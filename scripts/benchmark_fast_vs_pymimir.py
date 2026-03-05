#!/usr/bin/env python3
"""Direct parity benchmark: FastRelationalGNN vs pymimir-rgnn on real PDDL states.

By default this harness enforces strict apples-to-apples parity:
- same active relation set
- same per-relation arity
- same per-relation MLP linear topology
- same encoded inputs (state + goal)
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import torch

from relm.models import ArityMLPFactory, FastRelationalGNN


@dataclass(frozen=True)
class BenchStats:
    mean_ms: float
    p50_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    rounds: int


def _resolve_device(device: str) -> torch.device:
    query = device.strip().lower()
    if query == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    out = torch.device(device)
    if out.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
    return out


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _zero_grad(model: torch.nn.Module) -> None:
    model.zero_grad(set_to_none=True)


def _loss_from_output(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        if not output:
            raise RuntimeError("Expected non-empty dict output for benchmark loss.")
        return torch.stack([tensor.square().mean() for tensor in output.values()]).sum()
    if torch.is_tensor(output):
        return output.square().mean()
    raise RuntimeError(f"Unsupported output type for benchmark: {type(output)!r}.")


def _benchmark(
    *,
    run_once: Callable[[], Any],
    zero_grad: Callable[[], None] | None,
    device: torch.device,
    warmup: int,
    rounds: int,
    backward: bool,
) -> BenchStats:
    cudagraph_mark = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)

    for _ in range(int(warmup)):
        if callable(cudagraph_mark):
            cudagraph_mark()
        if zero_grad is not None:
            zero_grad()
        out = run_once()
        if backward:
            _loss_from_output(out).backward()
        _sync_if_needed(device)

    times: list[float] = []
    for _ in range(int(rounds)):
        if callable(cudagraph_mark):
            cudagraph_mark()
        if zero_grad is not None:
            zero_grad()
        t0 = time.perf_counter()
        out = run_once()
        if backward:
            _loss_from_output(out).backward()
        _sync_if_needed(device)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1_000.0)

    return BenchStats(
        mean_ms=float(statistics.fmean(times)),
        p50_ms=float(statistics.median(times)),
        stdev_ms=float(statistics.pstdev(times) if len(times) > 1 else 0.0),
        min_ms=float(min(times)),
        max_ms=float(max(times)),
        rounds=int(rounds),
    )


def _build_states(
    *,
    pddl_root: str,
    domain_case: str,
    problem_case: str,
    max_states: int,
    seed: int,
) -> tuple[Any, Any, list[Any]]:
    import pymimir  # type: ignore

    root = pathlib.Path(pddl_root).expanduser().resolve()
    domain_path = root / domain_case / "domain.pddl"
    problem_name = problem_case if str(problem_case).endswith(".pddl") else f"{problem_case}.pddl"
    problem_path = root / domain_case / problem_name
    if not domain_path.exists():
        raise FileNotFoundError(f"domain.pddl not found: {domain_path}")
    if not problem_path.exists():
        raise FileNotFoundError(f"problem file not found: {problem_path}")

    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")
    space, _ = pymimir.advanced.datasets.StateSpace.create(
        problem._search_context,
        pymimir.advanced.datasets.StateSpaceOptions(),
    )
    sampler = pymimir.wrapper_datasets.StateSpaceSampler(
        pymimir.advanced.datasets.StateSpaceSampler(space),
        problem,
    )
    all_states = list(sampler.get_states())
    total = len(all_states)
    n = min(total, int(max_states))
    if n <= 0:
        raise RuntimeError(
            f"No states available for {domain_case}/{problem_case} under {pddl_root}."
        )
    if n < total:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        ids = torch.randperm(total, generator=gen)[:n].tolist()
        states = [all_states[int(i)] for i in ids]
    else:
        states = all_states
    return domain, problem, states


def _extract_pymimir_relation_arities(
    *,
    pymimir_model: torch.nn.Module,
    embedding_size: int,
) -> dict[str, int]:
    relation_mlps = None
    try:
        relation_mlps = (
            pymimir_model._mpnn_module._relation_network._message._relation_mlps  # type: ignore[attr-defined]
        )
    except Exception:
        relation_mlps = None

    if relation_mlps is None:
        return {}

    out: dict[str, int] = {}
    for relation_name, module in relation_mlps.items():
        input_size = int(
            getattr(module, "input_size", getattr(getattr(module, "_inner", None), "in_features", 0))
        )
        if input_size <= 0:
            continue
        if input_size % int(embedding_size) != 0:
            continue
        out[str(relation_name)] = int(input_size // int(embedding_size))
    return out


def _canonical_pymimir_relation_name(name: str) -> str:
    text = str(name).strip()
    if text.startswith("relation_"):
        return text[len("relation_") :]
    return text.lower()


def _canonical_relm_relation_name(name: str) -> str:
    text = str(name).strip()
    lowered = text.lower()
    tags = {tag.strip().lower() for tag in re.findall(r"\[([^\]]*)\]", lowered)}
    base = re.sub(r"\[[^\]]*\]", "", text)
    base = base.replace("+", "").replace("-", "").strip().lower()
    if not base:
        return base
    is_goal = ("g" in tags) or ("goal" in tags)
    if is_goal:
        is_goal_true = ("sat" in tags) or ("true" in tags)
        return f"{base}_goal_{'true' if is_goal_true else 'false'}"
    return base


def _select_relm_parity_relations(
    *,
    relm_relation_dict: dict[str, int],
    pymimir_relation_arities: dict[str, int],
) -> tuple[dict[str, int], dict[str, Any]]:
    canonical_to_candidates: dict[str, list[tuple[str, int]]] = {}
    for relation_name, relation_arity in relm_relation_dict.items():
        arity = int(relation_arity)
        if arity <= 0:
            continue
        canonical = _canonical_relm_relation_name(str(relation_name))
        if not canonical:
            continue
        canonical_to_candidates.setdefault(canonical, []).append((str(relation_name), arity))

    selected: dict[str, int] = {}
    used_relm_relations: set[str] = set()
    mappings: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for pymimir_name, pymimir_arity in sorted(pymimir_relation_arities.items()):
        canonical = _canonical_pymimir_relation_name(pymimir_name)
        candidates = canonical_to_candidates.get(canonical, [])
        if not candidates:
            missing.append(
                {
                    "pymimir_relation": pymimir_name,
                    "canonical_name": canonical,
                    "expected_arity": int(pymimir_arity),
                    "reason": "no matching relm canonical name",
                }
            )
            continue

        def _score(item: tuple[str, int]) -> tuple[int, int, str]:
            name, arity = item
            penalty = 0
            if name != canonical:
                penalty += 1
            if "[" in name or "]" in name:
                penalty += 10
            if "+" in name or "-" in name:
                penalty += 5
            if "sat" in name:
                penalty += 3
            if int(arity) != int(pymimir_arity):
                penalty += 50
            return (penalty, len(name), name)

        chosen: tuple[str, int] | None = None
        for relm_name, relm_arity in sorted(candidates, key=_score):
            if relm_name in used_relm_relations:
                continue
            if int(relm_arity) != int(pymimir_arity):
                continue
            chosen = (relm_name, relm_arity)
            break

        if chosen is None:
            missing.append(
                {
                    "pymimir_relation": pymimir_name,
                    "canonical_name": canonical,
                    "expected_arity": int(pymimir_arity),
                    "reason": "no unused relm relation with matching arity",
                }
            )
            continue
        relm_name, relm_arity = chosen
        used_relm_relations.add(relm_name)
        selected[relm_name] = int(relm_arity)
        mappings.append(
            {
                "pymimir_relation": pymimir_name,
                "relm_relation": relm_name,
                "arity": int(relm_arity),
            }
        )

    meta = {
        "selected_count": int(len(selected)),
        "target_count": int(len(pymimir_relation_arities)),
        "missing": missing,
        "mappings": mappings,
    }
    return selected, meta


def _split_symbol_types(symbol_type_ids: str | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(symbol_type_ids, str):
        return (symbol_type_ids,)
    return tuple(str(x) for x in symbol_type_ids)


def _build_relm_inputs(
    *,
    batch: Any,
    keep_relations: set[str],
    symbol_types: tuple[str, ...],
) -> tuple[dict[str, torch.Tensor], dict[Any, torch.Tensor], dict[str, torch.Tensor]]:
    view = batch.as_hetero() if hasattr(batch, "as_hetero") else batch
    x_dict = dict(getattr(view, "x_dict", {}))
    edge_index_dict = dict(getattr(view, "edge_index_dict", {}))
    batch_dict = dict(getattr(view, "batch_dict", {}))
    keep_nodes = set(symbol_types) | set(keep_relations)

    x_out = {k: v for k, v in x_dict.items() if k in keep_nodes}
    batch_out = {k: v for k, v in batch_dict.items() if k in x_out}

    edge_out: dict[Any, torch.Tensor] = {}
    for edge_type, edge_index in edge_index_dict.items():
        src, _rel, dst = edge_type
        if src in x_out and dst in x_out:
            edge_out[edge_type] = edge_index
    return x_out, edge_out, batch_out


def _build_relm_inputs_from_pymimir_encoded(
    *,
    encoded: Any,
    relation_arities: dict[str, int],
    symbol_type: str,
) -> tuple[dict[str, torch.Tensor], dict[Any, torch.Tensor], dict[str, torch.Tensor]]:
    relations = getattr(encoded, "flattened_relations", {})
    device = (
        next(iter(relations.values())).device
        if relations
        else getattr(encoded, "node_sizes", torch.empty(0)).device
    )
    node_count = int(getattr(encoded, "node_count", 0))
    x_dict: dict[str, torch.Tensor] = {
        str(symbol_type): torch.zeros((node_count, 1), dtype=torch.float, device=device)
    }
    edge_index_dict: dict[Any, torch.Tensor] = {}

    for relation_name, relation_arity in relation_arities.items():
        arity = int(relation_arity)
        if arity <= 0:
            continue
        flat_ids = relations.get(relation_name)
        if flat_ids is None or flat_ids.numel() == 0:
            continue
        args = flat_ids.view(-1, arity).long()
        num_atoms = int(args.size(0))
        x_dict[str(relation_name)] = torch.zeros(
            (num_atoms, arity), dtype=torch.float, device=device
        )
        atom_idx = torch.arange(num_atoms, device=device, dtype=torch.long)
        for pos in range(arity):
            obj_idx = args[:, pos]
            edge_index_dict[(str(symbol_type), str(pos), str(relation_name))] = torch.stack(
                [obj_idx, atom_idx], dim=0
            )
            edge_index_dict[(str(relation_name), str(pos), str(symbol_type))] = torch.stack(
                [atom_idx, obj_idx], dim=0
            )

    node_sizes = getattr(encoded, "node_sizes", None)
    if torch.is_tensor(node_sizes) and node_sizes.numel() > 0:
        sizes = node_sizes.to(device=device, dtype=torch.long)
        batch_symbol = torch.repeat_interleave(
            torch.arange(int(sizes.numel()), device=device, dtype=torch.long),
            sizes,
        )
    else:
        batch_symbol = torch.zeros((node_count,), dtype=torch.long, device=device)
    batch_dict = {str(symbol_type): batch_symbol}
    return x_dict, edge_index_dict, batch_dict


def _make_pymimir_model(
    *,
    domain: Any,
    embedding_size: int,
    num_layers: int,
    aggr: str,
    include_goal: bool,
    normalize_updates: bool,
) -> tuple[torch.nn.Module, tuple[Any, ...]]:
    import pymimir_rgnn as pr  # type: ignore
    from pymimir_rgnn.encoders import get_input_from_encoders  # type: ignore

    aggr_q = aggr.strip().lower()
    if aggr_q == "sum":
        aggr_fn = pr.SumAggregation()
    elif aggr_q == "mean":
        aggr_fn = pr.MeanAggregation()
    elif aggr_q == "logsumexp":
        aggr_fn = pr.SmoothMaximumAggregation()
    else:
        raise ValueError(f"Unsupported aggregation for pymimir benchmark: {aggr!r}")

    if include_goal:
        input_spec = (pr.StateEncoder(), pr.GoalEncoder())
    else:
        input_spec = (pr.StateEncoder(),)
    hcfg = pr.HyperparameterConfig(
        domain=domain,
        embedding_size=int(embedding_size),
        num_layers=int(num_layers),
        normalize_updates=bool(normalize_updates),
        global_readout=False,
        residual_updates=True,
        binarize_updates=False,
    )
    module_cfg = pr.ModuleConfig(
        aggregation_function=aggr_fn,
        message_function=pr.PredicateMLPMessages(hcfg, input_spec),
        update_function=pr.MLPUpdates(hcfg),
    )
    model = pr.RelationalGraphNeuralNetwork(
        hcfg,
        module_cfg,
        input_spec=input_spec,
        output_spec=[],
    )
    return model, input_spec


def _extract_relation_linear_shapes(
    relation_modules: Any,
) -> dict[str, list[tuple[int, int]]]:
    out: dict[str, list[tuple[int, int]]] = {}
    for relation_name, module in relation_modules.items():
        shapes: list[tuple[int, int]] = []
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Linear):
                shapes.append((int(submodule.in_features), int(submodule.out_features)))
        out[str(relation_name)] = shapes
    return out


def _extract_relm_relation_linear_shapes(relm_model: torch.nn.Module) -> dict[str, list[tuple[int, int]]]:
    try:
        relation_modules = relm_model.fast_fused_rel_layer_mp.update_modules  # type: ignore[attr-defined]
    except Exception:
        return {}
    return _extract_relation_linear_shapes(relation_modules)


def _extract_pymimir_relation_linear_shapes(
    pymimir_model: torch.nn.Module,
) -> dict[str, list[tuple[int, int]]]:
    try:
        relation_modules = (
            pymimir_model._mpnn_module._relation_network._message._relation_mlps  # type: ignore[attr-defined]
        )
    except Exception:
        return {}
    return _extract_relation_linear_shapes(relation_modules)


def _build_relation_instance_counts(
    *,
    relm_x_dict: dict[str, torch.Tensor],
    relm_relation_dict: dict[str, int],
    pymimir_relations: dict[str, torch.Tensor],
    pymimir_relation_arities: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    relm_counts: dict[str, int] = {}
    for relation_name in relm_relation_dict:
        tensor = relm_x_dict.get(relation_name)
        if tensor is None:
            relm_counts[str(relation_name)] = 0
            continue
        relm_counts[str(relation_name)] = int(tensor.size(0))

    pymimir_counts: dict[str, int] = {}
    for relation_name, relation_arity in pymimir_relation_arities.items():
        tensor = pymimir_relations.get(relation_name)
        if tensor is None:
            pymimir_counts[str(relation_name)] = 0
            continue
        arity = int(relation_arity)
        if arity <= 0:
            pymimir_counts[str(relation_name)] = 0
            continue
        pymimir_counts[str(relation_name)] = int(tensor.numel() // arity)

    return relm_counts, pymimir_counts


def _compare_mapped_relation_values(
    *,
    mappings: list[dict[str, Any]],
    relm_values: dict[str, Any],
    pymimir_values: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for mapping in mappings:
        pymimir_relation = str(mapping["pymimir_relation"])
        relm_relation = str(mapping["relm_relation"])
        row = {
            "pymimir_relation": pymimir_relation,
            "relm_relation": relm_relation,
            "pymimir_value": pymimir_values.get(pymimir_relation),
            "relm_value": relm_values.get(relm_relation),
        }
        row["match"] = row["pymimir_value"] == row["relm_value"]
        rows.append(row)
        if not row["match"]:
            mismatches.append(row)
    return rows, mismatches


def _stats_to_dict(stats: BenchStats) -> dict[str, Any]:
    return {
        "mean_ms": stats.mean_ms,
        "p50_ms": stats.p50_ms,
        "stdev_ms": stats.stdev_ms,
        "min_ms": stats.min_ms,
        "max_ms": stats.max_ms,
        "rounds": stats.rounds,
    }


def _markdown_table(
    headers: list[str],
    rows: list[list[str]],
) -> str:
    sep = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}"
    except Exception:
        return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):+.1f}%"
    except Exception:
        return str(value)


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.3f}x"
    except Exception:
        return str(value)


def _build_comparisons(
    rows: list[dict[str, Any]],
    *,
    baseline_model: str = "pymimir_rgnn",
) -> list[dict[str, Any]]:
    ok_rows = [
        row
        for row in rows
        if ("error" not in row)
        and isinstance(row.get("mean_ms"), (float, int))
        and isinstance(row.get("lane"), str)
        and isinstance(row.get("mode"), str)
        and isinstance(row.get("model"), str)
    ]
    baseline_by_lane_mode = {
        (str(row["lane"]), str(row["mode"])): row
        for row in ok_rows
        if str(row["model"]) == baseline_model
    }
    comparisons: list[dict[str, Any]] = []
    for row in ok_rows:
        model = str(row["model"])
        lane = str(row["lane"])
        mode = str(row["mode"])
        if model == baseline_model:
            continue
        baseline = baseline_by_lane_mode.get((lane, mode))
        if baseline is None:
            continue
        candidate_mean = float(row["mean_ms"])
        baseline_mean = float(baseline["mean_ms"])
        delta_ms = candidate_mean - baseline_mean
        delta_pct = (delta_ms / baseline_mean * 100.0) if baseline_mean != 0 else None
        speedup_vs_baseline = (
            (baseline_mean / candidate_mean) if candidate_mean != 0 else None
        )
        winner = model if delta_ms < 0 else baseline_model
        comparisons.append(
            {
                "lane": lane,
                "mode": mode,
                "candidate_model": model,
                "baseline_model": baseline_model,
                "candidate_mean_ms": candidate_mean,
                "baseline_mean_ms": baseline_mean,
                "delta_ms": delta_ms,
                "delta_pct": delta_pct,
                "speedup_vs_baseline": speedup_vs_baseline,
                "winner": winner,
            }
        )
    return comparisons


def _print_results_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Model",
        "Lane",
        "Mode",
        "Mean ms",
        "P50 ms",
        "Stdev ms",
        "Min ms",
        "Max ms",
        "Rounds",
        "Status",
    ]
    table_rows: list[list[str]] = []
    for row in rows:
        if "error" in row:
            table_rows.append(
                [
                    str(row.get("model", "-")),
                    str(row.get("lane", "-")),
                    str(row.get("mode", "-")),
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    "-",
                    f"ERROR: {row.get('error', 'unknown')}",
                ]
            )
            continue
        table_rows.append(
            [
                str(row.get("model", "-")),
                str(row.get("lane", "-")),
                str(row.get("mode", "-")),
                _fmt_ms(row.get("mean_ms")),
                _fmt_ms(row.get("p50_ms")),
                _fmt_ms(row.get("stdev_ms")),
                _fmt_ms(row.get("min_ms")),
                _fmt_ms(row.get("max_ms")),
                str(row.get("rounds", "-")),
                "ok",
            ]
        )
    print("\nResults Table")
    print(_markdown_table(headers, table_rows))


def _print_comparison_table(comparisons: list[dict[str, Any]]) -> None:
    headers = [
        "Lane",
        "Mode",
        "Candidate",
        "Baseline",
        "Candidate ms",
        "Baseline ms",
        "Delta ms",
        "Delta %",
        "Speedup vs Baseline",
        "Winner",
    ]
    table_rows: list[list[str]] = []
    for row in comparisons:
        table_rows.append(
            [
                str(row["lane"]),
                str(row["mode"]),
                str(row["candidate_model"]),
                str(row["baseline_model"]),
                _fmt_ms(row.get("candidate_mean_ms")),
                _fmt_ms(row.get("baseline_mean_ms")),
                _fmt_ms(row.get("delta_ms")),
                _fmt_pct(row.get("delta_pct")),
                _fmt_ratio(row.get("speedup_vs_baseline")),
                str(row.get("winner", "-")),
            ]
        )
    print("\nComparison Table (vs pymimir_rgnn)")
    if not table_rows:
        print("No comparable rows available.")
        return
    print(_markdown_table(headers, table_rows))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--device", default=os.getenv("RELM_GNN_DEVICE", "cuda:0"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--aggr", choices=("sum", "mean", "logsumexp"), default="sum")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--pddl-root", default=str(repo_root / "data" / "pddl_domains"))
    parser.add_argument("--domain-case", default="blocks")
    parser.add_argument("--problem-case", default="probBLOCKS-4-0")
    parser.add_argument("--max-states", type=int, default=16)
    parser.add_argument("--relm-grouped-mlp", type=int, default=None)
    parser.add_argument("--relm-fanout-exp", type=int, default=None)
    parser.add_argument("--relm-fanin-reduce-exp", type=int, default=None)
    parser.add_argument("--relm-mlp-layers", type=int, default=1)
    parser.add_argument(
        "--strict-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, restrict relm to pymimir relation set and matching per-relation arities.",
    )
    parser.add_argument(
        "--include-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include goal encoding in pymimir inputs (relm encodes goal by default).",
    )
    parser.add_argument("--pymimir-normalize-updates", action="store_true")
    parser.add_argument("--tf32-high-precision", action="store_true")
    parser.add_argument("--include-compile-lane", action="store_true")
    parser.add_argument(
        "--json-out",
        default=str(repo_root / "docs" / "benchmark_fast_vs_pymimir_latest.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    pddl_root_path = pathlib.Path(args.pddl_root).expanduser().resolve()
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if args.tf32_high_precision:
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    # Optional env toggles for relm fast path.
    if args.relm_grouped_mlp is not None:
        os.environ["RELM_MODELS_MP_GROUPED_MLP"] = str(int(args.relm_grouped_mlp))
    if args.relm_fanout_exp is not None:
        os.environ["RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL"] = str(
            int(args.relm_fanout_exp)
        )
    if args.relm_fanin_reduce_exp is not None:
        os.environ["RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL"] = str(
            int(args.relm_fanin_reduce_exp)
        )

    domain, problem, states = _build_states(
        pddl_root=args.pddl_root,
        domain_case=args.domain_case,
        problem_case=args.problem_case,
        max_states=args.max_states,
        seed=args.seed,
    )

    import mifrost  # type: ignore
    import pymimir_rgnn as pr  # type: ignore
    from pymimir_rgnn.encoders import get_input_from_encoders  # type: ignore

    relm_encoder = mifrost.HGraphEncoder(domain)
    relm_batch = relm_encoder.encode_batch(states=states)
    relm_batch = relm_batch.to(device)
    relm_relation_dict_full = {str(k): int(v) for k, v in relm_encoder.relation_dict.items()}
    symbol_types = _split_symbol_types(relm_encoder.symbol_type_id)

    pymimir_model, input_spec = _make_pymimir_model(
        domain=domain,
        embedding_size=int(args.embedding_size),
        num_layers=int(args.num_layers),
        aggr=args.aggr,
        include_goal=bool(args.include_goal),
        normalize_updates=bool(args.pymimir_normalize_updates),
    )
    pymimir_model = pymimir_model.to(device)
    pymimir_relation_arities = _extract_pymimir_relation_arities(
        pymimir_model=pymimir_model,
        embedding_size=int(args.embedding_size),
    )

    relation_dict = dict(relm_relation_dict_full)
    parity_meta: dict[str, Any] = {
        "enabled": bool(args.strict_parity),
        "relm_relation_count_full": int(len(relm_relation_dict_full)),
        "pymimir_relation_count": int(len(pymimir_relation_arities)),
        "parity_mode": (
            "identity_from_pymimir_encoding" if bool(args.strict_parity) else "canonical_mapping"
        ),
    }

    goal_condition = problem.get_goal_condition()
    if args.include_goal:
        pymimir_inputs = [(state, goal_condition) for state in states]
    else:
        pymimir_inputs = [(state,) for state in states]
    pymimir_encoded = get_input_from_encoders(pymimir_inputs, input_spec, pymimir_model.get_device())
    if args.strict_parity:
        relation_dict = {str(k): int(v) for k, v in sorted(pymimir_relation_arities.items())}
        pymimir_encoded.flattened_relations = {
            str(k): v
            for k, v in pymimir_encoded.flattened_relations.items()
            if str(k) in relation_dict
        }
        parity_meta["selected_count"] = int(len(relation_dict))
        parity_meta["target_count"] = int(len(relation_dict))
        parity_meta["missing"] = []
        parity_meta["mappings"] = [
            {"pymimir_relation": str(name), "relm_relation": str(name), "arity": int(arity)}
            for name, arity in sorted(relation_dict.items())
        ]
        print(
            "[parity] identity relation mapping from pymimir encoding: "
            f"relations={len(relation_dict)}"
        )
        try:
            pymimir_relation_mlps = (
                pymimir_model._mpnn_module._relation_network._message._relation_mlps  # type: ignore[attr-defined]
            )
            for relation_name in list(pymimir_relation_mlps.keys()):
                if str(relation_name) not in relation_dict:
                    del pymimir_relation_mlps[relation_name]
        except Exception:
            pass
        relm_x_dict, relm_edge_index_dict, relm_batch_dict = _build_relm_inputs_from_pymimir_encoded(
            encoded=pymimir_encoded,
            relation_arities=relation_dict,
            symbol_type=str(symbol_types[0]),
        )
    else:
        relation_dict, selection_meta = _select_relm_parity_relations(
            relm_relation_dict=relm_relation_dict_full,
            pymimir_relation_arities=pymimir_relation_arities,
        )
        parity_meta.update(selection_meta)
        parity_meta["selected_count"] = int(len(relation_dict))
        parity_meta["target_count"] = int(len(pymimir_relation_arities))
        relm_x_dict, relm_edge_index_dict, relm_batch_dict = _build_relm_inputs(
            batch=relm_batch,
            keep_relations=set(relation_dict.keys()),
            symbol_types=symbol_types,
        )

    relation_factory = ArityMLPFactory(
        feature_size=int(args.embedding_size),
        residual=True,
        layers=int(args.relm_mlp_layers),
        activation="mish",
    )
    relm_eager = FastRelationalGNN(
        embedding_size=int(args.embedding_size),
        num_layer=int(args.num_layers),
        aggr=args.aggr,
        symbol_type_ids=relm_encoder.symbol_type_id,
        relation_dict=relation_dict,
        relation_module_factory=relation_factory,
        compile_forward=False,
    ).to(device)
    relm_compile = None
    if args.include_compile_lane:
        relm_compile = FastRelationalGNN(
            embedding_size=int(args.embedding_size),
            num_layer=int(args.num_layers),
            aggr=args.aggr,
            symbol_type_ids=relm_encoder.symbol_type_id,
            relation_dict=relation_dict,
            relation_module_factory=relation_factory,
            compile_forward=True,
        ).to(device)
    relm_linear_shapes = _extract_relm_relation_linear_shapes(relm_eager)
    pymimir_linear_shapes = _extract_pymimir_relation_linear_shapes(pymimir_model)
    mapped_linear_shapes_relm = {
        str(mapping["relm_relation"]): relm_linear_shapes.get(str(mapping["relm_relation"]), [])
        for mapping in parity_meta.get("mappings", [])
    }
    mapped_linear_shapes_pymimir = {
        str(mapping["pymimir_relation"]): pymimir_linear_shapes.get(str(mapping["pymimir_relation"]), [])
        for mapping in parity_meta.get("mappings", [])
    }
    linear_shape_rows, linear_shape_mismatches = _compare_mapped_relation_values(
        mappings=parity_meta.get("mappings", []),
        relm_values=mapped_linear_shapes_relm,
        pymimir_values=mapped_linear_shapes_pymimir,
    )
    relm_relation_counts, pymimir_relation_counts = _build_relation_instance_counts(
        relm_x_dict=relm_x_dict,
        relm_relation_dict=relation_dict,
        pymimir_relations=pymimir_encoded.flattened_relations,
        pymimir_relation_arities=pymimir_relation_arities,
    )
    relation_count_rows, relation_count_mismatches = _compare_mapped_relation_values(
        mappings=parity_meta.get("mappings", []),
        relm_values=relm_relation_counts,
        pymimir_values=pymimir_relation_counts,
    )

    if bool(args.strict_parity) and linear_shape_mismatches:
        raise RuntimeError(
            "Strict parity requested but relation MLP linear shapes do not match: "
            f"{linear_shape_mismatches}"
        )
    if bool(args.strict_parity) and relation_count_mismatches:
        raise RuntimeError(
            "Strict parity requested but mapped encoded relation instance counts do not match: "
            f"{relation_count_mismatches}"
        )

    def relm_compute() -> Any:
        return relm_eager(
            dict(relm_x_dict),
            dict(relm_edge_index_dict),
            dict(relm_batch_dict),
        )

    def relm_compute_compile() -> Any:
        if relm_compile is None:
            raise RuntimeError("Compile lane disabled.")
        return relm_compile(
            dict(relm_x_dict),
            dict(relm_edge_index_dict),
            dict(relm_batch_dict),
        )

    def relm_full() -> Any:
        if args.strict_parity:
            encoded = get_input_from_encoders(
                pymimir_inputs, input_spec, pymimir_model.get_device()
            )
            encoded.flattened_relations = {
                str(k): v
                for k, v in encoded.flattened_relations.items()
                if str(k) in relation_dict
            }
            x_dict, edge_index_dict, batch_dict = _build_relm_inputs_from_pymimir_encoded(
                encoded=encoded,
                relation_arities=relation_dict,
                symbol_type=str(symbol_types[0]),
            )
        else:
            batch = relm_encoder.encode_batch(states=states).to(device)
            x_dict, edge_index_dict, batch_dict = _build_relm_inputs(
                batch=batch,
                keep_relations=set(relation_dict.keys()),
                symbol_types=symbol_types,
            )
        return relm_eager(x_dict, edge_index_dict, batch_dict)

    def relm_full_compile() -> Any:
        if relm_compile is None:
            raise RuntimeError("Compile lane disabled.")
        if args.strict_parity:
            encoded = get_input_from_encoders(
                pymimir_inputs, input_spec, pymimir_model.get_device()
            )
            encoded.flattened_relations = {
                str(k): v
                for k, v in encoded.flattened_relations.items()
                if str(k) in relation_dict
            }
            x_dict, edge_index_dict, batch_dict = _build_relm_inputs_from_pymimir_encoded(
                encoded=encoded,
                relation_arities=relation_dict,
                symbol_type=str(symbol_types[0]),
            )
        else:
            batch = relm_encoder.encode_batch(states=states).to(device)
            x_dict, edge_index_dict, batch_dict = _build_relm_inputs(
                batch=batch,
                keep_relations=set(relation_dict.keys()),
                symbol_types=symbol_types,
            )
        return relm_compile(x_dict, edge_index_dict, batch_dict)

    def pymimir_compute() -> Any:
        return pymimir_model._mpnn_module.forward(pymimir_encoded)

    def pymimir_full() -> Any:
        encoded = get_input_from_encoders(pymimir_inputs, input_spec, pymimir_model.get_device())
        if args.strict_parity:
            encoded.flattened_relations = {
                str(k): v
                for k, v in encoded.flattened_relations.items()
                if str(k) in relation_dict
            }
        return pymimir_model._mpnn_module.forward(encoded)

    rows: list[dict[str, Any]] = []
    lanes: list[tuple[str, str, Callable[[], Any], Callable[[], None]]] = [
        ("relm_fast_fused", "compute_only", relm_compute, lambda: _zero_grad(relm_eager)),
        ("pymimir_rgnn", "compute_only", pymimir_compute, lambda: _zero_grad(pymimir_model)),
        ("relm_fast_fused", "full_with_encoding", relm_full, lambda: _zero_grad(relm_eager)),
        ("pymimir_rgnn", "full_with_encoding", pymimir_full, lambda: _zero_grad(pymimir_model)),
    ]
    if relm_compile is not None:
        lanes.insert(
            1,
            (
                "relm_fast_fused_compile",
                "compute_only",
                relm_compute_compile,
                lambda: _zero_grad(relm_compile),
            ),
        )
        lanes.insert(
            4,
            (
                "relm_fast_fused_compile",
                "full_with_encoding",
                relm_full_compile,
                lambda: _zero_grad(relm_compile),
            ),
        )

    for model_name, lane, fn, zgrad in lanes:
        for backward in (False, True):
            mode = "bwd" if backward else "fwd"
            try:
                stats = _benchmark(
                    run_once=fn,
                    zero_grad=zgrad,
                    device=device,
                    warmup=int(args.warmup),
                    rounds=int(args.rounds),
                    backward=backward,
                )
                row = {
                    "model": model_name,
                    "lane": lane,
                    "mode": mode,
                    **_stats_to_dict(stats),
                }
                rows.append(row)
                print(
                    f"[{model_name:>24}] [{lane:>18}] [{mode}] "
                    f"mean={stats.mean_ms:.3f}ms p50={stats.p50_ms:.3f}ms"
                )
            except Exception as exc:
                rows.append(
                    {
                        "model": model_name,
                        "lane": lane,
                        "mode": mode,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[{model_name:>24}] [{lane:>18}] [{mode}] "
                    f"ERROR: {type(exc).__name__}: {exc}"
                )

    comparisons = _build_comparisons(rows, baseline_model="pymimir_rgnn")
    _print_results_table(rows)
    _print_comparison_table(comparisons)

    total_nodes = int(sum(x.size(0) for x in relm_x_dict.values())) if relm_x_dict else 0
    total_edges = int(sum(e.size(1) for e in relm_edge_index_dict.values())) if relm_edge_index_dict else 0
    relm_msg_params = int(
        sum(
            p.numel()
            for name, p in relm_eager.named_parameters()
            if name.startswith("fast_fused_rel_layer_mp.update_modules.module_dict.")
        )
    )
    pymimir_msg_params = int(
        sum(
            p.numel()
            for name, p in pymimir_model.named_parameters()
            if name.startswith("_mpnn_module._relation_network._message._relation_mlps.")
        )
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "config": {
            "seed": int(args.seed),
            "embedding_size": int(args.embedding_size),
            "num_layers": int(args.num_layers),
            "aggr": args.aggr,
            "strict_parity": bool(args.strict_parity),
            "include_goal": bool(args.include_goal),
            "relm_mlp_layers": int(args.relm_mlp_layers),
            "warmup": int(args.warmup),
            "rounds": int(args.rounds),
            "pymimir_normalize_updates": bool(args.pymimir_normalize_updates),
            "relm_env": {
                "RELM_MODELS_MP_GROUPED_MLP": os.getenv("RELM_MODELS_MP_GROUPED_MLP"),
                "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": os.getenv(
                    "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL"
                ),
                "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": os.getenv(
                    "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL"
                ),
            },
        },
        "workload": {
            "source": "pddl",
            "pddl_root": str(pddl_root_path),
            "domain_case": args.domain_case,
            "problem_case": args.problem_case,
            "problem_name": getattr(problem, "name", None),
            "n_states": int(len(states)),
            "total_nodes": total_nodes,
            "total_edges": total_edges,
        },
        "parity": {
            **parity_meta,
            "relm_msg_params": relm_msg_params,
            "pymimir_msg_params": pymimir_msg_params,
            "relm_msg_param_ratio": (
                float(relm_msg_params) / float(pymimir_msg_params)
                if pymimir_msg_params > 0
                else None
            ),
            "selected_relm_relations": sorted(relation_dict.keys()),
            "pymimir_relation_arities": {
                str(k): int(v) for k, v in sorted(pymimir_relation_arities.items())
            },
            "relation_linear_shapes": {
                "relm": relm_linear_shapes,
                "pymimir": pymimir_linear_shapes,
            },
            "relation_linear_shape_comparison": {
                "rows": linear_shape_rows,
                "mismatch_count": int(len(linear_shape_mismatches)),
                "mismatches": linear_shape_mismatches,
            },
            "relation_instance_counts": {
                "relm": relm_relation_counts,
                "pymimir": pymimir_relation_counts,
            },
            "relation_instance_count_comparison": {
                "rows": relation_count_rows,
                "mismatch_count": int(len(relation_count_mismatches)),
                "mismatches": relation_count_mismatches,
            },
        },
        "comparisons": comparisons,
        "rows": rows,
    }

    out_path = pathlib.Path(args.json_out).expanduser()
    if not out_path.is_absolute():
        out_path = (repo_root / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nJSON written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
