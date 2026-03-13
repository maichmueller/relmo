#!/usr/bin/env python3
"""Direct parity benchmark: FlatRelationalGNN vs pymimir-rgnn on real PDDL states.

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
from torch_geometric.data import Data

from relmo.models import ArityMLPFactory, FlatExecutionPolicy, FlatRelationalGNN


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
    if hasattr(output, "entity") and torch.is_tensor(getattr(output, "entity")):
        return output.entity.square().mean()
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
    should_zero_grad = backward and zero_grad is not None

    for _ in range(int(warmup)):
        if callable(cudagraph_mark):
            cudagraph_mark()
        if should_zero_grad:
            zero_grad()
        out = run_once()
        if backward:
            _loss_from_output(out).backward()
        _sync_if_needed(device)

    times: list[float] = []
    for _ in range(int(rounds)):
        if callable(cudagraph_mark):
            cudagraph_mark()
        if should_zero_grad:
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


def _relm_relation_base_name(name: str) -> str:
    text = str(name).strip()
    base = re.sub(r"\[[^\]]*\]", "", text)
    return base.replace("+", "").replace("-", "").strip().lower()


def _canonical_relm_relation_name(name: str) -> str:
    lowered = str(name).strip().lower()
    tags = {tag.strip().lower() for tag in re.findall(r"\[([^\]]*)\]", lowered)}
    base = _relm_relation_base_name(name)
    if not base:
        return base
    if ("unsat" in tags) or ("false" in tags):
        return f"{base}_goal_false"
    if ("sat" in tags) or ("true" in tags):
        return f"{base}_goal_true"
    if ("g" in tags) or ("goal" in tags):
        return f"{base}_goal_all"
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
                    "reason": "no matching relmo canonical name",
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
            if "unsat" in name:
                penalty -= 3
            if "sat" in name:
                penalty -= 2
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
                    "reason": "no unused relmo relation with matching arity",
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


def _build_flat_relm_inputs_from_pymimir_encoded(
    *,
    encoded: Any,
    relation_arities: dict[str, int],
) -> dict[str, Any]:
    relations = getattr(encoded, "flattened_relations", {})
    node_sizes = getattr(encoded, "node_sizes", None)
    if torch.is_tensor(node_sizes) and node_sizes.numel() > 0:
        node_sizes_t = node_sizes.to(dtype=torch.long)
        node_count = int(node_sizes_t.sum().item())
    else:
        node_count = int(getattr(encoded, "node_count", 0))
        node_sizes_t = torch.tensor([node_count], dtype=torch.long)

    device = (
        next(iter(relations.values())).device
        if relations
        else node_sizes_t.device
    )
    relation_names = tuple(str(name) for name in relation_arities.keys())
    arities_t = torch.as_tensor(
        tuple(int(arity) for arity in relation_arities.values()),
        dtype=torch.long,
        device=device,
    )
    counts: list[int] = []
    arg_chunks: list[torch.Tensor] = []
    for relation_name, relation_arity in relation_arities.items():
        arity = int(relation_arity)
        if arity <= 0:
            counts.append(0)
            continue
        flat_ids = relations.get(str(relation_name))
        if flat_ids is None:
            counts.append(0)
            continue
        flat_ids = flat_ids.to(device=device, dtype=torch.long).view(-1)
        if flat_ids.numel() % arity != 0:
            raise ValueError(
                f"Relation {relation_name!r} has {int(flat_ids.numel())} flattened ids, "
                f"which is not divisible by its arity {arity}."
            )
        counts.append(int(flat_ids.numel() // arity))
        if flat_ids.numel() > 0:
            arg_chunks.append(flat_ids)

    relation_counts = torch.tensor([counts], dtype=torch.long, device=device)
    relation_args = (
        torch.cat(arg_chunks, dim=0)
        if arg_chunks
        else torch.empty((0,), dtype=torch.long, device=device)
    )
    return {
        "x": torch.zeros((node_count, 1), dtype=torch.float, device=device),
        "relation_counts": relation_counts,
        "relation_args": relation_args,
        "relation_arities": arities_t,
        "relation_names": relation_names,
        "node_sizes": node_sizes_t.to(device=device),
    }


def _build_flat_relm_inputs_from_flat_data(
    *,
    data: Any,
    keep_relations: Sequence[str] | None = None,
) -> dict[str, Any]:
    relation_names_full = tuple(str(name) for name in data.relation_names)
    relation_arities_full = tuple(int(arity) for arity in data.relation_arities)
    relation_counts_full = data.relation_counts
    if relation_counts_full.dim() == 1:
        relation_counts_full = relation_counts_full.unsqueeze(0)
    relation_counts_full = relation_counts_full.to(dtype=torch.long)
    relation_args_full = data.relation_args.to(dtype=torch.long).view(-1)

    if keep_relations is None:
        keep_indices = list(range(len(relation_names_full)))
    else:
        name_to_index = {
            relation_name: idx for idx, relation_name in enumerate(relation_names_full)
        }
        keep_indices = [
            name_to_index[str(relation_name)]
            for relation_name in keep_relations
            if str(relation_name) in name_to_index
        ]

    counts_total = relation_counts_full.sum(dim=0)
    slot_offsets = [0]
    cursor = 0
    for count_t, arity in zip(counts_total.tolist(), relation_arities_full):
        cursor += int(count_t) * int(arity)
        slot_offsets.append(cursor)

    relation_names = tuple(relation_names_full[idx] for idx in keep_indices)
    relation_arities = torch.as_tensor(
        [relation_arities_full[idx] for idx in keep_indices],
        dtype=torch.long,
        device=relation_args_full.device,
    )
    relation_counts = relation_counts_full.index_select(
        1,
        torch.as_tensor(keep_indices, dtype=torch.long, device=relation_counts_full.device),
    )
    arg_chunks = [
        relation_args_full[slot_offsets[idx] : slot_offsets[idx + 1]]
        for idx in keep_indices
        if slot_offsets[idx + 1] > slot_offsets[idx]
    ]
    relation_args = (
        torch.cat(arg_chunks, dim=0)
        if arg_chunks
        else torch.empty((0,), dtype=torch.long, device=relation_args_full.device)
    )

    x_value = getattr(data, "x", None)
    if x_value is None:
        node_sizes = getattr(data, "node_sizes", None)
        if torch.is_tensor(node_sizes) and node_sizes.numel() > 0:
            node_count = int(node_sizes.sum().item())
            x_device = node_sizes.device
        else:
            node_count = int(getattr(data, "num_nodes", 0))
            x_device = relation_args_full.device
        x_value = torch.zeros((node_count, 1), dtype=torch.float, device=x_device)

    payload: dict[str, Any] = {
        "x": x_value,
        "relation_counts": relation_counts,
        "relation_args": relation_args,
        "relation_arities": relation_arities,
        "relation_names": relation_names,
    }
    for key in (
        "relation_sources",
        "batch",
        "node_sizes",
        "object_indices",
        "object_sizes",
        "history_entity_indices",
        "history_entity_sizes",
        "history_entity_dt",
        "target_entity_indices",
        "target_entity_group_ids",
        "target_entity_sizes",
        "target_positions",
        "target_group_ids",
        "target_sizes",
        "target_indices",
        "target_candidate_ids",
    ):
        if hasattr(data, key):
            payload[key] = getattr(data, key)
    return payload


def _append_zero_count_relations_to_flat_payload(
    *,
    payload: dict[str, Any],
    extra_relations: dict[str, int],
) -> dict[str, Any]:
    if not extra_relations:
        return payload

    relation_counts = payload["relation_counts"]
    if relation_counts.dim() == 1:
        relation_counts = relation_counts.unsqueeze(0)
    relation_counts = relation_counts.to(dtype=torch.long)
    device = relation_counts.device
    extra_names = tuple(str(name) for name in extra_relations.keys())
    extra_arities = torch.as_tensor(
        [int(extra_relations[name]) for name in extra_names],
        dtype=torch.long,
        device=device,
    )
    extra_counts = torch.zeros(
        (int(relation_counts.size(0)), len(extra_names)),
        dtype=torch.long,
        device=device,
    )

    out = dict(payload)
    out["relation_counts"] = torch.cat([relation_counts, extra_counts], dim=1)
    out["relation_arities"] = torch.cat(
        [payload["relation_arities"].to(device=device, dtype=torch.long), extra_arities],
        dim=0,
    )
    out["relation_names"] = tuple(payload["relation_names"]) + extra_names
    if "relation_sources" in out:
        out["relation_sources"] = tuple(out["relation_sources"]) + tuple(
            "synthetic_zero" for _ in extra_names
        )
    return out


def _flat_payload_to_pyg_data(payload: dict[str, Any]) -> Data:
    data = Data(
        x=payload["x"],
        relation_counts=payload["relation_counts"],
        relation_args=payload["relation_args"],
        relation_arities=payload.get("relation_arities"),
    )
    for key, value in payload.items():
        if key in {"x", "relation_counts", "relation_args", "relation_arities"}:
            continue
        setattr(data, key, value)
    return data


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
    iterator = (
        relation_modules.items()
        if hasattr(relation_modules, "items")
        else enumerate(relation_modules)
    )
    for relation_name, module in iterator:
        shapes: list[tuple[int, int]] = []
        for submodule in module.modules():
            if isinstance(submodule, torch.nn.Linear):
                shapes.append((int(submodule.in_features), int(submodule.out_features)))
        out[str(relation_name)] = shapes
    return out


def _extract_relm_relation_linear_shapes(relm_model: torch.nn.Module) -> dict[str, list[tuple[int, int]]]:
    relation_modules = getattr(getattr(relm_model, "relational_layer", None), "update_modules", None)
    if relation_modules is None:
        return {}
    relation_names = getattr(relm_model, "relation_names", ())
    if relation_names and not hasattr(relation_modules, "items"):
        out: dict[str, list[tuple[int, int]]] = {}
        for relation_name, module in zip(relation_names, relation_modules):
            shapes: list[tuple[int, int]] = []
            for submodule in module.modules():
                if isinstance(submodule, torch.nn.Linear):
                    shapes.append((int(submodule.in_features), int(submodule.out_features)))
            out[str(relation_name)] = shapes
        return out
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


def _build_flat_relation_instance_counts(
    *,
    flat_payload: dict[str, Any],
) -> dict[str, int]:
    relation_names = tuple(str(name) for name in flat_payload["relation_names"])
    relation_counts = flat_payload["relation_counts"]
    if relation_counts.dim() == 1:
        relation_counts = relation_counts.unsqueeze(0)
    counts_total = relation_counts.sum(dim=0).tolist()
    return {
        relation_name: int(count)
        for relation_name, count in zip(relation_names, counts_total)
    }


def _build_pymimir_relation_instance_counts(
    *,
    pymimir_relations: dict[str, torch.Tensor],
    pymimir_relation_arities: dict[str, int],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for relation_name, relation_arity in pymimir_relation_arities.items():
        tensor = pymimir_relations.get(str(relation_name))
        if tensor is None:
            counts[str(relation_name)] = 0
            continue
        arity = int(relation_arity)
        if arity <= 0:
            counts[str(relation_name)] = 0
            continue
        counts[str(relation_name)] = int(tensor.numel() // arity)
    return counts


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
    parser.add_argument(
        "--relmo-relation-kernels",
        dest="relmo_relation_kernels",
        choices=("auto", "off"),
        default="auto",
    )
    parser.add_argument(
        "--relmo-program-kernels",
        dest="relmo_program_kernels",
        choices=("auto", "off"),
        default="auto",
    )
    parser.add_argument(
        "--relmo-relation-gather",
        dest="relmo_relation_gather",
        choices=("auto", "off", "on"),
        default="auto",
    )
    parser.add_argument(
        "--relmo-mlp-layers",
        dest="relmo_mlp_layers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--encoder-mode",
        choices=("native", "shared_pymimir"),
        default="native",
        help=(
            "Use native encoders for each stack (`native`) or force both stacks "
            "to consume a shared pymimir flat encoding (`shared_pymimir`)."
        ),
    )
    parser.add_argument(
        "--strict-parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When enabled, restrict relmo to pymimir relation set and matching per-relation arities.",
    )
    parser.add_argument(
        "--include-goal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include goal encoding in pymimir inputs (relmo encodes goal by default).",
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

    relation_dict: dict[str, int]
    parity_meta: dict[str, Any] = {
        "enabled": bool(args.strict_parity),
        "relm_relation_count_full": None,
        "pymimir_relation_count": int(len(pymimir_relation_arities)),
        "parity_mode": None,
        "encoder_mode": str(args.encoder_mode),
    }

    goal_condition = problem.get_goal_condition()
    relm_goals = list(goal_condition.get_literals()) if args.include_goal else None
    if args.include_goal:
        pymimir_inputs = [(state, goal_condition) for state in states]
    else:
        pymimir_inputs = [(state,) for state in states]
    pymimir_encoded = get_input_from_encoders(pymimir_inputs, input_spec, pymimir_model.get_device())
    relm_encoder = None
    relm_batch_pyg = None
    relm_native_keep_relations: list[str] = []
    relm_native_zero_pad_relations: dict[str, int] = {}

    if args.encoder_mode == "shared_pymimir":
        relation_dict = {str(k): int(v) for k, v in sorted(pymimir_relation_arities.items())}
        parity_meta["relm_relation_count_full"] = int(len(relation_dict))
        parity_meta["parity_mode"] = "identity_from_common_pymimir_flat_encoding"
        parity_meta["selected_count"] = int(len(relation_dict))
        parity_meta["target_count"] = int(len(relation_dict))
        parity_meta["missing"] = []
        parity_meta["mappings"] = [
            {"pymimir_relation": str(name), "relm_relation": str(name), "arity": int(arity)}
            for name, arity in sorted(relation_dict.items())
        ]
        print(
            "[parity] identity relation mapping from shared pymimir flat encoding: "
            f"relations={len(relation_dict)}"
        )
        relm_flat_payload = _build_flat_relm_inputs_from_pymimir_encoded(
            encoded=pymimir_encoded,
            relation_arities=relation_dict,
        )
    else:
        relm_encoder = mifrost.FlatRelationEncoder(
            domain,
            goal_satisfaction_derivations={
                mifrost.GoalSatisfaction.satisfied,
                mifrost.GoalSatisfaction.unsatisfied,
            },
        )
        relm_batch_pyg = relm_encoder.encode_batch(states=states, goals=relm_goals).to(device)
        relm_relation_dict_full = {
            str(name): int(arity)
            for name, arity in zip(relm_batch_pyg.relation_names, relm_batch_pyg.relation_arities)
        }
        parity_meta["relm_relation_count_full"] = int(len(relm_relation_dict_full))
        parity_meta["parity_mode"] = "canonical_mapping_native_encoders_with_goal_sat_unsat"
        relation_dict, selection_meta = _select_relm_parity_relations(
            relm_relation_dict=relm_relation_dict_full,
            pymimir_relation_arities=pymimir_relation_arities,
        )
        pymimir_initial_counts = _build_pymimir_relation_instance_counts(
            pymimir_relations=pymimir_encoded.flattened_relations,
            pymimir_relation_arities=pymimir_relation_arities,
        )
        zero_pad_relations = {
            str(row["pymimir_relation"]): int(row["expected_arity"])
            for row in selection_meta.get("missing", [])
            if int(pymimir_initial_counts.get(str(row["pymimir_relation"]), 0)) == 0
        }
        if zero_pad_relations:
            relation_dict.update(zero_pad_relations)
            padded_mappings = list(selection_meta.get("mappings", []))
            padded_mappings.extend(
                {
                    "pymimir_relation": relation_name,
                    "relm_relation": relation_name,
                    "arity": int(relation_arity),
                }
                for relation_name, relation_arity in zero_pad_relations.items()
            )
            padded_missing = [
                row
                for row in selection_meta.get("missing", [])
                if str(row["pymimir_relation"]) not in zero_pad_relations
            ]
            selection_meta = dict(selection_meta)
            selection_meta["mappings"] = padded_mappings
            selection_meta["missing"] = padded_missing
            selection_meta["selected_count"] = int(len(relation_dict))
        parity_meta.update(selection_meta)
        parity_meta["selected_count"] = int(len(relation_dict))
        parity_meta["target_count"] = int(len(pymimir_relation_arities))
        print(
            "[parity] canonical mapping between native encoders: "
            f"shared_relations={len(relation_dict)} "
            f"relm_full={len(relm_relation_dict_full)} "
            f"pymimir_full={len(pymimir_relation_arities)}"
        )
        relm_native_keep_relations = [
            name for name in relation_dict.keys() if name in relm_relation_dict_full
        ]
        relm_native_zero_pad_relations = {
            relation_name: relation_arity
            for relation_name, relation_arity in relation_dict.items()
            if relation_name not in relm_relation_dict_full
        }
        relm_flat_payload = _build_flat_relm_inputs_from_flat_data(
            data=relm_batch_pyg,
            keep_relations=relm_native_keep_relations,
        )
        relm_flat_payload = _append_zero_count_relations_to_flat_payload(
            payload=relm_flat_payload,
            extra_relations=relm_native_zero_pad_relations,
        )

    pymimir_selected_relation_arities = {
        str(mapping["pymimir_relation"]): int(mapping["arity"])
        for mapping in parity_meta.get("mappings", [])
    }
    if not pymimir_selected_relation_arities:
        pymimir_selected_relation_arities = {
            str(k): int(v) for k, v in sorted(pymimir_relation_arities.items())
        }
    pymimir_encoded.flattened_relations = {
        str(k): v
        for k, v in pymimir_encoded.flattened_relations.items()
        if str(k) in pymimir_selected_relation_arities
    }
    try:
        pymimir_relation_mlps = (
            pymimir_model._mpnn_module._relation_network._message._relation_mlps  # type: ignore[attr-defined]
        )
        for relation_name in list(pymimir_relation_mlps.keys()):
            if str(relation_name) not in pymimir_selected_relation_arities:
                del pymimir_relation_mlps[relation_name]
    except Exception:
        pass

    relation_factory = ArityMLPFactory(
        feature_size=int(args.embedding_size),
        residual=False,
        layers=int(args.relmo_mlp_layers),
        activation="mish",
    )
    execution_policy = FlatExecutionPolicy(
        relation_kernels=args.relmo_relation_kernels,
        program_kernels=args.relmo_program_kernels,
        relation_gather=args.relmo_relation_gather,
    )
    relm_eager = FlatRelationalGNN(
        embedding_size=int(args.embedding_size),
        num_layers=int(args.num_layers),
        aggregation=args.aggr,
        relations=relation_dict,
        relation_module_factory=relation_factory,
        execution_policy=execution_policy,
        compile_forward=False,
    ).to(device)
    relm_compile = None
    if args.include_compile_lane:
        relm_compile = FlatRelationalGNN(
            embedding_size=int(args.embedding_size),
            num_layers=int(args.num_layers),
            aggregation=args.aggr,
            relations=relation_dict,
            relation_module_factory=relation_factory,
            execution_policy=execution_policy,
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
    relm_relation_counts = _build_flat_relation_instance_counts(flat_payload=relm_flat_payload)
    pymimir_relation_counts = _build_pymimir_relation_instance_counts(
        pymimir_relations=pymimir_encoded.flattened_relations,
        pymimir_relation_arities=pymimir_selected_relation_arities,
    )
    relation_count_rows, relation_count_mismatches = _compare_mapped_relation_values(
        mappings=parity_meta.get("mappings", []),
        relm_values=relm_relation_counts,
        pymimir_values=pymimir_relation_counts,
    )

    strict_enforced = bool(args.strict_parity) and args.encoder_mode == "shared_pymimir"
    if strict_enforced and linear_shape_mismatches:
        raise RuntimeError(
            "Strict parity requested but relation MLP linear shapes do not match: "
            f"{linear_shape_mismatches}"
        )
    if strict_enforced and relation_count_mismatches:
        raise RuntimeError(
            "Strict parity requested but mapped encoded relation instance counts do not match: "
            f"{relation_count_mismatches}"
        )

    relm_flat_data = _flat_payload_to_pyg_data(relm_flat_payload)
    relm_prepared = relm_eager._prepare_batch(relm_flat_data)
    relm_prepared_compile = relm_compile._prepare_batch(relm_flat_data) if relm_compile is not None else None

    def relm_compute() -> Any:
        return relm_eager._compute_entity_embeddings_prepared(relm_prepared)

    def relm_compute_compile() -> Any:
        if relm_compile is None:
            raise RuntimeError("Compile lane disabled.")
        if relm_prepared_compile is None:
            raise RuntimeError("Compile prepared inputs missing.")
        return relm_compile._compute_entity_embeddings_prepared(relm_prepared_compile)

    def relm_full() -> Any:
        if args.encoder_mode == "shared_pymimir":
            encoded = get_input_from_encoders(
                pymimir_inputs, input_spec, pymimir_model.get_device()
            )
            encoded.flattened_relations = {
                str(k): v
                for k, v in encoded.flattened_relations.items()
                if str(k) in relation_dict
            }
            payload = _build_flat_relm_inputs_from_pymimir_encoded(
                encoded=encoded,
                relation_arities=relation_dict,
            )
        else:
            if relm_encoder is None:
                raise RuntimeError("Native relmo encoder is not initialized.")
            native_batch = relm_encoder.encode_batch(states=states, goals=relm_goals).to(device)
            payload = _build_flat_relm_inputs_from_flat_data(
                data=native_batch,
                keep_relations=relm_native_keep_relations,
            )
            payload = _append_zero_count_relations_to_flat_payload(
                payload=payload,
                extra_relations=relm_native_zero_pad_relations,
            )
        return relm_eager.compute_entity_embeddings(_flat_payload_to_pyg_data(payload))

    def relm_full_compile() -> Any:
        if relm_compile is None:
            raise RuntimeError("Compile lane disabled.")
        if args.encoder_mode == "shared_pymimir":
            encoded = get_input_from_encoders(
                pymimir_inputs, input_spec, pymimir_model.get_device()
            )
            encoded.flattened_relations = {
                str(k): v
                for k, v in encoded.flattened_relations.items()
                if str(k) in relation_dict
            }
            payload = _build_flat_relm_inputs_from_pymimir_encoded(
                encoded=encoded,
                relation_arities=relation_dict,
            )
        else:
            if relm_encoder is None:
                raise RuntimeError("Native relmo encoder is not initialized.")
            native_batch = relm_encoder.encode_batch(states=states, goals=relm_goals).to(device)
            payload = _build_flat_relm_inputs_from_flat_data(
                data=native_batch,
                keep_relations=relm_native_keep_relations,
            )
            payload = _append_zero_count_relations_to_flat_payload(
                payload=payload,
                extra_relations=relm_native_zero_pad_relations,
            )
        return relm_compile.compute_entity_embeddings(_flat_payload_to_pyg_data(payload))

    def pymimir_compute() -> Any:
        return pymimir_model._mpnn_module.forward(pymimir_encoded)

    def pymimir_full() -> Any:
        encoded = get_input_from_encoders(pymimir_inputs, input_spec, pymimir_model.get_device())
        if pymimir_selected_relation_arities:
            encoded.flattened_relations = {
                str(k): v
                for k, v in encoded.flattened_relations.items()
                if str(k) in pymimir_selected_relation_arities
            }
        return pymimir_model._mpnn_module.forward(encoded)

    rows: list[dict[str, Any]] = []
    lanes: list[tuple[str, str, Callable[[], Any], Callable[[], None]]] = [
        ("relm_flat", "compute_only", relm_compute, lambda: _zero_grad(relm_eager)),
        ("pymimir_rgnn", "compute_only", pymimir_compute, lambda: _zero_grad(pymimir_model)),
        ("relm_flat", "full_with_encoding", relm_full, lambda: _zero_grad(relm_eager)),
        ("pymimir_rgnn", "full_with_encoding", pymimir_full, lambda: _zero_grad(pymimir_model)),
    ]
    if relm_compile is not None:
        lanes.insert(
            1,
            (
                "relm_flat_compile",
                "compute_only",
                relm_compute_compile,
                lambda: _zero_grad(relm_compile),
            ),
        )
        lanes.insert(
            4,
            (
                "relm_flat_compile",
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

    total_nodes = int(relm_flat_payload["x"].size(0))
    total_relation_slots = int(relm_flat_payload["relation_args"].numel())
    total_relation_instances = int(relm_flat_payload["relation_counts"].sum().item())
    relmo_msg_params = int(
        sum(
            p.numel()
            for name, p in relm_eager.named_parameters()
            if name.startswith("relational_layer.update_modules.")
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
            "encoder_mode": str(args.encoder_mode),
            "strict_parity": bool(args.strict_parity),
            "include_goal": bool(args.include_goal),
            "relmo_mlp_layers": int(args.relmo_mlp_layers),
            "relmo_relation_kernels": args.relmo_relation_kernels,
            "relmo_program_kernels": args.relmo_program_kernels,
            "relmo_relation_gather": args.relmo_relation_gather,
            "warmup": int(args.warmup),
            "rounds": int(args.rounds),
            "pymimir_normalize_updates": bool(args.pymimir_normalize_updates),
            "relm_env": {
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
            "total_relation_instances": total_relation_instances,
            "total_relation_slots": total_relation_slots,
        },
        "parity": {
            **parity_meta,
            "relmo_msg_params": relmo_msg_params,
            "pymimir_msg_params": pymimir_msg_params,
            "relmo_msg_param_ratio": (
                float(relmo_msg_params) / float(pymimir_msg_params)
                if pymimir_msg_params > 0
                else None
            ),
            "selected_relmo_relations": sorted(relation_dict.keys()),
            "pymimir_relation_arities": {
                str(k): int(v) for k, v in sorted(pymimir_relation_arities.items())
            },
            "relation_linear_shapes": {
                "relmo": relm_linear_shapes,
                "pymimir": pymimir_linear_shapes,
            },
            "relation_linear_shape_comparison": {
                "rows": linear_shape_rows,
                "mismatch_count": int(len(linear_shape_mismatches)),
                "mismatches": linear_shape_mismatches,
            },
            "relation_instance_counts": {
                "relmo": relm_relation_counts,
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
