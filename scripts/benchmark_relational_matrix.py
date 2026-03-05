#!/usr/bin/env python3
"""Definitive matrix benchmark for relational model variants.

This script provides a single, rerunnable source of truth for comparing:
- pure Python/PyG execution (`python_only`)
- non-kernel optimizations (`optimized_general`)
- full optimization lane including custom mp kernels (`optimized_full`)

It emits machine-readable JSON/CSV artifacts for easy history tracking.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pathlib
import socket
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import torch

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import benchmark_relational_models as brm


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    env: dict[str, str]


VARIANTS: dict[str, Variant] = {
    "python_only": Variant(
        name="python_only",
        description="Disable custom mp ops and grouped MLP; use Python/PyG path only.",
        env={
            "RELM_MODELS_MP_OPS": "0",
            "RELM_MODELS_MP_FANIN": "0",
            "RELM_MODELS_MP_FANIN_FUSED": "0",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "0",
            "RELM_MODELS_MP_LOGSUMEXP": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "0",
            "RELM_MP_ENABLE": "0",
            "RELM_MP_FALLBACK": "python",
        },
    ),
    "optimized_general": Variant(
        name="optimized_general",
        description=(
            "Keep generic model optimizations (e.g., grouped compatible relation MLP path), "
            "but disable custom mp kernel lane."
        ),
        env={
            "RELM_MODELS_MP_OPS": "0",
            "RELM_MODELS_MP_FANIN": "0",
            "RELM_MODELS_MP_FANIN_FUSED": "0",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "0",
            "RELM_MP_ENABLE": "0",
            "RELM_MP_FALLBACK": "python",
        },
    ),
    "optimized_full": Variant(
        name="optimized_full",
        description=(
            "Enable full optimization lane: grouped path plus centralized custom fanin kernels "
            "(sum, and logsumexp when enabled)."
        ),
        env={
            "RELM_MODELS_MP_OPS": "1",
            "RELM_MODELS_MP_FANIN": "1",
            "RELM_MODELS_MP_FANIN_FUSED": "1",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "1",
            "RELM_MP_ENABLE": "1",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "0",
            "RELM_MP_FALLBACK": "error",
        },
    ),
    "batched_cpp_experimental": Variant(
        name="batched_cpp_experimental",
        description=(
            "Experimental decentralized batched fanout/fanin custom-op lane for direct "
            "A/B against Python batched path."
        ),
        env={
            "RELM_MODELS_MP_OPS": "1",
            "RELM_MODELS_MP_FANIN": "1",
            "RELM_MODELS_MP_FANIN_FUSED": "1",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "1",
            "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": "1",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "0",
            "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": "1",
            "RELM_MP_ENABLE": "1",
            "RELM_MP_FALLBACK": "error",
        },
    ),
    "batched_cpp_fanin_only": Variant(
        name="batched_cpp_fanin_only",
        description="Experimental decentralized batched fanin custom-op only.",
        env={
            "RELM_MODELS_MP_OPS": "1",
            "RELM_MODELS_MP_FANIN": "1",
            "RELM_MODELS_MP_FANIN_FUSED": "1",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "1",
            "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": "1",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "0",
            "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": "0",
            "RELM_MP_ENABLE": "1",
            "RELM_MP_FALLBACK": "error",
        },
    ),
    "batched_cpp_fanout_only": Variant(
        name="batched_cpp_fanout_only",
        description="Experimental decentralized batched fanout custom-op only.",
        env={
            "RELM_MODELS_MP_OPS": "1",
            "RELM_MODELS_MP_FANIN": "1",
            "RELM_MODELS_MP_FANIN_FUSED": "1",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "1",
            "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "0",
            "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": "1",
            "RELM_MP_ENABLE": "1",
            "RELM_MP_FALLBACK": "error",
        },
    ),
    "batched_cpp_scaffold_only": Variant(
        name="batched_cpp_scaffold_only",
        description=(
            "Experimental decentralized batched scaffold relay path only: C++ pack/scatter "
            "without custom fanin reduction kernel."
        ),
        env={
            "RELM_MODELS_MP_OPS": "1",
            "RELM_MODELS_MP_FANIN": "0",
            "RELM_MODELS_MP_FANIN_FUSED": "0",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "1",
            "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": "1",
            "RELM_MP_ENABLE": "1",
            "RELM_MP_FALLBACK": "error",
        },
    ),
    "batched_cpp_scaffold_fanin_only": Variant(
        name="batched_cpp_scaffold_fanin_only",
        description="Experimental decentralized batched C++ fanin packing only (no reduction kernel).",
        env={
            "RELM_MODELS_MP_OPS": "1",
            "RELM_MODELS_MP_FANIN": "0",
            "RELM_MODELS_MP_FANIN_FUSED": "0",
            "RELM_MODELS_MP_FANOUT": "0",
            "RELM_MODELS_MP_GROUPED_MLP": "1",
            "RELM_MODELS_MP_LOGSUMEXP": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": "0",
            "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": "1",
            "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": "0",
            "RELM_MP_ENABLE": "1",
            "RELM_MP_FALLBACK": "error",
        },
    ),
}


def _variant_env_for_aggr(variant: Variant, aggr: str) -> dict[str, str]:
    env = dict(variant.env)
    # Batched fanin reduction custom kernels only exist for sum/logsumexp.
    # Keep benchmark lanes apples-to-apples by disabling that flag on other aggregations.
    if aggr not in {"sum", "logsumexp"}:
        env["RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL"] = "0"
    return env


@contextlib.contextmanager
def _patched_env(overrides: dict[str, str]):
    old: dict[str, str | None] = {}
    for key, value in overrides.items():
        old[key] = os.environ.get(key)
        os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in old.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _resolve_device(query: str) -> torch.device:
    return brm._resolve_device(query)  # reuse existing behavior


def _build_workload(args: argparse.Namespace):
    if args.workload == "synthetic":
        return brm._build_synthetic_workload(
            seed=args.seed,
            num_graphs=args.num_graphs,
            num_symbols=args.num_symbols,
            num_predicates=args.num_predicates,
            min_arity=args.min_arity,
            max_arity=args.max_arity,
            atoms_per_predicate=args.atoms_per_predicate,
            symbol_type=args.symbol_type,
        )
    if args.workload == "rgnet":
        return brm._build_rgnet_workload(
            rgnet_root=args.rgnet_root,
            domain_case=args.domain_case,
            problem_case=args.problem_case,
            max_states=args.max_states,
            seed=args.seed,
        )
    return brm._build_pddl_workload(
        pddl_root=args.pddl_root,
        domain_case=args.domain_case,
        problem_case=args.problem_case,
        max_states=args.max_states,
        seed=args.seed,
    )


def _seed_all(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    workload = _build_workload(args)
    batch = workload.batch.to(device) if hasattr(workload.batch, "to") else workload.batch
    counts = brm._describe_tensor_counts(batch)

    model_kinds = [item.strip() for item in args.model_kinds.split(",") if item.strip()]
    aggregations = [item.strip() for item in args.aggregations.split(",") if item.strip()]
    variant_names = [item.strip() for item in args.variants.split(",") if item.strip()]

    unknown = [name for name in variant_names if name not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants requested: {unknown!r}. Known: {sorted(VARIANTS)}")

    print(
        f"device={device.type} workload={args.workload} rounds={args.rounds} warmup={args.warmup} "
        f"repeats={args.repeats} model_kinds={','.join(model_kinds)} aggregations={','.join(aggregations)}"
    )
    print(
        f"embedding_size={args.embedding_size} num_layer={args.num_layer} "
        f"rel_layer_mode={args.rel_layer_mode} central_layer_mode={args.central_layer_mode}"
    )
    print(
        f"workload_meta="
        + ", ".join(f"{k}={v}" for k, v in sorted(workload.meta.items()))
    )

    rows: list[dict[str, Any]] = []
    for aggr in aggregations:
        for backward in (False, True):
            for model_kind in model_kinds:
                per_variant_means: dict[str, list[float]] = {name: [] for name in variant_names}
                per_variant_p50s: dict[str, list[float]] = {name: [] for name in variant_names}
                per_variant_stdevs: dict[str, list[float]] = {name: [] for name in variant_names}

                for rep in range(int(args.repeats)):
                    # Alternate variant order by replicate to reduce warmup/drift bias.
                    ordered_variants = (
                        list(variant_names)
                        if (rep % 2 == 0)
                        else list(reversed(variant_names))
                    )
                    rep_seed = int(args.seed) + rep
                    for variant_name in ordered_variants:
                        variant = VARIANTS[variant_name]
                        variant_env = _variant_env_for_aggr(variant, aggr)
                        _seed_all(rep_seed, device)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                            torch.cuda.empty_cache()
                        with _patched_env(variant_env):
                            model = brm._make_model(
                                model_kind=model_kind,
                                embedding_size=args.embedding_size,
                                num_layer=args.num_layer,
                                aggr=aggr,
                                relation_dict=workload.relation_dict,
                                symbol_type_ids=workload.symbol_type_ids,
                                rel_layer_mode=args.rel_layer_mode,
                                central_layer_mode=args.central_layer_mode,
                            ).to(device)
                            times_ms = brm._benchmark(
                                model,
                                batch,
                                device=device,
                                rounds=args.rounds,
                                warmup=args.warmup,
                                backward=backward,
                            )
                        stats = brm._summarize(times_ms)
                        per_variant_means[variant_name].append(float(stats["mean_ms"]))
                        per_variant_p50s[variant_name].append(float(stats["p50_ms"]))
                        per_variant_stdevs[variant_name].append(float(stats["stdev_ms"]))
                        del model

                mode_label = "bwd" if backward else "fwd"
                for variant_name in variant_names:
                    variant = VARIANTS[variant_name]
                    replicate_means = per_variant_means[variant_name]
                    replicate_p50s = per_variant_p50s[variant_name]
                    replicate_stdevs = per_variant_stdevs[variant_name]
                    mean_ms = statistics.fmean(replicate_means)
                    p50_ms = statistics.fmean(replicate_p50s)
                    rep_stdev_ms = (
                        statistics.pstdev(replicate_means)
                        if len(replicate_means) > 1
                        else 0.0
                    )
                    within_stdev_ms = statistics.fmean(replicate_stdevs)

                    row = {
                        "variant": variant.name,
                        "variant_description": variant.description,
                        "aggr": aggr,
                        "backward": int(backward),
                        "model_kind": model_kind,
                        "repeats": int(args.repeats),
                        "rounds": int(args.rounds),
                        "warmup": int(args.warmup),
                        "mean_ms": float(mean_ms),
                        "p50_ms": float(p50_ms),
                        "replicate_stdev_ms": float(rep_stdev_ms),
                        "within_run_stdev_ms": float(within_stdev_ms),
                        "total_nodes": int(counts["total_nodes"]),
                        "total_edges": int(counts["total_edges"]),
                    }
                    rows.append(row)
                    print(
                        f"[{aggr:>9}] [{mode_label}] [{model_kind:>13}] [{variant.name:>16}] "
                        f"mean_ms={mean_ms:.3f} p50_ms={p50_ms:.3f} rep_stdev={rep_stdev_ms:.3f}"
                    )

    baseline: dict[tuple[str, int, str], float] = {}
    for row in rows:
        key = (row["aggr"], row["backward"], row["model_kind"])
        if row["variant"] == "python_only":
            baseline[key] = row["mean_ms"]
    for row in rows:
        key = (row["aggr"], row["backward"], row["model_kind"])
        base = baseline.get(key)
        if base is None or base == 0.0:
            row["delta_vs_python_only_ms"] = None
            row["delta_vs_python_only_pct"] = None
            row["speedup_vs_python_only"] = None
            continue
        row["delta_vs_python_only_ms"] = float(row["mean_ms"] - base)
        row["delta_vs_python_only_pct"] = float((row["mean_ms"] / base - 1.0) * 100.0)
        row["speedup_vs_python_only"] = float(base / row["mean_ms"])

    device_name = None
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": device_name,
        "args": {
            "workload": args.workload,
            "device": args.device,
            "seed": int(args.seed),
            "embedding_size": int(args.embedding_size),
            "num_layer": int(args.num_layer),
            "rounds": int(args.rounds),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "model_kinds": model_kinds,
            "aggregations": aggregations,
            "variants": variant_names,
            "rel_layer_mode": args.rel_layer_mode,
            "central_layer_mode": args.central_layer_mode,
            "workload_kwargs": {
                "num_graphs": int(args.num_graphs),
                "num_symbols": int(args.num_symbols),
                "num_predicates": int(args.num_predicates),
                "min_arity": int(args.min_arity),
                "max_arity": int(args.max_arity),
                "atoms_per_predicate": int(args.atoms_per_predicate),
                "symbol_type": args.symbol_type,
                "rgnet_root": args.rgnet_root,
                "pddl_root": args.pddl_root,
                "domain_case": args.domain_case,
                "problem_case": args.problem_case,
                "max_states": int(args.max_states),
            },
        },
        "workload_meta": workload.meta,
        "counts": counts,
        "rows": rows,
    }
    return result


def _write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = [
        "variant",
        "aggr",
        "backward",
        "model_kind",
        "repeats",
        "rounds",
        "warmup",
        "mean_ms",
        "p50_ms",
        "replicate_stdev_ms",
        "within_run_stdev_ms",
        "delta_vs_python_only_ms",
        "delta_vs_python_only_pct",
        "speedup_vs_python_only",
        "total_nodes",
        "total_edges",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in keys})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        choices=("synthetic", "rgnet", "pddl"),
        default=os.getenv("RELM_GNN_WORKLOAD", "pddl"),
    )
    parser.add_argument("--device", default=os.getenv("RELM_GNN_DEVICE", "auto"))
    parser.add_argument("--seed", type=int, default=int(os.getenv("RELM_GNN_SEED", "42")))
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=int(os.getenv("RELM_GNN_EMBEDDING", "32")),
    )
    parser.add_argument(
        "--num-layer", type=int, default=int(os.getenv("RELM_GNN_LAYERS", "6"))
    )
    parser.add_argument("--rounds", type=int, default=int(os.getenv("RELM_GNN_ROUNDS", "10")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("RELM_GNN_WARMUP", "2")))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--model-kinds",
        default=os.getenv("RELM_GNN_MODEL_KINDS", "decentralized,centralized"),
        help="Comma-separated list of model kinds.",
    )
    parser.add_argument(
        "--aggregations",
        default=os.getenv("RELM_GNN_MATRIX_AGGRS", "sum,logsumexp,mean"),
        help="Comma-separated aggregation names.",
    )
    parser.add_argument(
        "--variants",
        default="python_only,optimized_general,optimized_full",
        help="Comma-separated variant names.",
    )
    parser.add_argument(
        "--rel-layer-mode",
        choices=("modular", "batched_cached"),
        default=os.getenv("RELM_GNN_REL_LAYER_MODE", "batched_cached"),
    )
    parser.add_argument(
        "--central-layer-mode",
        choices=("fused", "modular"),
        default=os.getenv("RELM_GNN_CENTRAL_LAYER_MODE", "fused"),
    )
    parser.add_argument("--num-graphs", type=int, default=32)
    parser.add_argument("--num-symbols", type=int, default=32)
    parser.add_argument("--num-predicates", type=int, default=12)
    parser.add_argument("--min-arity", type=int, default=1)
    parser.add_argument("--max-arity", type=int, default=4)
    parser.add_argument("--atoms-per-predicate", type=int, default=128)
    parser.add_argument("--symbol-type", default="_symbol_")
    parser.add_argument("--rgnet-root", default="~/GitHub/rgnet")
    parser.add_argument("--pddl-root", default="~/GitHub/rgnet/test/pddl_instances")
    parser.add_argument("--domain-case", default="blocks")
    parser.add_argument("--problem-case", default="medium")
    parser.add_argument("--max-states", type=int, default=16)
    parser.add_argument(
        "--json-out",
        default="docs/benchmark_matrix_latest.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--csv-out",
        default="docs/benchmark_matrix_latest.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = _run_matrix(args)

    json_out = pathlib.Path(args.json_out).expanduser().resolve()
    csv_out = pathlib.Path(args.csv_out).expanduser().resolve()
    _write_json(json_out, payload)
    _write_csv(csv_out, payload["rows"])

    print(f"\nJSON written: {json_out}")
    print(f"CSV written:  {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
