#!/usr/bin/env python3
"""Profile decentralized batched variant lanes and report operator breakdowns."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts import benchmark_relational_models as brm
from scripts.benchmark_relational_matrix import (
    VARIANTS,
    _patched_env,
    _seed_all,
    _variant_env_for_aggr,
)


@dataclass(frozen=True)
class RunSpec:
    variant: str
    aggr: str
    backward: bool


_PATTERN_GROUPS: dict[str, tuple[str, ...]] = {
    "custom_pack": (
        "fanout_pack_multi",
        "fanin_pack_multi",
        "fanout_pack_from_edges",
        "fanin_pack_from_edges",
    ),
    "custom_scatter": ("fanout_scatter",),
    "custom_fanin_reduce": ("fanin_reduce",),
    "index_select": ("aten::index_select",),
    "cat": ("aten::cat",),
    "index_add": ("aten::index_add", "aten::index_add_"),
    "index_copy": ("aten::index_copy", "aten::index_copy_"),
    "scatter_reduce": ("aten::scatter_reduce", "aten::scatter_reduce_"),
    "scatter_add": ("aten::scatter_add", "aten::scatter_add_"),
    "linear_mm": ("aten::addmm", "aten::mm", "aten::matmul", "aten::bmm"),
    "relu": ("aten::relu", "aten::relu_", "aten::threshold_backward"),
}


def _device_self_cuda_us(event: Any) -> float:
    if hasattr(event, "self_cuda_time_total"):
        return float(event.self_cuda_time_total)
    if hasattr(event, "self_device_time_total"):
        return float(event.self_device_time_total)
    return 0.0


def _summarize_ops(events: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = []
    total_cuda_us = 0.0
    for ev in events:
        cuda_us = _device_self_cuda_us(ev)
        if cuda_us <= 0:
            continue
        totals.append((ev.key, cuda_us, int(ev.count)))
        total_cuda_us += cuda_us
    totals.sort(key=lambda row: row[1], reverse=True)

    grouped: dict[str, dict[str, float]] = {
        name: {"cuda_us": 0.0, "calls": 0.0} for name in _PATTERN_GROUPS
    }
    for key, cuda_us, calls in totals:
        for group, patterns in _PATTERN_GROUPS.items():
            if any(p in key for p in patterns):
                grouped[group]["cuda_us"] += float(cuda_us)
                grouped[group]["calls"] += float(calls)
    grouped_out = {}
    denom = total_cuda_us if total_cuda_us > 0 else 1.0
    for name, data in grouped.items():
        grouped_out[name] = {
            "self_cuda_ms": data["cuda_us"] / 1000.0,
            "calls": int(data["calls"]),
            "share_pct": (data["cuda_us"] / denom) * 100.0,
        }

    top_ops = [
        {
            "name": key,
            "self_cuda_ms": cuda_us / 1000.0,
            "calls": calls,
            "share_pct": (cuda_us / denom) * 100.0,
        }
        for key, cuda_us, calls in totals[:20]
    ]
    return (
        {
            "total_self_cuda_ms": total_cuda_us / 1000.0,
            "groups": grouped_out,
        },
        top_ops,
    )


def _time_steps(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: torch.device,
    backward: bool,
    warmup: int,
    rounds: int,
) -> dict[str, float]:
    for _ in range(int(warmup)):
        model.zero_grad(set_to_none=True)
        brm._run_once(model, batch, backward=backward)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    times_ms = []
    for _ in range(int(rounds)):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        brm._run_once(model, batch, backward=backward)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    return {
        "mean_ms": statistics.fmean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def _profile_ops(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: torch.device,
    backward: bool,
    profile_steps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(int(profile_steps)):
            model.zero_grad(set_to_none=True)
            brm._run_once(model, batch, backward=backward)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
    events = list(prof.key_averages())
    return _summarize_ops(events)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workload",
        choices=("synthetic", "rgnet", "pddl"),
        default=os.getenv("RELM_GNN_WORKLOAD", "pddl"),
    )
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-layer", type=int, default=6)
    parser.add_argument("--model-kind", default="decentralized")
    parser.add_argument("--rel-layer-mode", default="batched_cached")
    parser.add_argument("--central-layer-mode", default="fused")
    parser.add_argument("--num-graphs", type=int, default=12)
    parser.add_argument("--num-symbols", type=int, default=24)
    parser.add_argument("--num-predicates", type=int, default=10)
    parser.add_argument("--min-arity", type=int, default=1)
    parser.add_argument("--max-arity", type=int, default=4)
    parser.add_argument("--atoms-per-predicate", type=int, default=64)
    parser.add_argument("--symbol-type", default="_symbol_")
    parser.add_argument(
        "--rgnet-root",
        default=os.getenv("RELM_GNN_RGNET_ROOT", "~/GitHub/rgnet"),
    )
    parser.add_argument(
        "--pddl-root",
        default=os.getenv("RELM_GNN_PDDL_ROOT", "~/GitHub/rgnet/test/pddl_instances"),
    )
    parser.add_argument(
        "--domain-case",
        default=os.getenv("RELM_GNN_DOMAIN_CASE", "blocks"),
    )
    parser.add_argument(
        "--problem-case",
        default=os.getenv("RELM_GNN_PROBLEM_CASE", "medium"),
    )
    parser.add_argument(
        "--max-states",
        type=int,
        default=int(os.getenv("RELM_GNN_MAX_STATES", "16")),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=2)
    parser.add_argument(
        "--variants",
        default="python_only,optimized_full",
    )
    parser.add_argument("--aggregations", default="mean,sum")
    parser.add_argument("--json-out", default="docs/profile_batched_variants.json")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    aggregations = [a.strip() for a in args.aggregations.split(",") if a.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown!r}")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _seed_all(args.seed, device)

    workload = _build_workload(args)
    batch = workload.batch.to(device) if hasattr(workload.batch, "to") else workload.batch

    runs = [
        RunSpec(variant=variant, aggr=aggr, backward=backward)
        for aggr in aggregations
        for backward in (False, True)
        for variant in variants
    ]

    payload: dict[str, Any] = {
        "device": str(device),
        "seed": args.seed,
        "workload": dict(workload.meta),
        "runs": [],
    }

    for spec in runs:
        variant = VARIANTS[spec.variant]
        with _patched_env(_variant_env_for_aggr(variant, spec.aggr)):
            _seed_all(args.seed, device)
            model = brm._make_model(
                model_kind=args.model_kind,
                embedding_size=args.embedding_size,
                num_layer=args.num_layer,
                aggr=spec.aggr,
                relation_dict=workload.relation_dict,
                symbol_type_ids=workload.symbol_type_ids,
                rel_layer_mode=args.rel_layer_mode,
                central_layer_mode=args.central_layer_mode,
            ).to(device)
            model.train(mode=spec.backward)
            latency = _time_steps(
                model,
                batch,
                device=device,
                backward=spec.backward,
                warmup=args.warmup,
                rounds=args.rounds,
            )
            ops, top_ops = _profile_ops(
                model,
                batch,
                device=device,
                backward=spec.backward,
                profile_steps=args.profile_steps,
            )
            row = {
                "variant": spec.variant,
                "aggr": spec.aggr,
                "backward": bool(spec.backward),
                "latency": latency,
                "ops": ops,
                "top_ops": top_ops,
            }
            payload["runs"].append(row)
            print(
                f"[{spec.aggr:>9}] [{'bwd' if spec.backward else 'fwd'}] "
                f"[{spec.variant}] mean_ms={latency['mean_ms']:.3f} "
                f"total_self_cuda_ms={ops['total_self_cuda_ms']:.3f}"
            )

    out_path = Path(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"JSON written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
