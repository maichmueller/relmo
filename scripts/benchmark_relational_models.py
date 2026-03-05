#!/usr/bin/env python3
"""Benchmark RelationalGNN and CentralizedRelationalGNN runtimes."""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import torch
from torch_geometric.data import Batch, HeteroData

try:  # pragma: no cover - runtime dependency varies by environment
    import mifrost  # type: ignore
except Exception:
    mifrost = None  # type: ignore

from relm.models import ArityMLPFactory, CentralizedRelationalGNN, RelationalGNN


@dataclass
class Workload:
    batch: Any
    relation_dict: dict[str, int]
    symbol_type_ids: str | tuple[str, ...]
    meta: dict[str, Any]


def _resolve_device(query: str) -> torch.device:
    query = query.lower().strip()
    if query == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if query == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA but torch.cuda.is_available() is False.")
    if query == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS but torch.backends.mps.is_available() is False.")
    return torch.device(query)


def _build_synthetic_workload(
    *,
    seed: int,
    num_graphs: int,
    num_symbols: int,
    num_predicates: int,
    min_arity: int,
    max_arity: int,
    atoms_per_predicate: int,
    symbol_type: str = "_symbol_",
) -> Workload:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    relation_dict = {
        f"rel_{i}": int(
            torch.randint(min_arity, max_arity + 1, size=(1,), generator=gen).item()
        )
        for i in range(int(num_predicates))
    }

    graphs: list[HeteroData] = []
    for _ in range(int(num_graphs)):
        graph = HeteroData()
        graph[symbol_type].x = torch.zeros(int(num_symbols), 1, dtype=torch.float32)
        for pred, arity in relation_dict.items():
            n_atoms = int(atoms_per_predicate)
            graph[pred].x = torch.zeros(n_atoms, arity, dtype=torch.float32)
            atom_ids = torch.arange(n_atoms, dtype=torch.long)
            for pos in range(arity):
                src_ids = torch.randint(
                    0,
                    int(num_symbols),
                    size=(n_atoms,),
                    generator=gen,
                    dtype=torch.long,
                )
                edge_fwd = torch.stack([src_ids, atom_ids], dim=0)
                edge_bwd = torch.stack([atom_ids, src_ids], dim=0)
                graph[(symbol_type, str(pos), pred)].edge_index = edge_fwd
                graph[(pred, str(pos), symbol_type)].edge_index = edge_bwd
        graphs.append(graph)
    batch = Batch.from_data_list(graphs)
    return Workload(
        batch=batch,
        relation_dict=relation_dict,
        symbol_type_ids=symbol_type,
        meta={
            "source": "synthetic",
            "num_graphs": int(num_graphs),
            "num_symbols": int(num_symbols),
            "num_predicates": int(num_predicates),
            "atoms_per_predicate": int(atoms_per_predicate),
            "min_arity": int(min_arity),
            "max_arity": int(max_arity),
        },
    )


def _build_rgnet_workload(
    *,
    rgnet_root: str,
    domain_case: str,
    problem_case: str,
    max_states: int,
    seed: int,
) -> Workload:
    if mifrost is None:
        raise RuntimeError("rgnet workload mode requires mifrost.")
    rgnet_root_path = pathlib.Path(rgnet_root).expanduser().resolve()
    if not rgnet_root_path.exists():
        raise FileNotFoundError(f"rgnet root does not exist: {rgnet_root_path}")
    sys.path.insert(0, str(rgnet_root_path))
    sys.path.insert(0, str(rgnet_root_path / "src"))
    try:
        from test.test_utils import problem_setup  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime-only import
        raise RuntimeError(
            "Failed to import rgnet test helpers. "
            f"Expected test utilities under {rgnet_root_path}/test."
        ) from exc

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError("numpy is required for rgnet workload mode.") from exc

    space, domain, problem = problem_setup(domain_case, problem_case)
    rng = np.random.default_rng(int(seed))
    n_states = min(len(space), int(max_states))
    state_ids = rng.choice(len(space), size=n_states, replace=False)
    encoder = mifrost.HGraphEncoder(domain)
    batch = encoder.encode_batch(states=[space[int(state_id)] for state_id in state_ids])
    relation_dict = {k: int(v) for k, v in encoder.relation_dict.items()}
    return Workload(
        batch=batch,
        relation_dict=relation_dict,
        symbol_type_ids=encoder.symbol_type_id,
        meta={
            "source": "rgnet",
            "rgnet_root": str(rgnet_root_path),
            "domain_case": domain_case,
            "problem_case": problem_case,
            "n_states": int(n_states),
            "problem_name": getattr(problem, "name", str(problem_case)),
        },
    )


def _build_pddl_workload(
    *,
    pddl_root: str,
    domain_case: str,
    problem_case: str,
    max_states: int,
    seed: int,
) -> Workload:
    if mifrost is None:
        raise RuntimeError("pddl workload mode requires mifrost.")
    try:
        import pymimir  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime-only import
        raise RuntimeError("pddl workload mode requires pymimir.") from exc

    pddl_root_path = pathlib.Path(pddl_root).expanduser().resolve()
    domain_path = pddl_root_path / domain_case / "domain.pddl"
    problem_file = (
        problem_case if str(problem_case).endswith(".pddl") else f"{problem_case}.pddl"
    )
    problem_path = pddl_root_path / domain_case / problem_file
    if not domain_path.exists():
        raise FileNotFoundError(f"domain.pddl not found: {domain_path}")
    if not problem_path.exists():
        raise FileNotFoundError(f"problem file not found: {problem_path}")

    domain_obj = pymimir.Domain(domain_path)
    problem_obj = pymimir.Problem(domain_obj, problem_path, mode="lifted")
    state_space, _ = pymimir.advanced.datasets.StateSpace.create(
        problem_obj._search_context,
        pymimir.advanced.datasets.StateSpaceOptions(),
    )
    sampler = pymimir.wrapper_datasets.StateSpaceSampler(
        pymimir.advanced.datasets.StateSpaceSampler(state_space),
        problem_obj,
    )
    total_states = int(sampler.num_states())
    n_states = min(total_states, int(max_states))
    if n_states <= 0:
        raise RuntimeError(
            f"No states available for {domain_case}/{problem_case} at {pddl_root_path}."
        )
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    all_states = list(sampler.get_states())
    if len(all_states) != total_states:
        total_states = len(all_states)
        n_states = min(total_states, int(max_states))
    if n_states < total_states:
        state_ids = torch.randperm(total_states, generator=rng)[:n_states].tolist()
        states = [all_states[int(state_id)] for state_id in state_ids]
    else:
        states = all_states

    encoder = mifrost.HGraphEncoder(domain_obj)
    batch = encoder.encode_batch(states=states)
    relation_dict = {k: int(v) for k, v in encoder.relation_dict.items()}
    return Workload(
        batch=batch,
        relation_dict=relation_dict,
        symbol_type_ids=encoder.symbol_type_id,
        meta={
            "source": "pddl",
            "pddl_root": str(pddl_root_path),
            "domain_case": domain_case,
            "problem_case": problem_case,
            "n_states": int(n_states),
            "total_states": int(total_states),
        },
    )


def _make_model(
    *,
    model_kind: str,
    embedding_size: int,
    num_layer: int,
    aggr: str,
    relation_dict: dict[str, int],
    symbol_type_ids: str | tuple[str, ...],
    rel_layer_mode: str,
    central_layer_mode: str,
) -> torch.nn.Module:
    if model_kind == "decentralized":
        relation_factory = ArityMLPFactory(
            feature_size=int(embedding_size),
            residual=True,
            layers=2,
        )
        return RelationalGNN(
            embedding_size=int(embedding_size),
            num_layer=int(num_layer),
            aggr=aggr,
            symbol_type_ids=symbol_type_ids,
            relation_dict=relation_dict,
            relation_module_factory=relation_factory,
            rel_layer_mode=rel_layer_mode,
            compile_forward=False,
        )
    if model_kind == "centralized":
        condition_dim = max(1, int(math.sqrt(int(embedding_size))))
        central_factory = ArityMLPFactory(
            feature_size=int(embedding_size),
            in_condition_features=condition_dim,
            residual=True,
            layers=1,
        )
        return CentralizedRelationalGNN(
            embedding_size=int(embedding_size),
            num_layer=int(num_layer),
            aggr=aggr,
            symbol_type_ids=symbol_type_ids,
            relation_dict=relation_dict,
            relation_condition_dim=condition_dim,
            central_module_factory=central_factory,
            central_slot_mask=False,
            central_layer_mode=central_layer_mode,
            rel_layer_mode=rel_layer_mode,
            compile_forward=False,
        )
    raise ValueError(f"Unknown model kind: {model_kind!r}")


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_once(model: torch.nn.Module, batch: Any, *, backward: bool) -> None:
    out = model(batch)
    if not backward:
        return
    x_out = out[0] if isinstance(out, tuple) else out
    if not isinstance(x_out, dict) or not x_out:
        raise RuntimeError("Unexpected model output shape for backward benchmark.")
    loss = torch.stack([v.square().mean() for v in x_out.values()]).sum()
    loss.backward()


def _benchmark(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: torch.device,
    rounds: int,
    warmup: int,
    backward: bool,
) -> list[float]:
    model.train(mode=backward)
    for _ in range(int(warmup)):
        model.zero_grad(set_to_none=True)
        _run_once(model, batch, backward=backward)
        _sync_if_needed(device)
    times_ms: list[float] = []
    for _ in range(int(rounds)):
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        _run_once(model, batch, backward=backward)
        _sync_if_needed(device)
        end = time.perf_counter()
        times_ms.append((end - start) * 1_000.0)
    return times_ms


def _describe_tensor_counts(batch: Any) -> dict[str, int]:
    view = batch.as_hetero() if hasattr(batch, "as_hetero") else batch
    x_dict = getattr(view, "x_dict", {})
    edge_index_dict = getattr(view, "edge_index_dict", {})
    total_nodes = int(sum(x.size(0) for x in x_dict.values())) if x_dict else 0
    total_edges = (
        int(sum(edge.size(1) for edge in edge_index_dict.values()))
        if edge_index_dict
        else 0
    )
    return {"total_nodes": total_nodes, "total_edges": total_edges}


def _summarize(times_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(times_ms),
        "stdev_ms": statistics.pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "p50_ms": statistics.median(times_ms),
    }


def _print_summary(
    model_kind: str,
    stats: dict[str, float],
    *,
    params: int,
    meta: dict[str, Any],
    counts: dict[str, int],
) -> None:
    print(f"[{model_kind}]")
    print(
        "  runtime_ms"
        f" mean={stats['mean_ms']:.3f}"
        f" p50={stats['p50_ms']:.3f}"
        f" min={stats['min_ms']:.3f}"
        f" max={stats['max_ms']:.3f}"
        f" stdev={stats['stdev_ms']:.3f}"
    )
    print(f"  params={params}")
    print(f"  total_nodes={counts['total_nodes']} total_edges={counts['total_edges']}")
    print("  workload_meta=" + ", ".join(f"{k}={v}" for k, v in sorted(meta.items())))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RelationalGNN and CentralizedRelationalGNN."
    )
    parser.add_argument(
        "--workload",
        choices=("synthetic", "rgnet", "pddl"),
        default=os.getenv("RELM_GNN_WORKLOAD", "synthetic"),
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
    parser.add_argument("--aggr", default=os.getenv("RELM_GNN_AGGR", "sum"))
    parser.add_argument("--rounds", type=int, default=int(os.getenv("RELM_GNN_ROUNDS", "20")))
    parser.add_argument("--warmup", type=int, default=int(os.getenv("RELM_GNN_WARMUP", "5")))
    parser.add_argument("--backward", action="store_true")
    parser.add_argument(
        "--model-kinds",
        default=os.getenv("RELM_GNN_MODEL_KINDS", "decentralized,centralized"),
        help="Comma-separated list of model kinds.",
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
    parser.add_argument("--max-states", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = _resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))

    if args.workload == "synthetic":
        workload = _build_synthetic_workload(
            seed=args.seed,
            num_graphs=args.num_graphs,
            num_symbols=args.num_symbols,
            num_predicates=args.num_predicates,
            min_arity=args.min_arity,
            max_arity=args.max_arity,
            atoms_per_predicate=args.atoms_per_predicate,
            symbol_type=args.symbol_type,
        )
    elif args.workload == "rgnet":
        workload = _build_rgnet_workload(
            rgnet_root=args.rgnet_root,
            domain_case=args.domain_case,
            problem_case=args.problem_case,
            max_states=args.max_states,
            seed=args.seed,
        )
    else:
        workload = _build_pddl_workload(
            pddl_root=args.pddl_root,
            domain_case=args.domain_case,
            problem_case=args.problem_case,
            max_states=args.max_states,
            seed=args.seed,
        )

    batch = workload.batch.to(device) if hasattr(workload.batch, "to") else workload.batch
    counts = _describe_tensor_counts(batch)
    model_kinds = [k.strip() for k in args.model_kinds.split(",") if k.strip()]
    print(
        f"device={device.type} workload={args.workload} rounds={args.rounds} "
        f"warmup={args.warmup} backward={bool(args.backward)}"
    )
    print(
        f"embedding_size={args.embedding_size} num_layer={args.num_layer} "
        f"aggr={args.aggr} rel_layer_mode={args.rel_layer_mode} "
        f"central_layer_mode={args.central_layer_mode}"
    )
    for model_kind in model_kinds:
        model = _make_model(
            model_kind=model_kind,
            embedding_size=args.embedding_size,
            num_layer=args.num_layer,
            aggr=args.aggr,
            relation_dict=workload.relation_dict,
            symbol_type_ids=workload.symbol_type_ids,
            rel_layer_mode=args.rel_layer_mode,
            central_layer_mode=args.central_layer_mode,
        ).to(device)
        times_ms = _benchmark(
            model,
            batch,
            device=device,
            rounds=args.rounds,
            warmup=args.warmup,
            backward=bool(args.backward),
        )
        stats = _summarize(times_ms)
        params = sum(p.numel() for p in model.parameters())
        _print_summary(
            model_kind,
            stats,
            params=params,
            meta=workload.meta,
            counts=counts,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
