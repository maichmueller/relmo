from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time
from dataclasses import asdict, dataclass

import torch

import mifrost  # type: ignore

from relmo.models import (
    FlatExecutionPolicy,
    FlatLGANRelationalGNN,
    PostNormTwoLayerPointwiseRelationMLP,
    TwoLayerPointwiseRelationMLP,
)
from scripts.benchmark_fast_vs_pymimir import _build_states


@dataclass
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


def _resolve_problem_case(root: pathlib.Path, domain_case: str, problem_case: str | None) -> str:
    if problem_case:
        return problem_case
    problem_files = sorted(
        path.stem
        for path in (root / domain_case).glob("*.pddl")
        if path.name != "domain.pddl" and not path.name.startswith("._")
    )
    if not problem_files:
        raise FileNotFoundError(f"no problem files found under {(root / domain_case)!s}")
    return problem_files[0]


def _device_from_arg(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_call(fn, *, device: torch.device, rounds: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        _sync(device)
        fn()
        _sync(device)
    times: list[float] = []
    for _ in range(rounds):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - start) * 1000.0)
    return times


def _stats(times_ms: list[float]) -> TimingStats:
    return TimingStats(
        mean_ms=float(statistics.fmean(times_ms)),
        median_ms=float(statistics.median(times_ms)),
        min_ms=float(min(times_ms)),
        max_ms=float(max(times_ms)),
    )


def _build_model(*, batch, embedding_size: int, num_layers: int, device: torch.device):
    return _build_model_with_policy(
        batch=batch,
        embedding_size=embedding_size,
        num_layers=num_layers,
        device=device,
        execution_policy=FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        ),
    )


def _build_model_with_policy(
    *,
    batch,
    embedding_size: int,
    num_layers: int,
    device: torch.device,
    execution_policy: FlatExecutionPolicy,
    relation_family: str = "pointwise",
    activation: str = "silu",
):
    relation_names = tuple(str(name) for name in batch.relation_names)
    relation_arities = tuple(int(arity) for arity in batch.relation_arities)
    relations = dict(zip(relation_names, relation_arities))
    relation_modules = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(width, 2 * embedding_size)
        if relation_family == "pointwise":
            module = TwoLayerPointwiseRelationMLP(
                width,
                hidden,
                activation=activation,
            )
        elif relation_family == "postnorm_ln":
            module = PostNormTwoLayerPointwiseRelationMLP(
                width,
                hidden,
                activation=activation,
                norm="layernorm",
            )
        else:
            raise ValueError(f"Unsupported relation_family: {relation_family!r}")
        relation_modules[name] = module
    model = FlatLGANRelationalGNN(
        embedding_size=embedding_size,
        num_layers=num_layers,
        relations=relations,
        aggregation="sum",
        relation_modules=relation_modules,
        execution_policy=execution_policy,
    ).to(device)
    return model


def _run_forward(model, batch, *, device: torch.device) -> None:
    out = model(batch)
    loss = out.entity.sum()
    if device.type == "cuda":
        # Keep a device-side consumer in the timed region.
        loss = loss + out.entity_batch.float().sum()
    _ = loss


def _run_backward(model, batch, *, device: torch.device) -> None:
    model.zero_grad(set_to_none=True)
    out = model(batch)
    loss = out.entity.sum()
    loss.backward()


def _print_table(results: dict[str, TimingStats]) -> None:
    header = f"{'lane':<16} {'mean_ms':>10} {'median_ms':>10} {'min_ms':>10} {'max_ms':>10}"
    print(header)
    print("-" * len(header))
    for key, stats in results.items():
        print(
            f"{key:<16} {stats.mean_ms:10.3f} {stats.median_ms:10.3f} "
            f"{stats.min_ms:10.3f} {stats.max_ms:10.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser.add_argument("--pddl-root", type=str, default=str(repo_root / "data" / "pddl_domains"))
    parser.add_argument("--domain-case", type=str, default="blocks")
    parser.add_argument("--problem-case", type=str, default=None)
    parser.add_argument("--max-states", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--embedding-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--relation-kernels", choices=("auto", "off"), default="off")
    parser.add_argument("--program-kernels", choices=("auto", "off"), default="off")
    parser.add_argument("--relation-gather", choices=("auto", "on", "off"), default="off")
    parser.add_argument("--relation-family", choices=("pointwise", "postnorm_ln"), default="pointwise")
    parser.add_argument("--activation", choices=("silu", "mish", "gelu"), default="silu")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    pddl_root = pathlib.Path(args.pddl_root).expanduser().resolve()
    problem_case = _resolve_problem_case(pddl_root, args.domain_case, args.problem_case)
    device = _device_from_arg(args.device)

    domain, problem, states = _build_states(
        pddl_root=str(pddl_root),
        domain_case=args.domain_case,
        problem_case=problem_case,
        max_states=int(args.max_states),
        seed=int(args.seed),
    )
    goals = list(problem.get_goal_condition().get_literals())
    encoder = mifrost.FlatRelationEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=["goal"],
    )
    native_batch = encoder.encode_batch(states=states, goals=goals).to(device)
    pyg_batch = native_batch.as_pyg(as_batch=True).to(device)

    execution_policy = FlatExecutionPolicy(
        relation_kernels=args.relation_kernels,
        program_kernels=args.program_kernels,
        relation_gather=args.relation_gather,
    )
    model = _build_model_with_policy(
        batch=native_batch,
        embedding_size=int(args.embedding_size),
        num_layers=int(args.num_layers),
        device=device,
        execution_policy=execution_policy,
        relation_family=args.relation_family,
        activation=args.activation,
    )

    results = {
        "native_fwd": _stats(
            _time_call(
                lambda: _run_forward(model, native_batch, device=device),
                device=device,
                rounds=int(args.rounds),
                warmup=int(args.warmup),
            )
        ),
        "pyg_fwd": _stats(
            _time_call(
                lambda: _run_forward(model, pyg_batch, device=device),
                device=device,
                rounds=int(args.rounds),
                warmup=int(args.warmup),
            )
        ),
        "native_bwd": _stats(
            _time_call(
                lambda: _run_backward(model, native_batch, device=device),
                device=device,
                rounds=int(args.rounds),
                warmup=int(args.warmup),
            )
        ),
        "pyg_bwd": _stats(
            _time_call(
                lambda: _run_backward(model, pyg_batch, device=device),
                device=device,
                rounds=int(args.rounds),
                warmup=int(args.warmup),
            )
        ),
    }

    payload = {
        "config": {
            "pddl_root": str(pddl_root),
            "domain_case": args.domain_case,
            "problem_case": problem_case,
            "max_states": int(args.max_states),
            "seed": int(args.seed),
            "embedding_size": int(args.embedding_size),
            "num_layers": int(args.num_layers),
            "device": str(device),
            "relation_kernels": args.relation_kernels,
            "program_kernels": args.program_kernels,
            "relation_gather": args.relation_gather,
            "relation_family": args.relation_family,
            "activation": args.activation,
        },
        "carrier_meta": {
            "num_graphs": int(getattr(native_batch, "num_graphs", 1)),
            "num_nodes": int(getattr(native_batch, "num_nodes", 0)),
            "relation_args": int(native_batch.relation_args.numel()),
            "relation_instances": int(native_batch.relation_instance_sizes.sum().item()),
            "lgan_tn_edges": int(native_batch.lgan_tn_sizes.sum().item()),
            "lgan_nn_edges": int(native_batch.lgan_nn_sizes.sum().item()),
            "lgan_rr_edges": int(native_batch.lgan_rr_sizes.sum().item()),
        },
        "results": {key: asdict(value) for key, value in results.items()},
    }

    _print_table(results)
    print()
    print(
        "diff(native-pyg): "
        f"fwd={results['native_fwd'].mean_ms - results['pyg_fwd'].mean_ms:.3f} ms, "
        f"bwd={results['native_bwd'].mean_ms - results['pyg_bwd'].mean_ms:.3f} ms"
    )

    if args.json_out:
        out_path = pathlib.Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
