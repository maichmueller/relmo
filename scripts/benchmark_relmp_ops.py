from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class CaseSpec:
    name: str
    emb: int
    n_symbols: int
    total_slots: int
    edges: int
    dst_dim_size: int


@dataclass
class BenchRow:
    case: str
    op: str
    pass_kind: str
    impl: str
    mean_ms: float
    median_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    speedup_vs_python: float | None


CASES: dict[str, CaseSpec] = {
    "tiny": CaseSpec(
        name="tiny",
        emb=64,
        n_symbols=2048,
        total_slots=8192,
        edges=8192,
        dst_dim_size=1024,
    ),
    "small": CaseSpec(
        name="small",
        emb=128,
        n_symbols=8192,
        total_slots=32768,
        edges=32768,
        dst_dim_size=4096,
    ),
    "medium": CaseSpec(
        name="medium",
        emb=256,
        n_symbols=16384,
        total_slots=65536,
        edges=65536,
        dst_dim_size=8192,
    ),
}


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(
    fn: Callable[[], Any],
    *,
    device: torch.device,
    warmup: int,
    repeats: int,
    inner_iters: int,
) -> list[float]:
    inner_iters = max(1, int(inner_iters))
    for _ in range(max(0, warmup)):
        _ = fn()
        _sync_if_cuda(device)

    durations_ms: list[float] = []
    for _ in range(max(1, repeats)):
        _sync_if_cuda(device)
        t0 = time.perf_counter_ns()
        last = None
        for __ in range(inner_iters):
            last = fn()
        _ = last
        _sync_if_cuda(device)
        elapsed_ms = ((time.perf_counter_ns() - t0) / 1e6) / inner_iters
        durations_ms.append(elapsed_ms)
    return durations_ms


def _summarize(
    *,
    case: str,
    op: str,
    pass_kind: str,
    impl: str,
    durations_ms: list[float],
) -> BenchRow:
    return BenchRow(
        case=case,
        op=op,
        pass_kind=pass_kind,
        impl=impl,
        mean_ms=statistics.fmean(durations_ms),
        median_ms=statistics.median(durations_ms),
        stdev_ms=statistics.pstdev(durations_ms),
        min_ms=min(durations_ms),
        max_ms=max(durations_ms),
        speedup_vs_python=None,
    )


def _fanout_python(
    x_cat: torch.Tensor,
    src_global_idx: torch.Tensor,
    flat_dst: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    out = x_cat.new_zeros((int(out_rows), int(x_cat.size(-1))))
    if src_global_idx.numel() == 0 or int(out_rows) == 0:
        return out
    values = x_cat.index_select(0, src_global_idx)
    out.index_copy_(0, flat_dst, values)
    return out


def _fanin_python(
    rel_flat: torch.Tensor,
    flat_src: torch.Tensor,
    dst_idx: torch.Tensor,
    dim_size: int,
    mode: int,
) -> torch.Tensor:
    emb = int(rel_flat.size(-1))
    if mode == 0:
        out = rel_flat.new_zeros((int(dim_size), emb))
        if flat_src.numel() == 0 or int(dim_size) == 0:
            return out
        values = rel_flat.index_select(0, flat_src)
        out.index_add_(0, dst_idx, values)
        return out
    if mode == 1:
        out = rel_flat.new_full((int(dim_size), emb), float("-inf"))
        if flat_src.numel() == 0 or int(dim_size) == 0:
            return out
        values = rel_flat.index_select(0, flat_src)
        index = dst_idx.view(-1, 1).expand(-1, emb)
        amax = rel_flat.new_full((int(dim_size), emb), float("-inf"))
        amax.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
        offsets = amax.index_select(0, dst_idx)
        exps = (values - offsets).exp()
        exps_sum = rel_flat.new_zeros((int(dim_size), emb))
        exps_sum.scatter_add_(0, index, exps)
        return exps_sum.log() + amax
    raise ValueError(f"Unsupported mode: {mode!r}")


def _import_relmp(import_mode: str) -> Any:
    if import_mode == "source":
        root = Path(__file__).resolve().parents[1]
        src = root / "src"
        if str(src) not in sys.path:
            sys.path.insert(0, str(src))
    elif import_mode != "installed":
        raise ValueError(f"Unsupported import mode: {import_mode!r}")

    from relm.ops import relmp

    return relmp


def _sample_inputs(
    spec: CaseSpec, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(0)
    x_cat = torch.randn(
        spec.n_symbols, spec.emb, device=device, dtype=dtype, generator=g
    )
    rel_flat = torch.randn(
        spec.total_slots, spec.emb, device=device, dtype=dtype, generator=g
    )

    src_global_idx = torch.randint(
        low=0,
        high=spec.n_symbols,
        size=(spec.edges,),
        device=device,
        dtype=torch.int64,
        generator=g,
    )
    flat_src = torch.randint(
        low=0,
        high=spec.total_slots,
        size=(spec.edges,),
        device=device,
        dtype=torch.int64,
        generator=g,
    )
    dst_idx = torch.randint(
        low=0,
        high=spec.dst_dim_size,
        size=(spec.edges,),
        device=device,
        dtype=torch.int64,
        generator=g,
    )

    perm = torch.randperm(
        spec.total_slots, device=device, generator=g, dtype=torch.int64
    )
    flat_dst = perm[: spec.edges].contiguous()
    return {
        "x_cat": x_cat,
        "src_global_idx": src_global_idx,
        "flat_dst": flat_dst,
        "rel_flat": rel_flat,
        "flat_src": flat_src,
        "dst_idx": dst_idx,
    }


def _make_forward_fn(
    op: str, impl: str, relmp: Any, data: dict[str, torch.Tensor], spec: CaseSpec
):
    if op == "fanout":
        if impl == "python":
            return lambda: _fanout_python(
                data["x_cat"],
                data["src_global_idx"],
                data["flat_dst"],
                spec.total_slots,
            )
        return lambda: relmp.fanout_scatter(
            data["x_cat"], data["src_global_idx"], data["flat_dst"], spec.total_slots
        )

    mode = 0 if op == "fanin_sum" else 1
    if impl == "python":
        return lambda: _fanin_python(
            data["rel_flat"], data["flat_src"], data["dst_idx"], spec.dst_dim_size, mode
        )
    return lambda: relmp.fanin_reduce(
        data["rel_flat"], data["flat_src"], data["dst_idx"], spec.dst_dim_size, mode
    )


def _make_forward_backward_fn(
    op: str, impl: str, relmp: Any, data: dict[str, torch.Tensor], spec: CaseSpec
):
    if op == "fanout":

        def _run() -> torch.Tensor:
            x = data["x_cat"].detach().clone().requires_grad_(True)
            if impl == "python":
                out = _fanout_python(
                    x, data["src_global_idx"], data["flat_dst"], spec.total_slots
                )
            else:
                out = relmp.fanout_scatter(
                    x, data["src_global_idx"], data["flat_dst"], spec.total_slots
                )
            out.sum().backward()
            return x.grad  # type: ignore[return-value]

        return _run

    mode = 0 if op == "fanin_sum" else 1

    def _run() -> torch.Tensor:
        x = data["rel_flat"].detach().clone().requires_grad_(True)
        if impl == "python":
            out = _fanin_python(
                x, data["flat_src"], data["dst_idx"], spec.dst_dim_size, mode
            )
        else:
            out = relmp.fanin_reduce(
                x, data["flat_src"], data["dst_idx"], spec.dst_dim_size, mode
            )
        out.sum().backward()
        return x.grad  # type: ignore[return-value]

    return _run


def _compute_speedups(rows: list[BenchRow]) -> None:
    key_to_python: dict[tuple[str, str, str], float] = {}
    for row in rows:
        if row.impl == "python":
            key_to_python[(row.case, row.op, row.pass_kind)] = row.median_ms
    for row in rows:
        base = key_to_python.get((row.case, row.op, row.pass_kind))
        if base is None or row.median_ms <= 0:
            row.speedup_vs_python = None
        else:
            row.speedup_vs_python = base / row.median_ms


def _print_rows(rows: list[BenchRow]) -> None:
    if not rows:
        return
    print(
        "case     op            pass        impl      mean_ms   median_ms  "
        "stdev_ms  speedup_vs_python"
    )
    for row in rows:
        speed = (
            f"{row.speedup_vs_python:>8.3f}x"
            if row.speedup_vs_python is not None
            else "    n/a"
        )
        print(
            f"{row.case:<8} {row.op:<13} {row.pass_kind:<11} {row.impl:<9} "
            f"{row.mean_ms:>8.4f}  {row.median_ms:>9.4f}  {row.stdev_ms:>8.4f}  {speed}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark stable-ABI relmp custom ops vs Python references for fanout/fanin."
        )
    )
    parser.add_argument(
        "--import-mode",
        choices=("installed", "source"),
        default="installed",
        help="How to resolve relm import path.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Benchmark device.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Tensor dtype for benchmark payloads.",
    )
    parser.add_argument(
        "--cases",
        default="tiny,small",
        help="Comma-separated case names: tiny,small,medium",
    )
    parser.add_argument(
        "--ops",
        default="fanout,fanin_sum,fanin_logsumexp",
        help="Comma-separated ops: fanout,fanin_sum,fanin_logsumexp",
    )
    parser.add_argument(
        "--passes",
        default="forward,forward_backward",
        help="Comma-separated passes: forward,forward_backward",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=80)
    parser.add_argument("--inner-iters", type=int, default=5)
    parser.add_argument(
        "--require-custom",
        action="store_true",
        help="Fail if custom relmp ops are not available.",
    )
    parser.add_argument("--json-out", type=str, default="")
    args = parser.parse_args()

    relmp = _import_relmp(args.import_mode)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Requested CUDA benchmark but torch.cuda.is_available() is False."
        )

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    if device.type == "cpu" and dtype in {torch.float16, torch.bfloat16}:
        print(f"Warning: dtype={args.dtype} on CPU can be slower and less stable.")

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    os.environ["RELM_RELMP_ENABLE"] = "1"
    os.environ["RELM_RELMP_FALLBACK"] = "error"
    custom_available = bool(relmp.available())
    if args.require_custom and not custom_available:
        raise RuntimeError(
            "Custom relmp ops are unavailable in this environment. "
            "Use torch>=2.10 and build with RELM_RELMP_OPS enabled."
        )

    requested_cases = [x.strip().lower() for x in args.cases.split(",") if x.strip()]
    requested_ops = [x.strip().lower() for x in args.ops.split(",") if x.strip()]
    requested_passes = [x.strip().lower() for x in args.passes.split(",") if x.strip()]

    for case_name in requested_cases:
        if case_name not in CASES:
            raise ValueError(f"Unknown case {case_name!r}.")
    for op in requested_ops:
        if op not in {"fanout", "fanin_sum", "fanin_logsumexp"}:
            raise ValueError(f"Unknown op {op!r}.")
    for pass_kind in requested_passes:
        if pass_kind not in {"forward", "forward_backward"}:
            raise ValueError(f"Unknown pass kind {pass_kind!r}.")

    print(
        f"Benchmarking relm relmp on device={device}, dtype={dtype}, "
        f"torch={torch.__version__}, custom_available={custom_available}"
    )

    rows: list[BenchRow] = []
    impls = ["python"] + (["custom"] if custom_available else [])

    for case_name in requested_cases:
        spec = CASES[case_name]
        data = _sample_inputs(spec, device, dtype)

        for op in requested_ops:
            for pass_kind in requested_passes:
                for impl in impls:
                    if pass_kind == "forward":
                        fn = _make_forward_fn(op, impl, relmp, data, spec)
                    else:
                        fn = _make_forward_backward_fn(op, impl, relmp, data, spec)
                    durations = _bench(
                        fn,
                        device=device,
                        warmup=args.warmup,
                        repeats=args.repeats,
                        inner_iters=args.inner_iters,
                    )
                    rows.append(
                        _summarize(
                            case=case_name,
                            op=op,
                            pass_kind=pass_kind,
                            impl=impl,
                            durations_ms=durations,
                        )
                    )

    _compute_speedups(rows)
    _print_rows(rows)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "device": str(device),
                "dtype": args.dtype,
                "torch_version": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "custom_available": custom_available,
                "cases": requested_cases,
                "ops": requested_ops,
                "passes": requested_passes,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "inner_iters": args.inner_iters,
            },
            "rows": [asdict(r) for r in rows],
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote benchmark JSON to {out_path}")


if __name__ == "__main__":
    main()
