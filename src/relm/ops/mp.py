"""Relational message-passing operator wrappers."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict

try:  # pragma: no cover - exercised in environments without torch
    import torch
except Exception as exc:  # pragma: no cover - exercised in minimal wheels
    torch = None
    _TORCH_IMPORT_ERROR: Exception | None = exc
else:
    _TORCH_IMPORT_ERROR = None

_LIB_LOADED = False
_LIB_LOAD_ERROR: Exception | None = None
_BUILD_INFO_CACHE: Dict[str, str] | None = None
_OPS_NAMESPACE = "relm_mp"
_RUNTIME_COMPAT_VALIDATED = False

_BOOL_FALSE = {"0", "false", "no", "off"}
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")

_MODE_SUM = 0
_MODE_LOGSUMEXP = 1

_REQUIRED_NAMESPACE_OPS = (
    "fanout_scatter",
    "fanout_scatter_backward",
    "fanin_reduce",
    "fanin_reduce_sum_backward",
    "fanin_reduce_logsumexp_backward",
    "build_info",
)


def _env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() not in _BOOL_FALSE
    return default


def _env_first(names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def _fallback_mode() -> str:
    raw = (_env_first(("RELM_MP_FALLBACK", "RELM_RELMP_FALLBACK"), "python") or "python")
    raw = raw.strip().lower()
    return raw if raw in {"python", "error"} else "python"


def _runtime_cuda_tag() -> str:
    assert torch is not None
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return "cpu"
    return f"cu{str(cuda).replace('.', '')}"


def _major_minor(version: str) -> tuple[int, int]:
    match = _VERSION_RE.match(version)
    if match is None:
        raise RuntimeError(f"Could not parse torch version {version!r}.")
    return int(match.group(1)), int(match.group(2))


def _parse_build_info(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _candidate_libraries() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent.parent
    patterns = (
        "relm_mp_ops*.so",
        "relm_mp_ops*.dylib",
        "relm_mp_ops*.pyd",
        "librelm_mp_ops*.so",
        "librelm_mp_ops*.dylib",
        "relm_relmp_ops*.so",
        "relm_relmp_ops*.dylib",
        "relm_relmp_ops*.pyd",
        "librelm_relmp_ops*.so",
        "librelm_relmp_ops*.dylib",
    )
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(pkg_dir.glob(pattern)))
    deduped: list[Path] = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def _ensure_loaded() -> None:
    global _LIB_LOADED, _LIB_LOAD_ERROR, _OPS_NAMESPACE
    if _LIB_LOADED:
        return
    if torch is None:
        raise ModuleNotFoundError(
            "relm.ops.mp requires torch."
        ) from _TORCH_IMPORT_ERROR

    last_error: Exception | None = None
    for candidate in _candidate_libraries():
        try:
            torch.ops.load_library(str(candidate))

            namespace = getattr(torch.ops, "relm_mp", None)
            if namespace is None or any(
                not hasattr(namespace, op_name) for op_name in _REQUIRED_NAMESPACE_OPS
            ):
                raise RuntimeError(
                    f"Loaded {candidate} but torch.ops.relm_mp is incomplete."
                )
            _OPS_NAMESPACE = "relm_mp"
            _LIB_LOADED = True
            _LIB_LOAD_ERROR = None
            return
        except Exception as exc:  # pragma: no cover - depends on local build state
            last_error = exc

    _LIB_LOAD_ERROR = last_error
    if last_error is not None:
        raise RuntimeError(
            "Failed to load relm_mp custom op library."
        ) from last_error
    raise FileNotFoundError(
        "Could not find relm_mp custom op library in the relm package directory."
    )


def _ops_namespace():
    assert torch is not None
    return getattr(torch.ops, _OPS_NAMESPACE)


def _ensure_runtime_compat_once() -> None:
    global _RUNTIME_COMPAT_VALIDATED
    if _RUNTIME_COMPAT_VALIDATED:
        return
    assert_runtime_compat()
    _RUNTIME_COMPAT_VALIDATED = True


def assert_runtime_compat() -> Dict[str, str]:
    """Raise if runtime torch/cuda is incompatible with the built op library."""

    global _BUILD_INFO_CACHE
    if torch is None:
        raise ModuleNotFoundError("torch is required for mp compatibility checks.")
    _ensure_loaded()
    if _BUILD_INFO_CACHE is None:
        info_raw = _ops_namespace().build_info()
        _BUILD_INFO_CACHE = _parse_build_info(info_raw)

    info = _BUILD_INFO_CACHE
    build_torch = info.get("build_torch")
    if not build_torch:
        raise RuntimeError("mp build metadata is missing build_torch.")
    runtime_torch = str(torch.__version__)
    build_mm = _major_minor(build_torch)
    runtime_mm = _major_minor(runtime_torch)
    if build_mm != runtime_mm:
        raise RuntimeError(
            "mp torch version mismatch: "
            f"built against {build_torch}, runtime is {runtime_torch}. "
            "Expected matching major.minor."
        )

    build_cuda = info.get("build_cuda_tag", "cpu")
    runtime_cuda = _runtime_cuda_tag()
    if build_cuda != "cpu" and build_cuda != runtime_cuda:
        raise RuntimeError(
            "mp CUDA tag mismatch: "
            f"built for {build_cuda}, runtime is {runtime_cuda}."
        )
    return info


def available() -> bool:
    """Return True when custom ops are loadable and runtime-compatible."""

    if torch is None:
        return False
    try:
        _ensure_loaded()
        assert_runtime_compat()
    except Exception:
        return False
    return True


def _should_use_custom(op_name: str) -> bool:
    if not _env_bool_any(("RELM_MP_ENABLE", "RELM_RELMP_ENABLE"), True):
        return False
    try:
        _ensure_runtime_compat_once()
        return True
    except Exception as exc:
        if _fallback_mode() == "error":
            raise RuntimeError(f"Custom mp op {op_name} is unavailable.") from exc
        return False


def _fanout_scatter_python(
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


def _fanin_reduce_python(
    rel_flat: torch.Tensor,
    flat_src: torch.Tensor,
    dst_idx: torch.Tensor,
    dim_size: int,
    mode: int,
) -> torch.Tensor:
    emb = int(rel_flat.size(-1))
    if mode == _MODE_SUM:
        out = rel_flat.new_zeros((int(dim_size), emb))
        if flat_src.numel() == 0 or int(dim_size) == 0:
            return out
        values = rel_flat.index_select(0, flat_src)
        out.index_add_(0, dst_idx, values)
        return out

    if mode == _MODE_LOGSUMEXP:
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

    raise ValueError(f"Unsupported fanin mode {mode!r}. Supported: 0=sum, 1=logsumexp.")


if torch is not None:

    class _FanoutScatterFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x_cat: torch.Tensor,
            src_global_idx: torch.Tensor,
            flat_dst: torch.Tensor,
            out_rows: int,
        ) -> torch.Tensor:
            ctx.save_for_backward(src_global_idx, flat_dst)
            ctx.x_rows = int(x_cat.size(0))
            return _ops_namespace().fanout_scatter(
                x_cat, src_global_idx, flat_dst, int(out_rows)
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor, None, None, None]:
            src_global_idx, flat_dst = ctx.saved_tensors
            grad_x = _ops_namespace().fanout_scatter_backward(
                grad_out, src_global_idx, flat_dst, int(ctx.x_rows)
            )
            return grad_x, None, None, None

    class _FaninReduceSumFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            rel_flat: torch.Tensor,
            flat_src: torch.Tensor,
            dst_idx: torch.Tensor,
            dim_size: int,
        ) -> torch.Tensor:
            ctx.save_for_backward(flat_src, dst_idx)
            ctx.rel_rows = int(rel_flat.size(0))
            return _ops_namespace().fanin_reduce(
                rel_flat, flat_src, dst_idx, int(dim_size), _MODE_SUM
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor, None, None, None]:
            flat_src, dst_idx = ctx.saved_tensors
            grad_rel = _ops_namespace().fanin_reduce_sum_backward(
                grad_out, flat_src, dst_idx, int(ctx.rel_rows)
            )
            return grad_rel, None, None, None

    class _FaninReduceLogSumExpFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            rel_flat: torch.Tensor,
            flat_src: torch.Tensor,
            dst_idx: torch.Tensor,
            dim_size: int,
        ) -> torch.Tensor:
            ctx.rel_rows = int(rel_flat.size(0))
            out = _ops_namespace().fanin_reduce(
                rel_flat, flat_src, dst_idx, int(dim_size), _MODE_LOGSUMEXP
            )
            ctx.save_for_backward(rel_flat, flat_src, dst_idx, out)
            return out

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor, None, None, None]:
            rel_flat, flat_src, dst_idx, out = ctx.saved_tensors
            grad_rel = _ops_namespace().fanin_reduce_logsumexp_backward(
                grad_out,
                rel_flat,
                flat_src,
                dst_idx,
                out,
                int(ctx.rel_rows),
            )
            return grad_rel, None, None, None


def fanout_scatter(
    x_cat: torch.Tensor,
    src_global_idx: torch.Tensor,
    flat_dst: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError(
            "fanout_scatter requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("fanout_scatter"):
        return _FanoutScatterFunction.apply(
            x_cat, src_global_idx, flat_dst, int(out_rows)
        )
    return _fanout_scatter_python(x_cat, src_global_idx, flat_dst, int(out_rows))


def fanin_reduce(
    rel_flat: torch.Tensor,
    flat_src: torch.Tensor,
    dst_idx: torch.Tensor,
    dim_size: int,
    mode: int,
) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError(
            "fanin_reduce requires torch."
        ) from _TORCH_IMPORT_ERROR
    mode_int = int(mode)
    if mode_int not in (_MODE_SUM, _MODE_LOGSUMEXP):
        return _fanin_reduce_python(
            rel_flat, flat_src, dst_idx, int(dim_size), mode_int
        )
    if _should_use_custom("fanin_reduce"):
        if mode_int == _MODE_SUM:
            return _FaninReduceSumFunction.apply(
                rel_flat, flat_src, dst_idx, int(dim_size)
            )
        return _FaninReduceLogSumExpFunction.apply(
            rel_flat, flat_src, dst_idx, int(dim_size)
        )
    return _fanin_reduce_python(rel_flat, flat_src, dst_idx, int(dim_size), mode_int)


__all__ = [
    "fanout_scatter",
    "fanin_reduce",
    "available",
    "assert_runtime_compat",
]
