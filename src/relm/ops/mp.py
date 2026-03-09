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

_PW_IDENTITY = 0
_PW_RELU = 1
_PW_MISH = 2
_PW_GELU_NONE = 3
_PW_GELU_TANH = 4
_PW_SILU = 5
_PW_TANH = 6

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


def pointwise_code_from_signature(signature: tuple[object, ...] | None) -> int | None:
    if not signature:
        return None
    kind = str(signature[0])
    if kind == "identity":
        return _PW_IDENTITY
    if kind == "relu":
        return _PW_RELU
    if kind == "mish":
        return _PW_MISH
    if kind == "gelu":
        approximate = str(signature[1]) if len(signature) > 1 else "none"
        if approximate == "none":
            return _PW_GELU_NONE
        if approximate == "tanh":
            return _PW_GELU_TANH
        return None
    if kind == "silu":
        return _PW_SILU
    if kind == "tanh":
        return _PW_TANH
    return None


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


def _namespace_has_op(op_name: str) -> bool:
    if torch is None:
        return False
    try:
        _ensure_loaded()
    except Exception:
        return False
    return hasattr(_ops_namespace(), op_name)


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


def _fanout_pack_multi_python(
    x_parts: list[torch.Tensor],
    src_idx_parts: list[torch.Tensor],
    flat_dst_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not x_parts:
        raise ValueError("fanout_pack_multi requires at least one source tensor.")
    if not (len(x_parts) == len(src_idx_parts) == len(flat_dst_parts)):
        raise ValueError(
            "fanout_pack_multi expects x_parts, src_idx_parts, and flat_dst_parts with equal lengths."
        )
    row_offset = 0
    src_global_parts: list[torch.Tensor] = []
    x_cat_parts: list[torch.Tensor] = []
    dst_cat_parts: list[torch.Tensor] = []
    for x, src_idx, flat_dst in zip(x_parts, src_idx_parts, flat_dst_parts):
        if x.dim() != 2:
            raise ValueError("fanout_pack_multi expects each source tensor to be rank-2.")
        if src_idx.dim() != 1 or flat_dst.dim() != 1:
            raise ValueError("fanout_pack_multi expects rank-1 source/destination index tensors.")
        if src_idx.dtype != torch.int64 or flat_dst.dtype != torch.int64:
            raise ValueError("fanout_pack_multi expects int64 source/destination index tensors.")
        if src_idx.numel() != flat_dst.numel():
            raise ValueError(
                "fanout_pack_multi expects src_idx and flat_dst lengths to match per source."
            )
        x_cat_parts.append(x)
        src_global_parts.append(src_idx + int(row_offset))
        dst_cat_parts.append(flat_dst)
        row_offset += int(x.size(0))
    x_cat = x_cat_parts[0] if len(x_cat_parts) == 1 else torch.cat(x_cat_parts, dim=0)
    src_global = (
        src_global_parts[0]
        if len(src_global_parts) == 1
        else torch.cat(src_global_parts, dim=0)
    )
    flat_dst = dst_cat_parts[0] if len(dst_cat_parts) == 1 else torch.cat(dst_cat_parts, dim=0)
    return x_cat, src_global, flat_dst


def _fanin_pack_multi_python(
    rel_parts: list[torch.Tensor],
    flat_src_parts: list[torch.Tensor],
    dst_idx_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rel_parts:
        raise ValueError("fanin_pack_multi requires at least one relation tensor.")
    if not (len(rel_parts) == len(flat_src_parts) == len(dst_idx_parts)):
        raise ValueError(
            "fanin_pack_multi expects rel_parts, flat_src_parts, and dst_idx_parts with equal lengths."
        )
    row_offset = 0
    rel_cat_parts: list[torch.Tensor] = []
    src_cat_parts: list[torch.Tensor] = []
    dst_cat_parts: list[torch.Tensor] = []
    for rel, flat_src, dst_idx in zip(rel_parts, flat_src_parts, dst_idx_parts):
        if rel.dim() != 2:
            raise ValueError("fanin_pack_multi expects each relation tensor to be rank-2.")
        if flat_src.dim() != 1 or dst_idx.dim() != 1:
            raise ValueError("fanin_pack_multi expects rank-1 source/destination index tensors.")
        if flat_src.dtype != torch.int64 or dst_idx.dtype != torch.int64:
            raise ValueError("fanin_pack_multi expects int64 source/destination index tensors.")
        if flat_src.numel() != dst_idx.numel():
            raise ValueError(
                "fanin_pack_multi expects flat_src and dst_idx lengths to match per relation tensor."
            )
        rel_cat_parts.append(rel)
        src_cat_parts.append(flat_src + int(row_offset))
        dst_cat_parts.append(dst_idx)
        row_offset += int(rel.size(0))
    rel_cat = rel_cat_parts[0] if len(rel_cat_parts) == 1 else torch.cat(rel_cat_parts, dim=0)
    flat_src = src_cat_parts[0] if len(src_cat_parts) == 1 else torch.cat(src_cat_parts, dim=0)
    dst_idx = dst_cat_parts[0] if len(dst_cat_parts) == 1 else torch.cat(dst_cat_parts, dim=0)
    return rel_cat, flat_src, dst_idx


def _apply_pointwise_code(x: torch.Tensor, code: int) -> torch.Tensor:
    code_i = int(code)
    if code_i == _PW_IDENTITY:
        return x
    if code_i == _PW_RELU:
        return torch.relu(x)
    if code_i == _PW_MISH:
        return torch.nn.functional.mish(x)
    if code_i == _PW_GELU_NONE:
        return torch.nn.functional.gelu(x, approximate="none")
    if code_i == _PW_GELU_TANH:
        return torch.nn.functional.gelu(x, approximate="tanh")
    if code_i == _PW_SILU:
        return torch.nn.functional.silu(x)
    if code_i == _PW_TANH:
        return torch.tanh(x)
    raise ValueError(f"Unsupported pointwise code: {code_i!r}.")


def _fused_two_layer_pointwise_from_indices_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    pointwise_code: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError("fused_two_layer_pointwise_from_indices expects x to be rank-2.")
    if relation_args.dim() != 1:
        raise ValueError(
            "fused_two_layer_pointwise_from_indices expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "fused_two_layer_pointwise_from_indices expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError("fused_two_layer_pointwise_from_indices expects arity > 0.")

    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if w1_stack.dim() != 3 or w2_stack.dim() != 3:
        raise ValueError("fused_two_layer_pointwise_from_indices expects rank-3 weight stacks.")
    if int(w1_stack.size(0)) != groups or int(w2_stack.size(0)) != groups:
        raise ValueError("fused_two_layer_pointwise_from_indices weight stacks must match group count.")
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_two_layer_pointwise_from_indices weight stack dims do not match arity * emb."
        )

    rel_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        rel_idx = relation_args_i64.narrow(0, start, span)
        arg_emb = x.index_select(0, rel_idx)
        x_i = arg_emb.view(n, in_dim)
        hidden = torch.nn.functional.linear(
            x_i,
            w1_stack[i],
            b1_stack[i] if b1_stack.numel() > 0 else None,
        )
        hidden = _apply_pointwise_code(hidden, int(pointwise_code))
        out_i = torch.nn.functional.linear(
            hidden,
            w2_stack[i],
            b2_stack[i] if b2_stack.numel() > 0 else None,
        )
        rel_parts.append((x_i + out_i).view(span, emb))
        node_parts.append(rel_idx)

    if not rel_parts:
        return x.new_empty((0, emb)), torch.empty(0, device=x.device, dtype=torch.int64)
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    node_idx = node_parts[0] if len(node_parts) == 1 else torch.cat(node_parts, dim=0)
    return rel_cat, node_idx


def _fused_program_two_layer_silu_then_two_layer_silu_from_indices_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices expects arity > 0."
        )
    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    stacks = (
        ("w10_stack", w10_stack, 3),
        ("b10_stack", b10_stack, 2),
        ("w20_stack", w20_stack, 3),
        ("b20_stack", b20_stack, 2),
        ("w11_stack", w11_stack, 3),
        ("b11_stack", b11_stack, 2),
        ("w21_stack", w21_stack, 3),
        ("b21_stack", b21_stack, 2),
    )
    for name, tensor, rank in stacks:
        if tensor.dim() != rank:
            raise ValueError(
                f"fused_program_two_layer_silu_then_two_layer_silu_from_indices expects {name} rank-{rank}."
            )
        if int(tensor.size(0)) != groups:
            raise ValueError(
                "fused_program_two_layer_silu_then_two_layer_silu_from_indices expects all parameter stacks to match group count."
            )
    if int(w10_stack.size(2)) != in_dim or int(w20_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices stage-1 dims do not match arity * emb."
        )
    if int(w11_stack.size(2)) != in_dim or int(w21_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices stage-2 dims do not match arity * emb."
        )
    if int(w20_stack.size(2)) != int(w10_stack.size(1)) or int(b10_stack.size(1)) != int(w10_stack.size(1)):
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices stage-1 hidden dims do not match."
        )
    if int(w21_stack.size(2)) != int(w11_stack.size(1)) or int(b11_stack.size(1)) != int(w11_stack.size(1)):
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices stage-2 hidden dims do not match."
        )
    if int(b20_stack.size(1)) != in_dim or int(b21_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices output bias dims do not match arity * emb."
        )

    packed_rows_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for slot, rows in zip(slot_offsets, row_sizes):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        rel_idx = relation_args_i64.narrow(0, start, span)
        arg_emb = x.index_select(0, rel_idx)
        packed_rows_parts.append(arg_emb.view(n, in_dim))
        node_parts.append(rel_idx)

    if not packed_rows_parts:
        return x.new_empty((0, emb)), torch.empty(0, device=x.device, dtype=torch.int64)

    packed_rows = (
        packed_rows_parts[0] if len(packed_rows_parts) == 1 else torch.cat(packed_rows_parts, dim=0)
    )
    row_sizes_tensor = torch.as_tensor(row_sizes, device=x.device, dtype=torch.long)

    row_sizes_long = row_sizes_tensor.to(dtype=torch.long)
    max_rows = int(row_sizes_long.max().item()) if int(row_sizes_long.numel()) > 0 else 0
    row_offsets = torch.empty_like(row_sizes_long)
    if int(row_sizes_long.numel()) > 0:
        row_offsets[0] = 0
    if int(row_sizes_long.numel()) > 1:
        row_offsets[1:] = torch.cumsum(row_sizes_long[:-1], dim=0)
    base = torch.arange(max_rows, device=x.device, dtype=torch.long).unsqueeze(0)
    safe_sizes = row_sizes_long.clamp_min(1).unsqueeze(1)
    safe_idx = row_offsets.unsqueeze(1) + torch.minimum(base, safe_sizes - 1)
    mask = base < row_sizes_long.unsqueeze(1)
    x_rows = packed_rows.index_select(0, safe_idx.reshape(-1)).view(groups, max_rows, in_dim)
    mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
    x_rows = x_rows * mask_f

    pre1 = (torch.bmm(x_rows, w10_stack.transpose(1, 2)) + b10_stack.unsqueeze(1)) * mask_f
    stage1 = (
        torch.bmm(torch.nn.functional.silu(pre1), w20_stack.transpose(1, 2)) + b20_stack.unsqueeze(1)
    ) * mask_f
    pre2 = (torch.bmm(stage1, w11_stack.transpose(1, 2)) + b11_stack.unsqueeze(1)) * mask_f
    stage2 = (
        torch.bmm(torch.nn.functional.silu(pre2), w21_stack.transpose(1, 2)) + b21_stack.unsqueeze(1)
    ) * mask_f
    out_rows = x_rows + stage2

    packed_out = packed_rows.new_zeros((int(packed_rows.size(0)), in_dim))
    packed_out.index_add_(
        0,
        safe_idx.reshape(-1),
        (out_rows * mask_f).reshape(-1, in_dim),
    )
    rel_cat = packed_out.view(-1, emb)
    node_idx = node_parts[0] if len(node_parts) == 1 else torch.cat(node_parts, dim=0)
    return rel_cat, node_idx


def _fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
    ln_weight_stack: torch.Tensor,
    ln_bias_stack: torch.Tensor,
    ln_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices expects arity > 0."
        )
    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    stacks = (
        ("w10_stack", w10_stack, 3),
        ("b10_stack", b10_stack, 2),
        ("w20_stack", w20_stack, 3),
        ("b20_stack", b20_stack, 2),
        ("w11_stack", w11_stack, 3),
        ("b11_stack", b11_stack, 2),
        ("w21_stack", w21_stack, 3),
        ("b21_stack", b21_stack, 2),
    )
    for name, tensor, rank in stacks:
        if tensor.dim() != rank:
            raise ValueError(
                f"fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices expects {name} rank-{rank}."
            )
        if int(tensor.size(0)) != groups:
            raise ValueError(
                "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices expects all parameter stacks to match group count."
            )
    if int(w10_stack.size(2)) != in_dim or int(w20_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices stage-1 dims do not match arity * emb."
        )
    if int(w11_stack.size(2)) != in_dim or int(w21_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices stage-2 dims do not match arity * emb."
        )
    if int(w20_stack.size(2)) != int(w10_stack.size(1)) or int(b10_stack.size(1)) != int(w10_stack.size(1)):
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices stage-1 hidden dims do not match."
        )
    if int(w21_stack.size(2)) != int(w11_stack.size(1)) or int(b11_stack.size(1)) != int(w11_stack.size(1)):
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices stage-2 hidden dims do not match."
        )
    if int(b20_stack.size(1)) != in_dim or int(b21_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices output bias dims do not match arity * emb."
        )
    if ln_weight_stack.numel() > 0:
        if ln_weight_stack.dim() != 2 or int(ln_weight_stack.size(0)) != groups:
            raise ValueError(
                "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices ln_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices ln_weight_stack dims do not match arity * emb."
            )
    if ln_bias_stack.numel() > 0:
        if ln_bias_stack.dim() != 2 or int(ln_bias_stack.size(0)) != groups:
            raise ValueError(
                "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices ln_bias_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_bias_stack.size(1)) != in_dim:
            raise ValueError(
                "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices ln_bias_stack dims do not match arity * emb."
            )

    rel_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        rel_idx = relation_args_i64.narrow(0, start, span)
        arg_emb = x.index_select(0, rel_idx)
        x_i = arg_emb.view(n, in_dim)
        stage1 = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(x_i, w10_stack[i], b10_stack[i])),
            w20_stack[i],
            b20_stack[i],
        )
        stage2 = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(stage1, w11_stack[i], b11_stack[i])),
            w21_stack[i],
            b21_stack[i],
        )
        stage2 = torch.nn.functional.layer_norm(
            stage2,
            (in_dim,),
            weight=(ln_weight_stack[i] if ln_weight_stack.numel() > 0 else None),
            bias=(ln_bias_stack[i] if ln_bias_stack.numel() > 0 else None),
            eps=float(ln_eps),
        )
        rel_parts.append((x_i + stage2).view(span, emb))
        node_parts.append(rel_idx)

    if not rel_parts:
        return x.new_empty((0, emb)), torch.empty(0, device=x.device, dtype=torch.int64)
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    node_idx = node_parts[0] if len(node_parts) == 1 else torch.cat(node_parts, dim=0)
    return rel_cat, node_idx


def _fused_two_layer_mish_from_indices_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _fused_two_layer_pointwise_from_indices_python(
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        _PW_MISH,
    )


def _fused_postnorm_two_layer_pointwise_layernorm_from_indices_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    ln_weight_stack: torch.Tensor,
    ln_bias_stack: torch.Tensor,
    ln_eps: float,
    pointwise_code: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices expects arity > 0."
        )

    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if w1_stack.dim() != 3 or w2_stack.dim() != 3:
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices expects rank-3 weight stacks."
        )
    if int(w1_stack.size(0)) != groups or int(w2_stack.size(0)) != groups:
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices weight stacks must match group count."
        )
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices weight stack dims do not match arity * emb."
        )
    if ln_weight_stack.numel() > 0:
        if ln_weight_stack.dim() != 2 or int(ln_weight_stack.size(0)) != groups:
            raise ValueError(
                "fused_postnorm_two_layer_pointwise_layernorm_from_indices ln_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "fused_postnorm_two_layer_pointwise_layernorm_from_indices ln_weight_stack dims do not match arity * emb."
            )
    if ln_bias_stack.numel() > 0:
        if ln_bias_stack.dim() != 2 or int(ln_bias_stack.size(0)) != groups:
            raise ValueError(
                "fused_postnorm_two_layer_pointwise_layernorm_from_indices ln_bias_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_bias_stack.size(1)) != in_dim:
            raise ValueError(
                "fused_postnorm_two_layer_pointwise_layernorm_from_indices ln_bias_stack dims do not match arity * emb."
            )

    rel_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        rel_idx = relation_args_i64.narrow(0, start, span)
        arg_emb = x.index_select(0, rel_idx)
        x_i = arg_emb.view(n, in_dim)
        hidden = torch.nn.functional.linear(
            x_i,
            w1_stack[i],
            b1_stack[i] if b1_stack.numel() > 0 else None,
        )
        hidden = _apply_pointwise_code(hidden, int(pointwise_code))
        out_i = torch.nn.functional.linear(
            hidden,
            w2_stack[i],
            b2_stack[i] if b2_stack.numel() > 0 else None,
        )
        out_i = torch.nn.functional.layer_norm(
            out_i,
            (in_dim,),
            weight=(ln_weight_stack[i] if ln_weight_stack.numel() > 0 else None),
            bias=(ln_bias_stack[i] if ln_bias_stack.numel() > 0 else None),
            eps=float(ln_eps),
        )
        rel_parts.append((x_i + out_i).view(span, emb))
        node_parts.append(rel_idx)

    if not rel_parts:
        return x.new_empty((0, emb)), torch.empty(0, device=x.device, dtype=torch.int64)
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    node_idx = node_parts[0] if len(node_parts) == 1 else torch.cat(node_parts, dim=0)
    return rel_cat, node_idx


def _fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    rms_weight_stack: torch.Tensor,
    rms_eps: float,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    pointwise_code: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 2:
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices expects arity > 0."
        )

    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if rms_weight_stack.numel() > 0:
        if rms_weight_stack.dim() != 2 or int(rms_weight_stack.size(0)) != groups:
            raise ValueError(
                "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices rms_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(rms_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices rms_weight_stack dims do not match arity * emb."
            )
    if w1_stack.dim() != 3 or w2_stack.dim() != 3:
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices expects rank-3 weight stacks."
        )
    if int(w1_stack.size(0)) != groups or int(w2_stack.size(0)) != groups:
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices weight stacks must match group count."
        )
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices weight stack dims do not match arity * emb."
        )

    rel_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        rel_idx = relation_args_i64.narrow(0, start, span)
        arg_emb = x.index_select(0, rel_idx)
        x_i = arg_emb.view(n, in_dim)
        norm_i = torch.nn.functional.rms_norm(
            x_i,
            (in_dim,),
            weight=(rms_weight_stack[i] if rms_weight_stack.numel() > 0 else None),
            eps=float(rms_eps),
        )
        hidden = torch.nn.functional.linear(
            norm_i,
            w1_stack[i],
            b1_stack[i] if b1_stack.numel() > 0 else None,
        )
        hidden = _apply_pointwise_code(hidden, int(pointwise_code))
        out_i = torch.nn.functional.linear(
            hidden,
            w2_stack[i],
            b2_stack[i] if b2_stack.numel() > 0 else None,
        )
        rel_parts.append((x_i + out_i).view(span, emb))
        node_parts.append(rel_idx)

    if not rel_parts:
        return x.new_empty((0, emb)), torch.empty(0, device=x.device, dtype=torch.int64)
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    node_idx = node_parts[0] if len(node_parts) == 1 else torch.cat(node_parts, dim=0)
    return rel_cat, node_idx


def _fanout_pack_from_edges_python(
    x_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    src_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    slot_offset_parts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not x_parts:
        raise ValueError("fanout_pack_from_edges requires at least one source tensor.")
    n = len(edge_src_parts)
    if not (
        len(edge_dst_parts)
        == len(src_part_ids)
        == len(arity_parts)
        == len(pos_parts)
        == len(slot_offset_parts)
        == n
    ):
        raise ValueError(
            "fanout_pack_from_edges expects edge parts and metadata with equal lengths."
        )
    ref = x_parts[0]
    x_offsets: list[int] = []
    row_offset = 0
    for x in x_parts:
        if x.dim() != 2:
            raise ValueError("fanout_pack_from_edges expects each source tensor to be rank-2.")
        x_offsets.append(int(row_offset))
        row_offset += int(x.size(0))
    x_cat = x_parts[0] if len(x_parts) == 1 else torch.cat(x_parts, dim=0)

    src_global_parts: list[torch.Tensor] = []
    flat_dst_parts: list[torch.Tensor] = []
    for edge_src, edge_dst, src_part, arity, pos, slot_offset in zip(
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    ):
        if edge_src.dim() != 1 or edge_dst.dim() != 1:
            raise ValueError("fanout_pack_from_edges expects rank-1 edge src/dst tensors.")
        if edge_src.dtype != torch.int64 or edge_dst.dtype != torch.int64:
            raise ValueError("fanout_pack_from_edges expects int64 edge src/dst tensors.")
        if edge_src.numel() != edge_dst.numel():
            raise ValueError("fanout_pack_from_edges expects edge src/dst lengths to match.")
        if int(src_part) < 0 or int(src_part) >= len(x_parts):
            raise ValueError(f"fanout_pack_from_edges src_part_ids out of range: {src_part!r}.")
        arity_i = int(arity)
        pos_i = int(pos)
        if arity_i <= 0:
            raise ValueError("fanout_pack_from_edges expects arity > 0.")
        if pos_i < 0 or pos_i >= arity_i:
            raise ValueError(
                f"fanout_pack_from_edges expects pos in [0, arity), got pos={pos_i} arity={arity_i}."
            )
        src_global_parts.append(edge_src + int(x_offsets[int(src_part)]))
        flat_dst_parts.append(int(slot_offset) + edge_dst * arity_i + pos_i)

    if src_global_parts:
        src_global = (
            src_global_parts[0]
            if len(src_global_parts) == 1
            else torch.cat(src_global_parts, dim=0)
        )
        flat_dst = (
            flat_dst_parts[0] if len(flat_dst_parts) == 1 else torch.cat(flat_dst_parts, dim=0)
        )
    else:
        src_global = torch.empty(0, device=ref.device, dtype=torch.int64)
        flat_dst = torch.empty(0, device=ref.device, dtype=torch.int64)
    return x_cat, src_global, flat_dst


def _fanin_pack_from_edges_python(
    rel_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    rel_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not rel_parts:
        raise ValueError("fanin_pack_from_edges requires at least one relation tensor.")
    mode_i = int(mode)
    if mode_i not in (0, 1):
        raise ValueError(f"fanin_pack_from_edges expects mode in {{0,1}}, got {mode_i!r}.")
    n = len(edge_src_parts)
    if not (
        len(edge_dst_parts)
        == len(rel_part_ids)
        == len(arity_parts)
        == len(pos_parts)
        == n
    ):
        raise ValueError(
            "fanin_pack_from_edges expects edge parts and metadata with equal lengths."
        )
    ref = rel_parts[0]
    rel_offsets: list[int] = []
    row_offset = 0
    for rel in rel_parts:
        if rel.dim() != 2:
            raise ValueError("fanin_pack_from_edges expects each relation tensor to be rank-2.")
        rel_offsets.append(int(row_offset))
        row_offset += int(rel.size(0))
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)

    flat_src_parts: list[torch.Tensor] = []
    dst_cat_parts: list[torch.Tensor] = []
    for edge_src, edge_dst, rel_part, arity, pos in zip(
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
    ):
        if edge_src.dim() != 1 or edge_dst.dim() != 1:
            raise ValueError("fanin_pack_from_edges expects rank-1 edge src/dst tensors.")
        if edge_src.dtype != torch.int64 or edge_dst.dtype != torch.int64:
            raise ValueError("fanin_pack_from_edges expects int64 edge src/dst tensors.")
        if edge_src.numel() != edge_dst.numel():
            raise ValueError("fanin_pack_from_edges expects edge src/dst lengths to match.")
        rel_part_i = int(rel_part)
        if rel_part_i < 0 or rel_part_i >= len(rel_parts):
            raise ValueError(
                f"fanin_pack_from_edges rel_part_ids out of range: {rel_part_i!r}."
            )
        if mode_i == 1:
            flat_src_local = edge_src
        else:
            arity_i = int(arity)
            pos_i = int(pos)
            if arity_i <= 0:
                raise ValueError("fanin_pack_from_edges expects arity > 0 in relation mode.")
            if pos_i < 0 or pos_i >= arity_i:
                raise ValueError(
                    f"fanin_pack_from_edges expects pos in [0, arity), got pos={pos_i} arity={arity_i}."
                )
            flat_src_local = edge_src * arity_i + pos_i
        flat_src_parts.append(flat_src_local + int(rel_offsets[rel_part_i]))
        dst_cat_parts.append(edge_dst)

    if flat_src_parts:
        flat_src = (
            flat_src_parts[0] if len(flat_src_parts) == 1 else torch.cat(flat_src_parts, dim=0)
        )
        dst_idx = (
            dst_cat_parts[0] if len(dst_cat_parts) == 1 else torch.cat(dst_cat_parts, dim=0)
        )
    else:
        flat_src = torch.empty(0, device=ref.device, dtype=torch.int64)
        dst_idx = torch.empty(0, device=ref.device, dtype=torch.int64)
    return rel_cat, flat_src, dst_idx


if torch is not None:

    _CUSTOM_TWO_LAYER_POINTWISE_CODES = {
        _PW_MISH,
        _PW_GELU_NONE,
        _PW_GELU_TANH,
        _PW_SILU,
    }

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

    class _FusedTwoLayerPointwiseFromIndicesFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            relation_args: torch.Tensor,
            slot_offsets: list[int],
            row_sizes: list[int],
            arity: int,
            w1_stack: torch.Tensor,
            b1_stack: torch.Tensor,
            w2_stack: torch.Tensor,
            b2_stack: torch.Tensor,
            pointwise_code: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx.slot_offsets = [int(v) for v in slot_offsets]
            ctx.row_sizes = [int(v) for v in row_sizes]
            ctx.arity = int(arity)
            ctx.pointwise_code = int(pointwise_code)
            ctx.save_for_backward(x, relation_args, w1_stack, b1_stack, w2_stack, b2_stack)
            used_custom = (
                x.is_cuda
                and ctx.pointwise_code in _CUSTOM_TWO_LAYER_POINTWISE_CODES
                and _should_use_custom("fused_two_layer_pointwise_from_indices")
                and _namespace_has_op("fused_two_layer_pointwise_from_indices")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().fused_two_layer_pointwise_from_indices(
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w1_stack,
                    b1_stack,
                    w2_stack,
                    b2_stack,
                    int(ctx.pointwise_code),
                )
            return _fused_two_layer_pointwise_from_indices_python(
                x,
                relation_args,
                list(ctx.slot_offsets),
                list(ctx.row_sizes),
                int(ctx.arity),
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                int(ctx.pointwise_code),
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_rel: torch.Tensor,
            grad_node_idx: torch.Tensor | None,
        ) -> tuple[
            torch.Tensor | None,
            None,
            None,
            None,
            None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            None,
        ]:
            del grad_node_idx
            if grad_rel is None:
                return (None, None, None, None, None, None, None, None, None, None)

            x, relation_args, w1_stack, b1_stack, w2_stack, b2_stack = ctx.saved_tensors
            needs = ctx.needs_input_grad
            grad_map: list[torch.Tensor | None] = [None, None, None, None, None, None, None, None, None, None]
            if (
                bool(getattr(ctx, "used_custom", False))
                and grad_rel.is_cuda
                and _should_use_custom("fused_two_layer_pointwise_from_indices_backward")
                and _namespace_has_op("fused_two_layer_pointwise_from_indices_backward")
            ):
                grad_x, grad_w1, grad_b1, grad_w2, grad_b2 = (
                    _ops_namespace().fused_two_layer_pointwise_from_indices_backward(
                        grad_rel,
                        x,
                        relation_args,
                        list(ctx.slot_offsets),
                        list(ctx.row_sizes),
                        int(ctx.arity),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(ctx.pointwise_code),
                    )
                )
                if needs[0]:
                    grad_map[0] = grad_x
                if needs[5]:
                    grad_map[5] = grad_w1
                if needs[6] and b1_stack.numel() > 0:
                    grad_map[6] = grad_b1
                if needs[7]:
                    grad_map[7] = grad_w2
                if needs[8] and b2_stack.numel() > 0:
                    grad_map[8] = grad_b2
                return tuple(grad_map)  # type: ignore[return-value]
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(bool(needs[0]))
                w1_req = w1_stack.detach().requires_grad_(bool(needs[5]))
                b1_req = (
                    b1_stack.detach().requires_grad_(bool(needs[6] and b1_stack.numel() > 0))
                    if b1_stack.numel() > 0
                    else b1_stack
                )
                w2_req = w2_stack.detach().requires_grad_(bool(needs[7]))
                b2_req = (
                    b2_stack.detach().requires_grad_(bool(needs[8] and b2_stack.numel() > 0))
                    if b2_stack.numel() > 0
                    else b2_stack
                )
                rel_cat, _ = _fused_two_layer_pointwise_from_indices_python(
                    x_req,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w1_req,
                    b1_req,
                    w2_req,
                    b2_req,
                    int(ctx.pointwise_code),
                )
                grad_inputs: list[torch.Tensor] = []
                grad_targets: list[int] = []
                for pos, tensor in (
                    (0, x_req),
                    (5, w1_req),
                    (6, b1_req if b1_stack.numel() > 0 else None),
                    (7, w2_req),
                    (8, b2_req if b2_stack.numel() > 0 else None),
                ):
                    if tensor is not None and tensor.requires_grad:
                        grad_inputs.append(tensor)
                        grad_targets.append(pos)
                grads = (
                    torch.autograd.grad(rel_cat, grad_inputs, grad_rel, allow_unused=True)
                    if grad_inputs
                    else ()
                )
            for pos, grad in zip(grad_targets, grads):
                grad_map[pos] = grad
            return tuple(grad_map)  # type: ignore[return-value]

    class _FusedProgramTwoLayerSiLUThenTwoLayerSiLUFromIndicesFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            relation_args: torch.Tensor,
            slot_offsets: list[int],
            row_sizes: list[int],
            arity: int,
            w10_stack: torch.Tensor,
            b10_stack: torch.Tensor,
            w20_stack: torch.Tensor,
            b20_stack: torch.Tensor,
            w11_stack: torch.Tensor,
            b11_stack: torch.Tensor,
            w21_stack: torch.Tensor,
            b21_stack: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx.slot_offsets = [int(v) for v in slot_offsets]
            ctx.row_sizes = [int(v) for v in row_sizes]
            ctx.arity = int(arity)
            ctx.save_for_backward(
                x,
                relation_args,
                w10_stack,
                b10_stack,
                w20_stack,
                b20_stack,
                w11_stack,
                b11_stack,
                w21_stack,
                b21_stack,
            )
            used_custom = (
                x.is_cuda
                and _should_use_custom("fused_program_two_layer_silu_then_two_layer_silu_from_indices")
                and _namespace_has_op("fused_program_two_layer_silu_then_two_layer_silu_from_indices")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().fused_program_two_layer_silu_then_two_layer_silu_from_indices(
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                )
            return _fused_program_two_layer_silu_then_two_layer_silu_from_indices_python(
                x,
                relation_args,
                list(ctx.slot_offsets),
                list(ctx.row_sizes),
                int(ctx.arity),
                w10_stack,
                b10_stack,
                w20_stack,
                b20_stack,
                w11_stack,
                b11_stack,
                w21_stack,
                b21_stack,
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_rel: torch.Tensor,
            grad_node_idx: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            del grad_node_idx
            if grad_rel is None:
                return (None, None, None, None, None, None, None, None, None, None, None, None, None, None)
            (
                x,
                relation_args,
                w10_stack,
                b10_stack,
                w20_stack,
                b20_stack,
                w11_stack,
                b11_stack,
                w21_stack,
                b21_stack,
            ) = ctx.saved_tensors
            needs = ctx.needs_input_grad
            grad_map: list[torch.Tensor | None] = [None] * 14
            if (
                bool(getattr(ctx, "used_custom", False))
                and grad_rel.is_cuda
                and _should_use_custom(
                    "fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward"
                )
                and _namespace_has_op(
                    "fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward"
                )
            ):
                (
                    grad_x,
                    grad_w10,
                    grad_b10,
                    grad_w20,
                    grad_b20,
                    grad_w11,
                    grad_b11,
                    grad_w21,
                    grad_b21,
                ) = _ops_namespace().fused_program_two_layer_silu_then_two_layer_silu_from_indices_backward(
                    grad_rel,
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                )
                if needs[0]:
                    grad_map[0] = grad_x
                if needs[5]:
                    grad_map[5] = grad_w10
                if needs[6]:
                    grad_map[6] = grad_b10
                if needs[7]:
                    grad_map[7] = grad_w20
                if needs[8]:
                    grad_map[8] = grad_b20
                if needs[9]:
                    grad_map[9] = grad_w11
                if needs[10]:
                    grad_map[10] = grad_b11
                if needs[11]:
                    grad_map[11] = grad_w21
                if needs[12]:
                    grad_map[12] = grad_b21
                return tuple(grad_map)
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(bool(needs[0]))
                w10_req = w10_stack.detach().requires_grad_(bool(needs[5]))
                b10_req = b10_stack.detach().requires_grad_(bool(needs[6]))
                w20_req = w20_stack.detach().requires_grad_(bool(needs[7]))
                b20_req = b20_stack.detach().requires_grad_(bool(needs[8]))
                w11_req = w11_stack.detach().requires_grad_(bool(needs[9]))
                b11_req = b11_stack.detach().requires_grad_(bool(needs[10]))
                w21_req = w21_stack.detach().requires_grad_(bool(needs[11]))
                b21_req = b21_stack.detach().requires_grad_(bool(needs[12]))
                rel_cat, _ = _fused_program_two_layer_silu_then_two_layer_silu_from_indices_python(
                    x_req,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w10_req,
                    b10_req,
                    w20_req,
                    b20_req,
                    w11_req,
                    b11_req,
                    w21_req,
                    b21_req,
                )
                grad_inputs: list[torch.Tensor] = []
                grad_targets: list[int] = []
                for idx, tensor in (
                    (0, x_req),
                    (5, w10_req),
                    (6, b10_req),
                    (7, w20_req),
                    (8, b20_req),
                    (9, w11_req),
                    (10, b11_req),
                    (11, w21_req),
                    (12, b21_req),
                ):
                    if needs[idx]:
                        grad_inputs.append(tensor)
                        grad_targets.append(idx)
                grads = torch.autograd.grad(rel_cat, grad_inputs, grad_rel, allow_unused=True)
            for idx, grad in zip(grad_targets, grads):
                grad_map[idx] = grad
            return tuple(grad_map)

    class _FusedProgramTwoLayerSiLUThenPostNormTwoLayerSiLUFromIndicesFunction(
        torch.autograd.Function
    ):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            relation_args: torch.Tensor,
            slot_offsets: list[int],
            row_sizes: list[int],
            arity: int,
            w10_stack: torch.Tensor,
            b10_stack: torch.Tensor,
            w20_stack: torch.Tensor,
            b20_stack: torch.Tensor,
            w11_stack: torch.Tensor,
            b11_stack: torch.Tensor,
            w21_stack: torch.Tensor,
            b21_stack: torch.Tensor,
            ln_weight_stack: torch.Tensor,
            ln_bias_stack: torch.Tensor,
            ln_eps: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx.slot_offsets = [int(v) for v in slot_offsets]
            ctx.row_sizes = [int(v) for v in row_sizes]
            ctx.arity = int(arity)
            ctx.ln_eps = float(ln_eps)
            ctx.save_for_backward(
                x,
                relation_args,
                w10_stack,
                b10_stack,
                w20_stack,
                b20_stack,
                w11_stack,
                b11_stack,
                w21_stack,
                b21_stack,
                ln_weight_stack,
                ln_bias_stack,
            )
            used_custom = (
                x.is_cuda
                and _should_use_custom("fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices")
                and _namespace_has_op(
                    "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices"
                )
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices(
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                    ln_weight_stack,
                    ln_bias_stack,
                    float(ctx.ln_eps),
                )
            return _fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_python(
                x,
                relation_args,
                list(ctx.slot_offsets),
                list(ctx.row_sizes),
                int(ctx.arity),
                w10_stack,
                b10_stack,
                w20_stack,
                b20_stack,
                w11_stack,
                b11_stack,
                w21_stack,
                b21_stack,
                ln_weight_stack,
                ln_bias_stack,
                float(ctx.ln_eps),
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_rel: torch.Tensor,
            grad_node_idx: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            del grad_node_idx
            if grad_rel is None:
                return (None,) * 16
            (
                x,
                relation_args,
                w10_stack,
                b10_stack,
                w20_stack,
                b20_stack,
                w11_stack,
                b11_stack,
                w21_stack,
                b21_stack,
                ln_weight_stack,
                ln_bias_stack,
            ) = ctx.saved_tensors
            needs = ctx.needs_input_grad
            grad_map: list[torch.Tensor | None] = [None] * 16
            if (
                bool(getattr(ctx, "used_custom", False))
                and grad_rel.is_cuda
                and _should_use_custom(
                    "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward"
                )
                and _namespace_has_op(
                    "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward"
                )
            ):
                (
                    grad_x,
                    grad_w10,
                    grad_b10,
                    grad_w20,
                    grad_b20,
                    grad_w11,
                    grad_b11,
                    grad_w21,
                    grad_b21,
                    grad_ln_weight,
                    grad_ln_bias,
                ) = _ops_namespace().fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_backward(
                    grad_rel,
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                    ln_weight_stack,
                    ln_bias_stack,
                    float(ctx.ln_eps),
                )
                if needs[0]:
                    grad_map[0] = grad_x
                if needs[5]:
                    grad_map[5] = grad_w10
                if needs[6]:
                    grad_map[6] = grad_b10
                if needs[7]:
                    grad_map[7] = grad_w20
                if needs[8]:
                    grad_map[8] = grad_b20
                if needs[9]:
                    grad_map[9] = grad_w11
                if needs[10]:
                    grad_map[10] = grad_b11
                if needs[11]:
                    grad_map[11] = grad_w21
                if needs[12]:
                    grad_map[12] = grad_b21
                if needs[13] and ln_weight_stack.numel() > 0:
                    grad_map[13] = grad_ln_weight
                if needs[14] and ln_bias_stack.numel() > 0:
                    grad_map[14] = grad_ln_bias
                return tuple(grad_map)
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(bool(needs[0]))
                w10_req = w10_stack.detach().requires_grad_(bool(needs[5]))
                b10_req = b10_stack.detach().requires_grad_(bool(needs[6]))
                w20_req = w20_stack.detach().requires_grad_(bool(needs[7]))
                b20_req = b20_stack.detach().requires_grad_(bool(needs[8]))
                w11_req = w11_stack.detach().requires_grad_(bool(needs[9]))
                b11_req = b11_stack.detach().requires_grad_(bool(needs[10]))
                w21_req = w21_stack.detach().requires_grad_(bool(needs[11]))
                b21_req = b21_stack.detach().requires_grad_(bool(needs[12]))
                ln_w_req = (
                    ln_weight_stack.detach().requires_grad_(bool(needs[13] and ln_weight_stack.numel() > 0))
                    if ln_weight_stack.numel() > 0
                    else ln_weight_stack
                )
                ln_b_req = (
                    ln_bias_stack.detach().requires_grad_(bool(needs[14] and ln_bias_stack.numel() > 0))
                    if ln_bias_stack.numel() > 0
                    else ln_bias_stack
                )
                rel_cat, _ = _fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices_python(
                    x_req,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w10_req,
                    b10_req,
                    w20_req,
                    b20_req,
                    w11_req,
                    b11_req,
                    w21_req,
                    b21_req,
                    ln_w_req,
                    ln_b_req,
                    float(ctx.ln_eps),
                )
                grad_inputs: list[torch.Tensor] = []
                grad_targets: list[int] = []
                for idx, tensor in (
                    (0, x_req),
                    (5, w10_req),
                    (6, b10_req),
                    (7, w20_req),
                    (8, b20_req),
                    (9, w11_req),
                    (10, b11_req),
                    (11, w21_req),
                    (12, b21_req),
                    (13, ln_w_req if ln_weight_stack.numel() > 0 else None),
                    (14, ln_b_req if ln_bias_stack.numel() > 0 else None),
                ):
                    if tensor is not None and needs[idx]:
                        grad_inputs.append(tensor)
                        grad_targets.append(idx)
                grads = torch.autograd.grad(rel_cat, grad_inputs, grad_rel, allow_unused=True)
            for idx, grad in zip(grad_targets, grads):
                grad_map[idx] = grad
            return tuple(grad_map)


    class _FusedTwoLayerMishFromIndicesFunction(_FusedTwoLayerPointwiseFromIndicesFunction):
        pass

    class _FusedPostNormTwoLayerPointwiseLayerNormFromIndicesFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            relation_args: torch.Tensor,
            slot_offsets: list[int],
            row_sizes: list[int],
            arity: int,
            w1_stack: torch.Tensor,
            b1_stack: torch.Tensor,
            w2_stack: torch.Tensor,
            b2_stack: torch.Tensor,
            ln_weight_stack: torch.Tensor,
            ln_bias_stack: torch.Tensor,
            ln_eps: float,
            pointwise_code: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx.slot_offsets = [int(v) for v in slot_offsets]
            ctx.row_sizes = [int(v) for v in row_sizes]
            ctx.arity = int(arity)
            ctx.ln_eps = float(ln_eps)
            ctx.pointwise_code = int(pointwise_code)
            ctx.save_for_backward(
                x,
                relation_args,
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                ln_weight_stack,
                ln_bias_stack,
            )
            used_custom = (
                x.is_cuda
                and ctx.pointwise_code in _CUSTOM_TWO_LAYER_POINTWISE_CODES
                and _should_use_custom("fused_postnorm_two_layer_pointwise_layernorm_from_indices")
                and _namespace_has_op("fused_postnorm_two_layer_pointwise_layernorm_from_indices")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().fused_postnorm_two_layer_pointwise_layernorm_from_indices(
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w1_stack,
                    b1_stack,
                    w2_stack,
                    b2_stack,
                    ln_weight_stack,
                    ln_bias_stack,
                    float(ctx.ln_eps),
                    int(ctx.pointwise_code),
                )
            return _fused_postnorm_two_layer_pointwise_layernorm_from_indices_python(
                x,
                relation_args,
                list(ctx.slot_offsets),
                list(ctx.row_sizes),
                int(ctx.arity),
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                ln_weight_stack,
                ln_bias_stack,
                float(ctx.ln_eps),
                int(ctx.pointwise_code),
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_rel: torch.Tensor,
            grad_node_idx: torch.Tensor | None,
        ) -> tuple[
            torch.Tensor | None,
            None,
            None,
            None,
            None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            None,
            None,
        ]:
            del grad_node_idx
            if grad_rel is None:
                return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            x, relation_args, w1_stack, b1_stack, w2_stack, b2_stack, ln_weight_stack, ln_bias_stack = ctx.saved_tensors
            needs = ctx.needs_input_grad
            grad_map: list[torch.Tensor | None] = [None] * 13
            if (
                bool(getattr(ctx, "used_custom", False))
                and grad_rel.is_cuda
                and _should_use_custom(
                    "fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward"
                )
                and _namespace_has_op(
                    "fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward"
                )
            ):
                grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_ln_weight, grad_ln_bias = (
                    _ops_namespace().fused_postnorm_two_layer_pointwise_layernorm_from_indices_backward(
                        grad_rel,
                        x,
                        relation_args,
                        list(ctx.slot_offsets),
                        list(ctx.row_sizes),
                        int(ctx.arity),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        ln_weight_stack,
                        ln_bias_stack,
                        float(ctx.ln_eps),
                        int(ctx.pointwise_code),
                    )
                )
                if needs[0]:
                    grad_map[0] = grad_x
                if needs[5]:
                    grad_map[5] = grad_w1
                if needs[6] and b1_stack.numel() > 0:
                    grad_map[6] = grad_b1
                if needs[7]:
                    grad_map[7] = grad_w2
                if needs[8] and b2_stack.numel() > 0:
                    grad_map[8] = grad_b2
                if needs[9] and ln_weight_stack.numel() > 0:
                    grad_map[9] = grad_ln_weight
                if needs[10] and ln_bias_stack.numel() > 0:
                    grad_map[10] = grad_ln_bias
                return tuple(grad_map)  # type: ignore[return-value]

            with torch.enable_grad():
                x_req = x.detach().requires_grad_(bool(needs[0]))
                w1_req = w1_stack.detach().requires_grad_(bool(needs[5]))
                b1_req = (
                    b1_stack.detach().requires_grad_(bool(needs[6] and b1_stack.numel() > 0))
                    if b1_stack.numel() > 0
                    else b1_stack
                )
                w2_req = w2_stack.detach().requires_grad_(bool(needs[7]))
                b2_req = (
                    b2_stack.detach().requires_grad_(bool(needs[8] and b2_stack.numel() > 0))
                    if b2_stack.numel() > 0
                    else b2_stack
                )
                ln_w_req = (
                    ln_weight_stack.detach().requires_grad_(bool(needs[9] and ln_weight_stack.numel() > 0))
                    if ln_weight_stack.numel() > 0
                    else ln_weight_stack
                )
                ln_b_req = (
                    ln_bias_stack.detach().requires_grad_(bool(needs[10] and ln_bias_stack.numel() > 0))
                    if ln_bias_stack.numel() > 0
                    else ln_bias_stack
                )
                rel_cat, _ = _fused_postnorm_two_layer_pointwise_layernorm_from_indices_python(
                    x_req,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    w1_req,
                    b1_req,
                    w2_req,
                    b2_req,
                    ln_w_req,
                    ln_b_req,
                    float(ctx.ln_eps),
                    int(ctx.pointwise_code),
                )
                grad_inputs: list[torch.Tensor] = []
                grad_targets: list[int] = []
                for pos, tensor in (
                    (0, x_req),
                    (5, w1_req),
                    (6, b1_req if b1_stack.numel() > 0 else None),
                    (7, w2_req),
                    (8, b2_req if b2_stack.numel() > 0 else None),
                    (9, ln_w_req if ln_weight_stack.numel() > 0 else None),
                    (10, ln_b_req if ln_bias_stack.numel() > 0 else None),
                ):
                    if tensor is not None and tensor.requires_grad:
                        grad_inputs.append(tensor)
                        grad_targets.append(pos)
                grads = (
                    torch.autograd.grad(rel_cat, grad_inputs, grad_rel, allow_unused=True)
                    if grad_inputs
                    else ()
                )
            for pos, grad in zip(grad_targets, grads):
                grad_map[pos] = grad
            return tuple(grad_map)  # type: ignore[return-value]

    class _FusedPreNormTwoLayerPointwiseRMSNormFromIndicesFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            relation_args: torch.Tensor,
            slot_offsets: list[int],
            row_sizes: list[int],
            arity: int,
            rms_weight_stack: torch.Tensor,
            rms_eps: float,
            w1_stack: torch.Tensor,
            b1_stack: torch.Tensor,
            w2_stack: torch.Tensor,
            b2_stack: torch.Tensor,
            pointwise_code: int,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            ctx.slot_offsets = [int(v) for v in slot_offsets]
            ctx.row_sizes = [int(v) for v in row_sizes]
            ctx.arity = int(arity)
            ctx.rms_eps = float(rms_eps)
            ctx.pointwise_code = int(pointwise_code)
            ctx.save_for_backward(
                x,
                relation_args,
                rms_weight_stack,
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
            )
            used_custom = (
                x.is_cuda
                and ctx.pointwise_code in _CUSTOM_TWO_LAYER_POINTWISE_CODES
                and _should_use_custom("fused_prenorm_two_layer_pointwise_rmsnorm_from_indices")
                and _namespace_has_op("fused_prenorm_two_layer_pointwise_rmsnorm_from_indices")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().fused_prenorm_two_layer_pointwise_rmsnorm_from_indices(
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    rms_weight_stack,
                    float(ctx.rms_eps),
                    w1_stack,
                    b1_stack,
                    w2_stack,
                    b2_stack,
                    int(ctx.pointwise_code),
                )
            return _fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_python(
                x,
                relation_args,
                list(ctx.slot_offsets),
                list(ctx.row_sizes),
                int(ctx.arity),
                rms_weight_stack,
                float(ctx.rms_eps),
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                int(ctx.pointwise_code),
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_rel: torch.Tensor,
            grad_node_idx: torch.Tensor | None,
        ) -> tuple[
            torch.Tensor | None,
            None,
            None,
            None,
            None,
            torch.Tensor | None,
            None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            torch.Tensor | None,
            None,
        ]:
            del grad_node_idx
            if grad_rel is None:
                return (None, None, None, None, None, None, None, None, None, None, None, None)

            x, relation_args, rms_weight_stack, w1_stack, b1_stack, w2_stack, b2_stack = ctx.saved_tensors
            needs = ctx.needs_input_grad
            grad_map: list[torch.Tensor | None] = [None] * 12
            if (
                bool(getattr(ctx, "used_custom", False))
                and grad_rel.is_cuda
                and _should_use_custom("fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward")
                and _namespace_has_op("fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward")
            ):
                grad_x, grad_rms_weight, grad_w1, grad_b1, grad_w2, grad_b2 = (
                    _ops_namespace().fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_backward(
                        grad_rel,
                        x,
                        relation_args,
                        list(ctx.slot_offsets),
                        list(ctx.row_sizes),
                        int(ctx.arity),
                        rms_weight_stack,
                        float(ctx.rms_eps),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(ctx.pointwise_code),
                    )
                )
                if needs[0]:
                    grad_map[0] = grad_x
                if needs[5] and rms_weight_stack.numel() > 0:
                    grad_map[5] = grad_rms_weight
                if needs[7]:
                    grad_map[7] = grad_w1
                if needs[8] and b1_stack.numel() > 0:
                    grad_map[8] = grad_b1
                if needs[9]:
                    grad_map[9] = grad_w2
                if needs[10] and b2_stack.numel() > 0:
                    grad_map[10] = grad_b2
                return tuple(grad_map)  # type: ignore[return-value]

            with torch.enable_grad():
                x_req = x.detach().requires_grad_(bool(needs[0]))
                rms_w_req = (
                    rms_weight_stack.detach().requires_grad_(bool(needs[5] and rms_weight_stack.numel() > 0))
                    if rms_weight_stack.numel() > 0
                    else rms_weight_stack
                )
                w1_req = w1_stack.detach().requires_grad_(bool(needs[7]))
                b1_req = (
                    b1_stack.detach().requires_grad_(bool(needs[8] and b1_stack.numel() > 0))
                    if b1_stack.numel() > 0
                    else b1_stack
                )
                w2_req = w2_stack.detach().requires_grad_(bool(needs[9]))
                b2_req = (
                    b2_stack.detach().requires_grad_(bool(needs[10] and b2_stack.numel() > 0))
                    if b2_stack.numel() > 0
                    else b2_stack
                )
                rel_cat, _ = _fused_prenorm_two_layer_pointwise_rmsnorm_from_indices_python(
                    x_req,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    rms_w_req,
                    float(ctx.rms_eps),
                    w1_req,
                    b1_req,
                    w2_req,
                    b2_req,
                    int(ctx.pointwise_code),
                )
                grad_inputs: list[torch.Tensor] = []
                grad_targets: list[int] = []
                for pos, tensor in (
                    (0, x_req),
                    (5, rms_w_req if rms_weight_stack.numel() > 0 else None),
                    (7, w1_req),
                    (8, b1_req if b1_stack.numel() > 0 else None),
                    (9, w2_req),
                    (10, b2_req if b2_stack.numel() > 0 else None),
                ):
                    if tensor is not None and tensor.requires_grad:
                        grad_inputs.append(tensor)
                        grad_targets.append(pos)
                grads = (
                    torch.autograd.grad(rel_cat, grad_inputs, grad_rel, allow_unused=True)
                    if grad_inputs
                    else ()
                )
            for pos, grad in zip(grad_targets, grads):
                grad_map[pos] = grad
            return tuple(grad_map)  # type: ignore[return-value]


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


def fanout_pack_multi(
    x_parts: list[torch.Tensor],
    src_idx_parts: list[torch.Tensor],
    flat_dst_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fanout_pack_multi requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("fanout_pack_multi"):
        if _namespace_has_op("fanout_pack_multi"):
            return _ops_namespace().fanout_pack_multi(
                x_parts,
                src_idx_parts,
                flat_dst_parts,
            )
        if _fallback_mode() == "error":
            raise RuntimeError(
                "Custom mp op fanout_pack_multi is unavailable in the loaded relm_mp library."
            )
    return _fanout_pack_multi_python(x_parts, src_idx_parts, flat_dst_parts)


def fanin_pack_multi(
    rel_parts: list[torch.Tensor],
    flat_src_parts: list[torch.Tensor],
    dst_idx_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fanin_pack_multi requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("fanin_pack_multi"):
        if _namespace_has_op("fanin_pack_multi"):
            return _ops_namespace().fanin_pack_multi(
                rel_parts,
                flat_src_parts,
                dst_idx_parts,
            )
        if _fallback_mode() == "error":
            raise RuntimeError(
                "Custom mp op fanin_pack_multi is unavailable in the loaded relm_mp library."
            )
    return _fanin_pack_multi_python(rel_parts, flat_src_parts, dst_idx_parts)


def fanout_pack_from_edges(
    x_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    src_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    slot_offset_parts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fanout_pack_from_edges requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("fanout_pack_from_edges"):
        if _namespace_has_op("fanout_pack_from_edges"):
            return _ops_namespace().fanout_pack_from_edges(
                x_parts,
                edge_src_parts,
                edge_dst_parts,
                src_part_ids,
                arity_parts,
                pos_parts,
                slot_offset_parts,
            )
        if _fallback_mode() == "error":
            raise RuntimeError(
                "Custom mp op fanout_pack_from_edges is unavailable in the loaded relm_mp library."
            )
    return _fanout_pack_from_edges_python(
        x_parts,
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    )


def fanin_pack_from_edges(
    rel_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    rel_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fanin_pack_from_edges requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("fanin_pack_from_edges"):
        if _namespace_has_op("fanin_pack_from_edges"):
            return _ops_namespace().fanin_pack_from_edges(
                rel_parts,
                edge_src_parts,
                edge_dst_parts,
                rel_part_ids,
                arity_parts,
                pos_parts,
                int(mode),
            )
        if _fallback_mode() == "error":
            raise RuntimeError(
                "Custom mp op fanin_pack_from_edges is unavailable in the loaded relm_mp library."
            )
    return _fanin_pack_from_edges_python(
        rel_parts,
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
        int(mode),
    )


def fused_two_layer_pointwise_from_indices(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    pointwise_code: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fused_two_layer_pointwise_from_indices requires torch."
        ) from _TORCH_IMPORT_ERROR
    return _FusedTwoLayerPointwiseFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def fused_postnorm_two_layer_pointwise_layernorm_from_indices(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    ln_weight_stack: torch.Tensor,
    ln_bias_stack: torch.Tensor,
    ln_eps: float,
    pointwise_code: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fused_postnorm_two_layer_pointwise_layernorm_from_indices requires torch."
        ) from _TORCH_IMPORT_ERROR
    return _FusedPostNormTwoLayerPointwiseLayerNormFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        ln_weight_stack,
        ln_bias_stack,
        float(ln_eps),
        int(pointwise_code),
    )


def fused_prenorm_two_layer_pointwise_rmsnorm_from_indices(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    rms_weight_stack: torch.Tensor,
    rms_eps: float,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    pointwise_code: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices requires torch."
        ) from _TORCH_IMPORT_ERROR
    return _FusedPreNormTwoLayerPointwiseRMSNormFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        rms_weight_stack,
        float(rms_eps),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def fused_program_two_layer_silu_then_two_layer_silu_from_indices(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fused_program_two_layer_silu_then_two_layer_silu_from_indices requires torch."
        ) from _TORCH_IMPORT_ERROR
    return _FusedProgramTwoLayerSiLUThenTwoLayerSiLUFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w10_stack,
        b10_stack,
        w20_stack,
        b20_stack,
        w11_stack,
        b11_stack,
        w21_stack,
        b21_stack,
    )


def fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
    ln_weight_stack: torch.Tensor,
    ln_bias_stack: torch.Tensor,
    ln_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices requires torch."
        ) from _TORCH_IMPORT_ERROR
    return _FusedProgramTwoLayerSiLUThenPostNormTwoLayerSiLUFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w10_stack,
        b10_stack,
        w20_stack,
        b20_stack,
        w11_stack,
        b11_stack,
        w21_stack,
        b21_stack,
        ln_weight_stack,
        ln_bias_stack,
        float(ln_eps),
    )


def fused_two_layer_mish_from_indices(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "fused_two_layer_mish_from_indices requires torch."
        ) from _TORCH_IMPORT_ERROR
    return fused_two_layer_pointwise_from_indices(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        _PW_MISH,
    )


__all__ = [
    "fanout_scatter",
    "fanin_reduce",
    "fanout_pack_multi",
    "fanin_pack_multi",
    "fanout_pack_from_edges",
    "fanin_pack_from_edges",
    "fused_two_layer_pointwise_from_indices",
    "fused_postnorm_two_layer_pointwise_layernorm_from_indices",
    "fused_prenorm_two_layer_pointwise_rmsnorm_from_indices",
    "fused_program_two_layer_silu_then_two_layer_silu_from_indices",
    "fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices",
    "fused_two_layer_mish_from_indices",
    "pointwise_code_from_signature",
    "available",
    "assert_runtime_compat",
]
