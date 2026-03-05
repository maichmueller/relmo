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


def _grouped_stack_from_flat_python(
    flat: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
) -> torch.Tensor:
    if flat.dim() != 2:
        raise ValueError("grouped_stack_from_flat expects flat to be rank-2.")
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "grouped_stack_from_flat expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError("grouped_stack_from_flat expects arity > 0.")
    groups = len(slot_offsets)
    emb = int(flat.size(1))
    in_dim = int(arity_i * emb)
    max_rows = max((int(n) for n in row_sizes), default=0)
    out = flat.new_zeros((groups, max_rows, in_dim))
    if groups == 0 or max_rows == 0:
        return out
    flat_rows = int(flat.size(0))
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        if start < 0:
            raise ValueError("grouped_stack_from_flat expects non-negative slot offsets.")
        if start + span > flat_rows:
            raise ValueError(
                "grouped_stack_from_flat slice out of bounds: "
                f"start={start} span={span} flat_rows={flat_rows}."
            )
        src = flat.narrow(0, start, span).view(n, in_dim)
        out[i, :n, :].copy_(src)
    return out


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


def _grouped_residual_mlp_from_flat_python(
    flat: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    weight_stacks: list[torch.Tensor],
    bias_stacks: list[torch.Tensor],
    op_kinds: list[int],
    op_indices: list[int],
    pointwise_codes: list[int],
    truncated_dim: int,
    truncate_right: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if flat.dim() != 2:
        raise ValueError("grouped_residual_mlp_from_flat expects flat to be rank-2.")
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "grouped_residual_mlp_from_flat expects slot_offsets and row_sizes with equal lengths."
        )
    if len(op_kinds) != len(op_indices):
        raise ValueError(
            "grouped_residual_mlp_from_flat expects op_kinds and op_indices with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError("grouped_residual_mlp_from_flat expects arity > 0.")
    groups = len(slot_offsets)
    emb = int(flat.size(1))
    in_dim = int(emb * arity_i)
    max_rows = max((int(n) for n in row_sizes), default=0)
    x_stack = flat.new_zeros((groups, max_rows, in_dim))
    flat_rows = int(flat.size(0))
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        start = int(slot)
        span = int(n * arity_i)
        if start < 0:
            raise ValueError(
                "grouped_residual_mlp_from_flat expects non-negative slot offsets."
            )
        if start + span > flat_rows:
            raise ValueError(
                "grouped_residual_mlp_from_flat slice out of bounds: "
                f"start={start} span={span} flat_rows={flat_rows}."
            )
        x_stack[i, :n, :] = flat.narrow(0, start, span).view(n, in_dim)

    out_stack = x_stack
    for kind, op_idx in zip(op_kinds, op_indices):
        kind_i = int(kind)
        idx_i = int(op_idx)
        if kind_i == 0:
            if idx_i < 0 or idx_i >= len(weight_stacks):
                raise ValueError(
                    f"Linear op index out of range in grouped_residual_mlp_from_flat: {idx_i}."
                )
            w = weight_stacks[idx_i]
            out_stack = torch.matmul(out_stack, w.transpose(1, 2))
            if idx_i < len(bias_stacks):
                b = bias_stacks[idx_i]
                if b.numel() > 0:
                    out_stack = out_stack + b[:, None, :]
        elif kind_i == 1:
            if idx_i < 0 or idx_i >= len(pointwise_codes):
                raise ValueError(
                    f"Pointwise op index out of range in grouped_residual_mlp_from_flat: {idx_i}."
                )
            out_stack = _apply_pointwise_code(out_stack, int(pointwise_codes[idx_i]))
        else:
            raise ValueError(
                f"Unsupported op kind in grouped_residual_mlp_from_flat: {kind_i}."
            )

    trunc_dim = int(truncated_dim)
    if trunc_dim >= 0 and int(x_stack.size(-1)) != trunc_dim:
        if bool(truncate_right):
            out_stack = x_stack[..., :trunc_dim] + out_stack
        else:
            out_stack = x_stack[..., -trunc_dim:] + out_stack
    else:
        out_stack = x_stack + out_stack

    rel_parts: list[torch.Tensor] = []
    flat_parts: list[torch.Tensor] = []
    for i, (slot, rows) in enumerate(zip(slot_offsets, row_sizes)):
        n = int(rows)
        if n <= 0:
            continue
        rel_i = out_stack[i, :n, :].contiguous().view(n * arity_i, emb)
        flat_i = torch.arange(
            n * arity_i, device=flat.device, dtype=torch.int64
        ) + int(slot)
        rel_parts.append(rel_i)
        flat_parts.append(flat_i)
    if not rel_parts:
        return flat.new_empty((0, emb)), torch.empty(
            0, device=flat.device, dtype=torch.int64
        )
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    flat_dst = flat_parts[0] if len(flat_parts) == 1 else torch.cat(flat_parts, dim=0)
    return rel_cat, flat_dst


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


def grouped_stack_from_flat(
    flat: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
) -> torch.Tensor:
    if torch is None:
        raise ModuleNotFoundError(
            "grouped_stack_from_flat requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("grouped_stack_from_flat"):
        if _namespace_has_op("grouped_stack_from_flat"):
            return _ops_namespace().grouped_stack_from_flat(
                flat, slot_offsets, row_sizes, int(arity)
            )
        if _fallback_mode() == "error":
            raise RuntimeError(
                "Custom mp op grouped_stack_from_flat is unavailable in the loaded relm_mp library."
            )
    return _grouped_stack_from_flat_python(flat, slot_offsets, row_sizes, int(arity))


def grouped_residual_mlp_from_flat(
    flat: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    weight_stacks: list[torch.Tensor],
    bias_stacks: list[torch.Tensor],
    op_kinds: list[int],
    op_indices: list[int],
    pointwise_codes: list[int],
    truncated_dim: int,
    truncate_right: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ModuleNotFoundError(
            "grouped_residual_mlp_from_flat requires torch."
        ) from _TORCH_IMPORT_ERROR
    if _should_use_custom("grouped_residual_mlp_from_flat"):
        if _namespace_has_op("grouped_residual_mlp_from_flat"):
            return _ops_namespace().grouped_residual_mlp_from_flat(
                flat,
                slot_offsets,
                row_sizes,
                int(arity),
                weight_stacks,
                bias_stacks,
                op_kinds,
                op_indices,
                pointwise_codes,
                int(truncated_dim),
                bool(truncate_right),
            )
        if _fallback_mode() == "error":
            raise RuntimeError(
                "Custom mp op grouped_residual_mlp_from_flat is unavailable in the loaded relm_mp library."
            )
    return _grouped_residual_mlp_from_flat_python(
        flat,
        slot_offsets,
        row_sizes,
        int(arity),
        weight_stacks,
        bias_stacks,
        op_kinds,
        op_indices,
        pointwise_codes,
        int(truncated_dim),
        bool(truncate_right),
    )


__all__ = [
    "fanout_scatter",
    "fanin_reduce",
    "fanout_pack_multi",
    "fanin_pack_multi",
    "fanout_pack_from_edges",
    "fanin_pack_from_edges",
    "grouped_stack_from_flat",
    "grouped_residual_mlp_from_flat",
    "available",
    "assert_runtime_compat",
]
