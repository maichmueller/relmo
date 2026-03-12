"""Relational message-passing operator wrappers."""

from __future__ import annotations

import os
import re
import sys
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
_MODE_MEAN = 2

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
    raw = (_env_first(("RELM_MP_FALLBACK",), "python") or "python")
    raw = raw.strip().lower()
    return raw if raw in {"python", "error"} else "python"


def activation_code(signature: tuple[object, ...] | None) -> int | None:
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
    if sys.platform == "darwin":
        patterns = (
            "relm_mp_ops*.dylib",
            "librelm_mp_ops*.dylib",
            "relm_mp_ops*.so",
        )
    elif os.name == "nt":
        patterns = ("relm_mp_ops*.pyd",)
    else:
        patterns = (
            "relm_mp_ops*.so",
            "librelm_mp_ops*.so",
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
    if not _env_bool_any(("RELM_MP_ENABLE",), True):
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


def _lgan_pool_reduce_python(
    slot_messages: torch.Tensor,
    slot_to_relation_instance: torch.Tensor,
    relation_instance_arities: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    entity_dim_size: int,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if slot_messages.dim() != 2:
        raise ValueError("lgan_pool_reduce expects slot_messages to be rank-2.")
    relation_count = int(relation_instance_arities.numel())
    relation_pair_x = slot_messages.new_zeros(
        (relation_count, int(slot_messages.size(-1)))
    )
    if int(slot_messages.numel()) > 0:
        relation_pair_x.index_add_(0, slot_to_relation_instance, slot_messages)
        counts = relation_instance_arities.to(
            device=slot_messages.device,
            dtype=slot_messages.dtype,
        ).view(-1, 1).clamp_min_(1.0)
        relation_pair_x = relation_pair_x / counts

    def _indexed_reduce(
        source_embeddings: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        out = source_embeddings.new_zeros((int(dim_size), int(source_embeddings.size(-1))))
        if int(source_index.numel()) == 0 or int(dim_size) == 0:
            return out
        gathered = source_embeddings.index_select(0, source_index)
        out.index_add_(0, target_index, gathered)
        if mode == _MODE_MEAN:
            counts = out.new_zeros((int(dim_size), 1))
            counts.index_add_(
                0,
                target_index,
                torch.ones(
                    (int(target_index.numel()), 1),
                    device=out.device,
                    dtype=out.dtype,
                ),
            )
            out = out / counts.clamp_min_(1.0)
        return out

    if mode not in (_MODE_SUM, _MODE_MEAN):
        raise ValueError("lgan_pool_reduce supports only sum and mean modes.")
    rr_msgs = _indexed_reduce(relation_pair_x, rr_src, rr_dst, relation_count)
    relation_pair_x = relation_pair_x + rr_msgs
    tn_msgs = _indexed_reduce(relation_pair_x, tn_rel, tn_ent, int(entity_dim_size))
    nn_msgs = _indexed_reduce(relation_pair_x, nn_rel, nn_ent, int(entity_dim_size))
    return relation_pair_x, tn_msgs, nn_msgs


def _lgan_relation_graph_step_python(
    relation_pair_x: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    entity_dim_size: int,
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if relation_pair_x.dim() != 2:
        raise ValueError("lgan_relation_graph_step expects relation_pair_x to be rank-2.")

    def _indexed_reduce(
        source_embeddings: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        dim_size: int,
    ) -> torch.Tensor:
        out = source_embeddings.new_zeros((int(dim_size), int(source_embeddings.size(-1))))
        if int(source_index.numel()) == 0 or int(dim_size) == 0:
            return out
        gathered = source_embeddings.index_select(0, source_index)
        out.index_add_(0, target_index, gathered)
        if mode == _MODE_MEAN:
            counts = out.new_zeros((int(dim_size), 1))
            counts.index_add_(
                0,
                target_index,
                torch.ones(
                    (int(target_index.numel()), 1),
                    device=out.device,
                    dtype=out.dtype,
                ),
            )
            out = out / counts.clamp_min_(1.0)
        return out

    if mode not in (_MODE_SUM, _MODE_MEAN):
        raise ValueError("lgan_relation_graph_step supports only sum and mean modes.")
    rr_msgs = _indexed_reduce(relation_pair_x, rr_src, rr_dst, int(relation_pair_x.size(0)))
    relation_pair_x = relation_pair_x + rr_msgs
    tn_msgs = _indexed_reduce(relation_pair_x, tn_rel, tn_ent, int(entity_dim_size))
    nn_msgs = _indexed_reduce(relation_pair_x, nn_rel, nn_ent, int(entity_dim_size))
    return relation_pair_x, tn_msgs, nn_msgs


def _pool_block_messages_to_rows(
    rel_cat: torch.Tensor,
    row_count: int,
    arity: int,
) -> torch.Tensor:
    if int(rel_cat.numel()) == 0 or int(row_count) <= 0:
        return rel_cat.new_zeros((0, int(rel_cat.size(-1))))
    return rel_cat.view(int(row_count), int(arity), int(rel_cat.size(-1))).mean(dim=1)


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


def _block_pointwise_python(
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
        raise ValueError("block_pointwise expects x to be rank-2.")
    if relation_args.dim() != 1:
        raise ValueError(
            "block_pointwise expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "block_pointwise expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError("block_pointwise expects arity > 0.")

    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if w1_stack.dim() != 3 or w2_stack.dim() != 3:
        raise ValueError("block_pointwise expects rank-3 weight stacks.")
    if int(w1_stack.size(0)) != groups or int(w2_stack.size(0)) != groups:
        raise ValueError("block_pointwise weight stacks must match group count.")
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            "block_pointwise weight stack dims do not match arity * emb."
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


def _program_silu_pair_python(
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
            "program_silu_pair expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "program_silu_pair expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "program_silu_pair expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "program_silu_pair expects arity > 0."
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
                f"program_silu_pair expects {name} rank-{rank}."
            )
        if int(tensor.size(0)) != groups:
            raise ValueError(
                "program_silu_pair expects all parameter stacks to match group count."
            )
    if int(w10_stack.size(2)) != in_dim or int(w20_stack.size(1)) != in_dim:
        raise ValueError(
            "program_silu_pair stage-1 dims do not match arity * emb."
        )
    if int(w11_stack.size(2)) != in_dim or int(w21_stack.size(1)) != in_dim:
        raise ValueError(
            "program_silu_pair stage-2 dims do not match arity * emb."
        )
    if int(w20_stack.size(2)) != int(w10_stack.size(1)) or int(b10_stack.size(1)) != int(w10_stack.size(1)):
        raise ValueError(
            "program_silu_pair stage-1 hidden dims do not match."
        )
    if int(w21_stack.size(2)) != int(w11_stack.size(1)) or int(b11_stack.size(1)) != int(w11_stack.size(1)):
        raise ValueError(
            "program_silu_pair stage-2 hidden dims do not match."
        )
    if int(b20_stack.size(1)) != in_dim or int(b21_stack.size(1)) != in_dim:
        raise ValueError(
            "program_silu_pair output bias dims do not match arity * emb."
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


def _program_silu_postnorm_python(
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
            "program_silu_postnorm expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "program_silu_postnorm expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "program_silu_postnorm expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "program_silu_postnorm expects arity > 0."
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
                f"program_silu_postnorm expects {name} rank-{rank}."
            )
        if int(tensor.size(0)) != groups:
            raise ValueError(
                "program_silu_postnorm expects all parameter stacks to match group count."
            )
    if int(w10_stack.size(2)) != in_dim or int(w20_stack.size(1)) != in_dim:
        raise ValueError(
            "program_silu_postnorm stage-1 dims do not match arity * emb."
        )
    if int(w11_stack.size(2)) != in_dim or int(w21_stack.size(1)) != in_dim:
        raise ValueError(
            "program_silu_postnorm stage-2 dims do not match arity * emb."
        )
    if int(w20_stack.size(2)) != int(w10_stack.size(1)) or int(b10_stack.size(1)) != int(w10_stack.size(1)):
        raise ValueError(
            "program_silu_postnorm stage-1 hidden dims do not match."
        )
    if int(w21_stack.size(2)) != int(w11_stack.size(1)) or int(b11_stack.size(1)) != int(w11_stack.size(1)):
        raise ValueError(
            "program_silu_postnorm stage-2 hidden dims do not match."
        )
    if int(b20_stack.size(1)) != in_dim or int(b21_stack.size(1)) != in_dim:
        raise ValueError(
            "program_silu_postnorm output bias dims do not match arity * emb."
        )
    if ln_weight_stack.numel() > 0:
        if ln_weight_stack.dim() != 2 or int(ln_weight_stack.size(0)) != groups:
            raise ValueError(
                "program_silu_postnorm ln_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "program_silu_postnorm ln_weight_stack dims do not match arity * emb."
            )
    if ln_bias_stack.numel() > 0:
        if ln_bias_stack.dim() != 2 or int(ln_bias_stack.size(0)) != groups:
            raise ValueError(
                "program_silu_postnorm ln_bias_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_bias_stack.size(1)) != in_dim:
            raise ValueError(
                "program_silu_postnorm ln_bias_stack dims do not match arity * emb."
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


def _program_rmsnorm_silu_python(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    rms_weight_stack: torch.Tensor,
    rms_eps: float,
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
            "program_rmsnorm_silu expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "program_rmsnorm_silu expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "program_rmsnorm_silu expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "program_rmsnorm_silu expects arity > 0."
        )
    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if rms_weight_stack.numel() > 0:
        if rms_weight_stack.dim() != 2 or int(rms_weight_stack.size(0)) != groups:
            raise ValueError(
                "program_rmsnorm_silu rms_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(rms_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "program_rmsnorm_silu rms_weight_stack dims do not match arity * emb."
            )
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
                f"program_rmsnorm_silu expects {name} rank-{rank}."
            )
        if int(tensor.size(0)) != groups:
            raise ValueError(
                "program_rmsnorm_silu expects all parameter stacks to match group count."
            )
    if int(w10_stack.size(2)) != in_dim or int(w20_stack.size(1)) != in_dim:
        raise ValueError(
            "program_rmsnorm_silu stage-1 dims do not match arity * emb."
        )
    if int(w11_stack.size(2)) != in_dim or int(w21_stack.size(1)) != in_dim:
        raise ValueError(
            "program_rmsnorm_silu stage-2 dims do not match arity * emb."
        )
    if int(w20_stack.size(2)) != int(w10_stack.size(1)) or int(b10_stack.size(1)) != int(w10_stack.size(1)):
        raise ValueError(
            "program_rmsnorm_silu stage-1 hidden dims do not match."
        )
    if int(w21_stack.size(2)) != int(w11_stack.size(1)) or int(b11_stack.size(1)) != int(w11_stack.size(1)):
        raise ValueError(
            "program_rmsnorm_silu stage-2 hidden dims do not match."
        )
    if int(b20_stack.size(1)) != in_dim or int(b21_stack.size(1)) != in_dim:
        raise ValueError(
            "program_rmsnorm_silu output bias dims do not match arity * emb."
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
        sq_mean = x_i.square().mean(dim=-1, keepdim=True)
        stage0_in = x_i * torch.rsqrt(sq_mean + float(rms_eps))
        if rms_weight_stack.numel() > 0:
            stage0_in = stage0_in * rms_weight_stack[i].unsqueeze(0)
        stage1 = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(stage0_in, w10_stack[i], b10_stack[i])),
            w20_stack[i],
            b20_stack[i],
        )
        stage2 = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(stage1, w11_stack[i], b11_stack[i])),
            w21_stack[i],
            b21_stack[i],
        )
        rel_parts.append((x_i + stage2).view(span, emb))
        node_parts.append(rel_idx)

    if not rel_parts:
        return x.new_empty((0, emb)), torch.empty(0, device=x.device, dtype=torch.int64)
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    node_idx = node_parts[0] if len(node_parts) == 1 else torch.cat(node_parts, dim=0)
    return rel_cat, node_idx


def _block_postnorm_ln_python(
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
            "block_postnorm_ln expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "block_postnorm_ln expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "block_postnorm_ln expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "block_postnorm_ln expects arity > 0."
        )

    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if w1_stack.dim() != 3 or w2_stack.dim() != 3:
        raise ValueError(
            "block_postnorm_ln expects rank-3 weight stacks."
        )
    if int(w1_stack.size(0)) != groups or int(w2_stack.size(0)) != groups:
        raise ValueError(
            "block_postnorm_ln weight stacks must match group count."
        )
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            "block_postnorm_ln weight stack dims do not match arity * emb."
        )
    if ln_weight_stack.numel() > 0:
        if ln_weight_stack.dim() != 2 or int(ln_weight_stack.size(0)) != groups:
            raise ValueError(
                "block_postnorm_ln ln_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "block_postnorm_ln ln_weight_stack dims do not match arity * emb."
            )
    if ln_bias_stack.numel() > 0:
        if ln_bias_stack.dim() != 2 or int(ln_bias_stack.size(0)) != groups:
            raise ValueError(
                "block_postnorm_ln ln_bias_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(ln_bias_stack.size(1)) != in_dim:
            raise ValueError(
                "block_postnorm_ln ln_bias_stack dims do not match arity * emb."
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


def _block_prenorm_rms_python(
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
            "block_prenorm_rms expects x to be rank-2."
        )
    if relation_args.dim() != 1:
        raise ValueError(
            "block_prenorm_rms expects relation_args to be rank-1."
        )
    if len(slot_offsets) != len(row_sizes):
        raise ValueError(
            "block_prenorm_rms expects slot_offsets and row_sizes with equal lengths."
        )
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(
            "block_prenorm_rms expects arity > 0."
        )

    relation_args_i64 = relation_args.to(dtype=torch.int64)
    emb = int(x.size(1))
    in_dim = int(emb * arity_i)
    groups = len(slot_offsets)
    if rms_weight_stack.numel() > 0:
        if rms_weight_stack.dim() != 2 or int(rms_weight_stack.size(0)) != groups:
            raise ValueError(
                "block_prenorm_rms rms_weight_stack must have shape [groups, in_dim] when non-empty."
            )
        if int(rms_weight_stack.size(1)) != in_dim:
            raise ValueError(
                "block_prenorm_rms rms_weight_stack dims do not match arity * emb."
            )
    if w1_stack.dim() != 3 or w2_stack.dim() != 3:
        raise ValueError(
            "block_prenorm_rms expects rank-3 weight stacks."
        )
    if int(w1_stack.size(0)) != groups or int(w2_stack.size(0)) != groups:
        raise ValueError(
            "block_prenorm_rms weight stacks must match group count."
        )
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            "block_prenorm_rms weight stack dims do not match arity * emb."
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

    class _LGANPoolReduceFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            slot_messages: torch.Tensor,
            slot_to_relation_instance: torch.Tensor,
            relation_instance_arities: torch.Tensor,
            rr_src: torch.Tensor,
            rr_dst: torch.Tensor,
            tn_rel: torch.Tensor,
            tn_ent: torch.Tensor,
            nn_rel: torch.Tensor,
            nn_ent: torch.Tensor,
            entity_dim_size: int,
            mode: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ctx.entity_dim_size = int(entity_dim_size)
            ctx.mode = int(mode)
            ctx.save_for_backward(
                slot_messages,
                slot_to_relation_instance,
                relation_instance_arities,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
            )
            used_custom = (
                slot_messages.is_cuda
                and _should_use_custom("lgan_pool_reduce")
                and _namespace_has_op("lgan_pool_reduce")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().lgan_pool_reduce(
                    slot_messages,
                    slot_to_relation_instance,
                    relation_instance_arities,
                    rr_src,
                    rr_dst,
                    tn_rel,
                    tn_ent,
                    nn_rel,
                    nn_ent,
                    int(ctx.entity_dim_size),
                    int(ctx.mode),
                )
            return _lgan_pool_reduce_python(
                slot_messages,
                slot_to_relation_instance,
                relation_instance_arities,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
                int(ctx.entity_dim_size),
                int(ctx.mode),
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_relation_pair_x: torch.Tensor | None,
            grad_tn_msgs: torch.Tensor | None,
            grad_nn_msgs: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            (
                slot_messages,
                slot_to_relation_instance,
                relation_instance_arities,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
            ) = ctx.saved_tensors
            if grad_relation_pair_x is None and grad_tn_msgs is None and grad_nn_msgs is None:
                return (None,) * 11
            needs = ctx.needs_input_grad
            grad_slot_messages: torch.Tensor | None = None
            if needs[0]:
                use_custom_backward = (
                    bool(getattr(ctx, "used_custom", False))
                    and slot_messages.is_cuda
                    and _should_use_custom("lgan_pool_reduce_backward")
                    and _namespace_has_op("lgan_pool_reduce_backward")
                )
                grad_relation_pair_x_req = (
                    grad_relation_pair_x
                    if grad_relation_pair_x is not None
                    else torch.zeros(
                        (
                            int(relation_instance_arities.numel()),
                            int(slot_messages.size(-1)),
                        ),
                        device=slot_messages.device,
                        dtype=slot_messages.dtype,
                    )
                )
                grad_tn_msgs_req = (
                    grad_tn_msgs
                    if grad_tn_msgs is not None
                    else torch.zeros(
                        (
                            int(ctx.entity_dim_size),
                            int(slot_messages.size(-1)),
                        ),
                        device=slot_messages.device,
                        dtype=slot_messages.dtype,
                    )
                )
                grad_nn_msgs_req = (
                    grad_nn_msgs
                    if grad_nn_msgs is not None
                    else torch.zeros(
                        (
                            int(ctx.entity_dim_size),
                            int(slot_messages.size(-1)),
                        ),
                        device=slot_messages.device,
                        dtype=slot_messages.dtype,
                    )
                )
                if use_custom_backward:
                    return (
                        _ops_namespace().lgan_pool_reduce_backward(
                            grad_relation_pair_x_req,
                            grad_tn_msgs_req,
                            grad_nn_msgs_req,
                            slot_to_relation_instance,
                            relation_instance_arities,
                            rr_src,
                            rr_dst,
                            tn_rel,
                            tn_ent,
                            nn_rel,
                            nn_ent,
                            int(relation_instance_arities.numel()),
                            int(ctx.mode),
                        ),
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
                with torch.enable_grad():
                    slot_messages_req = slot_messages.detach().requires_grad_(True)
                    relation_pair_x, tn_msgs, nn_msgs = _lgan_pool_reduce_python(
                        slot_messages_req,
                        slot_to_relation_instance,
                        relation_instance_arities,
                        rr_src,
                        rr_dst,
                        tn_rel,
                        tn_ent,
                        nn_rel,
                        nn_ent,
                        int(ctx.entity_dim_size),
                        int(ctx.mode),
                    )
                    outputs = (relation_pair_x, tn_msgs, nn_msgs)
                    grad_outputs = tuple(
                        g if g is not None else torch.zeros_like(o)
                        for g, o in zip(
                            (grad_relation_pair_x, grad_tn_msgs, grad_nn_msgs),
                            outputs,
                        )
                    )
                    (grad_slot_messages,) = torch.autograd.grad(
                        outputs,
                        (slot_messages_req,),
                        grad_outputs=grad_outputs,
                        allow_unused=True,
                    )
            return (
                grad_slot_messages,
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

    class _LGANRelationGraphStepFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            relation_pair_x: torch.Tensor,
            rr_src: torch.Tensor,
            rr_dst: torch.Tensor,
            tn_rel: torch.Tensor,
            tn_ent: torch.Tensor,
            nn_rel: torch.Tensor,
            nn_ent: torch.Tensor,
            entity_dim_size: int,
            mode: int,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            ctx.entity_dim_size = int(entity_dim_size)
            ctx.mode = int(mode)
            ctx.save_for_backward(
                relation_pair_x,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
            )
            used_custom = (
                relation_pair_x.is_cuda
                and _should_use_custom("lgan_relation_graph_step")
                and _namespace_has_op("lgan_relation_graph_step")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().lgan_relation_graph_step(
                    relation_pair_x,
                    rr_src,
                    rr_dst,
                    tn_rel,
                    tn_ent,
                    nn_rel,
                    nn_ent,
                    int(ctx.entity_dim_size),
                    int(ctx.mode),
                )
            return _lgan_relation_graph_step_python(
                relation_pair_x,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
                int(ctx.entity_dim_size),
                int(ctx.mode),
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_relation_pair_x: torch.Tensor | None,
            grad_tn_msgs: torch.Tensor | None,
            grad_nn_msgs: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            relation_pair_x, rr_src, rr_dst, tn_rel, tn_ent, nn_rel, nn_ent = ctx.saved_tensors
            if grad_relation_pair_x is None and grad_tn_msgs is None and grad_nn_msgs is None:
                return (None,) * 9
            needs = ctx.needs_input_grad
            grad_relation_pair_x_req = (
                grad_relation_pair_x
                if grad_relation_pair_x is not None
                else torch.zeros_like(relation_pair_x)
            )
            grad_tn_msgs_req = (
                grad_tn_msgs
                if grad_tn_msgs is not None
                else torch.zeros(
                    (int(ctx.entity_dim_size), int(relation_pair_x.size(-1))),
                    device=relation_pair_x.device,
                    dtype=relation_pair_x.dtype,
                )
            )
            grad_nn_msgs_req = (
                grad_nn_msgs
                if grad_nn_msgs is not None
                else torch.zeros(
                    (int(ctx.entity_dim_size), int(relation_pair_x.size(-1))),
                    device=relation_pair_x.device,
                    dtype=relation_pair_x.dtype,
                )
            )
            if (
                needs[0]
                and bool(getattr(ctx, "used_custom", False))
                and relation_pair_x.is_cuda
                and _should_use_custom("lgan_relation_graph_step_backward")
                and _namespace_has_op("lgan_relation_graph_step_backward")
            ):
                return (
                    _ops_namespace().lgan_relation_graph_step_backward(
                        grad_relation_pair_x_req,
                        grad_tn_msgs_req,
                        grad_nn_msgs_req,
                        rr_src,
                        rr_dst,
                        tn_rel,
                        tn_ent,
                        nn_rel,
                        nn_ent,
                        int(relation_pair_x.size(0)),
                        int(ctx.mode),
                    ),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )
            grad_relation_pair_x_out: torch.Tensor | None = None
            if needs[0]:
                with torch.enable_grad():
                    relation_pair_x_req_in = relation_pair_x.detach().requires_grad_(True)
                    relation_pair_x_out_ref, tn_msgs_ref, nn_msgs_ref = _lgan_relation_graph_step_python(
                        relation_pair_x_req_in,
                        rr_src,
                        rr_dst,
                        tn_rel,
                        tn_ent,
                        nn_rel,
                        nn_ent,
                        int(ctx.entity_dim_size),
                        int(ctx.mode),
                    )
                    (grad_relation_pair_x_out,) = torch.autograd.grad(
                        (relation_pair_x_out_ref, tn_msgs_ref, nn_msgs_ref),
                        (relation_pair_x_req_in,),
                        grad_outputs=(grad_relation_pair_x_req, grad_tn_msgs_req, grad_nn_msgs_req),
                        allow_unused=True,
                    )
            return (
                grad_relation_pair_x_out,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

    class _LGANPointwiseBuildStepFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: torch.autograd.function.FunctionCtx,
            x: torch.Tensor,
            relation_args: torch.Tensor,
            seed_relation_pair_x: torch.Tensor,
            rr_src: torch.Tensor,
            rr_dst: torch.Tensor,
            tn_rel: torch.Tensor,
            tn_ent: torch.Tensor,
            nn_rel: torch.Tensor,
            nn_ent: torch.Tensor,
            entity_dim_size: int,
            mode: int,
            arities: tuple[int, ...],
            pointwise_codes: tuple[int, ...],
            slot_offsets_groups: tuple[tuple[int, ...], ...],
            row_sizes_groups: tuple[tuple[int, ...], ...],
            num_groups: int,
            *tensor_args: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            group_count = int(num_groups)
            if group_count < 0:
                raise ValueError("lgan_build_pointwise_step expects non-negative num_groups.")
            if len(arities) != group_count or len(pointwise_codes) != group_count:
                raise ValueError("lgan_build_pointwise_step metadata does not match group count.")
            if len(slot_offsets_groups) != group_count or len(row_sizes_groups) != group_count:
                raise ValueError("lgan_build_pointwise_step offset metadata does not match group count.")
            if len(tensor_args) != 5 * group_count:
                raise ValueError("lgan_build_pointwise_step expects row-index and parameter tensors per group.")

            row_index_groups = tuple(tensor_args[:group_count])
            param_tensors = tuple(tensor_args[group_count:])
            if len(param_tensors) != 4 * group_count:
                raise ValueError("lgan_build_pointwise_step expects four parameter tensors per group.")
            w1_groups = tuple(param_tensors[0:group_count])
            b1_groups = tuple(param_tensors[group_count : 2 * group_count])
            w2_groups = tuple(param_tensors[2 * group_count : 3 * group_count])
            b2_groups = tuple(param_tensors[3 * group_count : 4 * group_count])

            relation_pair_x = seed_relation_pair_x.clone()
            use_custom_block = (
                x.is_cuda
                and _should_use_custom("block_pointwise")
                and _namespace_has_op("block_pointwise")
            )
            for group_index in range(group_count):
                row_indices = row_index_groups[group_index]
                if int(row_indices.numel()) == 0:
                    continue
                w1_stack = w1_groups[group_index]
                b1_stack = b1_groups[group_index]
                w2_stack = w2_groups[group_index]
                b2_stack = b2_groups[group_index]
                if use_custom_block:
                    rel_cat, _ = _ops_namespace().block_pointwise(
                        x,
                        relation_args,
                        list(slot_offsets_groups[group_index]),
                        list(row_sizes_groups[group_index]),
                        int(arities[group_index]),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(pointwise_codes[group_index]),
                    )
                else:
                    rel_cat, _ = _block_pointwise_python(
                        x,
                        relation_args,
                        list(slot_offsets_groups[group_index]),
                        list(row_sizes_groups[group_index]),
                        int(arities[group_index]),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(pointwise_codes[group_index]),
                    )
                pooled = _pool_block_messages_to_rows(
                    rel_cat,
                    row_count=int(row_indices.numel()),
                    arity=int(arities[group_index]),
                )
                relation_pair_x.index_copy_(0, row_indices, pooled)

            use_custom_graph = (
                relation_pair_x.is_cuda
                and _should_use_custom("lgan_relation_graph_step")
                and _namespace_has_op("lgan_relation_graph_step")
            )
            if use_custom_graph:
                relation_pair_x_out, tn_msgs, nn_msgs = _ops_namespace().lgan_relation_graph_step(
                    relation_pair_x,
                    rr_src,
                    rr_dst,
                    tn_rel,
                    tn_ent,
                    nn_rel,
                    nn_ent,
                    int(entity_dim_size),
                    int(mode),
                )
            else:
                relation_pair_x_out, tn_msgs, nn_msgs = _lgan_relation_graph_step_python(
                    relation_pair_x,
                    rr_src,
                    rr_dst,
                    tn_rel,
                    tn_ent,
                    nn_rel,
                    nn_ent,
                    int(entity_dim_size),
                    int(mode),
                )

            ctx.group_count = group_count
            ctx.relation_pair_rows = int(seed_relation_pair_x.size(0))
            ctx.entity_dim_size = int(entity_dim_size)
            ctx.mode = int(mode)
            ctx.arities = tuple(int(v) for v in arities)
            ctx.pointwise_codes = tuple(int(v) for v in pointwise_codes)
            ctx.slot_offsets_groups = tuple(tuple(int(v) for v in values) for values in slot_offsets_groups)
            ctx.row_sizes_groups = tuple(tuple(int(v) for v in values) for values in row_sizes_groups)
            ctx.save_for_backward(
                x,
                relation_args,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
                *row_index_groups,
                *param_tensors,
            )
            return relation_pair_x_out, tn_msgs, nn_msgs

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx,
            grad_relation_pair_x: torch.Tensor | None,
            grad_tn_msgs: torch.Tensor | None,
            grad_nn_msgs: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            group_count = int(ctx.group_count)
            saved = ctx.saved_tensors
            x = saved[0]
            relation_args = saved[1]
            rr_src, rr_dst, tn_rel, tn_ent, nn_rel, nn_ent = saved[2:8]
            row_index_groups = saved[8 : 8 + group_count]
            param_tensors = saved[8 + group_count :]
            w1_groups = tuple(param_tensors[0:group_count])
            b1_groups = tuple(param_tensors[group_count : 2 * group_count])
            w2_groups = tuple(param_tensors[2 * group_count : 3 * group_count])
            b2_groups = tuple(param_tensors[3 * group_count : 4 * group_count])

            if grad_relation_pair_x is None and grad_tn_msgs is None and grad_nn_msgs is None:
                return (None,) * (16 + 5 * group_count)

            needs = ctx.needs_input_grad
            grad_relation_pair_x_req = (
                grad_relation_pair_x
                if grad_relation_pair_x is not None
                else torch.zeros(
                    (int(ctx.relation_pair_rows), int(x.size(-1))),
                    device=x.device,
                    dtype=x.dtype,
                )
            )
            grad_tn_msgs_req = (
                grad_tn_msgs
                if grad_tn_msgs is not None
                else torch.zeros(
                    (int(ctx.entity_dim_size), int(x.size(-1))),
                    device=x.device,
                    dtype=x.dtype,
                )
            )
            grad_nn_msgs_req = (
                grad_nn_msgs
                if grad_nn_msgs is not None
                else torch.zeros(
                    (int(ctx.entity_dim_size), int(x.size(-1))),
                    device=x.device,
                    dtype=x.dtype,
                )
            )

            if (
                x.is_cuda
                and _should_use_custom("lgan_relation_graph_step_backward")
                and _namespace_has_op("lgan_relation_graph_step_backward")
            ):
                grad_relation_pair_x_in = _ops_namespace().lgan_relation_graph_step_backward(
                    grad_relation_pair_x_req,
                    grad_tn_msgs_req,
                    grad_nn_msgs_req,
                    rr_src,
                    rr_dst,
                    tn_rel,
                    tn_ent,
                    nn_rel,
                    nn_ent,
                    int(grad_relation_pair_x_req.size(0)),
                    int(ctx.mode),
                )
            else:
                with torch.enable_grad():
                    relation_pair_x_req_in = grad_relation_pair_x_req.detach().new_zeros(
                        grad_relation_pair_x_req.shape
                    ).requires_grad_(True)
                    relation_pair_x_out_ref, tn_msgs_ref, nn_msgs_ref = _lgan_relation_graph_step_python(
                        relation_pair_x_req_in,
                        rr_src,
                        rr_dst,
                        tn_rel,
                        tn_ent,
                        nn_rel,
                        nn_ent,
                        int(ctx.entity_dim_size),
                        int(ctx.mode),
                    )
                    (grad_relation_pair_x_in,) = torch.autograd.grad(
                        (relation_pair_x_out_ref, tn_msgs_ref, nn_msgs_ref),
                        (relation_pair_x_req_in,),
                        grad_outputs=(grad_relation_pair_x_req, grad_tn_msgs_req, grad_nn_msgs_req),
                        allow_unused=False,
                    )

            grad_x_total = (
                torch.zeros_like(x) if needs[0] else None
            )
            grad_seed = grad_relation_pair_x_in if needs[2] else None
            grads: list[torch.Tensor | None] = [None] * (16 + 5 * group_count)
            if needs[0]:
                grads[0] = grad_x_total
            if needs[2]:
                grads[2] = grad_seed

            use_custom_block_backward = (
                x.is_cuda
                and _should_use_custom("block_pointwise_backward")
                and _namespace_has_op("block_pointwise_backward")
            )
            w1_base = 16 + group_count
            b1_base = w1_base + group_count
            w2_base = b1_base + group_count
            b2_base = w2_base + group_count
            for group_index in range(group_count):
                row_indices = row_index_groups[group_index]
                if int(row_indices.numel()) == 0:
                    continue
                arity = int(ctx.arities[group_index])
                grad_pooled = grad_relation_pair_x_in.index_select(0, row_indices)
                grad_rel = grad_pooled.repeat_interleave(arity, dim=0) / float(arity)
                w1_stack = w1_groups[group_index]
                b1_stack = b1_groups[group_index]
                w2_stack = w2_groups[group_index]
                b2_stack = b2_groups[group_index]
                if use_custom_block_backward:
                    grad_x_i, grad_w1, grad_b1, grad_w2, grad_b2 = _ops_namespace().block_pointwise_backward(
                        grad_rel,
                        x,
                        relation_args,
                        list(ctx.slot_offsets_groups[group_index]),
                        list(ctx.row_sizes_groups[group_index]),
                        arity,
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(ctx.pointwise_codes[group_index]),
                    )
                else:
                    with torch.enable_grad():
                        x_req = x.detach().requires_grad_(bool(needs[0]))
                        w1_req = w1_stack.detach().requires_grad_(bool(needs[w1_base + group_index]))
                        b1_req = b1_stack.detach().requires_grad_(bool(needs[b1_base + group_index] and b1_stack.numel() > 0))
                        w2_req = w2_stack.detach().requires_grad_(bool(needs[w2_base + group_index]))
                        b2_req = b2_stack.detach().requires_grad_(bool(needs[b2_base + group_index] and b2_stack.numel() > 0))
                        rel_cat, _ = _block_pointwise_python(
                            x_req,
                            relation_args,
                            list(ctx.slot_offsets_groups[group_index]),
                            list(ctx.row_sizes_groups[group_index]),
                            arity,
                            w1_req,
                            b1_req,
                            w2_req,
                            b2_req,
                            int(ctx.pointwise_codes[group_index]),
                        )
                        grads_tuple = torch.autograd.grad(
                            (rel_cat,),
                            (x_req, w1_req, b1_req, w2_req, b2_req),
                            grad_outputs=(grad_rel,),
                            allow_unused=True,
                        )
                        grad_x_i, grad_w1, grad_b1, grad_w2, grad_b2 = grads_tuple

                if needs[0] and grad_x_total is not None and grad_x_i is not None:
                    grad_x_total.add_(grad_x_i)
                if needs[w1_base + group_index]:
                    grads[w1_base + group_index] = grad_w1
                if needs[b1_base + group_index] and b1_stack.numel() > 0:
                    grads[b1_base + group_index] = grad_b1
                if needs[w2_base + group_index]:
                    grads[w2_base + group_index] = grad_w2
                if needs[b2_base + group_index] and b2_stack.numel() > 0:
                    grads[b2_base + group_index] = grad_b2

            if needs[0]:
                grads[0] = grad_x_total
            return tuple(grads)

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
                and _should_use_custom("block_pointwise")
                and _namespace_has_op("block_pointwise")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().block_pointwise(
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
            return _block_pointwise_python(
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
                and _should_use_custom("block_pointwise_backward")
                and _namespace_has_op("block_pointwise_backward")
            ):
                grad_x, grad_w1, grad_b1, grad_w2, grad_b2 = (
                    _ops_namespace().block_pointwise_backward(
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
                rel_cat, _ = _block_pointwise_python(
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
                and _should_use_custom("program_silu_pair")
                and _namespace_has_op("program_silu_pair")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().program_silu_pair(
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
            return _program_silu_pair_python(
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
                    "program_silu_pair_backward"
                )
                and _namespace_has_op(
                    "program_silu_pair_backward"
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
                ) = _ops_namespace().program_silu_pair_backward(
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
                rel_cat, _ = _program_silu_pair_python(
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
                and _should_use_custom("program_silu_postnorm")
                and _namespace_has_op(
                    "program_silu_postnorm"
                )
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().program_silu_postnorm(
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
            return _program_silu_postnorm_python(
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
                    "program_silu_postnorm_backward"
                )
                and _namespace_has_op(
                    "program_silu_postnorm_backward"
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
                ) = _ops_namespace().program_silu_postnorm_backward(
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
                rel_cat, _ = _program_silu_postnorm_python(
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

    class _FusedProgramPreNormTwoLayerSiLURMSNormThenTwoLayerSiLUFromIndicesFunction(
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
            rms_weight_stack: torch.Tensor,
            rms_eps: float,
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
            ctx.rms_eps = float(rms_eps)
            ctx.save_for_backward(
                x,
                relation_args,
                rms_weight_stack,
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
                and _should_use_custom(
                    "program_rmsnorm_silu"
                )
                and _namespace_has_op(
                    "program_rmsnorm_silu"
                )
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().program_rmsnorm_silu(
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    rms_weight_stack,
                    float(ctx.rms_eps),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                )
            return _program_rmsnorm_silu_python(
                x,
                relation_args,
                list(ctx.slot_offsets),
                list(ctx.row_sizes),
                int(ctx.arity),
                rms_weight_stack,
                float(ctx.rms_eps),
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
                return (None,) * 15
            (
                x,
                relation_args,
                rms_weight_stack,
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
            grad_map: list[torch.Tensor | None] = [None] * 15
            if (
                bool(getattr(ctx, "used_custom", False))
                and grad_rel.is_cuda
                and _should_use_custom(
                    "program_rmsnorm_silu_backward"
                )
                and _namespace_has_op(
                    "program_rmsnorm_silu_backward"
                )
            ):
                (
                    grad_x,
                    grad_rms_weight,
                    grad_w10,
                    grad_b10,
                    grad_w20,
                    grad_b20,
                    grad_w11,
                    grad_b11,
                    grad_w21,
                    grad_b21,
                ) = _ops_namespace().program_rmsnorm_silu_backward(
                    grad_rel,
                    x,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    rms_weight_stack,
                    float(ctx.rms_eps),
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
                if needs[5] and rms_weight_stack.numel() > 0:
                    grad_map[5] = grad_rms_weight
                if needs[7]:
                    grad_map[7] = grad_w10
                if needs[8]:
                    grad_map[8] = grad_b10
                if needs[9]:
                    grad_map[9] = grad_w20
                if needs[10]:
                    grad_map[10] = grad_b20
                if needs[11]:
                    grad_map[11] = grad_w11
                if needs[12]:
                    grad_map[12] = grad_b11
                if needs[13]:
                    grad_map[13] = grad_w21
                if needs[14]:
                    grad_map[14] = grad_b21
                return tuple(grad_map)
            with torch.enable_grad():
                x_req = x.detach().requires_grad_(bool(needs[0]))
                rms_w_req = (
                    rms_weight_stack.detach().requires_grad_(bool(needs[5] and rms_weight_stack.numel() > 0))
                    if rms_weight_stack.numel() > 0
                    else rms_weight_stack
                )
                w10_req = w10_stack.detach().requires_grad_(bool(needs[7]))
                b10_req = b10_stack.detach().requires_grad_(bool(needs[8]))
                w20_req = w20_stack.detach().requires_grad_(bool(needs[9]))
                b20_req = b20_stack.detach().requires_grad_(bool(needs[10]))
                w11_req = w11_stack.detach().requires_grad_(bool(needs[11]))
                b11_req = b11_stack.detach().requires_grad_(bool(needs[12]))
                w21_req = w21_stack.detach().requires_grad_(bool(needs[13]))
                b21_req = b21_stack.detach().requires_grad_(bool(needs[14]))
                rel_cat, _ = _program_rmsnorm_silu_python(
                    x_req,
                    relation_args,
                    list(ctx.slot_offsets),
                    list(ctx.row_sizes),
                    int(ctx.arity),
                    rms_w_req,
                    float(ctx.rms_eps),
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
                    (5, rms_w_req if rms_weight_stack.numel() > 0 else None),
                    (7, w10_req),
                    (8, b10_req),
                    (9, w20_req),
                    (10, b20_req),
                    (11, w11_req),
                    (12, b11_req),
                    (13, w21_req),
                    (14, b21_req),
                ):
                    if tensor is not None and needs[idx]:
                        grad_inputs.append(tensor)
                        grad_targets.append(idx)
                grads = torch.autograd.grad(rel_cat, grad_inputs, grad_rel, allow_unused=True)
            for idx, grad in zip(grad_targets, grads):
                grad_map[idx] = grad
            return tuple(grad_map)

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
                and _should_use_custom("block_postnorm_ln")
                and _namespace_has_op("block_postnorm_ln")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().block_postnorm_ln(
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
            return _block_postnorm_ln_python(
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
                    "block_postnorm_ln_backward"
                )
                and _namespace_has_op(
                    "block_postnorm_ln_backward"
                )
            ):
                grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_ln_weight, grad_ln_bias = (
                    _ops_namespace().block_postnorm_ln_backward(
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
                rel_cat, _ = _block_postnorm_ln_python(
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
                and _should_use_custom("block_prenorm_rms")
                and _namespace_has_op("block_prenorm_rms")
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return _ops_namespace().block_prenorm_rms(
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
            return _block_prenorm_rms_python(
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
                and _should_use_custom("block_prenorm_rms_backward")
                and _namespace_has_op("block_prenorm_rms_backward")
            ):
                grad_x, grad_rms_weight, grad_w1, grad_b1, grad_w2, grad_b2 = (
                    _ops_namespace().block_prenorm_rms_backward(
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
                rel_cat, _ = _block_prenorm_rms_python(
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


def block_pointwise(
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
    """Run a width-preserving 2-layer pointwise block over packed relation rows.

    Computes ``Linear -> activation -> Linear`` on each packed tuple row and
    returns:
    1. packed residual messages with the gathered tuple input added once
    2. packed destination entity indices
    """
    if torch is None:
        raise ModuleNotFoundError(
            "block_pointwise requires torch."
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


def block_postnorm_ln(
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
    """Run ``Linear -> activation -> Linear -> LayerNorm`` on packed relation rows."""
    if torch is None:
        raise ModuleNotFoundError(
            "block_postnorm_ln requires torch."
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


def block_prenorm_rms(
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
    """Run ``RMSNorm -> Linear -> activation -> Linear`` on packed relation rows."""
    if torch is None:
        raise ModuleNotFoundError(
            "block_prenorm_rms requires torch."
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


def program_silu_pair(
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
    """Run the exact 2-stage SiLU relation program on packed relation rows."""
    if torch is None:
        raise ModuleNotFoundError(
            "program_silu_pair requires torch."
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


def program_silu_postnorm(
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
    """Run the exact SiLU then post-norm SiLU relation program."""
    if torch is None:
        raise ModuleNotFoundError(
            "program_silu_postnorm requires torch."
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


def program_rmsnorm_silu(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    rms_weight_stack: torch.Tensor,
    rms_eps: float,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact pre-norm RMSNorm then SiLU relation program."""
    if torch is None:
        raise ModuleNotFoundError(
            "program_rmsnorm_silu requires torch."
        ) from _TORCH_IMPORT_ERROR
    return _FusedProgramPreNormTwoLayerSiLURMSNormThenTwoLayerSiLUFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        rms_weight_stack,
        float(rms_eps),
        w10_stack,
        b10_stack,
        w20_stack,
        b20_stack,
        w11_stack,
        b11_stack,
        w21_stack,
        b21_stack,
    )


def _lgan_build_pointwise_step(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    seed_relation_pair_x: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    *,
    entity_dim_size: int,
    mode: str,
    arities: tuple[int, ...] | list[int],
    pointwise_codes: tuple[int, ...] | list[int],
    slot_offsets_groups: tuple[tuple[int, ...], ...] | list[list[int]],
    row_sizes_groups: tuple[tuple[int, ...], ...] | list[list[int]],
    row_indices_groups: tuple[torch.Tensor, ...] | list[torch.Tensor],
    w1_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    b1_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    w2_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    b2_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact pointwise relation-instance rows and run the LGAN graph step.

    This is the integrated exact-family LGAN path for grouped two-layer pointwise
    relation kernels. It accepts a seeded relation-instance table for fallback
    rows and overwrites the exact-group rows before running RR/TN/NN propagation.
    """
    if torch is None:
        raise ModuleNotFoundError("_lgan_build_pointwise_step requires torch.") from _TORCH_IMPORT_ERROR
    mode_key = str(mode).strip().lower()
    if mode_key == "sum":
        reduce_mode = _MODE_SUM
    elif mode_key == "mean":
        reduce_mode = _MODE_MEAN
    else:
        raise ValueError("_lgan_build_pointwise_step supports only mode='sum' or mode='mean'.")
    group_count = len(tuple(arities))
    if not (
        len(tuple(pointwise_codes))
        == len(tuple(slot_offsets_groups))
        == len(tuple(row_sizes_groups))
        == len(tuple(row_indices_groups))
        == len(tuple(w1_stacks))
        == len(tuple(b1_stacks))
        == len(tuple(w2_stacks))
        == len(tuple(b2_stacks))
        == group_count
    ):
        raise ValueError("_lgan_build_pointwise_step expects one metadata/parameter entry per group.")
    return _LGANPointwiseBuildStepFunction.apply(
        x,
        relation_args,
        seed_relation_pair_x,
        rr_src,
        rr_dst,
        tn_rel,
        tn_ent,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        int(reduce_mode),
        tuple(int(v) for v in arities),
        tuple(int(v) for v in pointwise_codes),
        tuple(tuple(int(x) for x in values) for values in slot_offsets_groups),
        tuple(tuple(int(x) for x in values) for values in row_sizes_groups),
        int(group_count),
        *tuple(row_indices_groups),
        *tuple(w1_stacks),
        *tuple(b1_stacks),
        *tuple(w2_stacks),
        *tuple(b2_stacks),
    )


def _lgan_pool_reduce(
    slot_messages: torch.Tensor,
    slot_to_relation_instance: torch.Tensor,
    relation_instance_arities: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    *,
    entity_dim_size: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool slot messages to relation instances and run RR/TN/NN indexed reductions."""
    if torch is None:
        raise ModuleNotFoundError("_lgan_pool_reduce requires torch.") from _TORCH_IMPORT_ERROR
    mode_key = str(mode).strip().lower()
    if mode_key == "sum":
        reduce_mode = _MODE_SUM
    elif mode_key == "mean":
        reduce_mode = _MODE_MEAN
    else:
        raise ValueError("_lgan_pool_reduce supports only mode='sum' or mode='mean'.")
    return _LGANPoolReduceFunction.apply(
        slot_messages,
        slot_to_relation_instance,
        relation_instance_arities,
        rr_src,
        rr_dst,
        tn_rel,
        tn_ent,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        int(reduce_mode),
    )


def _lgan_relation_graph_step(
    relation_pair_x: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    *,
    entity_dim_size: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run RR/TN/NN propagation on pooled relation-instance embeddings."""
    if torch is None:
        raise ModuleNotFoundError("_lgan_relation_graph_step requires torch.") from _TORCH_IMPORT_ERROR
    mode_key = str(mode).strip().lower()
    if mode_key == "sum":
        reduce_mode = _MODE_SUM
    elif mode_key == "mean":
        reduce_mode = _MODE_MEAN
    else:
        raise ValueError("_lgan_relation_graph_step supports only mode='sum' or mode='mean'.")
    return _LGANRelationGraphStepFunction.apply(
        relation_pair_x,
        rr_src,
        rr_dst,
        tn_rel,
        tn_ent,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        int(reduce_mode),
    )
__all__ = [
    "fanout_scatter",
    "fanin_reduce",
    "fanout_pack_multi",
    "fanin_pack_multi",
    "fanout_pack_from_edges",
    "fanin_pack_from_edges",
    "block_pointwise",
    "block_postnorm_ln",
    "block_prenorm_rms",
    "program_silu_pair",
    "program_silu_postnorm",
    "program_rmsnorm_silu",
    "activation_code",
    "available",
    "assert_runtime_compat",
]
