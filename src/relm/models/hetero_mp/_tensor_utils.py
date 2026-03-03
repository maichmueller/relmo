from __future__ import annotations

from typing import Mapping

import torch


def _match_ntype(node_type: str, candidates: tuple[str, ...], strict: bool) -> bool:
    # Shared node-type filter semantics across MPs.
    if strict:
        return node_type in candidates
    return any(c in node_type for c in candidates)


def _cat_or_single(parts: list[torch.Tensor], *, dim: int = 0) -> torch.Tensor:
    # Avoid torch.cat when there is only a single tensor (common for small schemas).
    if not parts:
        raise ValueError("_cat_or_single got empty parts.")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


def _finalize_pair_lists(
    lhs_lists: Mapping[str, list[torch.Tensor]],
    rhs_lists: Mapping[str, list[torch.Tensor]],
    *,
    dim: int = 0,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    # Turns per-key lists of tensors into per-key concatenated tensors.
    # Used only on cache-miss routing builds.
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k, lhs in lhs_lists.items():
        rhs = rhs_lists.get(k)
        if not rhs:
            continue
        out[k] = (_cat_or_single(lhs, dim=dim), _cat_or_single(rhs, dim=dim))
    return out


def _get_or_make_buffer(
    buffers: dict,
    key: str,
    *,
    ref: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    zero: bool = True,
) -> torch.Tensor:
    # Tiny buffer pool helper:
    # - allocate if missing or shape/device/dtype mismatch
    # - optionally zero-fill for deterministic semantics (e.g. padding slots)
    #
    # This is called in the per-iteration hot path, but the checks are cheap and replace
    # repeated open-coded allocation/zeroing logic.
    buf = buffers.get(key)
    dtype = dtype or ref.dtype
    if (
        buf is None
        or buf.device != ref.device
        or buf.dtype != dtype
        or tuple(buf.shape) != tuple(shape)
    ):
        buf = ref.new_empty(shape, dtype=dtype)
        buffers[key] = buf
        if zero:
            buf.zero_()
        return buf
    if zero:
        buf.zero_()
    return buf
