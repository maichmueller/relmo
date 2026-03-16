"""Pure Python scatter, reduce, and packing helpers for mp kernels."""

from __future__ import annotations

from .mp_common import (
    cat_or_empty,
    cat_or_single,
    empty_index,
    require_int64,
    require_matching_lengths,
    require_non_empty,
    require_positive_arity,
    require_rank,
)
from .mp_constants import MODE_LOGSUMEXP, MODE_SUM
from .mp_runtime import torch


def fanout_scatter_python(
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


def fanin_reduce_python(
    rel_flat: torch.Tensor,
    flat_src: torch.Tensor,
    dst_idx: torch.Tensor,
    dim_size: int,
    mode: int,
) -> torch.Tensor:
    emb = int(rel_flat.size(-1))
    if mode == MODE_SUM:
        out = rel_flat.new_zeros((int(dim_size), emb))
        if flat_src.numel() == 0 or int(dim_size) == 0:
            return out
        values = rel_flat.index_select(0, flat_src)
        out.index_add_(0, dst_idx, values)
        return out

    if mode == MODE_LOGSUMEXP:
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


def fanout_pack_multi_python(
    x_parts: list[torch.Tensor],
    src_idx_parts: list[torch.Tensor],
    flat_dst_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op_name = "fanout_pack_multi"
    require_non_empty(op_name, x_parts, "source tensor")
    require_matching_lengths(
        op_name,
        ("x_parts", x_parts),
        ("src_idx_parts", src_idx_parts),
        ("flat_dst_parts", flat_dst_parts),
    )

    row_offset = 0
    src_global_parts: list[torch.Tensor] = []
    x_cat_parts: list[torch.Tensor] = []
    dst_cat_parts: list[torch.Tensor] = []
    for x, src_idx, flat_dst in zip(x_parts, src_idx_parts, flat_dst_parts, strict=True):
        require_rank(op_name, "source tensor", x, 2)
        require_rank(op_name, "src_idx", src_idx, 1)
        require_rank(op_name, "flat_dst", flat_dst, 1)
        require_int64(op_name, "src_idx", src_idx)
        require_int64(op_name, "flat_dst", flat_dst)
        if src_idx.numel() != flat_dst.numel():
            raise ValueError(
                f"{op_name} expects src_idx and flat_dst lengths to match per source."
            )
        x_cat_parts.append(x)
        src_global_parts.append(src_idx + int(row_offset))
        dst_cat_parts.append(flat_dst)
        row_offset += int(x.size(0))

    return (
        cat_or_single(x_cat_parts, dim=0),
        cat_or_single(src_global_parts, dim=0),
        cat_or_single(dst_cat_parts, dim=0),
    )


def fanin_pack_multi_python(
    rel_parts: list[torch.Tensor],
    flat_src_parts: list[torch.Tensor],
    dst_idx_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op_name = "fanin_pack_multi"
    require_non_empty(op_name, rel_parts, "relation tensor")
    require_matching_lengths(
        op_name,
        ("rel_parts", rel_parts),
        ("flat_src_parts", flat_src_parts),
        ("dst_idx_parts", dst_idx_parts),
    )

    row_offset = 0
    rel_cat_parts: list[torch.Tensor] = []
    src_cat_parts: list[torch.Tensor] = []
    dst_cat_parts: list[torch.Tensor] = []
    for rel, flat_src, dst_idx in zip(
        rel_parts,
        flat_src_parts,
        dst_idx_parts,
        strict=True,
    ):
        require_rank(op_name, "relation tensor", rel, 2)
        require_rank(op_name, "flat_src", flat_src, 1)
        require_rank(op_name, "dst_idx", dst_idx, 1)
        require_int64(op_name, "flat_src", flat_src)
        require_int64(op_name, "dst_idx", dst_idx)
        if flat_src.numel() != dst_idx.numel():
            raise ValueError(
                f"{op_name} expects flat_src and dst_idx lengths to match per relation tensor."
            )
        rel_cat_parts.append(rel)
        src_cat_parts.append(flat_src + int(row_offset))
        dst_cat_parts.append(dst_idx)
        row_offset += int(rel.size(0))

    return (
        cat_or_single(rel_cat_parts, dim=0),
        cat_or_single(src_cat_parts, dim=0),
        cat_or_single(dst_cat_parts, dim=0),
    )


def fanout_pack_from_edges_python(
    x_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    src_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    slot_offset_parts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op_name = "fanout_pack_from_edges"
    require_non_empty(op_name, x_parts, "source tensor")
    require_matching_lengths(
        op_name,
        ("edge_src_parts", edge_src_parts),
        ("edge_dst_parts", edge_dst_parts),
        ("src_part_ids", src_part_ids),
        ("arity_parts", arity_parts),
        ("pos_parts", pos_parts),
        ("slot_offset_parts", slot_offset_parts),
    )

    ref = x_parts[0]
    x_offsets: list[int] = []
    row_offset = 0
    for x in x_parts:
        require_rank(op_name, "source tensor", x, 2)
        x_offsets.append(int(row_offset))
        row_offset += int(x.size(0))
    x_cat = cat_or_single(x_parts, dim=0)

    src_global_parts: list[torch.Tensor] = []
    flat_dst_parts: list[torch.Tensor] = []
    for edge_src, edge_dst, src_part, arity, pos, slot_offset in zip(
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
        strict=True,
    ):
        require_rank(op_name, "edge_src", edge_src, 1)
        require_rank(op_name, "edge_dst", edge_dst, 1)
        require_int64(op_name, "edge_src", edge_src)
        require_int64(op_name, "edge_dst", edge_dst)
        if edge_src.numel() != edge_dst.numel():
            raise ValueError(f"{op_name} expects edge src/dst lengths to match.")

        src_part_i = int(src_part)
        if src_part_i < 0 or src_part_i >= len(x_parts):
            raise ValueError(f"{op_name} src_part_ids out of range: {src_part_i!r}.")

        arity_i = require_positive_arity(op_name, arity)
        pos_i = int(pos)
        if pos_i < 0 or pos_i >= arity_i:
            raise ValueError(
                f"{op_name} expects pos in [0, arity), got pos={pos_i} arity={arity_i}."
            )

        src_global_parts.append(edge_src + int(x_offsets[src_part_i]))
        flat_dst_parts.append(int(slot_offset) + edge_dst * arity_i + pos_i)

    return (
        x_cat,
        cat_or_empty(src_global_parts, empty=empty_index(ref.device), dim=0),
        cat_or_empty(flat_dst_parts, empty=empty_index(ref.device), dim=0),
    )


def fanin_pack_from_edges_python(
    rel_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    rel_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    op_name = "fanin_pack_from_edges"
    require_non_empty(op_name, rel_parts, "relation tensor")
    require_matching_lengths(
        op_name,
        ("edge_src_parts", edge_src_parts),
        ("edge_dst_parts", edge_dst_parts),
        ("rel_part_ids", rel_part_ids),
        ("arity_parts", arity_parts),
        ("pos_parts", pos_parts),
    )

    mode_i = int(mode)
    if mode_i not in (0, 1):
        raise ValueError(f"{op_name} expects mode in {{0,1}}, got {mode_i!r}.")

    ref = rel_parts[0]
    rel_offsets: list[int] = []
    row_offset = 0
    for rel in rel_parts:
        require_rank(op_name, "relation tensor", rel, 2)
        rel_offsets.append(int(row_offset))
        row_offset += int(rel.size(0))
    rel_cat = cat_or_single(rel_parts, dim=0)

    flat_src_parts: list[torch.Tensor] = []
    dst_cat_parts: list[torch.Tensor] = []
    for edge_src, edge_dst, rel_part, arity, pos in zip(
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
        strict=True,
    ):
        require_rank(op_name, "edge_src", edge_src, 1)
        require_rank(op_name, "edge_dst", edge_dst, 1)
        require_int64(op_name, "edge_src", edge_src)
        require_int64(op_name, "edge_dst", edge_dst)
        if edge_src.numel() != edge_dst.numel():
            raise ValueError(f"{op_name} expects edge src/dst lengths to match.")

        rel_part_i = int(rel_part)
        if rel_part_i < 0 or rel_part_i >= len(rel_parts):
            raise ValueError(f"{op_name} rel_part_ids out of range: {rel_part_i!r}.")

        if mode_i == 1:
            flat_src_local = edge_src
        else:
            arity_i = require_positive_arity(op_name, arity)
            pos_i = int(pos)
            if pos_i < 0 or pos_i >= arity_i:
                raise ValueError(
                    f"{op_name} expects pos in [0, arity), got pos={pos_i} arity={arity_i}."
                )
            flat_src_local = edge_src * arity_i + pos_i

        flat_src_parts.append(flat_src_local + int(rel_offsets[rel_part_i]))
        dst_cat_parts.append(edge_dst)

    return (
        rel_cat,
        cat_or_empty(flat_src_parts, empty=empty_index(ref.device), dim=0),
        cat_or_empty(dst_cat_parts, empty=empty_index(ref.device), dim=0),
    )
