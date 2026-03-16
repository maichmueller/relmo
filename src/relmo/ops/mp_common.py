"""Shared helpers for validation, packing, and indexed reductions."""

from __future__ import annotations

from typing import NamedTuple, Sequence

from .mp_constants import MODE_MEAN, MODE_SUM
from .mp_runtime import torch


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in tensor.shape)


def require_non_empty(op_name: str, values: Sequence[object], what: str) -> None:
    if not values:
        raise ValueError(f"{op_name} requires at least one {what}.")


def require_matching_lengths(
    op_name: str,
    *named_sequences: tuple[str, Sequence[object]],
) -> None:
    if not named_sequences:
        return
    expected = len(named_sequences[0][1])
    if all(len(seq) == expected for _, seq in named_sequences[1:]):
        return
    got = ", ".join(f"{name}={len(seq)}" for name, seq in named_sequences)
    raise ValueError(f"{op_name} expects matching sequence lengths, got {got}.")


def require_rank(op_name: str, name: str, tensor: torch.Tensor, rank: int) -> None:
    if tensor.dim() != rank:
        raise ValueError(
            f"{op_name} expects {name} to be rank-{rank}, got shape {_shape(tensor)}."
        )


def require_int64(op_name: str, name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype != torch.int64:
        raise ValueError(
            f"{op_name} expects {name} to have dtype int64, got {tensor.dtype}."
        )


def require_positive_arity(op_name: str, arity: int) -> int:
    arity_i = int(arity)
    if arity_i <= 0:
        raise ValueError(f"{op_name} expects arity > 0.")
    return arity_i


def validate_group_layout(
    op_name: str,
    slot_offsets: list[int] | tuple[int, ...],
    row_sizes: list[int] | tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    slot_offsets_t = tuple(int(v) for v in slot_offsets)
    row_sizes_t = tuple(int(v) for v in row_sizes)
    if len(slot_offsets_t) != len(row_sizes_t):
        raise ValueError(
            f"{op_name} expects slot_offsets and row_sizes with equal lengths, "
            f"got {len(slot_offsets_t)} and {len(row_sizes_t)}."
        )
    return slot_offsets_t, row_sizes_t


def cat_or_single(parts: Sequence[torch.Tensor], *, dim: int = 0) -> torch.Tensor:
    return parts[0] if len(parts) == 1 else torch.cat(list(parts), dim=dim)


def cat_or_empty(
    parts: Sequence[torch.Tensor],
    *,
    empty: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    if not parts:
        return empty
    return cat_or_single(parts, dim=dim)


def empty_index(device: torch.device) -> torch.Tensor:
    return torch.empty(0, device=device, dtype=torch.int64)


def indexed_sum_or_mean(
    source_embeddings: torch.Tensor,
    source_index: torch.Tensor,
    target_index: torch.Tensor,
    dim_size: int,
    mode: int,
    *,
    op_name: str,
) -> torch.Tensor:
    if mode not in (MODE_SUM, MODE_MEAN):
        raise ValueError(f"{op_name} supports only sum and mean modes.")

    out = source_embeddings.new_zeros((int(dim_size), int(source_embeddings.size(-1))))
    if int(source_index.numel()) == 0 or int(dim_size) == 0:
        return out

    gathered = source_embeddings.index_select(0, source_index)
    out.index_add_(0, target_index, gathered)
    if mode == MODE_MEAN:
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


class RelationKernelLayout(NamedTuple):
    relation_args_i64: torch.Tensor
    slot_offsets: tuple[int, ...]
    row_sizes: tuple[int, ...]
    arity: int
    emb: int
    in_dim: int
    groups: int


class RelationGroupInputs(NamedTuple):
    group_index: int
    rel_idx: torch.Tensor
    x_rows: torch.Tensor
    rows: int
    span: int


def prepare_relation_kernel_layout(
    op_name: str,
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int] | tuple[int, ...],
    row_sizes: list[int] | tuple[int, ...],
    arity: int,
) -> RelationKernelLayout:
    require_rank(op_name, "x", x, 2)
    require_rank(op_name, "relation_args", relation_args, 1)
    slot_offsets_t, row_sizes_t = validate_group_layout(op_name, slot_offsets, row_sizes)
    arity_i = require_positive_arity(op_name, arity)
    emb = int(x.size(1))
    return RelationKernelLayout(
        relation_args_i64=relation_args.to(dtype=torch.int64),
        slot_offsets=slot_offsets_t,
        row_sizes=row_sizes_t,
        arity=arity_i,
        emb=emb,
        in_dim=emb * arity_i,
        groups=len(slot_offsets_t),
    )


def iter_relation_group_inputs(
    x: torch.Tensor,
    layout: RelationKernelLayout,
):
    for group_index, (slot, rows) in enumerate(
        zip(layout.slot_offsets, layout.row_sizes, strict=True)
    ):
        row_count = int(rows)
        if row_count <= 0:
            continue
        start = int(slot)
        span = row_count * int(layout.arity)
        rel_idx = layout.relation_args_i64.narrow(0, start, span)
        arg_emb = x.index_select(0, rel_idx)
        yield RelationGroupInputs(
            group_index=group_index,
            rel_idx=rel_idx,
            x_rows=arg_emb.view(row_count, int(layout.in_dim)),
            rows=row_count,
            span=span,
        )


def build_relation_outputs(
    x: torch.Tensor,
    emb: int,
    rel_parts: Sequence[torch.Tensor],
    node_parts: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rel_parts:
        return x.new_empty((0, emb)), empty_index(x.device)
    return (
        cat_or_single(rel_parts, dim=0),
        cat_or_single(node_parts, dim=0),
    )


def validate_stack_rank(
    op_name: str,
    name: str,
    tensor: torch.Tensor,
    *,
    rank: int,
    groups: int,
) -> None:
    require_rank(op_name, name, tensor, rank)
    if int(tensor.size(0)) != groups:
        raise ValueError(
            f"{op_name} expects {name} group dimension to match group count {groups}, "
            f"got shape {_shape(tensor)}."
        )


def validate_optional_group_matrix(
    op_name: str,
    name: str,
    tensor: torch.Tensor,
    *,
    groups: int,
    width: int,
) -> None:
    if tensor.numel() == 0:
        return
    validate_stack_rank(op_name, name, tensor, rank=2, groups=groups)
    if int(tensor.size(1)) != width:
        raise ValueError(
            f"{op_name} expects {name} last dimension {width}, got shape {_shape(tensor)}."
        )


def validate_two_layer_stacks(
    op_name: str,
    *,
    groups: int,
    in_dim: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
) -> None:
    validate_stack_rank(op_name, "w1_stack", w1_stack, rank=3, groups=groups)
    validate_stack_rank(op_name, "w2_stack", w2_stack, rank=3, groups=groups)
    if int(w1_stack.size(2)) != in_dim or int(w2_stack.size(1)) != in_dim:
        raise ValueError(
            f"{op_name} weight stack dims do not match arity * emb={in_dim}."
        )

    hidden = int(w1_stack.size(1))
    if int(w2_stack.size(2)) != hidden:
        raise ValueError(
            f"{op_name} hidden dims do not match between w1_stack and w2_stack."
        )

    validate_optional_group_matrix(
        op_name,
        "b1_stack",
        b1_stack,
        groups=groups,
        width=hidden,
    )
    validate_optional_group_matrix(
        op_name,
        "b2_stack",
        b2_stack,
        groups=groups,
        width=in_dim,
    )


def validate_two_stage_program_stacks(
    op_name: str,
    *,
    groups: int,
    in_dim: int,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
) -> None:
    for name, tensor, rank in (
        ("w10_stack", w10_stack, 3),
        ("b10_stack", b10_stack, 2),
        ("w20_stack", w20_stack, 3),
        ("b20_stack", b20_stack, 2),
        ("w11_stack", w11_stack, 3),
        ("b11_stack", b11_stack, 2),
        ("w21_stack", w21_stack, 3),
        ("b21_stack", b21_stack, 2),
    ):
        validate_stack_rank(op_name, name, tensor, rank=rank, groups=groups)

    if int(w10_stack.size(2)) != in_dim or int(w20_stack.size(1)) != in_dim:
        raise ValueError(f"{op_name} stage-1 dims do not match arity * emb={in_dim}.")
    if int(w11_stack.size(2)) != in_dim or int(w21_stack.size(1)) != in_dim:
        raise ValueError(f"{op_name} stage-2 dims do not match arity * emb={in_dim}.")

    hidden1 = int(w10_stack.size(1))
    hidden2 = int(w11_stack.size(1))
    if int(w20_stack.size(2)) != hidden1 or int(b10_stack.size(1)) != hidden1:
        raise ValueError(f"{op_name} stage-1 hidden dims do not match.")
    if int(w21_stack.size(2)) != hidden2 or int(b11_stack.size(1)) != hidden2:
        raise ValueError(f"{op_name} stage-2 hidden dims do not match.")
    if int(b20_stack.size(1)) != in_dim or int(b21_stack.size(1)) != in_dim:
        raise ValueError(f"{op_name} output bias dims do not match arity * emb={in_dim}.")
