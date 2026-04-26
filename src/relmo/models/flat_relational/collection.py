"""Eager flat relation collection helpers.

The flat runtime intentionally stays on the eager path. The previous grouped
kernel dispatch surface is retained only as inert scaffolding so higher-level
APIs and imports remain stable while the execution path stays maintainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, cast

import torch
from torch import Tensor

from .topology import topology_cache_key
from .types import CentralizedBatchSpec, FlatTopology, KernelBatchPlan, RelationSlice


@dataclass(frozen=True)
class KernelExecutionLayout:
    """Retained scaffold for the removed grouped-kernel runtime."""

    groups: tuple[KernelBatchPlan, ...]
    fallback_indices: tuple[int, ...]


@dataclass(frozen=True)
class KernelExecutionContext:
    """Retained scaffold for the removed grouped-kernel runtime."""

    topology: FlatTopology
    layout: KernelExecutionLayout
    grouped_param_stacks: dict[tuple[Any, ...], Tensor]
    allow_persistent_stacks: bool
    fallback_arg_emb_all: Tensor | None = None


def run_relation_dispatch(
    *,
    context: KernelExecutionContext,
    on_group: Callable[[KernelBatchPlan, KernelExecutionContext], Iterable[int] | None],
    on_relation_slice: Callable[[RelationSlice, KernelExecutionContext], bool],
) -> set[int]:
    """Compatibility helper that now only drives fallback relation slices."""

    consumed: set[int] = set()
    topology = context.topology

    for grouped_batch in context.layout.groups:
        consumed_indices = on_group(grouped_batch, context)
        if consumed_indices is None:
            continue
        consumed.update(int(idx) for idx in consumed_indices)

    for relation_index in context.layout.fallback_indices:
        relation_slice = topology.relation_slices[int(relation_index)]
        if on_relation_slice(relation_slice, context):
            consumed.add(int(relation_index))

    for relation_slice in topology.relation_slices:
        relation_index = int(relation_slice.relation_index)
        if relation_index in consumed:
            continue
        if on_relation_slice(relation_slice, context):
            consumed.add(relation_index)

    return consumed


def build_centralized_batch_spec(layer: Any) -> CentralizedBatchSpec | None:
    if not layer.update_modules:
        return None
    first = layer.update_modules[0]
    required_attrs = (
        "central_module",
        "condition_embedding",
        "condition_index",
        "max_arity",
        "embedding_size",
        "condition_position",
        "include_slot_mask",
    )
    if any(not hasattr(first, attr) for attr in required_attrs):
        return None
    central_module = cast(torch.nn.Module, getattr(first, "central_module"))
    condition_embedding = cast(
        torch.nn.Embedding, getattr(first, "condition_embedding")
    )
    condition_position = str(getattr(first, "condition_position"))
    max_arity = int(getattr(first, "max_arity"))
    embedding_size = int(getattr(first, "embedding_size"))
    include_slot_mask = bool(getattr(first, "include_slot_mask"))
    condition_indices: list[int] = []
    for relation_index, module in enumerate(layer.update_modules):
        if any(not hasattr(module, attr) for attr in required_attrs):
            return None
        if getattr(module, "central_module") is not central_module:
            return None
        if getattr(module, "condition_embedding") is not condition_embedding:
            return None
        if str(getattr(module, "condition_position")) != condition_position:
            return None
        if int(getattr(module, "max_arity")) != max_arity:
            return None
        if int(getattr(module, "embedding_size")) != embedding_size:
            return None
        if bool(getattr(module, "include_slot_mask")) != include_slot_mask:
            return None
        if int(getattr(module, "arity", layer.relation_arities[relation_index])) != int(
            layer.relation_arities[relation_index]
        ):
            return None
        condition_indices.append(int(getattr(module, "condition_index")))
    if embedding_size != layer.embedding_size:
        return None
    return CentralizedBatchSpec(
        central_module=central_module,
        condition_embedding=condition_embedding,
        condition_position=condition_position,
        max_arity=max_arity,
        embedding_size=embedding_size,
        include_slot_mask=include_slot_mask,
        condition_indices=tuple(condition_indices),
    )


def _centralized_batch_spec(layer: Any) -> CentralizedBatchSpec | None:
    return layer._centralized_batch_spec_cache


def relation_row_starts(topology: FlatTopology) -> dict[int, int]:
    row_starts: dict[int, int] = {}
    cursor = 0
    for relation_slice in topology.relation_slices:
        row_starts[int(relation_slice.relation_index)] = cursor
        cursor += int(relation_slice.count)
    return row_starts


def build_kernel_execution_layout(
    layer: Any,
    topology: FlatTopology,
    *,
    cache: dict | None = None,
) -> KernelExecutionLayout:
    """Compatibility stub for the removed grouped-kernel runtime."""

    cache_key = topology_cache_key(topology)
    layout_cache_key = ("kernel_layout", cache_key)
    if cache is not None:
        cached = cache.get(layout_cache_key)
        if isinstance(cached, KernelExecutionLayout):
            return cached
    cached = layer._persistent_kernel_layout_cache.get(cache_key)
    if cached is not None:
        if cache is not None:
            cache[layout_cache_key] = cached
        return cached

    layout = KernelExecutionLayout(groups=tuple(), fallback_indices=tuple())
    layer._persistent_kernel_layout_cache[cache_key] = layout
    if cache is not None:
        cache[layout_cache_key] = layout
    return layout


def pool_grouped_kernel_messages(
    layer: Any,
    topology: FlatTopology,
    grouped_batch: KernelBatchPlan,
    relation_row_starts_map: dict[int, int],
    messages: Tensor,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Retained helper for compatibility with legacy kernel modules."""

    del topology
    row_index_dtype = torch.long
    if int(messages.numel()) == 0:
        return (
            messages.new_zeros((0, layer.embedding_size)),
            torch.empty((0,), device=device, dtype=row_index_dtype),
        )
    pooled_parts: list[Tensor] = []
    row_indices_parts: list[Tensor] = []
    msg_cursor = 0
    arity = int(grouped_batch.arity)
    for relation_index, row_count in zip(
        grouped_batch.relation_indices,
        grouped_batch.row_sizes,
        strict=True,
    ):
        row_count_i = int(row_count)
        if row_count_i <= 0:
            continue
        width = row_count_i * arity
        relation_msgs = messages[msg_cursor : msg_cursor + width]
        pooled_parts.append(
            relation_msgs.view(row_count_i, arity, layer.embedding_size).mean(dim=1)
        )
        row_indices_parts.append(
            torch.arange(
                relation_row_starts_map[int(relation_index)],
                relation_row_starts_map[int(relation_index)] + row_count_i,
                device=device,
                dtype=row_index_dtype,
            )
        )
        msg_cursor += width
    if not pooled_parts:
        return (
            messages.new_zeros((0, layer.embedding_size)),
            torch.empty((0,), device=device, dtype=row_index_dtype),
        )
    return torch.cat(pooled_parts, dim=0), torch.cat(row_indices_parts, dim=0)


def writeback_relation_instance_messages(
    relation_pair_x: Tensor,
    row_indices: Tensor,
    pooled_rows: Tensor,
) -> None:
    if int(row_indices.numel()) == 0:
        return
    relation_pair_x.index_copy_(0, row_indices, pooled_rows)


def collect_centralized_relation_messages(
    layer: Any,
    x: Tensor,
    relation_args: Tensor,
    topology: FlatTopology,
    *,
    spec: CentralizedBatchSpec,
    arg_emb_all: Tensor | None = None,
) -> tuple[Tensor, Tensor] | None:
    if int(relation_args.numel()) == 0:
        return None
    arg_emb_all = x.index_select(0, relation_args) if arg_emb_all is None else arg_emb_all
    target_width = int(spec.max_arity * layer.embedding_size)
    cond_dim = spec.condition_embedding.weight.size(-1)
    input_rows: list[Tensor] = []
    for relation_slice in topology.relation_slices:
        if relation_slice.count <= 0 or relation_slice.arity <= 0:
            continue
        rel_in = arg_emb_all[
            relation_slice.slot_start : relation_slice.slot_end
        ].view(relation_slice.count, relation_slice.arity * layer.embedding_size)
        if rel_in.size(-1) < target_width:
            pad = rel_in.new_zeros((rel_in.size(0), target_width - rel_in.size(-1)))
            rel_in = torch.cat([rel_in, pad], dim=-1)
        pieces = [rel_in]
        if spec.include_slot_mask:
            mask = rel_in.new_zeros((rel_in.size(0), spec.max_arity))
            mask[:, : relation_slice.arity] = 1.0
            pieces.append(mask)
        cond_idx = torch.tensor(
            spec.condition_indices[relation_slice.relation_index],
            device=x.device,
        )
        cond = spec.condition_embedding(cond_idx).view(1, cond_dim).expand(
            rel_in.size(0), cond_dim
        )
        if spec.condition_position == "pre":
            input_rows.append(torch.cat([cond, *pieces], dim=-1))
        else:
            input_rows.append(torch.cat([*pieces, cond], dim=-1))
    if not input_rows:
        return None
    central_in = torch.cat(input_rows, dim=0)
    central_out = spec.central_module(central_in)
    msg_chunks: list[Tensor] = []
    row_cursor = 0
    for relation_slice in topology.relation_slices:
        if relation_slice.count <= 0 or relation_slice.arity <= 0:
            continue
        row_end = row_cursor + relation_slice.count
        rel_out = central_out[
            row_cursor:row_end, : relation_slice.arity * layer.embedding_size
        ]
        msg_chunks.append(
            rel_out.contiguous().view(
                relation_slice.count * relation_slice.arity,
                layer.embedding_size,
            )
        )
        row_cursor = row_end
    rel_out_flat = torch.cat(msg_chunks, dim=0)
    return arg_emb_all + rel_out_flat, relation_args


def collect_eager_relation_messages(
    layer: Any,
    x: Tensor,
    relation_args: Tensor,
    relation_slice: RelationSlice,
    *,
    arg_emb_all: Tensor | None = None,
) -> tuple[Tensor, Tensor] | None:
    if relation_slice.count <= 0 or relation_slice.arity <= 0:
        return None
    flat_idx = relation_args[
        relation_slice.slot_start : relation_slice.slot_end
    ]
    module = layer.update_modules[relation_slice.relation_index]
    if arg_emb_all is not None:
        arg_emb = arg_emb_all[
            relation_slice.slot_start : relation_slice.slot_end
        ]
    else:
        arg_emb = x.index_select(0, flat_idx)
    rel_in = arg_emb.view(
        relation_slice.count,
        relation_slice.arity * layer.embedding_size,
    )
    rel_out = module(rel_in).view(
        relation_slice.count * relation_slice.arity,
        layer.embedding_size,
    )
    return arg_emb + rel_out, flat_idx


def _pool_eager_messages(
    layer: Any,
    relation_slice: RelationSlice,
    messages: Tensor,
    *,
    relation_row_starts_map: dict[int, int],
    relation_pair_x: Tensor,
) -> None:
    if relation_slice.count <= 0:
        return
    pooled = messages.view(
        relation_slice.count,
        relation_slice.arity,
        layer.embedding_size,
    ).mean(dim=1)
    row_start = relation_row_starts_map[int(relation_slice.relation_index)]
    relation_pair_x[row_start : row_start + relation_slice.count] = pooled


def collect_messages(
    layer: Any,
    x: Tensor,
    relation_args: Tensor,
    topology: FlatTopology,
    *,
    cache: dict | None = None,
) -> tuple[Tensor, Tensor] | None:
    slot_messages = collect_slot_messages(
        layer,
        x,
        relation_args,
        topology,
        cache=cache,
    )
    if slot_messages is None:
        return None
    return slot_messages, relation_args


def collect_slot_messages(
    layer: Any,
    x: Tensor,
    relation_args: Tensor,
    topology: FlatTopology,
    *,
    cache: dict | None = None,
) -> Tensor | None:
    """Materialize packed relation-slot messages in canonical slot order."""

    del cache
    centralized_spec = _centralized_batch_spec(layer)
    if centralized_spec is not None:
        centralized_arg_emb_all = (
            x.index_select(0, relation_args)
            if int(relation_args.numel()) > 0
            else None
        )
        centralized = collect_centralized_relation_messages(
            layer,
            x,
            relation_args,
            topology,
            spec=centralized_spec,
            arg_emb_all=centralized_arg_emb_all,
        )
        if centralized is None:
            return None
        msgs, _ = centralized
        return msgs
    if int(relation_args.numel()) == 0:
        return None

    slot_messages = x.new_zeros((int(relation_args.numel()), layer.embedding_size))
    arg_emb_all = (
        x.index_select(0, relation_args)
        if layer.use_relation_gather(x) and int(relation_args.numel()) > 0
        else None
    )
    for relation_slice in topology.relation_slices:
        direct = collect_eager_relation_messages(
            layer,
            x,
            relation_args,
            relation_slice,
            arg_emb_all=arg_emb_all,
        )
        if direct is None:
            continue
        msgs, _ = direct
        slot_messages[
            relation_slice.slot_start : relation_slice.slot_end
        ] = msgs
    return slot_messages


def collect_relation_instance_messages(
    layer: Any,
    x: Tensor,
    relation_args: Tensor,
    topology: FlatTopology,
    *,
    cache: dict | None = None,
) -> Tensor | None:
    del cache
    relation_instance_count = int(sum(int(s.count) for s in topology.relation_slices))
    if relation_instance_count == 0:
        return None

    row_starts = relation_row_starts(topology)
    centralized_spec = _centralized_batch_spec(layer)
    if centralized_spec is None:
        arg_emb_all = (
            x.index_select(0, relation_args)
            if layer.use_relation_gather(x) and int(relation_args.numel()) > 0
            else None
        )
        relation_pair_x = x.new_zeros((relation_instance_count, layer.embedding_size))
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0:
                continue
            direct = collect_eager_relation_messages(
                layer,
                x,
                relation_args,
                relation_slice,
                arg_emb_all=arg_emb_all,
            )
            if direct is None:
                continue
            msgs, _ = direct
            _pool_eager_messages(
                layer,
                relation_slice,
                msgs,
                relation_row_starts_map=row_starts,
                relation_pair_x=relation_pair_x,
            )
        return relation_pair_x

    slot_messages = collect_slot_messages(layer, x, relation_args, topology)
    if slot_messages is None:
        return None
    relation_pair_x = x.new_zeros((relation_instance_count, layer.embedding_size))
    for relation_slice in topology.relation_slices:
        if relation_slice.count <= 0:
            continue
        rel_slots = slot_messages[
            relation_slice.slot_start : relation_slice.slot_end
        ].view(relation_slice.count, relation_slice.arity, layer.embedding_size)
        row_start = row_starts[int(relation_slice.relation_index)]
        relation_pair_x[row_start : row_start + relation_slice.count] = rel_slots.mean(
            dim=1
        )
    return relation_pair_x


def run_lgan_pointwise_step(
    layer: Any,
    x: Tensor,
    relation_args: Tensor,
    topology: FlatTopology,
    *,
    rr_src: Tensor,
    rr_dst: Tensor,
    tn_rel: Tensor,
    tn_ent: Tensor,
    nn_rel: Tensor,
    nn_ent: Tensor,
    entity_dim_size: int,
    mode: str,
    cache: dict | None = None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Compatibility stub for the removed kernelized LGAN pointwise path."""

    del layer, x, relation_args, topology, rr_src, rr_dst, tn_rel, tn_ent
    del nn_rel, nn_ent, entity_dim_size, mode, cache
    return None


__all__ = [
    "KernelExecutionLayout",
    "KernelExecutionContext",
    "run_relation_dispatch",
    "build_centralized_batch_spec",
    "relation_row_starts",
    "build_kernel_execution_layout",
    "pool_grouped_kernel_messages",
    "writeback_relation_instance_messages",
    "collect_messages",
    "collect_slot_messages",
    "collect_relation_instance_messages",
    "run_lgan_pointwise_step",
]
