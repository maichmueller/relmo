"""Relation message collection and pooling paths."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from ...ops import mp as relm_mp_ops
from ..flat_kernel_runtime import (
    FlatKernelRuntime,
    KernelExecutionContext,
)
from .kernels import GELUBlockKernel, MishBlockKernel, SiLUBlockKernel
from .types import (
    CentralizedBatchSpec,
    FlatTopology,
    KernelBatchPlan,
    RelationSlice,
)


class FlatRelationCollectorMixin:
    def _build_centralized_batch_spec(self) -> CentralizedBatchSpec | None:
        if not self.update_modules:
            return None
        first = self.update_modules[0]
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
        for relation_index, module in enumerate(self.update_modules):
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
            if int(getattr(module, "arity", self.relation_arities[relation_index])) != int(
                self.relation_arities[relation_index]
            ):
                return None
            condition_indices.append(int(getattr(module, "condition_index")))
        if embedding_size != self.embedding_size:
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

    def _centralized_batch_spec(self) -> CentralizedBatchSpec | None:
        return self._centralized_batch_spec_cache

    def _collect_centralized_relation_messages(
        self,
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
        target_width = int(spec.max_arity * self.embedding_size)
        cond_dim = int(spec.condition_embedding.weight.size(-1))
        input_rows: list[Tensor] = []
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0 or relation_slice.arity <= 0:
                continue
            rel_in = arg_emb_all[
                relation_slice.slot_start : relation_slice.slot_end
            ].view(relation_slice.count, relation_slice.arity * self.embedding_size)
            if int(rel_in.size(-1)) < target_width:
                pad = rel_in.new_zeros((int(rel_in.size(0)), target_width - int(rel_in.size(-1))))
                rel_in = torch.cat([rel_in, pad], dim=-1)
            pieces = [rel_in]
            if spec.include_slot_mask:
                mask = rel_in.new_zeros((int(rel_in.size(0)), spec.max_arity))
                mask[:, : relation_slice.arity] = 1.0
                pieces.append(mask)
            cond_idx = torch.tensor(
                spec.condition_indices[relation_slice.relation_index],
                device=x.device,
            )
            cond = spec.condition_embedding(cond_idx).view(1, cond_dim).expand(
                int(rel_in.size(0)), cond_dim
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
            rel_out = central_out[row_cursor:row_end, : relation_slice.arity * self.embedding_size]
            msg_chunks.append(
                rel_out.contiguous().view(
                    relation_slice.count * relation_slice.arity,
                    self.embedding_size,
                )
            )
            row_cursor = row_end
        rel_out_flat = torch.cat(msg_chunks, dim=0)
        return arg_emb_all + rel_out_flat, relation_args

    def _collect_eager_relation_messages(
        self,
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
        module = self.update_modules[relation_slice.relation_index]
        if arg_emb_all is not None:
            arg_emb = arg_emb_all[
                relation_slice.slot_start : relation_slice.slot_end
            ]
        else:
            arg_emb = x.index_select(0, flat_idx)
        rel_in = arg_emb.view(
            relation_slice.count,
            relation_slice.arity * self.embedding_size,
        )
        rel_out = module(rel_in).view(
            relation_slice.count * relation_slice.arity,
            self.embedding_size,
        )
        return arg_emb + rel_out, flat_idx

    def _collect_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        slot_messages = self.collect_slot_messages(
            x, relation_args, topology, cache=cache
        )
        if slot_messages is None:
            return None
        return slot_messages, relation_args

    def collect_slot_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> Tensor | None:
        """Materialize packed relation-slot messages in canonical slot order."""
        centralized_spec = self._centralized_batch_spec()
        if centralized_spec is not None:
            centralized_arg_emb_all = (
                x.index_select(0, relation_args)
                if int(relation_args.numel()) > 0
                else None
            )
            centralized = self._collect_centralized_relation_messages(
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
        slot_messages = x.new_zeros((int(relation_args.numel()), self.embedding_size))
        use_any_kernels = self._use_relation_kernels(
            x
        ) or self._use_program_kernels(x)
        arg_emb_all = (
            x.index_select(0, relation_args)
            if (not use_any_kernels)
            and self._use_relation_gather(x)
            and int(relation_args.numel()) > 0
            else None
        )

        if use_any_kernels:
            layout = self._get_kernel_layout(topology)
            grouped_param_stacks = (
                cache.setdefault("kernel_param_stacks", {})
                if cache is not None
                else {}
            )
            allow_persistent_stacks = (not self.training) and (
                not torch.is_grad_enabled()
            )
            fallback_arg_emb_all = (
                x.index_select(0, relation_args)
                if layout.fallback_indices
                and self._use_relation_gather(x)
                and int(relation_args.numel()) > 0
                else None
            )

            def _handle_group(
                grouped_batch: KernelBatchPlan,
                context: KernelExecutionContext,
            ) -> tuple[int, ...] | None:
                grouped = grouped_batch.kernel.collect(
                    self,
                    x,
                    relation_args,
                    topology,
                    grouped_batch,
                    grouped_param_stacks=context.grouped_param_stacks,
                    allow_persistent_stacks=context.allow_persistent_stacks,
                )
                if grouped is None:
                    return None
                msgs, _ = grouped
                msg_cursor = 0
                for relation_index, row_count in zip(
                    grouped_batch.relation_indices,
                    grouped_batch.row_sizes,
                    strict=True,
                ):
                    relation_slice = topology.relation_slices[relation_index]
                    width = int(row_count) * int(grouped_batch.arity)
                    slot_messages[
                        relation_slice.slot_start : relation_slice.slot_end
                    ] = msgs[msg_cursor : msg_cursor + width]
                    msg_cursor += width
                return tuple(int(idx_i) for idx_i in grouped_batch.relation_indices)

            def _handle_relation(
                relation_slice: RelationSlice,
                context: KernelExecutionContext,
            ) -> bool:
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=context.fallback_arg_emb_all,
                )
                if direct is None:
                    return False
                msgs, _ = direct
                slot_messages[
                    relation_slice.slot_start : relation_slice.slot_end
                ] = msgs
                return True

            FlatKernelRuntime.run_relation_dispatch(
                context=KernelExecutionContext(
                    topology=topology,
                    layout=layout,
                    grouped_param_stacks=grouped_param_stacks,
                    allow_persistent_stacks=allow_persistent_stacks,
                    fallback_arg_emb_all=fallback_arg_emb_all,
                ),
                on_group=_handle_group,
                on_relation_slice=_handle_relation,
            )
        else:
            for relation_slice in topology.relation_slices:
                direct = self._collect_eager_relation_messages(
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

    def _pool_grouped_kernel_messages(
        self,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        relation_row_starts: dict[int, int],
        messages: Tensor,
        *,
        device: torch.device,
        index_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        row_index_dtype = torch.long
        if int(messages.numel()) == 0:
            return (
                messages.new_zeros((0, self.embedding_size)),
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
                relation_msgs.view(row_count_i, arity, self.embedding_size).mean(dim=1)
            )
            row_indices_parts.append(
                torch.arange(
                    relation_row_starts[int(relation_index)],
                    relation_row_starts[int(relation_index)] + row_count_i,
                    device=device,
                    dtype=row_index_dtype,
                )
            )
            msg_cursor += width
        if not pooled_parts:
            return (
                messages.new_zeros((0, self.embedding_size)),
                torch.empty((0,), device=device, dtype=row_index_dtype),
            )
        return torch.cat(pooled_parts, dim=0), torch.cat(row_indices_parts, dim=0)

    def _collect_relation_instance_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> Tensor | None:
        relation_instance_count = int(sum(int(s.count) for s in topology.relation_slices))
        if relation_instance_count == 0:
            return None
        relation_row_starts: dict[int, int] = {}
        row_cursor = 0
        for relation_slice in topology.relation_slices:
            relation_row_starts[int(relation_slice.relation_index)] = row_cursor
            row_cursor += int(relation_slice.count)

        centralized_spec = self._centralized_batch_spec()
        use_any_kernels = self._use_relation_kernels(x) or self._use_program_kernels(x)
        if centralized_spec is None and not use_any_kernels:
            arg_emb_all = (
                x.index_select(0, relation_args)
                if self._use_relation_gather(x) and int(relation_args.numel()) > 0
                else None
            )
            relation_pair_x = x.new_zeros((relation_instance_count, self.embedding_size))
            for relation_slice in topology.relation_slices:
                if relation_slice.count <= 0:
                    continue
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                pooled = msgs.view(
                    relation_slice.count,
                    relation_slice.arity,
                    self.embedding_size,
                ).mean(dim=1)
                row_start = relation_row_starts[int(relation_slice.relation_index)]
                relation_pair_x[row_start : row_start + relation_slice.count] = pooled
            return relation_pair_x

        if centralized_spec is None and use_any_kernels:
            relation_pair_x = x.new_zeros((relation_instance_count, self.embedding_size))
            layout = self._get_kernel_layout(topology)
            grouped_param_stacks = (
                cache.setdefault("kernel_param_stacks", {})
                if cache is not None
                else {}
            )
            allow_persistent_stacks = (not self.training) and (
                not torch.is_grad_enabled()
            )
            fallback_arg_emb_all = (
                x.index_select(0, relation_args)
                if layout.fallback_indices
                and self._use_relation_gather(x)
                and int(relation_args.numel()) > 0
                else None
            )

            def _pool_eager_messages(relation_slice: RelationSlice, messages: Tensor) -> None:
                if relation_slice.count <= 0:
                    return
                pooled = messages.view(
                    relation_slice.count,
                    relation_slice.arity,
                    self.embedding_size,
                ).mean(dim=1)
                row_start = relation_row_starts[int(relation_slice.relation_index)]
                relation_pair_x[row_start : row_start + relation_slice.count] = pooled

            def _handle_group(
                grouped_batch: KernelBatchPlan,
                context: KernelExecutionContext,
            ) -> tuple[int, ...] | None:
                pooled = grouped_batch.kernel.collect_relation_instances(
                    self,
                    x,
                    relation_args,
                    topology,
                    grouped_batch,
                    relation_row_starts=relation_row_starts,
                    grouped_param_stacks=context.grouped_param_stacks,
                    allow_persistent_stacks=context.allow_persistent_stacks,
                )
                if pooled is None:
                    return None
                pooled_rows, row_indices = pooled
                if int(row_indices.numel()) > 0:
                    relation_pair_x.index_copy_(0, row_indices, pooled_rows)
                return tuple(int(idx_i) for idx_i in grouped_batch.relation_indices)

            def _handle_relation(
                relation_slice: RelationSlice,
                context: KernelExecutionContext,
            ) -> bool:
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=context.fallback_arg_emb_all,
                )
                if direct is None:
                    return False
                msgs, _ = direct
                _pool_eager_messages(relation_slice, msgs)
                return True

            FlatKernelRuntime.run_relation_dispatch(
                context=KernelExecutionContext(
                    topology=topology,
                    layout=layout,
                    grouped_param_stacks=grouped_param_stacks,
                    allow_persistent_stacks=allow_persistent_stacks,
                    fallback_arg_emb_all=fallback_arg_emb_all,
                ),
                on_group=_handle_group,
                on_relation_slice=_handle_relation,
            )
            return relation_pair_x

        slot_messages = self.collect_slot_messages(
            x, relation_args, topology, cache=cache
        )
        if slot_messages is None:
            return None
        relation_pair_x = x.new_zeros((relation_instance_count, self.embedding_size))
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0:
                continue
            rel_slots = slot_messages[
                relation_slice.slot_start : relation_slice.slot_end
            ].view(relation_slice.count, relation_slice.arity, self.embedding_size)
            row_start = relation_row_starts[int(relation_slice.relation_index)]
            relation_pair_x[row_start : row_start + relation_slice.count] = rel_slots.mean(dim=1)
        return relation_pair_x

    def _run_lgan_pointwise_step(
        self,
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
        if not self._use_relation_kernels(x):
            return None
        relation_instance_count = int(sum(int(s.count) for s in topology.relation_slices))
        if relation_instance_count == 0:
            return x.new_zeros((0, self.embedding_size)), x.new_zeros((int(entity_dim_size), self.embedding_size)), x.new_zeros((int(entity_dim_size), self.embedding_size))

        relation_row_starts: dict[int, int] = {}
        row_cursor = 0
        for relation_slice in topology.relation_slices:
            relation_row_starts[int(relation_slice.relation_index)] = row_cursor
            row_cursor += int(relation_slice.count)

        relation_pair_seed = x.new_zeros((relation_instance_count, self.embedding_size))
        layout = self._get_kernel_layout(topology)
        grouped_param_stacks = (
            cache.setdefault("kernel_param_stacks", {})
            if cache is not None
            else {}
        )
        allow_persistent_stacks = (not self.training) and (not torch.is_grad_enabled())
        fallback_arg_emb_all = (
            x.index_select(0, relation_args)
            if (layout.fallback_indices or layout.groups)
            and self._use_relation_gather(x)
            and int(relation_args.numel()) > 0
            else None
        )

        pointwise_groups: list[KernelBatchPlan] = []
        pointwise_codes: list[int] = []
        w1_stacks: list[Tensor] = []
        b1_stacks: list[Tensor] = []
        w2_stacks: list[Tensor] = []
        b2_stacks: list[Tensor] = []
        slot_offsets_groups: list[list[int]] = []
        row_sizes_groups: list[list[int]] = []
        row_starts_groups: list[list[int]] = []

        def _pool_eager_messages(relation_slice: RelationSlice, messages: Tensor) -> None:
            if relation_slice.count <= 0:
                return
            pooled = messages.view(
                relation_slice.count,
                relation_slice.arity,
                self.embedding_size,
            ).mean(dim=1)
            row_start = relation_row_starts[int(relation_slice.relation_index)]
            relation_pair_seed[row_start : row_start + relation_slice.count] = pooled

        exact_kernel_types = (MishBlockKernel, SiLUBlockKernel, GELUBlockKernel)
        runtime_context = KernelExecutionContext(
            topology=topology,
            layout=layout,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            fallback_arg_emb_all=fallback_arg_emb_all,
        )

        def _fallback_group_relations(
            grouped_batch: KernelBatchPlan,
            context: KernelExecutionContext,
        ) -> tuple[int, ...]:
            consumed_indices: list[int] = []
            for relation_index in grouped_batch.relation_indices:
                relation_slice = topology.relation_slices[relation_index]
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=context.fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                _pool_eager_messages(relation_slice, msgs)
                consumed_indices.append(int(relation_index))
            return tuple(consumed_indices)

        def _handle_group(
            grouped_batch: KernelBatchPlan,
            context: KernelExecutionContext,
        ) -> tuple[int, ...] | None:
            if not isinstance(grouped_batch.kernel, exact_kernel_types):
                return _fallback_group_relations(grouped_batch, context)

            batch_items = self._collect_block_batch_items(
                topology=topology,
                grouped_batch=grouped_batch,
                expected_kernel_types=exact_kernel_types,
            )
            if not batch_items:
                return _fallback_group_relations(grouped_batch, context)

            pointwise_code = self._resolve_pointwise_code(batch_items)
            if pointwise_code is None:
                return _fallback_group_relations(grouped_batch, context)

            group_key = (
                "lgan_pointwise_step",
                type(grouped_batch.kernel),
                grouped_batch.arity,
                grouped_batch.signature,
            )
            w1_stack, b1_stack, w2_stack, b2_stack = self._stack_block_two_layer_params(
                batch_items=batch_items,
                group_key=group_key,
                grouped_param_stacks=context.grouped_param_stacks,
                allow_persistent_stacks=context.allow_persistent_stacks,
            )
            w1_stacks.append(w1_stack)
            b1_stacks.append(b1_stack)
            w2_stacks.append(w2_stack)
            b2_stacks.append(b2_stack)

            pointwise_groups.append(grouped_batch)
            pointwise_codes.append(int(pointwise_code))
            slot_offsets_groups.append([int(item[0].slot_start) for item in batch_items])
            row_sizes_groups.append([int(item[0].count) for item in batch_items])
            row_starts_groups.append(
                [
                    int(relation_row_starts[int(relation_slice.relation_index)])
                    for relation_slice, _ in batch_items
                ]
            )
            return tuple(int(idx_i) for idx_i in grouped_batch.relation_indices)

        def _handle_relation(
            relation_slice: RelationSlice,
            context: KernelExecutionContext,
        ) -> bool:
            direct = self._collect_eager_relation_messages(
                x,
                relation_args,
                relation_slice,
                arg_emb_all=context.fallback_arg_emb_all,
            )
            if direct is None:
                return False
            msgs, _ = direct
            _pool_eager_messages(relation_slice, msgs)
            return True

        FlatKernelRuntime.run_relation_dispatch(
            context=runtime_context,
            on_group=_handle_group,
            on_relation_slice=_handle_relation,
        )

        if not pointwise_groups:
            return None

        return relm_mp_ops._lgan_build_pointwise_step(
            x,
            relation_args,
            relation_pair_seed,
            rr_src,
            rr_dst,
            tn_rel,
            tn_ent,
            nn_rel,
            nn_ent,
            entity_dim_size=int(entity_dim_size),
            mode=str(mode),
            arities=tuple(int(group.arity) for group in pointwise_groups),
            pointwise_codes=tuple(pointwise_codes),
            slot_offsets_groups=tuple(tuple(values) for values in slot_offsets_groups),
            row_sizes_groups=tuple(tuple(values) for values in row_sizes_groups),
            row_starts_groups=tuple(tuple(values) for values in row_starts_groups),
            w1_stacks=tuple(w1_stacks),
            b1_stacks=tuple(b1_stacks),
            w2_stacks=tuple(w2_stacks),
            b2_stacks=tuple(b2_stacks),
        )


__all__ = ["FlatRelationCollectorMixin"]
