"""Custom autograd wrappers for mp custom ops and Python fallbacks."""

from __future__ import annotations

from .mp_constants import CUSTOM_TWO_LAYER_POINTWISE_CODES, MODE_LOGSUMEXP, MODE_SUM
from .mp_dispatch import namespace_has_op, ops_namespace, should_use_custom, use_custom_namespace_op
from .mp_fallbacks import (
    block_pointwise_pool_python,
    block_pointwise_python,
    block_postnorm_ln_python,
    block_prenorm_rms_python,
    fanin_reduce_python,
    fanout_scatter_python,
    lgan_pool_reduce_python,
    lgan_relation_graph_step_python,
    program_rmsnorm_silu_python,
    program_silu_pair_python,
    program_silu_postnorm_python,
)
from .mp_runtime import torch


def _detach_for_grad(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    return tensor.detach().requires_grad_(bool(enabled))


def _detach_optional_for_grad(tensor: torch.Tensor, enabled: bool) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    return tensor.detach().requires_grad_(bool(enabled))


def _collect_single_output_grads(
    output: torch.Tensor,
    grad_output: torch.Tensor,
    slots: tuple[tuple[int, torch.Tensor | None], ...],
    size: int,
) -> tuple[torch.Tensor | None, ...]:
    grad_map: list[torch.Tensor | None] = [None] * size
    grad_inputs: list[torch.Tensor] = []
    grad_targets: list[int] = []
    for idx, tensor in slots:
        if tensor is not None and tensor.requires_grad:
            grad_inputs.append(tensor)
            grad_targets.append(idx)
    grads = (
        torch.autograd.grad(output, grad_inputs, grad_output, allow_unused=True)
        if grad_inputs
        else ()
    )
    for idx, grad in zip(grad_targets, grads, strict=True):
        grad_map[idx] = grad
    return tuple(grad_map)


def _store_indexed_ctx(
    ctx: torch.autograd.function.FunctionCtx,
    *,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    **attrs,
) -> None:
    ctx.slot_offsets = [int(v) for v in slot_offsets]
    ctx.row_sizes = [int(v) for v in row_sizes]
    ctx.arity = int(arity)
    for name, value in attrs.items():
        setattr(ctx, name, value)


def _assign_requested_grads(
    size: int,
    needs: tuple[bool, ...],
    pairs: tuple[tuple[int, torch.Tensor | None], ...],
) -> tuple[torch.Tensor | None, ...]:
    grad_map: list[torch.Tensor | None] = [None] * size
    for idx, grad in pairs:
        if grad is not None and needs[idx]:
            grad_map[idx] = grad
    return tuple(grad_map)


def _use_custom_indexed_op(
    x: torch.Tensor,
    op_name: str,
    *,
    extra_condition: bool = True,
) -> bool:
    return bool(extra_condition) and use_custom_namespace_op(
        op_name,
        tensor=x,
        require_cuda=True,
    )


def _use_custom_backward(
    ctx: torch.autograd.function.FunctionCtx,
    grad_tensor: torch.Tensor,
    op_name: str,
) -> bool:
    return bool(getattr(ctx, "used_custom", False)) and use_custom_namespace_op(
        op_name,
        tensor=grad_tensor,
        require_cuda=True,
    )


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
            return ops_namespace().fanout_scatter(
                x_cat, src_global_idx, flat_dst, int(out_rows)
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor, None, None, None]:
            src_global_idx, flat_dst = ctx.saved_tensors
            grad_x = ops_namespace().fanout_scatter_backward(
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
            return ops_namespace().fanin_reduce(
                rel_flat, flat_src, dst_idx, int(dim_size), MODE_SUM
            )

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor, None, None, None]:
            flat_src, dst_idx = ctx.saved_tensors
            grad_rel = ops_namespace().fanin_reduce_sum_backward(
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
            out = ops_namespace().fanin_reduce(
                rel_flat, flat_src, dst_idx, int(dim_size), MODE_LOGSUMEXP
            )
            ctx.save_for_backward(rel_flat, flat_src, dst_idx, out)
            return out

        @staticmethod
        def backward(
            ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor
        ) -> tuple[torch.Tensor, None, None, None]:
            rel_flat, flat_src, dst_idx, out = ctx.saved_tensors
            grad_rel = ops_namespace().fanin_reduce_logsumexp_backward(
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
            used_custom = _use_custom_indexed_op(slot_messages, "lgan_pool_reduce")
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().lgan_pool_reduce(
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
            return lgan_pool_reduce_python(
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
                use_custom_backward = _use_custom_backward(
                    ctx,
                    slot_messages,
                    "lgan_pool_reduce_backward",
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
                        (int(ctx.entity_dim_size), int(slot_messages.size(-1))),
                        device=slot_messages.device,
                        dtype=slot_messages.dtype,
                    )
                )
                grad_nn_msgs_req = (
                    grad_nn_msgs
                    if grad_nn_msgs is not None
                    else torch.zeros(
                        (int(ctx.entity_dim_size), int(slot_messages.size(-1))),
                        device=slot_messages.device,
                        dtype=slot_messages.dtype,
                    )
                )
                if use_custom_backward:
                    return (
                        ops_namespace().lgan_pool_reduce_backward(
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
                    outputs = lgan_pool_reduce_python(
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
                    grad_outputs = tuple(
                        grad if grad is not None else torch.zeros_like(output)
                        for grad, output in zip(
                            (grad_relation_pair_x, grad_tn_msgs, grad_nn_msgs),
                            outputs,
                            strict=True,
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
            used_custom = _use_custom_indexed_op(
                relation_pair_x,
                "lgan_relation_graph_step",
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().lgan_relation_graph_step(
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
            return lgan_relation_graph_step_python(
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
            if needs[0] and _use_custom_backward(
                ctx,
                relation_pair_x,
                "lgan_relation_graph_step_backward",
            ):
                return (
                    ops_namespace().lgan_relation_graph_step_backward(
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
                    outputs = lgan_relation_graph_step_python(
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
                        outputs,
                        (relation_pair_x_req_in,),
                        grad_outputs=(
                            grad_relation_pair_x_req,
                            grad_tn_msgs_req,
                            grad_nn_msgs_req,
                        ),
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
            row_starts_groups: tuple[tuple[int, ...], ...],
            num_groups: int,
            *tensor_args: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            group_count = int(num_groups)
            if group_count < 0:
                raise ValueError("lgan_build_pointwise_step expects non-negative num_groups.")
            if len(arities) != group_count or len(pointwise_codes) != group_count:
                raise ValueError("lgan_build_pointwise_step metadata does not match group count.")
            if (
                len(slot_offsets_groups) != group_count
                or len(row_sizes_groups) != group_count
                or len(row_starts_groups) != group_count
            ):
                raise ValueError(
                    "lgan_build_pointwise_step offset metadata does not match group count."
                )
            if len(tensor_args) != 4 * group_count:
                raise ValueError(
                    "lgan_build_pointwise_step expects four parameter tensors per group."
                )

            param_tensors = tuple(tensor_args)
            w1_groups = tuple(param_tensors[0:group_count])
            b1_groups = tuple(param_tensors[group_count : 2 * group_count])
            w2_groups = tuple(param_tensors[2 * group_count : 3 * group_count])
            b2_groups = tuple(param_tensors[3 * group_count : 4 * group_count])

            relation_pair_x = seed_relation_pair_x.clone()
            use_custom_block = _use_custom_indexed_op(
                x,
                "block_pointwise",
            )
            for group_index in range(group_count):
                row_sizes = tuple(int(v) for v in row_sizes_groups[group_index])
                if sum(row_sizes) <= 0:
                    continue
                w1_stack = w1_groups[group_index]
                b1_stack = b1_groups[group_index]
                w2_stack = w2_groups[group_index]
                b2_stack = b2_groups[group_index]
                if use_custom_block and use_custom_namespace_op(
                    "block_pointwise_pool",
                    tensor=x,
                    require_cuda=True,
                ):
                    pooled = ops_namespace().block_pointwise_pool(
                        x,
                        relation_args,
                        list(slot_offsets_groups[group_index]),
                        list(row_sizes),
                        int(arities[group_index]),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(pointwise_codes[group_index]),
                    )
                else:
                    pooled = block_pointwise_pool_python(
                        x,
                        relation_args,
                        list(slot_offsets_groups[group_index]),
                        list(row_sizes),
                        int(arities[group_index]),
                        w1_stack,
                        b1_stack,
                        w2_stack,
                        b2_stack,
                        int(pointwise_codes[group_index]),
                    )
                pooled_offset = 0
                for row_start, row_count in zip(
                    row_starts_groups[group_index],
                    row_sizes,
                    strict=True,
                ):
                    row_count_i = int(row_count)
                    if row_count_i <= 0:
                        continue
                    relation_pair_x.narrow(0, int(row_start), row_count_i).copy_(
                        pooled.narrow(0, pooled_offset, row_count_i)
                    )
                    pooled_offset += row_count_i

            use_custom_graph = _use_custom_indexed_op(
                relation_pair_x,
                "lgan_relation_graph_step",
            )
            if use_custom_graph:
                relation_pair_x_out, tn_msgs, nn_msgs = ops_namespace().lgan_relation_graph_step(
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
                relation_pair_x_out, tn_msgs, nn_msgs = lgan_relation_graph_step_python(
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
            ctx.row_starts_groups = tuple(tuple(int(v) for v in values) for values in row_starts_groups)
            ctx.save_for_backward(
                x,
                relation_args,
                rr_src,
                rr_dst,
                tn_rel,
                tn_ent,
                nn_rel,
                nn_ent,
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
            param_tensors = saved[8:]
            w1_groups = tuple(param_tensors[0:group_count])
            b1_groups = tuple(param_tensors[group_count : 2 * group_count])
            w2_groups = tuple(param_tensors[2 * group_count : 3 * group_count])
            b2_groups = tuple(param_tensors[3 * group_count : 4 * group_count])

            if grad_relation_pair_x is None and grad_tn_msgs is None and grad_nn_msgs is None:
                return (None,) * (17 + 4 * group_count)

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

            if use_custom_namespace_op(
                "lgan_relation_graph_step_backward",
                tensor=x,
                require_cuda=True,
            ):
                grad_relation_pair_x_in = ops_namespace().lgan_relation_graph_step_backward(
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
                    outputs = lgan_relation_graph_step_python(
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
                        outputs,
                        (relation_pair_x_req_in,),
                        grad_outputs=(
                            grad_relation_pair_x_req,
                            grad_tn_msgs_req,
                            grad_nn_msgs_req,
                        ),
                        allow_unused=False,
                    )

            grad_x_total = torch.zeros_like(x) if needs[0] else None
            grad_seed = grad_relation_pair_x_in if needs[2] else None
            grads: list[torch.Tensor | None] = [None] * (17 + 4 * group_count)
            if needs[0]:
                grads[0] = grad_x_total
            if needs[2]:
                grads[2] = grad_seed

            use_custom_block_backward = use_custom_namespace_op(
                "block_pointwise_pool_backward",
                tensor=x,
                require_cuda=True,
            )
            w1_base = 17
            b1_base = w1_base + group_count
            w2_base = b1_base + group_count
            b2_base = w2_base + group_count
            for group_index in range(group_count):
                row_sizes = tuple(int(v) for v in ctx.row_sizes_groups[group_index])
                total_rows = int(sum(row_sizes))
                if total_rows <= 0:
                    continue
                arity = int(ctx.arities[group_index])
                grad_pooled = grad_relation_pair_x_in.new_empty((total_rows, int(x.size(-1))))
                pooled_offset = 0
                for row_start, row_count in zip(
                    ctx.row_starts_groups[group_index],
                    row_sizes,
                    strict=True,
                ):
                    row_count_i = int(row_count)
                    if row_count_i <= 0:
                        continue
                    grad_pooled.narrow(0, pooled_offset, row_count_i).copy_(
                        grad_relation_pair_x_in.narrow(0, int(row_start), row_count_i)
                    )
                    pooled_offset += row_count_i
                w1_stack = w1_groups[group_index]
                b1_stack = b1_groups[group_index]
                w2_stack = w2_groups[group_index]
                b2_stack = b2_groups[group_index]
                if use_custom_block_backward:
                    grad_x_i, grad_w1, grad_b1, grad_w2, grad_b2 = (
                        ops_namespace().block_pointwise_pool_backward(
                            grad_pooled,
                            x,
                            relation_args,
                            list(ctx.slot_offsets_groups[group_index]),
                            list(row_sizes),
                            arity,
                            w1_stack,
                            b1_stack,
                            w2_stack,
                            b2_stack,
                            int(ctx.pointwise_codes[group_index]),
                        )
                    )
                else:
                    with torch.enable_grad():
                        x_req = x.detach().requires_grad_(bool(needs[0]))
                        w1_req = _detach_for_grad(w1_stack, bool(needs[w1_base + group_index]))
                        b1_req = _detach_optional_for_grad(
                            b1_stack,
                            bool(needs[b1_base + group_index] and b1_stack.numel() > 0),
                        )
                        w2_req = _detach_for_grad(w2_stack, bool(needs[w2_base + group_index]))
                        b2_req = _detach_optional_for_grad(
                            b2_stack,
                            bool(needs[b2_base + group_index] and b2_stack.numel() > 0),
                        )
                        pooled_ref = block_pointwise_pool_python(
                            x_req,
                            relation_args,
                            list(ctx.slot_offsets_groups[group_index]),
                            list(row_sizes),
                            arity,
                            w1_req,
                            b1_req,
                            w2_req,
                            b2_req,
                            int(ctx.pointwise_codes[group_index]),
                        )
                        grad_x_i, grad_w1, grad_b1, grad_w2, grad_b2 = torch.autograd.grad(
                            (pooled_ref,),
                            (x_req, w1_req, b1_req, w2_req, b2_req),
                            grad_outputs=(grad_pooled,),
                            allow_unused=True,
                        )

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
            _store_indexed_ctx(
                ctx,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=arity,
                pointwise_code=int(pointwise_code),
            )
            ctx.save_for_backward(x, relation_args, w1_stack, b1_stack, w2_stack, b2_stack)
            used_custom = _use_custom_indexed_op(
                x,
                "block_pointwise",
                extra_condition=ctx.pointwise_code in CUSTOM_TWO_LAYER_POINTWISE_CODES,
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().block_pointwise(
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
            return block_pointwise_python(
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
            if _use_custom_backward(ctx, grad_rel, "block_pointwise_backward"):
                grad_x, grad_w1, grad_b1, grad_w2, grad_b2 = ops_namespace().block_pointwise_backward(
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
                return _assign_requested_grads(
                    10,
                    needs,
                    (
                        (0, grad_x),
                        (5, grad_w1),
                        (6, grad_b1 if b1_stack.numel() > 0 else None),
                        (7, grad_w2),
                        (8, grad_b2 if b2_stack.numel() > 0 else None),
                    ),
                )  # type: ignore[return-value]

            with torch.enable_grad():
                x_req = _detach_for_grad(x, bool(needs[0]))
                w1_req = _detach_for_grad(w1_stack, bool(needs[5]))
                b1_req = _detach_optional_for_grad(b1_stack, bool(needs[6] and b1_stack.numel() > 0))
                w2_req = _detach_for_grad(w2_stack, bool(needs[7]))
                b2_req = _detach_optional_for_grad(b2_stack, bool(needs[8] and b2_stack.numel() > 0))
                rel_cat, _ = block_pointwise_python(
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
                return _collect_single_output_grads(
                    rel_cat,
                    grad_rel,
                    (
                        (0, x_req),
                        (5, w1_req),
                        (6, b1_req if b1_stack.numel() > 0 else None),
                        (7, w2_req),
                        (8, b2_req if b2_stack.numel() > 0 else None),
                    ),
                    10,
                )  # type: ignore[return-value]


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
            _store_indexed_ctx(
                ctx,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=arity,
            )
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
            used_custom = _use_custom_indexed_op(x, "program_silu_pair")
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().program_silu_pair(
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
            return program_silu_pair_python(
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
            if _use_custom_backward(ctx, grad_rel, "program_silu_pair_backward"):
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
                ) = ops_namespace().program_silu_pair_backward(
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
                return _assign_requested_grads(
                    14,
                    needs,
                    (
                        (0, grad_x),
                        (5, grad_w10),
                        (6, grad_b10),
                        (7, grad_w20),
                        (8, grad_b20),
                        (9, grad_w11),
                        (10, grad_b11),
                        (11, grad_w21),
                        (12, grad_b21),
                    ),
                )

            with torch.enable_grad():
                x_req = _detach_for_grad(x, bool(needs[0]))
                w10_req = _detach_for_grad(w10_stack, bool(needs[5]))
                b10_req = _detach_for_grad(b10_stack, bool(needs[6]))
                w20_req = _detach_for_grad(w20_stack, bool(needs[7]))
                b20_req = _detach_for_grad(b20_stack, bool(needs[8]))
                w11_req = _detach_for_grad(w11_stack, bool(needs[9]))
                b11_req = _detach_for_grad(b11_stack, bool(needs[10]))
                w21_req = _detach_for_grad(w21_stack, bool(needs[11]))
                b21_req = _detach_for_grad(b21_stack, bool(needs[12]))
                rel_cat, _ = program_silu_pair_python(
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
                return _collect_single_output_grads(
                    rel_cat,
                    grad_rel,
                    (
                        (0, x_req),
                        (5, w10_req),
                        (6, b10_req),
                        (7, w20_req),
                        (8, b20_req),
                        (9, w11_req),
                        (10, b11_req),
                        (11, w21_req),
                        (12, b21_req),
                    ),
                    14,
                )


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
            _store_indexed_ctx(
                ctx,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=arity,
                ln_eps=float(ln_eps),
            )
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
            used_custom = _use_custom_indexed_op(x, "program_silu_postnorm")
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().program_silu_postnorm(
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
            return program_silu_postnorm_python(
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
            if _use_custom_backward(ctx, grad_rel, "program_silu_postnorm_backward"):
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
                ) = ops_namespace().program_silu_postnorm_backward(
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
                return _assign_requested_grads(
                    16,
                    needs,
                    (
                        (0, grad_x),
                        (5, grad_w10),
                        (6, grad_b10),
                        (7, grad_w20),
                        (8, grad_b20),
                        (9, grad_w11),
                        (10, grad_b11),
                        (11, grad_w21),
                        (12, grad_b21),
                        (13, grad_ln_weight if ln_weight_stack.numel() > 0 else None),
                        (14, grad_ln_bias if ln_bias_stack.numel() > 0 else None),
                    ),
                )

            with torch.enable_grad():
                x_req = _detach_for_grad(x, bool(needs[0]))
                w10_req = _detach_for_grad(w10_stack, bool(needs[5]))
                b10_req = _detach_for_grad(b10_stack, bool(needs[6]))
                w20_req = _detach_for_grad(w20_stack, bool(needs[7]))
                b20_req = _detach_for_grad(b20_stack, bool(needs[8]))
                w11_req = _detach_for_grad(w11_stack, bool(needs[9]))
                b11_req = _detach_for_grad(b11_stack, bool(needs[10]))
                w21_req = _detach_for_grad(w21_stack, bool(needs[11]))
                b21_req = _detach_for_grad(b21_stack, bool(needs[12]))
                ln_w_req = _detach_optional_for_grad(
                    ln_weight_stack,
                    bool(needs[13] and ln_weight_stack.numel() > 0),
                )
                ln_b_req = _detach_optional_for_grad(
                    ln_bias_stack,
                    bool(needs[14] and ln_bias_stack.numel() > 0),
                )
                rel_cat, _ = program_silu_postnorm_python(
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
                return _collect_single_output_grads(
                    rel_cat,
                    grad_rel,
                    (
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
                    ),
                    16,
                )


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
            _store_indexed_ctx(
                ctx,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=arity,
                rms_eps=float(rms_eps),
            )
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
            used_custom = _use_custom_indexed_op(x, "program_rmsnorm_silu")
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().program_rmsnorm_silu(
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
            return program_rmsnorm_silu_python(
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
            if _use_custom_backward(ctx, grad_rel, "program_rmsnorm_silu_backward"):
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
                ) = ops_namespace().program_rmsnorm_silu_backward(
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
                return _assign_requested_grads(
                    15,
                    needs,
                    (
                        (0, grad_x),
                        (5, grad_rms_weight if rms_weight_stack.numel() > 0 else None),
                        (7, grad_w10),
                        (8, grad_b10),
                        (9, grad_w20),
                        (10, grad_b20),
                        (11, grad_w11),
                        (12, grad_b11),
                        (13, grad_w21),
                        (14, grad_b21),
                    ),
                )

            with torch.enable_grad():
                x_req = _detach_for_grad(x, bool(needs[0]))
                rms_w_req = _detach_optional_for_grad(
                    rms_weight_stack,
                    bool(needs[5] and rms_weight_stack.numel() > 0),
                )
                w10_req = _detach_for_grad(w10_stack, bool(needs[7]))
                b10_req = _detach_for_grad(b10_stack, bool(needs[8]))
                w20_req = _detach_for_grad(w20_stack, bool(needs[9]))
                b20_req = _detach_for_grad(b20_stack, bool(needs[10]))
                w11_req = _detach_for_grad(w11_stack, bool(needs[11]))
                b11_req = _detach_for_grad(b11_stack, bool(needs[12]))
                w21_req = _detach_for_grad(w21_stack, bool(needs[13]))
                b21_req = _detach_for_grad(b21_stack, bool(needs[14]))
                rel_cat, _ = program_rmsnorm_silu_python(
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
                return _collect_single_output_grads(
                    rel_cat,
                    grad_rel,
                    (
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
                    ),
                    15,
                )


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
            _store_indexed_ctx(
                ctx,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=arity,
                ln_eps=float(ln_eps),
                pointwise_code=int(pointwise_code),
            )
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
            used_custom = _use_custom_indexed_op(
                x,
                "block_postnorm_ln",
                extra_condition=ctx.pointwise_code in CUSTOM_TWO_LAYER_POINTWISE_CODES,
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().block_postnorm_ln(
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
            return block_postnorm_ln_python(
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
            if _use_custom_backward(ctx, grad_rel, "block_postnorm_ln_backward"):
                grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_ln_weight, grad_ln_bias = (
                    ops_namespace().block_postnorm_ln_backward(
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
                return _assign_requested_grads(
                    13,
                    needs,
                    (
                        (0, grad_x),
                        (5, grad_w1),
                        (6, grad_b1 if b1_stack.numel() > 0 else None),
                        (7, grad_w2),
                        (8, grad_b2 if b2_stack.numel() > 0 else None),
                        (9, grad_ln_weight if ln_weight_stack.numel() > 0 else None),
                        (10, grad_ln_bias if ln_bias_stack.numel() > 0 else None),
                    ),
                )  # type: ignore[return-value]

            with torch.enable_grad():
                x_req = _detach_for_grad(x, bool(needs[0]))
                w1_req = _detach_for_grad(w1_stack, bool(needs[5]))
                b1_req = _detach_optional_for_grad(b1_stack, bool(needs[6] and b1_stack.numel() > 0))
                w2_req = _detach_for_grad(w2_stack, bool(needs[7]))
                b2_req = _detach_optional_for_grad(b2_stack, bool(needs[8] and b2_stack.numel() > 0))
                ln_w_req = _detach_optional_for_grad(
                    ln_weight_stack,
                    bool(needs[9] and ln_weight_stack.numel() > 0),
                )
                ln_b_req = _detach_optional_for_grad(
                    ln_bias_stack,
                    bool(needs[10] and ln_bias_stack.numel() > 0),
                )
                rel_cat, _ = block_postnorm_ln_python(
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
                return _collect_single_output_grads(
                    rel_cat,
                    grad_rel,
                    (
                        (0, x_req),
                        (5, w1_req),
                        (6, b1_req if b1_stack.numel() > 0 else None),
                        (7, w2_req),
                        (8, b2_req if b2_stack.numel() > 0 else None),
                        (9, ln_w_req if ln_weight_stack.numel() > 0 else None),
                        (10, ln_b_req if ln_bias_stack.numel() > 0 else None),
                    ),
                    13,
                )  # type: ignore[return-value]


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
            _store_indexed_ctx(
                ctx,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=arity,
                rms_eps=float(rms_eps),
                pointwise_code=int(pointwise_code),
            )
            ctx.save_for_backward(
                x,
                relation_args,
                rms_weight_stack,
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
            )
            used_custom = _use_custom_indexed_op(
                x,
                "block_prenorm_rms",
                extra_condition=ctx.pointwise_code in CUSTOM_TWO_LAYER_POINTWISE_CODES,
            )
            ctx.used_custom = bool(used_custom)
            if used_custom:
                return ops_namespace().block_prenorm_rms(
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
            return block_prenorm_rms_python(
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
            if _use_custom_backward(ctx, grad_rel, "block_prenorm_rms_backward"):
                grad_x, grad_rms_weight, grad_w1, grad_b1, grad_w2, grad_b2 = (
                    ops_namespace().block_prenorm_rms_backward(
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
                return _assign_requested_grads(
                    12,
                    needs,
                    (
                        (0, grad_x),
                        (5, grad_rms_weight if rms_weight_stack.numel() > 0 else None),
                        (7, grad_w1),
                        (8, grad_b1 if b1_stack.numel() > 0 else None),
                        (9, grad_w2),
                        (10, grad_b2 if b2_stack.numel() > 0 else None),
                    ),
                )  # type: ignore[return-value]

            with torch.enable_grad():
                x_req = _detach_for_grad(x, bool(needs[0]))
                rms_w_req = _detach_optional_for_grad(
                    rms_weight_stack,
                    bool(needs[5] and rms_weight_stack.numel() > 0),
                )
                w1_req = _detach_for_grad(w1_stack, bool(needs[7]))
                b1_req = _detach_optional_for_grad(b1_stack, bool(needs[8] and b1_stack.numel() > 0))
                w2_req = _detach_for_grad(w2_stack, bool(needs[9]))
                b2_req = _detach_optional_for_grad(b2_stack, bool(needs[10] and b2_stack.numel() > 0))
                rel_cat, _ = block_prenorm_rms_python(
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
                return _collect_single_output_grads(
                    rel_cat,
                    grad_rel,
                    (
                        (0, x_req),
                        (5, rms_w_req if rms_weight_stack.numel() > 0 else None),
                        (7, w1_req),
                        (8, b1_req if b1_stack.numel() > 0 else None),
                        (9, w2_req),
                        (10, b2_req if b2_stack.numel() > 0 else None),
                    ),
                    12,
                )  # type: ignore[return-value]

else:
    _FanoutScatterFunction = None
    _FaninReduceSumFunction = None
    _FaninReduceLogSumExpFunction = None
    _LGANPoolReduceFunction = None
    _LGANRelationGraphStepFunction = None
    _LGANPointwiseBuildStepFunction = None
    _FusedTwoLayerPointwiseFromIndicesFunction = None
    _FusedProgramTwoLayerSiLUThenTwoLayerSiLUFromIndicesFunction = None
    _FusedProgramTwoLayerSiLUThenPostNormTwoLayerSiLUFromIndicesFunction = None
    _FusedProgramPreNormTwoLayerSiLURMSNormThenTwoLayerSiLUFromIndicesFunction = None
    _FusedPostNormTwoLayerPointwiseLayerNormFromIndicesFunction = None
    _FusedPreNormTwoLayerPointwiseRMSNormFromIndicesFunction = None
