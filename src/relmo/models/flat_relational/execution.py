"""Grouped parameter stacking and exact kernel execution helpers."""

from __future__ import annotations

from typing import Any, Sequence, cast

import torch
from torch import Tensor

from ...ops import mp as relm_mp_ops
from .types import KernelBatchPlan, KernelMatch, RelationSlice


def _get_grouped_param_stack(
    *,
    cache_key: tuple[Any, ...],
    tensors: list[Tensor],
    forward_cache: dict[tuple[Any, ...], Tensor],
    allow_persistent: bool,
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
) -> Tensor:
    cached_forward = forward_cache.get(cache_key)
    if cached_forward is not None:
        return cached_forward

    if allow_persistent and tensors:
        versions = tuple(int(getattr(tensor, "_version", -1)) for tensor in tensors)
        persistent = persistent_grouped_param_stacks.get(cache_key)
        if persistent is not None:
            stacked = persistent.get("tensor")
            if (
                torch.is_tensor(stacked)
                and persistent.get("versions") == versions
                and tuple(stacked.shape) == tuple(persistent.get("shape", ()))
                and stacked.device == tensors[0].device
                and stacked.dtype == tensors[0].dtype
            ):
                forward_cache[cache_key] = stacked
                return stacked

    stacked = torch.stack(tensors, dim=0)
    forward_cache[cache_key] = stacked
    if allow_persistent and tensors:
        persistent_grouped_param_stacks[cache_key] = {
            "tensor": stacked,
            "versions": tuple(int(getattr(tensor, "_version", -1)) for tensor in tensors),
            "shape": tuple(stacked.shape),
        }
    return stacked


def _stack_optional_tensors(
    *,
    cache_key: tuple[Any, ...],
    tensors: Sequence[Tensor | None],
    forward_cache: dict[tuple[Any, ...], Tensor],
    allow_persistent: bool,
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    empty_like: Tensor,
) -> Tensor:
    present = [tensor for tensor in tensors if tensor is not None]
    if not present:
        return empty_like.new_empty((0,))
    return _get_grouped_param_stack(
        cache_key=cache_key,
        tensors=present,
        forward_cache=forward_cache,
        allow_persistent=allow_persistent,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )


def _stack_block_two_layer_params(
    *,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    group_key: tuple[Any, ...],
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    w1_stack = _get_grouped_param_stack(
        cache_key=("w1", group_key),
        tensors=[item[1].linears[0].weight for item in batch_items],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )
    w2_stack = _get_grouped_param_stack(
        cache_key=("w2", group_key),
        tensors=[item[1].linears[1].weight for item in batch_items],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )
    b1_stack = _stack_optional_tensors(
        cache_key=("b1", group_key),
        tensors=[item[1].linears[0].bias for item in batch_items],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        empty_like=w1_stack,
    )
    b2_stack = _stack_optional_tensors(
        cache_key=("b2", group_key),
        tensors=[item[1].linears[1].bias for item in batch_items],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        empty_like=w2_stack,
    )
    return w1_stack, b1_stack, w2_stack, b2_stack


def _stack_program_stage_params(
    *,
    stage_matches: Sequence[KernelMatch],
    program_key: tuple[Any, ...],
    stage_tag: str,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
    w1_key = f"w1{stage_tag}"
    b1_key = f"b1{stage_tag}"
    w2_key = f"w2{stage_tag}"
    b2_key = f"b2{stage_tag}"
    if any(
        match.linears[0].bias is None or match.linears[1].bias is None
        for match in stage_matches
    ):
        return None
    w1_stack = _get_grouped_param_stack(
        cache_key=(w1_key, program_key),
        tensors=[match.linears[0].weight for match in stage_matches],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )
    b1_stack = _get_grouped_param_stack(
        cache_key=(b1_key, program_key),
        tensors=[cast(Tensor, match.linears[0].bias) for match in stage_matches],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )
    w2_stack = _get_grouped_param_stack(
        cache_key=(w2_key, program_key),
        tensors=[match.linears[1].weight for match in stage_matches],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )
    b2_stack = _get_grouped_param_stack(
        cache_key=(b2_key, program_key),
        tensors=[cast(Tensor, match.linears[1].bias) for match in stage_matches],
        forward_cache=grouped_param_stacks,
        allow_persistent=allow_persistent_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
    )
    return w1_stack, b1_stack, w2_stack, b2_stack


def _resolve_pointwise_code(
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
) -> int | None:
    if not batch_items:
        return None
    pointwise_signature = batch_items[0][1].spec.pointwise_signature
    pointwise_code = relm_mp_ops.activation_code(pointwise_signature)
    if pointwise_code is None:
        return None
    if any(
        item[1].spec.pointwise_signature != pointwise_signature
        for item in batch_items[1:]
    ):
        return None
    return int(pointwise_code)


def _run_two_layer_pointwise_kernel(
    x: Tensor,
    relation_args: Tensor,
    grouped_batch: KernelBatchPlan,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    *,
    embedding_size: int,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor] | None:
    if not batch_items:
        return None
    pointwise_code = _resolve_pointwise_code(batch_items)
    if pointwise_code is None:
        return None

    group_key = (
        "block_pointwise",
        type(grouped_batch.kernel),
        grouped_batch.arity,
        grouped_batch.signature,
    )
    w1_stack, b1_stack, w2_stack, b2_stack = _stack_block_two_layer_params(
        batch_items=batch_items,
        group_key=group_key,
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    return relm_mp_ops.block_pointwise(
        x,
        relation_args,
        [int(item[0].slot_start) for item in batch_items],
        [int(item[0].count) for item in batch_items],
        int(grouped_batch.arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def _run_postnorm_layernorm_kernel(
    x: Tensor,
    relation_args: Tensor,
    grouped_batch: KernelBatchPlan,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    *,
    embedding_size: int,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor] | None:
    if not batch_items:
        return None
    for _relation_slice, match in batch_items:
        if match.spec.norm_kind != "layernorm" or match.spec.norm_position != "post":
            return None
        if len(match.norm_modules) != 1:
            return None

    pointwise_code = _resolve_pointwise_code(batch_items)
    if pointwise_code is None:
        return None

    norm_module = batch_items[0][1].norm_modules[0]
    eps = float(getattr(norm_module, "eps", 1e-5))
    affine = bool(getattr(norm_module, "elementwise_affine", True))
    if any(item[1].norm_modules[0].__class__ is not norm_module.__class__ for item in batch_items[1:]):
        return None
    group_key = (
        "block_postnorm_ln",
        type(grouped_batch.kernel),
        grouped_batch.arity,
        grouped_batch.signature,
    )
    w1_stack, b1_stack, w2_stack, b2_stack = _stack_block_two_layer_params(
        batch_items=batch_items,
        group_key=group_key,
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    if affine:
        ln_weight_stack = _stack_optional_tensors(
            cache_key=("ln_w", group_key),
            tensors=[cast(torch.nn.LayerNorm, item[1].norm_modules[0]).weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
            persistent_grouped_param_stacks=persistent_grouped_param_stacks,
            empty_like=w2_stack,
        )
        ln_bias_stack = _stack_optional_tensors(
            cache_key=("ln_b", group_key),
            tensors=[cast(torch.nn.LayerNorm, item[1].norm_modules[0]).bias for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
            persistent_grouped_param_stacks=persistent_grouped_param_stacks,
            empty_like=w2_stack,
        )
    else:
        ln_weight_stack = w2_stack.new_empty((0,))
        ln_bias_stack = w2_stack.new_empty((0,))
    return relm_mp_ops.block_postnorm_ln(
        x,
        relation_args,
        [int(item[0].slot_start) for item in batch_items],
        [int(item[0].count) for item in batch_items],
        int(grouped_batch.arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        ln_weight_stack,
        ln_bias_stack,
        float(eps),
        int(pointwise_code),
    )


def _run_prenorm_rmsnorm_kernel(
    x: Tensor,
    relation_args: Tensor,
    grouped_batch: KernelBatchPlan,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    *,
    embedding_size: int,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor] | None:
    if not batch_items:
        return None
    for _relation_slice, match in batch_items:
        if match.spec.norm_kind != "rmsnorm" or match.spec.norm_position != "pre":
            return None
        if len(match.norm_modules) != 1:
            return None

    pointwise_code = _resolve_pointwise_code(batch_items)
    if pointwise_code is None:
        return None

    norm_module = batch_items[0][1].norm_modules[0]
    eps = float(getattr(norm_module, "eps", 1e-5))
    affine = bool(getattr(norm_module, "elementwise_affine", True))
    if any(item[1].norm_modules[0].__class__ is not norm_module.__class__ for item in batch_items[1:]):
        return None
    group_key = (
        "block_prenorm_rms",
        type(grouped_batch.kernel),
        grouped_batch.arity,
        grouped_batch.signature,
    )
    w1_stack, b1_stack, w2_stack, b2_stack = _stack_block_two_layer_params(
        batch_items=batch_items,
        group_key=group_key,
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    if affine:
        rms_weight_stack = _stack_optional_tensors(
            cache_key=("rms_w", group_key),
            tensors=[cast(torch.nn.Module, item[1].norm_modules[0]).weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
            persistent_grouped_param_stacks=persistent_grouped_param_stacks,
            empty_like=w2_stack,
        )
    else:
        rms_weight_stack = w2_stack.new_empty((0,))
    return relm_mp_ops.block_prenorm_rms(
        x,
        relation_args,
        [int(item[0].slot_start) for item in batch_items],
        [int(item[0].count) for item in batch_items],
        int(grouped_batch.arity),
        rms_weight_stack,
        float(eps),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def _run_silu_pair_program_kernel(
    x: Tensor,
    relation_args: Tensor,
    grouped_batch: KernelBatchPlan,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    *,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor] | None:
    if not batch_items:
        return None
    row_sizes = [int(item[0].count) for item in batch_items]
    slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
    stage0_matches = [item[1].program_matches[0] for item in batch_items]
    stage1_matches = [item[1].program_matches[1] for item in batch_items]
    program_key = (
        "program_kernel",
        type(grouped_batch.kernel),
        grouped_batch.arity,
        grouped_batch.signature,
    )
    stage0_params = _stack_program_stage_params(
        stage_matches=stage0_matches,
        program_key=program_key,
        stage_tag="0",
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    stage1_params = _stack_program_stage_params(
        stage_matches=stage1_matches,
        program_key=program_key,
        stage_tag="1",
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    if stage0_params is None or stage1_params is None:
        return None
    w10_stack, b10_stack, w20_stack, b20_stack = stage0_params
    w11_stack, b11_stack, w21_stack, b21_stack = stage1_params
    return relm_mp_ops.program_silu_pair(
        x,
        relation_args,
        slot_offsets_global,
        row_sizes,
        int(grouped_batch.arity),
        w10_stack,
        b10_stack,
        w20_stack,
        b20_stack,
        w11_stack,
        b11_stack,
        w21_stack,
        b21_stack,
    )


def _run_silu_then_postnorm_program_kernel(
    x: Tensor,
    relation_args: Tensor,
    grouped_batch: KernelBatchPlan,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    *,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor] | None:
    if not batch_items:
        return None
    row_sizes = [int(item[0].count) for item in batch_items]
    slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
    stage0_matches = [item[1].program_matches[0] for item in batch_items]
    stage1_matches = [item[1].program_matches[1] for item in batch_items]
    program_key = (
        "program_kernel",
        type(grouped_batch.kernel),
        grouped_batch.arity,
        grouped_batch.signature,
    )
    norm_module = stage1_matches[0].norm_modules[0]
    norm_signature = getattr(norm_module, "normalized_shape", ())
    if isinstance(norm_signature, int):
        norm_shape = (int(norm_signature),)
    else:
        norm_shape = tuple(int(v) for v in norm_signature)
    if norm_shape != (int(stage1_matches[0].spec.output_dim),):
        return None
    ln_eps = float(getattr(norm_module, "eps", 1e-5))
    stage0_params = _stack_program_stage_params(
        stage_matches=stage0_matches,
        program_key=program_key,
        stage_tag="0",
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    stage1_params = _stack_program_stage_params(
        stage_matches=stage1_matches,
        program_key=program_key,
        stage_tag="1",
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    if stage0_params is None or stage1_params is None:
        return None
    w10_stack, b10_stack, w20_stack, b20_stack = stage0_params
    w11_stack, b11_stack, w21_stack, b21_stack = stage1_params
    if norm_module.weight is not None:
        ln_weight_stack = _stack_optional_tensors(
            cache_key=("ln_weight", program_key),
            tensors=[cast(Tensor, match.norm_modules[0].weight) for match in stage1_matches],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
            persistent_grouped_param_stacks=persistent_grouped_param_stacks,
            empty_like=w21_stack,
        )
    else:
        ln_weight_stack = w21_stack.new_empty((0,))
    if norm_module.bias is not None:
        ln_bias_stack = _stack_optional_tensors(
            cache_key=("ln_bias", program_key),
            tensors=[cast(Tensor, match.norm_modules[0].bias) for match in stage1_matches],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
            persistent_grouped_param_stacks=persistent_grouped_param_stacks,
            empty_like=w21_stack,
        )
    else:
        ln_bias_stack = w21_stack.new_empty((0,))
    return relm_mp_ops.program_silu_postnorm(
        x,
        relation_args,
        slot_offsets_global,
        row_sizes,
        int(grouped_batch.arity),
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


def _run_prenorm_rmsnorm_then_silu_program_kernel(
    x: Tensor,
    relation_args: Tensor,
    grouped_batch: KernelBatchPlan,
    batch_items: Sequence[tuple[RelationSlice, KernelMatch]],
    *,
    grouped_param_stacks: dict[tuple[Any, ...], Tensor],
    persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]],
    allow_persistent_stacks: bool,
) -> tuple[Tensor, Tensor] | None:
    if not batch_items:
        return None
    row_sizes = [int(item[0].count) for item in batch_items]
    slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
    stage0_matches = [item[1].program_matches[0] for item in batch_items]
    stage1_matches = [item[1].program_matches[1] for item in batch_items]
    program_key = (
        "program_kernel",
        type(grouped_batch.kernel),
        grouped_batch.arity,
        grouped_batch.signature,
    )
    norm_module = stage0_matches[0].norm_modules[0]
    norm_signature = getattr(norm_module, "normalized_shape", ())
    if isinstance(norm_signature, int):
        norm_shape = (int(norm_signature),)
    else:
        norm_shape = tuple(int(v) for v in norm_signature)
    if norm_shape != (int(stage0_matches[0].spec.output_dim),):
        return None
    rms_eps = float(getattr(norm_module, "eps", 1e-5))
    if norm_module.weight is not None:
        rms_weight_stack = _stack_optional_tensors(
            cache_key=("rms_weight", program_key),
            tensors=[cast(Tensor, match.norm_modules[0].weight) for match in stage0_matches],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
            persistent_grouped_param_stacks=persistent_grouped_param_stacks,
            empty_like=stage0_matches[0].linears[1].weight,
        )
    else:
        rms_weight_stack = stage0_matches[0].linears[1].weight.new_empty((0,))
    stage0_params = _stack_program_stage_params(
        stage_matches=stage0_matches,
        program_key=program_key,
        stage_tag="0",
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    stage1_params = _stack_program_stage_params(
        stage_matches=stage1_matches,
        program_key=program_key,
        stage_tag="1",
        grouped_param_stacks=grouped_param_stacks,
        persistent_grouped_param_stacks=persistent_grouped_param_stacks,
        allow_persistent_stacks=allow_persistent_stacks,
    )
    if stage0_params is None or stage1_params is None:
        return None
    w10_stack, b10_stack, w20_stack, b20_stack = stage0_params
    w11_stack, b11_stack, w21_stack, b21_stack = stage1_params
    return relm_mp_ops.program_rmsnorm_silu(
        x,
        relation_args,
        slot_offsets_global,
        row_sizes,
        int(grouped_batch.arity),
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


__all__ = [
    "_get_grouped_param_stack",
    "_stack_optional_tensors",
    "_stack_block_two_layer_params",
    "_stack_program_stage_params",
    "_resolve_pointwise_code",
    "_run_two_layer_pointwise_kernel",
    "_run_postnorm_layernorm_kernel",
    "_run_prenorm_rmsnorm_kernel",
    "_run_silu_pair_program_kernel",
    "_run_silu_then_postnorm_program_kernel",
    "_run_prenorm_rmsnorm_then_silu_program_kernel",
]
