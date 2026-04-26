"""Pure Python fallback implementations for grouped block/program kernels."""

from __future__ import annotations

from .mp_common import (
    build_relation_outputs,
    cat_or_single,
    iter_relation_group_inputs,
    prepare_relation_kernel_layout,
    validate_optional_group_matrix,
    validate_two_layer_stacks,
    validate_two_stage_program_stacks,
)
from .mp_constants import (
    PW_GELU_NONE,
    PW_GELU_TANH,
    PW_IDENTITY,
    PW_MISH,
    PW_RELU,
    PW_SILU,
    PW_TANH,
)
from .mp_runtime import torch


def _stack_bias(bias_stack: torch.Tensor, group_index: int):
    return bias_stack[group_index] if bias_stack.numel() > 0 else None


def _stack_linear(
    x: torch.Tensor,
    weight_stack: torch.Tensor,
    bias_stack: torch.Tensor,
    group_index: int,
) -> torch.Tensor:
    return torch.nn.functional.linear(
        x,
        weight_stack[group_index],
        _stack_bias(bias_stack, group_index),
    )


def _apply_optional_layer_norm(
    x: torch.Tensor,
    *,
    normalized_shape: tuple[int, ...],
    weight_stack: torch.Tensor,
    bias_stack: torch.Tensor,
    group_index: int,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape,
        weight=(weight_stack[group_index] if weight_stack.numel() > 0 else None),
        bias=(bias_stack[group_index] if bias_stack.numel() > 0 else None),
        eps=float(eps),
    )


def _apply_optional_rms_norm(
    x: torch.Tensor,
    *,
    normalized_shape: tuple[int, ...],
    weight_stack: torch.Tensor,
    group_index: int,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(
        x,
        normalized_shape,
        weight=(weight_stack[group_index] if weight_stack.numel() > 0 else None),
        eps=float(eps),
    )


def _two_layer_pointwise_update(
    x_i: torch.Tensor,
    *,
    group_index: int,
    w1_stack: torch.Tensor,
    b1_stack: torch.Tensor,
    w2_stack: torch.Tensor,
    b2_stack: torch.Tensor,
    pointwise_code: int,
) -> torch.Tensor:
    hidden = apply_pointwise_code(
        _stack_linear(x_i, w1_stack, b1_stack, group_index),
        int(pointwise_code),
    )
    return _stack_linear(hidden, w2_stack, b2_stack, group_index)


def _two_stage_silu_update(
    x_i: torch.Tensor,
    *,
    group_index: int,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
) -> torch.Tensor:
    stage1 = _stack_linear(
        torch.nn.functional.silu(_stack_linear(x_i, w10_stack, b10_stack, group_index)),
        w20_stack,
        b20_stack,
        group_index,
    )
    return _stack_linear(
        torch.nn.functional.silu(_stack_linear(stage1, w11_stack, b11_stack, group_index)),
        w21_stack,
        b21_stack,
        group_index,
    )


def apply_pointwise_code(x: torch.Tensor, code: int) -> torch.Tensor:
    code_i = int(code)
    if code_i == PW_IDENTITY:
        return x
    if code_i == PW_RELU:
        return torch.relu(x)
    if code_i == PW_MISH:
        return torch.nn.functional.mish(x)
    if code_i == PW_GELU_NONE:
        return torch.nn.functional.gelu(x, approximate="none")
    if code_i == PW_GELU_TANH:
        return torch.nn.functional.gelu(x, approximate="tanh")
    if code_i == PW_SILU:
        return torch.nn.functional.silu(x)
    if code_i == PW_TANH:
        return torch.tanh(x)
    raise ValueError(f"Unsupported pointwise code: {code_i!r}.")


def pool_block_messages_to_rows(
    rel_cat: torch.Tensor,
    row_count: int,
    arity: int,
) -> torch.Tensor:
    if int(rel_cat.numel()) == 0 or int(row_count) <= 0:
        return rel_cat.new_zeros((0, int(rel_cat.size(-1))))
    return rel_cat.view(int(row_count), int(arity), int(rel_cat.size(-1))).mean(dim=1)


def _run_grouped_residual_kernel(
    op_name: str,
    x: torch.Tensor,
    relation_args: torch.Tensor,
    slot_offsets: list[int],
    row_sizes: list[int],
    arity: int,
    kernel_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )

    rel_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for group in iter_relation_group_inputs(x, layout):
        out_i = kernel_fn(group.group_index, group.x_rows, layout.in_dim)
        if out_i.dim() != 2 or tuple(out_i.shape) != tuple(group.x_rows.shape):
            raise ValueError(
                f"{op_name} internal kernel must return shape {tuple(group.x_rows.shape)}, "
                f"got {tuple(int(v) for v in out_i.shape)}."
            )
        rel_parts.append((group.x_rows + out_i).view(group.span, layout.emb))
        node_parts.append(group.rel_idx)
    return build_relation_outputs(x, layout.emb, rel_parts, node_parts)


def _pack_group_rows(
    x: torch.Tensor,
    layout,
) -> tuple[torch.Tensor, torch.Tensor]:
    packed_rows_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for group in iter_relation_group_inputs(x, layout):
        packed_rows_parts.append(group.x_rows)
        node_parts.append(group.rel_idx)
    if not packed_rows_parts:
        return x.new_empty((0, layout.in_dim)), x.new_empty((0,), dtype=torch.int64)
    return cat_or_single(packed_rows_parts, dim=0), cat_or_single(node_parts, dim=0)


def _compute_group_padded_index(row_sizes: tuple[int, ...], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    row_sizes_long = torch.as_tensor(row_sizes, device=device, dtype=torch.long)
    max_rows = int(row_sizes_long.max().item()) if int(row_sizes_long.numel()) > 0 else 0
    row_offsets = torch.empty_like(row_sizes_long)
    if int(row_sizes_long.numel()) > 0:
        row_offsets[0] = 0
    if int(row_sizes_long.numel()) > 1:
        row_offsets[1:] = torch.cumsum(row_sizes_long[:-1], dim=0)
    base = torch.arange(max_rows, device=device, dtype=torch.long).unsqueeze(0)
    safe_sizes = row_sizes_long.clamp_min(1).unsqueeze(1)
    safe_idx = row_offsets.unsqueeze(1) + torch.minimum(base, safe_sizes - 1)
    mask = base < row_sizes_long.unsqueeze(1)
    return safe_idx, mask


def block_pointwise_python(
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
    op_name = "block_pointwise"
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )
    validate_two_layer_stacks(
        op_name,
        groups=layout.groups,
        in_dim=layout.in_dim,
        w1_stack=w1_stack,
        b1_stack=b1_stack,
        w2_stack=w2_stack,
        b2_stack=b2_stack,
    )

    return _run_grouped_residual_kernel(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        lambda i, x_i, _in_dim: _two_layer_pointwise_update(
            x_i,
            group_index=i,
            w1_stack=w1_stack,
            b1_stack=b1_stack,
            w2_stack=w2_stack,
            b2_stack=b2_stack,
            pointwise_code=int(pointwise_code),
        ),
    )


def block_pointwise_pool_python(
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
) -> torch.Tensor:
    rel_cat, _ = block_pointwise_python(
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        pointwise_code,
    )
    row_count = int(sum(int(v) for v in row_sizes))
    return pool_block_messages_to_rows(rel_cat, row_count=row_count, arity=int(arity))


def program_silu_pair_python(
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
    op_name = "program_silu_pair"
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )
    validate_two_stage_program_stacks(
        op_name,
        groups=layout.groups,
        in_dim=layout.in_dim,
        w10_stack=w10_stack,
        b10_stack=b10_stack,
        w20_stack=w20_stack,
        b20_stack=b20_stack,
        w11_stack=w11_stack,
        b11_stack=b11_stack,
        w21_stack=w21_stack,
        b21_stack=b21_stack,
    )

    packed_rows, node_idx = _pack_group_rows(x, layout)
    if int(packed_rows.numel()) == 0:
        return build_relation_outputs(x, layout.emb, [], [])

    safe_idx, mask = _compute_group_padded_index(layout.row_sizes, x.device)
    max_rows = int(mask.size(1)) if mask.dim() == 2 else 0

    x_rows = packed_rows.index_select(0, safe_idx.reshape(-1)).view(
        layout.groups,
        max_rows,
        layout.in_dim,
    )
    mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
    x_rows = x_rows * mask_f

    pre1 = (torch.bmm(x_rows, w10_stack.transpose(1, 2)) + b10_stack.unsqueeze(1)) * mask_f
    stage1 = (
        torch.bmm(torch.nn.functional.silu(pre1), w20_stack.transpose(1, 2))
        + b20_stack.unsqueeze(1)
    ) * mask_f
    pre2 = (torch.bmm(stage1, w11_stack.transpose(1, 2)) + b11_stack.unsqueeze(1)) * mask_f
    stage2 = (
        torch.bmm(torch.nn.functional.silu(pre2), w21_stack.transpose(1, 2))
        + b21_stack.unsqueeze(1)
    ) * mask_f
    out_rows = x_rows + stage2

    packed_out = packed_rows.new_zeros((int(packed_rows.size(0)), layout.in_dim))
    packed_out.index_add_(0, safe_idx.reshape(-1), (out_rows * mask_f).reshape(-1, layout.in_dim))
    rel_cat = packed_out.view(-1, layout.emb)
    return rel_cat, node_idx


def program_silu_postnorm_python(
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
    op_name = "program_silu_postnorm"
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )
    validate_two_stage_program_stacks(
        op_name,
        groups=layout.groups,
        in_dim=layout.in_dim,
        w10_stack=w10_stack,
        b10_stack=b10_stack,
        w20_stack=w20_stack,
        b20_stack=b20_stack,
        w11_stack=w11_stack,
        b11_stack=b11_stack,
        w21_stack=w21_stack,
        b21_stack=b21_stack,
    )
    validate_optional_group_matrix(
        op_name,
        "ln_weight_stack",
        ln_weight_stack,
        groups=layout.groups,
        width=layout.in_dim,
    )
    validate_optional_group_matrix(
        op_name,
        "ln_bias_stack",
        ln_bias_stack,
        groups=layout.groups,
        width=layout.in_dim,
    )

    def _kernel(i: int, x_i: torch.Tensor, in_dim: int) -> torch.Tensor:
        stage2 = _two_stage_silu_update(
            x_i,
            group_index=i,
            w10_stack=w10_stack,
            b10_stack=b10_stack,
            w20_stack=w20_stack,
            b20_stack=b20_stack,
            w11_stack=w11_stack,
            b11_stack=b11_stack,
            w21_stack=w21_stack,
            b21_stack=b21_stack,
        )
        return _apply_optional_layer_norm(
            stage2,
            normalized_shape=(in_dim,),
            weight_stack=ln_weight_stack,
            bias_stack=ln_bias_stack,
            group_index=i,
            eps=float(ln_eps),
        )

    return _run_grouped_residual_kernel(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        _kernel,
    )


def program_rmsnorm_silu_python(
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
    op_name = "program_rmsnorm_silu"
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )
    validate_two_stage_program_stacks(
        op_name,
        groups=layout.groups,
        in_dim=layout.in_dim,
        w10_stack=w10_stack,
        b10_stack=b10_stack,
        w20_stack=w20_stack,
        b20_stack=b20_stack,
        w11_stack=w11_stack,
        b11_stack=b11_stack,
        w21_stack=w21_stack,
        b21_stack=b21_stack,
    )
    validate_optional_group_matrix(
        op_name,
        "rms_weight_stack",
        rms_weight_stack,
        groups=layout.groups,
        width=layout.in_dim,
    )

    def _kernel(i: int, x_i: torch.Tensor, _in_dim: int) -> torch.Tensor:
        sq_mean = x_i.square().mean(dim=-1, keepdim=True)
        norm_i = x_i * torch.rsqrt(sq_mean + float(rms_eps))
        if rms_weight_stack.numel() > 0:
            norm_i = norm_i * rms_weight_stack[i].unsqueeze(0)
        return _two_stage_silu_update(
            norm_i,
            group_index=i,
            w10_stack=w10_stack,
            b10_stack=b10_stack,
            w20_stack=w20_stack,
            b20_stack=b20_stack,
            w11_stack=w11_stack,
            b11_stack=b11_stack,
            w21_stack=w21_stack,
            b21_stack=b21_stack,
        )

    return _run_grouped_residual_kernel(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        _kernel,
    )


def block_postnorm_ln_python(
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
    op_name = "block_postnorm_ln"
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )
    validate_two_layer_stacks(
        op_name,
        groups=layout.groups,
        in_dim=layout.in_dim,
        w1_stack=w1_stack,
        b1_stack=b1_stack,
        w2_stack=w2_stack,
        b2_stack=b2_stack,
    )
    validate_optional_group_matrix(
        op_name,
        "ln_weight_stack",
        ln_weight_stack,
        groups=layout.groups,
        width=layout.in_dim,
    )
    validate_optional_group_matrix(
        op_name,
        "ln_bias_stack",
        ln_bias_stack,
        groups=layout.groups,
        width=layout.in_dim,
    )

    return _run_grouped_residual_kernel(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        lambda i, x_i, in_dim: _apply_optional_layer_norm(
            _two_layer_pointwise_update(
                x_i,
                group_index=i,
                w1_stack=w1_stack,
                b1_stack=b1_stack,
                w2_stack=w2_stack,
                b2_stack=b2_stack,
                pointwise_code=int(pointwise_code),
            ),
            normalized_shape=(in_dim,),
            weight_stack=ln_weight_stack,
            bias_stack=ln_bias_stack,
            group_index=i,
            eps=float(ln_eps),
        ),
    )


def block_prenorm_rms_python(
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
    op_name = "block_prenorm_rms"
    layout = prepare_relation_kernel_layout(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
    )
    validate_optional_group_matrix(
        op_name,
        "rms_weight_stack",
        rms_weight_stack,
        groups=layout.groups,
        width=layout.in_dim,
    )
    validate_two_layer_stacks(
        op_name,
        groups=layout.groups,
        in_dim=layout.in_dim,
        w1_stack=w1_stack,
        b1_stack=b1_stack,
        w2_stack=w2_stack,
        b2_stack=b2_stack,
    )

    return _run_grouped_residual_kernel(
        op_name,
        x,
        relation_args,
        slot_offsets,
        row_sizes,
        arity,
        lambda i, x_i, in_dim: _two_layer_pointwise_update(
            _apply_optional_rms_norm(
                x_i,
                normalized_shape=(in_dim,),
                weight_stack=rms_weight_stack,
                group_index=i,
                eps=float(rms_eps),
            ),
            group_index=i,
            w1_stack=w1_stack,
            b1_stack=b1_stack,
            w2_stack=w2_stack,
            b2_stack=b2_stack,
            pointwise_code=int(pointwise_code),
        ),
    )
