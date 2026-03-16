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
        lambda i, x_i, _in_dim: torch.nn.functional.linear(
            apply_pointwise_code(
                torch.nn.functional.linear(
                    x_i,
                    w1_stack[i],
                    b1_stack[i] if b1_stack.numel() > 0 else None,
                ),
                int(pointwise_code),
            ),
            w2_stack[i],
            b2_stack[i] if b2_stack.numel() > 0 else None,
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

    packed_rows_parts: list[torch.Tensor] = []
    node_parts: list[torch.Tensor] = []
    for group in iter_relation_group_inputs(x, layout):
        packed_rows_parts.append(group.x_rows)
        node_parts.append(group.rel_idx)

    if not packed_rows_parts:
        return build_relation_outputs(x, layout.emb, [], [])

    packed_rows = cat_or_single(packed_rows_parts, dim=0)
    row_sizes_tensor = torch.as_tensor(layout.row_sizes, device=x.device, dtype=torch.long)
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
    return rel_cat, cat_or_single(node_parts, dim=0)


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
        return torch.nn.functional.layer_norm(
            stage2,
            (in_dim,),
            weight=(ln_weight_stack[i] if ln_weight_stack.numel() > 0 else None),
            bias=(ln_bias_stack[i] if ln_bias_stack.numel() > 0 else None),
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
        stage1 = torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(norm_i, w10_stack[i], b10_stack[i])),
            w20_stack[i],
            b20_stack[i],
        )
        return torch.nn.functional.linear(
            torch.nn.functional.silu(torch.nn.functional.linear(stage1, w11_stack[i], b11_stack[i])),
            w21_stack[i],
            b21_stack[i],
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
        lambda i, x_i, in_dim: torch.nn.functional.layer_norm(
            torch.nn.functional.linear(
                apply_pointwise_code(
                    torch.nn.functional.linear(
                        x_i,
                        w1_stack[i],
                        b1_stack[i] if b1_stack.numel() > 0 else None,
                    ),
                    int(pointwise_code),
                ),
                w2_stack[i],
                b2_stack[i] if b2_stack.numel() > 0 else None,
            ),
            (in_dim,),
            weight=(ln_weight_stack[i] if ln_weight_stack.numel() > 0 else None),
            bias=(ln_bias_stack[i] if ln_bias_stack.numel() > 0 else None),
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
        lambda i, x_i, in_dim: torch.nn.functional.linear(
            apply_pointwise_code(
                torch.nn.functional.linear(
                    torch.nn.functional.rms_norm(
                        x_i,
                        (in_dim,),
                        weight=(rms_weight_stack[i] if rms_weight_stack.numel() > 0 else None),
                        eps=float(rms_eps),
                    ),
                    w1_stack[i],
                    b1_stack[i] if b1_stack.numel() > 0 else None,
                ),
                int(pointwise_code),
            ),
            w2_stack[i],
            b2_stack[i] if b2_stack.numel() > 0 else None,
        ),
    )
