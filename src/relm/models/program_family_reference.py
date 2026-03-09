from __future__ import annotations

import torch


def _gather_padded_group_rows(
    packed_rows: torch.Tensor,
    row_sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_sizes_long = row_sizes.to(device=packed_rows.device, dtype=torch.long)
    group_count = int(row_sizes_long.numel())
    if group_count == 0:
        return packed_rows.new_empty((0, 0, int(packed_rows.size(1)))), row_sizes_long.new_empty((0, 0))
    max_rows = int(row_sizes_long.max().item())
    if max_rows <= 0:
        return (
            packed_rows.new_empty((group_count, 0, int(packed_rows.size(1)))),
            row_sizes_long.new_empty((group_count, 0)),
        )
    row_offsets = torch.empty_like(row_sizes_long)
    row_offsets[0] = 0
    if group_count > 1:
        row_offsets[1:] = torch.cumsum(row_sizes_long[:-1], dim=0)
    base = torch.arange(max_rows, device=packed_rows.device, dtype=torch.long).unsqueeze(0)
    safe_sizes = row_sizes_long.clamp_min(1).unsqueeze(1)
    safe_idx = row_offsets.unsqueeze(1) + torch.minimum(base, safe_sizes - 1)
    rows = packed_rows.index_select(0, safe_idx.reshape(-1)).view(
        group_count,
        max_rows,
        int(packed_rows.size(1)),
    )
    return rows, safe_idx


def _scatter_padded_group_rows(
    padded_rows: torch.Tensor,
    row_sizes: torch.Tensor,
    safe_idx: torch.Tensor,
    *,
    out_rows: int,
) -> torch.Tensor:
    out = padded_rows.new_zeros((int(out_rows), int(padded_rows.size(-1))))
    row_sizes_long = row_sizes.to(device=padded_rows.device, dtype=torch.long)
    if int(row_sizes_long.numel()) == 0 or int(padded_rows.numel()) == 0:
        return out
    max_rows = int(padded_rows.size(1))
    base = torch.arange(max_rows, device=padded_rows.device, dtype=torch.long).unsqueeze(0)
    mask = base < row_sizes_long.unsqueeze(1)
    flat_values = (padded_rows * mask.unsqueeze(-1).to(dtype=padded_rows.dtype)).reshape(
        -1,
        int(padded_rows.size(-1)),
    )
    out.index_add_(0, safe_idx.reshape(-1), flat_values)
    return out


def execute_program_two_layer_silu_then_two_layer_silu_reference(
    packed_rows: torch.Tensor,
    row_sizes: torch.Tensor,
    w10_stack: torch.Tensor,
    b10_stack: torch.Tensor,
    w20_stack: torch.Tensor,
    b20_stack: torch.Tensor,
    w11_stack: torch.Tensor,
    b11_stack: torch.Tensor,
    w21_stack: torch.Tensor,
    b21_stack: torch.Tensor,
) -> torch.Tensor:
    x_rows, safe_idx = _gather_padded_group_rows(packed_rows, row_sizes)
    if int(x_rows.numel()) == 0:
        return torch.empty_like(packed_rows)

    row_sizes_long = row_sizes.to(device=packed_rows.device, dtype=torch.long)
    base = torch.arange(int(x_rows.size(1)), device=packed_rows.device, dtype=torch.long).unsqueeze(0)
    mask = base < row_sizes_long.unsqueeze(1)
    mask_f = mask.unsqueeze(-1).to(dtype=packed_rows.dtype)

    x_rows = x_rows * mask_f
    pre1 = (torch.bmm(x_rows, w10_stack.transpose(1, 2)) + b10_stack.unsqueeze(1)) * mask_f
    stage1 = (torch.bmm(torch.nn.functional.silu(pre1), w20_stack.transpose(1, 2)) + b20_stack.unsqueeze(1)) * mask_f
    pre2 = (torch.bmm(stage1, w11_stack.transpose(1, 2)) + b11_stack.unsqueeze(1)) * mask_f
    stage2 = (torch.bmm(torch.nn.functional.silu(pre2), w21_stack.transpose(1, 2)) + b21_stack.unsqueeze(1)) * mask_f
    out_rows = x_rows + stage2
    return _scatter_padded_group_rows(out_rows, row_sizes_long, safe_idx, out_rows=int(packed_rows.size(0)))


def execute_program_two_layer_silu_then_postnorm_two_layer_silu_reference(
    packed_rows: torch.Tensor,
    row_sizes: torch.Tensor,
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
) -> torch.Tensor:
    x_rows, safe_idx = _gather_padded_group_rows(packed_rows, row_sizes)
    if int(x_rows.numel()) == 0:
        return torch.empty_like(packed_rows)

    row_sizes_long = row_sizes.to(device=packed_rows.device, dtype=torch.long)
    base = torch.arange(int(x_rows.size(1)), device=packed_rows.device, dtype=torch.long).unsqueeze(0)
    mask = base < row_sizes_long.unsqueeze(1)
    mask_f = mask.unsqueeze(-1).to(dtype=packed_rows.dtype)

    x_rows = x_rows * mask_f
    pre1 = (torch.bmm(x_rows, w10_stack.transpose(1, 2)) + b10_stack.unsqueeze(1)) * mask_f
    stage1 = (torch.bmm(torch.nn.functional.silu(pre1), w20_stack.transpose(1, 2)) + b20_stack.unsqueeze(1)) * mask_f
    pre2 = (torch.bmm(stage1, w11_stack.transpose(1, 2)) + b11_stack.unsqueeze(1)) * mask_f
    stage2 = (torch.bmm(torch.nn.functional.silu(pre2), w21_stack.transpose(1, 2)) + b21_stack.unsqueeze(1)) * mask_f
    norm_chunks: list[torch.Tensor] = []
    for gid in range(int(stage2.size(0))):
        norm_chunks.append(
            torch.nn.functional.layer_norm(
                stage2[gid],
                (int(stage2.size(-1)),),
                weight=(ln_weight_stack[gid] if int(ln_weight_stack.numel()) > 0 else None),
                bias=(ln_bias_stack[gid] if int(ln_bias_stack.numel()) > 0 else None),
                eps=float(ln_eps),
            )
        )
    norm_rows = (torch.stack(norm_chunks, dim=0) if norm_chunks else stage2.new_empty(stage2.shape)) * mask_f
    out_rows = x_rows + norm_rows
    return _scatter_padded_group_rows(out_rows, row_sizes_long, safe_idx, out_rows=int(packed_rows.size(0)))


__all__ = [
    "execute_program_two_layer_silu_then_two_layer_silu_reference",
    "execute_program_two_layer_silu_then_postnorm_two_layer_silu_reference",
]
