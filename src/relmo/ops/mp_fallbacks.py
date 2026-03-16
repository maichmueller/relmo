"""Pure Python fallback kernels used when custom mp ops are unavailable."""

from __future__ import annotations

from .mp_block_kernels import (
    apply_pointwise_code,
    block_pointwise_pool_python,
    block_pointwise_python,
    block_postnorm_ln_python,
    block_prenorm_rms_python,
    pool_block_messages_to_rows,
    program_rmsnorm_silu_python,
    program_silu_pair_python,
    program_silu_postnorm_python,
)
from .mp_common import indexed_sum_or_mean, require_rank
from .mp_pack import (
    fanin_pack_from_edges_python,
    fanin_pack_multi_python,
    fanin_reduce_python,
    fanout_pack_from_edges_python,
    fanout_pack_multi_python,
    fanout_scatter_python,
)
from .mp_runtime import torch


def lgan_pool_reduce_python(
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
    op_name = "lgan_pool_reduce"
    require_rank(op_name, "slot_messages", slot_messages, 2)

    relation_count = int(relation_instance_arities.numel())
    relation_pair_x = slot_messages.new_zeros(
        (relation_count, int(slot_messages.size(-1)))
    )
    if int(slot_messages.numel()) > 0:
        relation_pair_x.index_add_(0, slot_to_relation_instance, slot_messages)
        counts = relation_instance_arities.to(
            device=slot_messages.device,
            dtype=slot_messages.dtype,
        ).view(-1, 1).clamp_min_(1.0)
        relation_pair_x = relation_pair_x / counts

    rr_msgs = indexed_sum_or_mean(
        relation_pair_x,
        rr_src,
        rr_dst,
        relation_count,
        mode,
        op_name=op_name,
    )
    relation_pair_x = relation_pair_x + rr_msgs
    tn_msgs = indexed_sum_or_mean(
        relation_pair_x,
        tn_rel,
        tn_ent,
        int(entity_dim_size),
        mode,
        op_name=op_name,
    )
    nn_msgs = indexed_sum_or_mean(
        relation_pair_x,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        mode,
        op_name=op_name,
    )
    return relation_pair_x, tn_msgs, nn_msgs


def lgan_relation_graph_step_python(
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
    op_name = "lgan_relation_graph_step"
    require_rank(op_name, "relation_pair_x", relation_pair_x, 2)

    rr_msgs = indexed_sum_or_mean(
        relation_pair_x,
        rr_src,
        rr_dst,
        int(relation_pair_x.size(0)),
        mode,
        op_name=op_name,
    )
    relation_pair_x = relation_pair_x + rr_msgs
    tn_msgs = indexed_sum_or_mean(
        relation_pair_x,
        tn_rel,
        tn_ent,
        int(entity_dim_size),
        mode,
        op_name=op_name,
    )
    nn_msgs = indexed_sum_or_mean(
        relation_pair_x,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        mode,
        op_name=op_name,
    )
    return relation_pair_x, tn_msgs, nn_msgs


__all__ = [
    "apply_pointwise_code",
    "block_pointwise_pool_python",
    "block_pointwise_python",
    "block_postnorm_ln_python",
    "block_prenorm_rms_python",
    "fanin_pack_from_edges_python",
    "fanin_pack_multi_python",
    "fanin_reduce_python",
    "fanout_pack_from_edges_python",
    "fanout_pack_multi_python",
    "fanout_scatter_python",
    "lgan_pool_reduce_python",
    "lgan_relation_graph_step_python",
    "pool_block_messages_to_rows",
    "program_rmsnorm_silu_python",
    "program_silu_pair_python",
    "program_silu_postnorm_python",
]
