"""Public façade for relational message-passing operator wrappers.

Architecture:
- runtime/load policy lives in :mod:`relmo.ops.mp_runtime`
- custom-op dispatch policy lives in :mod:`relmo.ops.mp_dispatch`
- pure Python kernels live in :mod:`relmo.ops.mp_fallbacks`
- autograd wrappers live in :mod:`relmo.ops.mp_autograd`

This module keeps the historical public surface stable while delegating the
heavy lifting to smaller internal modules.
"""

from __future__ import annotations

from .mp_autograd import (
    _FaninReduceLogSumExpFunction,
    _FaninReduceSumFunction,
    _FanoutScatterFunction,
    _FusedPostNormTwoLayerPointwiseLayerNormFromIndicesFunction,
    _FusedPreNormTwoLayerPointwiseRMSNormFromIndicesFunction,
    _FusedProgramPreNormTwoLayerSiLURMSNormThenTwoLayerSiLUFromIndicesFunction,
    _FusedProgramTwoLayerSiLUThenPostNormTwoLayerSiLUFromIndicesFunction,
    _FusedProgramTwoLayerSiLUThenTwoLayerSiLUFromIndicesFunction,
    _FusedTwoLayerPointwiseFromIndicesFunction,
    _LGANPointwiseBuildStepFunction,
    _LGANPoolReduceFunction,
    _LGANRelationGraphStepFunction,
)
from .mp_constants import MODE_LOGSUMEXP, MODE_MEAN, MODE_SUM, activation_code
from .mp_dispatch import namespace_has_op, ops_namespace, require_available_custom_op, should_use_custom
from .mp_fallbacks import (
    block_pointwise_pool_python,
    fanin_pack_from_edges_python,
    fanin_pack_multi_python,
    fanin_reduce_python,
    fanout_pack_from_edges_python,
    fanout_pack_multi_python,
    fanout_scatter_python,
)
from .mp_runtime import TORCH_IMPORT_ERROR, assert_runtime_compat, available, require_torch, torch


def fanout_scatter(
    x_cat: torch.Tensor,
    src_global_idx: torch.Tensor,
    flat_dst: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    require_torch("fanout_scatter")
    if should_use_custom("fanout_scatter"):
        return _FanoutScatterFunction.apply(x_cat, src_global_idx, flat_dst, int(out_rows))
    return fanout_scatter_python(x_cat, src_global_idx, flat_dst, int(out_rows))


def fanin_reduce(
    rel_flat: torch.Tensor,
    flat_src: torch.Tensor,
    dst_idx: torch.Tensor,
    dim_size: int,
    mode: int,
) -> torch.Tensor:
    require_torch("fanin_reduce")
    mode_int = int(mode)
    if mode_int not in (MODE_SUM, MODE_LOGSUMEXP):
        return fanin_reduce_python(rel_flat, flat_src, dst_idx, int(dim_size), mode_int)
    if should_use_custom("fanin_reduce"):
        if mode_int == MODE_SUM:
            return _FaninReduceSumFunction.apply(rel_flat, flat_src, dst_idx, int(dim_size))
        return _FaninReduceLogSumExpFunction.apply(rel_flat, flat_src, dst_idx, int(dim_size))
    return fanin_reduce_python(rel_flat, flat_src, dst_idx, int(dim_size), mode_int)


def fanout_pack_multi(
    x_parts: list[torch.Tensor],
    src_idx_parts: list[torch.Tensor],
    flat_dst_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_torch("fanout_pack_multi")
    if should_use_custom("fanout_pack_multi"):
        if namespace_has_op("fanout_pack_multi"):
            return ops_namespace().fanout_pack_multi(x_parts, src_idx_parts, flat_dst_parts)
        require_available_custom_op("fanout_pack_multi")
    return fanout_pack_multi_python(x_parts, src_idx_parts, flat_dst_parts)


def fanin_pack_multi(
    rel_parts: list[torch.Tensor],
    flat_src_parts: list[torch.Tensor],
    dst_idx_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_torch("fanin_pack_multi")
    if should_use_custom("fanin_pack_multi"):
        if namespace_has_op("fanin_pack_multi"):
            return ops_namespace().fanin_pack_multi(rel_parts, flat_src_parts, dst_idx_parts)
        require_available_custom_op("fanin_pack_multi")
    return fanin_pack_multi_python(rel_parts, flat_src_parts, dst_idx_parts)


def fanout_pack_from_edges(
    x_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    src_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    slot_offset_parts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_torch("fanout_pack_from_edges")
    if should_use_custom("fanout_pack_from_edges"):
        if namespace_has_op("fanout_pack_from_edges"):
            return ops_namespace().fanout_pack_from_edges(
                x_parts,
                edge_src_parts,
                edge_dst_parts,
                src_part_ids,
                arity_parts,
                pos_parts,
                slot_offset_parts,
            )
        require_available_custom_op("fanout_pack_from_edges")
    return fanout_pack_from_edges_python(
        x_parts,
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    )


def fanin_pack_from_edges(
    rel_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    rel_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_torch("fanin_pack_from_edges")
    if should_use_custom("fanin_pack_from_edges"):
        if namespace_has_op("fanin_pack_from_edges"):
            return ops_namespace().fanin_pack_from_edges(
                rel_parts,
                edge_src_parts,
                edge_dst_parts,
                rel_part_ids,
                arity_parts,
                pos_parts,
                int(mode),
            )
        require_available_custom_op("fanin_pack_from_edges")
    return fanin_pack_from_edges_python(
        rel_parts,
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
        int(mode),
    )


def block_pointwise(
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
    """Run a width-preserving 2-layer pointwise block over packed relation rows."""

    require_torch("block_pointwise")
    return _FusedTwoLayerPointwiseFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def block_pointwise_pool(
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
    """Run a 2-layer pointwise block and return pooled relation-row outputs."""

    require_torch("block_pointwise_pool")
    use_custom = (
        x.is_cuda
        and should_use_custom("block_pointwise_pool")
        and namespace_has_op("block_pointwise_pool")
    )
    if use_custom:
        return ops_namespace().block_pointwise_pool(
            x,
            relation_args,
            list(slot_offsets),
            list(row_sizes),
            int(arity),
            w1_stack,
            b1_stack,
            w2_stack,
            b2_stack,
            int(pointwise_code),
        )
    if should_use_custom("block_pointwise_pool"):
        require_available_custom_op("block_pointwise_pool")
    return block_pointwise_pool_python(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def block_postnorm_ln(
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
    """Run ``Linear -> activation -> Linear -> LayerNorm`` on packed relation rows."""

    require_torch("block_postnorm_ln")
    return _FusedPostNormTwoLayerPointwiseLayerNormFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        ln_weight_stack,
        ln_bias_stack,
        float(ln_eps),
        int(pointwise_code),
    )


def block_prenorm_rms(
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
    """Run ``RMSNorm -> Linear -> activation -> Linear`` on packed relation rows."""

    require_torch("block_prenorm_rms")
    return _FusedPreNormTwoLayerPointwiseRMSNormFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        rms_weight_stack,
        float(rms_eps),
        w1_stack,
        b1_stack,
        w2_stack,
        b2_stack,
        int(pointwise_code),
    )


def program_silu_pair(
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
    """Run the exact 2-stage SiLU relation program on packed relation rows."""

    require_torch("program_silu_pair")
    return _FusedProgramTwoLayerSiLUThenTwoLayerSiLUFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
        w10_stack,
        b10_stack,
        w20_stack,
        b20_stack,
        w11_stack,
        b11_stack,
        w21_stack,
        b21_stack,
    )


def program_silu_postnorm(
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
    """Run the exact SiLU then post-norm SiLU relation program."""

    require_torch("program_silu_postnorm")
    return _FusedProgramTwoLayerSiLUThenPostNormTwoLayerSiLUFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
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


def program_rmsnorm_silu(
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
    """Run the exact pre-norm RMSNorm then SiLU relation program."""

    require_torch("program_rmsnorm_silu")
    return _FusedProgramPreNormTwoLayerSiLURMSNormThenTwoLayerSiLUFromIndicesFunction.apply(
        x,
        relation_args,
        list(slot_offsets),
        list(row_sizes),
        int(arity),
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


def _resolve_lgan_mode(op_name: str, mode: str) -> int:
    mode_key = str(mode).strip().lower()
    if mode_key == "sum":
        return MODE_SUM
    if mode_key == "mean":
        return MODE_MEAN
    raise ValueError(f"{op_name} supports only mode='sum' or mode='mean'.")


def _lgan_build_pointwise_step(
    x: torch.Tensor,
    relation_args: torch.Tensor,
    seed_relation_pair_x: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    *,
    entity_dim_size: int,
    mode: str,
    arities: tuple[int, ...] | list[int],
    pointwise_codes: tuple[int, ...] | list[int],
    slot_offsets_groups: tuple[tuple[int, ...], ...] | list[list[int]],
    row_sizes_groups: tuple[tuple[int, ...], ...] | list[list[int]],
    row_starts_groups: tuple[tuple[int, ...], ...] | list[list[int]],
    w1_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    b1_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    w2_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
    b2_stacks: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build exact pointwise relation-instance rows and run the LGAN graph step."""

    require_torch("_lgan_build_pointwise_step")
    reduce_mode = _resolve_lgan_mode("_lgan_build_pointwise_step", mode)
    group_count = len(tuple(arities))
    if not (
        len(tuple(pointwise_codes))
        == len(tuple(slot_offsets_groups))
        == len(tuple(row_sizes_groups))
        == len(tuple(row_starts_groups))
        == len(tuple(w1_stacks))
        == len(tuple(b1_stacks))
        == len(tuple(w2_stacks))
        == len(tuple(b2_stacks))
        == group_count
    ):
        raise ValueError("_lgan_build_pointwise_step expects one metadata/parameter entry per group.")
    return _LGANPointwiseBuildStepFunction.apply(
        x,
        relation_args,
        seed_relation_pair_x,
        rr_src,
        rr_dst,
        tn_rel,
        tn_ent,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        int(reduce_mode),
        tuple(int(v) for v in arities),
        tuple(int(v) for v in pointwise_codes),
        tuple(tuple(int(x) for x in values) for values in slot_offsets_groups),
        tuple(tuple(int(x) for x in values) for values in row_sizes_groups),
        tuple(tuple(int(x) for x in values) for values in row_starts_groups),
        int(group_count),
        *tuple(w1_stacks),
        *tuple(b1_stacks),
        *tuple(w2_stacks),
        *tuple(b2_stacks),
    )


def _lgan_pool_reduce(
    slot_messages: torch.Tensor,
    slot_to_relation_instance: torch.Tensor,
    relation_instance_arities: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    *,
    entity_dim_size: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool slot messages to relation instances and run RR/TN/NN indexed reductions."""

    require_torch("_lgan_pool_reduce")
    return _LGANPoolReduceFunction.apply(
        slot_messages,
        slot_to_relation_instance,
        relation_instance_arities,
        rr_src,
        rr_dst,
        tn_rel,
        tn_ent,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        int(_resolve_lgan_mode("_lgan_pool_reduce", mode)),
    )


def _lgan_relation_graph_step(
    relation_pair_x: torch.Tensor,
    rr_src: torch.Tensor,
    rr_dst: torch.Tensor,
    tn_rel: torch.Tensor,
    tn_ent: torch.Tensor,
    nn_rel: torch.Tensor,
    nn_ent: torch.Tensor,
    *,
    entity_dim_size: int,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run RR/TN/NN propagation on pooled relation-instance embeddings."""

    require_torch("_lgan_relation_graph_step")
    return _LGANRelationGraphStepFunction.apply(
        relation_pair_x,
        rr_src,
        rr_dst,
        tn_rel,
        tn_ent,
        nn_rel,
        nn_ent,
        int(entity_dim_size),
        int(_resolve_lgan_mode("_lgan_relation_graph_step", mode)),
    )


__all__ = [
    "fanout_scatter",
    "fanin_reduce",
    "fanout_pack_multi",
    "fanin_pack_multi",
    "fanout_pack_from_edges",
    "fanin_pack_from_edges",
    "block_pointwise",
    "block_postnorm_ln",
    "block_prenorm_rms",
    "program_silu_pair",
    "program_silu_postnorm",
    "program_rmsnorm_silu",
    "activation_code",
    "available",
    "assert_runtime_compat",
]
