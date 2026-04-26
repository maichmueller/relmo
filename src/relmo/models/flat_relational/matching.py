"""Explicit kernel-family matching and structural fallback helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import torch

from ..relation_blocks import RelationBlockSpec, RelationProgram
from .kernels import (
    FlatRelationKernel,
    GELUBlockKernel,
    MishBlockKernel,
    PostNormMishLayerNormKernel,
    PostNormSiLULayerNormKernel,
    PreNormRMSNormThenSiLUProgramKernel,
    PreNormSiLURMSNormKernel,
    SiLUBlockKernel,
    SiLUPairProgramKernel,
    SiLUThenPostNormProgramKernel,
)
from .types import BlockKernelSpec, KernelMatch, ProgramKernelSpec, RelationSlice


@dataclass(frozen=True)
class ExtractedLinearOp:
    linear_index: int
    module: torch.nn.Linear
    signature: tuple[int, int, bool]


@dataclass(frozen=True)
class ExtractedPointwiseOp:
    module: torch.nn.Module
    signature: tuple[Any, ...]


@dataclass(frozen=True)
class ExtractedNormOp:
    module: torch.nn.Module
    signature: tuple[Any, ...]


ExtractedBlockOp = ExtractedLinearOp | ExtractedPointwiseOp | ExtractedNormOp


@dataclass(frozen=True)
class ExtractedBlockSpec:
    signature: Hashable
    linears: tuple[torch.nn.Linear, ...]
    ops: tuple[ExtractedBlockOp, ...]


_KERNEL_SPEC_METHODS = ("kernel_spec",)
_SUPPORTED_POINTWISE_TYPES = (
    torch.nn.Identity,
    torch.nn.ReLU,
    torch.nn.Mish,
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.Tanh,
    torch.nn.ELU,
    torch.nn.LeakyReLU,
)
_LAYER_NORM_TYPE = torch.nn.LayerNorm
_RMS_NORM_TYPE = getattr(torch.nn, "RMSNorm", None)
_SUPPORTED_NORM_TYPES = tuple(
    norm_type for norm_type in (_LAYER_NORM_TYPE, _RMS_NORM_TYPE) if norm_type is not None
)


def pointwise_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
    if not isinstance(module, _SUPPORTED_POINTWISE_TYPES):
        return None
    if isinstance(module, torch.nn.GELU):
        return ("gelu", str(module.approximate))
    if isinstance(module, torch.nn.LeakyReLU):
        return (
            "leaky_relu",
            float(module.negative_slope),
            bool(module.inplace),
        )
    if isinstance(module, torch.nn.ELU):
        return (
            "elu",
            float(module.alpha),
            float(module.scale),
            float(module.input_scale),
            bool(module.inplace),
        )
    if isinstance(module, torch.nn.ReLU):
        return ("relu", bool(module.inplace))
    if isinstance(module, torch.nn.Identity):
        return ("identity",)
    if isinstance(module, torch.nn.Mish):
        return ("mish", bool(module.inplace))
    if isinstance(module, torch.nn.SiLU):
        return ("silu", bool(module.inplace))
    if isinstance(module, torch.nn.Tanh):
        return ("tanh",)
    return None


def norm_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
    normalized_shape = getattr(module, "normalized_shape", ())
    if isinstance(normalized_shape, int):
        shape = (int(normalized_shape),)
    else:
        shape = tuple(int(v) for v in normalized_shape)
    eps = getattr(module, "eps", None)
    eps_value = None if eps is None else float(eps)
    if isinstance(module, torch.nn.LayerNorm):
        return ("layernorm", shape, eps_value, bool(module.elementwise_affine))
    if _RMS_NORM_TYPE is not None and isinstance(module, _RMS_NORM_TYPE):
        return ("rmsnorm", shape, eps_value, bool(module.elementwise_affine))
    return None


def _coerce_relation_block_spec(
    spec: RelationBlockSpec | dict[str, Any],
) -> tuple[Sequence[torch.nn.Linear], Sequence[tuple[str, Any]], Hashable | None]:
    if isinstance(spec, RelationBlockSpec):
        return spec.linears, spec.ops, spec.signature
    if not isinstance(spec, dict):
        raise TypeError(
            f"kernel_spec() must return RelationBlockSpec|dict|None, got {type(spec)!r}."
        )
    linears = spec.get("linears", ())
    ops = spec.get("ops", ())
    signature = spec.get("signature", None)
    return linears, ops, signature


def extract_block(module: torch.nn.Module) -> ExtractedBlockSpec | None:
    for method_name in _KERNEL_SPEC_METHODS:
        method = getattr(module, method_name, None)
        if not callable(method):
            continue
        spec = method()
        if spec is None:
            return None
        linears_raw, ops_raw, signature = _coerce_relation_block_spec(spec)
        linears = tuple(linears_raw)
        ops_list: list[ExtractedBlockOp] = []
        if not linears or not ops_raw:
            return None
        for idx, op in enumerate(ops_raw):
            if not (isinstance(op, tuple) and len(op) == 2):
                raise TypeError(
                    f"{type(module).__name__}.{method_name}() ops[{idx}] must be tuple(kind, payload)."
                )
            kind, payload = op
            if kind == "linear":
                linear_index = int(payload)
                if linear_index < 0 or linear_index >= len(linears):
                    raise IndexError(
                        f"{type(module).__name__}.{method_name}() linear index {linear_index} is out of range."
                    )
                linear = linears[linear_index]
                if not isinstance(linear, torch.nn.Linear):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() linears[{linear_index}] must be torch.nn.Linear."
                    )
                ops_list.append(
                    ExtractedLinearOp(
                        linear_index=linear_index,
                        module=linear,
                        signature=(
                            int(linear.in_features),
                            int(linear.out_features),
                            bool(linear.bias is not None),
                        ),
                    )
                )
                continue
            if kind == "pointwise":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] pointwise payload must be torch.nn.Module."
                    )
                sig = pointwise_signature(payload)
                if sig is None:
                    return None
                ops_list.append(ExtractedPointwiseOp(module=payload, signature=sig))
                continue
            if kind == "norm":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] norm payload must be torch.nn.Module."
                    )
                sig = norm_signature(payload)
                if sig is None:
                    return None
                ops_list.append(ExtractedNormOp(module=payload, signature=sig))
                continue
            return None
        if signature is None:
            signature = tuple(
                (
                    "linear",
                    op.linear_index,
                    op.signature,
                )
                if isinstance(op, ExtractedLinearOp)
                else ("pointwise", op.signature)
                if isinstance(op, ExtractedPointwiseOp)
                else ("norm", op.signature)
                for op in ops_list
            )
        if isinstance(signature, list):
            signature = tuple(signature)
        hash(signature)
        return ExtractedBlockSpec(
            signature=signature,
            linears=linears,
            ops=tuple(ops_list),
        )
    return None


def _get_cached_extracted_block(
    layer: Any,
    module: torch.nn.Module,
) -> ExtractedBlockSpec | None:
    cache_key = id(module)
    cached = layer._relation_block_cache.get(cache_key)
    if cached is not None or cache_key in layer._relation_block_cache:
        return cached
    extracted = extract_block(module)
    layer._relation_block_cache[cache_key] = extracted
    return extracted


def _norm_matches_expected_dim(
    norm_sig: tuple[Any, ...],
    expected_dim: int,
    *,
    norm_kind: str | None = None,
) -> bool:
    kind = str(norm_sig[0])
    if norm_kind is not None and kind != norm_kind:
        return False
    shape = tuple(int(v) for v in norm_sig[1])
    return shape == (int(expected_dim),)


def _match_two_linear_pointwise_block(
    relation_slice: RelationSlice,
    block: ExtractedBlockSpec,
    *,
    kernel_type: type[FlatRelationKernel] | None,
    pointwise_kind: str | None,
    norm_position: str | None = None,
    norm_kind: str | None = None,
) -> KernelMatch | None:
    ops = tuple(block.ops)
    expected_dim = int(block.linears[0].in_features)
    if norm_position is None:
        if len(ops) != 3:
            return None
        if not (
            isinstance(ops[0], ExtractedLinearOp)
            and isinstance(ops[1], ExtractedPointwiseOp)
            and isinstance(ops[2], ExtractedLinearOp)
        ):
            return None
        linear0 = ops[0].module
        pointwise_module = ops[1].module
        pointwise_sig = ops[1].signature
        linear1 = ops[2].module
        norm_module = None
        norm_sig = None
    elif norm_position == "pre":
        if len(ops) != 4:
            return None
        if not (
            isinstance(ops[0], ExtractedNormOp)
            and isinstance(ops[1], ExtractedLinearOp)
            and isinstance(ops[2], ExtractedPointwiseOp)
            and isinstance(ops[3], ExtractedLinearOp)
        ):
            return None
        norm_module = ops[0].module
        norm_sig = ops[0].signature
        linear0 = ops[1].module
        pointwise_module = ops[2].module
        pointwise_sig = ops[2].signature
        linear1 = ops[3].module
    elif norm_position == "post":
        if len(ops) != 4:
            return None
        if not (
            isinstance(ops[0], ExtractedLinearOp)
            and isinstance(ops[1], ExtractedPointwiseOp)
            and isinstance(ops[2], ExtractedLinearOp)
            and isinstance(ops[3], ExtractedNormOp)
        ):
            return None
        linear0 = ops[0].module
        pointwise_module = ops[1].module
        pointwise_sig = ops[1].signature
        linear1 = ops[2].module
        norm_module = ops[3].module
        norm_sig = ops[3].signature
    else:
        raise ValueError(f"Unsupported norm_position: {norm_position!r}.")

    if pointwise_kind is not None and pointwise_sig[0] != pointwise_kind:
        return None
    if int(linear0.in_features) != expected_dim:
        return None
    if int(linear1.out_features) != expected_dim:
        return None
    if int(linear1.in_features) != int(linear0.out_features):
        return None
    if norm_sig is not None and not _norm_matches_expected_dim(
        norm_sig, expected_dim, norm_kind=norm_kind
    ):
        return None

    spec = BlockKernelSpec(
        kernel_type=kernel_type,
        signature=(
            int(relation_slice.arity),
            int(linear0.out_features),
            bool(linear0.bias is not None),
            bool(linear1.bias is not None),
            tuple(pointwise_sig),
            tuple(norm_sig) if norm_sig is not None else None,
            norm_position,
        ),
        arity=int(relation_slice.arity),
        input_dim=expected_dim,
        output_dim=expected_dim,
        hidden_dims=(int(linear0.out_features),),
        bias_flags=(
            bool(linear0.bias is not None),
            bool(linear1.bias is not None),
        ),
        pointwise_signature=tuple(pointwise_sig),
        norm_kind=str(norm_sig[0]) if norm_sig is not None else None,
        norm_position=norm_position,
    )
    return KernelMatch(
        spec=spec,
        linears=(linear0, linear1),
        pointwise_modules=(pointwise_module,),
        norm_modules=((norm_module,) if norm_module is not None else ()),
    )


def match_two_linear_pointwise_block(
    relation_slice: RelationSlice,
    block: torch.nn.Module,
    *,
    kernel_type: type[FlatRelationKernel] | None,
    pointwise_kind: str | None,
    norm_position: str | None = None,
    norm_kind: str | None = None,
) -> KernelMatch | None:
    extracted = extract_block(block)
    if extracted is None:
        return None
    return _match_two_linear_pointwise_block(
        relation_slice,
        extracted,
        kernel_type=kernel_type,
        pointwise_kind=pointwise_kind,
        norm_position=norm_position,
        norm_kind=norm_kind,
    )


def _match_three_linear_pointwise_block(
    relation_slice: RelationSlice,
    block: ExtractedBlockSpec,
    *,
    kernel_type: type[FlatRelationKernel] | None,
    pointwise_kind: str | None,
) -> KernelMatch | None:
    ops = tuple(block.ops)
    if len(ops) != 5:
        return None
    if not (
        isinstance(ops[0], ExtractedLinearOp)
        and isinstance(ops[1], ExtractedPointwiseOp)
        and isinstance(ops[2], ExtractedLinearOp)
        and isinstance(ops[3], ExtractedPointwiseOp)
        and isinstance(ops[4], ExtractedLinearOp)
    ):
        return None
    linear0 = ops[0].module
    pointwise0 = ops[1].module
    pointwise0_sig = ops[1].signature
    linear1 = ops[2].module
    pointwise1 = ops[3].module
    pointwise1_sig = ops[3].signature
    linear2 = ops[4].module
    expected_dim = int(linear0.in_features)
    if pointwise_kind is not None:
        if pointwise0_sig[0] != pointwise_kind or pointwise1_sig[0] != pointwise_kind:
            return None
    if int(linear0.in_features) != expected_dim:
        return None
    if int(linear2.out_features) != expected_dim:
        return None
    if int(linear1.in_features) != int(linear0.out_features):
        return None
    if int(linear2.in_features) != int(linear1.out_features):
        return None
    spec = BlockKernelSpec(
        kernel_type=kernel_type,
        signature=(
            int(relation_slice.arity),
            int(linear0.out_features),
            int(linear1.out_features),
            bool(linear0.bias is not None),
            bool(linear1.bias is not None),
            bool(linear2.bias is not None),
            tuple(pointwise0_sig),
            tuple(pointwise1_sig),
        ),
        arity=int(relation_slice.arity),
        input_dim=expected_dim,
        output_dim=expected_dim,
        hidden_dims=(int(linear0.out_features), int(linear1.out_features)),
        bias_flags=(
            bool(linear0.bias is not None),
            bool(linear1.bias is not None),
            bool(linear2.bias is not None),
        ),
        pointwise_signature=tuple(pointwise0_sig),
    )
    return KernelMatch(
        spec=spec,
        linears=(linear0, linear1, linear2),
        pointwise_modules=(pointwise0, pointwise1),
    )


def match_three_linear_pointwise_block(
    relation_slice: RelationSlice,
    block: torch.nn.Module,
    *,
    kernel_type: type[FlatRelationKernel] | None,
    pointwise_kind: str | None,
) -> KernelMatch | None:
    extracted = extract_block(block)
    if extracted is None:
        return None
    return _match_three_linear_pointwise_block(
        relation_slice,
        extracted,
        kernel_type=kernel_type,
        pointwise_kind=pointwise_kind,
    )


def _match_program_stage_block(
    relation_slice: RelationSlice,
    block: torch.nn.Module,
    *,
    allow_structural_fallback: bool,
) -> KernelMatch | None:
    extracted = extract_block(block)
    if extracted is None:
        return None
    for candidate in (
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=PreNormSiLURMSNormKernel,
            pointwise_kind="silu",
            norm_position="pre",
            norm_kind="rmsnorm",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=PostNormSiLULayerNormKernel,
            pointwise_kind="silu",
            norm_position="post",
            norm_kind="layernorm",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=MishBlockKernel,
            pointwise_kind="mish",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=SiLUBlockKernel,
            pointwise_kind="silu",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=GELUBlockKernel,
            pointwise_kind="gelu",
        ),
    ):
        match = candidate()
        if match is not None:
            return match
    if not allow_structural_fallback:
        return None
    for candidate in (
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
            norm_position="pre",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
            norm_position="post",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
        ),
        lambda: _match_three_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
        ),
    ):
        match = candidate()
        if match is not None:
            return match
    return None


def _match_program_kernel_spec(
    relation_slice: RelationSlice,
    *,
    stages: tuple[KernelMatch, ...],
    expected_dim: int,
    program_kernel_type: type[FlatRelationKernel] | None,
) -> ProgramKernelSpec | None:
    if len(stages) != 2:
        return None
    stage0, stage1 = stages
    if (
        stage0.spec.output_dim != expected_dim
        or stage1.spec.output_dim != expected_dim
    ):
        return None
    signature = (
        stage0.spec.kernel_type,
        stage0.spec.signature,
        stage1.spec.kernel_type,
        stage1.spec.signature,
    )
    return ProgramKernelSpec(
        kernel_type=program_kernel_type,
        signature=signature,
        arity=int(relation_slice.arity),
        input_dim=int(expected_dim),
        output_dim=int(expected_dim),
        block_specs=(stage0.spec, stage1.spec),
    )


def _match_executable_program_kernel_spec(
    relation_slice: RelationSlice,
    *,
    stages: tuple[KernelMatch, ...],
    expected_dim: int,
    program_kernel_type: type[FlatRelationKernel],
) -> ProgramKernelSpec | None:
    if len(stages) != 2:
        return None
    stage0, stage1 = stages
    if (
        stage0.spec.output_dim != expected_dim
        or stage1.spec.output_dim != expected_dim
    ):
        return None
    if (
        stage0.spec.kernel_type is SiLUBlockKernel
        and stage1.spec.kernel_type is SiLUBlockKernel
        and program_kernel_type is SiLUPairProgramKernel
    ):
        return _match_program_kernel_spec(
            relation_slice,
            stages=stages,
            expected_dim=expected_dim,
            program_kernel_type=program_kernel_type,
        )
    if (
        stage0.spec.kernel_type is SiLUBlockKernel
        and stage1.spec.kernel_type is PostNormSiLULayerNormKernel
        and program_kernel_type is SiLUThenPostNormProgramKernel
    ):
        return _match_program_kernel_spec(
            relation_slice,
            stages=stages,
            expected_dim=expected_dim,
            program_kernel_type=program_kernel_type,
        )
    if (
        stage0.spec.kernel_type is PreNormSiLURMSNormKernel
        and stage1.spec.kernel_type is SiLUBlockKernel
        and program_kernel_type is PreNormRMSNormThenSiLUProgramKernel
    ):
        return _match_program_kernel_spec(
            relation_slice,
            stages=stages,
            expected_dim=expected_dim,
            program_kernel_type=program_kernel_type,
        )
    return None


def match_exact_relation_program(
    layer: Any,
    relation_slice: RelationSlice,
    module: torch.nn.Module,
    *,
    program_kernel_type: type[FlatRelationKernel] | None = None,
    allow_structural_fallback: bool = False,
) -> KernelMatch | None:
    if not isinstance(module, RelationProgram):
        return None
    expected_dim = int(module.width)
    stages: list[KernelMatch] = []
    for block in module:
        stage_match = _match_program_stage_block(
            relation_slice,
            block,
            allow_structural_fallback=allow_structural_fallback,
        )
        if stage_match is None:
            return None
        stages.append(stage_match)
    stages_t = tuple(stages)
    if program_kernel_type is not None:
        program_spec = _match_executable_program_kernel_spec(
            relation_slice,
            stages=stages_t,
            expected_dim=expected_dim,
            program_kernel_type=program_kernel_type,
        )
        if program_spec is not None:
            return KernelMatch(
                spec=BlockKernelSpec(
                    kernel_type=program_spec.kernel_type,
                    signature=program_spec.signature,
                    arity=int(relation_slice.arity),
                    input_dim=expected_dim,
                    output_dim=expected_dim,
                    hidden_dims=tuple(),
                    bias_flags=tuple(),
                ),
                linears=tuple(),
                program_matches=stages_t,
                program_spec=program_spec,
            )
        if not allow_structural_fallback:
            return None
    if not allow_structural_fallback:
        return None
    program_spec = _match_program_kernel_spec(
        relation_slice,
        stages=stages_t,
        expected_dim=expected_dim,
        program_kernel_type=None,
    )
    if program_spec is None:
        return None
    return KernelMatch(
        spec=BlockKernelSpec(
            kernel_type=None,
            signature=program_spec.signature,
            arity=int(relation_slice.arity),
            input_dim=expected_dim,
            output_dim=expected_dim,
            hidden_dims=tuple(),
            bias_flags=tuple(),
        ),
        linears=tuple(),
        program_matches=stages_t,
        program_spec=program_spec,
    )


def _match_structural_relation_module(
    layer: Any,
    relation_slice: RelationSlice,
    module: torch.nn.Module,
) -> KernelMatch | None:
    extracted = _get_cached_extracted_block(layer, module)
    if extracted is None:
        return None
    for candidate in (
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
            norm_position="pre",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
            norm_position="post",
        ),
        lambda: _match_two_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
        ),
        lambda: _match_three_linear_pointwise_block(
            relation_slice,
            extracted,
            kernel_type=None,
            pointwise_kind=None,
        ),
    ):
        match = candidate()
        if match is not None:
            return match
    if isinstance(module, RelationProgram):
        return match_exact_relation_program(
            layer,
            relation_slice,
            module,
            program_kernel_type=None,
            allow_structural_fallback=True,
        )
    return None


def match_relation_kernel(
    layer: Any,
    relation_slice: RelationSlice,
) -> KernelMatch | None:
    module = layer.update_modules[relation_slice.relation_index]
    cache_key = (id(module), int(relation_slice.arity))
    cached = layer._kernel_match_cache.get(cache_key)
    if cached is not None or cache_key in layer._kernel_match_cache:
        return cached

    match: KernelMatch | None = None
    for kernel in layer.kernels:
        match = kernel.match(layer, relation_slice)
        if match is not None:
            break
    if match is None:
        match = _match_structural_relation_module(layer, relation_slice, module)
    layer._kernel_match_cache[cache_key] = match
    return match


__all__ = [
    "ExtractedLinearOp",
    "ExtractedPointwiseOp",
    "ExtractedNormOp",
    "ExtractedBlockSpec",
    "pointwise_signature",
    "norm_signature",
    "extract_block",
    "match_two_linear_pointwise_block",
    "match_three_linear_pointwise_block",
    "match_exact_relation_program",
    "match_relation_kernel",
]
