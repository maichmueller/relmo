"""Kernel-family matching and grouped-layout planning."""

from __future__ import annotations

from typing import Any, Hashable

import torch

from ..flat_kernel_runtime import KernelExecutionLayout
from ..relation_block_spec import RelationBlockSpec
from ..relation_blocks import RelationProgram
from .kernels import (
    FlatRelationKernel,
    GELUBlockKernel,
    MishBlockKernel,
    PostNormSiLULayerNormKernel,
    PreNormRMSNormThenSiLUProgramKernel,
    PreNormSiLURMSNormKernel,
    SiLUBlockKernel,
    SiLUPairProgramKernel,
    SiLUThenPostNormProgramKernel,
)
from .types import (
    BlockKernelSpec,
    FlatTopology,
    KernelBatchPlan,
    KernelMatch,
    ProgramKernelSpec,
    RelationSlice,
    topology_cache_key,
)


class _ReLUBlockShapeSpec:
    pass


class _ThreeLayerSiLUBlockShapeSpec:
    pass


_KERNEL_SPEC_METHODS = ("relmo_kernel_spec",)
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
_SUPPORTED_NORM_TYPES = (
    (_LAYER_NORM_TYPE,)
    if _RMS_NORM_TYPE is None
    else (_LAYER_NORM_TYPE, _RMS_NORM_TYPE)
)


def _pointwise_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
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


def _norm_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
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


def _extract_relation_block_info(
    module: torch.nn.Module,
) -> dict[str, Any] | None:
    for method_name in _KERNEL_SPEC_METHODS:
        method = getattr(module, method_name, None)
        if not callable(method):
            continue
        spec = method()
        if spec is None:
            return None
        if isinstance(spec, RelationBlockSpec):
            spec = {
                "linears": tuple(spec.linears),
                "ops": tuple(spec.ops),
                "signature": spec.signature,
            }
        if not isinstance(spec, dict):
            raise TypeError(
                f"{type(module).__name__}.{method_name}() must return RelationBlockSpec|dict|None, got {type(spec)!r}."
            )
        linears = tuple(spec.get("linears", ()))
        ops_raw = tuple(spec.get("ops", ()))
        if not linears or not ops_raw:
            return None
        ops: list[tuple[str, Any]] = []
        sig_ops: list[tuple[str, Any]] = []
        for idx, op in enumerate(ops_raw):
            if not (isinstance(op, tuple) and len(op) == 2):
                raise TypeError(
                    f"{type(module).__name__}.{method_name}() ops[{idx}] must be tuple(kind, payload)."
                )
            kind, payload = op
            if kind == "linear":
                lin_idx = int(payload)
                lin = linears[lin_idx]
                if not isinstance(lin, torch.nn.Linear):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() linears[{lin_idx}] must be torch.nn.Linear."
                    )
                ops.append(("linear", lin_idx))
                sig_ops.append(
                    (
                        "linear",
                        int(lin.in_features),
                        int(lin.out_features),
                        bool(lin.bias is not None),
                    )
                )
                continue
            if kind == "pointwise":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] pointwise payload must be torch.nn.Module."
                    )
                pointwise_sig = _pointwise_signature(payload)
                if pointwise_sig is None:
                    return None
                ops.append(("pointwise", payload))
                sig_ops.append(("pointwise", pointwise_sig))
                continue
            if kind == "norm":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] norm payload must be torch.nn.Module."
                    )
                norm_sig = _norm_signature(payload)
                if norm_sig is None:
                    return None
                ops.append(("norm", payload))
                sig_ops.append(("norm", norm_sig))
                continue
            return None
        signature = spec.get("signature", None)
        if signature is None:
            signature = tuple(sig_ops)
        if isinstance(signature, list):
            signature = tuple(signature)
        hash(signature)
        return {
            "signature": signature,
            "linears": linears,
            "ops": tuple(ops),
        }
    return None


class FlatRelationMatchingMixin:
    def _relation_block_info(
        self, module: torch.nn.Module
    ) -> dict[str, Any] | None:
        key = id(module)
        cached = self._relation_block_info_cache.get(key)
        if cached is not None or key in self._relation_block_info_cache:
            return cached
        info = _extract_relation_block_info(module)
        self._relation_block_info_cache[key] = info
        return info

    def _two_linear_pointwise_info(
        self,
        relation_slice: RelationSlice,
    ) -> dict[str, Any] | None:
        module = self.update_modules[relation_slice.relation_index]
        info = self._relation_block_info(module)
        if info is None:
            return None
        expected_dim = int(relation_slice.arity * self.embedding_size)
        linears = tuple(info["linears"])
        ops = tuple(info["ops"])
        if len(linears) != 2 or len(ops) != 3:
            return None
        if ops[0][0] != "linear" or int(ops[0][1]) != 0:
            return None
        if ops[1][0] != "pointwise":
            return None
        if ops[2][0] != "linear" or int(ops[2][1]) != 1:
            return None
        lin0, lin1 = linears
        pointwise_module = ops[1][1]
        if not isinstance(pointwise_module, torch.nn.Module):
            return None
        pointwise_signature = _pointwise_signature(pointwise_module)
        if pointwise_signature is None:
            return None
        if (
            int(lin0.in_features) != expected_dim
            or int(lin1.out_features) != expected_dim
        ):
            return None
        if int(lin1.in_features) != int(lin0.out_features):
            return None
        return {
            "expected_dim": expected_dim,
            "hidden_dim": int(lin0.out_features),
            "linear0": lin0,
            "linear1": lin1,
            "bias_flags": (
                bool(lin0.bias is not None),
                bool(lin1.bias is not None),
            ),
            "pointwise_module": pointwise_module,
            "pointwise_signature": pointwise_signature,
        }

    def _match_two_layer_mish_spec(
        self,
        relation_slice: RelationSlice,
    ) -> KernelMatch | None:
        return self._match_two_layer_pointwise_family(
            relation_slice, kernel_type=MishBlockKernel, pointwise_kind="mish"
        )

    def _normalized_grouped_ops_for_module(
        self,
        module: torch.nn.Module,
        *,
        expected_dim: int,
    ) -> dict[str, Any] | None:
        info = self._relation_block_info(module)
        if info is None:
            return None
        linears = tuple(info["linears"])
        ops = tuple(info["ops"])
        normalized_ops: list[tuple[str, Any, Any]] = []
        for kind, payload in ops:
            if kind == "linear":
                lin_idx = int(payload)
                lin = linears[lin_idx]
                normalized_ops.append(("linear", lin, None))
                continue
            if kind == "pointwise":
                if not isinstance(payload, torch.nn.Module):
                    return None
                pointwise_sig = _pointwise_signature(payload)
                if pointwise_sig is None:
                    return None
                normalized_ops.append(("pointwise", payload, pointwise_sig))
                continue
            if kind == "norm":
                if not isinstance(payload, torch.nn.Module):
                    return None
                norm_sig = _norm_signature(payload)
                if norm_sig is None:
                    return None
                normalized_ops.append(("norm", payload, norm_sig))
                continue
            return None
        return {
            "expected_dim": expected_dim,
            "linears": linears,
            "normalized_ops": tuple(normalized_ops),
        }

    def _normalized_grouped_ops(
        self,
        relation_slice: RelationSlice,
    ) -> dict[str, Any] | None:
        module = self.update_modules[relation_slice.relation_index]
        expected_dim = int(relation_slice.arity * self.embedding_size)
        return self._normalized_grouped_ops_for_module(
            module,
            expected_dim=expected_dim,
        )

    def _norm_matches_expected_dim(
        self,
        norm_signature: tuple[Any, ...],
        expected_dim: int,
        *,
        norm_kind: str | None = None,
    ) -> bool:
        kind = str(norm_signature[0])
        if norm_kind is not None and kind != norm_kind:
            return False
        shape = tuple(int(v) for v in norm_signature[1])
        return shape == (int(expected_dim),)

    def _match_two_layer_pointwise_ops(
        self,
        *,
        relation_slice: RelationSlice,
        ops: tuple[tuple[str, Any, Any], ...],
        expected_dim: int,
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
        norm_position: str | None = None,
        norm_kind: str | None = None,
    ) -> KernelMatch | None:
        if norm_position is None:
            if len(ops) != 3:
                return None
            if (
                ops[0][0] != "linear"
                or ops[1][0] != "pointwise"
                or ops[2][0] != "linear"
            ):
                return None
            norm_module = None
            norm_signature = None
            linear0 = ops[0][1]
            pointwise_module = ops[1][1]
            pointwise_signature = ops[1][2]
            linear1 = ops[2][1]
        elif norm_position == "pre":
            if len(ops) != 4:
                return None
            if (
                ops[0][0] != "norm"
                or ops[1][0] != "linear"
                or ops[2][0] != "pointwise"
                or ops[3][0] != "linear"
            ):
                return None
            norm_module = ops[0][1]
            norm_signature = ops[0][2]
            linear0 = ops[1][1]
            pointwise_module = ops[2][1]
            pointwise_signature = ops[2][2]
            linear1 = ops[3][1]
        elif norm_position == "post":
            if len(ops) != 4:
                return None
            if (
                ops[0][0] != "linear"
                or ops[1][0] != "pointwise"
                or ops[2][0] != "linear"
                or ops[3][0] != "norm"
            ):
                return None
            linear0 = ops[0][1]
            pointwise_module = ops[1][1]
            pointwise_signature = ops[1][2]
            linear1 = ops[2][1]
            norm_module = ops[3][1]
            norm_signature = ops[3][2]
        else:
            raise ValueError(f"Unsupported norm_position: {norm_position!r}.")

        if pointwise_signature[0] != pointwise_kind:
            return None
        if (
            int(linear0.in_features) != expected_dim
            or int(linear1.out_features) != expected_dim
        ):
            return None
        if int(linear1.in_features) != int(linear0.out_features):
            return None
        if norm_signature is not None and not self._norm_matches_expected_dim(
            norm_signature, expected_dim, norm_kind=norm_kind
        ):
            return None
        spec = BlockKernelSpec(
            kernel_type=kernel_type,
            signature=(
                int(relation_slice.arity),
                int(linear0.out_features),
                bool(linear0.bias is not None),
                bool(linear1.bias is not None),
                tuple(pointwise_signature),
                tuple(norm_signature) if norm_signature is not None else None,
            ),
            arity=int(relation_slice.arity),
            input_dim=expected_dim,
            output_dim=expected_dim,
            hidden_dims=(int(linear0.out_features),),
            bias_flags=(
                bool(linear0.bias is not None),
                bool(linear1.bias is not None),
            ),
            pointwise_signature=tuple(pointwise_signature),
            norm_kind=(
                str(norm_signature[0]) if norm_signature is not None else None
            ),
            norm_position=norm_position,
        )
        return KernelMatch(
            spec=spec,
            linears=(linear0, linear1),
            pointwise_modules=(pointwise_module,),
            norm_modules=((norm_module,) if norm_module is not None else ()),
        )

    def _match_two_layer_pointwise_family(
        self,
        relation_slice: RelationSlice,
        *,
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
        norm_position: str | None = None,
        norm_kind: str | None = None,
    ) -> KernelMatch | None:
        info = self._normalized_grouped_ops(relation_slice)
        if info is None:
            return None
        return self._match_two_layer_pointwise_ops(
            relation_slice=relation_slice,
            ops=tuple(info["normalized_ops"]),
            expected_dim=int(info["expected_dim"]),
            kernel_type=kernel_type,
            pointwise_kind=pointwise_kind,
            norm_position=norm_position,
            norm_kind=norm_kind,
        )

    def _match_three_layer_pointwise_ops(
        self,
        *,
        relation_slice: RelationSlice,
        ops: tuple[tuple[str, Any, Any], ...],
        expected_dim: int,
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
    ) -> KernelMatch | None:
        if len(ops) != 5:
            return None
        expected_kinds = (
            "linear",
            "pointwise",
            "linear",
            "pointwise",
            "linear",
        )
        if tuple(item[0] for item in ops) != expected_kinds:
            return None
        linear0 = ops[0][1]
        pointwise0 = ops[1]
        linear1 = ops[2][1]
        pointwise1 = ops[3]
        linear2 = ops[4][1]
        if (
            pointwise0[2][0] != pointwise_kind
            or pointwise1[2][0] != pointwise_kind
        ):
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
                tuple(pointwise0[2]),
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
            pointwise_signature=tuple(pointwise0[2]),
        )
        return KernelMatch(
            spec=spec,
            linears=(linear0, linear1, linear2),
            pointwise_modules=(pointwise0[1], pointwise1[1]),
        )

    def _match_three_layer_pointwise_family(
        self,
        relation_slice: RelationSlice,
        *,
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
    ) -> KernelMatch | None:
        info = self._normalized_grouped_ops(relation_slice)
        if info is None:
            return None
        return self._match_three_layer_pointwise_ops(
            relation_slice=relation_slice,
            ops=tuple(info["normalized_ops"]),
            expected_dim=int(info["expected_dim"]),
            kernel_type=kernel_type,
            pointwise_kind=pointwise_kind,
        )

    def _match_exact_relation_program(
        self,
        relation_slice: RelationSlice,
        *,
        program_kernel_type: type[FlatRelationKernel] | None = None,
    ) -> KernelMatch | None:
        module = self.update_modules[relation_slice.relation_index]
        if not isinstance(module, RelationProgram):
            return None
        expected_dim = int(relation_slice.arity * self.embedding_size)
        if int(module.width) != expected_dim:
            return None
        stages = []
        for block in module:
            stage_match = self._match_program_block(
                relation_slice=relation_slice,
                block=block,
                expected_dim=expected_dim,
            )
            if stage_match is None:
                return None
            stages.append(stage_match)
        if len(stages) <= 1:
            return None
        program_spec = self._match_program_kernel_spec(
            relation_slice,
            stages=tuple(stages),
            expected_dim=expected_dim,
        )
        if program_spec is None:
            return None
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
            program_matches=tuple(stages),
            program_spec=program_spec,
        )

    def _match_program_block(
        self,
        *,
        relation_slice: RelationSlice,
        block: torch.nn.Module,
        expected_dim: int,
    ) -> KernelMatch | None:
        info = self._normalized_grouped_ops_for_module(
            block, expected_dim=expected_dim
        )
        if info is None:
            return None
        ops = tuple(info["normalized_ops"])
        for candidate_len, matcher in (
            (
                4,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=PreNormSiLURMSNormKernel,
                    pointwise_kind="silu",
                    norm_position="pre",
                    norm_kind="rmsnorm",
                ),
            ),
            (
                4,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=PostNormSiLULayerNormKernel,
                    pointwise_kind="silu",
                    norm_position="post",
                    norm_kind="layernorm",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=MishBlockKernel,
                    pointwise_kind="mish",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=SiLUBlockKernel,
                    pointwise_kind="silu",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=GELUBlockKernel,
                    pointwise_kind="gelu",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=_ReLUBlockShapeSpec,
                    pointwise_kind="relu",
                ),
            ),
            (
                5,
                lambda stage_ops: self._match_three_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=_ThreeLayerSiLUBlockShapeSpec,
                    pointwise_kind="silu",
                ),
            ),
        ):
            if len(ops) != candidate_len:
                continue
            candidate = matcher(ops)
            if candidate is not None:
                return candidate
        return None

    def _match_program_kernel_spec(
        self,
        relation_slice: RelationSlice,
        *,
        stages: tuple[KernelMatch, ...],
        expected_dim: int,
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
        ):
            return ProgramKernelSpec(
                kernel_type=SiLUPairProgramKernel,
                signature=(
                    stage0.spec.kernel_type,
                    stage0.spec.signature,
                    stage1.spec.kernel_type,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        if (
            stage0.spec.kernel_type is SiLUBlockKernel
            and stage1.spec.kernel_type is PostNormSiLULayerNormKernel
        ):
            return ProgramKernelSpec(
                kernel_type=SiLUThenPostNormProgramKernel,
                signature=(
                    stage0.spec.kernel_type,
                    stage0.spec.signature,
                    stage1.spec.kernel_type,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        if (
            stage0.spec.kernel_type is PreNormSiLURMSNormKernel
            and stage1.spec.kernel_type is SiLUBlockKernel
        ):
            return ProgramKernelSpec(
                kernel_type=PreNormRMSNormThenSiLUProgramKernel,
                signature=(
                    stage0.spec.kernel_type,
                    stage0.spec.signature,
                    stage1.spec.kernel_type,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        return None

    def _match_kernel(
        self,
        relation_slice: RelationSlice,
    ) -> KernelMatch | None:
        module = self.update_modules[relation_slice.relation_index]
        cache_key = (id(module), int(relation_slice.arity))
        cached = self._kernel_match_cache.get(cache_key)
        if cached is not None or cache_key in self._kernel_match_cache:
            return cached
        match = None
        for kernel in self.kernels:
            match = kernel.match(self, relation_slice)
            if match is not None:
                break
        self._kernel_match_cache[cache_key] = match
        return match

    def _get_kernel_layout(
        self,
        topology: FlatTopology,
    ) -> KernelExecutionLayout:
        cache_key = topology_cache_key(topology)
        cached = self._persistent_kernel_layout_cache.get(cache_key)
        if cached is not None:
            return cached

        grouped_exec: dict[
            tuple[int, int, Hashable], tuple[FlatRelationKernel, list[int]]
        ] = {}
        fallback_indices: list[int] = []
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0 or relation_slice.arity <= 0:
                continue
            match = self._match_kernel(relation_slice)
            if match is None:
                fallback_indices.append(relation_slice.relation_index)
                continue
            if match.kernel is None:
                fallback_indices.append(relation_slice.relation_index)
                continue
            group_key = (
                int(id(match.kernel)),
                relation_slice.arity,
                match.spec.signature,
            )
            grouped_kernel, grouped_indices = grouped_exec.setdefault(
                group_key, (match.kernel, [])
            )
            grouped_indices.append(relation_slice.relation_index)

        groups: list[KernelBatchPlan] = []
        for (
            _kernel_id,
            arity,
            signature,
        ), (kernel, relation_indices) in grouped_exec.items():
            group_slices = tuple(
                topology.relation_slices[idx] for idx in relation_indices
            )
            row_sizes = tuple(
                int(relation_slice.count) for relation_slice in group_slices
            )
            groups.append(
                KernelBatchPlan(
                    kernel=kernel,
                    signature=signature,
                    arity=int(arity),
                    relation_indices=tuple(
                        int(idx) for idx in relation_indices
                    ),
                    max_rows=max(row_sizes) if row_sizes else 0,
                    row_sizes=row_sizes,
                )
            )

        layout = KernelExecutionLayout(
            groups=tuple(groups),
            fallback_indices=tuple(sorted(set(int(idx) for idx in fallback_indices))),
        )
        self._persistent_kernel_layout_cache[cache_key] = layout
        return layout


__all__ = ["FlatRelationMatchingMixin"]
