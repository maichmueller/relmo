"""Flat relation kernel interfaces and concrete kernel families."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, TYPE_CHECKING

from torch import Tensor

from .types import KernelBatchPlan, KernelMatch, RelationSlice

if TYPE_CHECKING:
    from .types import FlatTopology


class FlatRelationKernel(ABC):
    """Interface for exact flat-kernel families."""

    @abstractmethod
    def match(
        self,
        layer: "FlatRelationalLayer",
        relation_slice: RelationSlice,
    ) -> KernelMatch | None: ...

    @abstractmethod
    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None: ...

    def collect_relation_instances(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        relation_row_starts: dict[int, int],
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        grouped = self.collect(
            layer,
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )
        if grouped is None:
            return None
        msgs, _ = grouped
        return layer._pool_grouped_kernel_messages(
            topology,
            grouped_batch,
            relation_row_starts,
            msgs,
            device=x.device,
            index_dtype=relation_args.dtype,
        )


class MishBlockKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice, kernel_type=type(self), pointwise_kind="mish"
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(MishBlockKernel, SiLUBlockKernel, GELUBlockKernel),
        )


class SiLUBlockKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice, kernel_type=type(self), pointwise_kind="silu"
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(MishBlockKernel, SiLUBlockKernel, GELUBlockKernel),
        )


class GELUBlockKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice, kernel_type=type(self), pointwise_kind="gelu"
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(MishBlockKernel, SiLUBlockKernel, GELUBlockKernel),
        )


class PostNormMishLayerNormKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice,
                kernel_type=type(self),
                pointwise_kind="mish",
                norm_position="post",
                norm_kind="layernorm",
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_postnorm_layernorm_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(PostNormMishLayerNormKernel, PostNormSiLULayerNormKernel),
        )


class PostNormSiLULayerNormKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice,
                kernel_type=type(self),
                pointwise_kind="silu",
                norm_position="post",
                norm_kind="layernorm",
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_postnorm_layernorm_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(PostNormMishLayerNormKernel, PostNormSiLULayerNormKernel),
        )


class PreNormSiLURMSNormKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice,
                kernel_type=type(self),
                pointwise_kind="silu",
                norm_position="pre",
                norm_kind="rmsnorm",
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_prenorm_rmsnorm_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


def _match_expected_program_kernel(
    match: KernelMatch | None,
    kernel_type: type[FlatRelationKernel],
) -> KernelMatch | None:
    if (
        match is None
        or match.program_spec is None
        or match.program_spec.kernel_type is not kernel_type
    ):
        return None
    return match


class SiLUPairProgramKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            _match_expected_program_kernel(
                layer._match_exact_relation_program(
                    relation_slice, program_kernel_type=type(self)
                ),
                type(self),
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_silu_pair_program_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class SiLUThenPostNormProgramKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            _match_expected_program_kernel(
                layer._match_exact_relation_program(
                    relation_slice, program_kernel_type=type(self)
                ),
                type(self),
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_silu_then_postnorm_program_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class PreNormRMSNormThenSiLUProgramKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            _match_expected_program_kernel(
                layer._match_exact_relation_program(
                    relation_slice, program_kernel_type=type(self)
                ),
                type(self),
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: "FlatTopology",
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_prenorm_rmsnorm_then_silu_program_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


def build_default_kernel_registry() -> tuple[FlatRelationKernel, ...]:
    return (
        MishBlockKernel(),
        SiLUBlockKernel(),
        GELUBlockKernel(),
        PostNormMishLayerNormKernel(),
        PostNormSiLULayerNormKernel(),
        PreNormSiLURMSNormKernel(),
        SiLUPairProgramKernel(),
        SiLUThenPostNormProgramKernel(),
        PreNormRMSNormThenSiLUProgramKernel(),
    )


if TYPE_CHECKING:
    from ..flat_relational_layer import FlatRelationalLayer


__all__ = [
    "FlatRelationKernel",
    "MishBlockKernel",
    "SiLUBlockKernel",
    "GELUBlockKernel",
    "PostNormMishLayerNormKernel",
    "PostNormSiLULayerNormKernel",
    "PreNormSiLURMSNormKernel",
    "SiLUPairProgramKernel",
    "SiLUThenPostNormProgramKernel",
    "PreNormRMSNormThenSiLUProgramKernel",
    "build_default_kernel_registry",
]
