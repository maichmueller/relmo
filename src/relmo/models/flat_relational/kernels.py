"""Flat relation kernel adapters and the default kernel registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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
        from . import collection

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
        return collection.pool_grouped_kernel_messages(
            layer,
            topology,
            grouped_batch,
            relation_row_starts,
            msgs,
            device=x.device,
        )


def _bind_kernel_match(kernel: FlatRelationKernel, match: KernelMatch | None) -> KernelMatch | None:
    if match is None:
        return None
    return replace(match, kernel=kernel)


class MishBlockKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_two_linear_pointwise_block(
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                kernel_type=type(self),
                pointwise_kind="mish",
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            embedding_size=int(layer.embedding_size),
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class SiLUBlockKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_two_linear_pointwise_block(
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                kernel_type=type(self),
                pointwise_kind="silu",
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            embedding_size=int(layer.embedding_size),
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class GELUBlockKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_two_linear_pointwise_block(
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                kernel_type=type(self),
                pointwise_kind="gelu",
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            embedding_size=int(layer.embedding_size),
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class PostNormMishLayerNormKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_two_linear_pointwise_block(
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                kernel_type=type(self),
                pointwise_kind="mish",
                norm_position="post",
                norm_kind="layernorm",
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_postnorm_layernorm_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            embedding_size=int(layer.embedding_size),
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class PostNormSiLULayerNormKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_two_linear_pointwise_block(
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                kernel_type=type(self),
                pointwise_kind="silu",
                norm_position="post",
                norm_kind="layernorm",
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_postnorm_layernorm_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            embedding_size=int(layer.embedding_size),
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class PreNormSiLURMSNormKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_two_linear_pointwise_block(
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                kernel_type=type(self),
                pointwise_kind="silu",
                norm_position="pre",
                norm_kind="rmsnorm",
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_prenorm_rmsnorm_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            embedding_size=int(layer.embedding_size),
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class SiLUPairProgramKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_exact_relation_program(
                layer,
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                program_kernel_type=type(self),
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_silu_pair_program_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class SiLUThenPostNormProgramKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_exact_relation_program(
                layer,
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                program_kernel_type=type(self),
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_silu_then_postnorm_program_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class PreNormRMSNormThenSiLUProgramKernel(FlatRelationKernel):
    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        from . import matching

        return _bind_kernel_match(
            self,
            matching.match_exact_relation_program(
                layer,
                relation_slice,
                layer.update_modules[relation_slice.relation_index],
                program_kernel_type=type(self),
            ),
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
        from . import collection, execution

        batch_items = collection._collect_block_batch_items(layer, topology, grouped_batch)
        if not batch_items:
            return None
        return execution._run_prenorm_rmsnorm_then_silu_program_kernel(
            x,
            relation_args,
            grouped_batch,
            batch_items,
            grouped_param_stacks=grouped_param_stacks,
            persistent_grouped_param_stacks=layer._persistent_grouped_param_stacks,
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
