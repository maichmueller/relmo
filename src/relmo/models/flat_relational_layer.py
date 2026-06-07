"""Flat relation layer facade.

This module keeps the orchestration surface only. Matching, execution,
collection, and topology helpers live under ``relmo.models.flat_relational``.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch_geometric.nn.aggr
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver

from .aggr import LogSumExpAggregation
from .flat_relational.flat_contract import FlatExecutionPolicy
from .flat_relational import collection
from .flat_relational.kernels import FlatRelationKernel, build_default_kernel_registry
from .flat_relational.matching import match_relation_kernel
from .flat_relational.topology import (
    build_flat_topology,
    normalize_relation_arities,
    topology_cache_key,
)
from .flat_relational.types import FlatTopology


class FlatRelationalLayer(torch.nn.Module):
    """Flat relation message passing over packed relation tensors."""

    def __init__(
        self,
        *,
        update_modules: Sequence[torch.nn.Module],
        relation_names: Sequence[str],
        relation_arities: Tensor | Sequence[int] | Iterable[int],
        embedding_size: int,
        aggregation: str | torch_geometric.nn.aggr.Aggregation | None = None,
        execution_policy: FlatExecutionPolicy = FlatExecutionPolicy(),
        kernels: Sequence[FlatRelationKernel] | None = None,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.relation_names = tuple(str(name) for name in relation_names)
        self.relation_arities = normalize_relation_arities(
            relation_arities
        ).cpu()
        self.execution_policy = execution_policy
        if len(update_modules) != len(self.relation_names):
            raise ValueError(
                "update_modules and relation_names must have the same length, got "
                f"{len(update_modules)} vs {len(self.relation_names)}."
            )
        if len(self.relation_names) != int(self.relation_arities.numel()):
            raise ValueError(
                "relation_names and relation_arities must have the same length, got "
                f"{len(self.relation_names)} vs {int(self.relation_arities.numel())}."
            )
        self.update_modules = torch.nn.ModuleList(update_modules)
        aggr_query = aggregation or "logsumexp"
        if isinstance(aggr_query, str):
            if aggr_query.lower() == "logsumexp":
                self.aggr = LogSumExpAggregation()
            else:
                self.aggr = aggregation_resolver(aggr_query)
        else:
            self.aggr = aggr_query

        self._persistent_topology_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], FlatTopology
        ] = {}
        self._persistent_kernel_layout_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], collection.KernelExecutionLayout
        ] = {}
        self._kernel_match_cache: dict[tuple[int, int], object | None] = {}
        self._relation_block_cache: dict[int, object | None] = {}

        self.kernels = (
            tuple(kernels)
            if kernels is not None
            else build_default_kernel_registry()
        )
        self._centralized_batch_spec_cache = collection.build_centralized_batch_spec(
            self
        )

    def use_relation_kernels(self, x: Tensor) -> bool:
        return self.execution_policy.use_relation_kernels(device=x.device)

    def use_program_kernels(self, x: Tensor) -> bool:
        return self.execution_policy.use_program_kernels(device=x.device)

    def use_relation_gather(self, x: Tensor) -> bool:
        return self.execution_policy.use_relation_gather(device=x.device)

    def _use_relation_kernels(self, x: Tensor) -> bool:
        return self.use_relation_kernels(x)

    def _use_program_kernels(self, x: Tensor) -> bool:
        return self.use_program_kernels(x)

    def get_topology(
        self,
        relation_counts: Tensor,
        relation_arities: Tensor | Sequence[int] | Iterable[int] | None = None,
        *,
        cache: dict | None = None,
    ) -> FlatTopology:
        arities = (
            self.relation_arities
            if relation_arities is None
            else normalize_relation_arities(relation_arities)
        )
        topology = build_flat_topology(relation_counts, arities)
        cache_key = topology_cache_key(topology)
        if cache is not None:
            cached = cache.get(cache_key)
            if isinstance(cached, FlatTopology):
                return cached
        cached = self._persistent_topology_cache.get(cache_key)
        if cached is not None:
            if cache is not None:
                cache[cache_key] = cached
            return cached
        self._persistent_topology_cache[cache_key] = topology
        if cache is not None:
            cache[cache_key] = topology
        return topology

    def _aggregate_messages(
        self,
        *,
        x: Tensor,
        collected: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        aggregated = x.new_zeros((int(x.size(0)), self.embedding_size))
        if collected is None:
            return aggregated
        msgs, idx = collected
        return self.aggr(x=msgs, index=idx, dim=0, dim_size=int(x.size(0)))

    def _get_kernel_layout(
        self,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> collection.KernelExecutionLayout:
        return collection.build_kernel_execution_layout(self, topology, cache=cache)

    def _match_kernel(self, relation_slice):
        """Compatibility/debug hook for the quarantined kernel matcher."""
        return match_relation_kernel(self, relation_slice)

    def collect_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        return collection.collect_messages(
            self,
            x,
            relation_args,
            topology,
            cache=cache,
        )

    def collect_eager_relation_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        relation_slice,
        *,
        arg_emb_all: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        """Compatibility/debug hook for direct eager relation collection."""
        return collection.collect_eager_relation_messages(
            self,
            x,
            relation_args,
            relation_slice,
            arg_emb_all=arg_emb_all,
        )

    def collect_slot_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> Tensor | None:
        return collection.collect_slot_messages(
            self,
            x,
            relation_args,
            topology,
            cache=cache,
        )

    def collect_relation_instance_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> Tensor | None:
        return collection.collect_relation_instance_messages(
            self,
            x,
            relation_args,
            topology,
            cache=cache,
        )

    def _run_lgan_pointwise_step(self, *args, **kwargs):
        """Compatibility hook for the removed kernelized LGAN pointwise path."""
        return collection.run_lgan_pointwise_step(self, *args, **kwargs)


    def forward(
        self,
        x: Tensor,
        relation_counts: Tensor,
        relation_args: Tensor,
        *,
        relation_arities: Tensor | Sequence[int] | Iterable[int] | None = None,
        topology: FlatTopology | None = None,
        cache: dict | None = None,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be rank-2, got shape {tuple(x.shape)}.")
        if x.size(1) != self.embedding_size:
            raise ValueError(
                f"x must have feature size {self.embedding_size}, got {x.size(1)}."
            )
        relation_args = relation_args.to(device=x.device, dtype=torch.long).view(-1)
        if topology is None:
            topology = self.get_topology(
                relation_counts, relation_arities=relation_arities, cache=cache
            )
        if topology.slot_offsets[-1] != int(relation_args.numel()):
            raise ValueError(
                "relation_args length does not match the packed slot count implied by "
                f"relation_counts/relation_arities: {int(relation_args.numel())} vs "
                f"{int(topology.slot_offsets[-1])}."
            )

        collected = collection.collect_messages(
            self,
            x,
            relation_args,
            topology,
            cache=cache,
        )
        return self._aggregate_messages(x=x, collected=collected)


__all__ = ["FlatRelationalLayer"]
