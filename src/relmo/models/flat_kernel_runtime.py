"""Shared runtime helpers for flat kernel/fallback orchestration.

This module centralizes the relation-dispatch loops used by:
1. slot message collection
2. relation-instance pooling
3. LGAN integrated pointwise payload construction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from torch import Tensor


@dataclass(frozen=True)
class KernelExecutionLayout:
    """Precomputed grouped-kernel execution layout for one flat topology."""

    groups: tuple[Any, ...]
    fallback_indices: tuple[int, ...]


@dataclass(frozen=True)
class KernelExecutionContext:
    """Runtime state shared by grouped and eager-fallback dispatch."""

    topology: Any
    layout: KernelExecutionLayout
    grouped_param_stacks: dict[tuple[Any, ...], Tensor]
    allow_persistent_stacks: bool
    fallback_arg_emb_all: Tensor | None = None


class FlatKernelRuntime:
    """Dispatch grouped kernel batches with eager fallback once per relation."""

    @staticmethod
    def run_relation_dispatch(
        *,
        context: KernelExecutionContext,
        on_group: Callable[[Any, KernelExecutionContext], Iterable[int] | None],
        on_relation_slice: Callable[[Any, KernelExecutionContext], bool],
    ) -> set[int]:
        consumed: set[int] = set()
        topology = context.topology

        for grouped_batch in context.layout.groups:
            consumed_indices = on_group(grouped_batch, context)
            if consumed_indices is None:
                continue
            consumed.update(int(idx) for idx in consumed_indices)

        for relation_index in context.layout.fallback_indices:
            relation_slice = topology.relation_slices[int(relation_index)]
            if on_relation_slice(relation_slice, context):
                consumed.add(int(relation_index))

        for relation_slice in topology.relation_slices:
            relation_index = int(relation_slice.relation_index)
            if relation_index in consumed:
                continue
            if on_relation_slice(relation_slice, context):
                consumed.add(relation_index)

        return consumed

