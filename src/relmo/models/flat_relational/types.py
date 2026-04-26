"""Shared flat-relational contracts and typed batch plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable

import torch

if TYPE_CHECKING:
    from .kernels import FlatRelationKernel


@dataclass(frozen=True)
class RelationSlice:
    relation_index: int
    count: int
    arity: int
    slot_start: int
    slot_end: int


@dataclass(frozen=True)
class FlatTopology:
    relation_counts_total: tuple[int, ...]
    relation_arities: tuple[int, ...]
    relation_slices: tuple[RelationSlice, ...]
    slot_offsets: tuple[int, ...]


@dataclass(frozen=True)
class KernelBatchPlan:
    kernel: "FlatRelationKernel"
    signature: Hashable
    arity: int
    relation_indices: tuple[int, ...]
    max_rows: int
    row_sizes: tuple[int, ...]


@dataclass(frozen=True)
class BlockKernelSpec:
    kernel_type: type["FlatRelationKernel"] | None
    signature: Hashable
    arity: int
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    bias_flags: tuple[bool, ...]
    pointwise_signature: tuple[Any, ...] | None = None
    norm_kind: str | None = None
    norm_position: str | None = None


@dataclass(frozen=True)
class ProgramKernelSpec:
    kernel_type: type["FlatRelationKernel"] | None
    signature: Hashable
    arity: int
    input_dim: int
    output_dim: int
    block_specs: tuple[BlockKernelSpec, ...]


@dataclass(frozen=True)
class KernelMatch:
    spec: BlockKernelSpec
    linears: tuple[torch.nn.Linear, ...]
    kernel: "FlatRelationKernel | None" = None
    pointwise_modules: tuple[torch.nn.Module, ...] = ()
    norm_modules: tuple[torch.nn.Module, ...] = ()
    program_matches: tuple["KernelMatch", ...] = ()
    program_spec: ProgramKernelSpec | None = None


@dataclass(frozen=True)
class CentralizedBatchSpec:
    central_module: torch.nn.Module
    condition_embedding: torch.nn.Embedding
    condition_position: str
    max_arity: int
    embedding_size: int
    include_slot_mask: bool
    condition_indices: tuple[int, ...]


__all__ = [
    "RelationSlice",
    "FlatTopology",
    "KernelBatchPlan",
    "BlockKernelSpec",
    "ProgramKernelSpec",
    "KernelMatch",
    "CentralizedBatchSpec",
]
