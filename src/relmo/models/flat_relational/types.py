"""Shared flat-relational datatypes and topology helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, Iterable, Sequence

import torch
from torch import Tensor

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
    kernel_type: type["FlatRelationKernel"]
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
    kernel_type: type["FlatRelationKernel"]
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


def normalize_relation_arities(
    relation_arities: Tensor | Sequence[int] | Iterable[int],
    *,
    device: torch.device | None = None,
) -> Tensor:
    if torch.is_tensor(relation_arities):
        out = relation_arities.to(device=device, dtype=torch.long)
    else:
        out = torch.as_tensor(
            tuple(int(x) for x in relation_arities),
            dtype=torch.long,
            device=device,
        )
    if out.dim() != 1:
        raise ValueError(
            f"relation_arities must be 1D, got shape {tuple(out.shape)}."
        )
    return out


def normalize_relation_counts(
    relation_counts: Tensor,
    *,
    device: torch.device | None = None,
) -> Tensor:
    if not torch.is_tensor(relation_counts):
        raise TypeError("relation_counts must be a torch.Tensor.")
    out = relation_counts.to(device=device, dtype=torch.long)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    if out.dim() != 2:
        raise ValueError(
            f"relation_counts must have shape [R] or [B, R], got {tuple(out.shape)}."
        )
    return out


def build_flat_topology(
    relation_counts: Tensor,
    relation_arities: Tensor | Sequence[int] | Iterable[int],
) -> FlatTopology:
    counts_2d = normalize_relation_counts(relation_counts)
    arities_1d = normalize_relation_arities(
        relation_arities, device=counts_2d.device
    )
    if int(counts_2d.size(1)) != int(arities_1d.numel()):
        raise ValueError(
            "relation_counts and relation_arities disagree on relation dimension: "
            f"{tuple(counts_2d.shape)} vs {tuple(arities_1d.shape)}."
        )

    counts_total = counts_2d.sum(dim=0)
    relation_slices: list[RelationSlice] = []
    slot_offsets = [0]
    cursor = 0
    for relation_index, (count_t, arity_t) in enumerate(
        zip(counts_total, arities_1d)
    ):
        count = int(count_t.item())
        arity = int(arity_t.item())
        if arity < 0:
            raise ValueError(f"relation arity must be >= 0, got {arity}.")
        width = count * arity
        relation_slices.append(
            RelationSlice(
                relation_index=relation_index,
                count=count,
                arity=arity,
                slot_start=cursor,
                slot_end=cursor + width,
            )
        )
        cursor += width
        slot_offsets.append(cursor)
    return FlatTopology(
        relation_counts_total=tuple(int(x.item()) for x in counts_total),
        relation_arities=tuple(int(x.item()) for x in arities_1d),
        relation_slices=tuple(relation_slices),
        slot_offsets=tuple(int(x) for x in slot_offsets),
    )


def topology_cache_key(
    topology: FlatTopology,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return topology.relation_counts_total, topology.relation_arities


# Backward-compatible internal name used across the old flat layer internals.
_topology_cache_key = topology_cache_key


__all__ = [
    "RelationSlice",
    "FlatTopology",
    "KernelBatchPlan",
    "BlockKernelSpec",
    "ProgramKernelSpec",
    "KernelMatch",
    "CentralizedBatchSpec",
    "normalize_relation_arities",
    "normalize_relation_counts",
    "build_flat_topology",
    "topology_cache_key",
    "_topology_cache_key",
]
