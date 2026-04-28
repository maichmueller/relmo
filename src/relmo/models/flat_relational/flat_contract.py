"""Flat relation model contracts.

This module defines the public policy/output objects for the flat relational
runtime and the small tensor-normalization helpers used by the internal
prepared-batch carrier.

Vocabulary used throughout the flat stack:
    entity:
        Any row in the flat encoder's node table. In ``mifrost`` flat encodings
        this is the full set of addressable nodes the relation arguments point
        to. It is the canonical embedding table for message passing.
    object:
        The subset of entity rows corresponding to actual planning objects or
        constants. This is usually the subset consumed by object-level heads.
    target_entity:
        A task-defined subset of entities selected by the encoder for a
        dedicated prediction head. These are still entity rows; they are just
        marked as a special view.
    target:
        The final indexed prediction positions requested by the encoder. In the
        common case these are rows in the entity table, but the name reflects
        model-consumption semantics rather than ontology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor
from torch_geometric.data import Batch, Data

try:  # pragma: no cover - optional native dependency
    import mifrost  # type: ignore
except ImportError:  # pragma: no cover - optional native dependency
    mifrost = None

INT32_MAX = int(torch.iinfo(torch.int32).max)
# Public flat-model input boundary: either a native mifrost flat batch or an
# explicit PyG carrier with the flat relation tensors attached.
if mifrost is not None:
    FlatBatchInput = mifrost.BatchEncoding | Data | Batch
else:
    FlatBatchInput = Data | Batch


@dataclass(frozen=True)
class FlatExecutionPolicy:
    """Runtime execution policy for the flat relation stack.

    Attributes:
        relation_kernels:
            The public policy field is retained for API stability, but the
            current runtime keeps relation kernels disabled and always uses
            eager module execution.
        program_kernels:
            The public policy field is retained for API stability, but the
            current runtime keeps program kernels disabled and always uses
            eager module execution.
        relation_gather:
            The public policy field is retained for API stability, but the
            current runtime keeps the custom relation gather path disabled.
    """

    relation_kernels: str = "auto"
    program_kernels: str = "auto"
    relation_gather: str = "auto"

    def __post_init__(self) -> None:
        if self.relation_kernels not in {"auto", "off"}:
            raise ValueError(
                f"relation_kernels must be 'auto'|'off', got {self.relation_kernels!r}."
            )
        if self.program_kernels not in {"auto", "off"}:
            raise ValueError(
                f"program_kernels must be 'auto'|'off', got {self.program_kernels!r}."
            )
        if self.relation_gather not in {"auto", "off", "on"}:
            raise ValueError(
                f"relation_gather must be 'auto'|'off'|'on', got {self.relation_gather!r}."
            )

    def use_relation_kernels(self, *, device: torch.device) -> bool:
        del device
        return False

    def use_program_kernels(self, *, device: torch.device) -> bool:
        del device
        return False

    def use_relation_gather(self, *, device: torch.device) -> bool:
        del device
        return False


@dataclass(frozen=True)
class _FlatPreparedBatch:
    """Internal normalized carrier derived from a ``mifrost`` flat batch.

    This is not a second public encoder format. It is an internal cacheable
    carrier the model builds after receiving a native flat encoding.

    Shape contract:
        ``x``:
            ``[num_entities, feature_dim]`` placeholder node features. The flat
            relational stack only uses the row count plus dtype/device to size
            the entity embedding table.
        ``relation_counts``:
            ``[batch_size, num_relations]`` count of grounded tuples per graph
            and relation.
        ``relation_args``:
            ``[num_slots]`` packed entity indices. The total slot count must
            equal ``sum(relation_counts[:, r] * relation_arities[r])``.
        ``relation_arities``:
            ``[num_relations]`` relation arities in the same order as
            ``relation_counts`` and the model's ``relations`` mapping.
        ``batch``:
            ``[num_entities]`` graph id per entity row.

    Vocabulary for the optional index tensors:
        ``object_indices``:
            Rows of the entity table that correspond to planning objects or
            constants.
        ``history_entity_indices``:
            Rows selected by the encoder as the history-subset view.
        ``target_entity_indices``:
            Rows selected by the encoder as the target-entity subset view.
        ``target_positions``:
            Final prediction positions requested by the encoder. These are also
            entity-table rows, but they are treated as the dedicated target
            output view.

    All optional index tensors are 1D and point into the final entity embedding
    table produced by the model.

    LGAN-only optional tensors:
        ``lgan_tn_relation_indices`` / ``lgan_tn_entity_indices``:
            Flat target-neighbor edges from pooled relation-instance rows onto
            entity rows.
        ``lgan_nn_relation_indices`` / ``lgan_nn_entity_indices``:
            Flat neighbor-neighbor edges from pooled relation-instance rows
            onto entity rows.
        ``lgan_rr_src_relation_indices`` / ``lgan_rr_dst_relation_indices``:
            Flat relation-relation exchange edges over pooled relation-instance
            rows.
    """

    x: Tensor
    relation_counts: Tensor
    relation_args: Tensor
    relation_arities: Tensor
    batch: Tensor
    node_sizes: Tensor | None
    object_indices: Tensor | None = None
    object_sizes: Tensor | None = None
    history_entity_indices: Tensor | None = None
    history_entity_sizes: Tensor | None = None
    history_entity_dt: Tensor | None = None
    target_entity_indices: Tensor | None = None
    target_entity_group_ids: Tensor | None = None
    target_entity_sizes: Tensor | None = None
    target_positions: Tensor | None = None
    target_group_ids: Tensor | None = None
    target_sizes: Tensor | None = None
    target_indices: Tensor | None = None
    target_candidate_ids: Tensor | None = None
    topology: object | None = None
    lgan_tn_relation_indices: Tensor | None = None
    lgan_tn_entity_indices: Tensor | None = None
    lgan_nn_relation_indices: Tensor | None = None
    lgan_nn_entity_indices: Tensor | None = None
    lgan_rr_src_relation_indices: Tensor | None = None
    lgan_rr_dst_relation_indices: Tensor | None = None
    lgan_topology: object | None = None
    _relm_flat_prepared_batch: bool = True


@dataclass(frozen=True)
class FlatRelationalOutput:
    """Structured flat relation model output.

    Vocabulary:
        ``entity``:
            Embeddings for every row in the flat encoder's node table. This is
            the canonical output of the relational core.
        ``object``:
            ``entity`` restricted to the encoder-provided planning-object view
            via ``object_indices``.
        ``target_entity``:
            ``entity`` restricted to the encoder-provided target-entity view via
            ``target_entity_indices``.
        ``target``:
            ``entity`` restricted to the encoder-provided target positions via
            ``target_positions``.

    Batch-vector fields mirror the corresponding embedding tensor and store the
    graph id per row. Optional views are ``None`` when the encoder did not
    provide the corresponding index set.
    """

    entity: Tensor
    entity_batch: Tensor
    object: Tensor | None = None
    object_batch: Tensor | None = None
    target_entity: Tensor | None = None
    target_entity_batch: Tensor | None = None
    target: Tensor | None = None
    target_batch: Tensor | None = None


def normalize_optional_long_tensor(
    value: Tensor | Sequence[int] | None,
    *,
    device: torch.device,
    flatten: bool = True,
) -> Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=torch.long)
    else:
        out = torch.as_tensor(tuple(int(x) for x in value), device=device, dtype=torch.long)
    return out.view(-1) if flatten else out


def preferred_index_dtype(*, device: torch.device, max_index_bound: int) -> torch.dtype:
    del device, max_index_bound
    return torch.long


def normalize_optional_index_tensor(
    value: Tensor | Sequence[int] | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    flatten: bool = True,
) -> Tensor | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        out = value.to(device=device, dtype=dtype)
    else:
        out = torch.as_tensor(tuple(int(x) for x in value), device=device, dtype=dtype)
    return out.view(-1) if flatten else out


__all__ = [
    "FlatExecutionPolicy",
    "FlatBatchInput",
    "FlatRelationalOutput",
    "INT32_MAX",
    "normalize_optional_index_tensor",
    "normalize_optional_long_tensor",
    "preferred_index_dtype",
]
