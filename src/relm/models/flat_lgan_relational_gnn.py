"""Flat-native LGAN relational model.

This module implements the LGAN-v2 phase structure directly on top of the flat
relation carrier:
1. positional entity -> relation-slot construction
2. relation-instance pooling
3. RR exchange over pooled relation instances
4. TN aggregation onto entities
5. NN aggregation onto entities
6. entity fusion/update over ``[prev, tn, nn]``

The public input boundary matches ``FlatRelationalGNN``:
1. native ``mifrost.BatchEncoding`` flat batches
2. explicit PyG ``Data``/``Batch`` carriers

LGAN-specific topology must be present on the carrier via the flat index
tensors documented in ``docs/flat_lgan_plan.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Mapping, Sequence

import torch
import torch_geometric as pyg
from torch import Tensor

import mifrost

from ._compile import optional_compile
from .flat_contract import (
    FlatExecutionPolicy,
    _FlatPreparedBatch,
    normalize_optional_index_tensor,
)
from .flat_relational_gnn import FlatRelationalGNN
from .flat_relational_layer import FlatTopology
from .mlp import ArityMLPFactory, SimpleMLP


@dataclass(frozen=True)
class FlatLGANTopology:
    """Cached flat LGAN topology derived from the packed relation layout.

    Attributes:
        relation_instance_count:
            Total number of grounded relation instances across the batch.
        slot_to_relation_instance:
            Packed slot-to-relation-instance map of shape ``[num_slots]``.
        relation_instance_arities:
            Arity per pooled relation-instance row, shape
            ``[relation_instance_count]``.
    """

    relation_instance_count: int
    slot_to_relation_instance: Tensor
    relation_instance_arities: Tensor


class FlatLGANRelationalGNN(FlatRelationalGNN):
    """Flat-native LGAN-v2 model on packed relation tensors.

    This baseline preserves the hetero LGAN phase structure, but executes it on
    flat relation-instance and entity index tensors. The current implementation
    keeps relation construction eager so that slot order remains canonical for
    relation-instance pooling.

    Native ``mifrost`` flat carriers may also include:
    1. ``relation_instance_sizes``
    2. ``lgan_tn_sizes`` / ``lgan_nn_sizes`` / ``lgan_rr_sizes``
    3. ``lgan_tn_edge_pos`` / ``lgan_nn_edge_pos`` / ``lgan_rr_edge_pos``

    The model currently ignores those metadata fields because the global packed
    relation/entity indices are already sufficient for execution.
    """

    def __init__(
        self,
        embedding_size: int,
        num_layers: int,
        relations: Mapping[str, int],
        aggregation: str | pyg.nn.aggr.Aggregation | None = "sum",
        relation_modules: Mapping[str, torch.nn.Module] | Sequence[torch.nn.Module] | None = None,
        relation_module_factory: Callable[[int], torch.nn.Module] | ArityMLPFactory | None = None,
        execution_policy: FlatExecutionPolicy = FlatExecutionPolicy(),
        compile_forward: bool = False,
        activation: str | None = None,
    ) -> None:
        super().__init__(
            embedding_size=embedding_size,
            num_layers=num_layers,
            relations=relations,
            aggregation=aggregation,
            relation_modules=relation_modules,
            relation_module_factory=relation_module_factory,
            relation_kernels=None,
            execution_policy=execution_policy,
            compile_forward=compile_forward,
            activation=activation,
        )
        self.fusion_updater = SimpleMLP(
            in_size=3 * self.embedding_size,
            embedding_size=2 * self.embedding_size,
            out_size=self.embedding_size,
            activation=self.activation,
        )
        self._persistent_lgan_topology_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], torch.dtype, str],
            FlatLGANTopology,
        ] = {}

    def _build_lgan_topology(
        self,
        topology: FlatTopology,
        *,
        device: torch.device,
        index_dtype: torch.dtype,
    ) -> FlatLGANTopology:
        relation_ids: list[Tensor] = []
        relation_arities: list[Tensor] = []
        relation_cursor = 0
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0:
                continue
            arity = int(relation_slice.arity)
            relation_rows = torch.arange(
                relation_cursor,
                relation_cursor + int(relation_slice.count),
                device=device,
                dtype=index_dtype,
            )
            relation_arities.append(
                torch.full(
                    (int(relation_slice.count),),
                    arity,
                    device=device,
                    dtype=torch.long,
                )
            )
            if arity > 0:
                relation_ids.append(relation_rows.repeat_interleave(arity))
            relation_cursor += int(relation_slice.count)
        slot_to_relation_instance = (
            torch.cat(relation_ids, dim=0)
            if relation_ids
            else torch.empty((0,), device=device, dtype=index_dtype)
        )
        relation_instance_arities = (
            torch.cat(relation_arities, dim=0)
            if relation_arities
            else torch.empty((0,), device=device, dtype=torch.long)
        )
        if int(slot_to_relation_instance.numel()) != int(topology.slot_offsets[-1]):
            raise RuntimeError(
                "Flat LGAN topology construction produced an invalid slot map."
            )
        return FlatLGANTopology(
            relation_instance_count=int(relation_cursor),
            slot_to_relation_instance=slot_to_relation_instance,
            relation_instance_arities=relation_instance_arities,
        )

    def _get_lgan_topology(
        self,
        prepared: _FlatPreparedBatch,
        *,
        cache: dict | None = None,
    ) -> FlatLGANTopology:
        if prepared.topology is None:
            raise RuntimeError("Flat LGAN requires a prepared flat topology.")
        cache_key = (
            tuple(int(x) for x in prepared.topology.relation_counts_total),
            tuple(int(x) for x in prepared.topology.relation_arities),
            prepared.relation_args.dtype,
            str(prepared.x.device),
        )
        if cache is not None:
            store = cache.setdefault("lgan_topology_store", {})
            cached = store.get(cache_key)
            if isinstance(cached, FlatLGANTopology):
                return cached
        cached = self._persistent_lgan_topology_cache.get(cache_key)
        if cached is not None:
            if cache is not None:
                store[cache_key] = cached
            return cached
        built = self._build_lgan_topology(
            prepared.topology,
            device=prepared.x.device,
            index_dtype=prepared.relation_args.dtype,
        )
        self._persistent_lgan_topology_cache[cache_key] = built
        if cache is not None:
            store[cache_key] = built
        return built

    def _normalize_lgan_indices(
        self,
        prepared: _FlatPreparedBatch,
        *,
        lgan_tn_relation_indices: Tensor | Sequence[int] | None,
        lgan_tn_entity_indices: Tensor | Sequence[int] | None,
        lgan_nn_relation_indices: Tensor | Sequence[int] | None,
        lgan_nn_entity_indices: Tensor | Sequence[int] | None,
        lgan_rr_src_relation_indices: Tensor | Sequence[int] | None,
        lgan_rr_dst_relation_indices: Tensor | Sequence[int] | None,
        cache: dict | None,
    ) -> _FlatPreparedBatch:
        lgan_topology = self._get_lgan_topology(prepared, cache=cache)
        dtype = prepared.relation_args.dtype
        device = prepared.x.device
        fields = {
            "lgan_tn_relation_indices": normalize_optional_index_tensor(
                lgan_tn_relation_indices, device=device, dtype=dtype
            ),
            "lgan_tn_entity_indices": normalize_optional_index_tensor(
                lgan_tn_entity_indices, device=device, dtype=dtype
            ),
            "lgan_nn_relation_indices": normalize_optional_index_tensor(
                lgan_nn_relation_indices, device=device, dtype=dtype
            ),
            "lgan_nn_entity_indices": normalize_optional_index_tensor(
                lgan_nn_entity_indices, device=device, dtype=dtype
            ),
            "lgan_rr_src_relation_indices": normalize_optional_index_tensor(
                lgan_rr_src_relation_indices, device=device, dtype=dtype
            ),
            "lgan_rr_dst_relation_indices": normalize_optional_index_tensor(
                lgan_rr_dst_relation_indices, device=device, dtype=dtype
            ),
        }
        missing = tuple(name for name, value in fields.items() if value is None)
        if missing:
            raise ValueError(
                "FlatLGANRelationalGNN requires all LGAN flat index tensors, "
                f"missing={missing}."
            )

        tn_rel = fields["lgan_tn_relation_indices"]
        tn_ent = fields["lgan_tn_entity_indices"]
        nn_rel = fields["lgan_nn_relation_indices"]
        nn_ent = fields["lgan_nn_entity_indices"]
        rr_src = fields["lgan_rr_src_relation_indices"]
        rr_dst = fields["lgan_rr_dst_relation_indices"]
        assert tn_rel is not None and tn_ent is not None
        assert nn_rel is not None and nn_ent is not None
        assert rr_src is not None and rr_dst is not None

        self._validate_pair_lengths(tn_rel, tn_ent, "TN")
        self._validate_pair_lengths(nn_rel, nn_ent, "NN")
        self._validate_pair_lengths(rr_src, rr_dst, "RR")
        self._validate_index_range(
            tn_rel,
            upper_bound=lgan_topology.relation_instance_count,
            name="lgan_tn_relation_indices",
        )
        self._validate_index_range(
            nn_rel,
            upper_bound=lgan_topology.relation_instance_count,
            name="lgan_nn_relation_indices",
        )
        self._validate_index_range(
            rr_src,
            upper_bound=lgan_topology.relation_instance_count,
            name="lgan_rr_src_relation_indices",
        )
        self._validate_index_range(
            rr_dst,
            upper_bound=lgan_topology.relation_instance_count,
            name="lgan_rr_dst_relation_indices",
        )
        self._validate_index_range(
            tn_ent,
            upper_bound=int(prepared.x.size(0)),
            name="lgan_tn_entity_indices",
        )
        self._validate_index_range(
            nn_ent,
            upper_bound=int(prepared.x.size(0)),
            name="lgan_nn_entity_indices",
        )
        return replace(
            prepared,
            lgan_tn_relation_indices=tn_rel,
            lgan_tn_entity_indices=tn_ent,
            lgan_nn_relation_indices=nn_rel,
            lgan_nn_entity_indices=nn_ent,
            lgan_rr_src_relation_indices=rr_src,
            lgan_rr_dst_relation_indices=rr_dst,
            lgan_topology=lgan_topology,
        )

    def _validate_pair_lengths(self, left: Tensor, right: Tensor, name: str) -> None:
        if int(left.numel()) != int(right.numel()):
            raise ValueError(
                f"{name} LGAN index tensors must have the same length, got "
                f"{int(left.numel())} vs {int(right.numel())}."
            )

    def _validate_index_range(
        self,
        index: Tensor,
        *,
        upper_bound: int,
        name: str,
    ) -> None:
        if int(index.numel()) == 0:
            return
        min_index = int(index.min().item())
        max_index = int(index.max().item())
        if min_index < 0 or max_index >= int(upper_bound):
            raise ValueError(
                f"{name} must lie in [0, {int(upper_bound) - 1}], got [{min_index}, {max_index}]."
            )

    def _prepare_native_flat_batch(
        self,
        data: mifrost.BatchEncoding,
        *,
        cache: dict | None = None,
    ) -> _FlatPreparedBatch:
        x, relation_counts, relation_args, relation_arities, rest = self.unpack_native_flat(data)
        rest = dict(rest)
        prepared = self._prepare_from_unpacked(
            x=x,
            relation_counts=relation_counts,
            relation_args=relation_args,
            relation_arities=relation_arities,
            relation_names=rest.pop("relation_names", None),
            batch=rest.pop("batch", None),
            node_sizes=rest.pop("node_sizes", None),
            object_indices=rest.pop("object_indices", None),
            object_sizes=rest.pop("object_sizes", None),
            history_entity_indices=rest.pop("history_entity_indices", None),
            history_entity_sizes=rest.pop("history_entity_sizes", None),
            history_entity_dt=rest.pop("history_entity_dt", None),
            target_entity_indices=rest.pop("target_entity_indices", None),
            target_entity_group_ids=rest.pop("target_entity_group_ids", None),
            target_entity_sizes=rest.pop("target_entity_sizes", None),
            target_positions=rest.pop("target_positions", None),
            target_group_ids=rest.pop("target_group_ids", None),
            target_sizes=rest.pop("target_sizes", None),
            target_indices=rest.pop("target_indices", None),
            target_candidate_ids=rest.pop("target_candidate_ids", None),
            cache=cache,
        )
        return self._normalize_lgan_indices(
            prepared,
            lgan_tn_relation_indices=rest.pop("lgan_tn_relation_indices", None),
            lgan_tn_entity_indices=rest.pop("lgan_tn_entity_indices", None),
            lgan_nn_relation_indices=rest.pop("lgan_nn_relation_indices", None),
            lgan_nn_entity_indices=rest.pop("lgan_nn_entity_indices", None),
            lgan_rr_src_relation_indices=rest.pop("lgan_rr_src_relation_indices", None),
            lgan_rr_dst_relation_indices=rest.pop("lgan_rr_dst_relation_indices", None),
            cache=cache,
        )

    def _prepare_pyg_flat_batch(
        self,
        data: pyg.data.Data | pyg.data.Batch,
        *,
        cache: dict | None = None,
    ) -> _FlatPreparedBatch:
        if not (hasattr(data, "relation_counts") and hasattr(data, "relation_args")):
            raise TypeError("PyG flat inputs must carry relation_counts and relation_args.")
        x, relation_counts, relation_args, relation_arities, rest = self.unpack(data)
        rest = dict(rest)
        prepared = self._prepare_from_unpacked(
            x=x,
            relation_counts=relation_counts,
            relation_args=relation_args,
            relation_arities=relation_arities,
            relation_names=rest.pop("relation_names", None),
            batch=rest.pop("batch", None),
            node_sizes=rest.pop("node_sizes", None),
            object_indices=rest.pop("object_indices", None),
            object_sizes=rest.pop("object_sizes", None),
            history_entity_indices=rest.pop("history_entity_indices", None),
            history_entity_sizes=rest.pop("history_entity_sizes", None),
            history_entity_dt=rest.pop("history_entity_dt", None),
            target_entity_indices=rest.pop("target_entity_indices", None),
            target_entity_group_ids=rest.pop("target_entity_group_ids", None),
            target_entity_sizes=rest.pop("target_entity_sizes", None),
            target_positions=rest.pop("target_positions", None),
            target_group_ids=rest.pop("target_group_ids", None),
            target_sizes=rest.pop("target_sizes", None),
            target_indices=rest.pop("target_indices", None),
            target_candidate_ids=rest.pop("target_candidate_ids", None),
            cache=cache,
        )
        return self._normalize_lgan_indices(
            prepared,
            lgan_tn_relation_indices=rest.pop("lgan_tn_relation_indices", None),
            lgan_tn_entity_indices=rest.pop("lgan_tn_entity_indices", None),
            lgan_nn_relation_indices=rest.pop("lgan_nn_relation_indices", None),
            lgan_nn_entity_indices=rest.pop("lgan_nn_entity_indices", None),
            lgan_rr_src_relation_indices=rest.pop("lgan_rr_src_relation_indices", None),
            lgan_rr_dst_relation_indices=rest.pop("lgan_rr_dst_relation_indices", None),
            cache=cache,
        )

    def _build_relation_slot_messages(
        self,
        entity_embeddings: Tensor,
        prepared: _FlatPreparedBatch,
        *,
        cache: dict | None = None,
    ) -> Tensor:
        if prepared.topology is None:
            raise RuntimeError("Flat LGAN requires a prepared flat topology.")
        slot_messages = self.relational_layer.collect_slot_messages(
            entity_embeddings,
            prepared.relation_args,
            prepared.topology,
            cache=cache,
        )
        if slot_messages is None:
            return entity_embeddings.new_zeros((0, self.embedding_size))
        return slot_messages

    def _pool_relation_instances(
        self,
        slot_messages: Tensor,
        lgan_topology: FlatLGANTopology,
    ) -> Tensor:
        relation_pair_x = slot_messages.new_zeros(
            (int(lgan_topology.relation_instance_count), self.embedding_size)
        )
        if int(slot_messages.numel()) == 0:
            return relation_pair_x
        relation_pair_x.index_add_(0, lgan_topology.slot_to_relation_instance, slot_messages)
        counts = (
            lgan_topology.relation_instance_arities.to(
                device=slot_messages.device,
                dtype=slot_messages.dtype,
            )
            .view(-1, 1)
            .clamp_min_(1.0)
        )
        return relation_pair_x / counts

    def _aggregate_indexed(
        self,
        source_embeddings: Tensor,
        source_index: Tensor | None,
        target_index: Tensor | None,
        *,
        dim_size: int,
    ) -> Tensor:
        out = source_embeddings.new_zeros((int(dim_size), int(source_embeddings.size(-1))))
        if source_index is None or target_index is None or int(source_index.numel()) == 0:
            return out
        gathered = source_embeddings.index_select(0, source_index)
        return self.relational_layer.aggr(
            x=gathered,
            index=target_index,
            dim=0,
            dim_size=int(dim_size),
        )

    @optional_compile(enable_attr="_compile_forward", backend="inductor", dynamic=True)
    def _compute_entity_embeddings_prepared(
        self,
        prepared_batch: _FlatPreparedBatch,
        *,
        cache: dict | None = None,
    ) -> Tensor:
        if prepared_batch.lgan_topology is None:
            raise RuntimeError(
                "FlatLGANRelationalGNN requires prepared LGAN topology and index tensors."
            )
        lgan_topology = prepared_batch.lgan_topology
        entity_embeddings = self.initialize_embeddings(prepared_batch.x)
        for _ in range(self.num_layers):
            slot_messages = self._build_relation_slot_messages(
                entity_embeddings, prepared_batch, cache=cache
            )
            relation_pair_x = self._pool_relation_instances(slot_messages, lgan_topology)
            rr_msgs = self._aggregate_indexed(
                relation_pair_x,
                prepared_batch.lgan_rr_src_relation_indices,
                prepared_batch.lgan_rr_dst_relation_indices,
                dim_size=int(lgan_topology.relation_instance_count),
            )
            relation_pair_x = relation_pair_x + rr_msgs
            tn_msgs = self._aggregate_indexed(
                relation_pair_x,
                prepared_batch.lgan_tn_relation_indices,
                prepared_batch.lgan_tn_entity_indices,
                dim_size=int(entity_embeddings.size(0)),
            )
            nn_msgs = self._aggregate_indexed(
                relation_pair_x,
                prepared_batch.lgan_nn_relation_indices,
                prepared_batch.lgan_nn_entity_indices,
                dim_size=int(entity_embeddings.size(0)),
            )
            updated = self.fusion_updater(
                torch.cat([entity_embeddings, tn_msgs, nn_msgs], dim=1)
            )
            entity_embeddings = entity_embeddings + updated
        return entity_embeddings


__all__ = ["FlatLGANRelationalGNN", "FlatLGANTopology"]
