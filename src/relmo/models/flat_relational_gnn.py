"""Flat relational GNN public model API.

This module owns the high-level flat relation model surface:
1. input normalization into an internal prepared-batch carrier
2. embedding-state rollout over :class:`FlatRelationalLayer`
3. structured :class:`FlatRelationalOutput` construction

The actual relation message passing lives in ``flat_relational_layer``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver

from ._compile import optional_compile
from .aggr import LogSumExpAggregation
from .flat_contract import (
    FlatBatchInput,
    FlatExecutionPolicy,
    _FlatPreparedBatch,
    FlatRelationalOutput,
    normalize_optional_index_tensor,
    normalize_optional_long_tensor,
    preferred_index_dtype,
)
from .flat_relational_layer import (
    FlatRelationKernel,
    FlatRelationalLayer,
    normalize_relation_arities,
    normalize_relation_counts,
)
from .mlp import ArityMLPFactory, SimpleMLP
from .pyg_module import PyGFlatModule

import mifrost  # type: ignore

Relations = dict[str, int]


@dataclass(frozen=True)
class _NormalizedRelationCore:
    x: Tensor
    relation_counts: Tensor
    relation_args: Tensor
    relation_arities: Tensor
    index_dtype: torch.dtype


@dataclass(frozen=True)
class _NormalizedFlatViews:
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


class FlatRelationalGNN(PyGFlatModule):
    """Flat relation message-passing network.

    Public inputs are:
    1. native ``mifrost.BatchEncoding`` flat batches
    2. explicit PyG ``Data``/``Batch`` objects carrying the flat relation
       tensors

    The model returns a structured output object. Relation modules are
    width-preserving transforms over packed relation rows of shape
    ``[rows, arity * embedding]``. The layer applies the tuple residual exactly
    once when writing messages back to the entity table.
    """

    def __init__(
        self,
        embedding_size: int,
        num_layers: int,
        relations: Mapping[str, int],
        aggregation: str | pyg.nn.aggr.Aggregation | None = "sum",
        relation_modules: Mapping[str, torch.nn.Module] | Sequence[torch.nn.Module] | None = None,
        relation_module_factory: Callable[[int], torch.nn.Module] | ArityMLPFactory | None = None,
        relation_kernels: Sequence[FlatRelationKernel] | None = None,
        execution_policy: FlatExecutionPolicy = FlatExecutionPolicy(),
        compile_forward: bool = False,
        activation: str | Callable | None = None,
    ) -> None:
        super().__init__()
        self._compile_forward = bool(compile_forward)
        self._compile_public_api = False
        self.embedding_size = int(embedding_size)
        self.num_layers = int(num_layers)
        self.relations = {str(key): int(value) for key, value in relations.items()}
        self.relation_names = tuple(self.relations.keys())
        self.relation_arities = torch.as_tensor(
            tuple(int(v) for v in self.relations.values()), dtype=torch.long
        )
        self.activation = activation or "mish"
        self.execution_policy = execution_policy

        modules = self._build_relation_modules(
            relation_modules=relation_modules,
            relation_module_factory=relation_module_factory,
        )
        self._validate_relation_module_widths(modules)

        if isinstance(aggregation, str) or aggregation is None:
            if aggregation is None or aggregation.lower() == "logsumexp":
                resolved_aggregation: str | pyg.nn.aggr.Aggregation = LogSumExpAggregation()
            else:
                resolved_aggregation = aggregation_resolver(aggregation)
        else:
            resolved_aggregation = aggregation

        self.relational_layer = FlatRelationalLayer(
            update_modules=modules,
            relation_names=self.relation_names,
            relation_arities=self.relation_arities,
            embedding_size=self.embedding_size,
            aggregation=aggregation if isinstance(aggregation, str) else resolved_aggregation,
            execution_policy=execution_policy,
            kernels=relation_kernels,
        )
        self.embedding_updater = SimpleMLP(
            in_size=2 * self.embedding_size,
            embedding_size=2 * self.embedding_size,
            out_size=self.embedding_size,
            activation=self.activation,
        )

    def _build_relation_modules(
        self,
        *,
        relation_modules: Mapping[str, torch.nn.Module] | Sequence[torch.nn.Module] | None,
        relation_module_factory: Callable[[int], torch.nn.Module] | ArityMLPFactory | None,
    ) -> list[torch.nn.Module]:
        if relation_modules is not None and relation_module_factory is not None:
            raise ValueError(
                "Provide either relation_modules or relation_module_factory, not both."
            )
        if relation_modules is None and relation_module_factory is None:
            relation_module_factory = ArityMLPFactory(
                feature_size=self.embedding_size,
                added_arity=0,
                residual=False,
                padding=None,
                layers=1,
                activation=self.activation,
            )

        if relation_modules is not None:
            if isinstance(relation_modules, Mapping):
                keys = tuple(str(key) for key in relation_modules.keys())
                if set(keys) != set(self.relation_names):
                    missing = tuple(name for name in self.relation_names if name not in relation_modules)
                    extra = tuple(name for name in keys if name not in self.relation_names)
                    raise ValueError(
                        "relation_modules mapping keys must match relations exactly, "
                        f"missing={missing}, extra={extra}."
                    )
                return [relation_modules[name] for name in self.relation_names]
            modules = list(relation_modules)
            if len(modules) != len(self.relation_names):
                raise ValueError(
                    "relation_modules sequence length must match relations, got "
                    f"{len(modules)} vs {len(self.relation_names)}."
                )
            return modules

        assert relation_module_factory is not None
        return [relation_module_factory(arity) for arity in self.relation_arities.tolist()]

    def _validate_relation_module_widths(self, modules: Sequence[torch.nn.Module]) -> None:
        for relation_name, arity, module in zip(
            self.relation_names,
            self.relation_arities.tolist(),
            modules,
        ):
            width = int(arity * self.embedding_size)
            module_width = getattr(module, "width", None)
            if module_width is not None and int(module_width) != width:
                raise ValueError(
                    f"relation module {relation_name!r} declares width {module_width}, expected {width}."
                )

    def _normalize_relation_core(
        self,
        x: Tensor,
        relation_counts: Tensor,
        relation_args: Tensor,
        relation_arities: Tensor | Sequence[int] | None = None,
        *,
        relation_names: Sequence[str] | None = None,
    ) -> _NormalizedRelationCore:
        """Canonicalize the flat relation tensors required by the core layer."""
        if x.dim() != 2:
            raise ValueError(f"x must be rank-2, got shape {tuple(x.shape)}.")
        relation_counts_2d = normalize_relation_counts(relation_counts, device=x.device)
        index_dtype = preferred_index_dtype(device=x.device, max_index_bound=int(x.size(0)))
        relation_args_1d = relation_args.to(device=x.device, dtype=index_dtype).view(-1)
        arities_1d = (
            self.relation_arities.to(device=x.device)
            if relation_arities is None
            else normalize_relation_arities(relation_arities, device=x.device)
        )
        if relation_counts_2d.size(1) != arities_1d.numel():
            raise ValueError(
                "relation_counts and relation_arities must agree on the relation dimension, got "
                f"{tuple(relation_counts_2d.shape)} vs {tuple(arities_1d.shape)}."
            )
        if arities_1d.numel() != self.relation_arities.numel():
            raise ValueError("flat model relation dimension does not match constructor relations.")
        if not torch.equal(arities_1d.cpu(), self.relation_arities):
            raise ValueError("input relation_arities does not match the model relation order.")
        if relation_names is not None and tuple(str(name) for name in relation_names) != self.relation_names:
            raise ValueError("input relation_names does not match the model relation order.")
        return _NormalizedRelationCore(
            x=x,
            relation_counts=relation_counts_2d,
            relation_args=relation_args_1d,
            relation_arities=arities_1d,
            index_dtype=index_dtype,
        )

    def _resolve_entity_batch(
        self,
        x: Tensor,
        *,
        batch: Tensor | None = None,
        node_sizes: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Resolve the graph id per entity row from ``batch`` or ``node_sizes``."""
        node_sizes_t = (
            None if node_sizes is None else node_sizes.to(device=x.device, dtype=torch.long).view(-1)
        )
        if batch is None:
            if node_sizes_t is not None:
                if int(node_sizes_t.sum().item()) != int(x.size(0)):
                    raise ValueError(
                        "node_sizes does not sum to x.size(0): "
                        f"{int(node_sizes_t.sum().item())} vs {int(x.size(0))}."
                    )
                batch = torch.repeat_interleave(
                    torch.arange(int(node_sizes_t.numel()), device=x.device, dtype=torch.long),
                    node_sizes_t,
                )
            else:
                batch = torch.zeros((int(x.size(0)),), device=x.device, dtype=torch.long)
        else:
            batch = batch.to(device=x.device, dtype=torch.long).view(-1)
        return batch, node_sizes_t

    def _normalize_output_views(
        self,
        *,
        device: torch.device,
        index_dtype: torch.dtype,
        batch: Tensor,
        node_sizes: Tensor | None,
        object_indices: Tensor | Sequence[int] | None = None,
        object_sizes: Tensor | Sequence[int] | None = None,
        history_entity_indices: Tensor | Sequence[int] | None = None,
        history_entity_sizes: Tensor | Sequence[int] | None = None,
        history_entity_dt: Tensor | Sequence[int] | None = None,
        target_entity_indices: Tensor | Sequence[int] | None = None,
        target_entity_group_ids: Tensor | Sequence[int] | None = None,
        target_entity_sizes: Tensor | Sequence[int] | None = None,
        target_positions: Tensor | Sequence[int] | None = None,
        target_group_ids: Tensor | Sequence[int] | None = None,
        target_sizes: Tensor | Sequence[int] | None = None,
        target_indices: Tensor | Sequence[int] | None = None,
        target_candidate_ids: Tensor | Sequence[int] | None = None,
    ) -> _NormalizedFlatViews:
        """Canonicalize optional encoder-provided entity subset views."""
        return _NormalizedFlatViews(
            batch=batch,
            node_sizes=node_sizes,
            object_indices=normalize_optional_index_tensor(
                object_indices,
                device=device,
                dtype=index_dtype,
            ),
            object_sizes=normalize_optional_long_tensor(object_sizes, device=device),
            history_entity_indices=normalize_optional_index_tensor(
                history_entity_indices,
                device=device,
                dtype=index_dtype,
            ),
            history_entity_sizes=normalize_optional_long_tensor(
                history_entity_sizes, device=device
            ),
            history_entity_dt=normalize_optional_long_tensor(history_entity_dt, device=device),
            target_entity_indices=normalize_optional_index_tensor(
                target_entity_indices,
                device=device,
                dtype=index_dtype,
            ),
            target_entity_group_ids=normalize_optional_long_tensor(
                target_entity_group_ids, device=device
            ),
            target_entity_sizes=normalize_optional_long_tensor(
                target_entity_sizes, device=device
            ),
            target_positions=normalize_optional_index_tensor(
                target_positions,
                device=device,
                dtype=index_dtype,
            ),
            target_group_ids=normalize_optional_long_tensor(target_group_ids, device=device),
            target_sizes=normalize_optional_long_tensor(target_sizes, device=device),
            target_indices=normalize_optional_long_tensor(target_indices, device=device),
            target_candidate_ids=normalize_optional_long_tensor(
                target_candidate_ids, device=device
            ),
        )

    def _prepare_from_unpacked(
        self,
        *,
        x: Tensor,
        relation_counts: Tensor,
        relation_args: Tensor,
        relation_arities: Tensor | Sequence[int] | None,
        relation_names: Sequence[str] | None,
        batch: Tensor | None,
        node_sizes: Tensor | None,
        object_indices: Tensor | Sequence[int] | None,
        object_sizes: Tensor | Sequence[int] | None,
        history_entity_indices: Tensor | Sequence[int] | None,
        history_entity_sizes: Tensor | Sequence[int] | None,
        history_entity_dt: Tensor | Sequence[int] | None,
        target_entity_indices: Tensor | Sequence[int] | None,
        target_entity_group_ids: Tensor | Sequence[int] | None,
        target_entity_sizes: Tensor | Sequence[int] | None,
        target_positions: Tensor | Sequence[int] | None,
        target_group_ids: Tensor | Sequence[int] | None,
        target_sizes: Tensor | Sequence[int] | None,
        target_indices: Tensor | Sequence[int] | None,
        target_candidate_ids: Tensor | Sequence[int] | None,
        cache: dict | None,
    ) -> _FlatPreparedBatch:
        """Build the internal prepared carrier from already-unpacked flat fields."""
        core = self._normalize_relation_core(
            x,
            relation_counts,
            relation_args,
            relation_arities,
            relation_names=relation_names,
        )
        batch_t, node_sizes_t = self._resolve_entity_batch(
            core.x,
            batch=batch,
            node_sizes=node_sizes,
        )
        views = self._normalize_output_views(
            device=core.x.device,
            index_dtype=core.index_dtype,
            batch=batch_t,
            node_sizes=node_sizes_t,
            object_indices=object_indices,
            object_sizes=object_sizes,
            history_entity_indices=history_entity_indices,
            history_entity_sizes=history_entity_sizes,
            history_entity_dt=history_entity_dt,
            target_entity_indices=target_entity_indices,
            target_entity_group_ids=target_entity_group_ids,
            target_entity_sizes=target_entity_sizes,
            target_positions=target_positions,
            target_group_ids=target_group_ids,
            target_sizes=target_sizes,
            target_indices=target_indices,
            target_candidate_ids=target_candidate_ids,
        )
        topology_store = cache.setdefault("topology_store", {}) if cache is not None else None
        topology = self.relational_layer.get_topology(
            core.relation_counts,
            relation_arities=core.relation_arities,
            cache=topology_store,
        )
        return _FlatPreparedBatch(
            x=core.x,
            relation_counts=core.relation_counts,
            relation_args=core.relation_args,
            relation_arities=core.relation_arities,
            batch=views.batch,
            node_sizes=views.node_sizes,
            object_indices=views.object_indices,
            object_sizes=views.object_sizes,
            history_entity_indices=views.history_entity_indices,
            history_entity_sizes=views.history_entity_sizes,
            history_entity_dt=views.history_entity_dt,
            target_entity_indices=views.target_entity_indices,
            target_entity_group_ids=views.target_entity_group_ids,
            target_entity_sizes=views.target_entity_sizes,
            target_positions=views.target_positions,
            target_group_ids=views.target_group_ids,
            target_sizes=views.target_sizes,
            target_indices=views.target_indices,
            target_candidate_ids=views.target_candidate_ids,
            topology=topology,
        )

    def _prepare_native_flat_batch(
        self,
        data: mifrost.BatchEncoding,
        *,
        cache: dict | None = None,
    ) -> _FlatPreparedBatch:
        """Prepare a native mifrost flat batch for repeated model execution."""
        x, relation_counts, relation_args, relation_arities, rest = self.unpack_native_flat(data)
        rest = dict(rest)
        return self._prepare_from_unpacked(
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

    def _prepare_pyg_flat_batch(
        self,
        data: pyg.data.Data | pyg.data.Batch,
        *,
        cache: dict | None = None,
    ) -> _FlatPreparedBatch:
        """Prepare an explicit PyG flat carrier for repeated model execution."""
        if not (hasattr(data, "relation_counts") and hasattr(data, "relation_args")):
            raise TypeError("PyG flat inputs must carry relation_counts and relation_args.")
        x, relation_counts, relation_args, relation_arities, rest = self.unpack(data)
        rest = dict(rest)
        return self._prepare_from_unpacked(
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

    def initialize_embeddings(self, x: Tensor) -> Tensor:
        """Create the initial entity embedding table of shape ``[num_nodes, embedding_size]``."""
        return torch.zeros((int(x.size(0)), self.embedding_size), device=x.device, dtype=x.dtype)

    def _prepare_batch(
        self,
        data: FlatBatchInput | _FlatPreparedBatch,
        cache: dict | None = None,
    ) -> _FlatPreparedBatch:
        if getattr(data, "_relm_flat_prepared_batch", False):
            return data
        if (
            isinstance(data, mifrost.BatchEncoding)
            and hasattr(data, "relation_counts")
            and hasattr(data, "relation_args")
        ):
            return self._prepare_native_flat_batch(data, cache=cache)
        if isinstance(data, (pyg.data.Data, pyg.data.Batch)):
            return self._prepare_pyg_flat_batch(data, cache=cache)
        raise TypeError(
            "FlatRelationalGNN expects a mifrost flat BatchEncoding or a PyG "
            "Data/Batch carrying relation_counts and relation_args."
        )

    def _build_output(
        self,
        entity_embeddings: Tensor,
        batch: Tensor,
        *,
        object_indices: Tensor | None = None,
        target_entity_indices: Tensor | None = None,
        target_positions: Tensor | None = None,
    ) -> FlatRelationalOutput:
        # The relational core always produces the full entity table. All other
        # views are encoder-defined subsets of that same table.
        object_embeddings = None
        object_batch = None
        if object_indices is not None:
            object_embeddings = entity_embeddings.index_select(0, object_indices)
            object_batch = batch.index_select(0, object_indices)

        target_entity_embeddings = None
        target_entity_batch = None
        if target_entity_indices is not None:
            target_entity_embeddings = entity_embeddings.index_select(0, target_entity_indices)
            target_entity_batch = batch.index_select(0, target_entity_indices)

        target_embeddings = None
        target_batch = None
        if target_positions is not None:
            target_embeddings = entity_embeddings.index_select(0, target_positions)
            target_batch = batch.index_select(0, target_positions)

        return FlatRelationalOutput(
            entity=entity_embeddings,
            entity_batch=batch,
            object=object_embeddings,
            object_batch=object_batch,
            target_entity=target_entity_embeddings,
            target_entity_batch=target_entity_batch,
            target=target_embeddings,
            target_batch=target_batch,
        )

    @optional_compile(enable_attr="_compile_forward", backend="inductor", dynamic=True)
    def _compute_entity_embeddings_prepared(
        self,
        prepared_batch: _FlatPreparedBatch,
        *,
        cache: dict | None = None,
    ) -> Tensor:
        """Run the recurrent flat relation core and return entity embeddings.

        Input:
            ``prepared_batch`` with packed relation tensors.
        Output:
            ``[num_nodes, embedding_size]`` entity embeddings after all relation
            layers and updater residuals.
        """
        layer_cache = cache.setdefault("relational_layer", {}) if cache is not None else {}
        entity_embeddings = self.initialize_embeddings(prepared_batch.x)
        for _ in range(self.num_layers):
            relation_msgs = self.relational_layer(
                entity_embeddings,
                prepared_batch.relation_counts,
                prepared_batch.relation_args,
                relation_arities=prepared_batch.relation_arities,
                topology=prepared_batch.topology,
                cache=layer_cache,
            )
            updated = self.embedding_updater(torch.cat([entity_embeddings, relation_msgs], dim=1))
            entity_embeddings = entity_embeddings + updated
        return entity_embeddings

    @optional_compile(enable_attr="_compile_forward", backend="inductor", dynamic=True)
    def _forward_prepared_batch(
        self,
        prepared_batch: _FlatPreparedBatch,
        *,
        cache: dict | None = None,
    ) -> FlatRelationalOutput:
        entity_embeddings = self._compute_entity_embeddings_prepared(prepared_batch, cache=cache)
        return self._build_output(
            entity_embeddings,
            prepared_batch.batch,
            object_indices=prepared_batch.object_indices,
            target_entity_indices=prepared_batch.target_entity_indices,
            target_positions=prepared_batch.target_positions,
        )

    @optional_compile(enable_attr="_compile_public_api", backend="inductor", dynamic=True)
    def compute_entity_embeddings(
        self,
        data: FlatBatchInput,
        cache: dict | None = None,
    ) -> Tensor:
        """Return entity embeddings for a public flat batch carrier.

        Public inputs stay on the eager adapter path. Only the prepared flat
        tensor core is eligible for ``torch.compile``, which avoids compiling
        carrier normalization and batch/view reconstruction.
        """
        prepared_batch = self._prepare_batch(data, cache=cache)
        return self._compute_entity_embeddings_prepared(prepared_batch, cache=cache)

    @optional_compile(enable_attr="_compile_public_api", backend="inductor", dynamic=True)
    def forward(
        self,
        data: FlatBatchInput,
        cache: dict | None = None,
    ) -> FlatRelationalOutput:
        """Run the flat model on a public batch carrier and build structured output.

        As with :meth:`compute_entity_embeddings`, carrier preparation remains eager
        while the prepared recurrent core may be compiled separately.
        """
        prepared_batch = self._prepare_batch(data, cache=cache)
        return self._forward_prepared_batch(prepared_batch, cache=cache)


__all__ = [
    "FlatRelationalGNN",
    "FlatRelationalOutput",
    "FlatExecutionPolicy",
    "FlatBatchInput",
]
