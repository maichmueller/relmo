from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver

from ._compile import optional_compile
from .aggr import LogSumExpAggregation
from .flat_relational_layer import (
    FlatRelationalLayer,
    FlatTopology,
    normalize_relation_arities,
    normalize_relation_counts,
)
from .mlp import ArityMLPFactory, SimpleMLP
from .pyg_module import PyGFlatModule

try:  # pragma: no cover - optional runtime dependency
    import mifrost  # type: ignore
except Exception:  # pragma: no cover - keep module importable without mifrost
    mifrost = None  # type: ignore

RelationDict = dict[str, int]
INT32_MAX = int(torch.iinfo(torch.int32).max)


@dataclass(frozen=True)
class FlatPreparedInputs:
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
    topology: FlatTopology | None = None
    _relm_flat_prepared: bool = True


def _normalize_optional_long_tensor(
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


def _preferred_index_dtype(*, device: torch.device, max_index_bound: int) -> torch.dtype:
    if device.type == "cpu" and int(max_index_bound) <= INT32_MAX:
        return torch.int32
    return torch.long


def _normalize_optional_index_tensor(
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


class FlatRelationalGNN(PyGFlatModule):
    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        relation_dict: RelationDict,
        relation_module_factory: Callable[[int], torch.nn.Module]
        | ArityMLPFactory
        | None = None,
        activation: str | Callable | None = None,
        compile_forward: bool = False,
        fused_two_layer_mish_execution: bool | None = None,
        fused_two_layer_pointwise_execution: bool | None = None,
        fused_relation_gather: bool | None = None,
    ) -> None:
        super().__init__()
        self._compile_forward = bool(compile_forward)
        self.embedding_size = int(embedding_size)
        self.num_layer = int(num_layer)
        self.relation_dict = {str(key): int(value) for key, value in relation_dict.items()}
        self.relation_names = tuple(self.relation_dict.keys())
        self.relation_arities = torch.as_tensor(
            tuple(int(v) for v in self.relation_dict.values()), dtype=torch.long
        )
        self.activation = activation or "mish"

        if relation_module_factory is None:
            relation_module_factory = ArityMLPFactory(
                feature_size=self.embedding_size,
                added_arity=0,
                residual=False,
                padding=None,
                layers=1,
                activation=self.activation,
            )
        relation_modules = [relation_module_factory(arity) for arity in self.relation_arities.tolist()]

        resolved_aggr: str | pyg.nn.aggr.Aggregation
        if isinstance(aggr, str) or aggr is None:
            if aggr is None or aggr.lower() == "logsumexp":
                resolved_aggr = LogSumExpAggregation()
            else:
                resolved_aggr = aggregation_resolver(aggr)
        else:
            resolved_aggr = aggr

        self.relational_layer = FlatRelationalLayer(
            update_modules=relation_modules,
            relation_names=self.relation_names,
            relation_arities=self.relation_arities,
            embedding_size=self.embedding_size,
            aggr=aggr if isinstance(aggr, str) else resolved_aggr,
            fused_two_layer_mish_execution=fused_two_layer_mish_execution,
            fused_two_layer_pointwise_execution=fused_two_layer_pointwise_execution,
            fused_relation_gather=fused_relation_gather,
        )
        self.embedding_updater = SimpleMLP(
            in_size=2 * self.embedding_size,
            embedding_size=2 * self.embedding_size,
            out_size=self.embedding_size,
            activation=self.activation,
        )

    def _normalize_inputs(
        self,
        x: Tensor,
        relation_counts: Tensor,
        relation_args: Tensor,
        relation_arities: Tensor | Sequence[int] | None = None,
        *,
        relation_names: Sequence[str] | None = None,
        batch: Tensor | None = None,
        node_sizes: Tensor | None = None,
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
    ) -> dict[str, Any]:
        if x.dim() != 2:
            raise ValueError(f"x must be rank-2, got shape {tuple(x.shape)}.")
        relation_counts_2d = normalize_relation_counts(relation_counts, device=x.device)
        index_dtype = _preferred_index_dtype(
            device=x.device,
            max_index_bound=int(x.size(0)),
        )
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
            raise ValueError(
                "flat model relation arity dimension does not match constructor relation_dict."
            )
        if not torch.equal(arities_1d.cpu(), self.relation_arities):
            raise ValueError(
                "input relation_arities does not match the model relation_dict ordering."
            )
        if relation_names is not None and tuple(str(name) for name in relation_names) != self.relation_names:
            raise ValueError("input relation_names does not match model relation_dict ordering.")

        node_sizes_t = (
            None
            if node_sizes is None
            else node_sizes.to(device=x.device, dtype=torch.long).view(-1)
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

        return {
            "relation_counts": relation_counts_2d,
            "relation_args": relation_args_1d,
            "relation_arities": arities_1d,
            "batch": batch,
            "node_sizes": node_sizes_t,
            "object_indices": _normalize_optional_index_tensor(
                object_indices,
                device=x.device,
                dtype=index_dtype,
            ),
            "object_sizes": _normalize_optional_long_tensor(object_sizes, device=x.device),
            "history_entity_indices": _normalize_optional_index_tensor(
                history_entity_indices,
                device=x.device,
                dtype=index_dtype,
            ),
            "history_entity_sizes": _normalize_optional_long_tensor(
                history_entity_sizes, device=x.device
            ),
            "history_entity_dt": _normalize_optional_long_tensor(
                history_entity_dt, device=x.device
            ),
            "target_entity_indices": _normalize_optional_index_tensor(
                target_entity_indices,
                device=x.device,
                dtype=index_dtype,
            ),
            "target_entity_group_ids": _normalize_optional_long_tensor(
                target_entity_group_ids, device=x.device
            ),
            "target_entity_sizes": _normalize_optional_long_tensor(
                target_entity_sizes, device=x.device
            ),
            "target_positions": _normalize_optional_index_tensor(
                target_positions,
                device=x.device,
                dtype=index_dtype,
            ),
            "target_group_ids": _normalize_optional_long_tensor(
                target_group_ids, device=x.device
            ),
            "target_sizes": _normalize_optional_long_tensor(target_sizes, device=x.device),
            "target_indices": _normalize_optional_long_tensor(target_indices, device=x.device),
            "target_candidate_ids": _normalize_optional_long_tensor(
                target_candidate_ids, device=x.device
            ),
        }

    def initialize_embeddings(self, x: Tensor) -> Tensor:
        return torch.zeros((int(x.size(0)), self.embedding_size), device=x.device, dtype=x.dtype)

    def prepare(
        self,
        data: Any,
        *args,
        cache: dict | None = None,
        **kwargs,
    ) -> FlatPreparedInputs:
        if getattr(data, "_relm_flat_prepared", False):
            return data
        if torch.is_tensor(data):
            relation_counts = args[0] if len(args) > 0 else kwargs.pop("relation_counts")
            relation_args = args[1] if len(args) > 1 else kwargs.pop("relation_args")
            relation_arities = args[2] if len(args) > 2 else kwargs.pop("relation_arities", None)
            rest = kwargs
            x = data
        elif isinstance(data, dict):
            x = data["x"]
            relation_counts = data["relation_counts"]
            relation_args = data["relation_args"]
            relation_arities = data.get("relation_arities", kwargs.pop("relation_arities", None))
            rest = {
                key: value
                for key, value in data.items()
                if key not in {"x", "relation_counts", "relation_args", "relation_arities"}
            }
            rest.update(kwargs)
        else:
            if (
                mifrost is not None
                and isinstance(data, mifrost.BatchEncoding)
                and hasattr(data, "relation_counts")
                and hasattr(data, "relation_args")
            ):
                x, relation_counts, relation_args, relation_arities, rest = self.unpack_native_flat(data)
            else:
                if hasattr(data, "as_pyg") and hasattr(data, "relation_counts") and hasattr(data, "relation_args") and not hasattr(data, "x"):
                    data = data.as_pyg(as_batch=True)
                x, relation_counts, relation_args, relation_arities, rest = self.unpack_attr(data)
            rest = dict(rest)
            rest.update(kwargs)

        normalized = self._normalize_inputs(
            x,
            relation_counts,
            relation_args,
            relation_arities,
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
        )
        topology_store = cache.setdefault("topology_store", {}) if cache is not None else None
        topology = self.relational_layer.get_topology(
            normalized["relation_counts"],
            relation_arities=normalized["relation_arities"],
            cache=topology_store,
        )
        return FlatPreparedInputs(
            x=x,
            relation_counts=normalized["relation_counts"],
            relation_args=normalized["relation_args"],
            relation_arities=normalized["relation_arities"],
            batch=normalized["batch"],
            node_sizes=normalized["node_sizes"],
            object_indices=normalized["object_indices"],
            object_sizes=normalized["object_sizes"],
            history_entity_indices=normalized["history_entity_indices"],
            history_entity_sizes=normalized["history_entity_sizes"],
            history_entity_dt=normalized["history_entity_dt"],
            target_entity_indices=normalized["target_entity_indices"],
            target_entity_group_ids=normalized["target_entity_group_ids"],
            target_entity_sizes=normalized["target_entity_sizes"],
            target_positions=normalized["target_positions"],
            target_group_ids=normalized["target_group_ids"],
            target_sizes=normalized["target_sizes"],
            target_indices=normalized["target_indices"],
            target_candidate_ids=normalized["target_candidate_ids"],
            topology=topology,
        )

    def _build_outputs(
        self,
        entity_embeddings: Tensor,
        batch: Tensor,
        *,
        object_indices: Tensor | None = None,
        target_entity_indices: Tensor | None = None,
        target_positions: Tensor | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        out = {"entity": entity_embeddings}
        out_batch = {"entity": batch}
        if object_indices is not None:
            out["object"] = entity_embeddings.index_select(0, object_indices)
            out_batch["object"] = batch.index_select(0, object_indices)
        if target_entity_indices is not None:
            out["target_entity"] = entity_embeddings.index_select(0, target_entity_indices)
            out_batch["target_entity"] = batch.index_select(0, target_entity_indices)
        if target_positions is not None:
            out["target"] = entity_embeddings.index_select(0, target_positions)
            out_batch["target"] = batch.index_select(0, target_positions)
        return out, out_batch

    @optional_compile(
        enable_attr="_compile_forward",
        backend="inductor",
        dynamic=True,
    )
    def forward_prepared_entity_embeddings(
        self,
        prepared: FlatPreparedInputs,
        *,
        cache: dict | None = None,
    ) -> Tensor:
        layer_cache = cache.setdefault("relational_layer", {}) if cache is not None else {}
        entity_embeddings = self.initialize_embeddings(prepared.x)
        for _ in range(self.num_layer):
            relation_msgs = self.relational_layer(
                entity_embeddings,
                prepared.relation_counts,
                prepared.relation_args,
                relation_arities=prepared.relation_arities,
                topology=prepared.topology,
                cache=layer_cache,
            )
            updated = self.embedding_updater(
                torch.cat([entity_embeddings, relation_msgs], dim=1)
            )
            entity_embeddings = entity_embeddings + updated
        return entity_embeddings

    @optional_compile(
        enable_attr="_compile_forward",
        backend="inductor",
        dynamic=True,
    )
    def forward_prepared(
        self,
        prepared: FlatPreparedInputs,
        *,
        cache: dict | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        entity_embeddings = self.forward_prepared_entity_embeddings(
            prepared,
            cache=cache,
        )

        return self._build_outputs(
            entity_embeddings,
            prepared.batch,
            object_indices=prepared.object_indices,
            target_entity_indices=prepared.target_entity_indices,
            target_positions=prepared.target_positions,
        )

    @optional_compile(
        enable_attr="_compile_forward",
        backend="inductor",
        dynamic=True,
    )
    def forward(
        self,
        x: Tensor,
        relation_counts: Tensor,
        relation_args: Tensor,
        relation_arities: Tensor | Sequence[int] | None = None,
        *,
        relation_names: Sequence[str] | None = None,
        batch: Tensor | None = None,
        node_sizes: Tensor | None = None,
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
        relation_sources: Sequence[str] | Tensor | None = None,
        cache: dict | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        del relation_sources
        prepared = self.prepare(
            x,
            relation_counts,
            relation_args,
            relation_arities,
            relation_names=relation_names,
            batch=batch,
            node_sizes=node_sizes,
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
            cache=cache,
        )
        return self.forward_prepared(prepared)
