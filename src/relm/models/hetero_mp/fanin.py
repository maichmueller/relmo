from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch_geometric as pyg
import torch_geometric.nn
from torch import Tensor
from torch_geometric.typing import Adj, EdgeType, OptPairTensor

from ..aggr import LogSumExpAggregation
from .routing import HeteroRouting


class FanInMP(HeteroRouting):
    """ """

    def __init__(
        self,
        embedding_size: int,
        dst_types: Iterable[str],
        aggr: str | torch_geometric.nn.Aggregation | None = None,
        *,
        src_types: Iterable[str] | None = None,
        edge_labels: Iterable[str] | None = None,
        **kwargs,
    ) -> None:
        aggr = aggr or LogSumExpAggregation()
        super().__init__(aggr, **kwargs)
        self.embedding_size = int(embedding_size)
        self.edge_labels = (
            tuple(str(lbl) for lbl in edge_labels)
            if (edge_labels is not None and not isinstance(edge_labels, str))
            else ((str(edge_labels),) if isinstance(edge_labels, str) else None)
        )
        self.src_types = (
            tuple(src_types)
            if (src_types is not None and not isinstance(src_types, str))
            else ((src_types,) if isinstance(src_types, str) else None)
        )
        self._label_mode = self.edge_labels is not None
        self.select = IdentityMP() if self._label_mode else SelectMP(self.embedding_size)
        self.dst_types = tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)

    def _matches(self, node_type: str, candidates: tuple[str, ...]) -> bool:
        if self.strict_filter_mode:
            return node_type in candidates
        return any(candidate in node_type for candidate in candidates)

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        src, rel, dst = edge_type
        if not self._matches(dst, self.dst_types):
            return False
        if self.edge_labels is not None and str(rel) not in self.edge_labels:
            return False
        if self.src_types is not None and not self._matches(src, self.src_types):
            return False
        return True

    def _internal_forward(self, x, edges_index, edge_type, **kwargs):
        if self._label_mode:
            return self.select(x, edges_index)
        return self.select(x, edges_index, int(edge_type[1]))

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        aggregated = {}
        for dst, values in out_dict.items():
            inputs, indices, dim_sizes = zip(*values)
            flat_inputs = torch.cat(inputs)
            flat_indices = torch.cat(indices)
            out = self.aggr(x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_sizes[0])
            aggregated[dst] = out
        return aggregated


class CentralFanInMP(HeteroRouting):
    """
    Centralized/batched fan-in aggregation (relations -> symbols).

    Builds one padded relation tensor [total_rel, max_arity, emb] and aggregates to each
    destination symbol type with a single index_select + gather + aggr call.
    """

    def __init__(
        self,
        embedding_size: int,
        dst_types: Iterable[str],
        relation_arities: Mapping[str, int],
        max_arity: int,
        aggr: str | torch_geometric.nn.Aggregation | None = None,
        **kwargs,
    ) -> None:
        aggr = aggr or LogSumExpAggregation()
        super().__init__(aggr, **kwargs)
        self.embedding_size = int(embedding_size)
        self.dst_types = tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)
        self.relation_arities = dict(relation_arities)
        self.max_arity = int(max_arity)

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        *_, dst = edge_type
        if self.strict_filter_mode:
            return dst in self.dst_types
        else:
            return any(dst_type in dst for dst_type in self.dst_types)

    def _internal_forward(self, x, edges_index, edge_type: EdgeType, **kwargs):
        raise NotImplementedError()

    def forward(self, x_dict, edge_index_dict, **kwargs) -> Dict[str, Tensor]:
        if not edge_index_dict:
            return {}

        # Predicate block sizes and offsets.
        sizes: dict[str, int] = {}
        for pred, arity in self.relation_arities.items():
            if arity <= 0:
                continue
            sizes[pred] = int(x_dict[pred].size(0)) if pred in x_dict else 0

        order = sorted(pred for pred, n in sizes.items() if n > 0)
        aggregated: Dict[str, Tensor] = {}
        if not order:
            for dst in self.dst_types:
                if dst in x_dict:
                    aggregated[dst] = x_dict[dst].new_zeros(
                        (int(x_dict[dst].size(0)), self.embedding_size)
                    )
            return aggregated

        offsets: dict[str, int] = {}
        total_rel = 0
        for pred in order:
            offsets[pred] = total_rel
            total_rel += sizes[pred]

        ref = x_dict[order[0]]
        rel_all = ref.new_zeros((total_rel, self.max_arity, self.embedding_size))
        for pred in order:
            n = sizes[pred]
            if n == 0:
                continue
            arity = int(self.relation_arities[pred])
            x = x_dict[pred]
            exp = arity * self.embedding_size
            if x.size(-1) != exp:
                raise ValueError(
                    f"Predicate {pred!r} has arity {arity}, but embedding dim is {x.size(-1)} (expected {exp})."
                )
            rel_all[offsets[pred] : offsets[pred] + n, :arity, :] = x.view(
                n, arity, self.embedding_size
            )

        # Accumulate per-dst edge lists.
        per_dst_src: dict[str, list[torch.Tensor]] = defaultdict(list)
        per_dst_dst: dict[str, list[torch.Tensor]] = defaultdict(list)
        per_dst_pos: dict[str, list[torch.Tensor]] = defaultdict(list)
        for edge_type in filter(self._accepts_edge, edge_index_dict.keys()):
            src, rel, dst = edge_type
            if src not in offsets:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index is None or edge_index.numel() == 0:
                continue
            pos = int(rel)
            arity = int(self.relation_arities.get(src, 0))
            if pos < 0 or pos >= arity:
                continue
            src_global = edge_index[0] + offsets[src]
            dst_idx = edge_index[1]
            if src_global.numel() == 0:
                continue
            per_dst_src[dst].append(src_global)
            per_dst_dst[dst].append(dst_idx)
            per_dst_pos[dst].append(
                torch.full_like(dst_idx, pos, dtype=torch.long, device=dst_idx.device)
            )

        for dst in self.dst_types:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            src_parts = per_dst_src.get(dst)
            if not src_parts:
                aggregated[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue
            src_global = src_parts[0] if len(src_parts) == 1 else torch.cat(src_parts, dim=0)
            dst_parts = per_dst_dst[dst]
            dst_index = dst_parts[0] if len(dst_parts) == 1 else torch.cat(dst_parts, dim=0)
            pos_parts = per_dst_pos[dst]
            pos_all = pos_parts[0] if len(pos_parts) == 1 else torch.cat(pos_parts, dim=0)

            gathered = rel_all.index_select(0, src_global)  # [E, max_arity, emb]
            pos_idx = pos_all.view(-1, 1, 1).expand(-1, 1, self.embedding_size)
            msgs = torch.gather(gathered, 1, pos_idx).squeeze(1)  # [E, emb]
            aggregated[dst] = self.aggr(x=msgs, index=dst_index, dim=0, dim_size=dim_size)

        return aggregated


class SelectMP(pyg.nn.MessagePassing):
    def __init__(
        self,
        embedding_size: int,
        aggr: Optional[str | List[str]] = "sum",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            aggr,
            aggr_kwargs=aggr_kwargs,
        )
        self.embedding_size = embedding_size

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x, position=position)

    def message(self, x_j: Tensor, position: int = None) -> Tensor:
        # Take the i-th hidden-number of elements from the last dimension
        # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
        # alternatively:
        #   split = torch.split(x_j, self.embedding_size, dim=-1)
        #   return split[position]
        sliced = x_j[..., position * self.embedding_size : (position + 1) * self.embedding_size]
        return sliced

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        return inputs, index, dim_size


class IdentityMP(pyg.nn.MessagePassing):
    """Message passing that forwards full source features without positional slicing."""

    def __init__(
        self,
        aggr: Optional[str | List[str]] = "sum",
        aggr_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(aggr, aggr_kwargs=aggr_kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        return inputs, index, dim_size


class LabelFanInMP(FanInMP):
    """Backward-compatible alias for label-filtered FanInMP."""

    def __init__(
        self,
        dst_types: Iterable[str],
        edge_labels: Iterable[str],
        *,
        src_types: Iterable[str] | None = None,
        embedding_size: int = 1,
        aggr: str | torch_geometric.nn.Aggregation | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            embedding_size=embedding_size,
            dst_types=dst_types,
            src_types=src_types,
            edge_labels=edge_labels,
            aggr=aggr,
            **kwargs,
        )


class LGANNNAggregator(FanInMP):
    def _internal_forward(self, x, edges_index, edge_type, **kwargs):
        # x is (`src_x`, `dst_x`) where `src` is relation and `dst` is symbol
        relation_x = x[0]
        # Pool relations embeddings to a single embedding_size vector (mean over arity)
        # relation_x shape: [num_atoms, arity * embedding_size]
        emb_size = self.select.embedding_size
        pooled_x = relation_x.view(relation_x.shape[0], -1, emb_size).mean(dim=1)
        # Aggregate to symbols. SelectMP with pos=0 on pooled_x will take the whole thing.
        return self.select(pooled_x, edges_index, 0)
