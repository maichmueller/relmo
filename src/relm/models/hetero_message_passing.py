from __future__ import annotations

import abc
import itertools
import os
import operator
from abc import ABC
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import torch
import torch_geometric as pyg
import torch_geometric.nn
from torch import Tensor
from torch_geometric.nn import Aggregation, SimpleConv
from torch_geometric.nn.aggr import SumAggregation
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType, OptPairTensor

from ._logging import get_logger
from .aggr import LogSumExpAggregation
from .mixins import DeviceAwareMixin
from .patched_module_dict import PatchedModuleDict
from ._misc import stream_context

try:  # pragma: no cover - optional during minimal model-only imports
    from ..ops import mp as relm_mp_ops
except Exception:  # pragma: no cover
    relm_mp_ops = None  # type: ignore[assignment]

# NOTE: The centralized and decentralized "batched/cached" MPs share the same patterns:
# - Build routing indices once per forward() (edge_index_dict is stable across num_layer iterations).
# - Reuse preallocated buffers across iterations to avoid allocation churn.
# These helpers factor out that boilerplate while keeping the hot-path tensor ops unchanged
# (index_select, index_copy_, one MLP call, Aggregation).

_MODE_SUM = 0
_MODE_LOGSUMEXP = 1
_BOOL_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _BOOL_FALSE


def _use_model_mp_ops(ref: torch.Tensor) -> bool:
    if relm_mp_ops is None:
        return False
    if not _env_bool("RELM_MODELS_MP_OPS", True):
        return False
    # Model-side custom op integration is tuned for CUDA execution.
    if ref.device.type != "cuda":
        return False
    return ref.dtype.is_floating_point


def _use_model_mp_fanin(ref: torch.Tensor) -> bool:
    return _use_model_mp_ops(ref) and _env_bool("RELM_MODELS_MP_FANIN", True)


def _use_model_mp_fanout(ref: torch.Tensor) -> bool:
    return _use_model_mp_ops(ref) and _env_bool("RELM_MODELS_MP_FANOUT", False)


def _resolve_fanin_mode(aggr: Aggregation) -> int | None:
    if isinstance(aggr, SumAggregation):
        return _MODE_SUM
    # Optional: enable custom logsumexp reduce for model MPs when desired.
    if isinstance(aggr, LogSumExpAggregation) and _env_bool(
        "RELM_MODELS_MP_LOGSUMEXP", False
    ):
        return _MODE_LOGSUMEXP
    return None


def _fanout_scatter_multi_src(
    *,
    by_src: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    x_dict: Mapping[str, Tensor],
    out_rows: int,
) -> torch.Tensor:
    x_parts: list[torch.Tensor] = []
    flat_parts: list[torch.Tensor] = []
    src_global_parts: list[torch.Tensor] = []
    offset = 0
    for src, (flat_dst, src_idx) in by_src.items():
        if src not in x_dict:
            raise KeyError(f"Missing src node type {src!r} in x_dict.")
        x_src = x_dict[src]
        x_parts.append(x_src)
        flat_parts.append(flat_dst)
        src_global_parts.append(src_idx + int(offset))
        offset += int(x_src.size(0))
    x_cat = _cat_or_single(x_parts, dim=0)
    flat_dst = _cat_or_single(flat_parts, dim=0)
    src_global = _cat_or_single(src_global_parts, dim=0)
    return relm_mp_ops.fanout_scatter(  # type: ignore[union-attr]
        x_cat, src_global, flat_dst, int(out_rows)
    )


def _build_fanout_scatter_plan(
    *,
    by_src: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    x_dict: Mapping[str, Tensor],
) -> dict[str, Any] | None:
    src_order: list[str] = []
    flat_parts: list[torch.Tensor] = []
    src_global_parts: list[torch.Tensor] = []
    src_rows: list[int] = []
    offset = 0
    for src in sorted(by_src.keys()):
        if src not in x_dict:
            continue
        x_src = x_dict[src]
        rows = int(x_src.size(0))
        flat_dst, src_idx = by_src[src]
        src_order.append(src)
        src_rows.append(rows)
        flat_parts.append(flat_dst)
        src_global_parts.append(src_idx + int(offset))
        offset += rows
    if not src_order:
        return None
    return {
        "src_order": tuple(src_order),
        "src_rows": tuple(src_rows),
        "x_rows": int(offset),
        "flat_dst": _cat_or_single(flat_parts, dim=0),
        "src_global": _cat_or_single(src_global_parts, dim=0),
    }


def _fanout_scatter_from_plan(
    *,
    plan: Mapping[str, Any],
    x_dict: Mapping[str, Tensor],
    out_rows: int,
) -> torch.Tensor:
    src_order = plan["src_order"]
    src_rows = plan["src_rows"]
    x_parts: list[torch.Tensor] = []
    for src, expected_rows in zip(src_order, src_rows):
        if src not in x_dict:
            raise KeyError(f"Missing src node type {src!r} in x_dict.")
        x_src = x_dict[src]
        if int(x_src.size(0)) != int(expected_rows):
            raise ValueError(
                f"Source size changed for {src!r}: expected {expected_rows}, got {int(x_src.size(0))}."
            )
        x_parts.append(x_src)
    x_cat = _cat_or_single(x_parts, dim=0)
    if int(x_cat.size(0)) != int(plan["x_rows"]):
        raise ValueError(
            f"Fanout plan x_rows mismatch: expected {plan['x_rows']}, got {int(x_cat.size(0))}."
        )
    return relm_mp_ops.fanout_scatter(  # type: ignore[union-attr]
        x_cat, plan["src_global"], plan["flat_dst"], int(out_rows)
    )


def _match_ntype(node_type: str, candidates: tuple[str, ...], strict: bool) -> bool:
    # Shared node-type filter semantics across MPs.
    if strict:
        return node_type in candidates
    return any(c in node_type for c in candidates)


def _cat_or_single(parts: list[torch.Tensor], *, dim: int = 0) -> torch.Tensor:
    # Avoid torch.cat when there is only a single tensor (common for small schemas).
    if not parts:
        raise ValueError("_cat_or_single got empty parts.")
    return parts[0] if len(parts) == 1 else torch.cat(parts, dim=dim)


def _finalize_pair_lists(
    lhs_lists: Mapping[str, list[torch.Tensor]],
    rhs_lists: Mapping[str, list[torch.Tensor]],
    *,
    dim: int = 0,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    # Turns per-key lists of tensors into per-key concatenated tensors.
    # Used only on cache-miss routing builds.
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for k, lhs in lhs_lists.items():
        rhs = rhs_lists.get(k)
        if not rhs:
            continue
        out[k] = (_cat_or_single(lhs, dim=dim), _cat_or_single(rhs, dim=dim))
    return out


def _get_or_make_buffer(
    buffers: dict,
    key: str,
    *,
    ref: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
    zero: bool = True,
) -> torch.Tensor:
    # Tiny buffer pool helper:
    # - allocate if missing or shape/device/dtype mismatch
    # - optionally zero-fill for deterministic semantics (e.g. padding slots)
    #
    # This is called in the per-iteration hot path, but the checks are cheap and replace
    # repeated open-coded allocation/zeroing logic.
    buf = buffers.get(key)
    dtype = dtype or ref.dtype
    if (
        buf is None
        or buf.device != ref.device
        or buf.dtype != dtype
        or tuple(buf.shape) != tuple(shape)
    ):
        buf = ref.new_empty(shape, dtype=dtype)
        buffers[key] = buf
        if zero:
            buf.zero_()
        return buf
    if zero:
        buf.zero_()
    return buf


class HeteroRouting(torch.nn.Module, ABC):
    """
    Handles heterogeneous message passing very similar to pyg.nn.HeteroConv.
    Instead of specifying a convolution for each EdgeType more generic rules can be used.
    """

    def __init__(
        self, aggr: Optional[str | Aggregation] = None, strict_filter_mode: bool = False
    ) -> None:
        super().__init__()
        if isinstance(aggr, str):
            try:
                self.aggr = aggregation_resolver(query=aggr)
            except ValueError:
                if aggr != "cat" and aggr != "stack":
                    get_logger(__name__).warning(
                        "Failed to resolve aggregation: " + aggr
                    )
                self.aggr = aggr
        else:
            self.aggr = aggr
        self.strict_filter_mode = strict_filter_mode

    @abc.abstractmethod
    def _accepts_edge(self, edge_type: EdgeType) -> bool: ...

    @abc.abstractmethod
    def _internal_forward(self, x, edges_index, edge_type: EdgeType, **kwargs): ...

    def forward(self, x_dict, edge_index_dict, **kwargs) -> Dict[str, Tensor]:
        """
        Apply message passing to each edge_index key if the edge-type is accepted.

        Calls the internal forward with a normal homogenous signature of x, edge_index

        :param x_dict: Dictionary with a feature matrix for each node type
        :param edge_index_dict: One edge_index adjacency matrix for each edge type.
        :return: Dictionary with each processed dst as key and their updated embedding as value.
        """
        out_dict: Dict[str, Any] = dict()
        for edge_type in filter(self._accepts_edge, edge_index_dict.keys()):
            src, rel, dst = edge_type
            if src == dst and src in x_dict:
                x = x_dict[src]
            elif src in x_dict or dst in x_dict:
                x = (
                    x_dict.get(src, None),
                    x_dict.get(dst, None),
                )
            else:
                raise KeyError(
                    f"Neither src ({src}) nor destination ({dst}) found in x_dict ({x_dict})"
                )
            out = self._internal_forward(
                x, edge_index_dict[edge_type], edge_type, **kwargs
            )
            out_dict.setdefault(dst, []).append(out)
        return self._group_output(out_dict, **kwargs)

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        aggregated: Dict[str, Tensor] = {}
        for key, value in out_dict.items():
            # `hetero_conv.group` does not yet support Aggregation modules
            if isinstance(self.aggr, Aggregation):
                out = torch.stack(value, dim=0)
                out = self.aggr(out, dim=0).squeeze(0)
            else:
                out = group(value, self.aggr)
            aggregated[key] = out
        return aggregated


class FanOutMP(DeviceAwareMixin, HeteroRouting):
    """
    Perform the 'fanout' phase of message passing in a heterogeneous STRIPS-based graph (batch).

    Fanout refers to the number of outgoing edges of a node in the context of message passing.
    While this module can be used with generic relationships, we describe it in the STRIPS-graph case,
    the fanout refers to the first step of relational-message passing:
    Object-nodes pass their embeddings to the connected atom-nodes.

    We refer to this step also as the message-creation step of a Relational Graph Neural Network,
    since atom-embeddings store the created-messages with which object-nodes will be updated.

    Accepts `EdgeType`s whose attr `src` matches the parameter `src_type`.
    Processes the incoming edges by:
        1. For each destination, i.e. predicate, concatenate all incoming (object-)embeddings.
        2. Apply the destination specific Module to the concatenated embeddings.
        3. Save the new embedding under the destination key.

    FanOut should be aggregation free in theory.
    Every atom receives only as many messages as the arity of its predicate.

    :param update_modules: Dict, maps destination node-types to a Module (e.g. MLP) to compute messages with.
        Each Module input-and output-tensor needs to match the degree of incoming edges in shape at dim 0.
    :param src_types: The node-type whose outgoing edges should be accepted.
    """

    def __init__(
        self,
        update_modules: Dict[str, torch.nn.Module],
        src_types: Iterable[str],
        use_cuda_streams: bool = False,
        **kwargs,
    ) -> None:
        """ """
        super().__init__(**kwargs)
        self.update_modules = PatchedModuleDict(update_modules)
        self.use_cuda_streams = torch.cuda.is_available() and use_cuda_streams
        self._cuda_streams = None
        self._cuda_pool = None
        # simple conv merely groups the messages of source nodes to target nodes without modifying them.
        self.group_incoming_features = SimpleConv()
        self.src_types = (
            tuple(src_types) if not isinstance(src_types, str) else (src_types,)
        )

    @property
    def cuda_streams(self):
        if (
            self.use_cuda_streams
            and self._cuda_streams is None
            and self.device.type == "cuda"
        ):
            self._cuda_streams = [
                torch.cuda.Stream(self.device) for _ in range(len(self.update_modules))
            ]
            self._cuda_pool = itertools.cycle(self._cuda_streams)
        return self._cuda_streams

    def next_stream(self):
        if not self.use_cuda_streams or self.device.type != "cuda":
            return None
        assert self.cuda_streams
        return next(self._cuda_pool)

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        src, *_ = edge_type
        if self.strict_filter_mode:
            return src in self.src_types
        else:
            return any(src_type in src for src_type in self.src_types)

    def _internal_forward(self, x, edge_index, edge_type: EdgeType, **kwargs):
        position = int(edge_type[1])
        out = self.group_incoming_features(x, edge_index)
        return position, out

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        grouped = dict()
        for predicate, value in out_dict.items():
            sorted_out = sorted(value, key=operator.itemgetter(0))
            stacked = torch.cat(tuple(out for _, out in sorted_out), dim=1)
            update_module = self.update_modules[predicate]
            if (stream := self.next_stream()) is not None:
                with stream_context(stream):
                    grouped[predicate] = update_module(stacked)
            else:
                grouped[predicate] = update_module(stacked)
        self._sync_streams()
        return grouped

    def _sync_streams(self):
        if self.use_cuda_streams and (cuda_streams := self.cuda_streams) is not None:
            for stream in cuda_streams:
                stream.synchronize()


class ConditionalFanOutMP(FanOutMP):
    def _group_output(
        self, out_dict: Dict[str, List], condition: Tensor = None, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Group the output by predicate and apply the update module to each predicate's messages.

        Also requires a condition tensor that is concatenated to the messages at the end.
        """
        grouped = dict()
        for predicate, value in out_dict.items():
            sorted_out = sorted(value, key=operator.itemgetter(0))
            stacked = torch.cat(
                tuple(itertools.chain((out for _, out in sorted_out), condition)), dim=1
            )
            update_module = self.update_modules[predicate]
            with stream_context(self.next_stream()):
                grouped[predicate] = update_module(stacked)
        self._sync_streams()
        return grouped


class CentralFanOutMP(DeviceAwareMixin, HeteroRouting):
    """
    Fan-out that batches all predicate messages into a single central module call.
    """

    def __init__(
        self,
        central_module: torch.nn.Module,
        condition_embedding: torch.nn.Embedding,
        relation_condition_index: Dict[str, int],
        relation_arities: Dict[str, int],
        max_arity: int,
        embedding_size: int,
        condition_position: str,
        src_types: Iterable[str],
        include_slot_mask: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if condition_position not in ("pre", "post"):
            raise ValueError(
                f"condition_position must be 'pre' or 'post', got {condition_position!r}."
            )
        self.central_module = central_module
        self.condition_embedding = condition_embedding
        self.relation_condition_index = dict(relation_condition_index)
        self.relation_arities = dict(relation_arities)
        self.max_arity = max_arity
        self.embedding_size = embedding_size
        self.condition_position = condition_position
        self.include_slot_mask = bool(include_slot_mask)
        self.group_incoming_features = SimpleConv()
        self.src_types = (
            tuple(src_types) if not isinstance(src_types, str) else (src_types,)
        )
        for predicate in self.relation_arities:
            if predicate not in self.relation_condition_index:
                raise KeyError(
                    f"Missing condition index for predicate {predicate!r} in relation_condition_index."
                )
            arity = int(self.relation_arities[predicate])
            if arity < 0:
                raise ValueError(f"Arity must be >= 0, got {arity} for {predicate!r}.")
            if arity > self.max_arity:
                raise ValueError(
                    f"Predicate {predicate!r} has arity {arity}, but max_arity is {self.max_arity}."
                )
        max_condition_idx = max(self.relation_condition_index.values(), default=-1)
        slot_mask_table = torch.zeros(
            (max_condition_idx + 1, self.max_arity),
            dtype=self.condition_embedding.weight.dtype,
            device=self.condition_embedding.weight.device,
        )
        for predicate, arity in self.relation_arities.items():
            condition_idx = self.relation_condition_index[predicate]
            slot_mask_table[condition_idx, :arity] = 1.0
        self.register_buffer("_slot_mask_table", slot_mask_table, persistent=False)

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        src, *_ = edge_type
        if self.strict_filter_mode:
            return src in self.src_types
        else:
            return any(src_type in src for src_type in self.src_types)

    def _internal_forward(self, x, edge_index, edge_type: EdgeType, **kwargs):
        position = int(edge_type[1])
        out = self.group_incoming_features(x, edge_index)
        return position, out

    def forward(self, x_dict, edge_index_dict, **kwargs) -> Dict[str, Tensor]:
        """
        Batched fan-out gather (Proposal B):
        - Build a global args tensor [total_relation_nodes * max_arity, embedding_size]
          and fill it via batched index_select + index_copy_ grouped by src type.
        - Assemble central module input via slice writes (no per-predicate cat/pad).

        This bypasses HeteroRouting.forward() to avoid per-edge-type SimpleConv() calls.
        """
        if not edge_index_dict:
            return {}

        # Determine which predicate node-types exist and their counts.
        sizes: dict[str, int] = {}
        for predicate in self.relation_arities:
            if predicate in x_dict:
                sizes[predicate] = int(x_dict[predicate].size(0))
            else:
                sizes[predicate] = 0

        # Fallback: infer missing predicate counts from edge indices.
        for edge_type in filter(self._accepts_edge, edge_index_dict.keys()):
            src, rel, dst = edge_type
            if dst not in sizes or sizes[dst] > 0:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue
            sizes[dst] = int(edge_index[1].max().item()) + 1

        order = sorted(predicate for predicate, n in sizes.items() if n > 0)
        if not order:
            return {}

        offsets: dict[str, int] = {}
        total_relation_nodes = 0
        for predicate in order:
            offsets[predicate] = total_relation_nodes
            total_relation_nodes += sizes[predicate]

        # Find a reference symbol embedding tensor for dtype/device allocation.
        ref: torch.Tensor | None = None
        for key, val in x_dict.items():
            if not torch.is_tensor(val) or val.dim() != 2:
                continue
            if self.strict_filter_mode:
                if key in self.src_types:
                    ref = val
                    break
            else:
                if any(src_type in key for src_type in self.src_types):
                    ref = val
                    break
        if ref is None:
            ref = next(iter(x_dict.values()))
        if ref.size(-1) != self.embedding_size:
            raise ValueError(
                f"Expected symbol embeddings with last dim {self.embedding_size}, got {ref.size(-1)}."
            )

        expected = self.max_arity * self.embedding_size
        cond_dim = int(self.condition_embedding.weight.size(-1))
        mask_dim = self.max_arity if self.include_slot_mask else 0
        in_dim = cond_dim + mask_dim + expected

        # Flat args buffer: [total_relation_nodes * max_arity, embedding_size]
        # Initialized to zeros; missing slots remain 0 (same semantics as padding).
        args_flat = ref.new_zeros(
            (total_relation_nodes * self.max_arity, self.embedding_size)
        )

        flat_dst_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)
        src_idx_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)

        for edge_type in filter(self._accepts_edge, edge_index_dict.keys()):
            src, rel, dst = edge_type
            if dst not in offsets:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue
            pos = int(rel)
            if pos < 0 or pos >= self.max_arity:
                continue
            if src not in x_dict:
                raise KeyError(
                    f"Missing src node type {src!r} in x_dict for edge {edge_type!r}."
                )
            src_x = x_dict[src]
            if src_x.size(-1) != self.embedding_size:
                raise ValueError(
                    f"Expected src embeddings with last dim {self.embedding_size}, got {src_x.size(-1)} for {src!r}."
                )
            src_idx = edge_index[0]
            dst_idx = edge_index[1] + offsets[dst]
            flat_dst = dst_idx * self.max_arity + pos
            flat_dst_by_src[src].append(flat_dst)
            src_idx_by_src[src].append(src_idx)

        for src, flat_dsts in flat_dst_by_src.items():
            flat_dst = (
                flat_dsts[0] if len(flat_dsts) == 1 else torch.cat(flat_dsts, dim=0)
            )
            src_idx_list = src_idx_by_src[src]
            src_idx = (
                src_idx_list[0]
                if len(src_idx_list) == 1
                else torch.cat(src_idx_list, dim=0)
            )
            vals = x_dict[src].index_select(0, src_idx)
            # NOTE: duplicates (dst,slot) should not exist; if they do, later writes win.
            args_flat.index_copy_(0, flat_dst, vals)

        args_feat = args_flat.view(total_relation_nodes, expected)
        all_inputs = ref.new_empty((total_relation_nodes, in_dim))

        # Fill per predicate slice with its condition + optional slot mask, then args.
        for predicate in order:
            start = offsets[predicate]
            n = sizes[predicate]
            if n == 0:
                continue
            sl = all_inputs[start : start + n]
            cond_idx = self.relation_condition_index[predicate]
            cond = self.condition_embedding.weight[cond_idx]
            if cond.device != sl.device or cond.dtype != sl.dtype:
                cond = cond.to(device=sl.device, dtype=sl.dtype)
            if self.include_slot_mask:
                slot_mask_row = self._slot_mask_table[cond_idx]
                if slot_mask_row.device != sl.device or slot_mask_row.dtype != sl.dtype:
                    slot_mask_row = slot_mask_row.to(device=sl.device, dtype=sl.dtype)

            if self.condition_position == "pre":
                sl[:, 0:cond_dim] = cond
                if self.include_slot_mask:
                    sl[:, cond_dim : cond_dim + mask_dim] = slot_mask_row
                sl[:, cond_dim + mask_dim : cond_dim + mask_dim + expected] = args_feat[
                    start : start + n
                ]
            else:
                sl[:, 0:expected] = args_feat[start : start + n]
                if self.include_slot_mask:
                    sl[:, expected : expected + mask_dim] = slot_mask_row
                sl[:, expected + mask_dim : expected + mask_dim + cond_dim] = cond

        all_outputs = self.central_module(all_inputs)
        grouped: Dict[str, Tensor] = {}
        for predicate in order:
            start = offsets[predicate]
            n = sizes[predicate]
            if n == 0:
                continue
            arity = self.relation_arities[predicate]
            grouped[predicate] = all_outputs[
                start : start + n, : arity * self.embedding_size
            ]
        return grouped

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        if not out_dict:
            return {}
        order = sorted(out_dict.keys())
        batched_inputs = []
        split_sizes = []
        meta = []
        expected = self.max_arity * self.embedding_size
        cond_dim = int(self.condition_embedding.weight.size(-1))
        mask_dim = self.max_arity if self.include_slot_mask else 0
        for predicate in order:
            value = out_dict[predicate]
            arity = self.relation_arities[predicate]
            if arity < 0:
                raise ValueError(f"Arity must be >= 0, got {arity} for {predicate!r}.")
            if arity > self.max_arity:
                raise ValueError(
                    f"Predicate {predicate!r} has arity {arity}, but max_arity is {self.max_arity}."
                )
            if not value:
                continue
            n = value[0][1].size(0)
            # Build the full central-module input without repeated cat() calls.
            # Layout matches the previous implementation:
            # - condition_position == 'pre' : [cond, (slot_mask), args...]
            # - condition_position == 'post': [args..., (slot_mask), cond]
            in_dim = cond_dim + mask_dim + expected
            full = value[0][1].new_empty((n, in_dim))
            cond_idx = self.relation_condition_index[predicate]
            cond = self.condition_embedding.weight[cond_idx]
            if cond.device != full.device or cond.dtype != full.dtype:
                cond = cond.to(device=full.device, dtype=full.dtype)

            if self.include_slot_mask:
                slot_mask_row = self._slot_mask_table[cond_idx]
                if (
                    slot_mask_row.device != full.device
                    or slot_mask_row.dtype != full.dtype
                ):
                    slot_mask_row = slot_mask_row.to(
                        device=full.device, dtype=full.dtype
                    )

            if self.condition_position == "pre":
                cond_base = 0
                args_base = cond_dim + mask_dim
                full[:, cond_base : cond_base + cond_dim] = cond
                if self.include_slot_mask:
                    full[:, cond_dim : cond_dim + mask_dim] = slot_mask_row
            else:
                args_base = 0
                cond_base = expected + mask_dim
                full[:, cond_base : cond_base + cond_dim] = cond
                if self.include_slot_mask:
                    full[:, expected : expected + mask_dim] = slot_mask_row

            # Fill argument slots. Ensure every position [0..arity-1] is present.
            positions = [pos for pos, _ in value]
            if len(positions) != arity or set(positions) != set(range(arity)):
                raise ValueError(
                    f"Expected positions 0..{arity - 1} for predicate {predicate!r}, "
                    f"got {sorted(set(positions))}."
                )
            for pos, out in value:
                if out.size(0) != n:
                    raise ValueError(
                        f"Predicate {predicate!r}: inconsistent row counts in fanout outputs."
                    )
                start = args_base + pos * self.embedding_size
                end = start + self.embedding_size
                full[:, start:end] = out
            # Zero-pad remaining slots (arity < max_arity) to avoid garbage values.
            used = arity * self.embedding_size
            if used < expected:
                full[:, args_base + used : args_base + expected].zero_()

            batched_inputs.append(full)
            split_sizes.append(n)
            meta.append((predicate, arity))

        all_inputs = torch.cat(batched_inputs, dim=0)
        all_outputs = self.central_module(all_inputs)
        outputs = torch.split(all_outputs, split_sizes, dim=0)
        grouped: Dict[str, Tensor] = {}
        for (predicate, arity), pred_out in zip(meta, outputs):
            grouped[predicate] = pred_out[:, : arity * self.embedding_size]
        return grouped


class CentralFusedLayerMP(torch.nn.Module):
    """
    Fused centralized relational layer:
    - batched fanout gather (symbols -> relation slots)
    - one central module call
    - batched fanin aggregation (relation slots -> symbols)

    Returns (atom_msgs, symbol_msgs).
    """

    def __init__(
        self,
        *,
        central_module: torch.nn.Module,
        condition_embedding: torch.nn.Embedding,
        relation_condition_index: Dict[str, int],
        relation_arities: Mapping[str, int],
        max_arity: int,
        embedding_size: int,
        condition_position: str,
        include_slot_mask: bool,
        symbol_type_ids: Iterable[str] | str,
        dst_symbol_type_ids: Iterable[str] | str,
        aggr: Aggregation | str | None,
        strict_filter_mode: bool = False,
        validate_routing: bool = False,
    ) -> None:
        super().__init__()
        if condition_position not in ("pre", "post"):
            raise ValueError(
                f"condition_position must be 'pre' or 'post', got {condition_position!r}."
            )
        if isinstance(aggr, str) or aggr is None:
            resolved = aggregation_resolver(query=aggr or "logsumexp")
            if not isinstance(resolved, Aggregation):
                raise ValueError(
                    "CentralFusedLayerMP requires a PyG Aggregation module, "
                    f"got {type(resolved)!r} from query={aggr!r}."
                )
            aggr = resolved
        if not isinstance(aggr, Aggregation):
            raise ValueError(
                f"CentralFusedLayerMP requires a PyG Aggregation module, got {type(aggr)!r}."
            )

        self.central_module = central_module
        self.condition_embedding = condition_embedding
        self.relation_condition_index = dict(relation_condition_index)
        self.relation_arities = dict(relation_arities)
        self.max_arity = int(max_arity)
        self.embedding_size = int(embedding_size)
        self.condition_position = condition_position
        self.include_slot_mask = bool(include_slot_mask)
        self.strict_filter_mode = bool(strict_filter_mode)
        self.validate_routing = bool(validate_routing)

        self.symbol_type_ids = (
            (symbol_type_ids,)
            if isinstance(symbol_type_ids, str)
            else tuple(symbol_type_ids)
        )
        self.dst_symbol_type_ids = (
            (dst_symbol_type_ids,)
            if isinstance(dst_symbol_type_ids, str)
            else tuple(dst_symbol_type_ids)
        )
        self.aggr = aggr
        self._mp_fanin_mode = _resolve_fanin_mode(self.aggr)

        for predicate in self.relation_arities:
            if predicate not in self.relation_condition_index:
                raise KeyError(
                    f"Missing condition index for predicate {predicate!r} in relation_condition_index."
                )
            arity = int(self.relation_arities[predicate])
            if arity < 0:
                raise ValueError(f"Arity must be >= 0, got {arity} for {predicate!r}.")
            if arity > self.max_arity:
                raise ValueError(
                    f"Predicate {predicate!r} has arity {arity}, but max_arity is {self.max_arity}."
                )

        max_condition_idx = max(self.relation_condition_index.values(), default=-1)
        slot_mask_table = torch.zeros(
            (max_condition_idx + 1, self.max_arity),
            dtype=self.condition_embedding.weight.dtype,
            device=self.condition_embedding.weight.device,
        )
        for predicate, arity in self.relation_arities.items():
            condition_idx = self.relation_condition_index[predicate]
            slot_mask_table[condition_idx, : int(arity)] = 1.0
        self.register_buffer("_slot_mask_table", slot_mask_table, persistent=False)

    def _match_any(self, node_type: str, candidates: tuple[str, ...]) -> bool:
        return _match_ntype(node_type, candidates, self.strict_filter_mode)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *,
        cache: dict | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if not edge_index_dict:
            return {}, {}

        cache = cache if cache is not None else {}

        # Routing (indices for index_copy_/index_select) is built once per forward and cached.
        # For CentralizedRelationalGNN this cache persists across all num_layer iterations.
        routing = cache.get("routing")
        if routing is None:
            sizes: dict[str, int] = {}
            for predicate in self.relation_arities:
                sizes[predicate] = (
                    int(x_dict[predicate].size(0)) if predicate in x_dict else 0
                )

            # Fallback: infer missing predicate counts from fanout edges.
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                if dst not in sizes or sizes[dst] > 0:
                    continue
                if dst not in self.relation_arities:
                    continue
                if not self._match_any(src, self.symbol_type_ids):
                    continue
                if edge_index is None or edge_index.numel() == 0:
                    continue
                sizes[dst] = int(edge_index[1].max().item()) + 1

            order = sorted(predicate for predicate, n in sizes.items() if n > 0)
            offsets: dict[str, int] = {}
            total_relation_nodes = 0
            for predicate in order:
                offsets[predicate] = total_relation_nodes
                total_relation_nodes += sizes[predicate]

            # Precompute edge routing (fanout + fanin) for this batch.
            flat_dst_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)
            src_idx_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)

            per_dst_flat_src: dict[str, list[torch.Tensor]] = defaultdict(list)
            per_dst_dst: dict[str, list[torch.Tensor]] = defaultdict(list)

            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                if edge_index is None or edge_index.numel() == 0:
                    continue
                pos = int(rel)

                # Fanout: symbol -> predicate slot
                if dst in offsets and self._match_any(src, self.symbol_type_ids):
                    if 0 <= pos < self.max_arity:
                        src_idx = edge_index[0]
                        dst_global = edge_index[1] + offsets[dst]
                        flat_dst = dst_global * self.max_arity + pos
                        flat_dst_by_src[src].append(flat_dst)
                        src_idx_by_src[src].append(src_idx)

                # Fanin: predicate slot -> symbol
                if src in offsets and self._match_any(dst, self.dst_symbol_type_ids):
                    arity = int(self.relation_arities.get(src, 0))
                    if 0 <= pos < arity:
                        src_global = edge_index[0] + offsets[src]
                        flat_src = src_global * self.max_arity + pos
                        per_dst_flat_src[dst].append(flat_src)
                        per_dst_dst[dst].append(edge_index[1])
                    elif self.validate_routing and arity > 0:
                        raise AssertionError(
                            f"Fanin routing pos out of range: pred={src!r} pos={pos} arity={arity}."
                        )

            fanout_by_src = _finalize_pair_lists(flat_dst_by_src, src_idx_by_src)
            if self.validate_routing:
                for src, (flat_dst, _src_idx) in fanout_by_src.items():
                    if int(flat_dst.unique().numel()) != int(flat_dst.numel()):
                        raise AssertionError(
                            f"Fanout routing duplicates detected for src={src!r}."
                        )
            fanout_plan = _build_fanout_scatter_plan(
                by_src=fanout_by_src,
                x_dict=x_dict,
            )

            fanin_by_dst = _finalize_pair_lists(per_dst_flat_src, per_dst_dst)

            pred_meta = []
            for predicate in order:
                n = sizes[predicate]
                if n <= 0:
                    continue
                pred_meta.append(
                    (
                        predicate,
                        offsets[predicate],
                        n,
                        int(self.relation_arities[predicate]),
                        int(self.relation_condition_index[predicate]),
                    )
                )

            routing = {
                "sizes": sizes,
                "order": order,
                "offsets": offsets,
                "total_relation_nodes": total_relation_nodes,
                "fanout_by_src": fanout_by_src,
                "fanout_plan": fanout_plan,
                "fanin_by_dst": fanin_by_dst,
                "pred_meta": pred_meta,
            }
            cache["routing"] = routing

        order = routing["order"]
        total_relation_nodes = int(routing["total_relation_nodes"])
        if total_relation_nodes <= 0 or not order:
            symbol_msgs: dict[str, Tensor] = {}
            for dst in self.dst_symbol_type_ids:
                if dst in x_dict:
                    symbol_msgs[dst] = x_dict[dst].new_zeros(
                        (int(x_dict[dst].size(0)), self.embedding_size)
                    )
            return {}, symbol_msgs

        # Reference tensor for dtype/device.
        ref: torch.Tensor | None = None
        for k in self.symbol_type_ids:
            if k in x_dict:
                ref = x_dict[k]
                break
        if ref is None:
            for key, val in x_dict.items():
                if not torch.is_tensor(val) or val.dim() != 2:
                    continue
                if (
                    self._match_any(key, self.symbol_type_ids)
                    and int(val.size(-1)) == self.embedding_size
                ):
                    ref = val
                    break
        if ref is None:
            ref = next(iter(x_dict.values()))
        if int(ref.size(-1)) != self.embedding_size:
            raise ValueError(
                f"Expected symbol embeddings with last dim {self.embedding_size}, got {int(ref.size(-1))}."
            )

        # IMPORTANT (autograd correctness):
        # We must NOT reuse and overwrite the same storage across message-passing iterations when
        # gradients are enabled, because autograd may save these tensors for backward. Reusing the
        # same buffer and writing into it again would trigger "modified by an inplace operation"
        # errors during backward (version counter mismatch).
        # Only reuse buffers in inference/no-grad. In training (and under torch.compile/AOTAutograd),
        # the forward may execute with grad mode disabled but still requires saving intermediates for
        # parameter gradients. Buffer reuse across iterations would overwrite saved tensors.
        reuse_buffers = (not self.training) and (not torch.is_grad_enabled())

        # Buffers are reused across iterations only when gradients are disabled (profiling/inference).
        # Fanout gather: args_flat[rel_global * max_arity + pos] = x_sym[src_idx]
        buffers = cache.setdefault("buffers", {})
        if reuse_buffers:
            args_flat = _get_or_make_buffer(
                buffers,
                "args_flat",
                ref=ref,
                shape=(total_relation_nodes * self.max_arity, self.embedding_size),
                zero=True,
            )
        else:
            args_flat = ref.new_zeros(
                (total_relation_nodes * self.max_arity, self.embedding_size)
            )

        fanout_by_src: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing[
            "fanout_by_src"
        ]
        if fanout_by_src and _use_model_mp_fanout(ref):
            fanout_plan = routing.get("fanout_plan")
            if fanout_plan is not None:
                args_flat = _fanout_scatter_from_plan(
                    plan=fanout_plan,
                    x_dict=x_dict,
                    out_rows=(total_relation_nodes * self.max_arity),
                )
            else:
                args_flat = _fanout_scatter_multi_src(
                    by_src=fanout_by_src,
                    x_dict=x_dict,
                    out_rows=(total_relation_nodes * self.max_arity),
                )
        else:
            for src, (flat_dst, src_idxs) in fanout_by_src.items():
                if src not in x_dict:
                    raise KeyError(
                        f"Missing src node type {src!r} in x_dict for fused fanout."
                    )
                vals = x_dict[src].index_select(0, src_idxs)
                args_flat.index_copy_(0, flat_dst, vals)

        args_feat = args_flat.reshape(
            total_relation_nodes, self.max_arity * self.embedding_size
        )

        # Assemble central inputs.
        pred_meta = routing["pred_meta"]
        expected = self.max_arity * self.embedding_size
        cond_dim = int(self.condition_embedding.weight.size(-1))
        mask_dim = self.max_arity if self.include_slot_mask else 0
        context_cols = cond_dim + mask_dim
        in_dim = context_cols + expected

        context = buffers.get("all_inputs_context")
        if (
            context is None
            or context.device != args_feat.device
            or context.dtype != args_feat.dtype
            or tuple(context.shape) != (total_relation_nodes, context_cols)
        ):
            context = args_feat.new_zeros((total_relation_nodes, context_cols))
            for _predicate, start, n, _arity, cond_idx in pred_meta:
                cond = self.condition_embedding.weight[cond_idx] if cond_dim > 0 else None
                if cond is not None and (
                    cond.device != context.device or cond.dtype != context.dtype
                ):
                    cond = cond.to(device=context.device, dtype=context.dtype)

                slot_mask_row = None
                if self.include_slot_mask:
                    slot_mask_row = self._slot_mask_table[cond_idx]
                    if (
                        slot_mask_row.device != context.device
                        or slot_mask_row.dtype != context.dtype
                    ):
                        slot_mask_row = slot_mask_row.to(
                            device=context.device, dtype=context.dtype
                        )

                sl = context[start : start + n]
                if self.condition_position == "pre":
                    if cond is not None:
                        sl[:, 0:cond_dim] = cond
                    if self.include_slot_mask:
                        assert slot_mask_row is not None
                        sl[:, cond_dim : cond_dim + mask_dim] = slot_mask_row
                else:
                    if self.include_slot_mask:
                        assert slot_mask_row is not None
                        sl[:, 0:mask_dim] = slot_mask_row
                    if cond is not None:
                        sl[:, mask_dim : mask_dim + cond_dim] = cond
            buffers["all_inputs_context"] = context

        if reuse_buffers:
            all_inputs = _get_or_make_buffer(
                buffers,
                "all_inputs",
                ref=args_feat,
                shape=(total_relation_nodes, in_dim),
                zero=False,
            )
        else:
            all_inputs = args_feat.new_empty((total_relation_nodes, in_dim))

        if self.condition_position == "pre":
            if context_cols > 0:
                all_inputs[:, 0:context_cols] = context
            all_inputs[:, context_cols : context_cols + expected] = args_feat
        else:
            all_inputs[:, 0:expected] = args_feat
            if context_cols > 0:
                all_inputs[:, expected : expected + context_cols] = context

        all_outputs = self.central_module(all_inputs)

        atom_msgs: dict[str, Tensor] = {}
        for predicate, start, n, arity, _cond_idx in pred_meta:
            atom_msgs[predicate] = all_outputs[
                start : start + n, : arity * self.embedding_size
            ]

        # Fanin aggregation with flat indexing (no gather):
        out_flat = all_outputs.reshape(
            total_relation_nodes * self.max_arity, self.embedding_size
        )

        symbol_msgs: dict[str, Tensor] = {}
        fanin_by_dst: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing[
            "fanin_by_dst"
        ]
        for dst in self.dst_symbol_type_ids:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            pair = fanin_by_dst.get(dst)
            if not pair:
                symbol_msgs[dst] = x_dict[dst].new_zeros(
                    (dim_size, self.embedding_size)
                )
                continue
            flat_src, dst_index = pair
            if (
                self._mp_fanin_mode is not None
                and _use_model_mp_fanin(out_flat)
                and _env_bool("RELM_MODELS_MP_FANIN_FUSED", True)
            ):
                symbol_msgs[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                    out_flat,
                    flat_src,
                    dst_index,
                    dim_size,
                    self._mp_fanin_mode,
                )
            else:
                msgs = out_flat.index_select(0, flat_src)
                symbol_msgs[dst] = self.aggr(
                    x=msgs, index=dst_index, dim=0, dim_size=dim_size
                )

        return atom_msgs, symbol_msgs


class BatchedFanOutMP(DeviceAwareMixin, torch.nn.Module):
    """
    Decentralized fan-out (symbols -> predicate embeddings) using batched gather.

    This bypasses PyG's per-edge-type message passing and instead fills the predicate slot
    tensor directly via index_copy_.
    """

    def __init__(
        self,
        update_modules: Dict[str, torch.nn.Module],
        *,
        relation_arities: Mapping[str, int],
        embedding_size: int,
        src_types: Iterable[str],
        strict_filter_mode: bool = False,
        validate_routing: bool = False,
        use_cuda_streams: bool = False,
    ) -> None:
        super().__init__()
        self.update_modules = PatchedModuleDict(update_modules)
        self.relation_arities = dict(relation_arities)
        self.embedding_size = int(embedding_size)
        self.src_types = (
            tuple(src_types) if not isinstance(src_types, str) else (src_types,)
        )
        self.strict_filter_mode = bool(strict_filter_mode)
        self.validate_routing = bool(validate_routing)
        self.use_cuda_streams = torch.cuda.is_available() and use_cuda_streams
        self._cuda_streams = None
        self._cuda_pool = None

    def _match_src(self, src: str) -> bool:
        return _match_ntype(src, self.src_types, self.strict_filter_mode)

    @property
    def cuda_streams(self):
        if (
            self.use_cuda_streams
            and self._cuda_streams is None
            and self.device.type == "cuda"
        ):
            self._cuda_streams = [
                torch.cuda.Stream(self.device) for _ in range(len(self.update_modules))
            ]
            self._cuda_pool = itertools.cycle(self._cuda_streams)
        return self._cuda_streams

    def next_stream(self):
        if not self.use_cuda_streams or self.device.type != "cuda":
            return None
        assert self.cuda_streams
        return next(self._cuda_pool)

    def _sync_streams(self):
        if self.use_cuda_streams and (cuda_streams := self.cuda_streams) is not None:
            for stream in cuda_streams:
                stream.synchronize()

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *,
        cache: dict | None = None,
        **_: Any,
    ) -> Dict[str, Tensor]:
        cache = cache if cache is not None else {}
        layer_state = cache.setdefault("layer_state", {})
        # Layer-local handoff consumed by BatchedFanInMP in the same layer() call.
        layer_state["fanout_rel_flat_all"] = None
        layer_state["fanout_total_slots"] = 0
        if not edge_index_dict:
            return {}

        # Cache layout convention:
        # - cache["routing"]["fanout"] contains routing indices
        # - cache["buffers"] contains reusable tensors
        #
        # This is important because fanout and fanin share the same cache dict
        # (RelationalGNN passes one cache into both MPs).
        routing_root = cache.setdefault("routing", {})
        routing = routing_root.get("fanout")
        if routing is None:
            sizes: dict[str, int] = {}
            for pred in self.update_modules.keys():
                sizes[pred] = int(x_dict[pred].size(0)) if pred in x_dict else 0

            # Fallback: infer missing predicate counts from edges.
            for edge_type, edge_index in edge_index_dict.items():
                src, _, dst = edge_type
                if dst not in sizes or sizes[dst] > 0:
                    continue
                if dst not in self.update_modules:
                    continue
                if not self._match_src(src):
                    continue
                if edge_index is None or edge_index.numel() == 0:
                    continue
                sizes[dst] = int(edge_index[1].max().item()) + 1

            active = sorted(pred for pred, n in sizes.items() if n > 0)
            pred_meta = []
            total_slots = 0
            for pred in active:
                arity = int(self.relation_arities.get(pred, 0))
                if arity <= 0:
                    continue
                n = int(sizes[pred])
                slot_offset = total_slots
                total_slots += int(n * arity)
                pred_meta.append((pred, n, arity, slot_offset))

            slot_offsets = {pred: slot_offset for pred, _, _, slot_offset in pred_meta}

            # Global routed scatter by source type into one flattened slot tensor.
            tmp_flat_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)
            tmp_src_by_src: dict[str, list[torch.Tensor]] = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                if dst not in slot_offsets:
                    continue
                if dst not in self.update_modules:
                    continue
                if not self._match_src(src):
                    continue
                if edge_index is None or edge_index.numel() == 0:
                    continue
                pos = int(rel)
                arity = int(self.relation_arities.get(dst, 0))
                if pos < 0 or pos >= arity:
                    continue
                slot_offset = int(slot_offsets[dst])
                flat_dst = slot_offset + edge_index[1] * arity + pos
                tmp_flat_by_src[src].append(flat_dst)
                tmp_src_by_src[src].append(edge_index[0])

            fanout_by_src = _finalize_pair_lists(tmp_flat_by_src, tmp_src_by_src)
            if self.validate_routing:
                for src, (flat_dst, _src_idx) in fanout_by_src.items():
                    if int(flat_dst.unique().numel()) != int(flat_dst.numel()):
                        raise AssertionError(
                            f"Fanout routing duplicates detected for src={src!r}."
                        )
            fanout_plan = _build_fanout_scatter_plan(
                by_src=fanout_by_src,
                x_dict=x_dict,
            )

            pred_exec = []
            for pred, n, arity, slot_offset in pred_meta:
                if pred not in self.update_modules:
                    continue
                pred_exec.append(
                    (pred, n, arity, slot_offset, self.update_modules[pred])
                )

            routing = {
                "pred_meta": pred_meta,
                "pred_exec": pred_exec,
                "total_slots": int(total_slots),
                "fanout_by_src": fanout_by_src,
                "fanout_plan": fanout_plan,
            }
            routing_root["fanout"] = routing

        pred_exec = routing["pred_exec"]
        if not pred_exec:
            return {}

        # Find reference symbol tensor for dtype/device.
        ref = None
        for key in self.src_types:
            if key in x_dict:
                ref = x_dict[key]
                break
        if ref is None:
            for key, val in x_dict.items():
                if torch.is_tensor(val) and val.dim() == 2 and self._match_src(key):
                    ref = val
                    break
        if ref is None:
            ref = next(iter(x_dict.values()))

        # Same rationale as CentralFusedLayerMP: never reuse and overwrite storage in training,
        # because those tensors are needed to compute parameter gradients.
        reuse_buffers = (not self.training) and (not torch.is_grad_enabled())
        buffers_root = cache.setdefault("buffers", {})
        outputs: Dict[str, Tensor] = {}
        total_slots = int(routing["total_slots"])
        publish_rel_flat = _use_model_mp_fanin(ref) and _env_bool(
            "RELM_MODELS_MP_FANIN_BATCHED", True
        )
        rel_flat_all: torch.Tensor | None = None
        if publish_rel_flat:
            if reuse_buffers:
                rel_flat_all = _get_or_make_buffer(
                    buffers_root,
                    "fanout::rel_flat_all",
                    ref=ref,
                    shape=(total_slots, self.embedding_size),
                    zero=False,
                )
            else:
                rel_flat_all = ref.new_empty((total_slots, self.embedding_size))
        if reuse_buffers:
            args_flat_all = _get_or_make_buffer(
                buffers_root,
                "fanout::args_flat_all",
                ref=ref,
                shape=(total_slots, self.embedding_size),
                zero=True,
            )
        else:
            args_flat_all = ref.new_zeros((total_slots, self.embedding_size))

        fanout_by_src: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing[
            "fanout_by_src"
        ]
        if fanout_by_src and _use_model_mp_fanout(ref):
            plan = routing.get("fanout_plan")
            if plan is not None:
                args_flat_all = _fanout_scatter_from_plan(
                    plan=plan,
                    x_dict=x_dict,
                    out_rows=total_slots,
                )
            else:
                args_flat_all = _fanout_scatter_multi_src(
                    by_src=fanout_by_src,
                    x_dict=x_dict,
                    out_rows=total_slots,
                )
        else:
            for src, (flat_dst, src_idx) in fanout_by_src.items():
                if src not in x_dict:
                    raise KeyError(
                        f"Missing src node type {src!r} in x_dict for fanout scatter."
                    )
                vals = x_dict[src].index_select(0, src_idx)
                args_flat_all.index_copy_(0, flat_dst, vals)

        for pred, n, arity, slot_offset, update_module in pred_exec:
            args_flat = args_flat_all[slot_offset : slot_offset + (n * arity)]
            args_feat = args_flat.view(n, arity * self.embedding_size)
            if (stream := self.next_stream()) is not None:
                with stream_context(stream):
                    out_pred = update_module(args_feat)
                    outputs[pred] = out_pred
                    if rel_flat_all is not None:
                        rel_flat_all[slot_offset : slot_offset + (n * arity)] = (
                            out_pred.view(-1, self.embedding_size)
                        )
            else:
                out_pred = update_module(args_feat)
                outputs[pred] = out_pred
                if rel_flat_all is not None:
                    rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                        -1, self.embedding_size
                    )

        self._sync_streams()
        layer_state["fanout_rel_flat_all"] = rel_flat_all
        layer_state["fanout_total_slots"] = int(total_slots)
        return outputs


class BatchedFanInMP(torch.nn.Module):
    """
    Decentralized fan-in aggregation (predicates -> symbols) using batched index_select + aggr.
    """

    def __init__(
        self,
        *,
        embedding_size: int,
        dst_types: Iterable[str],
        relation_arities: Mapping[str, int],
        aggr: Aggregation | str | None,
        strict_filter_mode: bool = False,
        validate_routing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(aggr, str) or aggr is None:
            resolved = aggregation_resolver(query=aggr or "logsumexp")
            if not isinstance(resolved, Aggregation):
                raise ValueError(
                    "BatchedFanInMP requires a PyG Aggregation module, "
                    f"got {type(resolved)!r} from query={aggr!r}."
                )
            aggr = resolved
        if not isinstance(aggr, Aggregation):
            raise ValueError(
                f"BatchedFanInMP requires a PyG Aggregation module, got {type(aggr)!r}."
            )
        self.embedding_size = int(embedding_size)
        self.dst_types = (
            tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)
        )
        self.relation_arities = dict(relation_arities)
        self.aggr = aggr
        self._mp_fanin_mode = _resolve_fanin_mode(self.aggr)
        self.strict_filter_mode = bool(strict_filter_mode)
        self.validate_routing = bool(validate_routing)

    def _match_dst(self, dst: str) -> bool:
        return _match_ntype(dst, self.dst_types, self.strict_filter_mode)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *,
        cache: dict | None = None,
        **_: Any,
    ) -> Dict[str, Tensor]:
        if not edge_index_dict:
            return {}
        cache = cache if cache is not None else {}

        # See BatchedFanOutMP for the shared cache layout. We store fanin routing under
        # cache["routing"]["fanin"] to avoid collisions with fanout routing.
        routing_root = cache.setdefault("routing", {})
        routing = routing_root.get("fanin")
        if routing is None:
            # fanin_by_dst[dst][pred] = (flat_src, dst_idx)
            tmp: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = (
                defaultdict(list)
            )
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                if not self._match_dst(dst):
                    continue
                if src not in self.relation_arities:
                    continue
                if edge_index is None or edge_index.numel() == 0:
                    continue
                pos = int(rel)
                arity = int(self.relation_arities.get(src, 0))
                if pos < 0 or pos >= arity:
                    if self.validate_routing and arity > 0:
                        raise AssertionError(
                            f"Fanin routing pos out of range: pred={src!r} pos={pos} arity={arity}."
                        )
                    continue
                flat_src = edge_index[0] * arity + pos
                tmp[(dst, src)].append((flat_src, edge_index[1]))

            fanin_by_dst: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = (
                defaultdict(dict)
            )
            for (dst, pred), parts in tmp.items():
                flat_srcs = [p[0] for p in parts]
                dst_idxs = [p[1] for p in parts]
                flat_src = _cat_or_single(flat_srcs, dim=0)
                dst_idx = _cat_or_single(dst_idxs, dim=0)
                fanin_by_dst[dst][pred] = (flat_src, dst_idx)

            mp_fanin_plans: dict[str, dict[str, Any]] = {}
            for dst, per_pred in fanin_by_dst.items():
                pred_order: list[str] = []
                rel_rows: list[int] = []
                flat_src_parts: list[torch.Tensor] = []
                dst_parts: list[torch.Tensor] = []
                rel_offset = 0
                for pred in sorted(per_pred.keys()):
                    if pred not in x_dict:
                        continue
                    x_pred = x_dict[pred]
                    if x_pred.numel() == 0:
                        continue
                    arity = int(self.relation_arities.get(pred, 0))
                    rows = int(x_pred.size(0) * arity)
                    flat_src, dst_idx = per_pred[pred]
                    pred_order.append(pred)
                    rel_rows.append(rows)
                    flat_src_parts.append(flat_src + int(rel_offset))
                    dst_parts.append(dst_idx)
                    rel_offset += rows
                if pred_order:
                    mp_fanin_plans[dst] = {
                        "pred_order": tuple(pred_order),
                        "rel_rows": tuple(rel_rows),
                        "total_rel_rows": int(rel_offset),
                        "flat_src": _cat_or_single(flat_src_parts, dim=0),
                        "dst_idx": _cat_or_single(dst_parts, dim=0),
                    }

            mp_fanin_global_plans: dict[str, dict[str, Any]] = {}
            fanout_routing = routing_root.get("fanout")
            pred_slot_offsets: dict[str, int] = {}
            total_slots = 0
            if fanout_routing is not None:
                total_slots = int(fanout_routing.get("total_slots", 0))
                for pred, _n, _arity, slot_offset in fanout_routing.get("pred_meta", []):
                    pred_slot_offsets[str(pred)] = int(slot_offset)
            if pred_slot_offsets:
                for dst, per_pred in fanin_by_dst.items():
                    flat_src_parts: list[torch.Tensor] = []
                    dst_parts: list[torch.Tensor] = []
                    for pred in sorted(per_pred.keys()):
                        slot_offset = pred_slot_offsets.get(pred)
                        if slot_offset is None:
                            continue
                        flat_src, dst_idx = per_pred[pred]
                        flat_src_parts.append(flat_src + int(slot_offset))
                        dst_parts.append(dst_idx)
                    if flat_src_parts:
                        mp_fanin_global_plans[dst] = {
                            "flat_src": _cat_or_single(flat_src_parts, dim=0),
                            "dst_idx": _cat_or_single(dst_parts, dim=0),
                            "total_slots": int(total_slots),
                        }

            routing = {
                "fanin_by_dst": dict(fanin_by_dst),
                "mp_fanin_plans": mp_fanin_plans,
                "mp_fanin_global_plans": mp_fanin_global_plans,
            }
            routing_root["fanin"] = routing

        fanin_by_dst = routing["fanin_by_dst"]
        mp_fanin_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_plans", {})
        mp_fanin_global_plans: dict[str, dict[str, Any]] = routing.get(
            "mp_fanin_global_plans", {}
        )
        layer_state = cache.get("layer_state")
        rel_flat_shared = (
            layer_state.get("fanout_rel_flat_all")
            if isinstance(layer_state, dict)
            else None
        )
        out: Dict[str, Tensor] = {}

        for dst in self.dst_types:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            use_mp_fanin = self._mp_fanin_mode is not None and _use_model_mp_fanin(
                x_dict[dst]
            )
            use_mp_fanin = use_mp_fanin and _env_bool(
                "RELM_MODELS_MP_FANIN_BATCHED", True
            )
            if use_mp_fanin:
                global_plan = mp_fanin_global_plans.get(dst)
                if global_plan is not None and isinstance(rel_flat_shared, torch.Tensor):
                    if (
                        rel_flat_shared.device == x_dict[dst].device
                        and int(rel_flat_shared.size(-1)) == self.embedding_size
                        and (
                            (not self.validate_routing)
                            or int(rel_flat_shared.size(0))
                            == int(global_plan["total_slots"])
                        )
                    ):
                        out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                            rel_flat_shared,
                            global_plan["flat_src"],
                            global_plan["dst_idx"],
                            dim_size,
                            self._mp_fanin_mode,
                        )
                        continue
                plan = mp_fanin_plans.get(dst)
                if plan is not None:
                    rel_parts: list[torch.Tensor] = []
                    plan_ok = True
                    for pred in plan["pred_order"]:
                        rel_pred = x_dict.get(pred)
                        if rel_pred is None:
                            plan_ok = False
                            break
                        rel_parts.append(rel_pred.view(-1, self.embedding_size))
                    if plan_ok and rel_parts:
                        rel_cat = _cat_or_single(rel_parts, dim=0)
                        if (
                            (not self.validate_routing)
                            or int(rel_cat.size(0)) == int(plan["total_rel_rows"])
                        ):
                            out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                                rel_cat,
                                plan["flat_src"],
                                plan["dst_idx"],
                                dim_size,
                                self._mp_fanin_mode,
                            )
                            continue

            per_pred = fanin_by_dst.get(dst)
            if not per_pred:
                out[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue

            inputs = []
            indices = []
            rel_parts = []
            mp_src_parts = []
            mp_dst_parts = []
            rel_offset = 0
            for pred, (flat_src, dst_idx) in per_pred.items():
                if pred not in x_dict:
                    continue
                x_pred = x_dict[pred]
                if x_pred.numel() == 0:
                    continue
                arity = int(self.relation_arities.get(pred, 0))
                exp = arity * self.embedding_size
                if int(x_pred.size(-1)) != exp:
                    raise ValueError(
                        f"Predicate {pred!r} has arity {arity}, but embedding dim is {int(x_pred.size(-1))} (expected {exp})."
                    )
                rel_flat = x_pred.view(-1, self.embedding_size)
                if use_mp_fanin:
                    rel_parts.append(rel_flat)
                    mp_src_parts.append(flat_src + int(rel_offset))
                    mp_dst_parts.append(dst_idx)
                    rel_offset += int(rel_flat.size(0))
                else:
                    inputs.append(rel_flat.index_select(0, flat_src))
                    indices.append(dst_idx)

            if use_mp_fanin and rel_parts:
                rel_cat = _cat_or_single(rel_parts, dim=0)
                flat_src_all = _cat_or_single(mp_src_parts, dim=0)
                dst_all = _cat_or_single(mp_dst_parts, dim=0)
                out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                    rel_cat,
                    flat_src_all,
                    dst_all,
                    dim_size,
                    self._mp_fanin_mode,
                )
                continue

            if not inputs:
                out[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue

            flat_inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
            flat_indices = (
                indices[0] if len(indices) == 1 else torch.cat(indices, dim=0)
            )
            out[dst] = self.aggr(
                x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_size
            )

        return out


class FanInMP(HeteroRouting):
    """ """

    def __init__(
        self,
        embedding_size: int,
        dst_types: Iterable[str],
        aggr: str | torch_geometric.nn.Aggregation | None = None,
        **kwargs,
    ) -> None:
        aggr = aggr or LogSumExpAggregation()
        super().__init__(aggr, **kwargs)
        self.select = SelectMP(embedding_size)
        self.dst_types = (
            tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)
        )

    def _accepts_edge(self, edge_type: EdgeType) -> bool:
        *_, dst = edge_type
        if self.strict_filter_mode:
            return dst in self.dst_types
        else:
            return any(dst_type in dst for dst_type in self.dst_types)

    def _internal_forward(self, x, edges_index, edge_type, **kwargs):
        return self.select(x, edges_index, int(edge_type[1]))

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        aggregated = {}
        for dst, values in out_dict.items():
            assert self._accepts_edge(("", "", dst))
            inputs, indices, dim_sizes = zip(*values)
            flat_inputs = torch.cat(inputs)
            flat_indices = torch.cat(indices)
            out = self.aggr(
                x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_sizes[0]
            )
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
        self.dst_types = (
            tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)
        )
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
            src_global = (
                src_parts[0] if len(src_parts) == 1 else torch.cat(src_parts, dim=0)
            )
            dst_parts = per_dst_dst[dst]
            dst_index = (
                dst_parts[0] if len(dst_parts) == 1 else torch.cat(dst_parts, dim=0)
            )
            pos_parts = per_dst_pos[dst]
            pos_all = (
                pos_parts[0] if len(pos_parts) == 1 else torch.cat(pos_parts, dim=0)
            )

            gathered = rel_all.index_select(0, src_global)  # [E, max_arity, emb]
            pos_idx = pos_all.view(-1, 1, 1).expand(-1, 1, self.embedding_size)
            msgs = torch.gather(gathered, 1, pos_idx).squeeze(1)  # [E, emb]
            aggregated[dst] = self.aggr(
                x=msgs, index=dst_index, dim=0, dim_size=dim_size
            )

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

    def forward(
        self, x: Union[Tensor, OptPairTensor], edge_index: Adj, position: int
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        return self.propagate(edge_index, x=x, position=position)

    def message(self, x_j: Tensor, position: int = None) -> Tensor:
        # Take the i-th hidden-number of elements from the last dimension
        # e.g from [1, 2, 3, 4, 5, 6] with hidden=2 and position=1 -> [3, 4]
        # alternatively:
        #   split = torch.split(x_j, self.embedding_size, dim=-1)
        #   return split[position]
        sliced = x_j[
            ..., position * self.embedding_size : (position + 1) * self.embedding_size
        ]
        return sliced

    def aggregate(
        self,
        inputs: Tensor,
        index: Tensor,
        ptr: Optional[Tensor] = None,
        dim_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        return inputs, index, dim_size


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
