from __future__ import annotations

import itertools
import operator
from collections import defaultdict
from typing import Dict, Iterable, List

import torch
from torch import Tensor
from torch_geometric.nn import SimpleConv
from torch_geometric.typing import EdgeType

from .._misc import stream_context
from ..mixins import DeviceAwareMixin
from ..patched_module_dict import PatchedModuleDict
from .routing import HeteroRouting


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
            grouped[predicate] = all_outputs[start : start + n, : arity * self.embedding_size]
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
                if slot_mask_row.device != full.device or slot_mask_row.dtype != full.dtype:
                    slot_mask_row = slot_mask_row.to(device=full.device, dtype=full.dtype)

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
