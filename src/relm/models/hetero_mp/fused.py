from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType

from ._ops_env import (
    _env_bool,
    _resolve_fanin_mode,
    _use_model_mp_fanin,
    _use_model_mp_fanout,
    relm_mp_ops,
)
from ._scatter import (
    _build_fanout_scatter_plan,
    _fanout_scatter_from_plan,
    _fanout_scatter_multi_src,
)
from ._tensor_utils import _finalize_pair_lists, _get_or_make_buffer, _match_ntype


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
            (symbol_type_ids,) if isinstance(symbol_type_ids, str) else tuple(symbol_type_ids)
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
                sizes[predicate] = int(x_dict[predicate].size(0)) if predicate in x_dict else 0

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
            fanout_plan = _build_fanout_scatter_plan(by_src=fanout_by_src, x_dict=x_dict)

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
                if self._match_any(key, self.symbol_type_ids) and int(val.size(-1)) == self.embedding_size:
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
            args_flat = ref.new_zeros((total_relation_nodes * self.max_arity, self.embedding_size))

        fanout_by_src: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing["fanout_by_src"]
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
                    raise KeyError(f"Missing src node type {src!r} in x_dict for fused fanout.")
                vals = x_dict[src].index_select(0, src_idxs)
                args_flat.index_copy_(0, flat_dst, vals)

        args_feat = args_flat.reshape(total_relation_nodes, self.max_arity * self.embedding_size)

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
                if cond is not None and (cond.device != context.device or cond.dtype != context.dtype):
                    cond = cond.to(device=context.device, dtype=context.dtype)

                slot_mask_row = None
                if self.include_slot_mask:
                    slot_mask_row = self._slot_mask_table[cond_idx]
                    if slot_mask_row.device != context.device or slot_mask_row.dtype != context.dtype:
                        slot_mask_row = slot_mask_row.to(device=context.device, dtype=context.dtype)

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
            atom_msgs[predicate] = all_outputs[start : start + n, : arity * self.embedding_size]

        # Fanin aggregation with flat indexing (no gather):
        out_flat = all_outputs.reshape(total_relation_nodes * self.max_arity, self.embedding_size)

        symbol_msgs: dict[str, Tensor] = {}
        fanin_by_dst: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing["fanin_by_dst"]
        for dst in self.dst_symbol_type_ids:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            pair = fanin_by_dst.get(dst)
            if not pair:
                symbol_msgs[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue
            flat_src, dst_index = pair
            if (
                self._mp_fanin_mode is not None
                and _use_model_mp_fanin(out_flat)
                and _env_bool("RELM_MODELS_MP_FANIN_FUSED", True)
            ):
                symbol_msgs[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                    out_flat, flat_src, dst_index, dim_size, self._mp_fanin_mode
                )
            else:
                msgs = out_flat.index_select(0, flat_src)
                symbol_msgs[dst] = self.aggr(x=msgs, index=dst_index, dim=0, dim_size=dim_size)

        return atom_msgs, symbol_msgs
