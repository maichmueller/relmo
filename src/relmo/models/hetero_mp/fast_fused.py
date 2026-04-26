from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType

from ._ops_env import (
    _resolve_fanin_mode,
    _use_model_mp_batched_fanin_reduce,
    _use_model_mp_batched_fanout,
    relmo_mp_ops,
)
from ._scatter import (
    _build_fanout_scatter_plan,
    _fanout_scatter_from_plan,
    _fanout_scatter_multi_src,
)
from ._tensor_utils import _finalize_pair_lists, _get_or_make_buffer, _match_ntype
from ..patched_module_dict import PatchedModuleDict

class FastFusedRelationalLayerMP(torch.nn.Module):
    """
    Fast decentralized relational layer:
    - gather symbol->relation slots
    - relation-specific MLP updates
    - aggregate relation slots->symbols

    Routing is built once per forward and reused across all model layers.
    """

    def __init__(
        self,
        *,
        update_modules: Dict[str, torch.nn.Module],
        relation_arities: Mapping[str, int],
        embedding_size: int,
        src_types: Iterable[str],
        dst_types: Iterable[str],
        aggr: Aggregation | str | None,
        strict_filter_mode: bool = False,
        validate_routing: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(aggr, str) or aggr is None:
            resolved = aggregation_resolver(query=aggr or "logsumexp")
            if not isinstance(resolved, Aggregation):
                raise ValueError(
                    "FastFusedRelationalLayerMP requires a PyG Aggregation module, "
                    f"got {type(resolved)!r} from query={aggr!r}."
                )
            aggr = resolved
        if not isinstance(aggr, Aggregation):
            raise ValueError(
                f"FastFusedRelationalLayerMP requires a PyG Aggregation module, got {type(aggr)!r}."
            )

        self.update_modules = PatchedModuleDict(update_modules)
        self.relation_arities = dict(relation_arities)
        self.embedding_size = int(embedding_size)
        self.src_types = tuple(src_types) if not isinstance(src_types, str) else (src_types,)
        self.dst_types = tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)
        self.aggr = aggr
        self._mp_fanin_mode = _resolve_fanin_mode(self.aggr)
        self.strict_filter_mode = bool(strict_filter_mode)
        self.validate_routing = bool(validate_routing)
    def _match_src(self, src: str) -> bool:
        return _match_ntype(src, self.src_types, self.strict_filter_mode)

    def _match_dst(self, dst: str) -> bool:
        return _match_ntype(dst, self.dst_types, self.strict_filter_mode)

    def _build_routing(
        self,
        *,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> dict[str, Any]:
        sizes: dict[str, int] = {}
        for pred in self.update_modules.keys():
            sizes[pred] = int(x_dict[pred].size(0)) if pred in x_dict else 0

        # Fallback: infer relation row counts from symbol->relation edges.
        for edge_type, edge_index in edge_index_dict.items():
            src, _rel, dst = edge_type
            if dst not in sizes or sizes[dst] > 0:
                continue
            if dst not in self.update_modules:
                continue
            if not self._match_src(src):
                continue
            if edge_index is None or edge_index.numel() == 0:
                continue
            sizes[dst] = int(edge_index[1].max().item()) + 1

        pred_meta = []
        total_slots = 0
        for pred in sorted(sizes.keys()):
            n = int(sizes[pred])
            arity = int(self.relation_arities.get(pred, 0))
            if n <= 0 or arity <= 0:
                continue
            slot_offset = total_slots
            total_slots += int(n * arity)
            pred_meta.append(
                (pred, n, arity, slot_offset, self.update_modules[pred])
            )

        if not pred_meta:
            return {
                "pred_meta": [],
                "total_slots": 0,
                "fanout_by_src": {},
                "fanin_by_dst": {},
                "fanout_plan": None,
            }

        slot_offsets = {pred: slot_offset for pred, _n, _arity, slot_offset, _m in pred_meta}

        fanout_flat_by_src: dict[str, list[Tensor]] = defaultdict(list)
        fanout_src_by_src: dict[str, list[Tensor]] = defaultdict(list)
        fanin_src_by_dst: dict[str, list[Tensor]] = defaultdict(list)
        fanin_dst_by_dst: dict[str, list[Tensor]] = defaultdict(list)

        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            if edge_index is None or edge_index.numel() == 0:
                continue
            try:
                pos = int(rel)
            except (TypeError, ValueError):
                continue

            # Fanout: symbol -> relation slot.
            if dst in slot_offsets and self._match_src(src):
                arity = int(self.relation_arities.get(dst, 0))
                if 0 <= pos < arity:
                    slot_offset = int(slot_offsets[dst])
                    flat_dst = slot_offset + edge_index[1] * arity + pos
                    fanout_flat_by_src[src].append(flat_dst)
                    fanout_src_by_src[src].append(edge_index[0])

            # Fanin: relation slot -> symbol.
            if src in slot_offsets and self._match_dst(dst):
                arity = int(self.relation_arities.get(src, 0))
                if 0 <= pos < arity:
                    slot_offset = int(slot_offsets[src])
                    flat_src = slot_offset + edge_index[0] * arity + pos
                    fanin_src_by_dst[dst].append(flat_src)
                    fanin_dst_by_dst[dst].append(edge_index[1])

        fanout_by_src = _finalize_pair_lists(fanout_flat_by_src, fanout_src_by_src)
        if self.validate_routing:
            for src, (flat_dst, _src_idx) in fanout_by_src.items():
                if int(flat_dst.unique().numel()) != int(flat_dst.numel()):
                    raise AssertionError(
                        f"Fast fused fanout routing duplicates detected for src={src!r}."
                    )
        fanin_by_dst = _finalize_pair_lists(fanin_src_by_dst, fanin_dst_by_dst)

        return {
            "pred_meta": pred_meta,
            "total_slots": int(total_slots),
            "fanout_by_src": fanout_by_src,
            "fanin_by_dst": fanin_by_dst,
            "fanout_plan": _build_fanout_scatter_plan(by_src=fanout_by_src, x_dict=x_dict),
        }

    def _resolve_ref(self, x_dict: Dict[str, Tensor]) -> Tensor:
        for key in self.src_types:
            if key in x_dict:
                return x_dict[key]
        for key, val in x_dict.items():
            if torch.is_tensor(val) and val.dim() == 2 and self._match_src(key):
                return val
        return next(iter(x_dict.values()))

    def _empty_symbol_msgs(self, x_dict: Dict[str, Tensor]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        for dst in self.dst_types:
            if dst in x_dict:
                out[dst] = x_dict[dst].new_zeros((int(x_dict[dst].size(0)), self.embedding_size))
        return out

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *,
        cache: dict | None = None,
        **_: Any,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        if not edge_index_dict:
            return {}, self._empty_symbol_msgs(x_dict)

        cache = cache if cache is not None else {}
        routing = cache.get("routing")
        if routing is None:
            routing = self._build_routing(x_dict=x_dict, edge_index_dict=edge_index_dict)
            cache["routing"] = routing

        pred_meta = routing["pred_meta"]
        if not pred_meta:
            return {}, self._empty_symbol_msgs(x_dict)

        ref = self._resolve_ref(x_dict)
        total_slots = int(routing["total_slots"])
        reuse_buffers = (not self.training) and (not torch.is_grad_enabled())
        buffers = cache.setdefault("buffers", {})

        if reuse_buffers:
            args_flat_all = _get_or_make_buffer(
                buffers,
                "fast_fused::args_flat_all",
                ref=ref,
                shape=(total_slots, self.embedding_size),
                zero=True,
            )
            rel_flat_all = _get_or_make_buffer(
                buffers,
                "fast_fused::rel_flat_all",
                ref=ref,
                shape=(total_slots, self.embedding_size),
                zero=False,
            )
        else:
            args_flat_all = ref.new_zeros((total_slots, self.embedding_size))
            rel_flat_all = ref.new_empty((total_slots, self.embedding_size))

        fanout_by_src: dict[str, tuple[Tensor, Tensor]] = routing["fanout_by_src"]
        if fanout_by_src and _use_model_mp_batched_fanout(ref):
            fanout_plan = routing.get("fanout_plan")
            if fanout_plan is not None:
                try:
                    args_flat_all = _fanout_scatter_from_plan(
                        plan=fanout_plan,
                        x_dict=x_dict,
                        out_rows=total_slots,
                    )
                except Exception:
                    if self.validate_routing:
                        raise
                    args_flat_all = _fanout_scatter_multi_src(
                        by_src=fanout_by_src,
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
                        f"Missing src node type {src!r} in x_dict for fast fused fanout."
                    )
                vals = x_dict[src].index_select(0, src_idx)
                args_flat_all.index_copy_(0, flat_dst, vals)

        # Note: in RelationalGNN fast_fused mode, atom messages are not consumed;
        # we keep the return slot for API compatibility but avoid materializing it.
        atom_msgs: dict[str, Tensor] = {}
        for _pred, n, arity, slot_offset, update_module in pred_meta:
            args_flat = args_flat_all[slot_offset : slot_offset + (n * arity)]
            args_feat = args_flat.view(n, arity * self.embedding_size)
            out_pred = update_module(args_feat)
            rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                -1, self.embedding_size
            )

        symbol_msgs: dict[str, Tensor] = {}
        fanin_by_dst: dict[str, tuple[Tensor, Tensor]] = routing["fanin_by_dst"]
        use_mp_fanin_reduce = (
            self._mp_fanin_mode is not None and _use_model_mp_batched_fanin_reduce(ref)
        )
        for dst in self.dst_types:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            pair = fanin_by_dst.get(dst)
            if not pair:
                symbol_msgs[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue
            flat_src, dst_index = pair
            if use_mp_fanin_reduce and _use_model_mp_batched_fanin_reduce(x_dict[dst]):
                symbol_msgs[dst] = relmo_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                    rel_flat_all, flat_src, dst_index, dim_size, self._mp_fanin_mode
                )
            else:
                msgs = rel_flat_all.index_select(0, flat_src)
                symbol_msgs[dst] = self.aggr(x=msgs, index=dst_index, dim=0, dim_size=dim_size)

        return atom_msgs, symbol_msgs
