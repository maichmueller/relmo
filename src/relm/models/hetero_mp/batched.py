from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType

from .._misc import stream_context
from ..mixins import DeviceAwareMixin
from ..patched_module_dict import PatchedModuleDict
from ._ops_env import (
    _resolve_fanin_mode,
    _use_model_mp_batched_fanin_pack,
    _use_model_mp_batched_fanin_reduce,
    _use_model_mp_batched_fanout,
    relm_mp_ops,
)
from ._scatter import (
    _build_fanout_scatter_plan,
    _fanout_scatter_from_plan,
    _fanout_scatter_multi_src,
)
from ._tensor_utils import _cat_or_single, _finalize_pair_lists, _get_or_make_buffer, _match_ntype


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
        self.src_types = tuple(src_types) if not isinstance(src_types, str) else (src_types,)
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
            mp_edge_src_parts: list[torch.Tensor] = []
            mp_edge_dst_parts: list[torch.Tensor] = []
            mp_edge_src_labels: list[str] = []
            mp_arity_parts: list[int] = []
            mp_pos_parts: list[int] = []
            mp_slot_offset_parts: list[int] = []
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
                mp_edge_src_parts.append(edge_index[0])
                mp_edge_dst_parts.append(edge_index[1])
                mp_edge_src_labels.append(src)
                mp_arity_parts.append(int(arity))
                mp_pos_parts.append(int(pos))
                mp_slot_offset_parts.append(int(slot_offset))

            fanout_by_src = _finalize_pair_lists(tmp_flat_by_src, tmp_src_by_src)
            if self.validate_routing:
                for src, (flat_dst, _src_idx) in fanout_by_src.items():
                    if int(flat_dst.unique().numel()) != int(flat_dst.numel()):
                        raise AssertionError(
                            f"Fanout routing duplicates detected for src={src!r}."
                        )

            pred_exec = []
            for pred, n, arity, slot_offset in pred_meta:
                if pred not in self.update_modules:
                    continue
                pred_exec.append((pred, n, arity, slot_offset, self.update_modules[pred]))

            mp_src_order = tuple(sorted(set(mp_edge_src_labels)))
            mp_src_part_index = {src: idx for idx, src in enumerate(mp_src_order)}
            mp_src_part_ids = [int(mp_src_part_index[src]) for src in mp_edge_src_labels]
            mp_fanout_packed_plan: dict[str, Any] | None = None
            if mp_edge_src_parts:
                try:
                    x_parts: list[Tensor] = []
                    src_rows: list[int] = []
                    for src in mp_src_order:
                        x_src = x_dict.get(src)
                        if x_src is None:
                            raise KeyError(f"Missing src node type {src!r} in x_dict for fanout.")
                        x_parts.append(x_src)
                        src_rows.append(int(x_src.size(0)))
                    x_cat_pre, src_global_pre, flat_dst_pre = relm_mp_ops.fanout_pack_from_edges(  # type: ignore[union-attr]
                        x_parts,
                        mp_edge_src_parts,
                        mp_edge_dst_parts,
                        mp_src_part_ids,
                        mp_arity_parts,
                        mp_pos_parts,
                        mp_slot_offset_parts,
                    )
                    mp_fanout_packed_plan = {
                        "src_order": mp_src_order,
                        "src_rows": tuple(src_rows),
                        "x_rows": int(x_cat_pre.size(0)),
                        "src_global": src_global_pre,
                        "flat_dst": flat_dst_pre,
                    }
                except Exception:
                    if self.validate_routing:
                        raise
                    mp_fanout_packed_plan = None
            routing = {
                "pred_meta": pred_meta,
                "pred_exec": pred_exec,
                "total_slots": int(total_slots),
                "fanout_by_src": fanout_by_src,
                "mp_fanout_plan": _build_fanout_scatter_plan(
                    by_src=fanout_by_src,
                    x_dict=x_dict,
                ),
                "mp_fanout_edge_plan": {
                    "src_order": mp_src_order,
                    "edge_src_parts": tuple(mp_edge_src_parts),
                    "edge_dst_parts": tuple(mp_edge_dst_parts),
                    "src_part_ids": tuple(mp_src_part_ids),
                    "arity_parts": tuple(mp_arity_parts),
                    "pos_parts": tuple(mp_pos_parts),
                    "slot_offset_parts": tuple(mp_slot_offset_parts),
                },
                "mp_fanout_packed_plan": mp_fanout_packed_plan,
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
        use_mp_fanout = _use_model_mp_batched_fanout(ref)
        # Canonical handoff: always publish flattened relation slots for fanin.
        publish_rel_flat = True
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

        fanout_by_src: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing["fanout_by_src"]
        if use_mp_fanout and fanout_by_src:
            packed_plan = routing.get("mp_fanout_packed_plan")
            if packed_plan is not None:
                x_parts: list[Tensor] = []
                src_rows_actual: list[int] = []
                try:
                    for src in packed_plan["src_order"]:
                        if src not in x_dict:
                            raise KeyError(f"Missing src node type {src!r} in x_dict for fanout.")
                        x_src = x_dict[src]
                        x_parts.append(x_src)
                        src_rows_actual.append(int(x_src.size(0)))
                    if tuple(src_rows_actual) != tuple(packed_plan["src_rows"]):
                        raise ValueError(
                            "Fanout packed plan source row sizes changed: "
                            f"expected {packed_plan['src_rows']}, got {tuple(src_rows_actual)}."
                        )
                    x_cat = _cat_or_single(x_parts, dim=0)
                    if int(x_cat.size(0)) != int(packed_plan["x_rows"]):
                        raise ValueError(
                            f"Fanout packed plan x_rows mismatch: expected {packed_plan['x_rows']}, got {int(x_cat.size(0))}."
                        )
                    args_flat_all = relm_mp_ops.fanout_scatter(  # type: ignore[union-attr]
                        x_cat,
                        packed_plan["src_global"],
                        packed_plan["flat_dst"],
                        int(total_slots),
                    )
                except Exception:
                    if self.validate_routing:
                        raise
                    packed_plan = None
            if packed_plan is None:
                edge_plan = routing.get("mp_fanout_edge_plan")
                if edge_plan is not None and len(edge_plan["edge_src_parts"]) > 0:
                    try:
                        x_parts = []
                        for src in edge_plan["src_order"]:
                            if src not in x_dict:
                                raise KeyError(f"Missing src node type {src!r} in x_dict for fanout.")
                            x_parts.append(x_dict[src])
                        x_cat, src_global, flat_dst = relm_mp_ops.fanout_pack_from_edges(  # type: ignore[union-attr]
                            x_parts,
                            list(edge_plan["edge_src_parts"]),
                            list(edge_plan["edge_dst_parts"]),
                            list(edge_plan["src_part_ids"]),
                            list(edge_plan["arity_parts"]),
                            list(edge_plan["pos_parts"]),
                            list(edge_plan["slot_offset_parts"]),
                        )
                        args_flat_all = relm_mp_ops.fanout_scatter(  # type: ignore[union-attr]
                            x_cat, src_global, flat_dst, int(total_slots)
                        )
                    except Exception:
                        if self.validate_routing:
                            raise
                        edge_plan = None
                if edge_plan is None:
                    fanout_plan = routing.get("mp_fanout_plan")
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
                    raise KeyError(f"Missing src node type {src!r} in x_dict for fanout scatter.")
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
                        rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                            -1, self.embedding_size
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
        relation_arities: Mapping[str, int] | None,
        aggr: Aggregation | str | None,
        src_types: Iterable[str] | None = None,
        edge_labels: Iterable[str] | None = None,
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
        self.dst_types = tuple(dst_types) if not isinstance(dst_types, str) else (dst_types,)
        self.relation_arities = dict(relation_arities or {})
        self.src_types = (
            tuple(src_types)
            if (src_types is not None and not isinstance(src_types, str))
            else ((src_types,) if isinstance(src_types, str) else None)
        )
        self.edge_labels = (
            tuple(str(lbl) for lbl in edge_labels)
            if (edge_labels is not None and not isinstance(edge_labels, str))
            else ((str(edge_labels),) if isinstance(edge_labels, str) else None)
        )
        self._label_mode = self.edge_labels is not None
        self.aggr = aggr
        self._mp_fanin_mode = _resolve_fanin_mode(self.aggr)
        self.strict_filter_mode = bool(strict_filter_mode)
        self.validate_routing = bool(validate_routing)
        self._routing_cache_key = f"fanin::{id(self)}"

    def _match_dst(self, dst: str) -> bool:
        return _match_ntype(dst, self.dst_types, self.strict_filter_mode)

    def _match_src(self, src: str) -> bool:
        if self.src_types is None:
            return True
        return _match_ntype(src, self.src_types, self.strict_filter_mode)

    def _match_edge_label(self, rel: str) -> bool:
        if self.edge_labels is None:
            return True
        return str(rel) in self.edge_labels

    def _forward_label_lane(
        self,
        *,
        x_dict: Dict[str, Tensor],
        fanin_by_dst: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]],
        mp_fanin_label_plans: dict[str, dict[str, Any]],
        mp_fanin_edge_plans: dict[str, dict[str, Any]],
        mp_fanin_packed_plans: dict[str, dict[str, Any]],
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for dst in self.dst_types:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            use_mp_fanin_reduce = (
                self._mp_fanin_mode is not None
                and _use_model_mp_batched_fanin_reduce(x_dict[dst])
            )
            use_mp_fanin_pack = _use_model_mp_batched_fanin_pack(x_dict[dst])
            use_mp_fanin = use_mp_fanin_reduce or use_mp_fanin_pack

            packed_plan = mp_fanin_packed_plans.get(dst)
            if use_mp_fanin and packed_plan is not None:
                rel_parts: list[Tensor] = []
                plan_ok = True
                for src, expected_rows in zip(
                    packed_plan["rel_order"], packed_plan["rel_rows"]
                ):
                    rel_src = x_dict.get(src)
                    if rel_src is None:
                        plan_ok = False
                        break
                    if int(rel_src.size(-1)) != self.embedding_size:
                        raise ValueError(
                            f"Label fanin source {src!r} has embedding dim {int(rel_src.size(-1))} "
                            f"(expected {self.embedding_size})."
                        )
                    if self.validate_routing and int(rel_src.size(0)) != int(expected_rows):
                        raise ValueError(
                            f"Label fanin packed plan size changed for {src!r}: "
                            f"expected {int(expected_rows)}, got {int(rel_src.size(0))}."
                        )
                    rel_parts.append(rel_src)
                if plan_ok and rel_parts:
                    rel_cat = _cat_or_single(rel_parts, dim=0)
                    if self.validate_routing and int(rel_cat.size(0)) != int(
                        packed_plan["rel_cat_rows"]
                    ):
                        raise ValueError(
                            "Label fanin packed plan row mismatch: "
                            f"expected {int(packed_plan['rel_cat_rows'])}, got {int(rel_cat.size(0))}."
                        )
                    if use_mp_fanin_reduce:
                        out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                            rel_cat,
                            packed_plan["flat_src"],
                            packed_plan["dst_idx"],
                            dim_size,
                            self._mp_fanin_mode,
                        )
                    else:
                        flat_inputs = rel_cat.index_select(0, packed_plan["flat_src"])
                        out[dst] = self.aggr(
                            x=flat_inputs,
                            index=packed_plan["dst_idx"],
                            dim=0,
                            dim_size=dim_size,
                        )
                    continue

            edge_plan = mp_fanin_edge_plans.get(dst)
            if use_mp_fanin and edge_plan is not None and len(edge_plan["edge_src_parts"]) > 0:
                rel_parts = []
                plan_ok = True
                for src in edge_plan["rel_order"]:
                    rel_src = x_dict.get(src)
                    if rel_src is None:
                        plan_ok = False
                        break
                    if int(rel_src.size(-1)) != self.embedding_size:
                        raise ValueError(
                            f"Label fanin source {src!r} has embedding dim {int(rel_src.size(-1))} "
                            f"(expected {self.embedding_size})."
                        )
                    rel_parts.append(rel_src)
                if plan_ok and rel_parts:
                    rel_cat, flat_src_all, dst_all = relm_mp_ops.fanin_pack_from_edges(  # type: ignore[union-attr]
                        rel_parts,
                        list(edge_plan["edge_src_parts"]),
                        list(edge_plan["edge_dst_parts"]),
                        list(edge_plan["rel_part_ids"]),
                        list(edge_plan["arity_parts"]),
                        list(edge_plan["pos_parts"]),
                        int(edge_plan["mode"]),
                    )
                    if use_mp_fanin_reduce:
                        out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                            rel_cat,
                            flat_src_all,
                            dst_all,
                            dim_size,
                            self._mp_fanin_mode,
                        )
                    else:
                        flat_inputs = rel_cat.index_select(0, flat_src_all)
                        out[dst] = self.aggr(
                            x=flat_inputs,
                            index=dst_all,
                            dim=0,
                            dim_size=dim_size,
                        )
                    continue

            if use_mp_fanin_reduce:
                plan = mp_fanin_label_plans.get(dst)
                if plan is not None:
                    rel_parts: list[torch.Tensor] = []
                    plan_ok = True
                    for src in plan["src_order"]:
                        rel_src = x_dict.get(src)
                        if rel_src is None:
                            plan_ok = False
                            break
                        if int(rel_src.size(-1)) != self.embedding_size:
                            raise ValueError(
                                f"Label fanin source {src!r} has embedding dim {int(rel_src.size(-1))} "
                                f"(expected {self.embedding_size})."
                            )
                        rel_parts.append(rel_src)
                    if plan_ok and rel_parts:
                        rel_cat = _cat_or_single(rel_parts, dim=0)
                        if (not self.validate_routing) or int(rel_cat.size(0)) == int(
                            plan["total_src_rows"]
                        ):
                            out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                                rel_cat,
                                plan["flat_src"],
                                plan["dst_idx"],
                                dim_size,
                                self._mp_fanin_mode,
                            )
                            continue

            per_src = fanin_by_dst.get(dst)
            if not per_src:
                out[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue
            inputs = []
            indices = []
            rel_parts = []
            mp_src_parts = []
            mp_dst_parts = []
            for src, (src_idx, dst_idx) in per_src.items():
                if src not in x_dict:
                    continue
                x_src = x_dict[src]
                if x_src.numel() == 0:
                    continue
                if int(x_src.size(-1)) != self.embedding_size:
                    raise ValueError(
                        f"Label fanin source {src!r} has embedding dim {int(x_src.size(-1))} "
                        f"(expected {self.embedding_size})."
                    )
                if use_mp_fanin:
                    rel_parts.append(x_src)
                    mp_src_parts.append(src_idx)
                    mp_dst_parts.append(dst_idx)
                else:
                    inputs.append(x_src.index_select(0, src_idx))
                    indices.append(dst_idx)

            if use_mp_fanin and rel_parts:
                rel_cat, flat_src_all, dst_all = relm_mp_ops.fanin_pack_multi(  # type: ignore[union-attr]
                    rel_parts,
                    mp_src_parts,
                    mp_dst_parts,
                )
                if use_mp_fanin_reduce:
                    out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                        rel_cat,
                        flat_src_all,
                        dst_all,
                        dim_size,
                        self._mp_fanin_mode,
                    )
                else:
                    flat_inputs = rel_cat.index_select(0, flat_src_all)
                    out[dst] = self.aggr(
                        x=flat_inputs,
                        index=dst_all,
                        dim=0,
                        dim_size=dim_size,
                    )
                continue

            if not inputs:
                out[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue

            flat_inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
            flat_indices = indices[0] if len(indices) == 1 else torch.cat(indices, dim=0)
            out[dst] = self.aggr(x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_size)
        return out

    def _forward_relation_lane(
        self,
        *,
        x_dict: Dict[str, Tensor],
        fanin_by_dst: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]],
        mp_fanin_plans: dict[str, dict[str, Any]],
        mp_fanin_global_plans: dict[str, dict[str, Any]],
        mp_fanin_edge_plans: dict[str, dict[str, Any]],
        mp_fanin_packed_plans: dict[str, dict[str, Any]],
        rel_flat_shared: torch.Tensor | None,
    ) -> Dict[str, Tensor]:
        out: Dict[str, Tensor] = {}
        for dst in self.dst_types:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            use_mp_fanin_reduce = (
                self._mp_fanin_mode is not None
                and _use_model_mp_batched_fanin_reduce(x_dict[dst])
            )
            use_mp_fanin_pack = _use_model_mp_batched_fanin_pack(x_dict[dst])
            use_mp_fanin = use_mp_fanin_reduce or use_mp_fanin_pack

            # Prefer the canonical flattened fanout handoff when available.
            global_plan = mp_fanin_global_plans.get(dst)
            if global_plan is not None and isinstance(rel_flat_shared, torch.Tensor):
                if (
                    rel_flat_shared.device == x_dict[dst].device
                    and int(rel_flat_shared.size(-1)) == self.embedding_size
                    and (
                        (not self.validate_routing)
                        or int(rel_flat_shared.size(0)) == int(global_plan["total_slots"])
                    )
                ):
                    if use_mp_fanin_reduce:
                        out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                            rel_flat_shared,
                            global_plan["flat_src"],
                            global_plan["dst_idx"],
                            dim_size,
                            self._mp_fanin_mode,
                        )
                    else:
                        flat_inputs = rel_flat_shared.index_select(0, global_plan["flat_src"])
                        out[dst] = self.aggr(
                            x=flat_inputs,
                            index=global_plan["dst_idx"],
                            dim=0,
                            dim_size=dim_size,
                        )
                    continue

            packed_plan = mp_fanin_packed_plans.get(dst)
            if use_mp_fanin and packed_plan is not None:
                rel_parts: list[Tensor] = []
                plan_ok = True
                for pred, expected_rows in zip(packed_plan["rel_order"], packed_plan["rel_rows"]):
                    rel_pred = x_dict.get(pred)
                    if rel_pred is None:
                        plan_ok = False
                        break
                    arity = int(self.relation_arities.get(pred, 0))
                    exp = arity * self.embedding_size
                    if int(rel_pred.size(-1)) != exp:
                        raise ValueError(
                            f"Predicate {pred!r} has arity {arity}, but embedding dim is {int(rel_pred.size(-1))} (expected {exp})."
                        )
                    rel_flat = rel_pred.view(-1, self.embedding_size)
                    if self.validate_routing and int(rel_flat.size(0)) != int(expected_rows):
                        raise ValueError(
                            f"Relation fanin packed plan size changed for {pred!r}: "
                            f"expected {int(expected_rows)}, got {int(rel_flat.size(0))}."
                        )
                    rel_parts.append(rel_flat)
                if plan_ok and rel_parts:
                    rel_cat = _cat_or_single(rel_parts, dim=0)
                    if self.validate_routing and int(rel_cat.size(0)) != int(
                        packed_plan["rel_cat_rows"]
                    ):
                        raise ValueError(
                            "Relation fanin packed plan row mismatch: "
                            f"expected {int(packed_plan['rel_cat_rows'])}, got {int(rel_cat.size(0))}."
                        )
                    if use_mp_fanin_reduce:
                        out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                            rel_cat,
                            packed_plan["flat_src"],
                            packed_plan["dst_idx"],
                            dim_size,
                            self._mp_fanin_mode,
                        )
                    else:
                        flat_inputs = rel_cat.index_select(0, packed_plan["flat_src"])
                        out[dst] = self.aggr(
                            x=flat_inputs,
                            index=packed_plan["dst_idx"],
                            dim=0,
                            dim_size=dim_size,
                        )
                    continue

            edge_plan = mp_fanin_edge_plans.get(dst)
            if use_mp_fanin and edge_plan is not None and len(edge_plan["edge_src_parts"]) > 0:
                rel_parts = []
                plan_ok = True
                for pred in edge_plan["rel_order"]:
                    rel_pred = x_dict.get(pred)
                    if rel_pred is None:
                        plan_ok = False
                        break
                    arity = int(self.relation_arities.get(pred, 0))
                    exp = arity * self.embedding_size
                    if int(rel_pred.size(-1)) != exp:
                        raise ValueError(
                            f"Predicate {pred!r} has arity {arity}, but embedding dim is {int(rel_pred.size(-1))} (expected {exp})."
                        )
                    rel_parts.append(rel_pred.view(-1, self.embedding_size))
                if plan_ok and rel_parts:
                    rel_cat, flat_src_all, dst_all = relm_mp_ops.fanin_pack_from_edges(  # type: ignore[union-attr]
                        rel_parts,
                        list(edge_plan["edge_src_parts"]),
                        list(edge_plan["edge_dst_parts"]),
                        list(edge_plan["rel_part_ids"]),
                        list(edge_plan["arity_parts"]),
                        list(edge_plan["pos_parts"]),
                        int(edge_plan["mode"]),
                    )
                    if use_mp_fanin_reduce:
                        out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                            rel_cat,
                            flat_src_all,
                            dst_all,
                            dim_size,
                            self._mp_fanin_mode,
                        )
                    else:
                        flat_inputs = rel_cat.index_select(0, flat_src_all)
                        out[dst] = self.aggr(
                            x=flat_inputs,
                            index=dst_all,
                            dim=0,
                            dim_size=dim_size,
                        )
                    continue

            if use_mp_fanin_reduce:
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
                        if (not self.validate_routing) or int(rel_cat.size(0)) == int(
                            plan["total_rel_rows"]
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
                    mp_src_parts.append(flat_src)
                    mp_dst_parts.append(dst_idx)
                else:
                    inputs.append(rel_flat.index_select(0, flat_src))
                    indices.append(dst_idx)

            if use_mp_fanin and rel_parts:
                rel_cat, flat_src_all, dst_all = relm_mp_ops.fanin_pack_multi(  # type: ignore[union-attr]
                    rel_parts,
                    mp_src_parts,
                    mp_dst_parts,
                )
                if use_mp_fanin_reduce:
                    out[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                        rel_cat,
                        flat_src_all,
                        dst_all,
                        dim_size,
                        self._mp_fanin_mode,
                    )
                else:
                    flat_inputs = rel_cat.index_select(0, flat_src_all)
                    out[dst] = self.aggr(
                        x=flat_inputs,
                        index=dst_all,
                        dim=0,
                        dim_size=dim_size,
                    )
                continue

            if not inputs:
                out[dst] = x_dict[dst].new_zeros((dim_size, self.embedding_size))
                continue

            flat_inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=0)
            flat_indices = indices[0] if len(indices) == 1 else torch.cat(indices, dim=0)
            out[dst] = self.aggr(x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_size)

        return out

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        *,
        cache: dict | None = None,
        relation_messages: Mapping[str, Tensor] | None = None,
        **_: Any,
    ) -> Dict[str, Tensor]:
        if not edge_index_dict:
            return {}
        if relation_messages:
            merged_x_dict = dict(x_dict)
            merged_x_dict.update(relation_messages)
            x_dict = merged_x_dict
        cache = cache if cache is not None else {}

        # See BatchedFanOutMP for the shared cache layout. We store fanin routing under
        # a per-instance key in cache["routing"] to avoid collisions with fanout routing
        # and with other BatchedFanInMP modules sharing one cache dict.
        routing_root = cache.setdefault("routing", {})
        routing = routing_root.get(self._routing_cache_key)
        if routing is None:
            # fanin_by_dst[dst][pred] = (flat_src, dst_idx)
            tmp: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
            tmp_cpp_edges: dict[
                tuple[str, str], list[tuple[torch.Tensor, torch.Tensor, int, int]]
            ] = defaultdict(list)
            for edge_type, edge_index in edge_index_dict.items():
                src, rel, dst = edge_type
                if not self._match_dst(dst):
                    continue
                if edge_index is None or edge_index.numel() == 0:
                    continue

                if self._label_mode:
                    if not self._match_src(src):
                        continue
                    if not self._match_edge_label(rel):
                        continue
                    tmp[(dst, src)].append((edge_index[0], edge_index[1]))
                    tmp_cpp_edges[(dst, src)].append((edge_index[0], edge_index[1], 1, 0))
                    continue

                if src not in self.relation_arities:
                    continue
                try:
                    pos = int(rel)
                except (TypeError, ValueError):
                    if self.validate_routing:
                        raise AssertionError(
                            f"Fanin routing expects integer edge positions, got rel={rel!r} for pred={src!r}."
                        )
                    continue
                arity = int(self.relation_arities.get(src, 0))
                if pos < 0 or pos >= arity:
                    if self.validate_routing and arity > 0:
                        raise AssertionError(
                            f"Fanin routing pos out of range: pred={src!r} pos={pos} arity={arity}."
                        )
                    continue
                flat_src = edge_index[0] * arity + pos
                tmp[(dst, src)].append((flat_src, edge_index[1]))
                tmp_cpp_edges[(dst, src)].append((edge_index[0], edge_index[1], arity, pos))

            fanin_by_dst: dict[str, dict[str, tuple[torch.Tensor, torch.Tensor]]] = defaultdict(dict)
            for (dst, pred), parts in tmp.items():
                flat_srcs = [p[0] for p in parts]
                dst_idxs = [p[1] for p in parts]
                flat_src = _cat_or_single(flat_srcs, dim=0)
                dst_idx = _cat_or_single(dst_idxs, dim=0)
                fanin_by_dst[dst][pred] = (flat_src, dst_idx)

            mp_fanin_plans: dict[str, dict[str, Any]] = {}
            mp_fanin_global_plans: dict[str, dict[str, Any]] = {}
            mp_fanin_label_plans: dict[str, dict[str, Any]] = {}
            mp_fanin_edge_plans: dict[str, dict[str, Any]] = {}
            mp_fanin_packed_plans: dict[str, dict[str, Any]] = {}
            if self._label_mode:
                for dst, per_src in fanin_by_dst.items():
                    src_order: list[str] = []
                    src_rows: list[int] = []
                    flat_src_parts: list[torch.Tensor] = []
                    dst_parts: list[torch.Tensor] = []
                    src_offset = 0
                    for src in sorted(per_src.keys()):
                        if src not in x_dict:
                            continue
                        x_src = x_dict[src]
                        if x_src.numel() == 0:
                            continue
                        rows = int(x_src.size(0))
                        src_idx, dst_idx = per_src[src]
                        src_order.append(src)
                        src_rows.append(rows)
                        flat_src_parts.append(src_idx + int(src_offset))
                        dst_parts.append(dst_idx)
                        src_offset += rows
                    if src_order:
                        mp_fanin_label_plans[dst] = {
                            "src_order": tuple(src_order),
                            "src_rows": tuple(src_rows),
                            "total_src_rows": int(src_offset),
                            "flat_src": _cat_or_single(flat_src_parts, dim=0),
                            "dst_idx": _cat_or_single(dst_parts, dim=0),
                        }
            else:
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

            by_dst_cpp_edges: dict[str, dict[str, list[tuple[Tensor, Tensor, int, int]]]] = defaultdict(dict)
            for (dst, rel_name), entries in tmp_cpp_edges.items():
                by_dst_cpp_edges[dst][rel_name] = entries
            for dst, per_rel in by_dst_cpp_edges.items():
                rel_order = tuple(sorted(per_rel.keys()))
                rel_part_index = {rel_name: idx for idx, rel_name in enumerate(rel_order)}
                edge_src_parts: list[Tensor] = []
                edge_dst_parts: list[Tensor] = []
                rel_part_ids: list[int] = []
                arity_parts: list[int] = []
                pos_parts: list[int] = []
                for rel_name in rel_order:
                    for src_idx, dst_idx, arity, pos in per_rel[rel_name]:
                        edge_src_parts.append(src_idx)
                        edge_dst_parts.append(dst_idx)
                        rel_part_ids.append(int(rel_part_index[rel_name]))
                        arity_parts.append(int(arity))
                        pos_parts.append(int(pos))
                mp_fanin_edge_plans[dst] = {
                    "rel_order": rel_order,
                    "edge_src_parts": tuple(edge_src_parts),
                    "edge_dst_parts": tuple(edge_dst_parts),
                    "rel_part_ids": tuple(rel_part_ids),
                    "arity_parts": tuple(arity_parts),
                    "pos_parts": tuple(pos_parts),
                    "mode": int(1 if self._label_mode else 0),
                }

            for dst, edge_plan in mp_fanin_edge_plans.items():
                if len(edge_plan["edge_src_parts"]) == 0:
                    continue
                rel_parts: list[Tensor] = []
                rel_rows: list[int] = []
                plan_ok = True
                for rel_name in edge_plan["rel_order"]:
                    rel_tensor = x_dict.get(rel_name)
                    if rel_tensor is None:
                        plan_ok = False
                        break
                    if self._label_mode:
                        if int(rel_tensor.size(-1)) != self.embedding_size:
                            raise ValueError(
                                f"Label fanin source {rel_name!r} has embedding dim {int(rel_tensor.size(-1))} "
                                f"(expected {self.embedding_size})."
                            )
                        rel_flat = rel_tensor
                    else:
                        arity = int(self.relation_arities.get(rel_name, 0))
                        exp = arity * self.embedding_size
                        if int(rel_tensor.size(-1)) != exp:
                            raise ValueError(
                                f"Predicate {rel_name!r} has arity {arity}, but embedding dim is {int(rel_tensor.size(-1))} (expected {exp})."
                            )
                        rel_flat = rel_tensor.view(-1, self.embedding_size)
                    rel_parts.append(rel_flat)
                    rel_rows.append(int(rel_flat.size(0)))
                if not plan_ok or not rel_parts:
                    continue
                try:
                    rel_cat_pre, flat_src_pre, dst_pre = relm_mp_ops.fanin_pack_from_edges(  # type: ignore[union-attr]
                        rel_parts,
                        list(edge_plan["edge_src_parts"]),
                        list(edge_plan["edge_dst_parts"]),
                        list(edge_plan["rel_part_ids"]),
                        list(edge_plan["arity_parts"]),
                        list(edge_plan["pos_parts"]),
                        int(edge_plan["mode"]),
                    )
                except Exception:
                    if self.validate_routing:
                        raise
                    continue
                mp_fanin_packed_plans[dst] = {
                    "rel_order": edge_plan["rel_order"],
                    "rel_rows": tuple(rel_rows),
                    "rel_cat_rows": int(rel_cat_pre.size(0)),
                    "flat_src": flat_src_pre,
                    "dst_idx": dst_pre,
                }

            routing = {
                "fanin_by_dst": dict(fanin_by_dst),
                "mp_fanin_plans": mp_fanin_plans,
                "mp_fanin_global_plans": mp_fanin_global_plans,
                "mp_fanin_label_plans": mp_fanin_label_plans,
                "mp_fanin_edge_plans": mp_fanin_edge_plans,
                "mp_fanin_packed_plans": mp_fanin_packed_plans,
            }
            routing_root[self._routing_cache_key] = routing

        fanin_by_dst = routing["fanin_by_dst"]
        mp_fanin_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_plans", {})
        mp_fanin_global_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_global_plans", {})
        mp_fanin_label_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_label_plans", {})
        mp_fanin_edge_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_edge_plans", {})
        mp_fanin_packed_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_packed_plans", {})
        layer_state = cache.get("layer_state")
        rel_flat_shared = layer_state.get("fanout_rel_flat_all") if isinstance(layer_state, dict) else None
        if self._label_mode:
            return self._forward_label_lane(
                x_dict=x_dict,
                fanin_by_dst=fanin_by_dst,
                mp_fanin_label_plans=mp_fanin_label_plans,
                mp_fanin_edge_plans=mp_fanin_edge_plans,
                mp_fanin_packed_plans=mp_fanin_packed_plans,
            )

        return self._forward_relation_lane(
            x_dict=x_dict,
            fanin_by_dst=fanin_by_dst,
            mp_fanin_plans=mp_fanin_plans,
            mp_fanin_global_plans=mp_fanin_global_plans,
            mp_fanin_edge_plans=mp_fanin_edge_plans,
            mp_fanin_packed_plans=mp_fanin_packed_plans,
            rel_flat_shared=rel_flat_shared,
        )
