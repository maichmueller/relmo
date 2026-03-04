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
from ._grouped_exec import _apply_residual_truncate, _extract_grouped_residual_mlp_info
from ._ops_env import (
    _env_bool,
    _resolve_fanin_mode,
    _use_grouped_relation_mlp,
    _use_model_mp_fanin,
    _use_model_mp_fanout,
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
        self._grouped_mlp_info_cache: dict[int, dict[str, Any] | None] = {}

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

    def _grouped_mlp_info(self, module: torch.nn.Module) -> dict[str, Any] | None:
        key = id(module)
        cached = self._grouped_mlp_info_cache.get(key)
        if cached is not None or key in self._grouped_mlp_info_cache:
            return cached
        info = _extract_grouped_residual_mlp_info(module)
        self._grouped_mlp_info_cache[key] = info
        return info

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
            fanout_plan = _build_fanout_scatter_plan(by_src=fanout_by_src, x_dict=x_dict)

            pred_exec = []
            for pred, n, arity, slot_offset in pred_meta:
                if pred not in self.update_modules:
                    continue
                pred_exec.append((pred, n, arity, slot_offset, self.update_modules[pred]))

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

        fanout_by_src: dict[str, tuple[torch.Tensor, torch.Tensor]] = routing["fanout_by_src"]
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
                    raise KeyError(f"Missing src node type {src!r} in x_dict for fanout scatter.")
                vals = x_dict[src].index_select(0, src_idx)
                args_flat_all.index_copy_(0, flat_dst, vals)

        use_grouped_mlp = _use_grouped_relation_mlp(ref)
        if use_grouped_mlp:
            grouped_exec: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
            fallback_exec: list[tuple[Any, ...]] = []
            for entry in pred_exec:
                pred, n, arity, slot_offset, update_module = entry
                info = self._grouped_mlp_info(update_module)
                if info is None:
                    fallback_exec.append(entry)
                    continue
                grouped_exec[info["signature"]].append(
                    (pred, n, arity, slot_offset, update_module, info)
                )

            for group_items in grouped_exec.values():
                if len(group_items) <= 1:
                    pred, n, arity, slot_offset, update_module, _info = group_items[0]
                    args_flat = args_flat_all[slot_offset : slot_offset + (n * arity)]
                    args_feat = args_flat.view(n, arity * self.embedding_size)
                    out_pred = update_module(args_feat)
                    outputs[pred] = out_pred
                    if rel_flat_all is not None:
                        rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                            -1, self.embedding_size
                        )
                    continue

                batch_items = list(group_items)
                max_rows = max(int(n) for _pred, n, _arity, _slot, _mod, _info in batch_items)
                arity = int(batch_items[0][2])
                in_dim = int(arity * self.embedding_size)
                x_stack = args_flat_all.new_zeros((len(batch_items), max_rows, in_dim))
                valid_rows: list[int] = []
                for i, (_pred, n, _arity, slot_offset, _mod, _info) in enumerate(batch_items):
                    n_i = int(n)
                    valid_rows.append(n_i)
                    if n_i <= 0:
                        continue
                    args_flat = args_flat_all[slot_offset : slot_offset + (n_i * arity)]
                    x_stack[i, :n_i, :] = args_flat.view(n_i, in_dim)

                out_stack = x_stack
                template_info = batch_items[0][5]
                ops = template_info["ops"]
                for op_kind, op_payload in ops:
                    if op_kind == "linear":
                        lin_idx = int(op_payload)
                        weight_stack = torch.stack(
                            [item[5]["linears"][lin_idx].weight for item in batch_items],
                            dim=0,
                        )
                        out_stack = torch.matmul(out_stack, weight_stack.transpose(1, 2))
                        bias_ref = template_info["linears"][lin_idx].bias
                        if bias_ref is not None:
                            bias_stack = torch.stack(
                                [item[5]["linears"][lin_idx].bias for item in batch_items],
                                dim=0,
                            )
                            out_stack = out_stack + bias_stack[:, None, :]
                    else:
                        out_stack = op_payload(out_stack)

                out_stack = _apply_residual_truncate(
                    x=x_stack,
                    y=out_stack,
                    truncated_dim=template_info["truncated_dim"],
                    truncate_right=template_info["truncate_right"],
                )
                for i, (pred, n, _arity, slot_offset, _mod, _info) in enumerate(batch_items):
                    n_i = int(n)
                    out_pred = out_stack[i, :n_i, :]
                    outputs[pred] = out_pred
                    if rel_flat_all is not None:
                        rel_flat_all[slot_offset : slot_offset + (n_i * arity)] = out_pred.view(
                            -1, self.embedding_size
                        )

            for pred, n, arity, slot_offset, update_module in fallback_exec:
                args_flat = args_flat_all[slot_offset : slot_offset + (n * arity)]
                args_feat = args_flat.view(n, arity * self.embedding_size)
                out_pred = update_module(args_feat)
                outputs[pred] = out_pred
                if rel_flat_all is not None:
                    rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                        -1, self.embedding_size
                    )
        else:
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
        # a per-instance key in cache["routing"] to avoid collisions with fanout routing
        # and with other BatchedFanInMP modules sharing one cache dict.
        routing_root = cache.setdefault("routing", {})
        routing = routing_root.get(self._routing_cache_key)
        if routing is None:
            # fanin_by_dst[dst][pred] = (flat_src, dst_idx)
            tmp: dict[tuple[str, str], list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
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

            routing = {
                "fanin_by_dst": dict(fanin_by_dst),
                "mp_fanin_plans": mp_fanin_plans,
                "mp_fanin_global_plans": mp_fanin_global_plans,
                "mp_fanin_label_plans": mp_fanin_label_plans,
            }
            routing_root[self._routing_cache_key] = routing

        fanin_by_dst = routing["fanin_by_dst"]
        mp_fanin_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_plans", {})
        mp_fanin_global_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_global_plans", {})
        mp_fanin_label_plans: dict[str, dict[str, Any]] = routing.get("mp_fanin_label_plans", {})
        layer_state = cache.get("layer_state")
        rel_flat_shared = layer_state.get("fanout_rel_flat_all") if isinstance(layer_state, dict) else None
        out: Dict[str, Tensor] = {}

        for dst in self.dst_types:
            if dst not in x_dict:
                continue
            dim_size = int(x_dict[dst].size(0))
            use_mp_fanin = self._mp_fanin_mode is not None and _use_model_mp_fanin(x_dict[dst])
            use_mp_fanin = use_mp_fanin and _env_bool("RELM_MODELS_MP_FANIN_BATCHED", True)
            # Training-first default:
            # Batched decentralized fanin custom kernels can regress end-to-end training-step
            # throughput depending on graph mix. Keep them enabled by default for inference
            # and allow explicit opt-in for training via env override.
            if self.training and not _env_bool(
                "RELM_MODELS_MP_FANIN_BATCHED_TRAINING", False
            ):
                use_mp_fanin = False

            if self._label_mode:
                if use_mp_fanin:
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
                src_offset = 0
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
                        mp_src_parts.append(src_idx + int(src_offset))
                        mp_dst_parts.append(dst_idx)
                        src_offset += int(x_src.size(0))
                    else:
                        inputs.append(x_src.index_select(0, src_idx))
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
                flat_indices = indices[0] if len(indices) == 1 else torch.cat(indices, dim=0)
                out[dst] = self.aggr(x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_size)
                continue

            if use_mp_fanin:
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
            flat_indices = indices[0] if len(indices) == 1 else torch.cat(indices, dim=0)
            out[dst] = self.aggr(x=flat_inputs, index=flat_indices, dim=0, dim_size=dim_size)

        return out
