from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType

from ._grouped_exec import _apply_residual_truncate, _extract_grouped_residual_mlp_info
from ._ops_env import (
    _resolve_fanin_mode,
    _use_grouped_relation_mlp,
    _use_model_mp_batched_fanin_reduce,
    _use_model_mp_batched_fanout,
    relm_mp_ops,
)
from ._scatter import (
    _build_fanout_scatter_plan,
    _fanout_scatter_from_plan,
    _fanout_scatter_multi_src,
)
from ._tensor_utils import _finalize_pair_lists, _get_or_make_buffer, _match_ntype
from ..patched_module_dict import PatchedModuleDict

_PW_IDENTITY = 0
_PW_RELU = 1
_PW_MISH = 2
_PW_GELU_NONE = 3
_PW_GELU_TANH = 4
_PW_SILU = 5
_PW_TANH = 6


def _pointwise_op_code(module: torch.nn.Module) -> int | None:
    if isinstance(module, torch.nn.Identity):
        return _PW_IDENTITY
    if isinstance(module, torch.nn.ReLU):
        return _PW_RELU
    if isinstance(module, torch.nn.Mish):
        return _PW_MISH
    if isinstance(module, torch.nn.GELU):
        return _PW_GELU_TANH if str(module.approximate) == "tanh" else _PW_GELU_NONE
    if isinstance(module, torch.nn.SiLU):
        return _PW_SILU
    if isinstance(module, torch.nn.Tanh):
        return _PW_TANH
    return None


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
        self._grouped_mlp_info_cache: dict[int, dict[str, Any] | None] = {}
        self._persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]] = {}

    def _match_src(self, src: str) -> bool:
        return _match_ntype(src, self.src_types, self.strict_filter_mode)

    def _match_dst(self, dst: str) -> bool:
        return _match_ntype(dst, self.dst_types, self.strict_filter_mode)

    def _grouped_mlp_info(self, module: torch.nn.Module) -> dict[str, Any] | None:
        key = id(module)
        cached = self._grouped_mlp_info_cache.get(key)
        if cached is not None or key in self._grouped_mlp_info_cache:
            return cached
        info = _extract_grouped_residual_mlp_info(module)
        self._grouped_mlp_info_cache[key] = info
        return info

    def _get_grouped_param_stack(
        self,
        *,
        cache_key: tuple[Any, ...],
        tensors: list[Tensor],
        forward_cache: dict[tuple[Any, ...], Tensor],
        allow_persistent: bool,
    ) -> Tensor:
        cached_forward = forward_cache.get(cache_key)
        if cached_forward is not None:
            return cached_forward

        if allow_persistent and tensors:
            versions = tuple(int(getattr(tensor, "_version", -1)) for tensor in tensors)
            persistent = self._persistent_grouped_param_stacks.get(cache_key)
            if persistent is not None:
                stacked = persistent.get("tensor")
                if (
                    torch.is_tensor(stacked)
                    and persistent.get("versions") == versions
                    and tuple(stacked.shape) == tuple(persistent.get("shape", ()))
                    and stacked.device == tensors[0].device
                    and stacked.dtype == tensors[0].dtype
                ):
                    forward_cache[cache_key] = stacked
                    return stacked

        stacked = torch.stack(tensors, dim=0)
        forward_cache[cache_key] = stacked
        if allow_persistent and tensors:
            self._persistent_grouped_param_stacks[cache_key] = {
                "tensor": stacked,
                "versions": tuple(int(getattr(tensor, "_version", -1)) for tensor in tensors),
                "shape": tuple(stacked.shape),
            }
        return stacked

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
        use_grouped_mlp = _use_grouped_relation_mlp(ref)
        if use_grouped_mlp:
            grouped_layout = routing.get("grouped_layout")
            if grouped_layout is None:
                grouped_exec: dict[tuple[Any, ...], list[tuple[Any, ...]]] = defaultdict(list)
                fallback_exec: list[tuple[Any, ...]] = []
                for entry in pred_meta:
                    pred, n, arity, slot_offset, update_module = entry
                    info = self._grouped_mlp_info(update_module)
                    if info is None:
                        fallback_exec.append(entry)
                        continue
                    grouped_exec[info["signature"]].append(
                        (pred, n, arity, slot_offset, update_module, info)
                    )
                grouped_layout = {
                    "grouped_exec": grouped_exec,
                    "fallback_exec": fallback_exec,
                }
                routing["grouped_layout"] = grouped_layout
            grouped_exec = grouped_layout["grouped_exec"]
            fallback_exec = grouped_layout["fallback_exec"]
            # Per-forward cache: build grouped weight/bias stacks once and reuse across layers.
            grouped_param_stacks: dict[tuple[Any, ...], Tensor] = cache.setdefault(
                "grouped_param_stacks", {}
            )
            allow_persistent_stacks = reuse_buffers

            for group_items in grouped_exec.values():
                if len(group_items) <= 1:
                    pred, n, arity, slot_offset, update_module, _info = group_items[0]
                    args_flat = args_flat_all[slot_offset : slot_offset + (n * arity)]
                    args_feat = args_flat.view(n, arity * self.embedding_size)
                    out_pred = update_module(args_feat)
                    rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                        -1, self.embedding_size
                    )
                    continue

                batch_items = list(group_items)
                slot_offsets = [int(item[3]) for item in batch_items]
                row_sizes = [int(item[1]) for item in batch_items]
                arity = int(batch_items[0][2])
                template_info = batch_items[0][5]
                ops = template_info["ops"]
                group_pred_key = tuple(item[0] for item in batch_items)
                linear_indices = sorted(
                    {
                        int(payload)
                        for kind, payload in ops
                        if kind == "linear"
                    }
                )
                can_use_custom_grouped = (
                    relm_mp_ops is not None
                    and hasattr(relm_mp_ops, "grouped_residual_mlp_from_flat")
                )
                op_kinds: list[int] = []
                op_indices: list[int] = []
                pointwise_codes: list[int] = []
                linear_index_map = {lin_idx: i for i, lin_idx in enumerate(linear_indices)}
                if can_use_custom_grouped:
                    for op_kind, op_payload in ops:
                        if op_kind == "linear":
                            lin_idx = int(op_payload)
                            mapped_idx = linear_index_map.get(lin_idx)
                            if mapped_idx is None:
                                can_use_custom_grouped = False
                                break
                            op_kinds.append(0)
                            op_indices.append(int(mapped_idx))
                        elif op_kind == "pointwise":
                            code = _pointwise_op_code(op_payload)
                            if code is None:
                                can_use_custom_grouped = False
                                break
                            op_kinds.append(1)
                            op_indices.append(len(pointwise_codes))
                            pointwise_codes.append(int(code))
                        else:
                            can_use_custom_grouped = False
                            break

                weight_stacks: list[Tensor] = []
                bias_stacks: list[Tensor] = []
                for lin_idx in linear_indices:
                    weight_key = ("w", group_pred_key, lin_idx)
                    weight_stack = self._get_grouped_param_stack(
                        cache_key=weight_key,
                        tensors=[item[5]["linears"][lin_idx].weight for item in batch_items],
                        forward_cache=grouped_param_stacks,
                        allow_persistent=allow_persistent_stacks,
                    )
                    weight_stacks.append(weight_stack)
                    bias_ref = template_info["linears"][lin_idx].bias
                    if bias_ref is None:
                        bias_stacks.append(weight_stack.new_empty((0,)))
                    else:
                        bias_key = ("b", group_pred_key, lin_idx)
                        bias_stack = self._get_grouped_param_stack(
                            cache_key=bias_key,
                            tensors=[
                                item[5]["linears"][lin_idx].bias
                                for item in batch_items
                                if item[5]["linears"][lin_idx].bias is not None
                            ],
                            forward_cache=grouped_param_stacks,
                            allow_persistent=allow_persistent_stacks,
                        )
                        bias_stacks.append(bias_stack)

                truncated_dim = template_info["truncated_dim"]
                truncated_dim_i = int(truncated_dim) if truncated_dim is not None else -1
                truncated_dim_opt = (
                    None if truncated_dim is None else int(truncated_dim)
                )
                truncate_right_i = bool(template_info["truncate_right"])

                if can_use_custom_grouped:
                    rel_cat, flat_idx = relm_mp_ops.grouped_residual_mlp_from_flat(  # type: ignore[union-attr]
                        args_flat_all,
                        slot_offsets,
                        row_sizes,
                        arity,
                        weight_stacks,
                        bias_stacks,
                        op_kinds,
                        op_indices,
                        pointwise_codes,
                        truncated_dim_i,
                        truncate_right_i,
                    )
                    if rel_cat.numel() > 0:
                        rel_flat_all.index_copy_(0, flat_idx, rel_cat)
                    continue

                # Fallback: existing grouped Python path.
                max_rows = max(int(n) for _pred, n, _arity, _slot, _mod, _info in batch_items)
                in_dim = int(arity * self.embedding_size)
                x_stack = args_flat_all.new_zeros((len(batch_items), max_rows, in_dim))
                for i, (_pred, n, _arity, slot_offset, _mod, _info) in enumerate(batch_items):
                    n_i = int(n)
                    if n_i <= 0:
                        continue
                    args_flat = args_flat_all[slot_offset : slot_offset + (n_i * arity)]
                    x_stack[i, :n_i, :] = args_flat.view(n_i, in_dim)
                out_stack = x_stack
                for op_kind, op_payload in ops:
                    if op_kind == "linear":
                        lin_idx = int(op_payload)
                        mapped_idx = linear_index_map[lin_idx]
                        weight_stack = weight_stacks[mapped_idx]
                        out_stack = torch.matmul(out_stack, weight_stack.transpose(1, 2))
                        bias_stack = bias_stacks[mapped_idx]
                        if bias_stack.numel() > 0:
                            out_stack = out_stack + bias_stack[:, None, :]
                    else:
                        out_stack = op_payload(out_stack)

                out_stack = _apply_residual_truncate(
                    x=x_stack,
                    y=out_stack,
                    truncated_dim=truncated_dim_opt,
                    truncate_right=truncate_right_i,
                )
                for i, (pred, n, _arity, slot_offset, _mod, _info) in enumerate(batch_items):
                    n_i = int(n)
                    out_pred = out_stack[i, :n_i, :]
                    rel_flat_all[slot_offset : slot_offset + (n_i * arity)] = out_pred.view(
                        -1, self.embedding_size
                    )

            for pred, n, arity, slot_offset, update_module in fallback_exec:
                args_flat = args_flat_all[slot_offset : slot_offset + (n * arity)]
                args_feat = args_flat.view(n, arity * self.embedding_size)
                out_pred = update_module(args_feat)
                rel_flat_all[slot_offset : slot_offset + (n * arity)] = out_pred.view(
                    -1, self.embedding_size
                )
        else:
            for pred, n, arity, slot_offset, update_module in pred_meta:
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
                symbol_msgs[dst] = relm_mp_ops.fanin_reduce(  # type: ignore[union-attr]
                    rel_flat_all, flat_src, dst_index, dim_size, self._mp_fanin_mode
                )
            else:
                msgs = rel_flat_all.index_select(0, flat_src)
                symbol_msgs[dst] = self.aggr(x=msgs, index=dst_index, dim=0, dim_size=dim_size)

        return atom_msgs, symbol_msgs
