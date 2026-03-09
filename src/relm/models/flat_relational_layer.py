from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Sequence, cast

import torch
import torch_geometric.nn.aggr
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver

from ..ops import mp as relm_mp_ops
from .aggr import LogSumExpAggregation
from .grouped_mlp import GroupedMLPSpec


@dataclass(frozen=True)
class RelationSlice:
    relation_index: int
    count: int
    arity: int
    slot_start: int
    slot_end: int


@dataclass(frozen=True)
class FlatTopology:
    relation_counts_total: tuple[int, ...]
    relation_arities: tuple[int, ...]
    relation_slices: tuple[RelationSlice, ...]
    slot_offsets: tuple[int, ...]


@dataclass(frozen=True)
class GroupedRelationSliceBatch:
    family: str
    signature: Hashable
    arity: int
    relation_indices: tuple[int, ...]
    max_rows: int
    row_sizes: tuple[int, ...]


@dataclass(frozen=True)
class FusedRelationSpec:
    family: str
    signature: Hashable
    arity: int
    input_dim: int
    output_dim: int
    hidden_dims: tuple[int, ...]
    bias_flags: tuple[bool, ...]
    pointwise_signature: tuple[Any, ...] | None = None
    norm_kind: str | None = None
    norm_position: str | None = None


@dataclass(frozen=True)
class ProgramFamilySpec:
    family: str
    signature: Hashable
    arity: int
    input_dim: int
    output_dim: int
    block_specs: tuple[FusedRelationSpec, ...]


@dataclass(frozen=True)
class FusedRelationMatch:
    spec: FusedRelationSpec
    linears: tuple[torch.nn.Linear, ...]
    pointwise_modules: tuple[torch.nn.Module, ...] = ()
    norm_modules: tuple[torch.nn.Module, ...] = ()
    program_matches: tuple["FusedRelationMatch", ...] = ()
    program_family: ProgramFamilySpec | None = None


_GROUPED_SPEC_METHODS = ("relm_grouped_mlp_spec", "grouped_mlp_spec")
_SUPPORTED_POINTWISE_TYPES = (
    torch.nn.Identity,
    torch.nn.ReLU,
    torch.nn.Mish,
    torch.nn.GELU,
    torch.nn.SiLU,
    torch.nn.Tanh,
    torch.nn.ELU,
    torch.nn.LeakyReLU,
)
_LAYER_NORM_TYPE = torch.nn.LayerNorm
_RMS_NORM_TYPE = getattr(torch.nn, "RMSNorm", None)
_SUPPORTED_NORM_TYPES = (
    (_LAYER_NORM_TYPE,) if _RMS_NORM_TYPE is None else (_LAYER_NORM_TYPE, _RMS_NORM_TYPE)
)


def _pointwise_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
    if not isinstance(module, _SUPPORTED_POINTWISE_TYPES):
        return None
    if isinstance(module, torch.nn.GELU):
        return ("gelu", str(module.approximate))
    if isinstance(module, torch.nn.LeakyReLU):
        return ("leaky_relu", float(module.negative_slope), bool(module.inplace))
    if isinstance(module, torch.nn.ELU):
        return (
            "elu",
            float(module.alpha),
            float(module.scale),
            float(module.input_scale),
            bool(module.inplace),
        )
    if isinstance(module, torch.nn.ReLU):
        return ("relu", bool(module.inplace))
    if isinstance(module, torch.nn.Identity):
        return ("identity",)
    if isinstance(module, torch.nn.Mish):
        return ("mish", bool(module.inplace))
    if isinstance(module, torch.nn.SiLU):
        return ("silu", bool(module.inplace))
    if isinstance(module, torch.nn.Tanh):
        return ("tanh",)
    return None


def _norm_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
    normalized_shape = getattr(module, "normalized_shape", ())
    if isinstance(normalized_shape, int):
        shape = (int(normalized_shape),)
    else:
        shape = tuple(int(v) for v in normalized_shape)
    eps = getattr(module, "eps", None)
    eps_value = None if eps is None else float(eps)
    if isinstance(module, torch.nn.LayerNorm):
        return ("layernorm", shape, eps_value, bool(module.elementwise_affine))
    if _RMS_NORM_TYPE is not None and isinstance(module, _RMS_NORM_TYPE):
        return ("rmsnorm", shape, eps_value, bool(module.elementwise_affine))
    return None


def _extract_flat_grouped_mlp_info(module: torch.nn.Module) -> dict[str, Any] | None:
    for method_name in _GROUPED_SPEC_METHODS:
        method = getattr(module, method_name, None)
        if not callable(method):
            continue
        spec = method()
        if spec is None:
            return None
        if isinstance(spec, GroupedMLPSpec):
            spec = {
                "linears": tuple(spec.linears),
                "ops": tuple(spec.ops),
                "truncated_dim": spec.truncated_dim,
                "truncate_right": spec.truncate_right,
                "signature": spec.signature,
            }
        if not isinstance(spec, dict):
            raise TypeError(
                f"{type(module).__name__}.{method_name}() must return GroupedMLPSpec|dict|None, got {type(spec)!r}."
            )
        if spec.get("truncated_dim") is not None or spec.get("truncate_right") is not None:
            return None
        linears = tuple(spec.get("linears", ()))
        ops_raw = tuple(spec.get("ops", ()))
        if not linears or not ops_raw:
            return None
        ops: list[tuple[str, Any]] = []
        sig_ops: list[tuple[str, Any]] = []
        for idx, op in enumerate(ops_raw):
            if not (isinstance(op, tuple) and len(op) == 2):
                raise TypeError(
                    f"{type(module).__name__}.{method_name}() ops[{idx}] must be tuple(kind, payload)."
                )
            kind, payload = op
            if kind == "linear":
                lin_idx = int(payload)
                lin = linears[lin_idx]
                if not isinstance(lin, torch.nn.Linear):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() linears[{lin_idx}] must be torch.nn.Linear."
                    )
                ops.append(("linear", lin_idx))
                sig_ops.append(
                    (
                        "linear",
                        int(lin.in_features),
                        int(lin.out_features),
                        bool(lin.bias is not None),
                    )
                )
                continue
            if kind == "pointwise":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] pointwise payload must be torch.nn.Module."
                    )
                pointwise_sig = _pointwise_signature(payload)
                if pointwise_sig is None:
                    return None
                ops.append(("pointwise", payload))
                sig_ops.append(("pointwise", pointwise_sig))
                continue
            if kind == "norm":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] norm payload must be torch.nn.Module."
                    )
                norm_sig = _norm_signature(payload)
                if norm_sig is None:
                    return None
                ops.append(("norm", payload))
                sig_ops.append(("norm", norm_sig))
                continue
            return None
        signature = spec.get("signature", None)
        if signature is None:
            signature = tuple(sig_ops)
        if isinstance(signature, list):
            signature = tuple(signature)
        hash(signature)
        return {
            "signature": signature,
            "linears": linears,
            "ops": tuple(ops),
        }

    seq = None
    for attr_name in ("net", "mlp"):
        seq_attr = getattr(module, attr_name, None)
        if isinstance(seq_attr, torch.nn.Sequential):
            seq = seq_attr
            break
    if seq is None and isinstance(module, torch.nn.Sequential):
        seq = module
    if seq is None:
        return None

    linears: list[torch.nn.Linear] = []
    ops: list[tuple[str, Any]] = []
    sig_ops: list[tuple[str, Any]] = []
    for sub in seq:
        nested = _extract_flat_grouped_mlp_info(sub)
        if nested is not None:
            lin_offset = len(linears)
            nested_linears = tuple(nested["linears"])
            nested_ops = tuple(nested["ops"])
            linears.extend(nested_linears)
            for kind, payload in nested_ops:
                if kind == "linear":
                    lin_idx = lin_offset + int(payload)
                    lin = linears[lin_idx]
                    ops.append(("linear", lin_idx))
                    sig_ops.append(
                        (
                            "linear",
                            int(lin.in_features),
                            int(lin.out_features),
                            bool(lin.bias is not None),
                        )
                    )
                elif kind == "pointwise":
                    ops.append(("pointwise", payload))
                    sig_ops.append(("pointwise", _pointwise_signature(payload)))
                elif kind == "norm":
                    ops.append(("norm", payload))
                    sig_ops.append(("norm", _norm_signature(payload)))
                else:
                    return None
            continue
        if isinstance(sub, torch.nn.Linear):
            lin_idx = len(linears)
            linears.append(sub)
            ops.append(("linear", lin_idx))
            sig_ops.append(
                (
                    "linear",
                    int(sub.in_features),
                    int(sub.out_features),
                    bool(sub.bias is not None),
                )
            )
            continue
        pointwise_sig = _pointwise_signature(sub)
        if pointwise_sig is not None:
            ops.append(("pointwise", sub))
            sig_ops.append(("pointwise", pointwise_sig))
            continue
        norm_sig = _norm_signature(sub)
        if norm_sig is not None:
            ops.append(("norm", sub))
            sig_ops.append(("norm", norm_sig))
            continue
        return None

    if not linears:
        return None
    return {
        "signature": tuple(sig_ops),
        "linears": tuple(linears),
        "ops": tuple(ops),
    }


def normalize_relation_arities(
    relation_arities: Tensor | Sequence[int] | Iterable[int],
    *,
    device: torch.device | None = None,
) -> Tensor:
    if torch.is_tensor(relation_arities):
        out = relation_arities.to(device=device, dtype=torch.long)
    else:
        out = torch.as_tensor(tuple(int(x) for x in relation_arities), dtype=torch.long, device=device)
    if out.dim() != 1:
        raise ValueError(
            f"relation_arities must be 1D, got shape {tuple(out.shape)}."
        )
    return out


def normalize_relation_counts(
    relation_counts: Tensor,
    *,
    device: torch.device | None = None,
) -> Tensor:
    if not torch.is_tensor(relation_counts):
        raise TypeError("relation_counts must be a torch.Tensor.")
    out = relation_counts.to(device=device, dtype=torch.long)
    if out.dim() == 1:
        out = out.unsqueeze(0)
    if out.dim() != 2:
        raise ValueError(
            f"relation_counts must have shape [R] or [B, R], got {tuple(out.shape)}."
        )
    return out


def build_flat_topology(
    relation_counts: Tensor,
    relation_arities: Tensor | Sequence[int] | Iterable[int],
) -> FlatTopology:
    counts_2d = normalize_relation_counts(relation_counts)
    arities_1d = normalize_relation_arities(
        relation_arities, device=counts_2d.device
    )
    if int(counts_2d.size(1)) != int(arities_1d.numel()):
        raise ValueError(
            "relation_counts and relation_arities disagree on relation dimension: "
            f"{tuple(counts_2d.shape)} vs {tuple(arities_1d.shape)}."
        )

    counts_total = counts_2d.sum(dim=0)
    relation_slices: list[RelationSlice] = []
    slot_offsets = [0]
    cursor = 0
    for relation_index, (count_t, arity_t) in enumerate(zip(counts_total, arities_1d)):
        count = int(count_t.item())
        arity = int(arity_t.item())
        if arity < 0:
            raise ValueError(f"relation arity must be >= 0, got {arity}.")
        width = count * arity
        relation_slices.append(
            RelationSlice(
                relation_index=relation_index,
                count=count,
                arity=arity,
                slot_start=cursor,
                slot_end=cursor + width,
            )
        )
        cursor += width
        slot_offsets.append(cursor)
    return FlatTopology(
        relation_counts_total=tuple(int(x.item()) for x in counts_total),
        relation_arities=tuple(int(x.item()) for x in arities_1d),
        relation_slices=tuple(relation_slices),
        slot_offsets=tuple(int(x) for x in slot_offsets),
    )


def _topology_cache_key(topology: FlatTopology) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return topology.relation_counts_total, topology.relation_arities


class FlatRelationalLayer(torch.nn.Module):
    def __init__(
        self,
        *,
        update_modules: Sequence[torch.nn.Module],
        relation_names: Sequence[str],
        relation_arities: Tensor | Sequence[int] | Iterable[int],
        embedding_size: int,
        aggr: str | torch_geometric.nn.aggr.Aggregation | None = None,
        fused_two_layer_mish_execution: bool | None = None,
        fused_two_layer_pointwise_execution: bool | None = None,
        fused_relation_gather: bool | None = None,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.relation_names = tuple(str(name) for name in relation_names)
        self.relation_arities = normalize_relation_arities(relation_arities).cpu()
        if (
            fused_two_layer_mish_execution is not None
            and fused_two_layer_pointwise_execution is not None
            and bool(fused_two_layer_mish_execution) != bool(fused_two_layer_pointwise_execution)
        ):
            raise ValueError(
                "fused_two_layer_mish_execution and fused_two_layer_pointwise_execution "
                "must agree when both are provided."
            )
        resolved_pointwise_execution = (
            fused_two_layer_pointwise_execution
            if fused_two_layer_pointwise_execution is not None
            else fused_two_layer_mish_execution
        )
        self.fused_two_layer_pointwise_execution = resolved_pointwise_execution
        self.fused_two_layer_mish_execution = resolved_pointwise_execution
        self.fused_relation_gather = fused_relation_gather
        if len(update_modules) != len(self.relation_names):
            raise ValueError(
                "update_modules and relation_names must have the same length, got "
                f"{len(update_modules)} vs {len(self.relation_names)}."
            )
        if len(self.relation_names) != int(self.relation_arities.numel()):
            raise ValueError(
                "relation_names and relation_arities must have the same length, got "
                f"{len(self.relation_names)} vs {int(self.relation_arities.numel())}."
            )
        self.update_modules = torch.nn.ModuleList(update_modules)
        self._aggr_query = aggr or "logsumexp"
        if isinstance(self._aggr_query, str):
            if self._aggr_query.lower() == "logsumexp":
                self.aggr = LogSumExpAggregation()
            else:
                self.aggr = aggregation_resolver(self._aggr_query)
        else:
            self.aggr = self._aggr_query
        self._persistent_topology_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], FlatTopology
        ] = {}
        self._grouped_mlp_info_cache: dict[int, dict[str, Any] | None] = {}
        self._persistent_grouped_param_stacks: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._fused_relation_match_cache: dict[tuple[int, int], FusedRelationMatch | None] = {}
        self._persistent_fused_relation_layout_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            dict[str, tuple[GroupedRelationSliceBatch, ...] | tuple[int, ...]],
        ] = {}
        self._custom_fused_two_layer_mish_available: bool | None = None
        self._fused_relation_matchers = (
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice, family="two_layer_mish", pointwise_kind="mish"
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice, family="two_layer_silu", pointwise_kind="silu"
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice, family="two_layer_gelu", pointwise_kind="gelu"
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice, family="two_layer_relu", pointwise_kind="relu"
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice,
                family="prenorm_two_layer_mish",
                pointwise_kind="mish",
                norm_position="pre",
                norm_kind="layernorm",
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice,
                family="postnorm_two_layer_mish",
                pointwise_kind="mish",
                norm_position="post",
                norm_kind="layernorm",
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice,
                family="prenorm_two_layer_silu",
                pointwise_kind="silu",
                norm_position="pre",
                norm_kind="layernorm",
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice,
                family="postnorm_two_layer_silu",
                pointwise_kind="silu",
                norm_position="post",
                norm_kind="layernorm",
            ),
            lambda relation_slice: self._match_two_layer_pointwise_family(
                relation_slice,
                family="prenorm_two_layer_silu_rmsnorm",
                pointwise_kind="silu",
                norm_position="pre",
                norm_kind="rmsnorm",
            ),
            lambda relation_slice: self._match_three_layer_pointwise_family(
                relation_slice, family="three_layer_silu", pointwise_kind="silu"
            ),
            self._match_staged_program,
        )
        self._fused_relation_collectors = {
            "two_layer_mish": self._collect_fused_two_layer_pointwise_messages,
            "two_layer_silu": self._collect_fused_two_layer_pointwise_messages,
            "two_layer_gelu": self._collect_fused_two_layer_pointwise_messages,
            "postnorm_two_layer_mish": self._collect_fused_postnorm_two_layer_pointwise_layernorm_messages,
            "postnorm_two_layer_silu": self._collect_fused_postnorm_two_layer_pointwise_layernorm_messages,
            "prenorm_two_layer_silu_rmsnorm": self._collect_fused_prenorm_two_layer_pointwise_rmsnorm_messages,
            "program": self._collect_fused_program_messages,
        }

    def _grouped_mlp_info(self, module: torch.nn.Module) -> dict[str, Any] | None:
        key = id(module)
        cached = self._grouped_mlp_info_cache.get(key)
        if cached is not None or key in self._grouped_mlp_info_cache:
            return cached
        info = _extract_flat_grouped_mlp_info(module)
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

    def _use_fused_two_layer_pointwise_execution(self, x: Tensor) -> bool:
        if self.fused_two_layer_pointwise_execution is not None:
            return bool(self.fused_two_layer_pointwise_execution)
        return x.device.type == "cuda"

    def _use_fused_two_layer_mish_execution(self, x: Tensor) -> bool:
        return self._use_fused_two_layer_pointwise_execution(x)

    def _use_fused_relation_gather(self, x: Tensor) -> bool:
        if self.fused_relation_gather is not None:
            return bool(self.fused_relation_gather)
        return x.device.type == "cuda"

    def _can_use_custom_fused_two_layer_mish(self, x: Tensor) -> bool:
        if x.device.type != "cuda":
            return False
        cached = self._custom_fused_two_layer_mish_available
        if cached is not None:
            return cached
        try:
            cached = bool(
                relm_mp_ops.available()
                and hasattr(relm_mp_ops, "fused_two_layer_mish_from_indices")
            )
        except Exception:
            cached = False
        self._custom_fused_two_layer_mish_available = cached
        return cached

    def _two_linear_pointwise_info(
        self,
        relation_slice: RelationSlice,
    ) -> dict[str, Any] | None:
        module = self.update_modules[relation_slice.relation_index]
        info = self._grouped_mlp_info(module)
        if info is None:
            return None
        expected_dim = int(relation_slice.arity * self.embedding_size)
        linears = tuple(info["linears"])
        ops = tuple(info["ops"])
        if len(linears) != 2 or len(ops) != 3:
            return None
        if ops[0][0] != "linear" or int(ops[0][1]) != 0:
            return None
        if ops[1][0] != "pointwise":
            return None
        if ops[2][0] != "linear" or int(ops[2][1]) != 1:
            return None
        lin0, lin1 = linears
        pointwise_module = ops[1][1]
        if not isinstance(pointwise_module, torch.nn.Module):
            return None
        pointwise_signature = _pointwise_signature(pointwise_module)
        if pointwise_signature is None:
            return None
        if int(lin0.in_features) != expected_dim or int(lin1.out_features) != expected_dim:
            return None
        if int(lin1.in_features) != int(lin0.out_features):
            return None
        return {
            "expected_dim": expected_dim,
            "hidden_dim": int(lin0.out_features),
            "linear0": lin0,
            "linear1": lin1,
            "bias_flags": (bool(lin0.bias is not None), bool(lin1.bias is not None)),
            "pointwise_module": pointwise_module,
            "pointwise_signature": pointwise_signature,
        }

    def _match_two_layer_mish_spec(
        self,
        relation_slice: RelationSlice,
    ) -> FusedRelationMatch | None:
        return self._match_two_layer_pointwise_family(
            relation_slice, family="two_layer_mish", pointwise_kind="mish"
        )

    def _normalized_grouped_ops_for_module(
        self,
        module: torch.nn.Module,
        *,
        expected_dim: int,
    ) -> dict[str, Any] | None:
        info = self._grouped_mlp_info(module)
        if info is None:
            return None
        linears = tuple(info["linears"])
        ops = tuple(info["ops"])
        normalized_ops: list[tuple[str, Any, Any]] = []
        for kind, payload in ops:
            if kind == "linear":
                lin_idx = int(payload)
                lin = linears[lin_idx]
                normalized_ops.append(("linear", lin, None))
                continue
            if kind == "pointwise":
                if not isinstance(payload, torch.nn.Module):
                    return None
                pointwise_sig = _pointwise_signature(payload)
                if pointwise_sig is None:
                    return None
                normalized_ops.append(("pointwise", payload, pointwise_sig))
                continue
            if kind == "norm":
                if not isinstance(payload, torch.nn.Module):
                    return None
                norm_sig = _norm_signature(payload)
                if norm_sig is None:
                    return None
                normalized_ops.append(("norm", payload, norm_sig))
                continue
            return None
        return {
            "expected_dim": expected_dim,
            "linears": linears,
            "normalized_ops": tuple(normalized_ops),
        }

    def _normalized_grouped_ops(
        self,
        relation_slice: RelationSlice,
    ) -> dict[str, Any] | None:
        module = self.update_modules[relation_slice.relation_index]
        expected_dim = int(relation_slice.arity * self.embedding_size)
        return self._normalized_grouped_ops_for_module(
            module,
            expected_dim=expected_dim,
        )

    def _norm_matches_expected_dim(
        self,
        norm_signature: tuple[Any, ...],
        expected_dim: int,
        *,
        norm_kind: str | None = None,
    ) -> bool:
        kind = str(norm_signature[0])
        if norm_kind is not None and kind != norm_kind:
            return False
        shape = tuple(int(v) for v in norm_signature[1])
        return shape == (int(expected_dim),)

    def _match_two_layer_pointwise_ops(
        self,
        *,
        relation_slice: RelationSlice,
        ops: tuple[tuple[str, Any, Any], ...],
        expected_dim: int,
        family: str,
        pointwise_kind: str,
        norm_position: str | None = None,
        norm_kind: str | None = None,
    ) -> FusedRelationMatch | None:
        if norm_position is None:
            if len(ops) != 3:
                return None
            if ops[0][0] != "linear" or ops[1][0] != "pointwise" or ops[2][0] != "linear":
                return None
            norm_module = None
            norm_signature = None
            linear0 = ops[0][1]
            pointwise_module = ops[1][1]
            pointwise_signature = ops[1][2]
            linear1 = ops[2][1]
        elif norm_position == "pre":
            if len(ops) != 4:
                return None
            if (
                ops[0][0] != "norm"
                or ops[1][0] != "linear"
                or ops[2][0] != "pointwise"
                or ops[3][0] != "linear"
            ):
                return None
            norm_module = ops[0][1]
            norm_signature = ops[0][2]
            linear0 = ops[1][1]
            pointwise_module = ops[2][1]
            pointwise_signature = ops[2][2]
            linear1 = ops[3][1]
        elif norm_position == "post":
            if len(ops) != 4:
                return None
            if (
                ops[0][0] != "linear"
                or ops[1][0] != "pointwise"
                or ops[2][0] != "linear"
                or ops[3][0] != "norm"
            ):
                return None
            linear0 = ops[0][1]
            pointwise_module = ops[1][1]
            pointwise_signature = ops[1][2]
            linear1 = ops[2][1]
            norm_module = ops[3][1]
            norm_signature = ops[3][2]
        else:
            raise ValueError(f"Unsupported norm_position: {norm_position!r}.")

        if pointwise_signature[0] != pointwise_kind:
            return None
        if int(linear0.in_features) != expected_dim or int(linear1.out_features) != expected_dim:
            return None
        if int(linear1.in_features) != int(linear0.out_features):
            return None
        if norm_signature is not None and not self._norm_matches_expected_dim(
            norm_signature, expected_dim, norm_kind=norm_kind
        ):
            return None
        spec = FusedRelationSpec(
            family=family,
            signature=(
                int(relation_slice.arity),
                int(linear0.out_features),
                bool(linear0.bias is not None),
                bool(linear1.bias is not None),
                tuple(pointwise_signature),
                tuple(norm_signature) if norm_signature is not None else None,
            ),
            arity=int(relation_slice.arity),
            input_dim=expected_dim,
            output_dim=expected_dim,
            hidden_dims=(int(linear0.out_features),),
            bias_flags=(bool(linear0.bias is not None), bool(linear1.bias is not None)),
            pointwise_signature=tuple(pointwise_signature),
            norm_kind=(str(norm_signature[0]) if norm_signature is not None else None),
            norm_position=norm_position,
        )
        return FusedRelationMatch(
            spec=spec,
            linears=(linear0, linear1),
            pointwise_modules=(pointwise_module,),
            norm_modules=((norm_module,) if norm_module is not None else ()),
        )

    def _match_two_layer_pointwise_family(
        self,
        relation_slice: RelationSlice,
        *,
        family: str,
        pointwise_kind: str,
        norm_position: str | None = None,
        norm_kind: str | None = None,
    ) -> FusedRelationMatch | None:
        info = self._normalized_grouped_ops(relation_slice)
        if info is None:
            return None
        return self._match_two_layer_pointwise_ops(
            relation_slice=relation_slice,
            ops=tuple(info["normalized_ops"]),
            expected_dim=int(info["expected_dim"]),
            family=family,
            pointwise_kind=pointwise_kind,
            norm_position=norm_position,
            norm_kind=norm_kind,
        )

    def _match_three_layer_pointwise_ops(
        self,
        *,
        relation_slice: RelationSlice,
        ops: tuple[tuple[str, Any, Any], ...],
        expected_dim: int,
        family: str,
        pointwise_kind: str,
    ) -> FusedRelationMatch | None:
        if len(ops) != 5:
            return None
        expected_kinds = ("linear", "pointwise", "linear", "pointwise", "linear")
        if tuple(item[0] for item in ops) != expected_kinds:
            return None
        linear0 = ops[0][1]
        pointwise0 = ops[1]
        linear1 = ops[2][1]
        pointwise1 = ops[3]
        linear2 = ops[4][1]
        if pointwise0[2][0] != pointwise_kind or pointwise1[2][0] != pointwise_kind:
            return None
        if int(linear0.in_features) != expected_dim:
            return None
        if int(linear2.out_features) != expected_dim:
            return None
        if int(linear1.in_features) != int(linear0.out_features):
            return None
        if int(linear2.in_features) != int(linear1.out_features):
            return None
        spec = FusedRelationSpec(
            family=family,
            signature=(
                int(relation_slice.arity),
                int(linear0.out_features),
                int(linear1.out_features),
                bool(linear0.bias is not None),
                bool(linear1.bias is not None),
                bool(linear2.bias is not None),
                tuple(pointwise0[2]),
            ),
            arity=int(relation_slice.arity),
            input_dim=expected_dim,
            output_dim=expected_dim,
            hidden_dims=(int(linear0.out_features), int(linear1.out_features)),
            bias_flags=(
                bool(linear0.bias is not None),
                bool(linear1.bias is not None),
                bool(linear2.bias is not None),
            ),
            pointwise_signature=tuple(pointwise0[2]),
        )
        return FusedRelationMatch(
            spec=spec,
            linears=(linear0, linear1, linear2),
            pointwise_modules=(pointwise0[1], pointwise1[1]),
        )

    def _match_three_layer_pointwise_family(
        self,
        relation_slice: RelationSlice,
        *,
        family: str,
        pointwise_kind: str,
    ) -> FusedRelationMatch | None:
        info = self._normalized_grouped_ops(relation_slice)
        if info is None:
            return None
        return self._match_three_layer_pointwise_ops(
            relation_slice=relation_slice,
            ops=tuple(info["normalized_ops"]),
            expected_dim=int(info["expected_dim"]),
            family=family,
            pointwise_kind=pointwise_kind,
        )

    def _match_staged_program(
        self,
        relation_slice: RelationSlice,
    ) -> FusedRelationMatch | None:
        module = self.update_modules[relation_slice.relation_index]
        expected_dim = int(relation_slice.arity * self.embedding_size)
        info = self._normalized_grouped_ops_for_module(module, expected_dim=expected_dim)
        if info is None:
            return None
        ops = tuple(info["normalized_ops"])
        if len(ops) <= 5:
            return None

        cursor = 0
        stages: list[FusedRelationMatch] = []
        while cursor < len(ops):
            stage_match = None
            consumed = 0
            for candidate_len, matcher in (
                (
                    4,
                    lambda stage_ops: self._match_two_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="prenorm_two_layer_silu_rmsnorm",
                        pointwise_kind="silu",
                        norm_position="pre",
                        norm_kind="rmsnorm",
                    ),
                ),
                (
                    4,
                    lambda stage_ops: self._match_two_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="postnorm_two_layer_silu",
                        pointwise_kind="silu",
                        norm_position="post",
                        norm_kind="layernorm",
                    ),
                ),
                (
                    3,
                    lambda stage_ops: self._match_two_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="two_layer_mish",
                        pointwise_kind="mish",
                    ),
                ),
                (
                    3,
                    lambda stage_ops: self._match_two_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="two_layer_silu",
                        pointwise_kind="silu",
                    ),
                ),
                (
                    3,
                    lambda stage_ops: self._match_two_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="two_layer_gelu",
                        pointwise_kind="gelu",
                    ),
                ),
                (
                    3,
                    lambda stage_ops: self._match_two_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="two_layer_relu",
                        pointwise_kind="relu",
                    ),
                ),
                (
                    5,
                    lambda stage_ops: self._match_three_layer_pointwise_ops(
                        relation_slice=relation_slice,
                        ops=stage_ops,
                        expected_dim=expected_dim,
                        family="three_layer_silu",
                        pointwise_kind="silu",
                    ),
                ),
            ):
                if cursor + candidate_len > len(ops):
                    continue
                candidate_ops = ops[cursor : cursor + candidate_len]
                candidate = matcher(candidate_ops)
                if candidate is not None:
                    stage_match = candidate
                    consumed = candidate_len
                    break
            if stage_match is None:
                return None
            stages.append(stage_match)
            cursor += consumed

        if len(stages) <= 1:
            return None
        program_family = self._match_program_family(
            relation_slice,
            stages=tuple(stages),
            expected_dim=expected_dim,
        )
        return FusedRelationMatch(
            spec=FusedRelationSpec(
                family="program",
                signature=tuple((stage.spec.family, stage.spec.signature) for stage in stages),
                arity=int(relation_slice.arity),
                input_dim=expected_dim,
                output_dim=expected_dim,
                hidden_dims=tuple(),
                bias_flags=tuple(),
            ),
            linears=tuple(),
            program_matches=tuple(stages),
            program_family=program_family,
        )

    def _match_program_family(
        self,
        relation_slice: RelationSlice,
        *,
        stages: tuple[FusedRelationMatch, ...],
        expected_dim: int,
    ) -> ProgramFamilySpec | None:
        if len(stages) != 2:
            return None
        stage0, stage1 = stages
        if stage0.spec.output_dim != expected_dim or stage1.spec.output_dim != expected_dim:
            return None
        if stage0.spec.family == "two_layer_silu" and stage1.spec.family == "two_layer_silu":
            return ProgramFamilySpec(
                family="program_two_layer_silu_then_two_layer_silu",
                signature=(
                    stage0.spec.family,
                    stage0.spec.signature,
                    stage1.spec.family,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        if stage0.spec.family == "two_layer_silu" and stage1.spec.family == "postnorm_two_layer_silu":
            return ProgramFamilySpec(
                family="program_two_layer_silu_then_postnorm_two_layer_silu",
                signature=(
                    stage0.spec.family,
                    stage0.spec.signature,
                    stage1.spec.family,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        if (
            stage0.spec.family == "prenorm_two_layer_silu_rmsnorm"
            and stage1.spec.family == "two_layer_silu"
        ):
            return ProgramFamilySpec(
                family="program_prenorm_two_layer_silu_rmsnorm_then_two_layer_silu",
                signature=(
                    stage0.spec.family,
                    stage0.spec.signature,
                    stage1.spec.family,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        return None

    def _match_fused_relation(
        self,
        relation_slice: RelationSlice,
    ) -> FusedRelationMatch | None:
        module = self.update_modules[relation_slice.relation_index]
        cache_key = (id(module), int(relation_slice.arity))
        cached = self._fused_relation_match_cache.get(cache_key)
        if cached is not None or cache_key in self._fused_relation_match_cache:
            return cached
        match = None
        for matcher in self._fused_relation_matchers:
            match = matcher(relation_slice)
            if match is not None:
                break
        self._fused_relation_match_cache[cache_key] = match
        return match

    def _get_fused_relation_layout(
        self,
        topology: FlatTopology,
    ) -> dict[str, tuple[GroupedRelationSliceBatch, ...] | tuple[int, ...]]:
        cache_key = _topology_cache_key(topology)
        cached = self._persistent_fused_relation_layout_cache.get(cache_key)
        if cached is not None:
            return cached

        grouped_exec: dict[tuple[str, int, Hashable], list[int]] = defaultdict(list)
        fallback_indices: list[int] = []
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0 or relation_slice.arity <= 0:
                continue
            match = self._match_fused_relation(relation_slice)
            if match is None:
                fallback_indices.append(relation_slice.relation_index)
                continue
            if (
                match.spec.family == "program"
                and self.fused_two_layer_pointwise_execution is None
            ):
                fallback_indices.append(relation_slice.relation_index)
                continue
            grouped_exec[
                (match.spec.family, relation_slice.arity, match.spec.signature)
            ].append(
                relation_slice.relation_index
            )

        groups: list[GroupedRelationSliceBatch] = []
        for (family, arity, signature), relation_indices in grouped_exec.items():
            group_slices = tuple(topology.relation_slices[idx] for idx in relation_indices)
            row_sizes = tuple(int(relation_slice.count) for relation_slice in group_slices)
            groups.append(
                GroupedRelationSliceBatch(
                    family=str(family),
                    signature=signature,
                    arity=int(arity),
                    relation_indices=tuple(int(idx) for idx in relation_indices),
                    max_rows=max(row_sizes) if row_sizes else 0,
                    row_sizes=row_sizes,
                )
            )

        layout = {
            "groups": tuple(groups),
            "fallback_indices": tuple(sorted(set(int(idx) for idx in fallback_indices))),
        }
        self._persistent_fused_relation_layout_cache[cache_key] = layout
        return layout

    def _get_fused_two_layer_mish_layout(
        self,
        topology: FlatTopology,
    ) -> dict[str, tuple[GroupedRelationSliceBatch, ...] | tuple[int, ...]]:
        layout = self._get_fused_relation_layout(topology)
        return {
            "groups": tuple(group for group in layout["groups"] if group.family == "two_layer_mish"),
            "fallback_indices": layout["fallback_indices"],
        }

    def _collect_direct_relation_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        relation_slice: RelationSlice,
        *,
        arg_emb_all: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        if relation_slice.count <= 0 or relation_slice.arity <= 0:
            return None
        flat_idx = relation_args[relation_slice.slot_start : relation_slice.slot_end]
        module = self.update_modules[relation_slice.relation_index]
        if arg_emb_all is not None:
            arg_emb = arg_emb_all[relation_slice.slot_start : relation_slice.slot_end]
        else:
            arg_emb = x.index_select(0, flat_idx)
        rel_in = arg_emb.view(
            relation_slice.count,
            relation_slice.arity * self.embedding_size,
        )
        rel_out = module(rel_in).view(
            relation_slice.count * relation_slice.arity,
            self.embedding_size,
        )
        return arg_emb + rel_out, flat_idx

    def _collect_fused_two_layer_pointwise_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: GroupedRelationSliceBatch,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, FusedRelationMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_fused_relation(relation_slice)
            if match is None or match.spec.family not in {
                "two_layer_mish",
                "two_layer_silu",
                "two_layer_gelu",
            }:
                return None
            batch_items.append((relation_slice, match))

        pointwise_signature = batch_items[0][1].spec.pointwise_signature
        pointwise_code = relm_mp_ops.pointwise_code_from_signature(pointwise_signature)
        if pointwise_code is None:
            return None
        if any(item[1].spec.pointwise_signature != pointwise_signature for item in batch_items[1:]):
            return None

        group_key = ("fused_relation", grouped_batch.family, grouped_batch.arity, grouped_batch.signature)
        w1_stack = self._get_grouped_param_stack(
            cache_key=("w1", group_key),
            tensors=[item[1].linears[0].weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
        )
        w2_stack = self._get_grouped_param_stack(
            cache_key=("w2", group_key),
            tensors=[item[1].linears[1].weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
        )

        lin0_has_bias = batch_items[0][1].linears[0].bias is not None
        lin1_has_bias = batch_items[0][1].linears[1].bias is not None
        if lin0_has_bias:
            b1_stack = self._get_grouped_param_stack(
                cache_key=("b1", group_key),
                tensors=[item[1].linears[0].bias for item in batch_items if item[1].linears[0].bias is not None],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b1_stack = w1_stack.new_empty((0,))
        if lin1_has_bias:
            b2_stack = self._get_grouped_param_stack(
                cache_key=("b2", group_key),
                tensors=[item[1].linears[1].bias for item in batch_items if item[1].linears[1].bias is not None],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b2_stack = w2_stack.new_empty((0,))

        return relm_mp_ops.fused_two_layer_pointwise_from_indices(
            x,
            relation_args,
            [int(item[0].slot_start) for item in batch_items],
            [int(item[0].count) for item in batch_items],
            int(grouped_batch.arity),
            w1_stack,
            b1_stack,
            w2_stack,
            b2_stack,
            int(pointwise_code),
        )

    def _collect_fused_postnorm_two_layer_pointwise_layernorm_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: GroupedRelationSliceBatch,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, FusedRelationMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_fused_relation(relation_slice)
            if match is None or match.spec.family not in {
                "postnorm_two_layer_mish",
                "postnorm_two_layer_silu",
            }:
                return None
            if match.spec.norm_kind != "layernorm" or match.spec.norm_position != "post":
                return None
            if len(match.norm_modules) != 1:
                return None
            batch_items.append((relation_slice, match))

        pointwise_signature = batch_items[0][1].spec.pointwise_signature
        pointwise_code = relm_mp_ops.pointwise_code_from_signature(pointwise_signature)
        if pointwise_code is None:
            return None
        if any(item[1].spec.pointwise_signature != pointwise_signature for item in batch_items[1:]):
            return None

        norm_signature = batch_items[0][1].spec.signature[-1]
        if norm_signature is None or str(norm_signature[0]) != "layernorm":
            return None
        eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
        affine = bool(norm_signature[3])
        if any(item[1].spec.signature[-1] != norm_signature for item in batch_items[1:]):
            return None

        group_key = (
            "fused_postnorm_relation",
            grouped_batch.family,
            grouped_batch.arity,
            grouped_batch.signature,
        )
        w1_stack = self._get_grouped_param_stack(
            cache_key=("w1", group_key),
            tensors=[item[1].linears[0].weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
        )
        w2_stack = self._get_grouped_param_stack(
            cache_key=("w2", group_key),
            tensors=[item[1].linears[1].weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
        )
        lin0_has_bias = batch_items[0][1].linears[0].bias is not None
        lin1_has_bias = batch_items[0][1].linears[1].bias is not None
        if lin0_has_bias:
            b1_stack = self._get_grouped_param_stack(
                cache_key=("b1", group_key),
                tensors=[item[1].linears[0].bias for item in batch_items if item[1].linears[0].bias is not None],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b1_stack = w1_stack.new_empty((0,))
        if lin1_has_bias:
            b2_stack = self._get_grouped_param_stack(
                cache_key=("b2", group_key),
                tensors=[item[1].linears[1].bias for item in batch_items if item[1].linears[1].bias is not None],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b2_stack = w2_stack.new_empty((0,))

        if affine:
            ln_weight_stack = self._get_grouped_param_stack(
                cache_key=("ln_w", group_key),
                tensors=[cast(torch.nn.LayerNorm, item[1].norm_modules[0]).weight for item in batch_items],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            ln_bias_stack = self._get_grouped_param_stack(
                cache_key=("ln_b", group_key),
                tensors=[cast(torch.nn.LayerNorm, item[1].norm_modules[0]).bias for item in batch_items],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            ln_weight_stack = w2_stack.new_empty((0,))
            ln_bias_stack = w2_stack.new_empty((0,))

        return relm_mp_ops.fused_postnorm_two_layer_pointwise_layernorm_from_indices(
            x,
            relation_args,
            [int(item[0].slot_start) for item in batch_items],
            [int(item[0].count) for item in batch_items],
            int(grouped_batch.arity),
            w1_stack,
            b1_stack,
            w2_stack,
            b2_stack,
            ln_weight_stack,
            ln_bias_stack,
            float(eps),
            int(pointwise_code),
        )

    def _collect_fused_prenorm_two_layer_pointwise_rmsnorm_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: GroupedRelationSliceBatch,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, FusedRelationMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_fused_relation(relation_slice)
            if match is None or match.spec.family != "prenorm_two_layer_silu_rmsnorm":
                return None
            if match.spec.norm_kind != "rmsnorm" or match.spec.norm_position != "pre":
                return None
            if len(match.norm_modules) != 1:
                return None
            batch_items.append((relation_slice, match))

        pointwise_signature = batch_items[0][1].spec.pointwise_signature
        pointwise_code = relm_mp_ops.pointwise_code_from_signature(pointwise_signature)
        if pointwise_code is None:
            return None
        if any(item[1].spec.pointwise_signature != pointwise_signature for item in batch_items[1:]):
            return None

        norm_signature = batch_items[0][1].spec.signature[-1]
        if norm_signature is None or str(norm_signature[0]) != "rmsnorm":
            return None
        eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
        affine = bool(norm_signature[3])
        if any(item[1].spec.signature[-1] != norm_signature for item in batch_items[1:]):
            return None

        group_key = (
            "fused_prenorm_rmsnorm_relation",
            grouped_batch.family,
            grouped_batch.arity,
            grouped_batch.signature,
        )
        w1_stack = self._get_grouped_param_stack(
            cache_key=("w1", group_key),
            tensors=[item[1].linears[0].weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
        )
        w2_stack = self._get_grouped_param_stack(
            cache_key=("w2", group_key),
            tensors=[item[1].linears[1].weight for item in batch_items],
            forward_cache=grouped_param_stacks,
            allow_persistent=allow_persistent_stacks,
        )
        lin0_has_bias = batch_items[0][1].linears[0].bias is not None
        lin1_has_bias = batch_items[0][1].linears[1].bias is not None
        if lin0_has_bias:
            b1_stack = self._get_grouped_param_stack(
                cache_key=("b1", group_key),
                tensors=[item[1].linears[0].bias for item in batch_items if item[1].linears[0].bias is not None],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b1_stack = w1_stack.new_empty((0,))
        if lin1_has_bias:
            b2_stack = self._get_grouped_param_stack(
                cache_key=("b2", group_key),
                tensors=[item[1].linears[1].bias for item in batch_items if item[1].linears[1].bias is not None],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b2_stack = w2_stack.new_empty((0,))

        if affine:
            rms_weight_stack = self._get_grouped_param_stack(
                cache_key=("rms_w", group_key),
                tensors=[cast(Any, item[1].norm_modules[0]).weight for item in batch_items],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            rms_weight_stack = w2_stack.new_empty((0,))

        return relm_mp_ops.fused_prenorm_two_layer_pointwise_rmsnorm_from_indices(
            x,
            relation_args,
            [int(item[0].slot_start) for item in batch_items],
            [int(item[0].count) for item in batch_items],
            int(grouped_batch.arity),
            rms_weight_stack,
            float(eps),
            w1_stack,
            b1_stack,
            w2_stack,
            b2_stack,
            int(pointwise_code),
        )

    def _apply_dense_stage_match(
        self,
        rel_in: Tensor,
        match: FusedRelationMatch,
    ) -> Tensor:
        family = match.spec.family
        if family in {"two_layer_mish", "two_layer_silu", "two_layer_gelu", "two_layer_relu"}:
            return match.linears[1](match.pointwise_modules[0](match.linears[0](rel_in)))
        if family in {"postnorm_two_layer_mish", "postnorm_two_layer_silu"}:
            return match.norm_modules[0](
                match.linears[1](match.pointwise_modules[0](match.linears[0](rel_in)))
            )
        if family == "prenorm_two_layer_silu_rmsnorm":
            return match.linears[1](
                match.pointwise_modules[0](match.linears[0](match.norm_modules[0](rel_in)))
            )
        if family == "three_layer_silu":
            return match.linears[2](
                match.pointwise_modules[1](
                    match.linears[1](
                        match.pointwise_modules[0](match.linears[0](rel_in))
                    )
                )
            )
        raise ValueError(f"Unsupported dense staged family: {family!r}.")

    def _execute_dense_program_stage(
        self,
        stage_input: Tensor,
        *,
        row_sizes: Sequence[int],
        arity: int,
        stage_matches: Sequence[FusedRelationMatch],
    ) -> Tensor:
        chunks: list[Tensor] = []
        slot_cursor = 0
        for row_size, stage_match in zip(row_sizes, stage_matches):
            row_count = int(row_size)
            slot_count = row_count * int(arity)
            stage_slots = stage_input[slot_cursor : slot_cursor + slot_count]
            rel_in = stage_slots.view(row_count, int(arity) * self.embedding_size)
            rel_out = self._apply_dense_stage_match(rel_in, stage_match).view(
                slot_count,
                self.embedding_size,
            )
            chunks.append(rel_out)
            slot_cursor += slot_count
        if not chunks:
            return stage_input.new_empty((0, self.embedding_size))
        return torch.cat(chunks, dim=0)

    def _gather_stage_input_slots(
        self,
        stage_input: Tensor,
        stage_relation_args: Tensor,
        *,
        slot_offsets: Sequence[int],
        row_sizes: Sequence[int],
        arity: int,
    ) -> Tensor:
        chunks: list[Tensor] = []
        for slot_offset, row_size in zip(slot_offsets, row_sizes):
            slot_count = int(row_size) * int(arity)
            flat_idx = stage_relation_args[int(slot_offset) : int(slot_offset) + slot_count]
            chunks.append(stage_input.index_select(0, flat_idx))
        if not chunks:
            return stage_input.new_empty((0, self.embedding_size))
        return torch.cat(chunks, dim=0)

    def _execute_fused_program_stage(
        self,
        stage_input: Tensor,
        stage_relation_args: Tensor,
        *,
        slot_offsets: Sequence[int],
        row_sizes: Sequence[int],
        arity: int,
        stage_matches: Sequence[FusedRelationMatch],
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
        cache_key_prefix: tuple[Any, ...],
    ) -> tuple[Tensor, Tensor] | None:
        if not stage_matches:
            return None
        stage_family = stage_matches[0].spec.family
        if any(match.spec.family != stage_family for match in stage_matches[1:]):
            return None

        if stage_family in {"two_layer_mish", "two_layer_silu", "two_layer_gelu"}:
            pointwise_signature = stage_matches[0].spec.pointwise_signature
            pointwise_code = relm_mp_ops.pointwise_code_from_signature(pointwise_signature)
            if pointwise_code is None:
                return None
            if any(match.spec.pointwise_signature != pointwise_signature for match in stage_matches[1:]):
                return None
            w1_stack = self._get_grouped_param_stack(
                cache_key=("w1", *cache_key_prefix),
                tensors=[match.linears[0].weight for match in stage_matches],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            w2_stack = self._get_grouped_param_stack(
                cache_key=("w2", *cache_key_prefix),
                tensors=[match.linears[1].weight for match in stage_matches],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            if stage_matches[0].linears[0].bias is not None:
                b1_stack = self._get_grouped_param_stack(
                    cache_key=("b1", *cache_key_prefix),
                    tensors=[match.linears[0].bias for match in stage_matches if match.linears[0].bias is not None],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                b1_stack = w1_stack.new_empty((0,))
            if stage_matches[0].linears[1].bias is not None:
                b2_stack = self._get_grouped_param_stack(
                    cache_key=("b2", *cache_key_prefix),
                    tensors=[match.linears[1].bias for match in stage_matches if match.linears[1].bias is not None],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                b2_stack = w2_stack.new_empty((0,))
            residual, node_idx = relm_mp_ops.fused_two_layer_pointwise_from_indices(
                stage_input,
                stage_relation_args,
                [int(v) for v in slot_offsets],
                [int(v) for v in row_sizes],
                int(arity),
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                int(pointwise_code),
            )
            stage_slots = self._gather_stage_input_slots(
                stage_input,
                stage_relation_args,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=int(arity),
            )
            return residual - stage_slots, node_idx

        if stage_family in {"postnorm_two_layer_mish", "postnorm_two_layer_silu"}:
            pointwise_signature = stage_matches[0].spec.pointwise_signature
            pointwise_code = relm_mp_ops.pointwise_code_from_signature(pointwise_signature)
            if pointwise_code is None:
                return None
            if any(match.spec.pointwise_signature != pointwise_signature for match in stage_matches[1:]):
                return None
            norm_signature = stage_matches[0].spec.signature[-1]
            if norm_signature is None or str(norm_signature[0]) != "layernorm":
                return None
            if any(match.spec.signature[-1] != norm_signature for match in stage_matches[1:]):
                return None
            eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
            affine = bool(norm_signature[3])
            w1_stack = self._get_grouped_param_stack(
                cache_key=("w1", *cache_key_prefix),
                tensors=[match.linears[0].weight for match in stage_matches],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            w2_stack = self._get_grouped_param_stack(
                cache_key=("w2", *cache_key_prefix),
                tensors=[match.linears[1].weight for match in stage_matches],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            if stage_matches[0].linears[0].bias is not None:
                b1_stack = self._get_grouped_param_stack(
                    cache_key=("b1", *cache_key_prefix),
                    tensors=[match.linears[0].bias for match in stage_matches if match.linears[0].bias is not None],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                b1_stack = w1_stack.new_empty((0,))
            if stage_matches[0].linears[1].bias is not None:
                b2_stack = self._get_grouped_param_stack(
                    cache_key=("b2", *cache_key_prefix),
                    tensors=[match.linears[1].bias for match in stage_matches if match.linears[1].bias is not None],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                b2_stack = w2_stack.new_empty((0,))
            if affine:
                ln_weight_stack = self._get_grouped_param_stack(
                    cache_key=("ln_w", *cache_key_prefix),
                    tensors=[cast(torch.nn.LayerNorm, match.norm_modules[0]).weight for match in stage_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                ln_bias_stack = self._get_grouped_param_stack(
                    cache_key=("ln_b", *cache_key_prefix),
                    tensors=[cast(torch.nn.LayerNorm, match.norm_modules[0]).bias for match in stage_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                ln_weight_stack = w2_stack.new_empty((0,))
                ln_bias_stack = w2_stack.new_empty((0,))
            residual, node_idx = relm_mp_ops.fused_postnorm_two_layer_pointwise_layernorm_from_indices(
                stage_input,
                stage_relation_args,
                [int(v) for v in slot_offsets],
                [int(v) for v in row_sizes],
                int(arity),
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                ln_weight_stack,
                ln_bias_stack,
                float(eps),
                int(pointwise_code),
            )
            stage_slots = self._gather_stage_input_slots(
                stage_input,
                stage_relation_args,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=int(arity),
            )
            return residual - stage_slots, node_idx

        if stage_family == "prenorm_two_layer_silu_rmsnorm":
            pointwise_signature = stage_matches[0].spec.pointwise_signature
            pointwise_code = relm_mp_ops.pointwise_code_from_signature(pointwise_signature)
            if pointwise_code is None:
                return None
            if any(match.spec.pointwise_signature != pointwise_signature for match in stage_matches[1:]):
                return None
            norm_signature = stage_matches[0].spec.signature[-1]
            if norm_signature is None or str(norm_signature[0]) != "rmsnorm":
                return None
            if any(match.spec.signature[-1] != norm_signature for match in stage_matches[1:]):
                return None
            eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
            affine = bool(norm_signature[3])
            w1_stack = self._get_grouped_param_stack(
                cache_key=("w1", *cache_key_prefix),
                tensors=[match.linears[0].weight for match in stage_matches],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            w2_stack = self._get_grouped_param_stack(
                cache_key=("w2", *cache_key_prefix),
                tensors=[match.linears[1].weight for match in stage_matches],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            if stage_matches[0].linears[0].bias is not None:
                b1_stack = self._get_grouped_param_stack(
                    cache_key=("b1", *cache_key_prefix),
                    tensors=[match.linears[0].bias for match in stage_matches if match.linears[0].bias is not None],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                b1_stack = w1_stack.new_empty((0,))
            if stage_matches[0].linears[1].bias is not None:
                b2_stack = self._get_grouped_param_stack(
                    cache_key=("b2", *cache_key_prefix),
                    tensors=[match.linears[1].bias for match in stage_matches if match.linears[1].bias is not None],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                b2_stack = w2_stack.new_empty((0,))
            if affine:
                rms_weight_stack = self._get_grouped_param_stack(
                    cache_key=("rms_w", *cache_key_prefix),
                    tensors=[cast(Any, match.norm_modules[0]).weight for match in stage_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            else:
                rms_weight_stack = w2_stack.new_empty((0,))
            residual, node_idx = relm_mp_ops.fused_prenorm_two_layer_pointwise_rmsnorm_from_indices(
                stage_input,
                stage_relation_args,
                [int(v) for v in slot_offsets],
                [int(v) for v in row_sizes],
                int(arity),
                rms_weight_stack,
                float(eps),
                w1_stack,
                b1_stack,
                w2_stack,
                b2_stack,
                int(pointwise_code),
            )
            stage_slots = self._gather_stage_input_slots(
                stage_input,
                stage_relation_args,
                slot_offsets=slot_offsets,
                row_sizes=row_sizes,
                arity=int(arity),
            )
            return residual - stage_slots, node_idx
        return None

    def _collect_fused_program_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: GroupedRelationSliceBatch,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, FusedRelationMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_fused_relation(relation_slice)
            if match is None or match.spec.family != "program" or not match.program_matches:
                return None
            batch_items.append((relation_slice, match))

        stage_count = len(batch_items[0][1].program_matches)
        if any(len(item[1].program_matches) != stage_count for item in batch_items[1:]):
            return None

        row_sizes = [int(item[0].count) for item in batch_items]
        slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
        slot_sizes = [int(item[0].count * item[0].arity) for item in batch_items]
        slot_offsets_local: list[int] = []
        cursor = 0
        for slot_size in slot_sizes:
            slot_offsets_local.append(cursor)
            cursor += slot_size

        original_slots = self._gather_stage_input_slots(
            x,
            relation_args,
            slot_offsets=slot_offsets_global,
            row_sizes=row_sizes,
            arity=int(grouped_batch.arity),
        )
        original_idx = torch.cat(
            [
                relation_args[relation_slice.slot_start : relation_slice.slot_end]
                for relation_slice, _ in batch_items
            ],
            dim=0,
        )

        if stage_count == 2 and bool(self.fused_two_layer_pointwise_execution):
            stage0_matches = [item[1].program_matches[0] for item in batch_items]
            stage1_matches = [item[1].program_matches[1] for item in batch_items]
            if all(
                item[1].program_family is not None
                and item[1].program_family.family == "program_two_layer_silu_then_two_layer_silu"
                for item in batch_items
            ):
                program_key = (
                    "manual_program",
                    grouped_batch.family,
                    grouped_batch.arity,
                    grouped_batch.signature,
                )
                w10_stack = self._get_grouped_param_stack(
                    cache_key=("w10", program_key),
                    tensors=[match.linears[0].weight for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b10_stack = self._get_grouped_param_stack(
                    cache_key=("b10", program_key),
                    tensors=[cast(Tensor, match.linears[0].bias) for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w20_stack = self._get_grouped_param_stack(
                    cache_key=("w20", program_key),
                    tensors=[match.linears[1].weight for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b20_stack = self._get_grouped_param_stack(
                    cache_key=("b20", program_key),
                    tensors=[cast(Tensor, match.linears[1].bias) for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w11_stack = self._get_grouped_param_stack(
                    cache_key=("w11", program_key),
                    tensors=[match.linears[0].weight for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b11_stack = self._get_grouped_param_stack(
                    cache_key=("b11", program_key),
                    tensors=[cast(Tensor, match.linears[0].bias) for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w21_stack = self._get_grouped_param_stack(
                    cache_key=("w21", program_key),
                    tensors=[match.linears[1].weight for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b21_stack = self._get_grouped_param_stack(
                    cache_key=("b21", program_key),
                    tensors=[cast(Tensor, match.linears[1].bias) for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                return relm_mp_ops.fused_program_two_layer_silu_then_two_layer_silu_from_indices(
                    x,
                    relation_args,
                    slot_offsets_global,
                    row_sizes,
                    int(grouped_batch.arity),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                )
            if all(
                item[1].program_family is not None
                and item[1].program_family.family
                == "program_two_layer_silu_then_postnorm_two_layer_silu"
                for item in batch_items
            ):
                norm_signature = stage1_matches[0].spec.signature[-1]
                if norm_signature is None or str(norm_signature[0]) != "layernorm":
                    return None
                if any(match.spec.signature[-1] != norm_signature for match in stage1_matches[1:]):
                    return None
                ln_eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
                program_key = (
                    "manual_program",
                    grouped_batch.family,
                    grouped_batch.arity,
                    grouped_batch.signature,
                )
                w10_stack = self._get_grouped_param_stack(
                    cache_key=("w10", program_key),
                    tensors=[match.linears[0].weight for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b10_stack = self._get_grouped_param_stack(
                    cache_key=("b10", program_key),
                    tensors=[cast(Tensor, match.linears[0].bias) for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w20_stack = self._get_grouped_param_stack(
                    cache_key=("w20", program_key),
                    tensors=[match.linears[1].weight for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b20_stack = self._get_grouped_param_stack(
                    cache_key=("b20", program_key),
                    tensors=[cast(Tensor, match.linears[1].bias) for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w11_stack = self._get_grouped_param_stack(
                    cache_key=("w11", program_key),
                    tensors=[match.linears[0].weight for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b11_stack = self._get_grouped_param_stack(
                    cache_key=("b11", program_key),
                    tensors=[cast(Tensor, match.linears[0].bias) for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w21_stack = self._get_grouped_param_stack(
                    cache_key=("w21", program_key),
                    tensors=[match.linears[1].weight for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b21_stack = self._get_grouped_param_stack(
                    cache_key=("b21", program_key),
                    tensors=[cast(Tensor, match.linears[1].bias) for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                if stage1_matches[0].norm_modules[0].weight is not None:
                    ln_weight_stack = self._get_grouped_param_stack(
                        cache_key=("ln_weight", program_key),
                        tensors=[
                            cast(Tensor, match.norm_modules[0].weight)
                            for match in stage1_matches
                            if match.norm_modules[0].weight is not None
                        ],
                        forward_cache=grouped_param_stacks,
                        allow_persistent=allow_persistent_stacks,
                    )
                else:
                    ln_weight_stack = w21_stack.new_empty((0,))
                if stage1_matches[0].norm_modules[0].bias is not None:
                    ln_bias_stack = self._get_grouped_param_stack(
                        cache_key=("ln_bias", program_key),
                        tensors=[
                            cast(Tensor, match.norm_modules[0].bias)
                            for match in stage1_matches
                            if match.norm_modules[0].bias is not None
                        ],
                        forward_cache=grouped_param_stacks,
                        allow_persistent=allow_persistent_stacks,
                    )
                else:
                    ln_bias_stack = w21_stack.new_empty((0,))
                return relm_mp_ops.fused_program_two_layer_silu_then_postnorm_two_layer_silu_from_indices(
                    x,
                    relation_args,
                    slot_offsets_global,
                    row_sizes,
                    int(grouped_batch.arity),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                    ln_weight_stack,
                    ln_bias_stack,
                    float(ln_eps),
                )
            if all(
                item[1].program_family is not None
                and item[1].program_family.family
                == "program_prenorm_two_layer_silu_rmsnorm_then_two_layer_silu"
                for item in batch_items
            ):
                norm_signature = stage0_matches[0].spec.signature[5]
                if norm_signature is None or str(norm_signature[0]) != "rmsnorm":
                    return None
                if any(match.spec.signature[5] != norm_signature for match in stage0_matches[1:]):
                    return None
                rms_eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
                program_key = (
                    "manual_program",
                    grouped_batch.family,
                    grouped_batch.arity,
                    grouped_batch.signature,
                )
                if stage0_matches[0].norm_modules[0].weight is not None:
                    rms_weight_stack = self._get_grouped_param_stack(
                        cache_key=("rms_weight", program_key),
                        tensors=[
                            cast(Tensor, match.norm_modules[0].weight)
                            for match in stage0_matches
                            if match.norm_modules[0].weight is not None
                        ],
                        forward_cache=grouped_param_stacks,
                        allow_persistent=allow_persistent_stacks,
                    )
                else:
                    rms_weight_stack = stage0_matches[0].linears[1].weight.new_empty((0,))
                w10_stack = self._get_grouped_param_stack(
                    cache_key=("w10", program_key),
                    tensors=[match.linears[0].weight for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b10_stack = self._get_grouped_param_stack(
                    cache_key=("b10", program_key),
                    tensors=[cast(Tensor, match.linears[0].bias) for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w20_stack = self._get_grouped_param_stack(
                    cache_key=("w20", program_key),
                    tensors=[match.linears[1].weight for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b20_stack = self._get_grouped_param_stack(
                    cache_key=("b20", program_key),
                    tensors=[cast(Tensor, match.linears[1].bias) for match in stage0_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w11_stack = self._get_grouped_param_stack(
                    cache_key=("w11", program_key),
                    tensors=[match.linears[0].weight for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b11_stack = self._get_grouped_param_stack(
                    cache_key=("b11", program_key),
                    tensors=[cast(Tensor, match.linears[0].bias) for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                w21_stack = self._get_grouped_param_stack(
                    cache_key=("w21", program_key),
                    tensors=[match.linears[1].weight for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                b21_stack = self._get_grouped_param_stack(
                    cache_key=("b21", program_key),
                    tensors=[cast(Tensor, match.linears[1].bias) for match in stage1_matches],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
                return relm_mp_ops.fused_program_prenorm_two_layer_silu_rmsnorm_then_two_layer_silu_from_indices(
                    x,
                    relation_args,
                    slot_offsets_global,
                    row_sizes,
                    int(grouped_batch.arity),
                    rms_weight_stack,
                    float(rms_eps),
                    w10_stack,
                    b10_stack,
                    w20_stack,
                    b20_stack,
                    w11_stack,
                    b11_stack,
                    w21_stack,
                    b21_stack,
                )

        current_x = x
        current_relation_args = relation_args
        current_slot_offsets = slot_offsets_global
        local_relation_args: Tensor | None = None
        current_dense: Tensor | None = None

        for stage_idx in range(stage_count):
            stage_matches = [item[1].program_matches[stage_idx] for item in batch_items]
            stage_result = self._execute_fused_program_stage(
                current_x,
                current_relation_args,
                slot_offsets=current_slot_offsets,
                row_sizes=row_sizes,
                arity=int(grouped_batch.arity),
                stage_matches=stage_matches,
                grouped_param_stacks=grouped_param_stacks,
                allow_persistent_stacks=allow_persistent_stacks,
                cache_key_prefix=(
                    "program",
                    grouped_batch.family,
                    grouped_batch.arity,
                    grouped_batch.signature,
                    stage_idx,
                ),
            )
            if stage_result is None:
                dense_out = self._execute_dense_program_stage(
                    original_slots if stage_idx == 0 else cast(Tensor, current_dense),
                    row_sizes=row_sizes,
                    arity=int(grouped_batch.arity),
                    stage_matches=stage_matches,
                )
                current_dense = dense_out
                current_x = dense_out
                current_relation_args = (
                    torch.arange(
                        int(dense_out.size(0)),
                        device=dense_out.device,
                        dtype=relation_args.dtype,
                    )
                    if local_relation_args is None
                    else local_relation_args
                )
                current_slot_offsets = slot_offsets_local
                if local_relation_args is None:
                    local_relation_args = current_relation_args
                continue

            current_dense, _stage_idx_out = stage_result
            current_x = current_dense
            if local_relation_args is None:
                local_relation_args = torch.arange(
                    int(current_dense.size(0)),
                    device=current_dense.device,
                    dtype=relation_args.dtype,
                )
            current_relation_args = local_relation_args
            current_slot_offsets = slot_offsets_local

        if current_dense is None:
            return None
        return original_slots + current_dense, original_idx

    def _collect_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        msg_chunks: list[Tensor] = []
        idx_chunks: list[Tensor] = []
        use_fused_two_layer_mish = self._use_fused_two_layer_pointwise_execution(x)
        arg_emb_all = (
            x.index_select(0, relation_args)
            if (not use_fused_two_layer_mish)
            and self._use_fused_relation_gather(x)
            and int(relation_args.numel()) > 0
            else None
        )

        if use_fused_two_layer_mish:
            layout = self._get_fused_relation_layout(topology)
            grouped_param_stacks = (
                cache.setdefault("fused_relation_param_stacks", {}) if cache is not None else {}
            )
            allow_persistent_stacks = (not self.training) and (not torch.is_grad_enabled())
            consumed: set[int] = set()
            fallback_arg_emb_all = (
                x.index_select(0, relation_args)
                if layout["fallback_indices"]
                and self._use_fused_relation_gather(x)
                and int(relation_args.numel()) > 0
                else None
            )
            for grouped_batch in layout["groups"]:
                collector = self._fused_relation_collectors.get(grouped_batch.family)
                if collector is None:
                    continue
                grouped = collector(
                    x,
                    relation_args,
                    topology,
                    grouped_batch,
                    grouped_param_stacks=grouped_param_stacks,
                    allow_persistent_stacks=allow_persistent_stacks,
                )
                if grouped is None:
                    continue
                msgs, idx = grouped
                msg_chunks.append(msgs)
                idx_chunks.append(idx)
                consumed.update(int(idx_i) for idx_i in grouped_batch.relation_indices)
            for relation_index in layout["fallback_indices"]:
                relation_slice = topology.relation_slices[relation_index]
                direct = self._collect_direct_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, idx = direct
                msg_chunks.append(msgs)
                idx_chunks.append(idx)
                consumed.add(relation_index)
            for relation_slice in topology.relation_slices:
                if relation_slice.relation_index in consumed:
                    continue
                direct = self._collect_direct_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, idx = direct
                msg_chunks.append(msgs)
                idx_chunks.append(idx)
        else:
            for relation_slice in topology.relation_slices:
                direct = self._collect_direct_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, idx = direct
                msg_chunks.append(msgs)
                idx_chunks.append(idx)

        if not msg_chunks:
            return None
        return torch.cat(msg_chunks, dim=0), torch.cat(idx_chunks, dim=0)

    def get_topology(
        self,
        relation_counts: Tensor,
        relation_arities: Tensor | Sequence[int] | Iterable[int] | None = None,
        *,
        cache: dict | None = None,
    ) -> FlatTopology:
        arities = (
            self.relation_arities
            if relation_arities is None
            else normalize_relation_arities(relation_arities)
        )
        topology = build_flat_topology(relation_counts, arities)
        cache_key = _topology_cache_key(topology)
        if cache is not None:
            cached = cache.get(cache_key)
            if isinstance(cached, FlatTopology):
                return cached
        cached = self._persistent_topology_cache.get(cache_key)
        if cached is not None:
            if cache is not None:
                cache[cache_key] = cached
            return cached
        self._persistent_topology_cache[cache_key] = topology
        if cache is not None:
            cache[cache_key] = topology
        return topology

    def forward(
        self,
        x: Tensor,
        relation_counts: Tensor,
        relation_args: Tensor,
        *,
        relation_arities: Tensor | Sequence[int] | Iterable[int] | None = None,
        topology: FlatTopology | None = None,
        cache: dict | None = None,
    ) -> Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be rank-2, got shape {tuple(x.shape)}.")
        if int(x.size(1)) != self.embedding_size:
            raise ValueError(
                f"x must have feature size {self.embedding_size}, got {int(x.size(1))}."
            )
        relation_args = relation_args.to(device=x.device).view(-1)
        if relation_args.dtype not in (torch.int32, torch.int64):
            relation_args = relation_args.to(dtype=torch.long)
        if topology is None:
            topology = self.get_topology(
                relation_counts, relation_arities=relation_arities, cache=cache
            )
        if topology.slot_offsets[-1] != int(relation_args.numel()):
            raise ValueError(
                "relation_args length does not match the packed slot count implied by "
                f"relation_counts/relation_arities: {int(relation_args.numel())} vs "
                f"{int(topology.slot_offsets[-1])}."
            )

        if isinstance(self._aggr_query, str):
            aggr_name = self._aggr_query.lower()
        else:
            aggr_name = ""

        if aggr_name == "sum":
            collected = self._collect_messages(
                x, relation_args, topology, cache=cache
            )
            aggregated = x.new_zeros((int(x.size(0)), self.embedding_size))
            if collected is None:
                return aggregated
            msgs, idx = collected
            aggregated.index_add_(0, idx, msgs)
            return aggregated

        if aggr_name == "mean":
            collected = self._collect_messages(
                x, relation_args, topology, cache=cache
            )
            aggregated = x.new_zeros((int(x.size(0)), self.embedding_size))
            if collected is None:
                return aggregated
            msgs, idx = collected
            counts = x.new_zeros((int(x.size(0)), 1))
            aggregated.index_add_(0, idx, msgs)
            counts.index_add_(
                0,
                idx,
                torch.ones((int(idx.numel()), 1), device=x.device, dtype=x.dtype),
            )
            return aggregated / counts.clamp_min_(1.0)

        collected = self._collect_messages(
            x, relation_args, topology, cache=cache
        )
        if collected is None:
            return x.new_zeros((int(x.size(0)), self.embedding_size))
        msgs, idx = collected
        return self.aggr(x=msgs, index=idx, dim=0, dim_size=int(x.size(0)))
