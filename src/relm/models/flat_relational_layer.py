"""Flat relation message-passing kernel dispatch.

This module owns the packed relation topology helpers, exact kernel-family
matching, and runtime dispatch between:
1. exact single-block CUDA kernels
2. exact RelationProgram CUDA kernels
3. eager per-relation fallback execution
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Hashable, Iterable, Sequence, cast

import torch
import torch_geometric.nn.aggr
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver

from ..ops import mp as relm_mp_ops
from .aggr import LogSumExpAggregation
from .flat_contract import FlatExecutionPolicy
from .relation_block_spec import RelationBlockSpec
from .relation_blocks import RelationProgram


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
class KernelBatchPlan:
    kernel: "FlatRelationKernel"
    signature: Hashable
    arity: int
    relation_indices: tuple[int, ...]
    max_rows: int
    row_sizes: tuple[int, ...]


@dataclass(frozen=True)
class BlockKernelSpec:
    kernel_type: type["FlatRelationKernel"]
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
class ProgramKernelSpec:
    kernel_type: type["FlatRelationKernel"]
    signature: Hashable
    arity: int
    input_dim: int
    output_dim: int
    block_specs: tuple[BlockKernelSpec, ...]


@dataclass(frozen=True)
class KernelMatch:
    spec: BlockKernelSpec
    linears: tuple[torch.nn.Linear, ...]
    kernel: "FlatRelationKernel | None" = None
    pointwise_modules: tuple[torch.nn.Module, ...] = ()
    norm_modules: tuple[torch.nn.Module, ...] = ()
    program_matches: tuple["KernelMatch", ...] = ()
    program_spec: ProgramKernelSpec | None = None


@dataclass(frozen=True)
class CentralizedBatchSpec:
    central_module: torch.nn.Module
    condition_embedding: torch.nn.Embedding
    condition_position: str
    max_arity: int
    embedding_size: int
    include_slot_mask: bool
    condition_indices: tuple[int, ...]


class FlatRelationKernel(ABC):
    """Interface for exact flat-kernel families.

    A kernel object owns:
    1. matching a relation slice to a supported family/spec
    2. collecting packed slot messages for a grouped execution batch
    3. optionally collecting pooled relation-instance rows for grouped LGAN use
    """

    @abstractmethod
    def match(
        self,
        layer: "FlatRelationalLayer",
        relation_slice: RelationSlice,
    ) -> KernelMatch | None: ...

    @abstractmethod
    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None: ...

    def collect_relation_instances(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        relation_row_starts: dict[int, int],
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        grouped = self.collect(
            layer,
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )
        if grouped is None:
            return None
        msgs, _ = grouped
        return layer._pool_grouped_kernel_messages(
            topology,
            grouped_batch,
            relation_row_starts,
            msgs,
            device=x.device,
            index_dtype=relation_args.dtype,
        )

class MishBlockKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice, kernel_type=type(self), pointwise_kind="mish"
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(MishBlockKernel, SiLUBlockKernel, GELUBlockKernel),
        )




class SiLUBlockKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice, kernel_type=type(self), pointwise_kind="silu"
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(MishBlockKernel, SiLUBlockKernel, GELUBlockKernel),
        )




class GELUBlockKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice, kernel_type=type(self), pointwise_kind="gelu"
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_two_layer_pointwise_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(MishBlockKernel, SiLUBlockKernel, GELUBlockKernel),
        )




class PostNormMishLayerNormKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice,
                kernel_type=type(self),
                pointwise_kind="mish",
                norm_position="post",
                norm_kind="layernorm",
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_postnorm_layernorm_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(PostNormMishLayerNormKernel, PostNormSiLULayerNormKernel),
        )


class PostNormSiLULayerNormKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice,
                kernel_type=type(self),
                pointwise_kind="silu",
                norm_position="post",
                norm_kind="layernorm",
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_postnorm_layernorm_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
            expected_kernel_types=(PostNormMishLayerNormKernel, PostNormSiLULayerNormKernel),
        )


class PreNormSiLURMSNormKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            layer._match_two_layer_pointwise_family(
                relation_slice,
                kernel_type=type(self),
                pointwise_kind="silu",
                norm_position="pre",
                norm_kind="rmsnorm",
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_prenorm_rmsnorm_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class SiLUPairProgramKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            _match_expected_program_kernel(
                layer._match_exact_relation_program(
                    relation_slice, program_kernel_type=type(self)
                ),
                type(self),
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_silu_pair_program_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class SiLUThenPostNormProgramKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            _match_expected_program_kernel(
                layer._match_exact_relation_program(
                    relation_slice, program_kernel_type=type(self)
                ),
                type(self),
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_silu_then_postnorm_program_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class PreNormRMSNormThenSiLUProgramKernel(FlatRelationKernel):
    def _bind(self, match: KernelMatch | None) -> KernelMatch | None:
        if match is None:
            return None
        return replace(match, kernel=self)

    def match(self, layer: "FlatRelationalLayer", relation_slice: RelationSlice) -> KernelMatch | None:
        return self._bind(
            _match_expected_program_kernel(
                layer._match_exact_relation_program(
                    relation_slice, program_kernel_type=type(self)
                ),
                type(self),
            )
        )

    def collect(
        self,
        layer: "FlatRelationalLayer",
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        return layer._run_prenorm_rmsnorm_then_silu_program_kernel(
            x,
            relation_args,
            topology,
            grouped_batch,
            grouped_param_stacks=grouped_param_stacks,
            allow_persistent_stacks=allow_persistent_stacks,
        )


class _ReLUBlockShapeSpec:
    pass


class _ThreeLayerSiLUBlockShapeSpec:
    pass


_KERNEL_SPEC_METHODS = ("relm_kernel_spec",)
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
    (_LAYER_NORM_TYPE,)
    if _RMS_NORM_TYPE is None
    else (_LAYER_NORM_TYPE, _RMS_NORM_TYPE)
)


def _pointwise_signature(module: torch.nn.Module) -> tuple[Any, ...] | None:
    if not isinstance(module, _SUPPORTED_POINTWISE_TYPES):
        return None
    if isinstance(module, torch.nn.GELU):
        return ("gelu", str(module.approximate))
    if isinstance(module, torch.nn.LeakyReLU):
        return (
            "leaky_relu",
            float(module.negative_slope),
            bool(module.inplace),
        )
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


def _extract_relation_block_info(
    module: torch.nn.Module,
) -> dict[str, Any] | None:
    for method_name in _KERNEL_SPEC_METHODS:
        method = getattr(module, method_name, None)
        if not callable(method):
            continue
        spec = method()
        if spec is None:
            return None
        if isinstance(spec, RelationBlockSpec):
            spec = {
                "linears": tuple(spec.linears),
                "ops": tuple(spec.ops),
                "signature": spec.signature,
            }
        if not isinstance(spec, dict):
            raise TypeError(
                f"{type(module).__name__}.{method_name}() must return RelationBlockSpec|dict|None, got {type(spec)!r}."
            )
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
    return None


def _match_expected_program_kernel(
    match: KernelMatch | None,
    kernel_type: type[FlatRelationKernel],
) -> KernelMatch | None:
    if (
        match is None
        or match.program_spec is None
        or match.program_spec.kernel_type is not kernel_type
    ):
        return None
    return match


def normalize_relation_arities(
    relation_arities: Tensor | Sequence[int] | Iterable[int],
    *,
    device: torch.device | None = None,
) -> Tensor:
    if torch.is_tensor(relation_arities):
        out = relation_arities.to(device=device, dtype=torch.long)
    else:
        out = torch.as_tensor(
            tuple(int(x) for x in relation_arities),
            dtype=torch.long,
            device=device,
        )
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
    for relation_index, (count_t, arity_t) in enumerate(
        zip(counts_total, arities_1d)
    ):
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


def _topology_cache_key(
    topology: FlatTopology,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return topology.relation_counts_total, topology.relation_arities


class FlatRelationalLayer(torch.nn.Module):
    """Flat relation message passing over packed relation tensors.

    Input contract:
        ``x`` has shape ``[num_nodes, embedding_size]``.
        ``relation_counts`` has shape ``[batch_size, num_relations]``.
        ``relation_args`` is a 1D packed node-index tensor whose length matches
        the slot count implied by ``relation_counts`` and ``relation_arities``.

    Output contract:
        returns aggregated relation messages of shape
        ``[num_nodes, embedding_size]``.

    Residual semantics:
        relation blocks and relation programs do not add the outer tuple
        residual. This layer adds the gathered tuple slots exactly once when it
        materializes messages.
    """

    def __init__(
        self,
        *,
        update_modules: Sequence[torch.nn.Module],
        relation_names: Sequence[str],
        relation_arities: Tensor | Sequence[int] | Iterable[int],
        embedding_size: int,
        aggregation: str | torch_geometric.nn.aggr.Aggregation | None = None,
        execution_policy: FlatExecutionPolicy = FlatExecutionPolicy(),
        kernels: Sequence[FlatRelationKernel] | None = None,
    ) -> None:
        super().__init__()
        self.embedding_size = int(embedding_size)
        self.relation_names = tuple(str(name) for name in relation_names)
        self.relation_arities = normalize_relation_arities(
            relation_arities
        ).cpu()
        self.execution_policy = execution_policy
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
        aggr_query = aggregation or "logsumexp"
        if isinstance(aggr_query, str):
            if aggr_query.lower() == "logsumexp":
                self.aggr = LogSumExpAggregation()
            else:
                self.aggr = aggregation_resolver(aggr_query)
        else:
            self.aggr = aggr_query
        self._persistent_topology_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]], FlatTopology
        ] = {}
        self._relation_block_info_cache: dict[int, dict[str, Any] | None] = {}
        self._persistent_grouped_param_stacks: dict[
            tuple[Any, ...], dict[str, Any]
        ] = {}
        self._kernel_match_cache: dict[tuple[int, int], KernelMatch | None] = {}
        self._persistent_kernel_layout_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...]],
            dict[str, tuple[KernelBatchPlan, ...] | tuple[int, ...]],
        ] = {}
        self.kernels = tuple(kernels) if kernels is not None else self._build_default_kernels()
        self._centralized_batch_spec_cache = self._build_centralized_batch_spec()

    def _build_centralized_batch_spec(self) -> CentralizedBatchSpec | None:
        if not self.update_modules:
            return None
        first = self.update_modules[0]
        required_attrs = (
            "central_module",
            "condition_embedding",
            "condition_index",
            "max_arity",
            "embedding_size",
            "condition_position",
            "include_slot_mask",
        )
        if any(not hasattr(first, attr) for attr in required_attrs):
            return None
        central_module = cast(torch.nn.Module, getattr(first, "central_module"))
        condition_embedding = cast(
            torch.nn.Embedding, getattr(first, "condition_embedding")
        )
        condition_position = str(getattr(first, "condition_position"))
        max_arity = int(getattr(first, "max_arity"))
        embedding_size = int(getattr(first, "embedding_size"))
        include_slot_mask = bool(getattr(first, "include_slot_mask"))
        condition_indices: list[int] = []
        for relation_index, module in enumerate(self.update_modules):
            if any(not hasattr(module, attr) for attr in required_attrs):
                return None
            if getattr(module, "central_module") is not central_module:
                return None
            if getattr(module, "condition_embedding") is not condition_embedding:
                return None
            if str(getattr(module, "condition_position")) != condition_position:
                return None
            if int(getattr(module, "max_arity")) != max_arity:
                return None
            if int(getattr(module, "embedding_size")) != embedding_size:
                return None
            if bool(getattr(module, "include_slot_mask")) != include_slot_mask:
                return None
            if int(getattr(module, "arity", self.relation_arities[relation_index])) != int(
                self.relation_arities[relation_index]
            ):
                return None
            condition_indices.append(int(getattr(module, "condition_index")))
        if embedding_size != self.embedding_size:
            return None
        return CentralizedBatchSpec(
            central_module=central_module,
            condition_embedding=condition_embedding,
            condition_position=condition_position,
            max_arity=max_arity,
            embedding_size=embedding_size,
            include_slot_mask=include_slot_mask,
            condition_indices=tuple(condition_indices),
        )

    def _centralized_batch_spec(self) -> CentralizedBatchSpec | None:
        return self._centralized_batch_spec_cache

    def _collect_centralized_relation_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        spec: CentralizedBatchSpec,
        arg_emb_all: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        if int(relation_args.numel()) == 0:
            return None
        arg_emb_all = x.index_select(0, relation_args) if arg_emb_all is None else arg_emb_all
        target_width = int(spec.max_arity * self.embedding_size)
        cond_dim = int(spec.condition_embedding.weight.size(-1))
        input_rows: list[Tensor] = []
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0 or relation_slice.arity <= 0:
                continue
            rel_in = arg_emb_all[
                relation_slice.slot_start : relation_slice.slot_end
            ].view(relation_slice.count, relation_slice.arity * self.embedding_size)
            if int(rel_in.size(-1)) < target_width:
                pad = rel_in.new_zeros((int(rel_in.size(0)), target_width - int(rel_in.size(-1))))
                rel_in = torch.cat([rel_in, pad], dim=-1)
            pieces = [rel_in]
            if spec.include_slot_mask:
                mask = rel_in.new_zeros((int(rel_in.size(0)), spec.max_arity))
                mask[:, : relation_slice.arity] = 1.0
                pieces.append(mask)
            cond_idx = torch.tensor(
                spec.condition_indices[relation_slice.relation_index],
                device=x.device,
            )
            cond = spec.condition_embedding(cond_idx).view(1, cond_dim).expand(
                int(rel_in.size(0)), cond_dim
            )
            if spec.condition_position == "pre":
                input_rows.append(torch.cat([cond, *pieces], dim=-1))
            else:
                input_rows.append(torch.cat([*pieces, cond], dim=-1))
        if not input_rows:
            return None
        central_in = torch.cat(input_rows, dim=0)
        central_out = spec.central_module(central_in)
        msg_chunks: list[Tensor] = []
        row_cursor = 0
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0 or relation_slice.arity <= 0:
                continue
            row_end = row_cursor + relation_slice.count
            rel_out = central_out[row_cursor:row_end, : relation_slice.arity * self.embedding_size]
            msg_chunks.append(
                rel_out.contiguous().view(
                    relation_slice.count * relation_slice.arity,
                    self.embedding_size,
                )
            )
            row_cursor = row_end
        rel_out_flat = torch.cat(msg_chunks, dim=0)
        return arg_emb_all + rel_out_flat, relation_args

    def _relation_block_info(
        self, module: torch.nn.Module
    ) -> dict[str, Any] | None:
        key = id(module)
        cached = self._relation_block_info_cache.get(key)
        if cached is not None or key in self._relation_block_info_cache:
            return cached
        info = _extract_relation_block_info(module)
        self._relation_block_info_cache[key] = info
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
            versions = tuple(
                int(getattr(tensor, "_version", -1)) for tensor in tensors
            )
            persistent = self._persistent_grouped_param_stacks.get(cache_key)
            if persistent is not None:
                stacked = persistent.get("tensor")
                if (
                    torch.is_tensor(stacked)
                    and persistent.get("versions") == versions
                    and tuple(stacked.shape)
                    == tuple(persistent.get("shape", ()))
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
                "versions": tuple(
                    int(getattr(tensor, "_version", -1)) for tensor in tensors
                ),
                "shape": tuple(stacked.shape),
            }
        return stacked

    def _use_relation_kernels(self, x: Tensor) -> bool:
        return self.execution_policy.use_relation_kernels(device=x.device)

    def _use_program_kernels(self, x: Tensor) -> bool:
        return self.execution_policy.use_program_kernels(device=x.device)

    def _use_relation_gather(self, x: Tensor) -> bool:
        return self.execution_policy.use_relation_gather(device=x.device)

    def _build_default_kernels(self) -> tuple[FlatRelationKernel, ...]:
        return (
            MishBlockKernel(),
            SiLUBlockKernel(),
            GELUBlockKernel(),
            PostNormMishLayerNormKernel(),
            PostNormSiLULayerNormKernel(),
            PreNormSiLURMSNormKernel(),
            SiLUPairProgramKernel(),
            SiLUThenPostNormProgramKernel(),
            PreNormRMSNormThenSiLUProgramKernel(),
        )

    def _two_linear_pointwise_info(
        self,
        relation_slice: RelationSlice,
    ) -> dict[str, Any] | None:
        module = self.update_modules[relation_slice.relation_index]
        info = self._relation_block_info(module)
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
        if (
            int(lin0.in_features) != expected_dim
            or int(lin1.out_features) != expected_dim
        ):
            return None
        if int(lin1.in_features) != int(lin0.out_features):
            return None
        return {
            "expected_dim": expected_dim,
            "hidden_dim": int(lin0.out_features),
            "linear0": lin0,
            "linear1": lin1,
            "bias_flags": (
                bool(lin0.bias is not None),
                bool(lin1.bias is not None),
            ),
            "pointwise_module": pointwise_module,
            "pointwise_signature": pointwise_signature,
        }

    def _match_two_layer_mish_spec(
        self,
        relation_slice: RelationSlice,
    ) -> KernelMatch | None:
        return self._match_two_layer_pointwise_family(
            relation_slice, kernel_type=MishBlockKernel, pointwise_kind="mish"
        )

    def _normalized_grouped_ops_for_module(
        self,
        module: torch.nn.Module,
        *,
        expected_dim: int,
    ) -> dict[str, Any] | None:
        info = self._relation_block_info(module)
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
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
        norm_position: str | None = None,
        norm_kind: str | None = None,
    ) -> KernelMatch | None:
        if norm_position is None:
            if len(ops) != 3:
                return None
            if (
                ops[0][0] != "linear"
                or ops[1][0] != "pointwise"
                or ops[2][0] != "linear"
            ):
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
        if (
            int(linear0.in_features) != expected_dim
            or int(linear1.out_features) != expected_dim
        ):
            return None
        if int(linear1.in_features) != int(linear0.out_features):
            return None
        if norm_signature is not None and not self._norm_matches_expected_dim(
            norm_signature, expected_dim, norm_kind=norm_kind
        ):
            return None
        spec = BlockKernelSpec(
            kernel_type=kernel_type,
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
            bias_flags=(
                bool(linear0.bias is not None),
                bool(linear1.bias is not None),
            ),
            pointwise_signature=tuple(pointwise_signature),
            norm_kind=(
                str(norm_signature[0]) if norm_signature is not None else None
            ),
            norm_position=norm_position,
        )
        return KernelMatch(
            spec=spec,
            linears=(linear0, linear1),
            pointwise_modules=(pointwise_module,),
            norm_modules=((norm_module,) if norm_module is not None else ()),
        )

    def _match_two_layer_pointwise_family(
        self,
        relation_slice: RelationSlice,
        *,
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
        norm_position: str | None = None,
        norm_kind: str | None = None,
    ) -> KernelMatch | None:
        info = self._normalized_grouped_ops(relation_slice)
        if info is None:
            return None
        return self._match_two_layer_pointwise_ops(
            relation_slice=relation_slice,
            ops=tuple(info["normalized_ops"]),
            expected_dim=int(info["expected_dim"]),
            kernel_type=kernel_type,
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
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
    ) -> KernelMatch | None:
        if len(ops) != 5:
            return None
        expected_kinds = (
            "linear",
            "pointwise",
            "linear",
            "pointwise",
            "linear",
        )
        if tuple(item[0] for item in ops) != expected_kinds:
            return None
        linear0 = ops[0][1]
        pointwise0 = ops[1]
        linear1 = ops[2][1]
        pointwise1 = ops[3]
        linear2 = ops[4][1]
        if (
            pointwise0[2][0] != pointwise_kind
            or pointwise1[2][0] != pointwise_kind
        ):
            return None
        if int(linear0.in_features) != expected_dim:
            return None
        if int(linear2.out_features) != expected_dim:
            return None
        if int(linear1.in_features) != int(linear0.out_features):
            return None
        if int(linear2.in_features) != int(linear1.out_features):
            return None
        spec = BlockKernelSpec(
            kernel_type=kernel_type,
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
        return KernelMatch(
            spec=spec,
            linears=(linear0, linear1, linear2),
            pointwise_modules=(pointwise0[1], pointwise1[1]),
        )

    def _match_three_layer_pointwise_family(
        self,
        relation_slice: RelationSlice,
        *,
        kernel_type: type[FlatRelationKernel],
        pointwise_kind: str,
    ) -> KernelMatch | None:
        info = self._normalized_grouped_ops(relation_slice)
        if info is None:
            return None
        return self._match_three_layer_pointwise_ops(
            relation_slice=relation_slice,
            ops=tuple(info["normalized_ops"]),
            expected_dim=int(info["expected_dim"]),
            kernel_type=kernel_type,
            pointwise_kind=pointwise_kind,
        )

    def _match_exact_relation_program(
        self,
        relation_slice: RelationSlice,
        *,
        program_kernel_type: type[FlatRelationKernel] | None = None,
    ) -> KernelMatch | None:
        module = self.update_modules[relation_slice.relation_index]
        if not isinstance(module, RelationProgram):
            return None
        expected_dim = int(relation_slice.arity * self.embedding_size)
        if int(module.width) != expected_dim:
            return None
        stages = []
        for block in module:
            stage_match = self._match_program_block(
                relation_slice=relation_slice,
                block=block,
                expected_dim=expected_dim,
            )
            if stage_match is None:
                return None
            stages.append(stage_match)
        if len(stages) <= 1:
            return None
        program_spec = self._match_program_kernel_spec(
            relation_slice,
            stages=tuple(stages),
            expected_dim=expected_dim,
        )
        if program_spec is None:
            return None
        return KernelMatch(
            spec=BlockKernelSpec(
                kernel_type=program_spec.kernel_type,
                signature=program_spec.signature,
                arity=int(relation_slice.arity),
                input_dim=expected_dim,
                output_dim=expected_dim,
                hidden_dims=tuple(),
                bias_flags=tuple(),
            ),
            linears=tuple(),
            program_matches=tuple(stages),
            program_spec=program_spec,
        )

    def _match_program_block(
        self,
        *,
        relation_slice: RelationSlice,
        block: torch.nn.Module,
        expected_dim: int,
    ) -> KernelMatch | None:
        info = self._normalized_grouped_ops_for_module(
            block, expected_dim=expected_dim
        )
        if info is None:
            return None
        ops = tuple(info["normalized_ops"])
        for candidate_len, matcher in (
            (
                4,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=PreNormSiLURMSNormKernel,
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
                    kernel_type=PostNormSiLULayerNormKernel,
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
                    kernel_type=MishBlockKernel,
                    pointwise_kind="mish",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=SiLUBlockKernel,
                    pointwise_kind="silu",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=GELUBlockKernel,
                    pointwise_kind="gelu",
                ),
            ),
            (
                3,
                lambda stage_ops: self._match_two_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=_ReLUBlockShapeSpec,
                    pointwise_kind="relu",
                ),
            ),
            (
                5,
                lambda stage_ops: self._match_three_layer_pointwise_ops(
                    relation_slice=relation_slice,
                    ops=stage_ops,
                    expected_dim=expected_dim,
                    kernel_type=_ThreeLayerSiLUBlockShapeSpec,
                    pointwise_kind="silu",
                ),
            ),
        ):
            if len(ops) != candidate_len:
                continue
            candidate = matcher(ops)
            if candidate is not None:
                return candidate
        return None

    def _match_program_kernel_spec(
        self,
        relation_slice: RelationSlice,
        *,
        stages: tuple[KernelMatch, ...],
        expected_dim: int,
    ) -> ProgramKernelSpec | None:
        if len(stages) != 2:
            return None
        stage0, stage1 = stages
        if (
            stage0.spec.output_dim != expected_dim
            or stage1.spec.output_dim != expected_dim
        ):
            return None
        if (
            stage0.spec.kernel_type is SiLUBlockKernel
            and stage1.spec.kernel_type is SiLUBlockKernel
        ):
            return ProgramKernelSpec(
                kernel_type=SiLUPairProgramKernel,
                signature=(
                    stage0.spec.kernel_type,
                    stage0.spec.signature,
                    stage1.spec.kernel_type,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        if (
            stage0.spec.kernel_type is SiLUBlockKernel
            and stage1.spec.kernel_type is PostNormSiLULayerNormKernel
        ):
            return ProgramKernelSpec(
                kernel_type=SiLUThenPostNormProgramKernel,
                signature=(
                    stage0.spec.kernel_type,
                    stage0.spec.signature,
                    stage1.spec.kernel_type,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        if (
            stage0.spec.kernel_type is PreNormSiLURMSNormKernel
            and stage1.spec.kernel_type is SiLUBlockKernel
        ):
            return ProgramKernelSpec(
                kernel_type=PreNormRMSNormThenSiLUProgramKernel,
                signature=(
                    stage0.spec.kernel_type,
                    stage0.spec.signature,
                    stage1.spec.kernel_type,
                    stage1.spec.signature,
                ),
                arity=int(relation_slice.arity),
                input_dim=int(expected_dim),
                output_dim=int(expected_dim),
                block_specs=(stage0.spec, stage1.spec),
            )
        return None

    def _match_kernel(
        self,
        relation_slice: RelationSlice,
    ) -> KernelMatch | None:
        module = self.update_modules[relation_slice.relation_index]
        cache_key = (id(module), int(relation_slice.arity))
        cached = self._kernel_match_cache.get(cache_key)
        if cached is not None or cache_key in self._kernel_match_cache:
            return cached
        match = None
        for kernel in self.kernels:
            match = kernel.match(self, relation_slice)
            if match is not None:
                break
        self._kernel_match_cache[cache_key] = match
        return match

    def _get_kernel_layout(
        self,
        topology: FlatTopology,
    ) -> dict[str, tuple[KernelBatchPlan, ...] | tuple[int, ...]]:
        cache_key = _topology_cache_key(topology)
        cached = self._persistent_kernel_layout_cache.get(cache_key)
        if cached is not None:
            return cached

        grouped_exec: dict[
            tuple[int, int, Hashable], tuple[FlatRelationKernel, list[int]]
        ] = {}
        fallback_indices: list[int] = []
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0 or relation_slice.arity <= 0:
                continue
            match = self._match_kernel(relation_slice)
            if match is None:
                fallback_indices.append(relation_slice.relation_index)
                continue
            if match.kernel is None:
                fallback_indices.append(relation_slice.relation_index)
                continue
            group_key = (
                int(id(match.kernel)),
                relation_slice.arity,
                match.spec.signature,
            )
            grouped_kernel, grouped_indices = grouped_exec.setdefault(
                group_key, (match.kernel, [])
            )
            grouped_indices.append(relation_slice.relation_index)

        groups: list[KernelBatchPlan] = []
        for (
            _kernel_id,
            arity,
            signature,
        ), (kernel, relation_indices) in grouped_exec.items():
            group_slices = tuple(
                topology.relation_slices[idx] for idx in relation_indices
            )
            row_sizes = tuple(
                int(relation_slice.count) for relation_slice in group_slices
            )
            groups.append(
                KernelBatchPlan(
                    kernel=kernel,
                    signature=signature,
                    arity=int(arity),
                    relation_indices=tuple(
                        int(idx) for idx in relation_indices
                    ),
                    max_rows=max(row_sizes) if row_sizes else 0,
                    row_sizes=row_sizes,
                )
            )

        layout = {
            "groups": tuple(groups),
            "fallback_indices": tuple(
                sorted(set(int(idx) for idx in fallback_indices))
            ),
        }
        self._persistent_kernel_layout_cache[cache_key] = layout
        return layout

    def _collect_eager_relation_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        relation_slice: RelationSlice,
        *,
        arg_emb_all: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        if relation_slice.count <= 0 or relation_slice.arity <= 0:
            return None
        flat_idx = relation_args[
            relation_slice.slot_start : relation_slice.slot_end
        ]
        module = self.update_modules[relation_slice.relation_index]
        if arg_emb_all is not None:
            arg_emb = arg_emb_all[
                relation_slice.slot_start : relation_slice.slot_end
            ]
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

    def _run_two_layer_pointwise_kernel(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
        expected_kernel_types: tuple[type[FlatRelationKernel], ...],
    ) -> tuple[Tensor, Tensor] | None:
        if not self._use_relation_kernels(x):
            return None
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, KernelMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_kernel(relation_slice)
            if match is None or not isinstance(
                match.kernel,
                expected_kernel_types,
            ):
                return None
            batch_items.append((relation_slice, match))

        pointwise_signature = batch_items[0][1].spec.pointwise_signature
        pointwise_code = relm_mp_ops.activation_code(
            pointwise_signature
        )
        if pointwise_code is None:
            return None
        if any(
            item[1].spec.pointwise_signature != pointwise_signature
            for item in batch_items[1:]
        ):
            return None

        group_key = (
            "block_pointwise",
            type(grouped_batch.kernel),
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
                tensors=[
                    item[1].linears[0].bias
                    for item in batch_items
                    if item[1].linears[0].bias is not None
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b1_stack = w1_stack.new_empty((0,))
        if lin1_has_bias:
            b2_stack = self._get_grouped_param_stack(
                cache_key=("b2", group_key),
                tensors=[
                    item[1].linears[1].bias
                    for item in batch_items
                    if item[1].linears[1].bias is not None
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b2_stack = w2_stack.new_empty((0,))

        return relm_mp_ops.block_pointwise(
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

    def _run_postnorm_layernorm_kernel(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
        expected_kernel_types: tuple[type[FlatRelationKernel], ...],
    ) -> tuple[Tensor, Tensor] | None:
        if not self._use_relation_kernels(x):
            return None
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, KernelMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_kernel(relation_slice)
            if match is None or not isinstance(
                match.kernel,
                expected_kernel_types,
            ):
                return None
            if (
                match.spec.norm_kind != "layernorm"
                or match.spec.norm_position != "post"
            ):
                return None
            if len(match.norm_modules) != 1:
                return None
            batch_items.append((relation_slice, match))

        pointwise_signature = batch_items[0][1].spec.pointwise_signature
        pointwise_code = relm_mp_ops.activation_code(
            pointwise_signature
        )
        if pointwise_code is None:
            return None
        if any(
            item[1].spec.pointwise_signature != pointwise_signature
            for item in batch_items[1:]
        ):
            return None

        norm_signature = batch_items[0][1].spec.signature[-1]
        if norm_signature is None or str(norm_signature[0]) != "layernorm":
            return None
        eps = float(
            norm_signature[2] if norm_signature[2] is not None else 1e-5
        )
        affine = bool(norm_signature[3])
        if any(
            item[1].spec.signature[-1] != norm_signature
            for item in batch_items[1:]
        ):
            return None

        group_key = (
            "block_postnorm_ln",
            type(grouped_batch.kernel),
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
                tensors=[
                    item[1].linears[0].bias
                    for item in batch_items
                    if item[1].linears[0].bias is not None
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b1_stack = w1_stack.new_empty((0,))
        if lin1_has_bias:
            b2_stack = self._get_grouped_param_stack(
                cache_key=("b2", group_key),
                tensors=[
                    item[1].linears[1].bias
                    for item in batch_items
                    if item[1].linears[1].bias is not None
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b2_stack = w2_stack.new_empty((0,))

        if affine:
            ln_weight_stack = self._get_grouped_param_stack(
                cache_key=("ln_w", group_key),
                tensors=[
                    cast(torch.nn.LayerNorm, item[1].norm_modules[0]).weight
                    for item in batch_items
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
            ln_bias_stack = self._get_grouped_param_stack(
                cache_key=("ln_b", group_key),
                tensors=[
                    cast(torch.nn.LayerNorm, item[1].norm_modules[0]).bias
                    for item in batch_items
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            ln_weight_stack = w2_stack.new_empty((0,))
            ln_bias_stack = w2_stack.new_empty((0,))

        return relm_mp_ops.block_postnorm_ln(
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

    def _run_prenorm_rmsnorm_kernel(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if not self._use_relation_kernels(x):
            return None
        relation_indices = grouped_batch.relation_indices
        if not relation_indices:
            return None

        batch_items: list[tuple[RelationSlice, KernelMatch]] = []
        for relation_index in relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_kernel(relation_slice)
            if (
                match is None
                or not isinstance(match.kernel, PreNormSiLURMSNormKernel)
            ):
                return None
            if (
                match.spec.norm_kind != "rmsnorm"
                or match.spec.norm_position != "pre"
            ):
                return None
            if len(match.norm_modules) != 1:
                return None
            batch_items.append((relation_slice, match))

        pointwise_signature = batch_items[0][1].spec.pointwise_signature
        pointwise_code = relm_mp_ops.activation_code(
            pointwise_signature
        )
        if pointwise_code is None:
            return None
        if any(
            item[1].spec.pointwise_signature != pointwise_signature
            for item in batch_items[1:]
        ):
            return None

        norm_signature = batch_items[0][1].spec.signature[-1]
        if norm_signature is None or str(norm_signature[0]) != "rmsnorm":
            return None
        eps = float(
            norm_signature[2] if norm_signature[2] is not None else 1e-5
        )
        affine = bool(norm_signature[3])
        if any(
            item[1].spec.signature[-1] != norm_signature
            for item in batch_items[1:]
        ):
            return None

        group_key = (
            "block_prenorm_rms",
            type(grouped_batch.kernel),
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
                tensors=[
                    item[1].linears[0].bias
                    for item in batch_items
                    if item[1].linears[0].bias is not None
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b1_stack = w1_stack.new_empty((0,))
        if lin1_has_bias:
            b2_stack = self._get_grouped_param_stack(
                cache_key=("b2", group_key),
                tensors=[
                    item[1].linears[1].bias
                    for item in batch_items
                    if item[1].linears[1].bias is not None
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            b2_stack = w2_stack.new_empty((0,))

        if affine:
            rms_weight_stack = self._get_grouped_param_stack(
                cache_key=("rms_w", group_key),
                tensors=[
                    cast(Any, item[1].norm_modules[0]).weight
                    for item in batch_items
                ],
                forward_cache=grouped_param_stacks,
                allow_persistent=allow_persistent_stacks,
            )
        else:
            rms_weight_stack = w2_stack.new_empty((0,))

        return relm_mp_ops.block_prenorm_rms(
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

    def _collect_program_batch_items(
        self,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        expected_kernel_type: type[FlatRelationKernel],
    ) -> list[tuple[RelationSlice, KernelMatch]] | None:
        batch_items: list[tuple[RelationSlice, KernelMatch]] = []
        for relation_index in grouped_batch.relation_indices:
            relation_slice = topology.relation_slices[relation_index]
            match = self._match_kernel(relation_slice)
            if (
                match is None
                or match.program_spec is None
                or not match.program_matches
                or not isinstance(match.kernel, expected_kernel_type)
            ):
                return None
            batch_items.append((relation_slice, match))
        return batch_items

    def _run_silu_pair_program_kernel(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if not self._use_program_kernels(x):
            return None
        batch_items = self._collect_program_batch_items(
            topology, grouped_batch, SiLUPairProgramKernel
        )
        if not batch_items:
            return None
        row_sizes = [int(item[0].count) for item in batch_items]
        slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
        stage0_matches = [item[1].program_matches[0] for item in batch_items]
        stage1_matches = [item[1].program_matches[1] for item in batch_items]
        program_key = ("program_kernel", type(grouped_batch.kernel), grouped_batch.arity, grouped_batch.signature)
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
        return relm_mp_ops.program_silu_pair(
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

    def _run_silu_then_postnorm_program_kernel(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if not self._use_program_kernels(x):
            return None
        batch_items = self._collect_program_batch_items(
            topology, grouped_batch, SiLUThenPostNormProgramKernel
        )
        if not batch_items:
            return None
        row_sizes = [int(item[0].count) for item in batch_items]
        slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
        stage0_matches = [item[1].program_matches[0] for item in batch_items]
        stage1_matches = [item[1].program_matches[1] for item in batch_items]
        program_key = ("program_kernel", type(grouped_batch.kernel), grouped_batch.arity, grouped_batch.signature)
        norm_signature = stage1_matches[0].spec.signature[-1]
        if norm_signature is None or str(norm_signature[0]) != "layernorm":
            return None
        if any(match.spec.signature[-1] != norm_signature for match in stage1_matches[1:]):
            return None
        ln_eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
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
        return relm_mp_ops.program_silu_postnorm(
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

    def _run_prenorm_rmsnorm_then_silu_program_kernel(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        *,
        grouped_param_stacks: dict[tuple[Any, ...], Tensor],
        allow_persistent_stacks: bool,
    ) -> tuple[Tensor, Tensor] | None:
        if not self._use_program_kernels(x):
            return None
        batch_items = self._collect_program_batch_items(
            topology, grouped_batch, PreNormRMSNormThenSiLUProgramKernel
        )
        if not batch_items:
            return None
        row_sizes = [int(item[0].count) for item in batch_items]
        slot_offsets_global = [int(item[0].slot_start) for item in batch_items]
        stage0_matches = [item[1].program_matches[0] for item in batch_items]
        stage1_matches = [item[1].program_matches[1] for item in batch_items]
        program_key = ("program_kernel", type(grouped_batch.kernel), grouped_batch.arity, grouped_batch.signature)
        norm_signature = stage0_matches[0].spec.signature[-1]
        if norm_signature is None or str(norm_signature[0]) != "rmsnorm":
            return None
        if any(match.spec.signature[-1] != norm_signature for match in stage0_matches[1:]):
            return None
        rms_eps = float(norm_signature[2] if norm_signature[2] is not None else 1e-5)
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
        return relm_mp_ops.program_rmsnorm_silu(
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

    def _collect_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> tuple[Tensor, Tensor] | None:
        slot_messages = self.collect_slot_messages(
            x, relation_args, topology, cache=cache
        )
        if slot_messages is None:
            return None
        return slot_messages, relation_args

    def collect_slot_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> Tensor | None:
        """Materialize packed relation-slot messages in canonical slot order.

        The returned tensor has shape ``[num_slots, embedding_size]`` and aligns
        exactly with the packed slot order implied by ``relation_args`` and
        ``topology``. This is the reusable boundary for downstream models such as
        flat LGAN, which need slot-local messages before any entity aggregation.
        """
        centralized_spec = self._centralized_batch_spec()
        if centralized_spec is not None:
            centralized_arg_emb_all = (
                x.index_select(0, relation_args)
                if int(relation_args.numel()) > 0
                else None
            )
            centralized = self._collect_centralized_relation_messages(
                x,
                relation_args,
                topology,
                spec=centralized_spec,
                arg_emb_all=centralized_arg_emb_all,
            )
            if centralized is None:
                return None
            msgs, _ = centralized
            return msgs
        if int(relation_args.numel()) == 0:
            return None
        slot_messages = x.new_zeros((int(relation_args.numel()), self.embedding_size))
        use_any_kernels = self._use_relation_kernels(
            x
        ) or self._use_program_kernels(x)
        arg_emb_all = (
            x.index_select(0, relation_args)
            if (not use_any_kernels)
            and self._use_relation_gather(x)
            and int(relation_args.numel()) > 0
            else None
        )

        if use_any_kernels:
            layout = self._get_kernel_layout(topology)
            grouped_param_stacks = (
                cache.setdefault("kernel_param_stacks", {})
                if cache is not None
                else {}
            )
            allow_persistent_stacks = (not self.training) and (
                not torch.is_grad_enabled()
            )
            consumed: set[int] = set()
            fallback_arg_emb_all = (
                x.index_select(0, relation_args)
                if layout["fallback_indices"]
                and self._use_relation_gather(x)
                and int(relation_args.numel()) > 0
                else None
            )
            for grouped_batch in layout["groups"]:
                grouped = grouped_batch.kernel.collect(
                    self,
                    x,
                    relation_args,
                    topology,
                    grouped_batch,
                    grouped_param_stacks=grouped_param_stacks,
                    allow_persistent_stacks=allow_persistent_stacks,
                )
                if grouped is None:
                    continue
                msgs, _ = grouped
                msg_cursor = 0
                for relation_index, row_count in zip(
                    grouped_batch.relation_indices,
                    grouped_batch.row_sizes,
                    strict=True,
                ):
                    relation_slice = topology.relation_slices[relation_index]
                    width = int(row_count) * int(grouped_batch.arity)
                    slot_messages[
                        relation_slice.slot_start : relation_slice.slot_end
                    ] = msgs[msg_cursor : msg_cursor + width]
                    msg_cursor += width
                consumed.update(
                    int(idx_i) for idx_i in grouped_batch.relation_indices
                )
            for relation_index in layout["fallback_indices"]:
                relation_slice = topology.relation_slices[relation_index]
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                slot_messages[
                    relation_slice.slot_start : relation_slice.slot_end
                ] = msgs
                consumed.add(relation_index)
            for relation_slice in topology.relation_slices:
                if relation_slice.relation_index in consumed:
                    continue
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                slot_messages[
                    relation_slice.slot_start : relation_slice.slot_end
                ] = msgs
        else:
            for relation_slice in topology.relation_slices:
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                slot_messages[
                    relation_slice.slot_start : relation_slice.slot_end
                ] = msgs
        return slot_messages

    def _pool_grouped_kernel_messages(
        self,
        topology: FlatTopology,
        grouped_batch: KernelBatchPlan,
        relation_row_starts: dict[int, int],
        messages: Tensor,
        *,
        device: torch.device,
        index_dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Pool one grouped exact-kernel batch to relation-instance rows.

        Returns ``(pooled_rows, row_indices)`` where ``pooled_rows`` has one row per
        grounded relation instance in the grouped batch and ``row_indices`` maps those
        rows back into the global relation-instance table.
        """
        if int(messages.numel()) == 0:
            return (
                messages.new_zeros((0, self.embedding_size)),
                torch.empty((0,), device=device, dtype=index_dtype),
            )
        pooled_parts: list[Tensor] = []
        row_indices_parts: list[Tensor] = []
        msg_cursor = 0
        arity = int(grouped_batch.arity)
        for relation_index, row_count in zip(
            grouped_batch.relation_indices,
            grouped_batch.row_sizes,
            strict=True,
        ):
            row_count_i = int(row_count)
            if row_count_i <= 0:
                continue
            width = row_count_i * arity
            relation_msgs = messages[msg_cursor : msg_cursor + width]
            pooled_parts.append(
                relation_msgs.view(row_count_i, arity, self.embedding_size).mean(dim=1)
            )
            row_indices_parts.append(
                torch.arange(
                    relation_row_starts[int(relation_index)],
                    relation_row_starts[int(relation_index)] + row_count_i,
                    device=device,
                    dtype=index_dtype,
                )
            )
            msg_cursor += width
        if not pooled_parts:
            return (
                messages.new_zeros((0, self.embedding_size)),
                torch.empty((0,), device=device, dtype=index_dtype),
            )
        return torch.cat(pooled_parts, dim=0), torch.cat(row_indices_parts, dim=0)

    def _collect_relation_instance_messages(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        cache: dict | None = None,
    ) -> Tensor | None:
        """Collect one pooled embedding per grounded relation instance.

        The returned tensor has shape ``[num_relation_instances, embedding_size]`` in
        the canonical relation-instance order implied by ``topology``. This is the
        phase boundary needed by flat LGAN and similar relation-graph models.
        """
        relation_instance_count = int(sum(int(s.count) for s in topology.relation_slices))
        if relation_instance_count == 0:
            return None
        relation_row_starts: dict[int, int] = {}
        row_cursor = 0
        for relation_slice in topology.relation_slices:
            relation_row_starts[int(relation_slice.relation_index)] = row_cursor
            row_cursor += int(relation_slice.count)

        centralized_spec = self._centralized_batch_spec()
        use_any_kernels = self._use_relation_kernels(x) or self._use_program_kernels(x)
        if centralized_spec is None and not use_any_kernels:
            arg_emb_all = (
                x.index_select(0, relation_args)
                if self._use_relation_gather(x) and int(relation_args.numel()) > 0
                else None
            )
            relation_pair_x = x.new_zeros((relation_instance_count, self.embedding_size))
            for relation_slice in topology.relation_slices:
                if relation_slice.count <= 0:
                    continue
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                pooled = msgs.view(
                    relation_slice.count,
                    relation_slice.arity,
                    self.embedding_size,
                ).mean(dim=1)
                row_start = relation_row_starts[int(relation_slice.relation_index)]
                relation_pair_x[row_start : row_start + relation_slice.count] = pooled
            return relation_pair_x

        if centralized_spec is None and use_any_kernels:
            relation_pair_x = x.new_zeros((relation_instance_count, self.embedding_size))
            layout = self._get_kernel_layout(topology)
            grouped_param_stacks = (
                cache.setdefault("kernel_param_stacks", {})
                if cache is not None
                else {}
            )
            allow_persistent_stacks = (not self.training) and (
                not torch.is_grad_enabled()
            )
            consumed: set[int] = set()
            fallback_arg_emb_all = (
                x.index_select(0, relation_args)
                if layout["fallback_indices"]
                and self._use_relation_gather(x)
                and int(relation_args.numel()) > 0
                else None
            )

            def _pool_eager_messages(relation_slice: RelationSlice, messages: Tensor) -> None:
                if relation_slice.count <= 0:
                    return
                pooled = messages.view(
                    relation_slice.count,
                    relation_slice.arity,
                    self.embedding_size,
                ).mean(dim=1)
                row_start = relation_row_starts[int(relation_slice.relation_index)]
                relation_pair_x[row_start : row_start + relation_slice.count] = pooled

            for grouped_batch in layout["groups"]:
                pooled = grouped_batch.kernel.collect_relation_instances(
                    self,
                    x,
                    relation_args,
                    topology,
                    grouped_batch,
                    relation_row_starts=relation_row_starts,
                    grouped_param_stacks=grouped_param_stacks,
                    allow_persistent_stacks=allow_persistent_stacks,
                )
                if pooled is None:
                    continue
                pooled_rows, row_indices = pooled
                if int(row_indices.numel()) > 0:
                    relation_pair_x.index_copy_(0, row_indices, pooled_rows)
                consumed.update(int(idx_i) for idx_i in grouped_batch.relation_indices)

            for relation_index in layout["fallback_indices"]:
                relation_slice = topology.relation_slices[relation_index]
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                _pool_eager_messages(relation_slice, msgs)
                consumed.add(relation_index)

            for relation_slice in topology.relation_slices:
                if relation_slice.relation_index in consumed:
                    continue
                direct = self._collect_eager_relation_messages(
                    x,
                    relation_args,
                    relation_slice,
                    arg_emb_all=fallback_arg_emb_all,
                )
                if direct is None:
                    continue
                msgs, _ = direct
                _pool_eager_messages(relation_slice, msgs)
            return relation_pair_x

        slot_messages = self.collect_slot_messages(
            x, relation_args, topology, cache=cache
        )
        if slot_messages is None:
            return None
        relation_pair_x = x.new_zeros((relation_instance_count, self.embedding_size))
        for relation_slice in topology.relation_slices:
            if relation_slice.count <= 0:
                continue
            rel_slots = slot_messages[
                relation_slice.slot_start : relation_slice.slot_end
            ].view(relation_slice.count, relation_slice.arity, self.embedding_size)
            row_start = relation_row_starts[int(relation_slice.relation_index)]
            relation_pair_x[row_start : row_start + relation_slice.count] = rel_slots.mean(dim=1)
        return relation_pair_x

    def _run_lgan_pointwise_step(
        self,
        x: Tensor,
        relation_args: Tensor,
        topology: FlatTopology,
        *,
        rr_src: Tensor,
        rr_dst: Tensor,
        tn_rel: Tensor,
        tn_ent: Tensor,
        nn_rel: Tensor,
        nn_ent: Tensor,
        entity_dim_size: int,
        mode: str,
        cache: dict | None = None,
    ) -> tuple[Tensor, Tensor, Tensor] | None:
        """Run the integrated exact pointwise LGAN path when supported.

        This path currently targets grouped exact two-layer pointwise block kernels.
        Unsupported grouped kernels and all fallback relations are materialized
        eagerly into the seeded relation-instance table before the integrated LGAN
        graph propagation step runs.
        """
        if not self._use_relation_kernels(x):
            return None
        relation_instance_count = int(sum(int(s.count) for s in topology.relation_slices))
        if relation_instance_count == 0:
            return x.new_zeros((0, self.embedding_size)), x.new_zeros((int(entity_dim_size), self.embedding_size)), x.new_zeros((int(entity_dim_size), self.embedding_size))

        relation_row_starts: dict[int, int] = {}
        row_cursor = 0
        for relation_slice in topology.relation_slices:
            relation_row_starts[int(relation_slice.relation_index)] = row_cursor
            row_cursor += int(relation_slice.count)

        relation_pair_seed = x.new_zeros((relation_instance_count, self.embedding_size))
        layout = self._get_kernel_layout(topology)
        grouped_param_stacks = (
            cache.setdefault("kernel_param_stacks", {})
            if cache is not None
            else {}
        )
        allow_persistent_stacks = (not self.training) and (not torch.is_grad_enabled())
        fallback_arg_emb_all = (
            x.index_select(0, relation_args)
            if (layout["fallback_indices"] or layout["groups"])
            and self._use_relation_gather(x)
            and int(relation_args.numel()) > 0
            else None
        )

        pointwise_groups: list[KernelBatchPlan] = []
        pointwise_codes: list[int] = []
        pointwise_row_indices: list[Tensor] = []
        w1_stacks: list[Tensor] = []
        b1_stacks: list[Tensor] = []
        w2_stacks: list[Tensor] = []
        b2_stacks: list[Tensor] = []
        slot_offsets_groups: list[list[int]] = []
        row_sizes_groups: list[list[int]] = []
        consumed: set[int] = set()

        def _pool_eager_messages(relation_slice: RelationSlice, messages: Tensor) -> None:
            if relation_slice.count <= 0:
                return
            pooled = messages.view(
                relation_slice.count,
                relation_slice.arity,
                self.embedding_size,
            ).mean(dim=1)
            row_start = relation_row_starts[int(relation_slice.relation_index)]
            relation_pair_seed[row_start : row_start + relation_slice.count] = pooled

        exact_kernel_types = (MishBlockKernel, SiLUBlockKernel, GELUBlockKernel)
        for grouped_batch in layout["groups"]:
            if not isinstance(grouped_batch.kernel, exact_kernel_types):
                for relation_index in grouped_batch.relation_indices:
                    relation_slice = topology.relation_slices[relation_index]
                    direct = self._collect_eager_relation_messages(
                        x,
                        relation_args,
                        relation_slice,
                        arg_emb_all=fallback_arg_emb_all,
                    )
                    if direct is None:
                        continue
                    msgs, _ = direct
                    _pool_eager_messages(relation_slice, msgs)
                    consumed.add(relation_index)
                continue

            batch_items: list[tuple[RelationSlice, KernelMatch]] = []
            valid_group = True
            for relation_index in grouped_batch.relation_indices:
                relation_slice = topology.relation_slices[relation_index]
                match = self._match_kernel(relation_slice)
                if match is None or not isinstance(match.kernel, exact_kernel_types):
                    valid_group = False
                    break
                batch_items.append((relation_slice, match))
            if not valid_group or not batch_items:
                for relation_index in grouped_batch.relation_indices:
                    relation_slice = topology.relation_slices[relation_index]
                    direct = self._collect_eager_relation_messages(
                        x,
                        relation_args,
                        relation_slice,
                        arg_emb_all=fallback_arg_emb_all,
                    )
                    if direct is None:
                        continue
                    msgs, _ = direct
                    _pool_eager_messages(relation_slice, msgs)
                    consumed.add(relation_index)
                continue

            pointwise_signature = batch_items[0][1].spec.pointwise_signature
            pointwise_code = relm_mp_ops.activation_code(pointwise_signature)
            if pointwise_code is None or any(
                item[1].spec.pointwise_signature != pointwise_signature
                for item in batch_items[1:]
            ):
                for relation_slice, _ in batch_items:
                    direct = self._collect_eager_relation_messages(
                        x,
                        relation_args,
                        relation_slice,
                        arg_emb_all=fallback_arg_emb_all,
                    )
                    if direct is None:
                        continue
                    msgs, _ = direct
                    _pool_eager_messages(relation_slice, msgs)
                    consumed.add(relation_slice.relation_index)
                continue

            group_key = (
                "lgan_pointwise_step",
                type(grouped_batch.kernel),
                grouped_batch.arity,
                grouped_batch.signature,
            )
            w1_stacks.append(
                self._get_grouped_param_stack(
                    cache_key=("w1", group_key),
                    tensors=[item[1].linears[0].weight for item in batch_items],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            )
            w2_stacks.append(
                self._get_grouped_param_stack(
                    cache_key=("w2", group_key),
                    tensors=[item[1].linears[1].weight for item in batch_items],
                    forward_cache=grouped_param_stacks,
                    allow_persistent=allow_persistent_stacks,
                )
            )
            lin0_has_bias = batch_items[0][1].linears[0].bias is not None
            lin1_has_bias = batch_items[0][1].linears[1].bias is not None
            if lin0_has_bias:
                b1_stacks.append(
                    self._get_grouped_param_stack(
                        cache_key=("b1", group_key),
                        tensors=[
                            item[1].linears[0].bias
                            for item in batch_items
                            if item[1].linears[0].bias is not None
                        ],
                        forward_cache=grouped_param_stacks,
                        allow_persistent=allow_persistent_stacks,
                    )
                )
            else:
                b1_stacks.append(w1_stacks[-1].new_empty((0,)))
            if lin1_has_bias:
                b2_stacks.append(
                    self._get_grouped_param_stack(
                        cache_key=("b2", group_key),
                        tensors=[
                            item[1].linears[1].bias
                            for item in batch_items
                            if item[1].linears[1].bias is not None
                        ],
                        forward_cache=grouped_param_stacks,
                        allow_persistent=allow_persistent_stacks,
                    )
                )
            else:
                b2_stacks.append(w2_stacks[-1].new_empty((0,)))

            pointwise_groups.append(grouped_batch)
            pointwise_codes.append(int(pointwise_code))
            slot_offsets_groups.append([int(item[0].slot_start) for item in batch_items])
            row_sizes_groups.append([int(item[0].count) for item in batch_items])
            row_indices_parts: list[Tensor] = []
            for relation_slice, _ in batch_items:
                row_start = relation_row_starts[int(relation_slice.relation_index)]
                row_indices_parts.append(
                    torch.arange(
                        row_start,
                        row_start + int(relation_slice.count),
                        device=x.device,
                        dtype=relation_args.dtype,
                    )
                )
            pointwise_row_indices.append(
                row_indices_parts[0]
                if len(row_indices_parts) == 1
                else torch.cat(row_indices_parts, dim=0)
            )
            consumed.update(int(idx_i) for idx_i in grouped_batch.relation_indices)

        for relation_index in layout["fallback_indices"]:
            relation_slice = topology.relation_slices[relation_index]
            direct = self._collect_eager_relation_messages(
                x,
                relation_args,
                relation_slice,
                arg_emb_all=fallback_arg_emb_all,
            )
            if direct is None:
                continue
            msgs, _ = direct
            _pool_eager_messages(relation_slice, msgs)
            consumed.add(relation_index)

        for relation_slice in topology.relation_slices:
            if relation_slice.relation_index in consumed:
                continue
            direct = self._collect_eager_relation_messages(
                x,
                relation_args,
                relation_slice,
                arg_emb_all=fallback_arg_emb_all,
            )
            if direct is None:
                continue
            msgs, _ = direct
            _pool_eager_messages(relation_slice, msgs)

        if not pointwise_groups:
            return None

        return relm_mp_ops._lgan_build_pointwise_step(
            x,
            relation_args,
            relation_pair_seed,
            rr_src,
            rr_dst,
            tn_rel,
            tn_ent,
            nn_rel,
            nn_ent,
            entity_dim_size=int(entity_dim_size),
            mode=str(mode),
            arities=tuple(int(group.arity) for group in pointwise_groups),
            pointwise_codes=tuple(pointwise_codes),
            slot_offsets_groups=tuple(tuple(values) for values in slot_offsets_groups),
            row_sizes_groups=tuple(tuple(values) for values in row_sizes_groups),
            row_indices_groups=tuple(pointwise_row_indices),
            w1_stacks=tuple(w1_stacks),
            b1_stacks=tuple(b1_stacks),
            w2_stacks=tuple(w2_stacks),
            b2_stacks=tuple(b2_stacks),
        )

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

    def _aggregate_messages(
        self,
        *,
        x: Tensor,
        collected: tuple[Tensor, Tensor] | None,
    ) -> Tensor:
        """Reduce packed relation messages back into the entity table."""
        aggregated = x.new_zeros((int(x.size(0)), self.embedding_size))
        if collected is None:
            return aggregated
        msgs, idx = collected
        return self.aggr(x=msgs, index=idx, dim=0, dim_size=int(x.size(0)))

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

        collected = self._collect_messages(
            x, relation_args, topology, cache=cache
        )
        return self._aggregate_messages(x=x, collected=collected)
