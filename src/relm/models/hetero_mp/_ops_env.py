from __future__ import annotations

import os

import torch
from torch_geometric.nn import Aggregation
from torch_geometric.nn.aggr import SumAggregation

from ..aggr import LogSumExpAggregation

try:  # pragma: no cover - optional during minimal model-only imports
    from ...ops import mp as relm_mp_ops
except Exception:  # pragma: no cover
    relm_mp_ops = None  # type: ignore[assignment]

_MODE_SUM = 0
_MODE_LOGSUMEXP = 1
_BOOL_FALSE = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _BOOL_FALSE


def _use_grouped_relation_mlp(ref: torch.Tensor) -> bool:
    # Optional, CUDA-focused fast path that vectorizes compatible per-relation MLPs.
    return ref.device.type == "cuda" and _env_bool("RELM_MODELS_MP_GROUPED_MLP", True)


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
