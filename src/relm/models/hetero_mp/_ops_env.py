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
_PRESET_ENV = "RELM_MODELS_MP_PRESET"

_BASELINE_FLAGS: dict[str, bool] = {
    # Global model-side MP integration gates.
    "RELM_MODELS_MP_OPS": True,
    "RELM_MODELS_MP_FANIN": True,
    "RELM_MODELS_MP_FANOUT": False,
    "RELM_MODELS_MP_LOGSUMEXP": False,
    # Generic relation MLP grouping path.
    "RELM_MODELS_MP_GROUPED_MLP": True,
    # Central fused custom fanin reduction.
    "RELM_MODELS_MP_FANIN_FUSED": True,
    # Decentralized batched experimental lanes.
    "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": False,
    "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": False,
    "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": False,
}

_PRESET_FLAGS: dict[str, dict[str, bool]] = {
    # Historical behavior before introducing presets.
    "baseline": dict(_BASELINE_FLAGS),
    # Training-oriented defaults: keep grouped relation MLP enabled and
    # avoid pack-only fanin lane, which raised backward cost on real PDDL runs.
    "tuned_train": {
        **_BASELINE_FLAGS,
    },
    # Inference-oriented defaults: reduce Python-side relation routing overhead.
    "tuned_infer": {
        **_BASELINE_FLAGS,
        "RELM_MODELS_MP_GROUPED_MLP": False,
        "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": True,
    },
    # Explicit full decentralized C++ experiment lane.
    "full_cpp_exp": {
        **_BASELINE_FLAGS,
        "RELM_MODELS_MP_GROUPED_MLP": False,
        "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL": True,
        "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL": True,
        "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL": True,
    },
}

_PRESET_ALIASES = {
    "default": "tuned_train",
    "train": "tuned_train",
    "training": "tuned_train",
    "infer": "tuned_infer",
    "inference": "tuned_infer",
    "full_cpp": "full_cpp_exp",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in _BOOL_FALSE


def _mp_preset_name() -> str:
    raw = os.getenv(_PRESET_ENV, "tuned_train")
    key = raw.strip().lower()
    key = _PRESET_ALIASES.get(key, key)
    return key if key in _PRESET_FLAGS else "tuned_train"


def _mp_flag(name: str, legacy_default: bool) -> bool:
    preset = _PRESET_FLAGS.get(_mp_preset_name(), _PRESET_FLAGS["tuned_train"])
    default = preset.get(name, legacy_default)
    return _env_bool(name, default)


def _use_grouped_relation_mlp(ref: torch.Tensor) -> bool:
    return ref.device.type == "cuda" and _mp_flag("RELM_MODELS_MP_GROUPED_MLP", True)


def _use_model_mp_ops(ref: torch.Tensor) -> bool:
    if relm_mp_ops is None:
        return False
    if not _mp_flag("RELM_MODELS_MP_OPS", True):
        return False
    # Model-side custom op integration is tuned for CUDA execution.
    if ref.device.type != "cuda":
        return False
    return ref.dtype.is_floating_point


def _use_model_mp_fanin(ref: torch.Tensor) -> bool:
    return _use_model_mp_ops(ref) and _mp_flag("RELM_MODELS_MP_FANIN", True)


def _use_model_mp_fanout(ref: torch.Tensor) -> bool:
    return _use_model_mp_ops(ref) and _mp_flag("RELM_MODELS_MP_FANOUT", False)


def _use_model_mp_fanin_fused(ref: torch.Tensor) -> bool:
    return _use_model_mp_fanin(ref) and _mp_flag("RELM_MODELS_MP_FANIN_FUSED", True)


def _use_model_mp_batched_fanout(ref: torch.Tensor) -> bool:
    return _use_model_mp_ops(ref) and _mp_flag(
        "RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL", False
    )


def _use_model_mp_batched_fanin_reduce(ref: torch.Tensor) -> bool:
    return _use_model_mp_fanin(ref) and _mp_flag(
        "RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL", False
    )


def _use_model_mp_batched_fanin_pack(ref: torch.Tensor) -> bool:
    return _use_model_mp_ops(ref) and _mp_flag(
        "RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL", False
    )


def _resolve_fanin_mode(aggr: Aggregation) -> int | None:
    if isinstance(aggr, SumAggregation):
        return _MODE_SUM
    # Optional: enable custom logsumexp reduce for model MPs when desired.
    if isinstance(aggr, LogSumExpAggregation) and _mp_flag(
        "RELM_MODELS_MP_LOGSUMEXP", False
    ):
        return _MODE_LOGSUMEXP
    return None
