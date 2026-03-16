"""Constants shared by message-passing runtime, fallbacks, and dispatch."""

from __future__ import annotations

from typing import Final

MODE_SUM: Final[int] = 0
MODE_LOGSUMEXP: Final[int] = 1
MODE_MEAN: Final[int] = 2

PW_IDENTITY: Final[int] = 0
PW_RELU: Final[int] = 1
PW_MISH: Final[int] = 2
PW_GELU_NONE: Final[int] = 3
PW_GELU_TANH: Final[int] = 4
PW_SILU: Final[int] = 5
PW_TANH: Final[int] = 6

REQUIRED_NAMESPACE_OPS: Final[tuple[str, ...]] = (
    "fanout_scatter",
    "fanout_scatter_backward",
    "fanin_reduce",
    "fanin_reduce_sum_backward",
    "fanin_reduce_logsumexp_backward",
    "build_info",
)

CUSTOM_TWO_LAYER_POINTWISE_CODES: Final[set[int]] = {
    PW_MISH,
    PW_GELU_NONE,
    PW_GELU_TANH,
    PW_SILU,
}


def activation_code(signature: tuple[object, ...] | None) -> int | None:
    """Map a structured activation signature to the custom-op enum code."""

    if not signature:
        return None

    kind = str(signature[0])
    if kind == "identity":
        return PW_IDENTITY
    if kind == "relu":
        return PW_RELU
    if kind == "mish":
        return PW_MISH
    if kind == "gelu":
        approximate = str(signature[1]) if len(signature) > 1 else "none"
        if approximate == "none":
            return PW_GELU_NONE
        if approximate == "tanh":
            return PW_GELU_TANH
        return None
    if kind == "silu":
        return PW_SILU
    if kind == "tanh":
        return PW_TANH
    return None
