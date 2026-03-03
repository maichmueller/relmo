from __future__ import annotations

from typing import Any

import torch

from ..grouped_mlp import GroupedMLPSpec
from ..residual import ResidualModule

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


def _extract_grouped_residual_mlp_info(
    module: torch.nn.Module,
) -> dict[str, Any] | None:
    # Explicit user interface takes precedence over auto-detection.
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
        if "linears" not in spec or "ops" not in spec:
            raise KeyError(
                f"{type(module).__name__}.{method_name}() must define 'linears' and 'ops'."
            )
        linears = tuple(spec["linears"])
        if not linears:
            raise ValueError(
                f"{type(module).__name__}.{method_name}() returned empty 'linears'."
            )
        for idx, lin in enumerate(linears):
            if not isinstance(lin, torch.nn.Linear):
                raise TypeError(
                    f"{type(module).__name__}.{method_name}() linears[{idx}] is {type(lin)!r}, expected torch.nn.Linear."
                )

        ops: list[tuple[str, Any]] = []
        sig_ops: list[tuple[str, Any]] = []
        for idx, op in enumerate(spec["ops"]):
            if not (isinstance(op, tuple) and len(op) == 2):
                raise TypeError(
                    f"{type(module).__name__}.{method_name}() ops[{idx}] must be tuple(kind, payload)."
                )
            kind, payload = op
            if kind == "linear":
                lin_idx = int(payload)
                if lin_idx < 0 or lin_idx >= len(linears):
                    raise IndexError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] references linear index {lin_idx}, "
                        f"but only {len(linears)} linears are available."
                    )
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
                continue
            if kind == "pointwise":
                if not isinstance(payload, torch.nn.Module):
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] pointwise payload must be torch.nn.Module."
                    )
                pointwise_sig = _pointwise_signature(payload)
                if pointwise_sig is None:
                    raise TypeError(
                        f"{type(module).__name__}.{method_name}() ops[{idx}] uses unsupported pointwise module {type(payload)!r}."
                    )
                ops.append(("pointwise", payload))
                sig_ops.append(("pointwise", pointwise_sig))
                continue
            raise ValueError(
                f"{type(module).__name__}.{method_name}() ops[{idx}] has invalid kind {kind!r}; use 'linear' or 'pointwise'."
            )

        truncated_dim = spec.get("truncated_dim", None)
        if truncated_dim is not None:
            truncated_dim = int(truncated_dim)
        truncate_right = spec.get("truncate_right", None)
        if truncate_right is not None:
            truncate_right = bool(truncate_right)

        signature = spec.get("signature", None)
        if signature is None:
            signature = (tuple(sig_ops), truncated_dim, truncate_right)
        if isinstance(signature, list):
            signature = tuple(signature)
        try:
            hash(signature)
        except TypeError as exc:
            raise TypeError(
                f"{type(module).__name__}.{method_name}() returned non-hashable signature."
            ) from exc

        return {
            "signature": signature,
            "linears": linears,
            "ops": tuple(ops),
            "truncated_dim": truncated_dim,
            "truncate_right": truncate_right,
        }

    # Supports only simple ResidualModule(module=Sequential[Linear, pointwise, ...]).
    # All other module types use the existing per-predicate execution path.
    if not isinstance(module, ResidualModule):
        return None
    inner = getattr(module, "module", None)
    net = getattr(inner, "net", None)
    if not isinstance(net, torch.nn.Sequential):
        return None

    ops: list[tuple[str, Any]] = []
    sig_ops: list[tuple[str, Any]] = []
    linears: list[torch.nn.Linear] = []
    for sub in net:
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
        if pointwise_sig is None:
            return None
        ops.append(("pointwise", sub))
        sig_ops.append(("pointwise", pointwise_sig))

    if not linears:
        return None

    signature = (
        tuple(sig_ops),
        int(module.truncated_dim) if module.truncated_dim is not None else None,
        None if module.truncate_right is None else bool(module.truncate_right),
    )
    return {
        "signature": signature,
        "linears": tuple(linears),
        "ops": tuple(ops),
        "truncated_dim": module.truncated_dim,
        "truncate_right": module.truncate_right,
    }


def _apply_residual_truncate(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    truncated_dim: int | None,
    truncate_right: bool | None,
) -> torch.Tensor:
    if truncated_dim is None or int(x.size(-1)) == int(truncated_dim):
        return x + y
    if truncate_right:
        return x[..., : int(truncated_dim)] + y
    return x[..., -int(truncated_dim) :] + y
