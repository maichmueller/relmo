"""Dispatch policy between custom ops and Python fallbacks."""

from __future__ import annotations

from .mp_runtime import (
    ensure_loaded,
    ensure_runtime_compat_once,
    env_bool_any,
    fallback_mode,
    ops_namespace,
    torch,
)


def should_use_custom(op_name: str) -> bool:
    """Return whether the current policy allows dispatching to a custom op."""

    if not env_bool_any(("RELM_MP_ENABLE",), True):
        return False
    try:
        ensure_runtime_compat_once()
        return True
    except Exception as exc:
        if fallback_mode() == "error":
            raise RuntimeError(f"Custom mp op {op_name} is unavailable.") from exc
        return False


def namespace_has_op(op_name: str) -> bool:
    if torch is None:
        return False
    try:
        ensure_loaded()
    except Exception:
        return False
    return hasattr(ops_namespace(), op_name)


def require_available_custom_op(op_name: str) -> None:
    if fallback_mode() == "error":
        raise RuntimeError(
            f"Custom mp op {op_name} is unavailable in the loaded relm_mp library."
        )


def use_custom_namespace_op(
    op_name: str,
    *,
    tensor: object | None = None,
    require_cuda: bool = False,
) -> bool:
    """Return whether a direct namespace call should be used for this operation."""

    if require_cuda and (tensor is None or not bool(getattr(tensor, "is_cuda", False))):
        return False
    return should_use_custom(op_name) and namespace_has_op(op_name)


def dispatch_namespace_op(
    op_name: str,
    *args,
    fallback,
    tensor: object | None = None,
    require_cuda: bool = False,
    enforce_fallback_policy: bool = True,
):
    """Apply the standard direct custom-op vs fallback dispatch pattern."""

    if use_custom_namespace_op(
        op_name,
        tensor=tensor,
        require_cuda=require_cuda,
    ):
        return getattr(ops_namespace(), op_name)(*args)

    if enforce_fallback_policy and should_use_custom(op_name):
        require_available_custom_op(op_name)
    return fallback(*args)
