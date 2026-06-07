"""Compatibility wrapper for experimental flat kernel runtime helpers."""

from __future__ import annotations

from warnings import warn

from .experimental.kernel_runtime import (
    FlatKernelRuntime,
    KernelExecutionContext,
    KernelExecutionLayout,
)

warn(
    "relmo.models.flat_relational.flat_kernel_runtime is experimental; "
    "use relmo.models.flat_relational.experimental.kernel_runtime for new code.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "FlatKernelRuntime",
    "KernelExecutionContext",
    "KernelExecutionLayout",
]
