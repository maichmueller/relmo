"""Experimental flat-relational runtime helpers.

These modules are not part of the stable user-facing flat model API. They are
kept for benchmark history, kernel experiments, and compatibility with older
internal imports.
"""

from .kernel_runtime import FlatKernelRuntime, KernelExecutionContext, KernelExecutionLayout

__all__ = [
    "FlatKernelRuntime",
    "KernelExecutionContext",
    "KernelExecutionLayout",
]
