"""Internal flat relational implementation package.

User-facing flat model entrypoints live in ``relmo.models.flat`` and
``relmo.models.builders``. Kernel experimentation notes are preserved in
``docs/flat_lgan_plan.md`` and ``docs/manual_fused_program_kernel_plan.md``.
"""

from . import collection, execution, flat_contract, kernels, matching, topology, types

__all__ = [
    "collection",
    "execution",
    "flat_contract",
    "kernels",
    "matching",
    "topology",
    "types",
]
