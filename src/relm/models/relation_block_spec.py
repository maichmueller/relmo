"""Kernel-spec protocol for relation blocks.

Relation blocks may expose a declarative spec that allows the flat runtime to
match them against exact CUDA kernel families. Modules without a spec remain
fully supported through the eager fallback path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Protocol, Sequence, TypeAlias

import torch

RelationBlockOpPayload: TypeAlias = int | torch.nn.Module
RelationBlockOp: TypeAlias = tuple[str, RelationBlockOpPayload]


@dataclass(frozen=True)
class RelationBlockSpec:
    """Declarative description of a width-preserving relation block.

    Attributes:
        linears:
            Linear layers referenced by ``("linear", idx)`` operations.
        ops:
            Ordered operations. Supported items are:
            ``("linear", int-index-into-linears)``
            ``("pointwise", torch.nn.Module)``
            ``("norm", torch.nn.Module)``
        signature:
            Optional explicit grouping key. If omitted, relm derives one from
            the ordered op structure.
    """

    linears: Sequence[torch.nn.Linear]
    ops: Sequence[RelationBlockOp]
    signature: Hashable | None = None


class RelationBlockCompatible(Protocol):
    """Optional protocol for modules that want kernel-family matching."""

    def relm_kernel_spec(self) -> RelationBlockSpec | dict[str, Any] | None: ...


__all__ = [
    "RelationBlockCompatible",
    "RelationBlockOp",
    "RelationBlockOpPayload",
    "RelationBlockSpec",
]
