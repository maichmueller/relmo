from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Protocol, Sequence, TypeAlias

import torch

GroupedMLPOpPayload: TypeAlias = int | torch.nn.Module
GroupedMLPOp: TypeAlias = tuple[str, GroupedMLPOpPayload]


@dataclass(frozen=True)
class GroupedMLPSpec:
    """
    Declarative spec for grouped per-relation MLP execution in BatchedFanOutMP.

    Attributes:
        linears:
            Linear layers referenced by ("linear", idx) operations.
        ops:
            Ordered operations. Each item must be:
            - ("linear", int-index-into-linears)
            - ("pointwise", torch.nn.Module)
            - ("norm", torch.nn.Module)
        truncated_dim:
            Optional residual truncation dim (same semantics as ResidualModule).
        truncate_right:
            Optional residual truncation direction.
        signature:
            Optional custom grouping key. If omitted, relm derives one from ops + truncation.
    """

    linears: Sequence[torch.nn.Linear]
    ops: Sequence[GroupedMLPOp]
    truncated_dim: int | None = None
    truncate_right: bool | None = None
    signature: Hashable | None = None


class GroupedMLPCompatible(Protocol):
    """
    Optional user-facing protocol for modules that expose grouped MLP specs.
    """

    def relm_grouped_mlp_spec(self) -> GroupedMLPSpec | dict[str, Any] | None: ...
