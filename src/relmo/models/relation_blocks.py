"""Public typed relation blocks for the flat relational runtime.

These blocks define width-preserving relation transforms over packed relation
rows of shape ``[rows, width]``. They do not apply the outer tuple residual.
The flat layer or exact kernel path is responsible for adding the gathered
input slots exactly once after block/program execution.
"""

from __future__ import annotations

from typing import Iterable, Literal

import torch

from .relation_block_spec import RelationBlockSpec

PointwiseKind = Literal["identity", "relu", "mish", "gelu", "silu", "tanh"]
NormKind = Literal["layernorm", "rmsnorm"]


def _make_pointwise(
    activation: PointwiseKind | torch.nn.Module,
    *,
    gelu_approximate: str = "none",
) -> torch.nn.Module:
    if isinstance(activation, torch.nn.Module):
        return activation
    name = str(activation).strip().lower()
    if name == "identity":
        return torch.nn.Identity()
    if name == "relu":
        return torch.nn.ReLU()
    if name == "mish":
        return torch.nn.Mish()
    if name == "gelu":
        return torch.nn.GELU(approximate=gelu_approximate)
    if name == "silu":
        return torch.nn.SiLU()
    if name == "tanh":
        return torch.nn.Tanh()
    raise ValueError(f"Unsupported pointwise activation: {activation!r}.")


def _make_norm(
    width: int,
    *,
    norm: NormKind,
    eps: float = 1e-5,
    affine: bool = True,
) -> torch.nn.Module:
    norm_name = str(norm).strip().lower()
    if norm_name == "layernorm":
        return torch.nn.LayerNorm(width, eps=eps, elementwise_affine=affine)
    if norm_name == "rmsnorm":
        rmsnorm_cls = getattr(torch.nn, "RMSNorm", None)
        if rmsnorm_cls is None:
            raise RuntimeError("RMSNorm is unavailable in this torch build.")
        return rmsnorm_cls(width, eps=eps, elementwise_affine=affine)
    raise ValueError(f"Unsupported norm kind: {norm!r}.")


class TwoLayerPointwiseRelationMLP(torch.nn.Module):
    """Width-preserving ``Linear -> Pointwise -> Linear`` relation block."""

    def __init__(
        self,
        width: int,
        hidden: int,
        *,
        activation: PointwiseKind | torch.nn.Module = "mish",
        gelu_approximate: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.lin1 = torch.nn.Linear(width, hidden, bias=bias)
        self.act = _make_pointwise(activation, gelu_approximate=gelu_approximate)
        self.lin2 = torch.nn.Linear(hidden, width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

    def relmo_kernel_spec(self) -> RelationBlockSpec:
        return RelationBlockSpec(
            linears=[self.lin1, self.lin2],
            ops=[
                ("linear", 0),
                ("pointwise", self.act),
                ("linear", 1),
            ],
        )


class PreNormTwoLayerPointwiseRelationMLP(torch.nn.Module):
    """Width-preserving ``Norm -> Linear -> Pointwise -> Linear`` block."""

    def __init__(
        self,
        width: int,
        hidden: int,
        *,
        activation: PointwiseKind | torch.nn.Module = "silu",
        norm: NormKind = "layernorm",
        gelu_approximate: str = "none",
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.norm = _make_norm(width, norm=norm, eps=norm_eps, affine=norm_affine)
        self.lin1 = torch.nn.Linear(width, hidden, bias=bias)
        self.act = _make_pointwise(activation, gelu_approximate=gelu_approximate)
        self.lin2 = torch.nn.Linear(hidden, width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(self.norm(x))))

    def relmo_kernel_spec(self) -> RelationBlockSpec:
        return RelationBlockSpec(
            linears=[self.lin1, self.lin2],
            ops=[
                ("norm", self.norm),
                ("linear", 0),
                ("pointwise", self.act),
                ("linear", 1),
            ],
        )


class PostNormTwoLayerPointwiseRelationMLP(torch.nn.Module):
    """Width-preserving ``Linear -> Pointwise -> Linear -> Norm`` block."""

    def __init__(
        self,
        width: int,
        hidden: int,
        *,
        activation: PointwiseKind | torch.nn.Module = "silu",
        norm: NormKind = "layernorm",
        gelu_approximate: str = "none",
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.lin1 = torch.nn.Linear(width, hidden, bias=bias)
        self.act = _make_pointwise(activation, gelu_approximate=gelu_approximate)
        self.lin2 = torch.nn.Linear(hidden, width, bias=bias)
        self.norm = _make_norm(width, norm=norm, eps=norm_eps, affine=norm_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.lin2(self.act(self.lin1(x))))

    def relmo_kernel_spec(self) -> RelationBlockSpec:
        return RelationBlockSpec(
            linears=[self.lin1, self.lin2],
            ops=[
                ("linear", 0),
                ("pointwise", self.act),
                ("linear", 1),
                ("norm", self.norm),
            ],
        )


class ThreeLayerPointwiseRelationMLP(torch.nn.Module):
    """Width-preserving ``Linear -> Pointwise -> Linear -> Pointwise -> Linear`` block."""

    def __init__(
        self,
        width: int,
        hidden1: int,
        hidden2: int,
        *,
        activation: PointwiseKind | torch.nn.Module = "silu",
        gelu_approximate: str = "none",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.lin1 = torch.nn.Linear(width, hidden1, bias=bias)
        self.act1 = _make_pointwise(activation, gelu_approximate=gelu_approximate)
        self.lin2 = torch.nn.Linear(hidden1, hidden2, bias=bias)
        self.act2 = _make_pointwise(activation, gelu_approximate=gelu_approximate)
        self.lin3 = torch.nn.Linear(hidden2, width, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin3(self.act2(self.lin2(self.act1(self.lin1(x)))))

    def relmo_kernel_spec(self) -> RelationBlockSpec:
        return RelationBlockSpec(
            linears=[self.lin1, self.lin2, self.lin3],
            ops=[
                ("linear", 0),
                ("pointwise", self.act1),
                ("linear", 1),
                ("pointwise", self.act2),
                ("linear", 2),
            ],
        )


class RelationProgram(torch.nn.Module):
    """Explicit composed relation program.

    A relation program is an ordered sequence of width-preserving relation
    blocks. It does not apply the outer tuple residual. Exact CUDA program
    kernels are matched only from this wrapper, not from arbitrary
    ``torch.nn.Sequential`` modules.
    """

    def __init__(self, *blocks: torch.nn.Module) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("RelationProgram requires at least one block.")
        self.blocks = torch.nn.ModuleList(blocks)
        widths = [getattr(block, "width", None) for block in self.blocks]
        if any(width is None for width in widths):
            raise ValueError(
                "All RelationProgram blocks must expose a 'width' attribute for validation."
            )
        unique_widths = {int(width) for width in widths}
        if len(unique_widths) != 1:
            raise ValueError(
                f"RelationProgram blocks must share one width, got {sorted(unique_widths)}."
            )
        self.width = int(next(iter(unique_widths)))

    def __iter__(self):
        return iter(self.blocks)

    def __len__(self) -> int:
        return len(self.blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out


__all__ = [
    "NormKind",
    "PointwiseKind",
    "PostNormTwoLayerPointwiseRelationMLP",
    "PreNormTwoLayerPointwiseRelationMLP",
    "RelationProgram",
    "ThreeLayerPointwiseRelationMLP",
    "TwoLayerPointwiseRelationMLP",
]
