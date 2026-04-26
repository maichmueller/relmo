"""Public builders and helper constructors for relation modules."""

from __future__ import annotations

from typing import Mapping

import torch

from .flat_relational.flat_contract import FlatExecutionPolicy
from .mlp import ArityMLPFactory
from .relation_blocks import (
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    RelationBlockProto,
    RelationBlockSpec,
    RelationProgram,
    ThreeLayerPointwiseRelationMLP,
    TwoLayerPointwiseRelationMLP,
)


def build_relations(batch) -> dict[str, int]:
    return {
        name: int(arity)
        for name, arity in zip(batch.relation_names, batch.relation_arities)
    }


def build_typed_relation_modules(
    relations: Mapping[str, int],
    *,
    embedding_size: int = 32,
    activation: str = "silu",
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(64, width)
        modules[name] = TwoLayerPointwiseRelationMLP(
            width=width,
            hidden=hidden,
            activation=activation,
        )
    return modules


def build_program_relation_modules(
    relations: Mapping[str, int],
    *,
    embedding_size: int = 32,
    activation: str = "silu",
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(64, width)
        modules[name] = RelationProgram(
            TwoLayerPointwiseRelationMLP(width=width, hidden=hidden, activation=activation),
            TwoLayerPointwiseRelationMLP(width=width, hidden=hidden, activation=activation),
        )
    return modules


def build_eager_fallback_modules(
    relations: Mapping[str, int],
    *,
    embedding_size: int = 32,
) -> dict[str, torch.nn.Module]:
    modules: dict[str, torch.nn.Module] = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(64, width)
        modules[name] = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, width),
        )
    return modules


EAGER_POLICY = FlatExecutionPolicy(
    relation_kernels="off",
    program_kernels="off",
    relation_gather="off",
)


__all__ = [
    "ArityMLPFactory",
    "EAGER_POLICY",
    "FlatExecutionPolicy",
    "PostNormTwoLayerPointwiseRelationMLP",
    "PreNormTwoLayerPointwiseRelationMLP",
    "RelationBlockProto",
    "RelationBlockSpec",
    "RelationProgram",
    "ThreeLayerPointwiseRelationMLP",
    "TwoLayerPointwiseRelationMLP",
    "build_eager_fallback_modules",
    "build_program_relation_modules",
    "build_relations",
    "build_typed_relation_modules",
]