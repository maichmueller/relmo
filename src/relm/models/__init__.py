"""Relational GNN models and message passing modules."""

from .grouped_mlp import GroupedMLPCompatible, GroupedMLPSpec
from .mlp import ArityMLPFactory
from .relational_gnn import (
    CentralizedRelationalGNN,
    FastRelationalGNN,
    LGANRelationalGNN,
    RelationalGNN,
)

__all__ = [
    "ArityMLPFactory",
    "GroupedMLPSpec",
    "GroupedMLPCompatible",
    "RelationalGNN",
    "FastRelationalGNN",
    "CentralizedRelationalGNN",
    "LGANRelationalGNN",
]
