"""Public facade for heterogeneous relational model variants."""

from .relational_gnn import (
    CentralizedRelationalGNN,
    FastRelationalGNN,
    LGANRelationalGNN,
    RelationalGNN,
)

__all__ = [
    "CentralizedRelationalGNN",
    "FastRelationalGNN",
    "LGANRelationalGNN",
    "RelationalGNN",
]