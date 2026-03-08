"""Relational GNN models and message passing modules."""

from .grouped_mlp import GroupedMLPCompatible, GroupedMLPSpec
from .flat_relational_gnn import FlatPreparedInputs, FlatRelationalGNN
from .mlp import ArityMLPFactory
from .relation_mlp_stacks import (
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    ThreeLayerPointwiseRelationMLP,
    TwoLayerPointwiseRelationMLP,
)
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
    "FlatPreparedInputs",
    "FlatRelationalGNN",
    "TwoLayerPointwiseRelationMLP",
    "PreNormTwoLayerPointwiseRelationMLP",
    "PostNormTwoLayerPointwiseRelationMLP",
    "ThreeLayerPointwiseRelationMLP",
    "RelationalGNN",
    "FastRelationalGNN",
    "CentralizedRelationalGNN",
    "LGANRelationalGNN",
]
