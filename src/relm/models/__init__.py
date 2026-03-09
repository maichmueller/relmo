"""Relational GNN models and flat relation building blocks."""

from .flat_contract import FlatBatchInput, FlatExecutionPolicy, FlatRelationalOutput
from .flat_relational_gnn import FlatRelationalGNN
from .flat_relational_layer import FlatRelationKernel
from .mlp import ArityMLPFactory
from .relation_block_spec import RelationBlockCompatible, RelationBlockSpec
from .relation_blocks import (
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    RelationProgram,
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
    "FlatExecutionPolicy",
    "FlatBatchInput",
    "FlatRelationKernel",
    "FlatRelationalGNN",
    "FlatRelationalOutput",
    "RelationBlockCompatible",
    "RelationBlockSpec",
    "RelationProgram",
    "TwoLayerPointwiseRelationMLP",
    "PreNormTwoLayerPointwiseRelationMLP",
    "PostNormTwoLayerPointwiseRelationMLP",
    "ThreeLayerPointwiseRelationMLP",
    "RelationalGNN",
    "FastRelationalGNN",
    "CentralizedRelationalGNN",
    "LGANRelationalGNN",
]
