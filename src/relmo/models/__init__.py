"""Relational GNN models and flat relation building blocks."""

from .flat_contract import FlatBatchInput, FlatExecutionPolicy, FlatRelationalOutput
from .centralized_flat_relational_gnn import (
    CentralizedFlatRelationModule,
    CentralizedFlatRelationalGNN,
)
from .flat_lgan_relational_gnn import FlatLGANRelationalGNN
from .flat_relational_gnn import FlatRelationalGNN
from .flat_relational import FlatRelationKernel
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
    "CentralizedFlatRelationModule",
    "CentralizedFlatRelationalGNN",
    "FlatRelationKernel",
    "FlatLGANRelationalGNN",
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
