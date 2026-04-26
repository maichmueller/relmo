"""Public facade for flat relational model variants."""

from .centralized_flat_relational_gnn import (
    CentralizedFlatRelationModule,
    CentralizedFlatRelationalGNN,
)
from .flat_lgan_relational_gnn import FlatLGANRelationalGNN
from .flat_relational.flat_contract import FlatBatchInput, FlatExecutionPolicy, FlatRelationalOutput
from .flat_relational_gnn import FlatRelationalGNN

__all__ = [
    "CentralizedFlatRelationModule",
    "CentralizedFlatRelationalGNN",
    "FlatBatchInput",
    "FlatExecutionPolicy",
    "FlatLGANRelationalGNN",
    "FlatRelationalGNN",
    "FlatRelationalOutput",
]