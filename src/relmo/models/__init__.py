"""Relational GNN models and model-construction facades."""

try:  # pragma: no cover - exercised by import-safety subprocess tests
    import torch_geometric as _torch_geometric  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "relmo.models requires the optional 'models' dependencies. "
        "Install them with `pip install relmo[models]`."
    ) from exc

from . import builders, flat, hetero
from .builders import (
    ArityMLPFactory,
    EAGER_POLICY,
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    RelationBlockProto,
    RelationBlockSpec,
    RelationProgram,
    ThreeLayerPointwiseRelationMLP,
    TwoLayerPointwiseRelationMLP,
    build_eager_fallback_modules,
    build_program_relation_modules,
    build_relations,
    build_typed_relation_modules,
)
from .flat import (
    CentralizedFlatRelationModule,
    CentralizedFlatRelationalGNN,
    FlatBatchInput,
    FlatExecutionPolicy,
    FlatLGANRelationalGNN,
    FlatRelationalGNN,
    FlatRelationalOutput,
)
from .hetero import (
    CentralizedRelationalGNN,
    FastRelationalGNN,
    LGANRelationalGNN,
    RelationalGNN,
)

__all__ = [
    "ArityMLPFactory",
    "builders",
    "build_eager_fallback_modules",
    "build_program_relation_modules",
    "build_relations",
    "build_typed_relation_modules",
    "FlatExecutionPolicy",
    "FlatBatchInput",
    "CentralizedFlatRelationModule",
    "CentralizedFlatRelationalGNN",
    "EAGER_POLICY",
    "flat",
    "FlatLGANRelationalGNN",
    "FlatRelationalGNN",
    "FlatRelationalOutput",
    "hetero",
    "RelationBlockProto",
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
