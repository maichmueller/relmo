"""Flat relational internals split by subsystem."""

from .collectors import FlatRelationCollectorMixin
from .kernels import (
    FlatRelationKernel,
    GELUBlockKernel,
    MishBlockKernel,
    PostNormMishLayerNormKernel,
    PostNormSiLULayerNormKernel,
    PreNormRMSNormThenSiLUProgramKernel,
    PreNormSiLURMSNormKernel,
    SiLUBlockKernel,
    SiLUPairProgramKernel,
    SiLUThenPostNormProgramKernel,
    build_default_kernel_registry,
)
from .matching import FlatRelationMatchingMixin
from .runners import FlatRelationRunnerMixin
from .types import (
    BlockKernelSpec,
    CentralizedBatchSpec,
    FlatTopology,
    KernelBatchPlan,
    KernelMatch,
    ProgramKernelSpec,
    RelationSlice,
    build_flat_topology,
    normalize_relation_arities,
    normalize_relation_counts,
    topology_cache_key,
)

__all__ = [
    "FlatRelationCollectorMixin",
    "FlatRelationKernel",
    "FlatRelationMatchingMixin",
    "FlatRelationRunnerMixin",
    "GELUBlockKernel",
    "MishBlockKernel",
    "PostNormMishLayerNormKernel",
    "PostNormSiLULayerNormKernel",
    "PreNormRMSNormThenSiLUProgramKernel",
    "PreNormSiLURMSNormKernel",
    "SiLUBlockKernel",
    "SiLUPairProgramKernel",
    "SiLUThenPostNormProgramKernel",
    "build_default_kernel_registry",
    "BlockKernelSpec",
    "CentralizedBatchSpec",
    "FlatTopology",
    "KernelBatchPlan",
    "KernelMatch",
    "ProgramKernelSpec",
    "RelationSlice",
    "build_flat_topology",
    "normalize_relation_arities",
    "normalize_relation_counts",
    "topology_cache_key",
]
