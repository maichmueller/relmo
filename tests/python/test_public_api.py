from __future__ import annotations

import relmo
import relmo.models.flat_relational as flat_relational


def test_package_exports_models_and_ops() -> None:
    assert relmo.models is not None
    assert relmo.ops is not None


def test_model_facades_expose_stable_entrypoints() -> None:
    assert relmo.models.flat.FlatRelationalGNN is relmo.models.FlatRelationalGNN
    assert relmo.models.hetero.RelationalGNN is relmo.models.RelationalGNN
    assert relmo.models.builders.ArityMLPFactory is relmo.models.ArityMLPFactory
    assert relmo.models.builders.build_typed_relation_modules is not None
    assert not hasattr(relmo.models, "FlatRelationKernel")
    assert not hasattr(flat_relational, "FlatRelationKernel")