from __future__ import annotations

import os
import subprocess
import sys
import textwrap

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


def _run_import_probe(source: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    repo_src = os.fspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
    env["PYTHONPATH"] = repo_src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_root_import_does_not_import_optional_models() -> None:
    result = _run_import_probe(
        """
        import importlib.abc
        import sys

        class BlockOptionalModels(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch_geometric" or fullname.startswith("torch_geometric."):
                    raise ModuleNotFoundError(fullname)
                if fullname == "mifrost" or fullname.startswith("mifrost."):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalModels())
        import relmo
        assert "relmo.models" not in sys.modules
        from relmo.ops import mp
        assert mp.available() in {True, False}
        """
    )
    assert result.returncode == 0, result.stderr


def test_model_import_reports_missing_models_extra() -> None:
    result = _run_import_probe(
        """
        import importlib.abc
        import relmo
        import sys

        class BlockTorchGeometric(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch_geometric" or fullname.startswith("torch_geometric."):
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockTorchGeometric())
        try:
            relmo.models
        except ModuleNotFoundError as exc:
            assert "relmo[models]" in str(exc)
        else:
            raise AssertionError("relmo.models unexpectedly imported without torch_geometric")
        """
    )
    assert result.returncode == 0, result.stderr
