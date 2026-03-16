from __future__ import annotations

import os
import subprocess
import sys


def test_mp_public_facade_preserves_expected_symbols() -> None:
    import relmo.ops.mp as mp

    assert "fanout_scatter" in mp.__all__
    assert "fanin_reduce" in mp.__all__
    assert "activation_code" in mp.__all__
    assert "available" in mp.__all__
    assert hasattr(mp, "_lgan_pool_reduce")
    assert hasattr(mp, "_lgan_relation_graph_step")
    assert hasattr(mp, "_lgan_build_pointwise_step")


def test_mp_import_without_torch_preserves_lazy_failure() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    code = """
import builtins
import importlib
import sys

real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("simulated missing torch")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import
for key in list(sys.modules):
    if key == "torch" or key.startswith("torch.") or key.startswith("relmo.ops"):
        sys.modules.pop(key, None)

mp = importlib.import_module("relmo.ops.mp")
assert mp.torch is None
assert mp.available() is False
assert mp.activation_code(("relu",)) == 1
try:
    mp.fanout_scatter(None, None, None, 0)
except ModuleNotFoundError as exc:
    assert "requires torch" in str(exc)
else:
    raise AssertionError("fanout_scatter should fail lazily when torch is missing")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=os.getcwd(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
