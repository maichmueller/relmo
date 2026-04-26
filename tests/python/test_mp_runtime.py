from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from relmo.ops import mp_dispatch, mp_runtime


@pytest.fixture(autouse=True)
def _reset_mp_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp_runtime, "_LIB_LOADED", False)
    monkeypatch.setattr(mp_runtime, "_LIB_LOAD_ERROR", None)
    monkeypatch.setattr(mp_runtime, "_BUILD_INFO_CACHE", None)
    monkeypatch.setattr(mp_runtime, "_RUNTIME_COMPAT_VALIDATED", False)


def _fake_torch(*, version: str, cuda: str | None = None, namespace: object | None = None):
    ops = SimpleNamespace(load_library=lambda path: None)
    if namespace is not None:
        setattr(ops, "relm_mp", namespace)
    return SimpleNamespace(
        __version__=version,
        version=SimpleNamespace(cuda=cuda),
        ops=ops,
    )


def test_runtime_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELM_MP_ENABLE", "false")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    monkeypatch.setenv("RELM_MP_TORCH_VERSION_POLICY", "strict")

    assert mp_runtime.env_bool_any(("RELM_MP_ENABLE",), True) is False
    assert mp_runtime.fallback_mode() == "error"
    assert mp_runtime.torch_version_policy() == "strict"


def test_assert_runtime_compat_forward_policy_warns_on_newer_minor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = SimpleNamespace(build_info=lambda: "build_torch=2.5.1;build_cuda_tag=cpu")
    monkeypatch.setattr(
        mp_runtime,
        "torch",
        _fake_torch(version="2.6.0", cuda=None, namespace=namespace),
    )
    monkeypatch.setattr(mp_runtime, "ensure_loaded", lambda: None)
    monkeypatch.setattr(mp_runtime, "ops_namespace", lambda: namespace)
    monkeypatch.setattr(mp_runtime, "torch_version_policy", lambda: "forward")

    with pytest.warns(RuntimeWarning, match="version drift"):
        info = mp_runtime.assert_runtime_compat()

    assert info["build_torch"] == "2.5.1"


def test_assert_runtime_compat_strict_policy_rejects_minor_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = SimpleNamespace(build_info=lambda: "build_torch=2.5.1;build_cuda_tag=cpu")
    monkeypatch.setattr(
        mp_runtime,
        "torch",
        _fake_torch(version="2.6.0", cuda=None, namespace=namespace),
    )
    monkeypatch.setattr(mp_runtime, "ensure_loaded", lambda: None)
    monkeypatch.setattr(mp_runtime, "ops_namespace", lambda: namespace)
    monkeypatch.setattr(mp_runtime, "torch_version_policy", lambda: "strict")

    with pytest.raises(RuntimeError, match="Expected matching major.minor"):
        mp_runtime.assert_runtime_compat()


def test_assert_runtime_compat_rejects_older_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    namespace = SimpleNamespace(build_info=lambda: "build_torch=2.6.0;build_cuda_tag=cpu")
    monkeypatch.setattr(
        mp_runtime,
        "torch",
        _fake_torch(version="2.5.1", cuda=None, namespace=namespace),
    )
    monkeypatch.setattr(mp_runtime, "ensure_loaded", lambda: None)
    monkeypatch.setattr(mp_runtime, "ops_namespace", lambda: namespace)
    monkeypatch.setattr(mp_runtime, "torch_version_policy", lambda: "forward")

    with pytest.raises(RuntimeError, match="runtime is older than the build torch"):
        mp_runtime.assert_runtime_compat()


def test_assert_runtime_compat_rejects_cuda_tag_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = SimpleNamespace(build_info=lambda: "build_torch=2.5.1;build_cuda_tag=cu124")
    monkeypatch.setattr(
        mp_runtime,
        "torch",
        _fake_torch(version="2.5.1", cuda="12.1", namespace=namespace),
    )
    monkeypatch.setattr(mp_runtime, "ensure_loaded", lambda: None)
    monkeypatch.setattr(mp_runtime, "ops_namespace", lambda: namespace)

    with pytest.raises(RuntimeError, match="CUDA tag mismatch"):
        mp_runtime.assert_runtime_compat()


def test_available_returns_false_when_compat_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mp_runtime, "torch", object())
    monkeypatch.setattr(mp_runtime, "ensure_loaded", lambda: None)
    monkeypatch.setattr(
        mp_runtime,
        "assert_runtime_compat",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert mp_runtime.available() is False


def test_ensure_loaded_raises_when_library_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp_runtime, "candidate_libraries", lambda: [])
    monkeypatch.setattr(mp_runtime, "torch", _fake_torch(version="2.5.1", cuda=None))

    with pytest.raises(FileNotFoundError, match="Could not find relm_mp custom op library"):
        mp_runtime.ensure_loaded()


def test_ensure_loaded_raises_when_namespace_is_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeOps:
        def __init__(self) -> None:
            self.relm_mp = SimpleNamespace(fanout_scatter=lambda *args: None)

        def load_library(self, path: str) -> None:
            del path

    monkeypatch.setattr(mp_runtime, "candidate_libraries", lambda: [Path("/tmp/fake.so")])
    monkeypatch.setattr(
        mp_runtime,
        "torch",
        SimpleNamespace(
            __version__="2.5.1",
            version=SimpleNamespace(cuda=None),
            ops=FakeOps(),
        ),
    )

    with pytest.raises(RuntimeError, match="Failed to load relm_mp custom op library"):
        mp_runtime.ensure_loaded()


def test_dispatch_disable_env_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELM_MP_ENABLE", "0")

    assert mp_dispatch.should_use_custom("fanout_scatter") is False


def test_dispatch_error_mode_raises_when_custom_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RELM_MP_ENABLE", raising=False)
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    monkeypatch.setattr(
        mp_dispatch,
        "ensure_runtime_compat_once",
        lambda: (_ for _ in ()).throw(RuntimeError("missing custom op")),
    )

    with pytest.raises(RuntimeError, match="Custom mp op fanout_scatter is unavailable"):
        mp_dispatch.should_use_custom("fanout_scatter")


def test_namespace_has_op_uses_loaded_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mp_dispatch, "torch", object())
    monkeypatch.setattr(mp_dispatch, "ensure_loaded", lambda: None)
    monkeypatch.setattr(
        mp_dispatch,
        "ops_namespace",
        lambda: SimpleNamespace(example_op=lambda *args: None),
    )

    assert mp_dispatch.namespace_has_op("example_op") is True
    assert mp_dispatch.namespace_has_op("missing_op") is False
