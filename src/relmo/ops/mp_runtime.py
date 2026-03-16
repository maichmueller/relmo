"""Runtime loading and compatibility policy for ``relmo.ops.mp``.

This module isolates the side-effectful parts of the mp stack:

- optional ``torch`` import handling
- environment-variable policy parsing
- custom library discovery and ``torch.ops`` loading
- build metadata parsing and runtime compatibility checks
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path

from .mp_constants import REQUIRED_NAMESPACE_OPS

try:  # pragma: no cover - exercised in environments without torch
    import torch
except Exception as exc:  # pragma: no cover - exercised in minimal wheels
    torch = None
    TORCH_IMPORT_ERROR: Exception | None = exc
else:
    TORCH_IMPORT_ERROR = None

_LIB_LOADED = False
_LIB_LOAD_ERROR: Exception | None = None
_BUILD_INFO_CACHE: dict[str, str] | None = None
_OPS_NAMESPACE = "relm_mp"
_RUNTIME_COMPAT_VALIDATED = False

_BOOL_FALSE = {"0", "false", "no", "off"}
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


def require_torch(feature_name: str) -> None:
    if torch is None:
        raise ModuleNotFoundError(f"{feature_name} requires torch.") from TORCH_IMPORT_ERROR


def env_bool_any(names: tuple[str, ...], default: bool) -> bool:
    for name in names:
        raw = os.getenv(name)
        if raw is not None:
            return raw.strip().lower() not in _BOOL_FALSE
    return default


def env_first(names: tuple[str, ...], default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None:
            return value
    return default


def fallback_mode() -> str:
    raw = (env_first(("RELM_MP_FALLBACK",), "python") or "python").strip().lower()
    return raw if raw in {"python", "error"} else "python"


def torch_version_policy() -> str:
    raw = (
        env_first(("RELM_MP_TORCH_VERSION_POLICY",), "forward") or "forward"
    ).strip().lower()
    return raw if raw in {"forward", "strict"} else "forward"


def runtime_cuda_tag() -> str:
    require_torch("mp runtime compatibility checks")
    cuda = getattr(torch.version, "cuda", None)
    if not cuda:
        return "cpu"
    return f"cu{str(cuda).replace('.', '')}"


def major_minor(version: str) -> tuple[int, int]:
    match = _VERSION_RE.match(version)
    if match is None:
        raise RuntimeError(f"Could not parse torch version {version!r}.")
    return int(match.group(1)), int(match.group(2))


def parse_build_info(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in raw.split(";"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def candidate_libraries() -> list[Path]:
    pkg_dir = Path(__file__).resolve().parent.parent
    if sys.platform == "darwin":
        patterns = (
            "relm_mp_ops*.dylib",
            "librelm_mp_ops*.dylib",
            "relm_mp_ops*.so",
        )
    elif os.name == "nt":
        patterns = ("relm_mp_ops*.pyd",)
    else:
        patterns = (
            "relm_mp_ops*.so",
            "librelm_mp_ops*.so",
        )

    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(pkg_dir.glob(pattern)))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        deduped.append(resolved)
        seen.add(resolved)
    return deduped


def ensure_loaded() -> None:
    global _LIB_LOADED, _LIB_LOAD_ERROR, _OPS_NAMESPACE

    if _LIB_LOADED:
        return
    require_torch("relmo.ops.mp")

    last_error: Exception | None = None
    for candidate in candidate_libraries():
        try:
            torch.ops.load_library(str(candidate))

            namespace = getattr(torch.ops, "relm_mp", None)
            if namespace is None or any(
                not hasattr(namespace, op_name) for op_name in REQUIRED_NAMESPACE_OPS
            ):
                raise RuntimeError(
                    f"Loaded {candidate} but torch.ops.relm_mp is incomplete."
                )
            _OPS_NAMESPACE = "relm_mp"
            _LIB_LOADED = True
            _LIB_LOAD_ERROR = None
            return
        except Exception as exc:  # pragma: no cover - depends on local build state
            last_error = exc

    _LIB_LOAD_ERROR = last_error
    if last_error is not None:
        raise RuntimeError("Failed to load relm_mp custom op library.") from last_error
    raise FileNotFoundError(
        "Could not find relm_mp custom op library in the relmo package directory."
    )


def ops_namespace():
    require_torch("mp custom ops")
    return getattr(torch.ops, _OPS_NAMESPACE)


def ensure_runtime_compat_once() -> None:
    global _RUNTIME_COMPAT_VALIDATED
    if _RUNTIME_COMPAT_VALIDATED:
        return
    assert_runtime_compat()
    _RUNTIME_COMPAT_VALIDATED = True


def assert_runtime_compat() -> dict[str, str]:
    """Raise if runtime torch/cuda is incompatible with the built op library."""

    global _BUILD_INFO_CACHE

    require_torch("mp compatibility checks")
    ensure_loaded()
    if _BUILD_INFO_CACHE is None:
        _BUILD_INFO_CACHE = parse_build_info(ops_namespace().build_info())

    info = _BUILD_INFO_CACHE
    build_torch = info.get("build_torch")
    if not build_torch:
        raise RuntimeError("mp build metadata is missing build_torch.")

    runtime_torch = str(torch.__version__)
    build_mm = major_minor(build_torch)
    runtime_mm = major_minor(runtime_torch)
    if build_mm != runtime_mm:
        policy = torch_version_policy()
        if policy == "strict":
            raise RuntimeError(
                "mp torch version mismatch: "
                f"built against {build_torch}, runtime is {runtime_torch}. "
                "Expected matching major.minor."
            )
        if runtime_mm[0] != build_mm[0]:
            raise RuntimeError(
                "mp torch major version mismatch: "
                f"built against {build_torch}, runtime is {runtime_torch}."
            )
        if runtime_mm < build_mm:
            raise RuntimeError(
                "mp torch runtime is older than the build torch: "
                f"built against {build_torch}, runtime is {runtime_torch}. "
                "Rebuild against the older torch or upgrade the runtime torch."
            )
        warnings.warn(
            "mp torch version drift: "
            f"built against {build_torch}, runtime is {runtime_torch}. "
            "Continuing because RELM_MP_TORCH_VERSION_POLICY=forward. "
            "Treat this as compatibility-by-test, not ABI-guaranteed.",
            RuntimeWarning,
            stacklevel=2,
        )

    build_cuda = info.get("build_cuda_tag", "cpu")
    runtime_cuda = runtime_cuda_tag()
    if build_cuda != "cpu" and build_cuda != runtime_cuda:
        raise RuntimeError(
            "mp CUDA tag mismatch: "
            f"built for {build_cuda}, runtime is {runtime_cuda}."
        )
    return info


def available() -> bool:
    """Return ``True`` when custom ops are loadable and runtime-compatible."""

    if torch is None:
        return False
    try:
        ensure_loaded()
        assert_runtime_compat()
    except Exception:
        return False
    return True
