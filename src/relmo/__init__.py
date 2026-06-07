"""Core package facade for relmo.

The root package intentionally avoids importing optional model dependencies.
Use ``relmo.ops`` for the core operator wrappers and ``relmo.models`` for the
model stack when the model extras are installed.
"""

from importlib import import_module
from types import ModuleType

__all__ = ["models", "ops"]


def __getattr__(name: str) -> ModuleType:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted((*globals().keys(), *__all__))
