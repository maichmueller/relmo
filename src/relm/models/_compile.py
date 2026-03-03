"""Optional torch.compile decorator for instance methods."""

from __future__ import annotations

import types
from functools import wraps
from typing import Any, Callable

import torch

from ._logging import get_logger


def optional_compile(
    enable_attr: str | None = None,
    kwargs_attr: str | None = None,
    cache_attr: str | None = None,
    **default_compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Optionally compile an instance method with torch.compile on first call."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = func.__name__
        _enable_attr = enable_attr or f"_compile_{func_name}_enable"
        _kwargs_attr = kwargs_attr or f"_compile_{func_name}_kwargs"
        _cache_attr = cache_attr or f"_compiled_{func_name}_cached"

        @wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any) -> Any:
            enabled = bool(getattr(self, _enable_attr, False))
            if not enabled:
                return func(self, *args, **kwargs)

            compiled_bound = getattr(self, _cache_attr, None)
            if compiled_bound is None:
                compile_kwargs: dict[str, Any] = dict(default_compile_kwargs)
                inst_kwargs = getattr(self, _kwargs_attr, None)
                if isinstance(inst_kwargs, dict):
                    compile_kwargs.update(inst_kwargs)
                try:
                    compiled_unbound = torch.compile(func, **compile_kwargs)
                    compiled_bound = types.MethodType(compiled_unbound, self)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    get_logger(__name__).warning(
                        "optional_compile: failed for %s on %s; using eager.",
                        func_name,
                        type(self).__name__,
                        exc_info=True,
                    )
                    orig = types.MethodType(func, self)
                    setattr(self, _cache_attr, orig)
                    return orig(*args, **kwargs)
                setattr(self, _cache_attr, compiled_bound)
            return compiled_bound(*args, **kwargs)

        return wrapper

    return decorator
