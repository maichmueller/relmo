"""Misc helpers for relmo models."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def stream_context(stream):
    if stream is not None:
        with torch.cuda.stream(stream):
            yield
    else:
        with nullcontext():
            yield
