"""Flat batch preparation boundary.

This module owns public-carrier dispatch for the flat model stack. The model
core should operate on prepared flat batches; adapter details for PyG and
native mifrost carriers stay here.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch_geometric as pyg
import mifrost
from ..flat_relational.flat_contract import _FlatPreparedBatch


PrepareFn = Callable[..., _FlatPreparedBatch]


class FlatBatchPreparer:
    """Dispatch public flat carriers into the internal prepared-batch contract."""

    def __init__(
        self,
        *,
        prepare_native: PrepareFn,
        prepare_pyg: PrepareFn,
    ) -> None:
        self._prepare_native = prepare_native
        self._prepare_pyg = prepare_pyg

    def prepare(self, data: Any, *, cache: dict | None = None) -> _FlatPreparedBatch:
        if getattr(data, "_relm_flat_prepared_batch", False):
            return data
        if mifrost is not None and isinstance(data, mifrost.BatchEncoding):
            return self._prepare_native(data, cache=cache)
        if isinstance(data, (pyg.data.Data, pyg.data.Batch)):
            return self._prepare_pyg(data, cache=cache)
        raise TypeError(
            "FlatRelationalGNN expects a mifrost flat BatchEncoding or a PyG "
            "Data/Batch carrying relation_counts and relation_args."
        )


__all__ = ["FlatBatchPreparer"]
