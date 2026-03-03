from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.typing import Adj

from .mixins import DeviceAwareMixin

try:  # pragma: no cover - optional runtime dependency
    import mifrost  # type: ignore
except Exception:  # pragma: no cover - keep module importable without mifrost
    mifrost = None  # type: ignore


class PyGModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor = None,
        **kwargs,
    ) -> Tensor: ...

    def __call__(self, *args, **kwargs):
        if not args:
            raise NotImplementedError("No input for '__call__'")
        data = args[0]
        if isinstance(data, dict):
            x = data
            edge_index = args[1]
            batch = args[2] if len(args) > 2 else kwargs.pop("batch", None)
            return super().__call__(x, edge_index, batch, *args[3:], **kwargs)
        if isinstance(data, (Data, Batch)):
            return super().__call__(*self.unpack(data), *args[1:], **kwargs)
        if mifrost is not None and isinstance(
            data, (mifrost.BatchEncoding, mifrost.HomoBatchEncodingView)
        ):
            return super().__call__(*self.unpack_native(data), *args[1:], **kwargs)
        raise NotImplementedError(f"Invalid input type {type(data)!r} for '__call__'")

    def unpack_native(self, data):
        if mifrost is None:
            raise RuntimeError("mifrost is not available.")
        if isinstance(data, mifrost.BatchEncoding):
            data = data.as_homo()
        return (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.batch.to(self.device),
        )

    @classmethod
    def unpack(cls, data: Data | Batch):
        return data.x, data.edge_index, data.batch


class PyGHeteroModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[str, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ): ...

    def __call__(self, *args, **kwargs):
        if not args:
            raise NotImplementedError("No input for '__call__'")
        data = args[0]
        if isinstance(data, dict):
            x_dict = data
            edge_index_dict = args[1]
            batch_dict = args[2] if len(args) > 2 else kwargs.pop("batch_dict", None)
            return super().__call__(
                x_dict, edge_index_dict, batch_dict, *args[3:], **kwargs
            )
        if isinstance(data, HeteroData):
            return super().__call__(
                data.x_dict, data.edge_index_dict, data.batch_dict, *args[1:], **kwargs
            )
        if isinstance(data, Batch):
            return super().__call__(*self.unpack(data), *args[1:], **kwargs)
        if mifrost is not None and isinstance(
            data, (mifrost.BatchEncoding, mifrost.HeteroBatchEncodingView)
        ):
            return super().__call__(*self.unpack_native(data), *args[1:], **kwargs)
        raise NotImplementedError(f"Invalid input type {type(data)!r} for '__call__'")

    def unpack_native(self, data):
        if mifrost is None:
            raise RuntimeError("mifrost is not available.")
        if isinstance(data, mifrost.BatchEncoding):
            data = data.as_hetero()
        data = data.to(self.device)
        x_dict = {key: value.to(self.device) for key, value in data.x_dict.items()}
        edge_index_dict = {
            key: value.to(self.device) for key, value in data.edge_index_dict.items()
        }
        batch_dict = {
            key: value.to(self.device) for key, value in data.batch_dict.items()
        }
        return x_dict, edge_index_dict, batch_dict

    @classmethod
    def unpack(cls, data: HeteroData | Batch):
        return data.x_dict, data.edge_index_dict, data.batch_dict
