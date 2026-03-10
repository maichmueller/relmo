from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.typing import Adj

from .flat_contract import FlatBatchInput
from .mixins import DeviceAwareMixin

import mifrost  # type: ignore


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
        if isinstance(data, (mifrost.BatchEncoding, mifrost.HomoBatchEncodingView)):
            return super().__call__(*self.unpack_native(data), *args[1:], **kwargs)
        raise NotImplementedError(f"Invalid input type {type(data)!r} for '__call__'")

    def unpack_native(self, data):
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
        if isinstance(data, (mifrost.BatchEncoding, mifrost.HeteroBatchEncodingView)):
            return super().__call__(*self.unpack_native(data), *args[1:], **kwargs)
        raise NotImplementedError(f"Invalid input type {type(data)!r} for '__call__'")

    def unpack_native(self, data):
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


class PyGFlatModule(DeviceAwareMixin, Module, ABC):
    @abstractmethod
    def forward(self, data: FlatBatchInput, **kwargs): ...

    def __call__(self, *args, **kwargs):
        if not args:
            raise NotImplementedError("No input for '__call__'")
        data = args[0]
        if getattr(data, "_relm_flat_prepared_batch", False):
            forward_prepared = getattr(self, "_forward_prepared_batch", None)
            if not callable(forward_prepared):
                raise NotImplementedError(
                    f"{type(self)!r} does not implement internal prepared flat-batch dispatch."
                )
            return forward_prepared(data, *args[1:], **kwargs)
        if isinstance(data, (Data, Batch)) and hasattr(data, "relation_counts") and hasattr(data, "relation_args"):
            return super().__call__(data, *args[1:], **kwargs)
        if (
            isinstance(data, mifrost.BatchEncoding)
            and hasattr(data, "relation_counts")
            and hasattr(data, "relation_args")
        ):
            return super().__call__(data, *args[1:], **kwargs)
        raise TypeError(
            "Flat relational models accept only mifrost flat BatchEncoding inputs "
            "or PyG Data/Batch objects with relation_counts and relation_args."
        )

    @classmethod
    def unpack(cls, data: Data | Batch):
        return cls.unpack_pyg_flat(data)

    def unpack_native_flat(self, data):
        if isinstance(data, mifrost.BatchEncoding):
            data = data.to(self.device)

        relation_counts = data.relation_counts
        relation_args = data.relation_args
        relation_arities = getattr(data, "relation_arities", None)

        x = getattr(data, "x", None)
        if x is None:
            node_sizes = getattr(data, "node_sizes", None)
            if torch.is_tensor(node_sizes) and node_sizes.numel() > 0:
                node_count = int(node_sizes.sum().item())
                device = node_sizes.device
            else:
                node_count = int(getattr(data, "num_nodes", 0))
                device = relation_args.device if torch.is_tensor(relation_args) else torch.device("cpu")
            x = torch.zeros((node_count, 1), dtype=torch.float, device=device)

        extras = {}
        for key in (
            "relation_names",
            "relation_sources",
            "batch",
            "node_sizes",
            "object_indices",
            "object_sizes",
            "history_entity_indices",
            "history_entity_sizes",
            "history_entity_dt",
            "target_entity_indices",
            "target_entity_group_ids",
            "target_entity_sizes",
            "target_positions",
            "target_group_ids",
            "target_sizes",
            "target_indices",
            "target_candidate_ids",
            "relation_instance_sizes",
            "lgan_tn_relation_indices",
            "lgan_tn_entity_indices",
            "lgan_tn_sizes",
            "lgan_nn_relation_indices",
            "lgan_nn_entity_indices",
            "lgan_nn_sizes",
            "lgan_rr_src_relation_indices",
            "lgan_rr_dst_relation_indices",
            "lgan_rr_sizes",
            "lgan_tn_edge_pos",
            "lgan_nn_edge_pos",
            "lgan_rr_edge_pos",
        ):
            if not hasattr(data, key):
                continue
            value = getattr(data, key)
            extras[key] = value
        return x, relation_counts, relation_args, relation_arities, extras

    @classmethod
    def unpack_pyg_flat(cls, data: Data | Batch):
        extras = {}
        for key in (
            "relation_names",
            "relation_sources",
            "batch",
            "node_sizes",
            "object_indices",
            "object_sizes",
            "history_entity_indices",
            "history_entity_sizes",
            "history_entity_dt",
            "target_entity_indices",
            "target_entity_group_ids",
            "target_entity_sizes",
            "target_positions",
            "target_group_ids",
            "target_sizes",
            "target_indices",
            "target_candidate_ids",
            "relation_instance_sizes",
            "lgan_tn_relation_indices",
            "lgan_tn_entity_indices",
            "lgan_tn_sizes",
            "lgan_nn_relation_indices",
            "lgan_nn_entity_indices",
            "lgan_nn_sizes",
            "lgan_rr_src_relation_indices",
            "lgan_rr_dst_relation_indices",
            "lgan_rr_sizes",
            "lgan_tn_edge_pos",
            "lgan_nn_edge_pos",
            "lgan_rr_edge_pos",
        ):
            if hasattr(data, key):
                extras[key] = getattr(data, key)
        relation_arities = getattr(data, "relation_arities", None)
        return data.x, data.relation_counts, data.relation_args, relation_arities, extras
