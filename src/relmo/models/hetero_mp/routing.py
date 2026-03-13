from __future__ import annotations

import abc
from abc import ABC
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch_geometric.nn import Aggregation
from torch_geometric.nn.conv.hetero_conv import group
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import EdgeType

from .._logging import get_logger


class HeteroRouting(torch.nn.Module, ABC):
    """
    Handles heterogeneous message passing very similar to pyg.nn.HeteroConv.
    Instead of specifying a convolution for each EdgeType more generic rules can be used.
    """

    def __init__(
        self, aggr: Optional[str | Aggregation] = None, strict_filter_mode: bool = False
    ) -> None:
        super().__init__()
        if isinstance(aggr, str):
            try:
                self.aggr = aggregation_resolver(query=aggr)
            except ValueError:
                if aggr != "cat" and aggr != "stack":
                    get_logger(__name__).warning(
                        "Failed to resolve aggregation: " + aggr
                    )
                self.aggr = aggr
        else:
            self.aggr = aggr
        self.strict_filter_mode = strict_filter_mode

    @abc.abstractmethod
    def _accepts_edge(self, edge_type: EdgeType) -> bool: ...

    @abc.abstractmethod
    def _internal_forward(self, x, edges_index, edge_type: EdgeType, **kwargs): ...

    def forward(self, x_dict, edge_index_dict, **kwargs) -> Dict[str, Tensor]:
        """
        Apply message passing to each edge_index key if the edge-type is accepted.

        Calls the internal forward with a normal homogenous signature of x, edge_index

        :param x_dict: Dictionary with a feature matrix for each node type
        :param edge_index_dict: One edge_index adjacency matrix for each edge type.
        :return: Dictionary with each processed dst as key and their updated embedding as value.
        """
        out_dict: Dict[str, Any] = dict()
        for edge_type in filter(self._accepts_edge, edge_index_dict.keys()):
            src, rel, dst = edge_type
            if src == dst and src in x_dict:
                x = x_dict[src]
            elif src in x_dict or dst in x_dict:
                x = (
                    x_dict.get(src, None),
                    x_dict.get(dst, None),
                )
            else:
                raise KeyError(
                    f"Neither src ({src}) nor destination ({dst}) found in x_dict ({x_dict})"
                )
            out = self._internal_forward(
                x, edge_index_dict[edge_type], edge_type, **kwargs
            )
            out_dict.setdefault(dst, []).append(out)
        return self._group_output(out_dict, **kwargs)

    def _group_output(self, out_dict: Dict[str, List], **kwargs) -> Dict[str, Tensor]:
        aggregated: Dict[str, Tensor] = {}
        for key, value in out_dict.items():
            # `hetero_conv.group` does not yet support Aggregation modules
            if isinstance(self.aggr, Aggregation):
                out = torch.stack(value, dim=0)
                out = self.aggr(out, dim=0).squeeze(0)
            else:
                out = group(value, self.aggr)
            aggregated[key] = out
        return aggregated
