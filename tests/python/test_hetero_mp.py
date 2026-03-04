from __future__ import annotations

import torch
from torch_geometric.nn import SumAggregation

from relm.models.hetero_mp import FanInMP, HeteroRouting, SelectMP


class _DummyRouting(HeteroRouting):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_x = None

    def _accepts_edge(self, edge_type):
        return True

    def _internal_forward(self, x, edges_index, edge_type, **kwargs):
        self.last_x = x
        return x


def _manual_fanin_sum(
    rel_x: torch.Tensor,
    edge_pos0: torch.Tensor,
    edge_pos1: torch.Tensor,
    *,
    embedding_size: int,
    dim_size: int,
) -> torch.Tensor:
    rel_flat = rel_x.view(-1, embedding_size)
    src0 = edge_pos0[0] * 2 + 0
    src1 = edge_pos1[0] * 2 + 1
    dst0 = edge_pos0[1]
    dst1 = edge_pos1[1]
    gathered = torch.cat(
        [rel_flat.index_select(0, src0), rel_flat.index_select(0, src1)],
        dim=0,
    )
    dst = torch.cat([dst0, dst1], dim=0)
    out = rel_x.new_zeros((dim_size, embedding_size))
    out.index_add_(0, dst, gathered)
    return out


def test_hetero_routing_unknown_aggr_keeps_string() -> None:
    routing = _DummyRouting(aggr="not_an_aggr")
    assert routing.aggr == "not_an_aggr"


def test_hetero_routing_forward_src_equals_dst_uses_tensor() -> None:
    routing = _DummyRouting(aggr="stack")
    x_dict = {"node": torch.zeros((2, 1))}
    edge_index_dict = {("node", "0", "node"): torch.tensor([[0], [1]])}
    routing.forward(x_dict, edge_index_dict)
    assert isinstance(routing.last_x, torch.Tensor)


def test_hetero_routing_forward_missing_x_raises() -> None:
    routing = _DummyRouting(aggr="stack")
    edge_index_dict = {("src", "0", "dst"): torch.tensor([[0], [1]])}
    try:
        routing.forward({}, edge_index_dict)
    except KeyError:
        return
    raise AssertionError("Expected KeyError for missing src and dst node types.")


def test_hetero_routing_group_output_aggregation() -> None:
    routing = _DummyRouting(aggr=SumAggregation())
    out = routing._group_output({"dst": [torch.ones((2, 3)), torch.zeros((2, 3))]})
    assert out["dst"].shape == (2, 3)
    assert torch.allclose(out["dst"], torch.ones((2, 3)))


def test_hetero_routing_group_output_cat() -> None:
    routing = _DummyRouting(aggr="cat")
    out = routing._group_output({"dst": [torch.ones((2, 1)), torch.zeros((2, 1))]})
    assert out["dst"].shape == (2, 2)


def test_select_mp_hidden2() -> None:
    hidden = 2
    selector = SelectMP(hidden)
    aggr = SumAggregation()

    # relation embeddings: each row is [arg0(2 dims), arg1(2 dims)]
    relation_x = torch.tensor([[3.0, 3.0, 4.0, 4.0], [1.0, 1.0, 2.0, 2.0]])
    dst_x = torch.zeros((4, hidden))
    edge_pos0 = torch.tensor([[0, 1], [2, 0]])  # rel0->obj2, rel1->obj0
    edge_pos1 = torch.tensor([[0, 1], [3, 1]])  # rel0->obj3, rel1->obj1

    def _aggregate(select_result):
        inputs, index, dim_size = select_result
        return aggr(inputs, index=index, ptr=None, dim_size=dim_size, dim=-2)

    out_pos0 = _aggregate(selector((relation_x, dst_x), edge_pos0, 0))
    out_pos1 = _aggregate(selector((relation_x, dst_x), edge_pos1, 1))

    assert torch.allclose(
        out_pos0,
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [3.0, 3.0], [0.0, 0.0]]),
    )
    assert torch.allclose(
        out_pos1,
        torch.tensor([[0.0, 0.0], [2.0, 2.0], [0.0, 0.0], [4.0, 4.0]]),
    )


def test_fanin_positional_matches_manual_and_gradients() -> None:
    embedding_size = 2
    fanin = FanInMP(
        embedding_size=embedding_size,
        dst_types=("obj",),
        aggr="sum",
        strict_filter_mode=True,
    )
    relation_x_ref = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.5, 0.0], [2.0, 1.0, -3.0, 2.0]],
        requires_grad=True,
    )
    relation_x_tst = relation_x_ref.detach().clone().requires_grad_(True)
    obj_x = torch.zeros((4, embedding_size))
    edge_pos0 = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    edge_pos1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_index_dict = {
        ("rel", "0", "obj"): edge_pos0,
        ("rel", "1", "obj"): edge_pos1,
    }

    out_model = fanin({"rel": relation_x_tst, "obj": obj_x}, edge_index_dict)["obj"]
    out_ref = _manual_fanin_sum(
        relation_x_ref,
        edge_pos0,
        edge_pos1,
        embedding_size=embedding_size,
        dim_size=obj_x.size(0),
    )
    assert torch.allclose(out_model, out_ref, atol=1e-6, rtol=1e-5)

    out_model.square().sum().backward()
    out_ref.square().sum().backward()
    assert torch.allclose(
        relation_x_tst.grad, relation_x_ref.grad, atol=1e-6, rtol=1e-5
    )
