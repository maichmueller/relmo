from __future__ import annotations

import pytest
import torch
from torch_geometric.nn import SumAggregation

import relmo.models.hetero_mp.batched as batched_impl
import relmo.models.hetero_mp._scatter as scatter_impl
from relmo.models.hetero_mp import (
    BatchedFanInMP,
    BatchedFanOutMP,
    FanInMP,
    HeteroRouting,
    SelectMP,
)


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


def _fanout_pack_edges_ref(
    x_parts,
    edge_src_parts,
    edge_dst_parts,
    src_part_ids,
    arity_parts,
    pos_parts,
    slot_offset_parts,
):
    x_cat = x_parts[0] if len(x_parts) == 1 else torch.cat(x_parts, dim=0)
    offsets = []
    offset = 0
    for x in x_parts:
        offsets.append(int(offset))
        offset += int(x.size(0))
    src_global_parts = []
    flat_dst_parts = []
    for edge_src, edge_dst, src_part, arity, pos, slot_offset in zip(
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    ):
        src_global_parts.append(edge_src + int(offsets[int(src_part)]))
        flat_dst_parts.append(int(slot_offset) + edge_dst * int(arity) + int(pos))
    src_global = (
        src_global_parts[0]
        if len(src_global_parts) == 1
        else torch.cat(src_global_parts, dim=0)
    )
    flat_dst = (
        flat_dst_parts[0] if len(flat_dst_parts) == 1 else torch.cat(flat_dst_parts, dim=0)
    )
    return x_cat, src_global, flat_dst


def _fanin_pack_edges_ref(
    rel_parts,
    edge_src_parts,
    edge_dst_parts,
    rel_part_ids,
    arity_parts,
    pos_parts,
    mode,
):
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    offsets = []
    offset = 0
    for rel in rel_parts:
        offsets.append(int(offset))
        offset += int(rel.size(0))
    flat_src_parts = []
    dst_parts = []
    for edge_src, edge_dst, rel_part, arity, pos in zip(
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
    ):
        if int(mode) == 1:
            local_src = edge_src
        else:
            local_src = edge_src * int(arity) + int(pos)
        flat_src_parts.append(local_src + int(offsets[int(rel_part)]))
        dst_parts.append(edge_dst)
    flat_src = (
        flat_src_parts[0] if len(flat_src_parts) == 1 else torch.cat(flat_src_parts, dim=0)
    )
    dst_idx = dst_parts[0] if len(dst_parts) == 1 else torch.cat(dst_parts, dim=0)
    return rel_cat, flat_src, dst_idx


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


def test_batched_fanin_uses_pyg_path_even_when_custom_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RaisingOps:
        @staticmethod
        def fanin_reduce(*_args, **_kwargs):
            raise AssertionError("decentralized BatchedFanInMP should not call custom kernels")

    monkeypatch.setattr(batched_impl, "relm_mp_ops", _RaisingOps())
    monkeypatch.setenv("RELM_MODELS_MP_OPS", "1")
    monkeypatch.setenv("RELM_MODELS_MP_FANIN", "1")
    monkeypatch.setenv("RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL", "0")

    embedding_size = 2
    mp = BatchedFanInMP(
        embedding_size=embedding_size,
        dst_types=("obj",),
        relation_arities={"rel": 2},
        aggr="sum",
        strict_filter_mode=True,
    )
    rel_x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.5, 0.0], [2.0, 1.0, -3.0, 2.0]]
    )
    obj_x = torch.zeros((4, embedding_size))
    edge_pos0 = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    edge_pos1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_index_dict = {
        ("rel", "0", "obj"): edge_pos0,
        ("rel", "1", "obj"): edge_pos1,
    }

    out = mp({"rel": rel_x, "obj": obj_x}, edge_index_dict)["obj"]
    ref = _manual_fanin_sum(
        rel_x,
        edge_pos0,
        edge_pos1,
        embedding_size=embedding_size,
        dim_size=obj_x.size(0),
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_batched_fanin_experimental_flag_uses_custom_kernel_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CountingOps:
        calls = 0

        @staticmethod
        def fanin_pack_from_edges(
            rel_parts,
            edge_src_parts,
            edge_dst_parts,
            rel_part_ids,
            arity_parts,
            pos_parts,
            mode,
        ):
            return _fanin_pack_edges_ref(
                rel_parts,
                edge_src_parts,
                edge_dst_parts,
                rel_part_ids,
                arity_parts,
                pos_parts,
                int(mode),
            )

        @staticmethod
        def fanin_reduce(rel_flat, flat_src, dst_idx, dim_size, mode):
            _CountingOps.calls += 1
            assert int(mode) == 0
            out = rel_flat.new_zeros((int(dim_size), int(rel_flat.size(1))))
            if int(flat_src.numel()) > 0 and int(dim_size) > 0:
                msgs = rel_flat.index_select(0, flat_src)
                out.index_add_(0, dst_idx, msgs)
            return out

    monkeypatch.setattr(batched_impl, "relm_mp_ops", _CountingOps())
    monkeypatch.setattr(
        batched_impl, "_use_model_mp_batched_fanin_reduce", lambda _ref: True
    )
    monkeypatch.setenv("RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL", "1")

    embedding_size = 2
    mp = BatchedFanInMP(
        embedding_size=embedding_size,
        dst_types=("obj",),
        relation_arities={"rel": 2},
        aggr="sum",
        strict_filter_mode=True,
    )
    rel_x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.5, 0.0], [2.0, 1.0, -3.0, 2.0]]
    )
    obj_x = torch.zeros((4, embedding_size))
    edge_pos0 = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    edge_pos1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_index_dict = {
        ("rel", "0", "obj"): edge_pos0,
        ("rel", "1", "obj"): edge_pos1,
    }

    out = mp({"rel": rel_x, "obj": obj_x}, edge_index_dict)["obj"]
    ref = _manual_fanin_sum(
        rel_x,
        edge_pos0,
        edge_pos1,
        embedding_size=embedding_size,
        dim_size=obj_x.size(0),
    )
    assert _CountingOps.calls > 0
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_batched_fanin_pack_only_path_skips_reduce_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PackOnlyOps:
        pack_calls = 0
        reduce_calls = 0

        @staticmethod
        def fanin_pack_from_edges(
            rel_parts,
            edge_src_parts,
            edge_dst_parts,
            rel_part_ids,
            arity_parts,
            pos_parts,
            mode,
        ):
            _PackOnlyOps.pack_calls += 1
            return _fanin_pack_edges_ref(
                rel_parts,
                edge_src_parts,
                edge_dst_parts,
                rel_part_ids,
                arity_parts,
                pos_parts,
                int(mode),
            )

        @staticmethod
        def fanin_pack_multi(rel_parts, flat_src_parts, dst_idx_parts):
            _PackOnlyOps.pack_calls += 1
            rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
            src_cat = (
                flat_src_parts[0]
                if len(flat_src_parts) == 1
                else torch.cat(flat_src_parts, dim=0)
            )
            dst_cat = (
                dst_idx_parts[0]
                if len(dst_idx_parts) == 1
                else torch.cat(dst_idx_parts, dim=0)
            )
            return rel_cat, src_cat, dst_cat

        @staticmethod
        def fanin_reduce(*_args, **_kwargs):
            _PackOnlyOps.reduce_calls += 1
            raise AssertionError("pack-only path must not call fanin_reduce")

    monkeypatch.setattr(batched_impl, "relm_mp_ops", _PackOnlyOps())
    monkeypatch.setattr(
        batched_impl, "_use_model_mp_batched_fanin_pack", lambda _ref: True
    )
    monkeypatch.setenv("RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL", "0")
    monkeypatch.setenv("RELM_MODELS_MP_FANIN_BATCHED_PACK_EXPERIMENTAL", "1")

    embedding_size = 2
    mp = BatchedFanInMP(
        embedding_size=embedding_size,
        dst_types=("obj",),
        relation_arities={"rel": 2},
        aggr="mean",
        strict_filter_mode=True,
    )
    rel_x = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, -1.0, 2.5, 0.0], [2.0, 1.0, -3.0, 2.0]]
    )
    obj_x = torch.zeros((4, embedding_size))
    edge_pos0 = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    edge_pos1 = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_index_dict = {
        ("rel", "0", "obj"): edge_pos0,
        ("rel", "1", "obj"): edge_pos1,
    }

    out = mp({"rel": rel_x, "obj": obj_x}, edge_index_dict)["obj"]
    ref = torch.stack(
        [
            rel_x[0, 0:2],
            rel_x[0, 2:4] + rel_x[1, 0:2],
            rel_x[1, 2:4] + rel_x[2, 0:2],
            rel_x[2, 2:4],
        ],
        dim=0,
    )
    ref = ref / torch.tensor([[1.0], [2.0], [2.0], [1.0]])
    assert _PackOnlyOps.pack_calls > 0
    assert _PackOnlyOps.reduce_calls == 0
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_batched_fanout_experimental_flag_uses_custom_kernel_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _CountingOps:
        calls = 0

        @staticmethod
        def fanout_pack_from_edges(
            x_parts,
            edge_src_parts,
            edge_dst_parts,
            src_part_ids,
            arity_parts,
            pos_parts,
            slot_offset_parts,
        ):
            return _fanout_pack_edges_ref(
                x_parts,
                edge_src_parts,
                edge_dst_parts,
                src_part_ids,
                arity_parts,
                pos_parts,
                slot_offset_parts,
            )

        @staticmethod
        def fanout_scatter(x_cat, src_global_idx, flat_dst, out_rows):
            _CountingOps.calls += 1
            out = x_cat.new_zeros((int(out_rows), int(x_cat.size(1))))
            if int(src_global_idx.numel()) > 0 and int(out_rows) > 0:
                vals = x_cat.index_select(0, src_global_idx)
                out.index_copy_(0, flat_dst, vals)
            return out

    monkeypatch.setattr(batched_impl, "relm_mp_ops", _CountingOps())
    monkeypatch.setattr(scatter_impl, "relm_mp_ops", _CountingOps())
    monkeypatch.setattr(batched_impl, "_use_model_mp_batched_fanout", lambda _ref: True)
    monkeypatch.setenv("RELM_MODELS_MP_FANOUT_BATCHED_EXPERIMENTAL", "1")

    mp = BatchedFanOutMP(
        update_modules={"rel": torch.nn.Identity()},
        relation_arities={"rel": 1},
        embedding_size=2,
        src_types=("obj",),
        strict_filter_mode=True,
    )
    x_dict = {
        "obj": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "rel": torch.zeros((1, 2)),
    }
    edge_index_dict = {
        ("obj", "0", "rel"): torch.tensor([[1], [0]], dtype=torch.long),
    }

    out = mp(x_dict, edge_index_dict)["rel"]
    assert _CountingOps.calls > 0
    assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))
