from __future__ import annotations

import pytest
import torch

from relm.models import LGANRelationalGNN, RelationalGNN
from relm.models.hetero_mp import BatchedFanInMP, FanInMP

from ._graph_fixtures import add_lgan_edges, build_relation_graph, clone_graph


def _nonzero_grads(model: torch.nn.Module) -> int:
    count = 0
    for param in model.parameters():
        if not param.requires_grad or param.grad is None:
            continue
        if torch.isfinite(param.grad).all() and float(param.grad.abs().sum()) > 0.0:
            count += 1
    return count


def _run_backward(model: torch.nn.Module, x_dict, edge_index_dict) -> dict[str, torch.Tensor]:
    out, _ = model(x_dict, edge_index_dict)
    loss = torch.stack([value.square().mean() for value in out.values()]).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def test_lgan_relational_gnn_with_custom_label_ported() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    custom_tn = "custom_tn"
    custom_nn = "custom_nn"
    custom_rr = "custom_rr"
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 3, "rel_b": 2},
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relation_dict=relation_dict,
        symbol_type=symbol_type,
        tn_label=custom_tn,
        nn_label=custom_nn,
        rr_label=custom_rr,
    )

    model = LGANRelationalGNN(
        embedding_size=8,
        num_layer=1,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
        lgan_tn_edge_pos=custom_tn,
        lgan_nn_edge_pos=custom_nn,
        lgan_rr_edge_pos=custom_rr,
    ).eval()

    assert model.lgan_tn_edge_pos == custom_tn
    assert model.lgan_nn_edge_pos == custom_nn
    assert model.lgan_rr_edge_pos == custom_rr

    out, _ = model(*clone_graph(x_dict, edge_index_dict))
    assert symbol_type in out
    assert tuple(out[symbol_type].shape) == (x_dict[symbol_type].size(0), 8)


def test_lgan_relational_gnn_default_labels_ported() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 3, "rel_b": 2},
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relation_dict=relation_dict,
        symbol_type=symbol_type,
    )

    model = LGANRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
    ).eval()

    assert model.lgan_tn_edge_pos == "_lgan_tn_"
    assert model.lgan_nn_edge_pos == "_lgan_nn_"
    assert model.lgan_rr_edge_pos == "_lgan_rr_"

    out, _ = model(*clone_graph(x_dict, edge_index_dict))
    assert symbol_type in out
    assert tuple(out[symbol_type].shape) == (x_dict[symbol_type].size(0), 4)


def test_lgan_relational_gnn_requires_all_lgan_edge_families() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict=relation_dict,
        symbol_type=symbol_type,
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relation_dict=relation_dict,
        symbol_type=symbol_type,
    )
    for edge_type in list(edge_index_dict.keys()):
        if edge_type[1] == "_lgan_nn_":
            edge_index_dict.pop(edge_type)

    model = LGANRelationalGNN(
        embedding_size=8,
        num_layer=1,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
    )

    with pytest.raises(ValueError, match="missing"):
        model(*clone_graph(x_dict, edge_index_dict))


def test_lgan_relational_gnn_batched_uses_batched_label_fanin_modules() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    model_batched = LGANRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="_symbol_",
        relation_dict=relation_dict,
        rel_layer_mode="batched_cached",
    )
    model_modular = LGANRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="_symbol_",
        relation_dict=relation_dict,
        rel_layer_mode="modular",
    )

    assert isinstance(model_batched.tn_relations_to_symbols_mp, BatchedFanInMP)
    assert isinstance(model_batched.nn_relations_to_symbols_mp, BatchedFanInMP)
    assert isinstance(model_batched.rr_relations_to_relations_mp, BatchedFanInMP)
    assert isinstance(model_modular.tn_relations_to_symbols_mp, FanInMP)
    assert isinstance(model_modular.nn_relations_to_symbols_mp, FanInMP)
    assert isinstance(model_modular.rr_relations_to_relations_mp, FanInMP)


def test_relational_gnn_batched_cached_matches_modular_forward_and_gradients() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 3},
    )

    torch.manual_seed(0)
    modular = RelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
        rel_layer_mode="modular",
    )
    torch.manual_seed(0)
    batched = RelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
        rel_layer_mode="batched_cached",
    )
    batched.load_state_dict(modular.state_dict(), strict=True)

    out_mod, _ = modular(*clone_graph(x_dict, edge_index_dict))
    out_bat, _ = batched(*clone_graph(x_dict, edge_index_dict))
    assert torch.allclose(out_mod[symbol_type], out_bat[symbol_type], atol=1e-6, rtol=1e-5)

    grad_mod = _run_backward(modular, *clone_graph(x_dict, edge_index_dict))
    grad_bat = _run_backward(batched, *clone_graph(x_dict, edge_index_dict))
    assert set(grad_mod.keys()) == set(grad_bat.keys())
    for name in grad_mod:
        assert torch.allclose(grad_mod[name], grad_bat[name], atol=1e-6, rtol=1e-5), name
    assert _nonzero_grads(modular) > 0
    assert _nonzero_grads(batched) > 0


def test_lgan_relational_gnn_batched_cached_matches_modular_forward_and_gradients() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 3},
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relation_dict=relation_dict,
        symbol_type=symbol_type,
    )

    torch.manual_seed(0)
    modular = LGANRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
        rel_layer_mode="modular",
    )
    torch.manual_seed(0)
    batched = LGANRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        symbol_type_ids=symbol_type,
        relation_dict=relation_dict,
        rel_layer_mode="batched_cached",
    )
    batched.load_state_dict(modular.state_dict(), strict=True)

    out_mod, _ = modular(*clone_graph(x_dict, edge_index_dict))
    out_bat, _ = batched(*clone_graph(x_dict, edge_index_dict))
    assert torch.allclose(out_mod[symbol_type], out_bat[symbol_type], atol=1e-6, rtol=1e-5)

    grad_mod = _run_backward(modular, *clone_graph(x_dict, edge_index_dict))
    grad_bat = _run_backward(batched, *clone_graph(x_dict, edge_index_dict))
    assert set(grad_mod.keys()) == set(grad_bat.keys())
    for name in grad_mod:
        assert torch.allclose(grad_mod[name], grad_bat[name], atol=1e-6, rtol=1e-5), name
    assert _nonzero_grads(modular) > 0
    assert _nonzero_grads(batched) > 0
