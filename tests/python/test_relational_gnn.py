from __future__ import annotations

import pytest
import torch

from relmo.models import FastRelationalGNN, LGANRelationalGNN, RelationalGNN
from relmo.models.hetero_mp import BatchedFanInMP, FanInMP
from relmo.ops import mp as mp_ops

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


def _map_modular_to_fast_fused_grad_keys(
    grads: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, grad in grads.items():
        if name.startswith("symbols_to_relations_mp.update_modules."):
            suffix = name[len("symbols_to_relations_mp.update_modules.") :]
            mapped = f"fast_fused_rel_layer_mp.update_modules.{suffix}"
        elif name.startswith("relations_to_symbols_mp.aggr."):
            suffix = name[len("relations_to_symbols_mp.aggr.") :]
            mapped = f"fast_fused_rel_layer_mp.aggr.{suffix}"
        else:
            mapped = name
        out[mapped] = grad
    return out


def _to_device(
    x_dict: dict[str, torch.Tensor],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    *,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    return (
        {k: v.to(device=device) for k, v in x_dict.items()},
        {k: v.to(device=device) for k, v in edge_index_dict.items()},
    )


def test_lgan_relational_gnn_with_custom_label() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    custom_tn = "custom_tn"
    custom_nn = "custom_nn"
    custom_rr = "custom_rr"
    x_dict, edge_index_dict = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 3, "rel_b": 2},
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relations=relation_dict,
        symbol_type=symbol_type,
        tn_label=custom_tn,
        nn_label=custom_nn,
        rr_label=custom_rr,
    )

    model = LGANRelationalGNN(
        embedding_size=8,
        num_layers=1,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
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


def test_lgan_relational_gnn_default_labels() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 3, "rel_b": 2},
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relations=relation_dict,
        symbol_type=symbol_type,
    )

    model = LGANRelationalGNN(
        embedding_size=4,
        num_layers=1,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
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
        relations=relation_dict,
        symbol_type=symbol_type,
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relations=relation_dict,
        symbol_type=symbol_type,
    )
    for edge_type in list(edge_index_dict.keys()):
        if edge_type[1] == "_lgan_nn_":
            edge_index_dict.pop(edge_type)

    model = LGANRelationalGNN(
        embedding_size=8,
        num_layers=1,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
    )

    with pytest.raises(ValueError, match="missing"):
        model(*clone_graph(x_dict, edge_index_dict))


def test_lgan_relational_gnn_batched_uses_batched_label_fanin_modules() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    model_batched = LGANRelationalGNN(
        embedding_size=4,
        num_layers=1,
        aggregation="sum",
        symbol_type_ids="_symbol_",
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    )
    model_modular = LGANRelationalGNN(
        embedding_size=4,
        num_layers=1,
        aggregation="sum",
        symbol_type_ids="_symbol_",
        relations=relation_dict,
        rel_layer_mode="modular",
    )

    assert isinstance(model_batched.tn_relations_to_symbols_mp, BatchedFanInMP)
    assert isinstance(model_batched.nn_relations_to_symbols_mp, BatchedFanInMP)
    assert isinstance(model_batched.rr_relations_to_relations_mp, BatchedFanInMP)
    assert isinstance(model_modular.tn_relations_to_symbols_mp, FanInMP)
    assert isinstance(model_modular.nn_relations_to_symbols_mp, FanInMP)
    assert isinstance(model_modular.rr_relations_to_relations_mp, FanInMP)


@pytest.mark.parametrize("aggr", ["sum", "logsumexp"])
def test_relational_gnn_batched_cached_matches_modular_forward_and_gradients(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 3},
    )

    torch.manual_seed(0)
    modular = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation=aggr,
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="modular",
    )
    torch.manual_seed(0)
    batched = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation=aggr,
        symbol_type_ids=symbol_type,
        relations=relation_dict,
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


@pytest.mark.parametrize("aggr", ["sum", "logsumexp"])
def test_relational_gnn_fast_fused_matches_modular_forward_and_gradients(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 3},
    )

    torch.manual_seed(0)
    modular = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation=aggr,
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="modular",
    )
    torch.manual_seed(0)
    fast = FastRelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation=aggr,
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        compile_forward=False,
    )
    fast.embedding_updater.load_state_dict(modular.embedding_updater.state_dict(), strict=True)
    fast.fast_fused_rel_layer_mp.update_modules.load_state_dict(
        modular.symbols_to_relations_mp.update_modules.state_dict(),  # type: ignore[attr-defined]
        strict=True,
    )
    fast.fast_fused_rel_layer_mp.aggr.load_state_dict(  # type: ignore[union-attr]
        modular.relations_to_symbols_mp.aggr.state_dict(),  # type: ignore[attr-defined]
        strict=True,
    )

    out_mod, _ = modular(*clone_graph(x_dict, edge_index_dict))
    out_fast, _ = fast(*clone_graph(x_dict, edge_index_dict))
    assert torch.allclose(out_mod[symbol_type], out_fast[symbol_type], atol=1e-6, rtol=1e-5)

    grad_mod = _run_backward(modular, *clone_graph(x_dict, edge_index_dict))
    grad_fast = _run_backward(fast, *clone_graph(x_dict, edge_index_dict))
    grad_mod = _map_modular_to_fast_fused_grad_keys(grad_mod)
    assert set(grad_mod.keys()) == set(grad_fast.keys())
    for name in grad_mod:
        assert torch.allclose(grad_mod[name], grad_fast[name], atol=1e-6, rtol=1e-5), name
    assert _nonzero_grads(modular) > 0
    assert _nonzero_grads(fast) > 0


def test_fast_relational_gnn_rejects_non_fast_mode() -> None:
    with pytest.raises(ValueError, match="only supports rel_layer_mode='fast_fused'"):
        FastRelationalGNN(
            embedding_size=8,
            num_layers=1,
            aggregation="sum",
            symbol_type_ids="_symbol_",
            relations={"rel_a": 2},
            rel_layer_mode="modular",
        )


@pytest.mark.parametrize("aggr", ["sum"])
def test_lgan_relational_gnn_batched_cached_matches_modular_forward_and_gradients(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict, edge_index_dict = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 3},
    )
    add_lgan_edges(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        relations=relation_dict,
        symbol_type=symbol_type,
    )

    torch.manual_seed(0)
    modular = LGANRelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation=aggr,
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="modular",
    )
    torch.manual_seed(0)
    batched = LGANRelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation=aggr,
        symbol_type_ids=symbol_type,
        relations=relation_dict,
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


def test_lgan_relational_gnn_batched_cached_rejects_logsumexp_string() -> None:
    with pytest.raises(ValueError, match="Could not resolve 'logsumexp'"):
        LGANRelationalGNN(
            embedding_size=8,
            num_layers=1,
            aggregation="logsumexp",
            symbol_type_ids="_symbol_",
            relations={"rel_a": 2, "rel_b": 1},
            rel_layer_mode="batched_cached",
        )


def test_relational_gnn_batched_cached_schema_churn_matches_fresh_model() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_a, e_a = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 2},
        num_symbols=6,
    )
    x_b, e_b = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 2, "rel_b": 5},
        num_symbols=5,
    )

    torch.manual_seed(123)
    model_a_then_b = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    )
    model_b_only = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    )
    model_b_only.load_state_dict(model_a_then_b.state_dict(), strict=True)

    model_a_then_b(*clone_graph(x_a, e_a))
    out_after_churn, _ = model_a_then_b(*clone_graph(x_b, e_b))
    out_fresh, _ = model_b_only(*clone_graph(x_b, e_b))

    assert torch.allclose(
        out_after_churn[symbol_type], out_fresh[symbol_type], atol=1e-6, rtol=1e-5
    )


def test_lgan_relational_gnn_batched_cached_schema_churn_matches_fresh_model() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_a, e_a = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 4, "rel_b": 2},
        num_symbols=6,
    )
    add_lgan_edges(
        x_dict=x_a,
        edge_index_dict=e_a,
        relations=relation_dict,
        symbol_type=symbol_type,
    )
    x_b, e_b = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 2, "rel_b": 5},
        num_symbols=5,
    )
    add_lgan_edges(
        x_dict=x_b,
        edge_index_dict=e_b,
        relations=relation_dict,
        symbol_type=symbol_type,
    )

    torch.manual_seed(123)
    model_a_then_b = LGANRelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    )
    model_b_only = LGANRelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    )
    model_b_only.load_state_dict(model_a_then_b.state_dict(), strict=True)

    model_a_then_b(*clone_graph(x_a, e_a))
    out_after_churn, _ = model_a_then_b(*clone_graph(x_b, e_b))
    out_fresh, _ = model_b_only(*clone_graph(x_b, e_b))

    assert torch.allclose(
        out_after_churn[symbol_type], out_fresh[symbol_type], atol=1e-6, rtol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only model parity check.")
def test_relational_gnn_batched_cached_cuda_custom_ops_matches_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not mp_ops.available():
        pytest.skip("mp custom ops are not available in this environment.")

    relation_dict = {"rel_a": 2, "rel_b": 1}
    symbol_type = "_symbol_"
    x_dict_cpu, edge_index_cpu = build_relation_graph(
        relations=relation_dict,
        symbol_type=symbol_type,
        relation_sizes={"rel_a": 5, "rel_b": 4},
        num_symbols=7,
    )
    x_dict, edge_index_dict = _to_device(
        x_dict_cpu, edge_index_cpu, device=torch.device("cuda")
    )

    torch.manual_seed(7)
    model_python = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    ).cuda()
    model_custom = RelationalGNN(
        embedding_size=8,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=symbol_type,
        relations=relation_dict,
        rel_layer_mode="batched_cached",
    ).cuda()
    model_custom.load_state_dict(model_python.state_dict(), strict=True)

    monkeypatch.setenv("RELM_MODELS_MP_OPS", "0")
    out_py, _ = model_python(*clone_graph(x_dict, edge_index_dict))
    grad_py = _run_backward(model_python, *clone_graph(x_dict, edge_index_dict))

    monkeypatch.setenv("RELM_MODELS_MP_OPS", "1")
    monkeypatch.setenv("RELM_MODELS_MP_FANIN", "1")
    monkeypatch.setenv("RELM_MODELS_MP_FANIN_BATCHED_EXPERIMENTAL", "1")
    monkeypatch.setenv("RELM_MODELS_MP_LOGSUMEXP", "0")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    out_custom, _ = model_custom(*clone_graph(x_dict, edge_index_dict))
    grad_custom = _run_backward(model_custom, *clone_graph(x_dict, edge_index_dict))

    assert torch.allclose(
        out_py[symbol_type], out_custom[symbol_type], atol=1e-6, rtol=1e-5
    )
    assert set(grad_py.keys()) == set(grad_custom.keys())
    for name in grad_py:
        assert torch.allclose(grad_py[name], grad_custom[name], atol=1e-6, rtol=1e-5), name
