from __future__ import annotations

import pytest
import torch

from relm.models import ArityMLPFactory, CentralizedRelationalGNN
from relm.models.film import CentralFiLMFactory, FiLMConcatMLP, FiLMConcatResMLP
from relm.models.hetero_mp import CentralFanOutMP
from relm.models.relational_gnn import (
    BoundedValueHead,
    CentralRelationModule,
    ZeroOut,
)
from relm.models.residual import ResidualModule

from ._graph_fixtures import build_relation_graph, clone_graph


class _CountingModule(torch.nn.Module):
    def __init__(self, out_size: int):
        super().__init__()
        self.out_size = int(out_size)
        self.calls = 0
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.last_input = x.detach().clone()
        return x[:, : self.out_size]


class _CaptureModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_input: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input = x.detach().clone()
        return x


def _build_symbol_inputs():
    x_dict = {
        "obj": torch.tensor([[1.0], [2.0]]),
        "relA": torch.zeros((1, 1)),
        "relB": torch.zeros((1, 2)),
    }
    edge_index_dict = {
        ("obj", "0", "relA"): torch.tensor([[0], [0]]),
        ("obj", "0", "relB"): torch.tensor([[0], [0]]),
        ("obj", "1", "relB"): torch.tensor([[1], [0]]),
        ("relA", "0", "obj"): torch.tensor([[0], [0]]),
        ("relB", "0", "obj"): torch.tensor([[0], [0]]),
        ("relB", "1", "obj"): torch.tensor([[0], [1]]),
    }
    return x_dict, edge_index_dict


def _build_relation_dict() -> dict[str, int]:
    return {"relA": 1, "relB": 2}


def test_central_fanout_batches_single_call_condition_pre_ported() -> None:
    embedding_size = 2
    cond_dim = 2
    relation_arities = {"relA": 1, "relB": 2}
    central = _CountingModule(out_size=embedding_size * max(relation_arities.values()))
    condition = torch.nn.Embedding(2, cond_dim)
    with torch.no_grad():
        condition.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    fanout = CentralFanOutMP(
        central_module=central,
        condition_embedding=condition,
        relation_condition_index={"relA": 0, "relB": 1},
        relation_arities=relation_arities,
        max_arity=2,
        embedding_size=embedding_size,
        condition_position="pre",
        src_types=("obj",),
        strict_filter_mode=True,
    )
    x_dict = {
        "obj": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        "relA": torch.zeros((1, embedding_size)),
        "relB": torch.zeros((1, embedding_size)),
    }
    _, edge_index_dict = _build_symbol_inputs()
    out = fanout(x_dict, edge_index_dict)

    assert central.calls == 1
    assert out["relA"].shape == (1, embedding_size)
    assert out["relB"].shape == (1, embedding_size * 2)
    assert central.last_input is not None
    assert central.last_input.shape == (2, embedding_size * 2 + cond_dim + 2)
    assert torch.allclose(central.last_input[0, :cond_dim], condition.weight[0])
    assert torch.allclose(central.last_input[1, :cond_dim], condition.weight[1])


def test_central_fanout_condition_post_ported() -> None:
    embedding_size = 2
    cond_dim = 2
    relation_arities = {"relA": 1, "relB": 2}
    central = _CountingModule(out_size=embedding_size * max(relation_arities.values()))
    condition = torch.nn.Embedding(2, cond_dim)
    with torch.no_grad():
        condition.weight.copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0]]))

    fanout = CentralFanOutMP(
        central_module=central,
        condition_embedding=condition,
        relation_condition_index={"relA": 0, "relB": 1},
        relation_arities=relation_arities,
        max_arity=2,
        embedding_size=embedding_size,
        condition_position="post",
        src_types=("obj",),
        strict_filter_mode=True,
    )
    x_dict = {
        "obj": torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        "relA": torch.zeros((1, embedding_size)),
        "relB": torch.zeros((1, embedding_size)),
    }
    _, edge_index_dict = _build_symbol_inputs()
    fanout(x_dict, edge_index_dict)

    assert central.last_input is not None
    assert torch.allclose(central.last_input[0, -cond_dim:], condition.weight[0])
    assert torch.allclose(central.last_input[1, -cond_dim:], condition.weight[1])


def test_central_fanout_raises_on_mismatched_max_arity_ported() -> None:
    with pytest.raises(ValueError):
        CentralFanOutMP(
            central_module=_CountingModule(out_size=2),
            condition_embedding=torch.nn.Embedding(2, 1),
            relation_condition_index={"relA": 0, "relB": 1},
            relation_arities={"relA": 1, "relB": 2},
            max_arity=1,
            embedding_size=2,
            condition_position="pre",
            src_types=("obj",),
            strict_filter_mode=True,
        )


def test_central_relation_module_pads_and_truncates_ported() -> None:
    capture = _CaptureModule()
    condition = torch.nn.Embedding(1, 1)
    with torch.no_grad():
        condition.weight.copy_(torch.tensor([[9.0]]))
    module = CentralRelationModule(
        central_module=capture,
        condition_embedding=condition,
        condition_index=0,
        arity=1,
        max_arity=2,
        embedding_size=2,
        condition_position="post",
        truncate_output=True,
    )
    out = module(torch.tensor([[1.0, 2.0]]))
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[1.0, 2.0]]))
    assert capture.last_input is not None
    assert capture.last_input.shape == (1, 5)


def test_central_relation_module_rejects_oversize_input_ported() -> None:
    module = CentralRelationModule(
        central_module=torch.nn.Identity(),
        condition_embedding=torch.nn.Embedding(1, 1),
        condition_index=0,
        arity=1,
        max_arity=1,
        embedding_size=2,
        condition_position="post",
    )
    with pytest.raises(ValueError):
        module(torch.zeros((1, 3)))


def test_centralized_rgnn_rejects_conflicting_central_args_ported() -> None:
    with pytest.raises(ValueError):
        CentralizedRelationalGNN(
            embedding_size=2,
            num_layer=1,
            aggr="sum",
            symbol_type_ids="obj",
            relation_dict=_build_relation_dict(),
            central_module=torch.nn.Identity(),
            central_module_factory=lambda *_: torch.nn.Identity(),
        )


def test_centralized_rgnn_rejects_invalid_condition_dim_ported() -> None:
    with pytest.raises(ValueError):
        CentralizedRelationalGNN(
            embedding_size=2,
            num_layer=1,
            aggr="sum",
            symbol_type_ids="obj",
            relation_dict=_build_relation_dict(),
            relation_condition_dim=0,
        )


def test_centralized_rgnn_condition_embedding_can_be_static_ported() -> None:
    model = CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        relation_condition_learnable=False,
    )
    assert model.relation_condition_embedding.weight.requires_grad is False


def test_centralized_rgnn_default_central_module_types_ported() -> None:
    model_residual = CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_residual=True,
    )
    assert isinstance(model_residual.central_module, ResidualModule)

    model_plain = CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_residual=False,
    )
    assert isinstance(model_plain.central_module, FiLMConcatMLP)


def test_centralized_rgnn_uses_relation_module_factory_alias_ported() -> None:
    calls: list[int] = []

    def factory(max_arity):
        calls.append(int(max_arity))
        return torch.nn.Identity()

    CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        relation_module_factory=factory,
    )
    assert calls == [2]


def test_centralized_rgnn_resolves_central_module_factories_ported() -> None:
    calls: list[tuple] = []

    def factory_one(max_arity):
        calls.append(("one", max_arity))
        return torch.nn.Identity()

    def factory_two(embedding_size, max_arity):
        calls.append(("two", embedding_size, max_arity))
        return torch.nn.Identity()

    def factory_three(embedding_size, max_arity, cond_dim):
        calls.append(("three", embedding_size, max_arity, cond_dim))
        return torch.nn.Identity()

    CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_module_factory=factory_one,
    )
    CentralizedRelationalGNN(
        embedding_size=3,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_module_factory=factory_two,
    )
    CentralizedRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_module_factory=factory_three,
    )

    assert calls[0] == ("one", 2)
    assert calls[1] == ("two", 3, 2)
    assert calls[2] == ("three", 4, 2, 2)


def test_centralized_rgnn_single_central_call_ported() -> None:
    relation_dict = _build_relation_dict()
    embedding_size = 2
    cond_dim = 2
    central = _CountingModule(out_size=embedding_size * 2)
    model = CentralizedRelationalGNN(
        embedding_size=embedding_size,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=relation_dict,
        relation_condition_dim=cond_dim,
        central_module=central,
        condition_position="pre",
        central_layer_mode="modular",
    )
    with torch.no_grad():
        model.relation_condition_embedding.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    x_dict, edge_index_dict = _build_symbol_inputs()
    model(*clone_graph(x_dict, edge_index_dict))

    assert central.calls == 1
    assert central.last_input is not None
    assert torch.allclose(central.last_input[0, :cond_dim], torch.tensor([1.0, 2.0]))
    assert torch.allclose(central.last_input[1, :cond_dim], torch.tensor([3.0, 4.0]))


def test_centralized_relational_gnn_condition_embedding_gets_gradients() -> None:
    relation_dict = {"relA": 1, "relB": 2}
    embedding_size = 4
    cond_dim = 2
    max_arity = max(relation_dict.values())
    central = torch.nn.Linear(cond_dim + max_arity * embedding_size, max_arity * embedding_size)
    model = CentralizedRelationalGNN(
        embedding_size=embedding_size,
        num_layer=2,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=relation_dict,
        relation_condition_dim=cond_dim,
        central_module=central,
        condition_position="pre",
        central_layer_mode="modular",
        central_slot_mask=False,
    )
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict=relation_dict,
        symbol_type="obj",
        relation_sizes={"relA": 4, "relB": 3},
        num_symbols=6,
    )
    out, _ = model(*clone_graph(x_dict, edge_index_dict))
    loss = torch.stack([value.square().mean() for value in out.values()]).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()

    assert model.relation_condition_embedding.weight.grad is not None
    assert torch.isfinite(model.relation_condition_embedding.weight.grad).all()
    assert float(model.relation_condition_embedding.weight.grad.abs().sum()) > 0.0
    assert model.central_module.weight.grad is not None
    assert float(model.central_module.weight.grad.abs().sum()) > 0.0


def test_centralized_relational_gnn_static_condition_embedding_stays_grad_free() -> None:
    model = CentralizedRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict={"relA": 1, "relB": 2},
        relation_condition_learnable=False,
        central_layer_mode="modular",
        central_slot_mask=False,
    )
    x_dict, edge_index_dict = build_relation_graph(
        relation_dict={"relA": 1, "relB": 2},
        symbol_type="obj",
        relation_sizes={"relA": 3, "relB": 2},
        num_symbols=4,
    )
    out, _ = model(*clone_graph(x_dict, edge_index_dict))
    loss = torch.stack([value.square().mean() for value in out.values()]).sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.relation_condition_embedding.weight.grad is None


def test_centralized_rgnn_uses_arity_factory_ported() -> None:
    factory = ArityMLPFactory(feature_size=2, layers=1)
    model = CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_module_factory=factory,
        central_slot_mask=False,
    )
    assert isinstance(model.central_module, torch.nn.Module)


def test_centralized_rgnn_accepts_central_film_factory_with_mask_ported() -> None:
    factory = CentralFiLMFactory(layers=["x2"])
    model = CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict=_build_relation_dict(),
        central_module_factory=factory,
        condition_position="post",
        central_slot_mask=True,
        central_residual=False,
    )
    assert isinstance(model.central_module, FiLMConcatResMLP)
    assert model.central_module.in_dim == 6  # max_arity*emb + max_arity mask = 4 + 2
    assert model.central_module.condition_position == "post"


def test_centralized_rgnn_default_zero_arity_module_ported() -> None:
    model = CentralizedRelationalGNN(
        embedding_size=2,
        num_layer=1,
        aggr="sum",
        symbol_type_ids="obj",
        relation_dict={"relZero": 0},
    )
    assert isinstance(model.central_module, ZeroOut)


def test_bounded_value_head_unbounded_path_ported() -> None:
    value_net = torch.nn.Linear(1, 1)
    with torch.no_grad():
        value_net.weight.fill_(2.0)
        value_net.bias.fill_(1.0)
    head = BoundedValueHead(value_net, lower_bound=None, upper_bound=1.0)
    x = torch.tensor([[3.0]])
    assert torch.allclose(head(x), value_net(x))


def test_bounded_value_head_bounded_path_ported() -> None:
    value_net = torch.nn.Linear(1, 1)
    with torch.no_grad():
        value_net.weight.fill_(0.0)
        value_net.bias.fill_(0.0)
    head = BoundedValueHead(value_net, lower_bound=-2.0, upper_bound=2.0)
    out = head(torch.tensor([[5.0]]))
    assert torch.allclose(out, torch.tensor([[0.0]]))
