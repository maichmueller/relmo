from __future__ import annotations

from pathlib import Path
import copy

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation

from relm.ops import mp as relm_mp_module
from relm.models import FlatRelationalGNN
from relm.models import (
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    ThreeLayerPointwiseRelationMLP,
    TwoLayerPointwiseRelationMLP,
)
from relm.models import flat_relational_layer as flat_relational_layer_module
from relm.models.flat_relational_layer import build_flat_topology
from relm.models.grouped_mlp import GroupedMLPSpec
from relm.models.program_family_reference import (
    execute_program_two_layer_silu_then_two_layer_silu_reference,
)


class _SpecPostNormTwoLayerSiLU(torch.nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(width, hidden)
        self.act = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(hidden, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.lin2(self.act(self.lin1(x))))

    def relm_grouped_mlp_spec(self) -> GroupedMLPSpec:
        return GroupedMLPSpec(
            linears=[self.lin1, self.lin2],
            ops=[
                ("linear", 0),
                ("pointwise", self.act),
                ("linear", 1),
                ("norm", self.norm),
            ],
        )


class _SpecPreNormTwoLayerSiLURMS(torch.nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        rms_norm_cls = getattr(torch.nn, "RMSNorm", None)
        if rms_norm_cls is None:
            raise RuntimeError("RMSNorm is unavailable in this torch build.")
        self.norm = rms_norm_cls(width)
        self.lin1 = torch.nn.Linear(width, hidden)
        self.act = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(self.norm(x))))

    def relm_grouped_mlp_spec(self) -> GroupedMLPSpec:
        return GroupedMLPSpec(
            linears=[self.lin1, self.lin2],
            ops=[
                ("norm", self.norm),
                ("linear", 0),
                ("pointwise", self.act),
                ("linear", 1),
            ],
        )


def _make_family_model(
    module: torch.nn.Module,
    *,
    arity: int = 2,
    embedding_size: int = 4,
) -> tuple[FlatRelationalGNN, flat_relational_layer_module.RelationSlice]:
    relation_dict = {"rel": arity}
    model = FlatRelationalGNN(
        embedding_size=embedding_size,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=True,
        fused_relation_gather=False,
    )
    model.relational_layer.update_modules[0] = module
    relation_counts = torch.tensor([[1]], dtype=torch.long)
    relation_arities = torch.tensor([arity], dtype=torch.long)
    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    return model, topology.relation_slices[0]


def _replace_relation_modules(
    model: FlatRelationalGNN,
    modules: list[torch.nn.Module],
) -> None:
    model.relational_layer.update_modules = torch.nn.ModuleList(modules)


def _manual_sum_messages(
    x: torch.Tensor,
    relation_counts: torch.Tensor,
    relation_args: torch.Tensor,
    relation_arities: torch.Tensor,
    relation_modules: torch.nn.ModuleList,
) -> torch.Tensor:
    counts_2d = relation_counts if relation_counts.dim() == 2 else relation_counts.unsqueeze(0)
    counts_total = counts_2d.sum(dim=0)
    cursor = 0
    out = x.new_zeros(x.shape)
    embedding_size = int(x.size(1))
    for relation_index, (count_t, arity_t) in enumerate(zip(counts_total, relation_arities)):
        count = int(count_t.item())
        arity = int(arity_t.item())
        width = count * arity
        flat_idx = relation_args[cursor : cursor + width]
        if count > 0 and arity > 0:
            arg_emb = x.index_select(0, flat_idx)
            rel_in = arg_emb.view(count, arity * embedding_size)
            rel_out = relation_modules[relation_index](rel_in).view(-1, embedding_size)
            out.index_add_(0, flat_idx, arg_emb + rel_out)
        cursor += width
    return out


def _manual_mean_messages(
    x: torch.Tensor,
    relation_counts: torch.Tensor,
    relation_args: torch.Tensor,
    relation_arities: torch.Tensor,
    relation_modules: torch.nn.ModuleList,
) -> torch.Tensor:
    summed = _manual_sum_messages(
        x,
        relation_counts,
        relation_args,
        relation_arities,
        relation_modules,
    )
    counts_2d = relation_counts if relation_counts.dim() == 2 else relation_counts.unsqueeze(0)
    counts_total = counts_2d.sum(dim=0)
    cursor = 0
    counts = x.new_zeros((int(x.size(0)), 1))
    for count_t, arity_t in zip(counts_total, relation_arities):
        count = int(count_t.item())
        arity = int(arity_t.item())
        width = count * arity
        flat_idx = relation_args[cursor : cursor + width]
        if width > 0:
            counts.index_add_(
                0,
                flat_idx,
                torch.ones((width, 1), dtype=x.dtype, device=x.device),
            )
        cursor += width
    return summed / counts.clamp_min_(1.0)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_blocks_problem():
    pymimir = pytest.importorskip("pymimir")
    root = _repo_root() / "data" / "pddl_domains" / "blocks"
    domain = pymimir.Domain(root / "domain.pddl")
    problem = pymimir.Problem(domain, root / "probBLOCKS-4-0.pddl", mode="lifted")
    goals = list(problem.get_goal_condition().get_literals())
    state = problem.get_initial_state()
    return domain, state, goals


def _relation_dict_from_data(data: Data) -> dict[str, int]:
    arities = (
        data.relation_arities.tolist()
        if hasattr(data.relation_arities, "tolist")
        else list(data.relation_arities)
    )
    return {
        str(name): int(arity)
        for name, arity in zip(tuple(data.relation_names), arities)
    }


def test_build_flat_topology_counts_and_offsets() -> None:
    relation_counts = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    relation_arities = torch.tensor([2, 1], dtype=torch.long)

    topology = build_flat_topology(relation_counts, relation_arities)

    assert topology.relation_counts_total == (3, 3)
    assert topology.relation_arities == (2, 1)
    assert topology.slot_offsets == (0, 6, 9)
    assert topology.relation_slices[0].slot_start == 0
    assert topology.relation_slices[0].slot_end == 6
    assert topology.relation_slices[1].slot_start == 6
    assert topology.relation_slices[1].slot_end == 9


def test_flat_relational_layer_sum_matches_reference_forward_and_gradients() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
    )
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(5, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    relation_arities = torch.tensor([2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
    )
    ref = _manual_sum_messages(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities,
        ref_model.relational_layer.update_modules,
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)
    for (name, param), (name_ref, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        assert name == name_ref
        assert param.grad is not None
        assert param_ref.grad is not None
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-6, rtol=1e-5), name


def test_flat_relational_layer_mean_matches_reference_forward_and_gradients() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="mean",
        relation_dict=relation_dict,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="mean",
        relation_dict=relation_dict,
    )
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(5, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    relation_arities = torch.tensor([2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
    )
    ref = _manual_mean_messages(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities,
        ref_model.relational_layer.update_modules,
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)
    for (name, param), (name_ref, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        assert name == name_ref
        assert param.grad is not None
        assert param_ref.grad is not None
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-6, rtol=1e-5), name


def test_flat_relational_layer_accepts_pyg_aggregation_object() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=MeanAggregation(),
        relation_dict=relation_dict,
    )

    x = torch.randn(5, 4)
    relation_counts = torch.tensor([[2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    relation_arities = torch.tensor([2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
    )
    ref = _manual_mean_messages(
        x,
        relation_counts,
        relation_args,
        relation_arities,
        model.relational_layer.update_modules,
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)


def test_flat_relational_layer_fused_relation_gather_matches_direct_execution() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_relation_gather=True,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_relation_gather=False,
    )
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(5, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
    relation_arities = torch.tensor([2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


def test_flat_relational_layer_builds_fused_relation_specs_and_layout() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=True,
        fused_relation_gather=False,
    )
    model.relational_layer.update_modules[2] = torch.nn.Sequential(
        torch.nn.Linear(4, 8),
        torch.nn.GELU(),
        torch.nn.Linear(8, 4),
    )

    relation_counts = torch.tensor([[2, 1, 1]], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)
    topology = model.relational_layer.get_topology(relation_counts, relation_arities)

    match0 = model.relational_layer._match_fused_relation(topology.relation_slices[0])
    match1 = model.relational_layer._match_fused_relation(topology.relation_slices[1])
    match2 = model.relational_layer._match_fused_relation(topology.relation_slices[2])

    assert match0 is not None
    assert match1 is not None
    assert match2 is not None
    assert match0.spec.family == "two_layer_mish"
    assert match1.spec.family == "two_layer_mish"
    assert match2.spec.family == "two_layer_gelu"
    assert match0.spec.pointwise_signature is not None
    assert match0.spec.pointwise_signature[0] == "mish"

    layout = model.relational_layer._get_fused_relation_layout(topology)
    grouped_indices = {
        int(idx)
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_indices == {0, 1, 2}
    assert tuple(int(idx) for idx in layout["fallback_indices"]) == ()
    families_by_relation = {
        int(idx): batch.family
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert families_by_relation == {
        0: "two_layer_mish",
        1: "two_layer_mish",
        2: "two_layer_gelu",
    }
    mish_layout = model.relational_layer._get_fused_two_layer_mish_layout(topology)
    mish_indices = {
        int(idx)
        for batch in mish_layout["groups"]
        for idx in batch.relation_indices
    }
    assert mish_indices == {0, 1}


@pytest.mark.parametrize(
    ("module_factory", "expected_family"),
    [
        (
            lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="mish"),
            "two_layer_mish",
        ),
        (
            lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="silu"),
            "two_layer_silu",
        ),
        (
            lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="gelu"),
            "two_layer_gelu",
        ),
        (
            lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="relu"),
            "two_layer_relu",
        ),
        (
            lambda width: PreNormTwoLayerPointwiseRelationMLP(
                width, 16, activation="mish", norm="layernorm"
            ),
            "prenorm_two_layer_mish",
        ),
        (
            lambda width: PostNormTwoLayerPointwiseRelationMLP(
                width, 16, activation="silu", norm="layernorm"
            ),
            "postnorm_two_layer_silu",
        ),
        (
            lambda width: _SpecPostNormTwoLayerSiLU(width, 16),
            "postnorm_two_layer_silu",
        ),
        (
            lambda width: ThreeLayerPointwiseRelationMLP(
                width, 16, 12, activation="silu"
            ),
            "three_layer_silu",
        ),
    ],
)
def test_flat_relational_layer_matches_candidate_small_mlp_families(
    module_factory,
    expected_family: str,
) -> None:
    width = 8
    model, relation_slice = _make_family_model(module_factory(width), embedding_size=4, arity=2)
    match = model.relational_layer._match_fused_relation(relation_slice)
    assert match is not None
    assert match.spec.family == expected_family


@pytest.mark.skipif(
    getattr(torch.nn, "RMSNorm", None) is None,
    reason="RMSNorm unavailable in this torch build",
)
def test_flat_relational_layer_matches_rmsnorm_candidate_family() -> None:
    model, relation_slice = _make_family_model(
        PreNormTwoLayerPointwiseRelationMLP(
            width=8,
            hidden=16,
            activation="silu",
            norm="rmsnorm",
        ),
        embedding_size=4,
        arity=2,
    )
    match = model.relational_layer._match_fused_relation(relation_slice)
    assert match is not None
    assert match.spec.family == "prenorm_two_layer_silu_rmsnorm"
    assert match.spec.norm_kind == "rmsnorm"
    assert match.spec.norm_position == "pre"


def test_flat_relational_layer_matches_staged_two_layer_silu_program() -> None:
    model, relation_slice = _make_family_model(
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ),
        embedding_size=4,
        arity=2,
    )
    match = model.relational_layer._match_fused_relation(relation_slice)
    assert match is not None
    assert match.spec.family == "program"
    assert tuple(stage.spec.family for stage in match.program_matches) == (
        "two_layer_silu",
        "two_layer_silu",
    )
    assert match.program_family is not None
    assert match.program_family.family == "program_two_layer_silu_then_two_layer_silu"


def test_flat_relational_layer_non_exact_program_has_no_manual_program_family() -> None:
    model, relation_slice = _make_family_model(
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="gelu"),
        ),
        embedding_size=4,
        arity=2,
    )
    match = model.relational_layer._match_fused_relation(relation_slice)
    assert match is not None
    assert match.spec.family == "program"
    assert match.program_family is None


def test_manual_program_reference_matches_loop_reference_forward_and_gradients() -> None:
    def loop_reference(
        packed_rows: torch.Tensor,
        row_sizes: torch.Tensor,
        w10_stack: torch.Tensor,
        b10_stack: torch.Tensor,
        w20_stack: torch.Tensor,
        b20_stack: torch.Tensor,
        w11_stack: torch.Tensor,
        b11_stack: torch.Tensor,
        w21_stack: torch.Tensor,
        b21_stack: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(packed_rows)
        cursor = 0
        for gid, row_size_t in enumerate(row_sizes):
            row_count = int(row_size_t.item())
            rows = packed_rows[cursor : cursor + row_count]
            stage1 = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(rows, w10_stack[gid], b10_stack[gid])),
                w20_stack[gid],
                b20_stack[gid],
            )
            stage2 = torch.nn.functional.linear(
                torch.nn.functional.silu(torch.nn.functional.linear(stage1, w11_stack[gid], b11_stack[gid])),
                w21_stack[gid],
                b21_stack[gid],
            )
            out[cursor : cursor + row_count] = rows + stage2
            cursor += row_count
        return out

    torch.manual_seed(0)
    packed_rows = torch.randn(7, 8, requires_grad=True)
    packed_rows_ref = packed_rows.detach().clone().requires_grad_(True)
    row_sizes = torch.tensor([2, 3, 2], dtype=torch.long)
    params = [
        torch.randn(3, 16, 8, requires_grad=True),
        torch.randn(3, 16, requires_grad=True),
        torch.randn(3, 8, 16, requires_grad=True),
        torch.randn(3, 8, requires_grad=True),
        torch.randn(3, 12, 8, requires_grad=True),
        torch.randn(3, 12, requires_grad=True),
        torch.randn(3, 8, 12, requires_grad=True),
        torch.randn(3, 8, requires_grad=True),
    ]
    params_ref = [param.detach().clone().requires_grad_(True) for param in params]

    out = execute_program_two_layer_silu_then_two_layer_silu_reference(
        packed_rows,
        row_sizes,
        *params,
    )
    ref = loop_reference(
        packed_rows_ref,
        row_sizes,
        *params_ref,
    )

    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    out.square().sum().backward()
    ref.square().sum().backward()
    assert torch.allclose(packed_rows.grad, packed_rows_ref.grad, atol=1e-4, rtol=1e-4)
    for param, param_ref in zip(params, params_ref):
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-4, rtol=1e-4)


def test_flat_relational_layer_staged_program_is_opt_in_not_auto() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_relation_gather=False,
    )
    modules = [
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ),
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ),
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(4, 12, activation="silu"),
            TwoLayerPointwiseRelationMLP(4, 10, activation="silu"),
        ),
    ]
    _replace_relation_modules(model, modules)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)
    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_relation_layout(topology)

    assert layout["groups"] == ()
    assert layout["fallback_indices"] == (0, 1, 2)


def test_relation_mlp_stack_modules_expose_grouped_specs() -> None:
    modules = [
        TwoLayerPointwiseRelationMLP(8, 16, activation="mish"),
        TwoLayerPointwiseRelationMLP(8, 16, activation="gelu", gelu_approximate="tanh"),
        PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="layernorm"),
        PostNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="layernorm"),
        ThreeLayerPointwiseRelationMLP(8, 16, 12, activation="silu"),
    ]
    for module in modules:
        spec = module.relm_grouped_mlp_spec()
        assert isinstance(spec, GroupedMLPSpec)
        assert len(spec.linears) >= 2
        assert len(spec.ops) >= 3


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.Dropout(p=0.1),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
            torch.nn.LayerNorm(7),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
            torch.nn.Tanh(),
        ),
    ],
)
def test_flat_relational_layer_rejects_unsupported_or_misaligned_families(
    module: torch.nn.Module,
) -> None:
    model, relation_slice = _make_family_model(module, embedding_size=4, arity=2)
    assert model.relational_layer._match_fused_relation(relation_slice) is None


@pytest.mark.parametrize("aggr", ["sum", "mean"])
def test_flat_relational_layer_fused_two_layer_mish_matches_direct_execution(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=True,
        fused_relation_gather=False,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=False,
        fused_relation_gather=False,
    )
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(6, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_two_layer_mish_layout(topology)
    grouped_indices = {
        int(idx)
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_indices == {0, 1, 2}

    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("aggr", ["sum", "mean"])
def test_flat_relational_layer_fused_two_layer_silu_matches_direct_execution(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=True,
        fused_relation_gather=False,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=False,
        fused_relation_gather=False,
    )

    modules = [
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.SiLU(),
            torch.nn.Linear(12, 4),
        ),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_relation_layout(topology)
    grouped_families = {
        int(idx): batch.family
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_families == {
        0: "two_layer_silu",
        1: "two_layer_silu",
        2: "two_layer_silu",
    }

    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("aggr", ["sum", "mean"])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_flat_relational_layer_fused_two_layer_gelu_matches_direct_execution(
    aggr: str,
    approximate: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    )

    modules = [
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(approximate=approximate),
            torch.nn.Linear(16, 8),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(approximate=approximate),
            torch.nn.Linear(16, 8),
        ),
        torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.GELU(approximate=approximate),
            torch.nn.Linear(12, 4),
        ),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_relation_layout(topology)
    grouped_families = {
        int(idx): batch.family
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_families == {
        0: "two_layer_gelu",
        1: "two_layer_gelu",
        2: "two_layer_gelu",
    }

    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


def test_flat_relational_layer_accepts_pointwise_flag_alias() -> None:
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=1,
        aggr="sum",
        relation_dict={"rel": 1},
        fused_two_layer_pointwise_execution=True,
    )
    assert model.relational_layer.fused_two_layer_pointwise_execution is True
    assert model.relational_layer.fused_two_layer_mish_execution is True


@pytest.mark.parametrize("aggr", ["sum", "mean"])
def test_flat_relational_layer_fused_postnorm_two_layer_silu_matches_direct_execution(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    )

    modules = [
        PostNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="layernorm"),
        PostNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="layernorm"),
        PostNormTwoLayerPointwiseRelationMLP(4, 12, activation="silu", norm="layernorm"),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_relation_layout(topology)
    grouped_families = {
        int(idx): batch.family
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_families == {
        0: "postnorm_two_layer_silu",
        1: "postnorm_two_layer_silu",
        2: "postnorm_two_layer_silu",
    }

    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(
    getattr(torch.nn, "RMSNorm", None) is None,
    reason="RMSNorm unavailable in this torch build",
)
@pytest.mark.parametrize("aggr", ["sum", "mean"])
def test_flat_relational_layer_fused_prenorm_two_layer_silu_rmsnorm_matches_direct_execution(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    )

    modules = [
        PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm"),
        PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm"),
        PreNormTwoLayerPointwiseRelationMLP(4, 12, activation="silu", norm="rmsnorm"),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_relation_layout(topology)
    grouped_families = {
        int(idx): batch.family
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_families == {
        0: "prenorm_two_layer_silu_rmsnorm",
        1: "prenorm_two_layer_silu_rmsnorm",
        2: "prenorm_two_layer_silu_rmsnorm",
    }

    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("aggr", ["sum", "mean"])
def test_flat_relational_layer_fused_staged_two_layer_silu_program_matches_direct_execution(
    aggr: str,
) -> None:
    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    )
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr=aggr,
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    )

    modules = [
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ),
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ),
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(4, 12, activation="silu"),
            TwoLayerPointwiseRelationMLP(4, 10, activation="silu"),
        ),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long)
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long)
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long)

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    layout = model.relational_layer._get_fused_relation_layout(topology)
    grouped_families = {
        int(idx): batch.family
        for batch in layout["groups"]
        for idx in batch.relation_indices
    }
    assert grouped_families == {
        0: "program",
        1: "program",
        2: "program",
    }

    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flat_relational_layer_fused_two_layer_mish_cuda_custom_backward_matches_direct() -> None:
    if not relm_mp_module.available():
        pytest.skip("Custom mp ops are unavailable.")

    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=True,
        fused_relation_gather=False,
    ).cuda()
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=False,
        fused_relation_gather=False,
    ).cuda()
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long, device="cuda")
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long, device="cuda")
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (_, param), (_, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        if param.grad is None or param_ref.grad is None:
            continue
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flat_relational_layer_fused_two_layer_silu_cuda_custom_backward_matches_direct() -> None:
    if not relm_mp_module.available():
        pytest.skip("Custom mp ops are unavailable.")

    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=True,
        fused_relation_gather=False,
    ).cuda()
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_mish_execution=False,
        fused_relation_gather=False,
    ).cuda()
    modules = [
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        ).cuda(),
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 8),
        ).cuda(),
        torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.SiLU(),
            torch.nn.Linear(12, 4),
        ).cuda(),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long, device="cuda")
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long, device="cuda")
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (_, param), (_, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        if param.grad is None or param_ref.grad is None:
            continue
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_flat_relational_layer_fused_two_layer_gelu_cuda_custom_backward_matches_direct(
    approximate: str,
) -> None:
    if not relm_mp_module.available():
        pytest.skip("Custom mp ops are unavailable.")

    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    ).cuda()
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    ).cuda()
    modules = [
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(approximate=approximate),
            torch.nn.Linear(16, 8),
        ).cuda(),
        torch.nn.Sequential(
            torch.nn.Linear(8, 16),
            torch.nn.GELU(approximate=approximate),
            torch.nn.Linear(16, 8),
        ).cuda(),
        torch.nn.Sequential(
            torch.nn.Linear(4, 12),
            torch.nn.GELU(approximate=approximate),
            torch.nn.Linear(12, 4),
        ).cuda(),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long, device="cuda")
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long, device="cuda")
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (_, param), (_, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        if param.grad is None or param_ref.grad is None:
            continue
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flat_relational_layer_fused_postnorm_two_layer_silu_cuda_custom_backward_matches_direct() -> None:
    if not relm_mp_module.available():
        pytest.skip("Custom mp ops are unavailable.")

    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    ).cuda()
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    ).cuda()
    modules = [
        PostNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="layernorm").cuda(),
        PostNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="layernorm").cuda(),
        PostNormTwoLayerPointwiseRelationMLP(4, 12, activation="silu", norm="layernorm").cuda(),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long, device="cuda")
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long, device="cuda")
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (_, param), (_, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        if param.grad is None or param_ref.grad is None:
            continue
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(
    getattr(torch.nn, "RMSNorm", None) is None,
    reason="RMSNorm unavailable in this torch build",
)
def test_flat_relational_layer_fused_prenorm_two_layer_silu_rmsnorm_cuda_custom_backward_matches_direct() -> None:
    if not relm_mp_module.available():
        pytest.skip("Custom mp ops are unavailable.")

    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    ).cuda()
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    ).cuda()
    modules = [
        PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm").cuda(),
        PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm").cuda(),
        PreNormTwoLayerPointwiseRelationMLP(4, 12, activation="silu", norm="rmsnorm").cuda(),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long, device="cuda")
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long, device="cuda")
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (_, param), (_, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        if param.grad is None or param_ref.grad is None:
            continue
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flat_relational_layer_fused_staged_two_layer_silu_program_cuda_custom_backward_matches_direct() -> None:
    if not relm_mp_module.available():
        pytest.skip("Custom mp ops are unavailable.")

    relation_dict = {"rel_a": 2, "rel_b": 2, "rel_c": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=True,
        fused_relation_gather=False,
    ).cuda()
    torch.manual_seed(0)
    ref_model = FlatRelationalGNN(
        embedding_size=4,
        num_layer=1,
        aggr="sum",
        relation_dict=relation_dict,
        fused_two_layer_pointwise_execution=False,
        fused_relation_gather=False,
    ).cuda()
    modules = [
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ).cuda(),
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        ).cuda(),
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(4, 12, activation="silu"),
            TwoLayerPointwiseRelationMLP(4, 10, activation="silu"),
        ).cuda(),
    ]
    _replace_relation_modules(model, modules)
    _replace_relation_modules(ref_model, copy.deepcopy(modules))

    x = torch.randn(6, 4, device="cuda", requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2, 2, 1]], dtype=torch.long, device="cuda")
    relation_args = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2], dtype=torch.long, device="cuda")
    relation_arities = torch.tensor([2, 2, 1], dtype=torch.long, device="cuda")

    topology = model.relational_layer.get_topology(relation_counts, relation_arities)
    out = model.relational_layer(
        x,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    ref = ref_model.relational_layer(
        x_ref,
        relation_counts,
        relation_args,
        relation_arities=relation_arities,
        topology=topology,
        cache={},
    )
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    loss = out.square().sum()
    loss_ref = ref.square().sum()
    loss.backward()
    loss_ref.backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-5)
    for (_, param), (_, param_ref) in zip(
        model.relational_layer.named_parameters(),
        ref_model.relational_layer.named_parameters(),
    ):
        if param.grad is None or param_ref.grad is None:
            continue
        assert torch.allclose(param.grad, param_ref.grad, atol=1e-5, rtol=1e-5)


def test_flat_relational_gnn_normalizes_single_graph_counts_and_reads_data() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=relation_dict,
    ).eval()

    data = Data(
        x=torch.zeros((5, 1), dtype=torch.float),
        relation_counts=torch.tensor([2, 1], dtype=torch.long),
        relation_args=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        relation_arities=torch.tensor([2, 1], dtype=torch.long),
        node_sizes=torch.tensor([5], dtype=torch.long),
        object_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        target_entity_indices=torch.tensor([1, 3], dtype=torch.long),
        target_positions=torch.tensor([0, 2], dtype=torch.long),
    )

    out, out_batch = model(data)

    assert tuple(out["entity"].shape) == (5, 8)
    assert tuple(out["object"].shape) == (5, 8)
    assert tuple(out["target_entity"].shape) == (2, 8)
    assert tuple(out["target"].shape) == (2, 8)
    assert torch.equal(out_batch["entity"], torch.zeros(5, dtype=torch.long))
    assert torch.equal(out_batch["target_entity"], torch.zeros(2, dtype=torch.long))


def test_flat_relational_gnn_single_and_batched_relation_count_inputs_match() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=relation_dict,
    ).eval()

    kwargs = dict(
        relation_args=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        relation_arities=torch.tensor([2, 1], dtype=torch.long),
        node_sizes=torch.tensor([5], dtype=torch.long),
        object_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        target_entity_indices=torch.tensor([1, 3], dtype=torch.long),
        target_positions=torch.tensor([0, 2], dtype=torch.long),
    )
    x = torch.zeros((5, 1), dtype=torch.float)
    out_single, batch_single = model(
        x,
        torch.tensor([2, 1], dtype=torch.long),
        **kwargs,
    )
    out_batch, batch_batch = model(
        x,
        torch.tensor([[2, 1]], dtype=torch.long),
        **kwargs,
    )

    for key in out_single:
        assert torch.allclose(out_single[key], out_batch[key], atol=1e-6, rtol=1e-5), key
        assert torch.equal(batch_single[key], batch_batch[key]), key


def test_flat_relational_gnn_prepare_matches_raw_forward() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=relation_dict,
    ).eval()

    data = Data(
        x=torch.zeros((5, 1), dtype=torch.float),
        relation_counts=torch.tensor([2, 1], dtype=torch.long),
        relation_args=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        relation_arities=torch.tensor([2, 1], dtype=torch.long),
        node_sizes=torch.tensor([5], dtype=torch.long),
        object_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        target_entity_indices=torch.tensor([1, 3], dtype=torch.long),
        target_positions=torch.tensor([0, 2], dtype=torch.long),
    )

    prepared = model.prepare(data)
    out_raw, batch_raw = model(data)
    out_prepared, batch_prepared = model(prepared)

    assert prepared.topology is not None
    for key in out_raw:
        assert torch.allclose(out_raw[key], out_prepared[key], atol=1e-6, rtol=1e-5), key
        assert torch.equal(batch_raw[key], batch_prepared[key]), key


def test_flat_relational_gnn_forward_prepared_entity_embeddings_matches_entity_output() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    torch.manual_seed(0)
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=relation_dict,
    ).eval()

    data = Data(
        x=torch.zeros((5, 1), dtype=torch.float),
        relation_counts=torch.tensor([2, 1], dtype=torch.long),
        relation_args=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        relation_arities=torch.tensor([2, 1], dtype=torch.long),
        node_sizes=torch.tensor([5], dtype=torch.long),
        object_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        target_entity_indices=torch.tensor([1, 3], dtype=torch.long),
        target_positions=torch.tensor([0, 2], dtype=torch.long),
    )

    prepared = model.prepare(data)
    entity = model.forward_prepared_entity_embeddings(prepared)
    out, _out_batch = model(prepared)

    assert torch.allclose(entity, out["entity"], atol=1e-6, rtol=1e-5)


def test_flat_relational_gnn_prepare_prefers_int32_indices_on_cpu() -> None:
    relation_dict = {"rel_a": 2, "rel_b": 1}
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=relation_dict,
    ).eval()

    data = Data(
        x=torch.zeros((5, 1), dtype=torch.float),
        relation_counts=torch.tensor([2, 1], dtype=torch.long),
        relation_args=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        relation_arities=torch.tensor([2, 1], dtype=torch.long),
        node_sizes=torch.tensor([5], dtype=torch.long),
        object_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        target_entity_indices=torch.tensor([1, 3], dtype=torch.long),
        target_positions=torch.tensor([0, 2], dtype=torch.long),
    )

    prepared = model.prepare(data)

    assert prepared.relation_args.dtype == torch.int32
    assert prepared.object_indices is not None
    assert prepared.target_entity_indices is not None
    assert prepared.target_positions is not None
    assert prepared.object_indices.dtype == torch.int32
    assert prepared.target_entity_indices.dtype == torch.int32
    assert prepared.target_positions.dtype == torch.int32


def test_flat_topology_matches_real_encoder_helper_offsets() -> None:
    mifrost = pytest.importorskip("mifrost")
    domain, state, goals = _load_blocks_problem()
    encoder = mifrost.FlatRelationEncoder(domain)
    data = encoder.encode_pyg(state, goals=goals)

    topology = build_flat_topology(data.relation_counts, data.relation_arities)

    assert topology.relation_counts_total == tuple(
        int(x) for x in data.relation_instance_counts_total().tolist()
    )
    assert topology.slot_offsets == tuple(int(x) for x in data.relation_slot_offsets().tolist())


def test_flat_relational_gnn_accepts_real_encoder_single_graph_data() -> None:
    mifrost = pytest.importorskip("mifrost")
    domain, state, goals = _load_blocks_problem()
    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal"])
    data = encoder.encode_pyg(state, goals=goals)
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=_relation_dict_from_data(data),
    ).eval()

    out, out_batch = model(data)

    assert tuple(out["entity"].shape) == (int(data.x.size(0)), 8)
    assert tuple(out["object"].shape) == (int(data.object_indices.numel()), 8)
    assert tuple(out["target_entity"].shape) == (
        int(data.target_entity_indices.numel()),
        8,
    )
    assert tuple(out["target"].shape) == (int(data.target_positions.numel()), 8)
    assert torch.equal(out_batch["entity"], torch.zeros(int(data.x.size(0)), dtype=torch.long))
    assert torch.equal(
        out_batch["target"],
        torch.zeros(int(data.target_positions.numel()), dtype=torch.long),
    )


def test_flat_relational_gnn_accepts_native_batchencoding_and_pyg_batch() -> None:
    mifrost = pytest.importorskip("mifrost")
    domain, state, goals = _load_blocks_problem()
    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal"])
    native_batch = encoder.encode_batch(states=[state, state], goals=goals)
    pyg_batch = native_batch.as_pyg(as_batch=True)
    model = FlatRelationalGNN(
        embedding_size=8,
        num_layer=2,
        aggr="sum",
        relation_dict=_relation_dict_from_data(pyg_batch),
    ).eval()

    out_native, batch_native = model(native_batch)
    out_pyg, batch_pyg = model(pyg_batch)
    prepared = model.prepare(native_batch)
    out_prepared, batch_prepared = model(prepared)

    for key in out_native:
        assert torch.allclose(out_native[key], out_pyg[key], atol=1e-6, rtol=1e-5), key
        assert torch.equal(batch_native[key], batch_pyg[key]), key
        assert torch.allclose(out_native[key], out_prepared[key], atol=1e-6, rtol=1e-5), key
        assert torch.equal(batch_native[key], batch_prepared[key]), key
