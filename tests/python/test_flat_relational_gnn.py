from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.nn.aggr import MeanAggregation

from relmo.models import (
    CentralizedFlatRelationalGNN,
    FlatExecutionPolicy,
    FlatRelationalGNN,
    FlatRelationalOutput,
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    RelationBlockSpec,
    RelationProgram,
    ThreeLayerPointwiseRelationMLP,
    TwoLayerPointwiseRelationMLP,
)
from relmo.models import flat_relational_layer as flat_relational_layer_module
from tests.python.support.program_family_reference import (
    execute_program_prenorm_two_layer_silu_rmsnorm_then_two_layer_silu_reference,
    execute_program_two_layer_silu_then_postnorm_two_layer_silu_reference,
    execute_program_two_layer_silu_then_two_layer_silu_reference,
)


class _CustomSpecPostNormTwoLayerSiLU(torch.nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.width = int(width)
        self.lin1 = torch.nn.Linear(width, hidden)
        self.act = torch.nn.SiLU()
        self.lin2 = torch.nn.Linear(hidden, width)
        self.norm = torch.nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.lin2(self.act(self.lin1(x))))

    def relmo_kernel_spec(self) -> RelationBlockSpec:
        return RelationBlockSpec(
            linears=[self.lin1, self.lin2],
            ops=[
                ("linear", 0),
                ("pointwise", self.act),
                ("linear", 1),
                ("norm", self.norm),
            ],
        )


class _UnsupportedCustomBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.lin = torch.nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.lin(x))


class _CountingCentralModule(torch.nn.Module):
    def __init__(self, out_size: int) -> None:
        super().__init__()
        self.out_size = int(out_size)
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return x[:, : self.out_size]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_blocks_problem():
    pymimir = pytest.importorskip("pymimir")
    root = _repo_root() / "data" / "pddl_domains" / "blocks"
    domain = pymimir.Domain(root / "domain.pddl")
    problem_files = sorted(
        path for path in root.glob("*.pddl") if path.name != "domain.pddl"
    )
    if not problem_files:
        pytest.skip("no Blocks problem files available in test fixture root")
    problem = pymimir.Problem(domain, problem_files[0], mode="lifted")
    goals = list(problem.get_goal_condition().get_literals())
    state = problem.get_initial_state()
    return domain, state, goals


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
    summed = _manual_sum_messages(x, relation_counts, relation_args, relation_arities, relation_modules)
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
            counts.index_add_(0, flat_idx, torch.ones((width, 1), dtype=x.dtype, device=x.device))
        cursor += width
    return summed / counts.clamp_min_(1.0)


def _make_small_payload(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    device = torch.device(device)
    return {
        "x": torch.zeros((5, 1), device=device),
        "relation_counts": torch.tensor([[2, 1]], dtype=torch.long, device=device),
        "relation_args": torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device),
        "relation_arities": torch.tensor([2, 1], dtype=torch.long, device=device),
        "node_sizes": torch.tensor([5], dtype=torch.long, device=device),
        "object_indices": torch.tensor([0, 1, 3], dtype=torch.long, device=device),
        "target_entity_indices": torch.tensor([1, 4], dtype=torch.long, device=device),
        "target_positions": torch.tensor([0, 4], dtype=torch.long, device=device),
    }


def _make_small_data(device: torch.device | str = "cpu") -> Data:
    payload = _make_small_payload(device=device)
    data = Data(
        x=payload["x"],
        relation_counts=payload["relation_counts"],
        relation_args=payload["relation_args"],
        relation_arities=payload["relation_arities"],
    )
    for key, value in payload.items():
        if key in {"x", "relation_counts", "relation_args", "relation_arities"}:
            continue
        setattr(data, key, value)
    return data


def _make_model(
    relation_modules,
    *,
    aggregation: str | torch.nn.Module | None = "sum",
    execution_policy: FlatExecutionPolicy | None = None,
) -> FlatRelationalGNN:
    if execution_policy is None:
        execution_policy = FlatExecutionPolicy(relation_gather="off")
    return FlatRelationalGNN(
        embedding_size=4,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        aggregation=aggregation,
        relation_modules=relation_modules,
        execution_policy=execution_policy,
    )


def _make_program_model(
    module: torch.nn.Module,
    *,
    arity: int = 2,
    relation_kernels=None,
) -> tuple[FlatRelationalGNN, flat_relational_layer_module.RelationSlice]:
    model = FlatRelationalGNN(
        embedding_size=4,
        num_layers=1,
        relations={"rel": arity},
        relation_modules={"rel": module},
        relation_kernels=relation_kernels,
        execution_policy=FlatExecutionPolicy(relation_kernels="auto", program_kernels="auto", relation_gather="off"),
    )
    topology = model.relational_layer.get_topology(torch.tensor([[1]]), relation_arities=torch.tensor([arity]))
    return model, topology.relation_slices[0]


def test_flat_execution_policy_resolves_modes() -> None:
    policy = FlatExecutionPolicy(relation_kernels="auto", program_kernels="auto", relation_gather="auto")
    assert policy.use_relation_kernels(device=torch.device("cpu")) is False
    assert policy.use_program_kernels(device=torch.device("cpu")) is False
    assert policy.use_relation_gather(device=torch.device("cpu")) is False
    assert policy.use_relation_gather(device=torch.device("cuda")) is True


def test_flat_execution_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        FlatExecutionPolicy(relation_kernels="on")
    with pytest.raises(ValueError):
        FlatExecutionPolicy(program_kernels="maybe")
    with pytest.raises(ValueError):
        FlatExecutionPolicy(relation_gather="maybe")


def test_compile_boundary_stays_on_prepared_core_only() -> None:
    assert hasattr(FlatRelationalGNN.compute_entity_embeddings, "__wrapped__")
    assert hasattr(FlatRelationalGNN.forward, "__wrapped__")
    assert hasattr(FlatRelationalGNN._compute_entity_embeddings_prepared, "__wrapped__")
    assert hasattr(FlatRelationalGNN._forward_prepared_batch, "__wrapped__")
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 8), "rel_b": TwoLayerPointwiseRelationMLP(4, 4)})
    assert model._compile_forward is False
    assert model._compile_public_api is False


def test_relation_program_validates_widths() -> None:
    with pytest.raises(ValueError):
        RelationProgram(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(12, 16, activation="silu"),
        )


def test_constructor_rejects_conflicting_relation_module_sources() -> None:
    with pytest.raises(ValueError):
        FlatRelationalGNN(
            embedding_size=4,
            num_layers=1,
            relations={"rel": 2},
            relation_modules={"rel": TwoLayerPointwiseRelationMLP(8, 16)},
            relation_module_factory=lambda arity: TwoLayerPointwiseRelationMLP(arity * 4, 16),
        )


def test_centralized_flat_rejects_conflicting_central_module_sources() -> None:
    with pytest.raises(ValueError):
        CentralizedFlatRelationalGNN(
            embedding_size=4,
            num_layers=1,
            relations={"rel_a": 2},
            central_module=torch.nn.Identity(),
            central_module_factory=lambda max_arity: torch.nn.Identity(),
        )


def test_centralized_flat_forward_and_shared_central_params() -> None:
    model = CentralizedFlatRelationalGNN(
        embedding_size=4,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        aggregation="sum",
        execution_policy=FlatExecutionPolicy(relation_kernels="off", program_kernels="off", relation_gather="off"),
    )
    data = _make_small_data()
    out = model(data)
    assert isinstance(out, FlatRelationalOutput)
    assert out.entity.shape == (5, 4)
    rel_a = model.relational_layer.update_modules[0]
    rel_b = model.relational_layer.update_modules[1]
    assert rel_a.central_module is rel_b.central_module
    assert rel_a.condition_embedding is rel_b.condition_embedding

    loss = out.entity.square().sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    central_grads = [
        param.grad for param in model.central_module.parameters() if param.requires_grad
    ]
    assert central_grads
    assert all(grad is not None for grad in central_grads)
    assert model.relation_condition_embedding.weight.grad is not None


def test_centralized_flat_static_condition_embedding_stays_grad_free() -> None:
    model = CentralizedFlatRelationalGNN(
        embedding_size=4,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        relation_condition_learnable=False,
        aggregation="sum",
        execution_policy=FlatExecutionPolicy(relation_kernels="off", program_kernels="off", relation_gather="off"),
    )
    out = model(_make_small_data())
    loss = out.entity.square().sum()
    model.zero_grad(set_to_none=True)
    loss.backward()
    assert model.relation_condition_embedding.weight.requires_grad is False
    assert model.relation_condition_embedding.weight.grad is None


def test_centralized_flat_uses_central_module_factory() -> None:
    calls: list[int] = []

    def factory(max_arity: int) -> torch.nn.Module:
        calls.append(int(max_arity))
        return torch.nn.Identity()

    model = CentralizedFlatRelationalGNN(
        embedding_size=4,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        central_module_factory=factory,
        central_residual=False,
        central_slot_mask=False,
        central_conditioning="concat",
        execution_policy=FlatExecutionPolicy(relation_kernels="off", program_kernels="off", relation_gather="off"),
    )
    assert calls == [2]
    out = model(_make_small_data())
    assert out.entity.shape == (5, 4)


def test_centralized_flat_batches_shared_central_module_once_per_layer() -> None:
    central = _CountingCentralModule(out_size=8)
    model = CentralizedFlatRelationalGNN(
        embedding_size=4,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        central_module=central,
        central_residual=False,
        central_slot_mask=False,
        central_conditioning="concat",
        execution_policy=FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        ),
    )
    out = model(_make_small_data())
    assert out.entity.shape == (5, 4)
    assert central.calls == 1


def test_constructor_validates_relation_module_mapping_keys() -> None:
    with pytest.raises(ValueError):
        FlatRelationalGNN(
            embedding_size=4,
            num_layers=1,
            relations={"rel_a": 2, "rel_b": 1},
            relation_modules={"rel_a": TwoLayerPointwiseRelationMLP(8, 16)},
        )


def test_constructor_validates_relation_module_sequence_length() -> None:
    with pytest.raises(ValueError):
        FlatRelationalGNN(
            embedding_size=4,
            num_layers=1,
            relations={"rel_a": 2, "rel_b": 1},
            relation_modules=[TwoLayerPointwiseRelationMLP(8, 16)],
        )


def test_constructor_validates_declared_relation_block_width() -> None:
    with pytest.raises(ValueError):
        FlatRelationalGNN(
            embedding_size=4,
            num_layers=1,
            relations={"rel": 2},
            relation_modules={"rel": TwoLayerPointwiseRelationMLP(12, 16)},
        )


def test_build_flat_topology_counts_and_offsets() -> None:
    relation_counts = torch.tensor([[1, 2], [2, 1]], dtype=torch.long)
    relation_arities = torch.tensor([2, 1], dtype=torch.long)
    topology = flat_relational_layer_module.build_flat_topology(relation_counts, relation_arities)
    assert topology.relation_counts_total == (3, 3)
    assert topology.relation_arities == (2, 1)
    assert topology.slot_offsets == (0, 6, 9)


def test_flat_relational_layer_sum_matches_reference_forward_and_gradients() -> None:
    modules = {
        "rel_a": TwoLayerPointwiseRelationMLP(8, 16, activation="mish"),
        "rel_b": TwoLayerPointwiseRelationMLP(4, 12, activation="mish"),
    }
    torch.manual_seed(0)
    model = _make_model(copy.deepcopy(modules), aggregation="sum")
    torch.manual_seed(0)
    ref_model = _make_model(copy.deepcopy(modules), aggregation="sum")
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(5, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    payload = _make_small_payload()
    topology = model.relational_layer.get_topology(payload["relation_counts"], payload["relation_arities"])

    out = model.relational_layer(x, payload["relation_counts"], payload["relation_args"], relation_arities=payload["relation_arities"], topology=topology)
    ref = _manual_sum_messages(x_ref, payload["relation_counts"], payload["relation_args"], payload["relation_arities"], ref_model.relational_layer.update_modules)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    out.square().sum().backward()
    ref.square().sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


def test_flat_relational_layer_mean_matches_reference_forward_and_gradients() -> None:
    modules = {
        "rel_a": TwoLayerPointwiseRelationMLP(8, 16, activation="mish"),
        "rel_b": TwoLayerPointwiseRelationMLP(4, 12, activation="mish"),
    }
    torch.manual_seed(0)
    model = _make_model(copy.deepcopy(modules), aggregation="mean")
    torch.manual_seed(0)
    ref_model = _make_model(copy.deepcopy(modules), aggregation="mean")
    ref_model.load_state_dict(model.state_dict(), strict=True)

    x = torch.randn(5, 4, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    payload = _make_small_payload()
    topology = model.relational_layer.get_topology(payload["relation_counts"], payload["relation_arities"])

    out = model.relational_layer(x, payload["relation_counts"], payload["relation_args"], relation_arities=payload["relation_arities"], topology=topology)
    ref = _manual_mean_messages(x_ref, payload["relation_counts"], payload["relation_args"], payload["relation_arities"], ref_model.relational_layer.update_modules)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    out.square().sum().backward()
    ref.square().sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5)


def test_flat_relational_layer_accepts_pyg_aggregation_object() -> None:
    model = _make_model(
        {
            "rel_a": TwoLayerPointwiseRelationMLP(8, 16),
            "rel_b": TwoLayerPointwiseRelationMLP(4, 12),
        },
        aggregation=MeanAggregation(),
    )
    payload = _make_small_payload()
    out = model.relational_layer(
        torch.randn(5, 4),
        payload["relation_counts"],
        payload["relation_args"],
        relation_arities=payload["relation_arities"],
    )
    assert out.shape == (5, 4)


@pytest.mark.parametrize(
    ("module_factory", "expected_kernel_type"),
    [
        (lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="mish"), flat_relational_layer_module.MishBlockKernel),
        (lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="silu"), flat_relational_layer_module.SiLUBlockKernel),
        (lambda width: TwoLayerPointwiseRelationMLP(width, 16, activation="gelu"), flat_relational_layer_module.GELUBlockKernel),
        (lambda width: PostNormTwoLayerPointwiseRelationMLP(width, 16, activation="silu", norm="layernorm"), flat_relational_layer_module.PostNormSiLULayerNormKernel),
        (lambda width: _CustomSpecPostNormTwoLayerSiLU(width, 16), flat_relational_layer_module.PostNormSiLULayerNormKernel),
    ],
)
def test_kernel_matcher_identifies_supported_block_kernels(module_factory, expected_kernel_type: type[object]) -> None:
    model, relation_slice = _make_program_model(module_factory(8))
    match = model.relational_layer._match_kernel(relation_slice)
    assert match is not None
    assert match.kernel is not None
    assert isinstance(match.kernel, expected_kernel_type)
    assert match.spec.kernel_type is expected_kernel_type


@pytest.mark.skipif(getattr(torch.nn, "RMSNorm", None) is None, reason="RMSNorm unavailable")
def test_kernel_matcher_identifies_rmsnorm_block_family() -> None:
    model, relation_slice = _make_program_model(
        PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm")
    )
    match = model.relational_layer._match_kernel(relation_slice)
    assert match is not None
    assert match.kernel is not None
    assert isinstance(match.kernel, flat_relational_layer_module.PreNormSiLURMSNormKernel)
    assert match.spec.kernel_type is flat_relational_layer_module.PreNormSiLURMSNormKernel


def test_exact_program_matching_requires_relation_program_wrapper() -> None:
    model, relation_slice = _make_program_model(
        torch.nn.Sequential(
            TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
            TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
        )
    )
    assert model.relational_layer._match_kernel(relation_slice) is None


@pytest.mark.parametrize(
    ("program", "expected_kernel_type"),
    [
        (
            RelationProgram(
                TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
                TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
            ),
            flat_relational_layer_module.SiLUPairProgramKernel,
        ),
        (
            RelationProgram(
                TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
                PostNormTwoLayerPointwiseRelationMLP(8, 12, activation="silu", norm="layernorm"),
            ),
            flat_relational_layer_module.SiLUThenPostNormProgramKernel,
        ),
        (
            RelationProgram(
                PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm"),
                TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
            ),
            flat_relational_layer_module.PreNormRMSNormThenSiLUProgramKernel,
        ),
    ],
)
def test_exact_program_matching_identifies_registered_program_kernels(program, expected_kernel_type: type[object]) -> None:
    model, relation_slice = _make_program_model(program)
    match = model.relational_layer._match_kernel(relation_slice)
    assert match is not None
    assert match.kernel is not None
    assert isinstance(match.kernel, expected_kernel_type)
    assert match.program_spec is not None
    assert match.program_spec.kernel_type is expected_kernel_type
    assert match.spec.kernel_type is expected_kernel_type


def test_unsupported_custom_module_falls_back_to_eager_execution() -> None:
    model, relation_slice = _make_program_model(_UnsupportedCustomBlock(8))
    assert model.relational_layer._match_kernel(relation_slice) is None


@pytest.mark.parametrize(
    "module",
    [
        TwoLayerPointwiseRelationMLP(8, 16, activation="relu"),
        ThreeLayerPointwiseRelationMLP(8, 16, 12, activation="silu"),
    ],
)
def test_non_kernel_modules_fall_back_to_eager_execution(module: torch.nn.Module) -> None:
    model, relation_slice = _make_program_model(module)
    assert model.relational_layer._match_kernel(relation_slice) is None


def test_custom_kernel_sequence_is_used_as_is() -> None:
    module = TwoLayerPointwiseRelationMLP(8, 16, activation="silu")
    model, relation_slice = _make_program_model(module, relation_kernels=())
    assert model.relational_layer._match_kernel(relation_slice) is None


def test_compute_entity_embeddings_rejects_raw_payload() -> None:
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 16), "rel_b": TwoLayerPointwiseRelationMLP(4, 12)})
    with pytest.raises(TypeError):
        model.compute_entity_embeddings(_make_small_payload())


def test_model_call_rejects_raw_payload() -> None:
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 16), "rel_b": TwoLayerPointwiseRelationMLP(4, 12)})
    with pytest.raises(TypeError):
        model(_make_small_payload())


def test_compute_entity_embeddings_accepts_pyg_data() -> None:
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 16), "rel_b": TwoLayerPointwiseRelationMLP(4, 12)})
    data = _make_small_data()
    data.relation_names = tuple(model.relation_names)
    entity = model.compute_entity_embeddings(data)
    assert entity.shape == (5, 4)


def test_compute_entity_embeddings_accepts_native_mifrost_flat_batch() -> None:
    try:
        import mifrost  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent editable rebuild path
        pytest.skip(f"mifrost unavailable in this test environment: {exc}")
    try:
        flat_relation_encoder = mifrost.FlatRelationEncoder
    except Exception as exc:  # pragma: no cover - env-dependent optional wrapper path
        pytest.skip(f"mifrost FlatRelationEncoder wrapper unavailable: {exc}")
    if flat_relation_encoder is None:
        pytest.skip("mifrost FlatRelationEncoder wrapper unavailable in this test environment")
    domain, state, goals = _load_blocks_problem()
    encoder = flat_relation_encoder(
        domain,
        target_sources=["goal"],
        goal_satisfaction_derivations={
            mifrost.GoalSatisfaction.satisfied,
            mifrost.GoalSatisfaction.unsatisfied,
        },
    )
    native_batch = encoder.encode_batch(states=[state], goals=goals)
    relations = {
        str(name): int(arity)
        for name, arity in zip(native_batch.relation_names, native_batch.relation_arities)
    }
    model = FlatRelationalGNN(embedding_size=8, num_layers=1, relations=relations)
    entity = model.compute_entity_embeddings(native_batch)
    assert entity.dim() == 2


def test_model_returns_structured_output_and_optional_fields() -> None:
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 16), "rel_b": TwoLayerPointwiseRelationMLP(4, 12)})
    output = model(_make_small_data())
    assert isinstance(output, FlatRelationalOutput)
    assert output.entity.shape == (5, 4)
    assert output.entity_batch.shape == (5,)
    assert output.object is not None and output.object_batch is not None
    assert output.target_entity is not None and output.target_entity_batch is not None
    assert output.target is not None and output.target_batch is not None


def test_model_returns_none_for_absent_optional_views() -> None:
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 16), "rel_b": TwoLayerPointwiseRelationMLP(4, 12)})
    data = _make_small_data()
    del data.object_indices
    del data.target_entity_indices
    del data.target_positions
    output = model(data)
    assert output.object is None and output.object_batch is None
    assert output.target_entity is None and output.target_entity_batch is None
    assert output.target is None and output.target_batch is None


def test_compute_entity_embeddings_matches_output_entity() -> None:
    model = _make_model({"rel_a": TwoLayerPointwiseRelationMLP(8, 16), "rel_b": TwoLayerPointwiseRelationMLP(4, 12)})
    data = _make_small_data()
    entity = model.compute_entity_embeddings(data)
    output = model(data)
    assert torch.allclose(entity, output.entity)


def test_auto_policy_cpu_falls_back_cleanly_for_exact_programs() -> None:
    program = RelationProgram(
        TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
        TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
    )
    modules = {"rel_a": program, "rel_b": TwoLayerPointwiseRelationMLP(4, 12, activation="silu")}
    eager = _make_model(modules, execution_policy=FlatExecutionPolicy(relation_kernels="off", program_kernels="off", relation_gather="off"))
    auto = _make_model(copy.deepcopy(modules), execution_policy=FlatExecutionPolicy(relation_kernels="auto", program_kernels="auto", relation_gather="off"))
    auto.load_state_dict(eager.state_dict(), strict=True)
    data = _make_small_data()
    eager_out = eager(data)
    auto_out = auto(data)
    assert torch.allclose(eager_out.entity, auto_out.entity, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    "module",
    [
        TwoLayerPointwiseRelationMLP(8, 16, activation="mish"),
        TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
        TwoLayerPointwiseRelationMLP(8, 16, activation="gelu"),
    ],
)
def test_cuda_single_block_auto_kernel_matches_eager(module: torch.nn.Module) -> None:
    device = torch.device("cuda")
    modules = {"rel_a": module.to(device), "rel_b": TwoLayerPointwiseRelationMLP(4, 12, activation="silu").to(device)}
    eager = _make_model(copy.deepcopy(modules), execution_policy=FlatExecutionPolicy(relation_kernels="off", program_kernels="off", relation_gather="off")).to(device)
    auto = _make_model(copy.deepcopy(modules), execution_policy=FlatExecutionPolicy(relation_kernels="auto", program_kernels="auto", relation_gather="off")).to(device)
    auto.load_state_dict(eager.state_dict(), strict=True)
    payload = _make_small_payload(device=device)
    x = torch.randn(5, 4, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    out = auto.relational_layer(x, payload["relation_counts"], payload["relation_args"], relation_arities=payload["relation_arities"])
    ref = eager.relational_layer(x_ref, payload["relation_counts"], payload["relation_args"], relation_arities=payload["relation_arities"])
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    out.square().sum().backward()
    ref.square().sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize(
    ("program", "ref_fn"),
    [
        (
            RelationProgram(
                TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
                TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
            ),
            execute_program_two_layer_silu_then_two_layer_silu_reference,
        ),
        (
            RelationProgram(
                TwoLayerPointwiseRelationMLP(8, 16, activation="silu"),
                PostNormTwoLayerPointwiseRelationMLP(8, 12, activation="silu", norm="layernorm"),
            ),
            execute_program_two_layer_silu_then_postnorm_two_layer_silu_reference,
        ),
        (
            RelationProgram(
                PreNormTwoLayerPointwiseRelationMLP(8, 16, activation="silu", norm="rmsnorm"),
                TwoLayerPointwiseRelationMLP(8, 12, activation="silu"),
            ),
            execute_program_prenorm_two_layer_silu_rmsnorm_then_two_layer_silu_reference,
        ),
    ],
)
def test_cuda_exact_program_auto_kernel_matches_eager(program: RelationProgram, ref_fn) -> None:
    device = torch.device("cuda")
    eager_module = copy.deepcopy(program).to(device)
    auto_module = copy.deepcopy(program).to(device)
    eager, relation_slice = _make_program_model(eager_module)
    auto, _ = _make_program_model(auto_module)
    eager = eager.to(device)
    auto = auto.to(device)
    eager.relational_layer.execution_policy = FlatExecutionPolicy(relation_kernels="off", program_kernels="off", relation_gather="off")
    auto.relational_layer.execution_policy = FlatExecutionPolicy(relation_kernels="auto", program_kernels="auto", relation_gather="off")
    auto.load_state_dict(eager.state_dict(), strict=True)

    x = torch.randn(6, 4, device=device, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    relation_counts = torch.tensor([[2]], dtype=torch.long, device=device)
    relation_args = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
    relation_arities = torch.tensor([2], dtype=torch.long, device=device)

    out = auto.relational_layer(x, relation_counts, relation_args, relation_arities=relation_arities)
    ref = eager.relational_layer(x_ref, relation_counts, relation_args, relation_arities=relation_arities)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)
    out.square().sum().backward()
    ref.square().sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-5, rtol=1e-4)
