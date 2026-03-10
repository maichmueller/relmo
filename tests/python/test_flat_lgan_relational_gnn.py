from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from relm.models import (
    FlatExecutionPolicy,
    FlatLGANRelationalGNN,
    FlatRelationalOutput,
    TwoLayerPointwiseRelationMLP,
)
from relm.ops import mp as mp_ops


class _ZeroBlock(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class _InputInitializedFlatLGAN(FlatLGANRelationalGNN):
    def initialize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class _IndexInitializedFlatLGAN(FlatLGANRelationalGNN):
    def initialize_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        num_nodes = int(x.size(0))
        values = torch.arange(
            num_nodes * self.embedding_size,
            device=x.device,
            dtype=x.dtype,
        )
        return values.view(num_nodes, self.embedding_size)


def _make_lgan_data() -> Data:
    return Data(
        x=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ],
            dtype=torch.float,
        ),
        relation_counts=torch.tensor([[2, 1]], dtype=torch.long),
        relation_args=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        relation_arities=torch.tensor([2, 1], dtype=torch.long),
        node_sizes=torch.tensor([5], dtype=torch.long),
        object_indices=torch.tensor([0, 1, 3], dtype=torch.long),
        target_entity_indices=torch.tensor([1, 4], dtype=torch.long),
        target_positions=torch.tensor([0, 4], dtype=torch.long),
        lgan_tn_relation_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        lgan_tn_entity_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        lgan_nn_relation_indices=torch.tensor([1, 2], dtype=torch.long),
        lgan_nn_entity_indices=torch.tensor([2, 4], dtype=torch.long),
        lgan_rr_src_relation_indices=torch.tensor([0, 1], dtype=torch.long),
        lgan_rr_dst_relation_indices=torch.tensor([1, 2], dtype=torch.long),
    )


def _make_lgan_model() -> FlatLGANRelationalGNN:
    model = _InputInitializedFlatLGAN(
        embedding_size=2,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        aggregation="sum",
        relation_modules={
            "rel_a": _ZeroBlock(4),
            "rel_b": _ZeroBlock(2),
        },
        execution_policy=FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        ),
    )
    fusion = torch.nn.Linear(6, 2, bias=False)
    with torch.no_grad():
        fusion.weight.zero_()
        fusion.weight[0, 2] = 1.0
        fusion.weight[0, 4] = 1.0
        fusion.weight[1, 3] = 1.0
        fusion.weight[1, 5] = 1.0
    model.fusion_updater = fusion
    return model


def _load_native_lgan_batch():
    pymimir = pytest.importorskip("pymimir")
    try:
        import mifrost  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"mifrost unavailable: {exc}")
    try:
        flat_relation_encoder = mifrost.FlatRelationEncoder
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"mifrost FlatRelationEncoder wrapper unavailable: {exc}")

    root = Path(__file__).resolve().parents[2] / "data" / "pddl_domains" / "blocks"
    problem_files = sorted(
        path
        for path in root.glob("*.pddl")
        if path.name != "domain.pddl" and not path.name.startswith("._")
    )
    if not problem_files:
        pytest.skip("no Blocks problem files available in test fixture root")

    domain = pymimir.Domain(root / "domain.pddl")
    problem = pymimir.Problem(domain, problem_files[0], mode="lifted")
    goals = list(problem.get_goal_condition().get_literals())
    encoder = flat_relation_encoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=["goal"],
    )
    native_batch = encoder.encode_batch(states=[problem.get_initial_state()], goals=goals)
    pyg_batch = native_batch.as_pyg(as_batch=True)
    return native_batch, pyg_batch


def _manual_lgan_reference(data: Data) -> torch.Tensor:
    x = data.x
    rel0 = x.index_select(0, torch.tensor([0, 1])).mean(dim=0)
    rel1 = x.index_select(0, torch.tensor([2, 3])).mean(dim=0)
    rel2 = x.index_select(0, torch.tensor([4])).mean(dim=0)
    relation_pair_x = torch.stack([rel0, rel1, rel2], dim=0)

    rr_msgs = torch.zeros_like(relation_pair_x)
    rr_msgs[1] += relation_pair_x[0]
    rr_msgs[2] += relation_pair_x[1]
    relation_pair_x = relation_pair_x + rr_msgs

    tn_msgs = torch.zeros_like(x)
    tn_msgs[0] += relation_pair_x[0]
    tn_msgs[1] += relation_pair_x[1]
    tn_msgs[2] += relation_pair_x[2]

    nn_msgs = torch.zeros_like(x)
    nn_msgs[2] += relation_pair_x[1]
    nn_msgs[4] += relation_pair_x[2]

    return x + tn_msgs + nn_msgs


def test_flat_lgan_requires_lgan_indices() -> None:
    data = Data(
        x=torch.zeros((3, 2)),
        relation_counts=torch.tensor([[1]], dtype=torch.long),
        relation_args=torch.tensor([0], dtype=torch.long),
        relation_arities=torch.tensor([1], dtype=torch.long),
    )
    model = _InputInitializedFlatLGAN(
        embedding_size=2,
        num_layers=1,
        relations={"rel": 1},
        relation_modules={"rel": _ZeroBlock(2)},
        execution_policy=FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        ),
    )
    with pytest.raises(ValueError, match="requires all LGAN flat index tensors"):
        model(data)


def test_flat_lgan_matches_manual_reference_and_structured_views() -> None:
    data = _make_lgan_data()
    model = _make_lgan_model()
    out = model(data)
    assert isinstance(out, FlatRelationalOutput)
    expected = _manual_lgan_reference(data)
    assert torch.allclose(out.entity, expected, atol=1e-6, rtol=0.0)
    assert torch.equal(out.object, expected.index_select(0, data.object_indices))
    assert torch.equal(
        out.target_entity,
        expected.index_select(0, data.target_entity_indices),
    )
    assert torch.equal(out.target, expected.index_select(0, data.target_positions))


def test_flat_lgan_rejects_out_of_range_relation_indices() -> None:
    data = _make_lgan_data()
    data.lgan_rr_dst_relation_indices = torch.tensor([1, 3], dtype=torch.long)
    model = _make_lgan_model()
    with pytest.raises(ValueError, match="lgan_rr_dst_relation_indices"):
        model(data)


def test_flat_lgan_accepts_native_mifrost_batch_and_matches_pyg() -> None:
    native_batch, pyg_batch = _load_native_lgan_batch()
    relations = {
        str(name): int(arity)
        for name, arity in zip(native_batch.relation_names, native_batch.relation_arities)
    }
    relation_modules = {
        name: _ZeroBlock(int(arity) * 4) for name, arity in relations.items()
    }
    model = _IndexInitializedFlatLGAN(
        embedding_size=4,
        num_layers=1,
        relations=relations,
        aggregation="sum",
        relation_modules=relation_modules,
        execution_policy=FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        ),
    )
    prepared_native = model._prepare_batch(native_batch)
    assert prepared_native.lgan_topology is not None
    assert prepared_native.lgan_tn_relation_indices is not None
    assert prepared_native.lgan_nn_relation_indices is not None
    assert prepared_native.lgan_rr_src_relation_indices is not None
    assert prepared_native.lgan_rr_dst_relation_indices is not None
    assert int(prepared_native.lgan_topology.relation_instance_count) == int(
        native_batch.relation_instance_sizes.sum().item()
    )
    assert int(prepared_native.lgan_tn_relation_indices.numel()) == int(
        native_batch.lgan_tn_sizes.sum().item()
    )
    assert int(prepared_native.lgan_nn_relation_indices.numel()) == int(
        native_batch.lgan_nn_sizes.sum().item()
    )
    assert int(prepared_native.lgan_rr_src_relation_indices.numel()) == int(
        native_batch.lgan_rr_sizes.sum().item()
    )
    # LGAN anchor rows are structural. They may coincide with target metadata on
    # the current mifrost carrier, but the model does not require prediction
    # targets to be present just to execute LGAN.
    assert hasattr(native_batch, "lgan_tn_relation_indices")

    out_native = model(native_batch)
    out_pyg = model(pyg_batch)
    assert torch.allclose(out_native.entity, out_pyg.entity, atol=1e-6, rtol=0.0)
    assert torch.equal(out_native.entity_batch, out_pyg.entity_batch)
    if out_native.object is not None and out_pyg.object is not None:
        assert torch.allclose(out_native.object, out_pyg.object, atol=1e-6, rtol=0.0)
    if out_native.target_entity is not None and out_pyg.target_entity is not None:
        assert torch.allclose(
            out_native.target_entity,
            out_pyg.target_entity,
            atol=1e-6,
            rtol=0.0,
        )
    if out_native.target is not None and out_pyg.target is not None:
        assert torch.allclose(out_native.target, out_pyg.target, atol=1e-6, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_flat_lgan_cuda_phase1_kernel_matches_eager() -> None:
    if not mp_ops.available():
        pytest.skip("relm_mp ops unavailable")
    device = torch.device("cuda")
    modules = {
        "rel_a": TwoLayerPointwiseRelationMLP(4, 8, activation="silu").to(device),
        "rel_b": TwoLayerPointwiseRelationMLP(2, 8, activation="silu").to(device),
    }
    eager = _InputInitializedFlatLGAN(
        embedding_size=2,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        aggregation="sum",
        relation_modules=copy.deepcopy(modules),
        execution_policy=FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        ),
    ).to(device)
    auto = _InputInitializedFlatLGAN(
        embedding_size=2,
        num_layers=1,
        relations={"rel_a": 2, "rel_b": 1},
        aggregation="sum",
        relation_modules=copy.deepcopy(modules),
        execution_policy=FlatExecutionPolicy(
            relation_kernels="auto",
            program_kernels="auto",
            relation_gather="off",
        ),
    ).to(device)
    auto.load_state_dict(eager.state_dict(), strict=True)

    data = _make_lgan_data().to(device)
    out = auto(data)
    ref = eager(data)
    assert torch.allclose(out.entity, ref.entity, atol=1e-5, rtol=1e-4)
