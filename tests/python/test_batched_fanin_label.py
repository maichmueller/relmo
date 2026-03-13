import torch

from relmo.models.hetero_mp import BatchedFanInMP, FanInMP


def _make_inputs() -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    x_dict = {
        "obj": torch.zeros(3, 4, dtype=torch.float32),
        "rel_a": torch.tensor(
            [[1.0, 0.0, 0.5, -1.0], [0.2, 0.3, -0.1, 0.7]], dtype=torch.float32
        ),
        "rel_b": torch.tensor(
            [[-0.1, 1.2, 0.0, 0.4], [0.3, -0.2, 0.8, 0.0], [0.5, 0.5, -0.3, 0.1]],
            dtype=torch.float32,
        ),
    }
    edge_index_dict = {
        ("rel_a", "_lgan_tn_", "obj"): torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        ("rel_b", "_lgan_tn_", "obj"): torch.tensor([[0, 2], [1, 2]], dtype=torch.long),
        ("rel_b", "_lgan_nn_", "obj"): torch.tensor([[1], [0]], dtype=torch.long),
    }
    return x_dict, edge_index_dict


def test_batched_label_fanin_matches_fanin_sum() -> None:
    x_dict, edge_index_dict = _make_inputs()
    x_ref = {k: v.clone().requires_grad_(k.startswith("rel_")) for k, v in x_dict.items()}
    x_batched = {k: v.clone().requires_grad_(k.startswith("rel_")) for k, v in x_dict.items()}

    ref_mp = FanInMP(
        embedding_size=4,
        dst_types=("obj",),
        src_types=("rel_a", "rel_b"),
        edge_labels=("_lgan_tn_",),
        aggr="sum",
        strict_filter_mode=True,
    )
    batched_mp = BatchedFanInMP(
        embedding_size=4,
        dst_types=("obj",),
        relation_arities={"rel_a": 2, "rel_b": 3},
        src_types=("rel_a", "rel_b"),
        edge_labels=("_lgan_tn_",),
        aggr="sum",
        strict_filter_mode=True,
    )

    out_ref = ref_mp(x_ref, edge_index_dict)["obj"]
    out_batched = batched_mp(x_batched, edge_index_dict)["obj"]
    assert torch.allclose(out_batched, out_ref, atol=1e-6, rtol=1e-5)

    loss_ref = out_ref.square().sum()
    loss_batched = out_batched.square().sum()
    loss_ref.backward()
    loss_batched.backward()
    assert torch.allclose(x_batched["rel_a"].grad, x_ref["rel_a"].grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(x_batched["rel_b"].grad, x_ref["rel_b"].grad, atol=1e-6, rtol=1e-5)


def test_batched_label_fanin_shared_cache_no_collision() -> None:
    x_dict, edge_index_dict = _make_inputs()
    cache: dict = {}
    tn_mp = BatchedFanInMP(
        embedding_size=4,
        dst_types=("obj",),
        relation_arities={"rel_a": 2, "rel_b": 3},
        src_types=("rel_a", "rel_b"),
        edge_labels=("_lgan_tn_",),
        aggr="sum",
        strict_filter_mode=True,
    )
    nn_mp = BatchedFanInMP(
        embedding_size=4,
        dst_types=("obj",),
        relation_arities={"rel_a": 2, "rel_b": 3},
        src_types=("rel_a", "rel_b"),
        edge_labels=("_lgan_nn_",),
        aggr="sum",
        strict_filter_mode=True,
    )

    ref_tn = FanInMP(
        embedding_size=4,
        dst_types=("obj",),
        src_types=("rel_a", "rel_b"),
        edge_labels=("_lgan_tn_",),
        aggr="sum",
        strict_filter_mode=True,
    )
    ref_nn = FanInMP(
        embedding_size=4,
        dst_types=("obj",),
        src_types=("rel_a", "rel_b"),
        edge_labels=("_lgan_nn_",),
        aggr="sum",
        strict_filter_mode=True,
    )

    out_tn = tn_mp(x_dict, edge_index_dict, cache=cache)["obj"]
    out_nn = nn_mp(x_dict, edge_index_dict, cache=cache)["obj"]
    assert torch.allclose(out_tn, ref_tn(x_dict, edge_index_dict)["obj"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_nn, ref_nn(x_dict, edge_index_dict)["obj"], atol=1e-6, rtol=1e-5)
