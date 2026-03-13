import torch

from relmo.models import LGANRelationalGNN


def test_lgan_batched_cached_label_fanin_smoke() -> None:
    embedding_size = 8
    relation_dict = {"rel_a": 2, "rel_b": 1}
    model = LGANRelationalGNN(
        embedding_size=embedding_size,
        num_layers=2,
        aggregation="sum",
        symbol_type_ids=("_symbol_",),
        relations=relation_dict,
        rel_layer_mode="batched_cached",
        include_lgan_edges=True,
    )

    x_dict = {
        "_symbol_": torch.zeros(4, 1, dtype=torch.float32),
        "rel_a": torch.zeros(3, 2, dtype=torch.float32),
        "rel_b": torch.zeros(2, 1, dtype=torch.float32),
    }
    edge_index_dict = {
        ("_symbol_", "0", "rel_a"): torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        ("_symbol_", "1", "rel_a"): torch.tensor([[1, 2, 3], [0, 1, 2]], dtype=torch.long),
        ("_symbol_", "0", "rel_b"): torch.tensor([[0, 3], [0, 1]], dtype=torch.long),
        ("rel_a", "_lgan_tn_", "_symbol_"): torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
        ("rel_b", "_lgan_tn_", "_symbol_"): torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
        ("rel_a", "_lgan_nn_", "_symbol_"): torch.tensor([[0, 2], [1, 3]], dtype=torch.long),
        ("rel_b", "_lgan_nn_", "_symbol_"): torch.tensor([[0], [0]], dtype=torch.long),
        ("rel_a", "_lgan_rr_", "rel_a"): torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
        ("rel_b", "_lgan_rr_", "rel_b"): torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    }

    out_dict, _ = model(x_dict, edge_index_dict)
    assert "_symbol_" in out_dict
    assert out_dict["_symbol_"].shape == (4, embedding_size)
