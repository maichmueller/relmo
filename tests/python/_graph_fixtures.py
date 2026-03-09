from __future__ import annotations

from collections.abc import Mapping

import torch


def build_relation_graph(
    *,
    relations: Mapping[str, int],
    symbol_type: str = "_symbol_",
    num_symbols: int = 5,
    relation_sizes: Mapping[str, int] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    """Build a tiny deterministic heterogeneous relation graph."""
    relation_sizes = dict(relation_sizes or {})
    x_dict: dict[str, torch.Tensor] = {symbol_type: torch.zeros((int(num_symbols), 1))}
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
    n_symbols = int(num_symbols)

    for predicate, arity in relations.items():
        arity = int(arity)
        n_rel = int(relation_sizes.get(predicate, max(2, arity + 1)))
        rel_ids = torch.arange(n_rel, dtype=torch.long)
        x_dict[predicate] = torch.zeros((n_rel, arity))
        for pos in range(arity):
            src_ids = (rel_ids + pos) % n_symbols
            edge_index_dict[(symbol_type, str(pos), predicate)] = torch.stack(
                [src_ids, rel_ids]
            )
            edge_index_dict[(predicate, str(pos), symbol_type)] = torch.stack(
                [rel_ids, src_ids]
            )

    return x_dict, edge_index_dict


def add_lgan_edges(
    *,
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    relations: Mapping[str, int],
    symbol_type: str = "_symbol_",
    tn_label: str = "_lgan_tn_",
    nn_label: str = "_lgan_nn_",
    rr_label: str = "_lgan_rr_",
) -> None:
    """Add deterministic TN/NN/RR edge families required by LGANRelationalGNN."""
    n_symbols = int(x_dict[symbol_type].size(0))
    for predicate in relations:
        n_rel = int(x_dict[predicate].size(0))
        rel_ids = torch.arange(n_rel, dtype=torch.long)
        edge_index_dict[(predicate, tn_label, symbol_type)] = torch.stack(
            [rel_ids, (rel_ids + 1) % n_symbols]
        )
        edge_index_dict[(predicate, nn_label, symbol_type)] = torch.stack(
            [rel_ids, (rel_ids + 2) % n_symbols]
        )
        edge_index_dict[(predicate, rr_label, predicate)] = torch.stack(
            [rel_ids, (rel_ids + 1) % n_rel]
        )


def clone_graph(
    x_dict: Mapping[str, torch.Tensor],
    edge_index_dict: Mapping[tuple[str, str, str], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    return (
        {k: v.clone() for k, v in x_dict.items()},
        {k: v.clone() for k, v in edge_index_dict.items()},
    )
