"""Centralized flat relational GNN.

This module provides a flat-native analogue of ``CentralizedRelationalGNN``.
It keeps the flat ``mifrost`` carrier and flat message-passing layer, but
shares a single relation transform across all relations via relation-specific
conditioning embeddings.
"""

from __future__ import annotations

import inspect
import math
from typing import Callable, Mapping, Sequence

import torch
import torch_geometric as pyg

from .film import CentralFiLMFactory, FiLMConcatMLP
from .flat_contract import FlatExecutionPolicy
from .flat_relational_gnn import FlatRelationalGNN
from .flat_relational_layer import FlatRelationKernel
from .mlp import ArityMLPFactory, SimpleMLP
from .residual import ResidualModule


class _ZeroOut(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., :0]


class CentralizedFlatRelationModule(torch.nn.Module):
    """Per-relation flat wrapper around a shared central module.

    Input shape:
        ``[rows, arity * embedding_size]``

    Output shape:
        ``[rows, arity * embedding_size]``

    The wrapper pads to ``max_arity * embedding_size``, appends the relation
    condition embedding (and optional slot mask), applies the shared central
    module, and truncates back to the active relation width.
    """

    def __init__(
        self,
        *,
        central_module: torch.nn.Module,
        condition_embedding: torch.nn.Embedding,
        condition_index: int,
        arity: int,
        max_arity: int,
        embedding_size: int,
        condition_position: str,
        include_slot_mask: bool,
    ) -> None:
        super().__init__()
        if condition_position not in ("pre", "post"):
            raise ValueError(
                f"condition_position must be 'pre' or 'post', got {condition_position!r}."
            )
        self.central_module = central_module
        self.condition_embedding = condition_embedding
        self.condition_index = int(condition_index)
        self.arity = int(arity)
        self.max_arity = int(max_arity)
        self.embedding_size = int(embedding_size)
        self.condition_position = condition_position
        self.include_slot_mask = bool(include_slot_mask)
        self.width = int(self.arity * self.embedding_size)
        self.mask_dim = int(self.max_arity if self.include_slot_mask else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = int(self.arity * self.embedding_size)
        if x.size(-1) != expected:
            raise ValueError(
                f"Expected relation rows of width {expected}, got {int(x.size(-1))}."
            )
        target_width = int(self.max_arity * self.embedding_size)
        if x.size(-1) < target_width:
            pad = x.new_zeros((int(x.size(0)), target_width - int(x.size(-1))))
            x = torch.cat([x, pad], dim=-1)

        pieces = [x]
        if self.include_slot_mask:
            mask = x.new_zeros((int(x.size(0)), self.max_arity))
            if self.arity > 0:
                mask[:, : self.arity] = 1.0
            pieces.append(mask)

        cond_idx = torch.tensor(self.condition_index, device=x.device)
        cond = self.condition_embedding(cond_idx).view(1, -1).expand(int(x.size(0)), -1)
        if self.condition_position == "pre":
            inp = torch.cat([cond, *pieces], dim=-1)
        else:
            inp = torch.cat([*pieces, cond], dim=-1)

        out = self.central_module(inp)
        return out[:, :expected].contiguous()


class CentralizedFlatRelationalGNN(FlatRelationalGNN):
    """Flat-native centralized relational GNN with shared relation parameters.

    This model consumes the same flat ``mifrost`` batches as
    :class:`FlatRelationalGNN`, but relation transforms are represented by a
    single shared ``central_module`` conditioned on a learned relation-id
    embedding.
    """

    def __init__(
        self,
        embedding_size: int,
        num_layers: int,
        relations: Mapping[str, int],
        aggregation: str | pyg.nn.aggr.Aggregation | None = "sum",
        *,
        relation_condition_dim: int | None = None,
        relation_condition_learnable: bool = True,
        condition_position: str = "pre",
        central_module: torch.nn.Module | None = None,
        central_module_factory: Callable[..., torch.nn.Module]
        | ArityMLPFactory
        | CentralFiLMFactory
        | None = None,
        central_residual: bool = True,
        central_conditioning: str = "film",
        central_slot_mask: bool = True,
        relation_kernels: Sequence[FlatRelationKernel] | None = None,
        execution_policy: FlatExecutionPolicy = FlatExecutionPolicy(),
        compile_forward: bool = False,
        activation: str | Callable | None = None,
    ) -> None:
        if central_module is not None and central_module_factory is not None:
            raise ValueError(
                "Pass either central_module or central_module_factory, not both."
            )
        if condition_position not in ("pre", "post"):
            raise ValueError(
                f"condition_position must be 'pre' or 'post', got {condition_position!r}."
            )
        if central_conditioning not in ("concat", "film"):
            raise ValueError(
                "central_conditioning must be 'concat' or 'film', "
                f"got {central_conditioning!r}."
            )

        self._relation_condition_dim = relation_condition_dim
        self.relation_condition_learnable = bool(relation_condition_learnable)
        self.condition_position = condition_position
        self.central_residual = bool(central_residual)
        self.central_conditioning = central_conditioning
        self.central_slot_mask = bool(central_slot_mask)
        self.activation = activation or "mish"
        self._central_module_factory = central_module_factory
        object.__setattr__(self, "_central_module_cache", central_module)

        relation_items = tuple((str(name), int(arity)) for name, arity in relations.items())
        self.relation_condition_dim = self._resolve_relation_condition_dim(embedding_size)
        self.relation_condition_index = {
            predicate: idx for idx, (predicate, _) in enumerate(relation_items)
        }
        relation_condition_embedding = torch.nn.Embedding(
            len(relation_items), self.relation_condition_dim
        )
        if not self.relation_condition_learnable:
            relation_condition_embedding.weight.requires_grad_(False)

        self.max_relation_arity = max((arity for _, arity in relation_items), default=0)
        self.central_mask_dim = self.max_relation_arity if self.central_slot_mask else 0
        central_module = self._resolve_central_module(embedding_size)
        object.__setattr__(self, "_central_module_ref", central_module)
        object.__setattr__(self, "_relation_condition_embedding_ref", relation_condition_embedding)

        relation_modules = {
            predicate: CentralizedFlatRelationModule(
                central_module=central_module,
                condition_embedding=relation_condition_embedding,
                condition_index=self.relation_condition_index[predicate],
                arity=arity,
                max_arity=self.max_relation_arity,
                embedding_size=embedding_size,
                condition_position=self.condition_position,
                include_slot_mask=self.central_slot_mask,
            )
            for predicate, arity in relation_items
        }

        super().__init__(
            embedding_size=embedding_size,
            num_layers=num_layers,
            relations=dict(relation_items),
            aggregation=aggregation,
            relation_modules=relation_modules,
            relation_module_factory=None,
            relation_kernels=relation_kernels,
            execution_policy=execution_policy,
            compile_forward=compile_forward,
            activation=activation,
        )

    @property
    def central_module(self) -> torch.nn.Module:
        return self._central_module_ref

    @property
    def relation_condition_embedding(self) -> torch.nn.Embedding:
        return self._relation_condition_embedding_ref

    def _resolve_relation_condition_dim(self, embedding_size: int) -> int:
        if self._relation_condition_dim is None:
            return max(1, int(math.sqrt(int(embedding_size))))
        if int(self._relation_condition_dim) < 1:
            raise ValueError(
                f"relation_condition_dim must be >= 1, got {self._relation_condition_dim}."
            )
        return int(self._relation_condition_dim)

    def _build_default_central_module(self, embedding_size: int) -> torch.nn.Module:
        if self.max_relation_arity == 0:
            return _ZeroOut()
        feature_in_size = int(self.max_relation_arity * embedding_size + self.central_mask_dim)
        in_size = int(feature_in_size + self.relation_condition_dim)
        out_size = int(self.max_relation_arity * embedding_size)
        hidden_size = max(in_size, out_size)
        if self.central_conditioning == "film":
            mlp = FiLMConcatMLP(
                in_dim=feature_in_size,
                cond_dim=self.relation_condition_dim,
                hidden_dims=[hidden_size],
                out_dim=out_size,
                condition_position=self.condition_position,
                activation=self.activation,
            )
        else:
            mlp = SimpleMLP(
                in_size=in_size,
                embedding_size=hidden_size,
                out_size=out_size,
                activation=self.activation,
            )
        if not self.central_residual:
            return mlp
        truncate_right = self.condition_position == "post"
        return ResidualModule(
            module=mlp,
            truncated_dim=out_size,
            truncate_right=truncate_right,
        )

    def _resolve_central_module(self, embedding_size: int) -> torch.nn.Module:
        if self._central_module_cache is not None:
            return self._central_module_cache
        factory = self._central_module_factory
        if factory is None:
            return self._build_default_central_module(embedding_size)
        if isinstance(factory, CentralFiLMFactory):
            return factory(
                embedding_size=embedding_size,
                max_arity=self.max_relation_arity,
                cond_dim=self.relation_condition_dim,
                mask_dim=self.central_mask_dim,
                condition_position=self.condition_position,
            )
        if isinstance(factory, ArityMLPFactory):
            if self.central_mask_dim:
                raise ValueError(
                    "ArityMLPFactory does not support centralized slot-mask features. "
                    "Use CentralFiLMFactory or set central_slot_mask=False."
                )
            if factory.in_condition_features == 0 and self.relation_condition_dim > 0:
                factory.in_condition_features = self.relation_condition_dim
            elif factory.in_condition_features != self.relation_condition_dim:
                raise ValueError(
                    "CentralizedFlatRelationalGNN expects central_module_factory inputs to "
                    f"match relation_condition_dim={self.relation_condition_dim}, got "
                    f"in_condition_features={factory.in_condition_features}."
                )
            factory.in_condition_position = self.condition_position
            return factory(self.max_relation_arity)
        signature = inspect.signature(factory)
        params = [
            p
            for p in signature.parameters.values()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(params) <= 1:
            return factory(self.max_relation_arity)
        if len(params) == 2:
            return factory(embedding_size, self.max_relation_arity)
        if len(params) == 3:
            return factory(embedding_size, self.max_relation_arity, self.relation_condition_dim)
        if len(params) == 4:
            return factory(
                embedding_size,
                self.max_relation_arity,
                self.relation_condition_dim,
                self.central_mask_dim,
            )
        return factory(
            embedding_size,
            self.max_relation_arity,
            self.relation_condition_dim,
            self.central_mask_dim,
            self.condition_position,
        )


__all__ = [
    "CentralizedFlatRelationModule",
    "CentralizedFlatRelationalGNN",
]
