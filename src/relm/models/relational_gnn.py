from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Union

import torch
import torch_geometric as pyg
from torch import Tensor
from torch_geometric.nn.resolver import aggregation_resolver
from torch_geometric.typing import Adj, EdgeType

from ._compile import optional_compile
from ._logging import get_logger

from .aggr import LogSumExpAggregation
from .film import CentralFilmFactory, FiLMConcatMLP
from .hetero_mp import (
    CentralFanOutMP,
    CentralFusedLayerMP,
    FanInMP,
    FanOutMP,
)
from .mlp import ArityMLPFactory, SimpleMLP
from .pyg_module import PyGHeteroModule
from .residual import ResidualModule

RelationDict = dict[str, int]


class CentralRelationModule(torch.nn.Module):
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
        truncate_output: bool = True,
    ) -> None:
        super().__init__()
        if condition_position not in ("pre", "post"):
            raise ValueError(
                f"condition_position must be 'pre' or 'post', got {condition_position!r}."
            )
        self.central_module = central_module
        self.condition_embedding = condition_embedding
        self.condition_index = int(condition_index)
        self.arity = arity
        self.max_arity = max_arity
        self.embedding_size = embedding_size
        self.condition_position = condition_position
        self.truncate_output = truncate_output

    def forward(self, x: Tensor) -> Tensor:
        expected = self.max_arity * self.embedding_size
        if x.size(-1) > expected:
            raise ValueError(
                f"Input has {x.size(-1)} features, but max arity expects {expected}."
            )
        if x.size(-1) < expected:
            pad = x.new_zeros(x.size(0), expected - x.size(-1))
            x = torch.cat([x, pad], dim=1)

        idx = torch.tensor(self.condition_index, device=x.device)
        condition = self.condition_embedding(idx).view(1, -1).expand(x.size(0), -1)
        if self.condition_position == "pre":
            x = torch.cat([condition, x], dim=1)
        else:
            x = torch.cat([x, condition], dim=1)

        out = self.central_module(x)
        if self.truncate_output:
            out = out[:, : self.arity * self.embedding_size]
        return out


class ZeroOut(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x[..., :0]


class BoundedValueHead(torch.nn.Module):
    def __init__(
        self,
        value_net,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        super().__init__()
        self.value_net = value_net  # last layer outputs any real z
        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        if (
            lower_bound is None
            or lower_bound == -float("inf")
            or upper_bound is None
            or upper_bound == float("inf")
        ):
            get_logger(__name__).warning(
                f"At most one of lower_bound and upper_bound is finite, got {lower_bound=}, {upper_bound=}). Ignoring bounds."
            )
            self._forward = self._forward_unbounded
        else:
            self._forward = self._forward_bounded

    def forward(self, x):
        return self._forward(x)

    def _forward_bounded(self, x):
        z = self.value_net(x)
        s = torch.sigmoid(z)  # modulate to [0,1]
        return (
            self.lower_bound + (self.upper_bound - self.lower_bound) * s
        )  # in [lower_bound, upper_bound]

    def _forward_unbounded(self, x):
        return self.value_net(x)


class RelationalGNN(PyGHeteroModule):
    """
    RelationalGNN is a Graph Neural Network designed for learning over relational (heterogeneous) graphs,
    where nodes represent objects and atoms (predicates with arguments), and edges encode relationships
    between them. It performs iterative message passing between objects and atoms, using predicate-specific
    MLPs to aggregate and propagate information. The primary use case is to compute object embeddings in
    planning instances through their atom-relationships.

    Core Functionality:
      - Initializes object and atom embeddings.
      - For each layer, passes messages from objects to atoms (using predicate-specific message modules),
        then from atoms back to objects (using aggregation), and updates object embeddings.
      - Supports customizable aggregation functions, activation functions, and predicate module construction.
      - Handles variable arity predicates and flexible initialization, including random initialization.

    Usage:
      - Instantiate with embedding size, number of layers, object type id, arity dictionary, and optional customizations.
      - Call forward() with x_dict (node features), edge_index_dict (edges), and optional batch or info dicts.
      - Returns object embeddings and their batch indices for downstream tasks.
    """

    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        symbol_type_ids: Iterable[str] | str,
        relation_dict: RelationDict,
        relation_module_factory: Callable[[str, int], torch.nn.Module]
        | ArityMLPFactory
        | None = None,
        activation: Union[str, Callable, None] = None,
        ignore_zero_arity_relations: bool = True,
        random_init: bool = False,
        random_init_dims: int | None = None,
        random_init_percent: float | None = None,
        strict_ntype_filter: bool = True,
        compile_forward: bool = False,
        compile_prudent: bool = False,
        compile_prudent_kwargs: dict | None = None,
        rel_layer_mode: str = "modular",
        rel_validate_routing: bool = False,
    ):
        """
        :param embedding_size: The size of object embeddings.
        :param num_layer: Total number of message exchange iterations.
        :param aggr: Aggregation-function to be used for message passing.
        :param symbol_type_ids: The type identifier(s) of objects in the x_dict.
        :param relation_dict: A dictionary mapping predicates names to their arity.
        :param activation: The activation function for all MLPs
            (Default: "mish", other options: "relu", "gelu", "silu", or a callable).
        :param relation_module_factory: A factory function that takes a predicate name and its arity
            as input and returns a torch.nn.Module to be used for message passing for that predicate.
            If None, a default factory creating one-layer MLPs is used.
        :param ignore_zero_arity_relations: If True, predicates with zero arity are skipped
            when creating predicate-specific modules. This is useful if zero-arity predicates
            (propositions) do not carry meaningful information. (Default: True)
        :param random_init: If True, object and atom embeddings are initialized with random values
            in addition to zeros. (Default: False)
        :param random_init_dims: If set, this number of dimensions in the embeddings are randomly
            initialized. If None, random_init_percent is used instead. (Default: None)
        :param random_init_percent: If set, this percentage of dimensions in the embeddings are
            randomly initialized. If None, random_init_dims is used instead. (Default: None)
        :param strict_ntype_filter: If True, only edges where the source node type matches
            the expected object type are considered during message passing.
            Otherwise, containment of the object type is enough. (Default: False)
        :param compile_forward: If True, wraps `forward` in `torch.compile`. This can be
            very beneficial on stable graph schemas, but may lead to frequent
            recompilations if the set of node/edge types varies across calls.
        :param compile_prudent: If True, compiles only the dense, schema-stable MLP
            submodules (e.g. embedding updater; and for centralized variants the
            central module). This is typically more robust for highly variable
            STRIPS graphs than compiling the full `forward`. (Default: False)

        Note that predicates refer to any relations provided.
        """
        super().__init__()
        self._compile_forward = compile_forward
        self._compile_prudent = bool(compile_prudent)
        self._compile_prudent_kwargs = compile_prudent_kwargs
        if rel_layer_mode not in ("modular", "batched_cached"):
            raise ValueError(
                "rel_layer_mode must be one of {'modular','batched_cached'}, "
                f"got {rel_layer_mode!r}."
            )
        self.rel_layer_mode = rel_layer_mode
        self.rel_validate_routing = bool(rel_validate_routing)
        self.embedding_size: int = embedding_size
        self.random_initialization: bool = random_init
        self.random_initialization_dims: int
        if random_init_dims is not None:
            self.random_initialization_dims = random_init_dims
        elif random_init_percent is not None:
            self.random_initialization_dims = int(random_init_percent * embedding_size)
        else:
            self.random_initialization_dims = embedding_size
        self.random_initialization_dims = max(
            min(self.random_initialization_dims, embedding_size), 0
        )
        if self.random_initialization_dims == 0:
            get_logger(__name__).warning(
                "Random initialization dimensions are set to 0, random initialization will not be applied."
            )
            self.random_initialization = False

        self.num_layer: int = num_layer
        self.symbol_type_ids: tuple[str, ...] = (
            (symbol_type_ids,)
            if isinstance(symbol_type_ids, str)
            else tuple(symbol_type_ids)
        )
        self.relation_dict: RelationDict = relation_dict

        self.strict_ntype_filter = strict_ntype_filter
        # the module to pass node-features from symbols (objects, action, ...) to relations (atoms, action-schemas, ...)
        self.symbols_to_relations_mp: torch.nn.Module
        # the module to pass created messages from relations back to symbols
        self.relations_to_symbols_mp: torch.nn.Module
        # the module to update object embeddings based on the final object messages
        self.embedding_updater: torch.nn.Module
        # resolve activation
        self.activation = activation or "mish"
        # resolve aggregation
        if isinstance(aggr, str) or aggr is None:
            if aggr is None or aggr.lower() == "logsumexp":
                aggr = LogSumExpAggregation()
            else:
                aggr = aggregation_resolver(aggr)
        else:
            raise ValueError(f"Invalid aggregation type: {aggr}")
        self._init_modules(
            embedding_size,
            num_layer,
            aggr,
            relation_module_factory,
            ignore_zero_arity_relations,
        )

    def _maybe_compile_prudent_module(
        self, module: torch.nn.Module, *, name: str
    ) -> torch.nn.Module:
        if not self._compile_prudent:
            return module
        if not hasattr(torch, "compile"):
            get_logger(__name__).warning(
                "compile_prudent=True requested, but torch.compile is unavailable; '%s' will run eagerly.",
                name,
            )
            return module

        compile_kwargs = {"backend": "inductor", "dynamic": True}
        inst_kwargs = self._compile_prudent_kwargs
        if isinstance(inst_kwargs, dict):
            compile_kwargs.update(inst_kwargs)

        try:
            return torch.compile(module, **compile_kwargs)
        except KeyboardInterrupt:
            raise
        except Exception:
            get_logger(__name__).warning(
                "compile_prudent=True: failed to torch.compile '%s'; running eagerly.",
                name,
                exc_info=True,
            )
            return module

    def _init_modules(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        relation_module_factory: Callable[[str, int], torch.nn.Module],
        ignore_zero_arity_relations: bool = True,
    ):
        """
        Initializes the modules for the RelationalGNN.
        """
        if relation_module_factory is None:
            # One MLP per relation
            # For a relation p(o1,...,ok) the corresponding MLP gets k symbol
            # embeddings as input and generates k outputs, one for each symbol.
            relation_module_factory = ArityMLPFactory(
                feature_size=embedding_size,
                added_arity=0,  # no additional arity
                residual=True,
                padding=None,  # no padding
                layers=1,  # one layer per predicate
                activation=self.activation,
            )
        relation_module_dict = {
            pred: relation_module_factory(arity)
            for pred, arity in self.relation_dict.items()
            if arity > 0 or not ignore_zero_arity_relations
        }

        if self.rel_layer_mode == "batched_cached":
            from .hetero_mp import BatchedFanInMP, BatchedFanOutMP

            relation_arities = {
                pred: arity
                for pred, arity in self.relation_dict.items()
                if arity > 0 or not ignore_zero_arity_relations
            }
            self.symbols_to_relations_mp = BatchedFanOutMP(
                relation_module_dict,
                relation_arities=relation_arities,
                embedding_size=embedding_size,
                src_types=self.symbol_type_ids,
                strict_filter_mode=self.strict_ntype_filter,
                validate_routing=self.rel_validate_routing,
            )
            self.relations_to_symbols_mp = BatchedFanInMP(
                embedding_size=embedding_size,
                dst_types=self.symbol_type_ids,
                relation_arities=relation_arities,
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
                validate_routing=self.rel_validate_routing,
            )
        else:
            self.symbols_to_relations_mp = FanOutMP(
                relation_module_dict,
                src_types=self.symbol_type_ids,
                strict_filter_mode=self.strict_ntype_filter,
            )
            self.relations_to_symbols_mp = FanInMP(
                embedding_size=embedding_size,
                dst_types=self.symbol_type_ids,
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
            )
        # Updates object embedding from embedding of last iteration and current iteration:
        # `X_o = comb([X_o, m_o])` where `m_o` is the final object message
        self.embedding_updater = SimpleMLP(
            in_size=2 * embedding_size,
            embedding_size=2 * embedding_size,
            out_size=embedding_size,
            activation=self.activation,
        )
        self.embedding_updater = self._maybe_compile_prudent_module(
            self.embedding_updater, name="embedding_updater"
        )

    def initialize_embeddings(
        self, x_dict: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Any]:
        # Initialize embeddings for objects and atoms with 0s.
        # embedding-dims of objects = embedding_size
        # embedding-dims of atoms = (arity of predicate) * embedding_size
        for key, x in x_dict.items():
            assert x.dim() == 2
            init_embed = torch.zeros(
                x.shape[0], x.shape[1] * self.embedding_size, device=x.device
            )
            if self.random_initialization:
                # Random initialization of embeddings
                if self.random_initialization_dims is not None:
                    init_embed = torch.nn.init.xavier_normal_(
                        init_embed[:, -self.random_initialization_dims :]
                    )
            x_dict[key] = init_embed
        if self.rel_layer_mode == "batched_cached":
            return x_dict, {"rel_batched_cache": {}}
        return x_dict, None

    def symbol_ntype_filter(self, key: str) -> bool:
        if self.strict_ntype_filter:
            return key in self.symbol_type_ids
        return any(symbol_type_id in key for symbol_type_id in self.symbol_type_ids)

    def contained_symbol_ntypes(self, keys: Iterable[str]) -> list[str]:
        return sorted(key for key in keys if self.symbol_ntype_filter(key))

    def layer(
        self, x_dict, edge_index_dict, *, extra=None, symbol_ntype_ids: Sequence[str]
    ):
        """
        Groups object embeddings that are part of an atom and applies predicate-specific Module (e.g. MLP) based on the edge type.
        """
        # Spread the object embeddings to the atoms via message passing.
        # Note: unlike symbol embeddings, relation embeddings are always simply replaced, instead of updated.
        if self.rel_layer_mode == "batched_cached":
            if extra is None:
                extra = {}
            cache = extra.setdefault("rel_batched_cache", {})
            atom_msgs = self.symbols_to_relations_mp(
                x_dict, edge_index_dict, cache=cache
            )
        else:
            atom_msgs = self.symbols_to_relations_mp(x_dict, edge_index_dict)
        x_dict.update(atom_msgs)

        for symbol_ntype_id in symbol_ntype_ids:
            # Distribute the relation embeddings back to the corresponding symbols via message passing.
            if self.rel_layer_mode == "batched_cached":
                if extra is None:
                    extra = {}
                cache = extra.setdefault("rel_batched_cache", {})
                symbol_msgs = self.relations_to_symbols_mp(
                    x_dict, edge_index_dict, cache=cache
                )[symbol_ntype_id]
            else:
                symbol_msgs = self.relations_to_symbols_mp(x_dict, edge_index_dict)[
                    symbol_ntype_id
                ]
            # perform update step of message passing, but for symbol-nodes only.
            # The symbol/object embeddings are updated based on the previous embedding `X_o` and final symbol message `m_o`.
            # In formula: `X_o = comb([X_o, m_o])`
            updated_obj_emb = self.embedding_updater(
                torch.cat([x_dict[symbol_ntype_id], symbol_msgs], dim=1)
            )
            # residual update (current + updates)
            x_dict[symbol_ntype_id] = x_dict[symbol_ntype_id] + updated_obj_emb

    @optional_compile(
        enable_attr="_compile_forward",
        backend="inductor",
        dynamic=True,
    )
    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        info_dict: Optional[Dict[str, Tensor]] = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Compute object embeddings for each state.
        The states represent graphs and their objects represent nodes.
        The graphs also contain atoms as nodes, but only the object embeddings are returned.
        :param x_dict: The node features for each node type.
            The keys should contain self.symbol_type_id.
        :param edge_index_dict: The edges between heterogeneous nodes.
        :param batch_dict: Optional information which node is associated to which state.
            If you pass more than one state (graph) to this function, you should pass the batch_dict too.
        :param info_dict: Optional information about the states.
        :return: A tuple containing:
        - The first tensor contains object embeddings with shape [N, embedding_size], where N is the total number of objects across all states in the batch.
        - The second tensor contains batch indices with shape [N], mapping each object (node) to its corresponding state (graph).
        Note that the number of objects is not necessarily equal for each state. This tuple can be used to instantiate a `SetEmbedding`.
        We do not return this object directly since pytorch would refuse to accept hooks with this return type.
        """
        batch_dict = batch_dict or {}
        x_dict, edge_index_dict = self._filter_out_dummies(x_dict, edge_index_dict)
        symbol_ntype_ids = self.contained_symbol_ntypes(x_dict.keys())

        # Initialize embeddings for objects and atoms.
        x_dict, extra = self.initialize_embeddings(x_dict)

        for _ in range(self.num_layer):
            self.layer(
                x_dict, edge_index_dict, extra=extra, symbol_ntype_ids=symbol_ntype_ids
            )

        x_dict_out = {
            symbol_ntype_id: x_dict[symbol_ntype_id]
            for symbol_ntype_id in symbol_ntype_ids
        }
        batch_dict_out = {
            symbol_ntype_id: batch_dict[symbol_ntype_id]
            if symbol_ntype_id in batch_dict
            else torch.zeros(
                x_dict_out[symbol_ntype_id].shape[0],
                dtype=torch.long,
                device=x_dict_out[symbol_ntype_id].device,
            )
            for symbol_ntype_id in symbol_ntype_ids
        }
        return x_dict_out, batch_dict_out

    @staticmethod
    def _filter_out_dummies(x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items() if v.numel() != 0}
        edge_index_dict = {k: v for k, v in edge_index_dict.items() if v.numel() != 0}
        return x_dict, edge_index_dict


class CentralizedRelationalGNN(RelationalGNN):
    """
    CentralizedRelationGNN shares a single message module across all relations and conditions it
    on a per-relation embedding.
    """

    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        symbol_type_ids: Iterable[str] | str,
        relation_dict: RelationDict,
        relation_module_factory: Callable[[int], torch.nn.Module] | None = None,
        *,
        relation_condition_dim: int | None = None,
        relation_condition_learnable: bool = True,
        condition_position: str = "pre",
        central_module: torch.nn.Module | None = None,
        central_module_factory: Callable[..., torch.nn.Module]
        | ArityMLPFactory
        | CentralFilmFactory
        | None = None,
        central_residual: bool = True,
        central_conditioning: str = "film",
        central_slot_mask: bool = True,
        central_layer_mode: str = "fused",
        central_validate_routing: bool = False,
        **kwargs,
    ):
        if central_module is not None and (
            central_module_factory is not None or relation_module_factory is not None
        ):
            raise ValueError(
                "Pass either central_module or central_module_factory/relation_module_factory, not both."
            )
        if central_module_factory is None and relation_module_factory is not None:
            central_module_factory = relation_module_factory

        self._relation_condition_dim = relation_condition_dim
        self.relation_condition_learnable = relation_condition_learnable
        self.condition_position = condition_position
        self.central_residual = central_residual
        self.central_slot_mask = bool(central_slot_mask)
        if central_layer_mode not in ("fused", "modular"):
            raise ValueError(
                "central_layer_mode must be one of {'fused','modular'}, "
                f"got {central_layer_mode!r}."
            )
        self.central_layer_mode = central_layer_mode
        self.central_validate_routing = bool(central_validate_routing)
        if central_conditioning not in ("concat", "film"):
            raise ValueError(
                "central_conditioning must be 'concat' or 'film', "
                f"got {central_conditioning!r}."
            )
        self.central_conditioning = central_conditioning
        self._central_module_factory = central_module_factory
        self._central_module = None
        # Initialized in _init_modules after relation metadata is available.
        self.relation_condition_dim: int | None = None
        self.relation_condition_index: dict[str, int] | None = None
        self.relation_condition_embedding: torch.nn.Embedding | None = None
        self.max_relation_arity: int | None = None
        self.central_mask_dim: int = 0
        self.central_module: torch.nn.Module | None = None
        # Cache module pre-__init__ to avoid nn.Module attribute registration errors.
        object.__setattr__(self, "_central_module_cache", central_module)

        super().__init__(
            embedding_size=embedding_size,
            num_layer=num_layer,
            aggr=aggr,
            symbol_type_ids=symbol_type_ids,
            relation_dict=relation_dict,
            relation_module_factory=None,
            **kwargs,
        )

    def _init_modules(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        relation_module_factory: Callable[[str, int], torch.nn.Module],
        ignore_zero_arity_relations: bool = True,
    ):
        if self._relation_condition_dim is None:
            self._relation_condition_dim = max(1, int(math.sqrt(embedding_size)))
        if self._relation_condition_dim < 1:
            raise ValueError(
                f"relation_condition_dim must be >= 1, got {self._relation_condition_dim}."
            )
        self.relation_condition_dim = self._relation_condition_dim
        self.relation_condition_index = {
            predicate: idx for idx, predicate in enumerate(self.relation_dict)
        }
        self.relation_condition_embedding = torch.nn.Embedding(
            len(self.relation_dict), self.relation_condition_dim
        )
        if not self.relation_condition_learnable:
            self.relation_condition_embedding.weight.requires_grad_(False)

        self.max_relation_arity: int = max(self.relation_dict.values(), default=0)
        self.central_mask_dim = self.max_relation_arity if self.central_slot_mask else 0
        self.central_module = self._resolve_central_module(
            embedding_size,
            self.max_relation_arity,
            self.relation_condition_dim,
            self.central_mask_dim,
        )
        self.central_module = self._maybe_compile_prudent_module(
            self.central_module, name="central_module"
        )
        self._central_module = self.central_module

        relation_arities = (
            {
                pred: arity
                for pred, arity in self.relation_dict.items()
                if arity > 0 or not ignore_zero_arity_relations
            }
            if ignore_zero_arity_relations
            else dict(self.relation_dict.items())
        )
        if self.central_layer_mode == "fused":
            self.central_fused_layer_mp = CentralFusedLayerMP(
                central_module=self.central_module,
                condition_embedding=self.relation_condition_embedding,
                relation_condition_index=self.relation_condition_index,
                relation_arities=relation_arities,
                max_arity=self.max_relation_arity,
                embedding_size=embedding_size,
                condition_position=self.condition_position,
                include_slot_mask=self.central_slot_mask,
                symbol_type_ids=self.symbol_type_ids,
                dst_symbol_type_ids=self.symbol_type_ids,
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
                validate_routing=self.central_validate_routing,
            )
            # Unused in fused mode, but kept as valid nn.Module attributes.
            self.symbols_to_relations_mp = torch.nn.Identity()
            self.relations_to_symbols_mp = torch.nn.Identity()
        else:
            self.symbols_to_relations_mp = CentralFanOutMP(
                central_module=self.central_module,
                condition_embedding=self.relation_condition_embedding,
                relation_condition_index=self.relation_condition_index,
                relation_arities=relation_arities,
                max_arity=self.max_relation_arity,
                embedding_size=embedding_size,
                condition_position=self.condition_position,
                include_slot_mask=self.central_slot_mask,
                src_types=self.symbol_type_ids,
                strict_filter_mode=self.strict_ntype_filter,
            )
            self.relations_to_symbols_mp = FanInMP(
                embedding_size=embedding_size,
                dst_types=self.symbol_type_ids,
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
            )
        self.embedding_updater = SimpleMLP(
            in_size=2 * embedding_size,
            embedding_size=2 * embedding_size,
            out_size=embedding_size,
            activation=self.activation,
        )
        self.embedding_updater = self._maybe_compile_prudent_module(
            self.embedding_updater, name="embedding_updater"
        )

    def layer(
        self, x_dict, edge_index_dict, *, extra=None, symbol_ntype_ids: Sequence[str]
    ):
        if self.central_layer_mode != "fused":
            return super().layer(
                x_dict,
                edge_index_dict,
                extra=extra,
                symbol_ntype_ids=symbol_ntype_ids,
            )

        if extra is None:
            extra = {}
        cache = extra.setdefault("central_fused_cache", {})
        atom_msgs, symbol_msgs_dict = self.central_fused_layer_mp(
            x_dict, edge_index_dict, cache=cache
        )
        x_dict.update(atom_msgs)
        for symbol_ntype_id in symbol_ntype_ids:
            symbol_msgs = symbol_msgs_dict.get(symbol_ntype_id)
            if symbol_msgs is None:
                symbol_msgs = x_dict[symbol_ntype_id].new_zeros(
                    (int(x_dict[symbol_ntype_id].size(0)), self.embedding_size)
                )
            updated_obj_emb = self.embedding_updater(
                torch.cat([x_dict[symbol_ntype_id], symbol_msgs], dim=1)
            )
            x_dict[symbol_ntype_id] = x_dict[symbol_ntype_id] + updated_obj_emb

    def initialize_embeddings(
        self, x_dict: Dict[str, Tensor]
    ) -> tuple[Dict[str, Tensor], Any]:
        x_dict, extra = super().initialize_embeddings(x_dict)
        if self.central_layer_mode == "fused":
            # Mutable cache shared across iterations of a single forward() call.
            extra = {"central_fused_cache": {}}
        return x_dict, extra

    def _build_default_central_module(
        self, embedding_size: int, max_arity: int, condition_dim: int, mask_dim: int
    ) -> torch.nn.Module:
        if max_arity == 0:
            return ZeroOut()
        feature_in_size = max_arity * embedding_size + mask_dim
        in_size = feature_in_size + condition_dim
        out_size = max_arity * embedding_size
        hidden_size = max(in_size, out_size)
        if self.central_conditioning == "film":
            mlp = FiLMConcatMLP(
                in_dim=feature_in_size,
                cond_dim=condition_dim,
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
            module=mlp, truncated_dim=out_size, truncate_right=truncate_right
        )

    def _resolve_central_module(
        self, embedding_size: int, max_arity: int, condition_dim: int, mask_dim: int
    ) -> torch.nn.Module:
        if self._central_module is not None:
            return self._central_module
        if getattr(self, "_central_module_cache", None) is not None:
            self._central_module = self._central_module_cache
            self._central_module_cache = None
            return self._central_module
        if self._central_module_factory is None:
            return self._build_default_central_module(
                embedding_size, max_arity, condition_dim, mask_dim
            )
        factory = self._central_module_factory
        if isinstance(factory, CentralFilmFactory):
            return factory(
                embedding_size=embedding_size,
                max_arity=max_arity,
                cond_dim=condition_dim,
                mask_dim=mask_dim,
                condition_position=self.condition_position,
            )
        if isinstance(factory, ArityMLPFactory):
            if mask_dim:
                raise ValueError(
                    "ArityMLPFactory does not support centralized slot-mask features. "
                    "Use CentralFiLMFactory or set central_slot_mask=False."
                )
            if factory.in_condition_features == 0 and condition_dim > 0:
                factory.in_condition_features = condition_dim
            elif factory.in_condition_features != condition_dim:
                raise ValueError(
                    "CentralizedRelationalGNN expects central_module_factory inputs to "
                    f"match relation_condition_dim={condition_dim}, got "
                    f"in_condition_features={factory.in_condition_features}. "
                    "Set relation_condition_dim or in_condition_features to the same value."
                )
            factory.in_condition_position = self.condition_position
            return factory(max_arity)
        signature = inspect.signature(factory)
        params = [
            p
            for p in signature.parameters.values()
            if p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(params) <= 1:
            return factory(max_arity)
        if len(params) == 2:
            return factory(embedding_size, max_arity)
        if len(params) == 3:
            return factory(embedding_size, max_arity, condition_dim)
        if len(params) == 4:
            return factory(embedding_size, max_arity, condition_dim, mask_dim)
        return factory(
            embedding_size, max_arity, condition_dim, mask_dim, self.condition_position
        )


class LGANRelationalGNN(RelationalGNN):
    """
    LGANRelationalGNN implements the explicit LGAN-v2 stack with dual target aggregation:
    AGGR_t (target-neighbor), AGGR_n (neighbor-neighbor), and relation-relation exchange (RR).
    """

    def __init__(
        self,
        embedding_size: int,
        num_layer: int,
        aggr: Optional[str | pyg.nn.aggr.Aggregation],
        symbol_type_ids: Iterable[str] | str,
        relation_dict: RelationDict,
        relation_module_factory: Callable[[str, int], torch.nn.Module] | None = None,
        ignore_zero_arity_relations: bool = True,
        include_lgan_edges: bool = True,
        lgan_tn_edge_pos: str = "_lgan_tn_",
        lgan_nn_edge_pos: str = "_lgan_nn_",
        lgan_rr_edge_pos: str = "_lgan_rr_",
        **kwargs,
    ):
        self.include_lgan_edges = bool(include_lgan_edges)
        if not self.include_lgan_edges:
            raise ValueError(
                "LGANRelationalGNN requires include_lgan_edges=True. "
                "Use RelationalGNN for non-LGAN message passing."
            )
        self.lgan_tn_edge_pos = str(lgan_tn_edge_pos)
        self.lgan_nn_edge_pos = str(lgan_nn_edge_pos)
        self.lgan_rr_edge_pos = str(lgan_rr_edge_pos)
        super().__init__(
            embedding_size=embedding_size,
            num_layer=num_layer,
            aggr=aggr,
            symbol_type_ids=symbol_type_ids,
            relation_dict=relation_dict,
            relation_module_factory=relation_module_factory,
            ignore_zero_arity_relations=ignore_zero_arity_relations,
            **kwargs,
        )
        self._relation_type_ids = tuple(str(k) for k in self.relation_dict.keys())
        if self.rel_layer_mode == "batched_cached":
            from .hetero_mp import BatchedFanInMP

            self.tn_relations_to_symbols_mp = BatchedFanInMP(
                embedding_size=self.embedding_size,
                dst_types=self.symbol_type_ids,
                relation_arities=self.relation_dict,
                src_types=self._relation_type_ids,
                edge_labels=(self.lgan_tn_edge_pos,),
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
                validate_routing=self.rel_validate_routing,
            )
            self.nn_relations_to_symbols_mp = BatchedFanInMP(
                embedding_size=self.embedding_size,
                dst_types=self.symbol_type_ids,
                relation_arities=self.relation_dict,
                src_types=self._relation_type_ids,
                edge_labels=(self.lgan_nn_edge_pos,),
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
                validate_routing=self.rel_validate_routing,
            )
            self.rr_relations_to_relations_mp = BatchedFanInMP(
                embedding_size=self.embedding_size,
                dst_types=self._relation_type_ids,
                relation_arities=self.relation_dict,
                src_types=self._relation_type_ids,
                edge_labels=(self.lgan_rr_edge_pos,),
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
                validate_routing=self.rel_validate_routing,
            )
        else:
            self.tn_relations_to_symbols_mp = FanInMP(
                embedding_size=self.embedding_size,
                dst_types=self.symbol_type_ids,
                src_types=self._relation_type_ids,
                edge_labels=(self.lgan_tn_edge_pos,),
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
            )
            self.nn_relations_to_symbols_mp = FanInMP(
                embedding_size=self.embedding_size,
                dst_types=self.symbol_type_ids,
                src_types=self._relation_type_ids,
                edge_labels=(self.lgan_nn_edge_pos,),
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
            )
            self.rr_relations_to_relations_mp = FanInMP(
                embedding_size=self.embedding_size,
                dst_types=self._relation_type_ids,
                src_types=self._relation_type_ids,
                edge_labels=(self.lgan_rr_edge_pos,),
                aggr=aggr,
                strict_filter_mode=self.strict_ntype_filter,
            )
        self.fusion_updater = SimpleMLP(
            in_size=3 * embedding_size,
            embedding_size=2 * embedding_size,
            out_size=embedding_size,
            activation=self.activation,
        )

    def _matches_ntype(self, node_type: str, candidates: Sequence[str]) -> bool:
        if self.strict_ntype_filter:
            return node_type in candidates
        return any(candidate in node_type for candidate in candidates)

    def _is_symbol_ntype(self, node_type: str) -> bool:
        return self._matches_ntype(node_type, self.symbol_type_ids)

    def _is_relation_ntype(self, node_type: str) -> bool:
        return self._matches_ntype(node_type, self._relation_type_ids)

    def _filter_directional_edges(
        self,
        edge_index_dict: Dict[EdgeType, Adj],
        *,
        edge_label: str,
        relation_to_relation: bool,
    ) -> Dict[EdgeType, Adj]:
        filtered: Dict[EdgeType, Adj] = {}
        for edge_type, edge_index in edge_index_dict.items():
            src, rel, dst = edge_type
            if str(rel) != edge_label:
                continue
            if relation_to_relation:
                if not (self._is_relation_ntype(src) and self._is_relation_ntype(dst)):
                    continue
            else:
                if not (self._is_relation_ntype(src) and self._is_symbol_ntype(dst)):
                    continue
            filtered[edge_type] = edge_index
        return filtered

    def _ensure_required_edge_families(self, edge_index_dict: Dict[EdgeType, Adj]) -> None:
        tn_edges = self._filter_directional_edges(
            edge_index_dict,
            edge_label=self.lgan_tn_edge_pos,
            relation_to_relation=False,
        )
        nn_edges = self._filter_directional_edges(
            edge_index_dict,
            edge_label=self.lgan_nn_edge_pos,
            relation_to_relation=False,
        )
        rr_edges = self._filter_directional_edges(
            edge_index_dict,
            edge_label=self.lgan_rr_edge_pos,
            relation_to_relation=True,
        )
        missing: list[str] = []
        if not tn_edges:
            missing.append(self.lgan_tn_edge_pos)
        if not nn_edges:
            missing.append(self.lgan_nn_edge_pos)
        if not rr_edges:
            missing.append(self.lgan_rr_edge_pos)
        if missing:
            raise ValueError(
                "LGAN enabled but required LGAN edge families are missing from the "
                f"encoded graph. missing={missing}; available_labels="
                f"{sorted({str(edge_type[1]) for edge_type in edge_index_dict.keys()})}. "
                "Ensure the encoder emits TN/NN/RR labels."
            )

    def _pool_relation_embeddings(self, x_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pooled: Dict[str, Tensor] = {}
        for relation_type in self._relation_type_ids:
            if relation_type not in x_dict:
                continue
            rel_x = x_dict[relation_type]
            if rel_x.dim() != 2:
                raise ValueError(
                    f"Expected 2D relation embedding for {relation_type!r}, got dim={rel_x.dim()}."
                )
            arity = int(self.relation_dict[relation_type])
            if arity <= 0:
                pooled[relation_type] = rel_x.new_zeros((rel_x.size(0), self.embedding_size))
                continue
            expected_dim = arity * self.embedding_size
            if rel_x.size(-1) != expected_dim:
                raise ValueError(
                    f"Relation embedding width mismatch for {relation_type!r}: "
                    f"got {rel_x.size(-1)}, expected {expected_dim}."
                )
            pooled[relation_type] = rel_x.view(rel_x.size(0), arity, self.embedding_size).mean(
                dim=1
            )
        return pooled

    def layer(
        self, x_dict, edge_index_dict, *, extra=None, symbol_ntype_ids: Sequence[str]
    ):
        if not symbol_ntype_ids:
            raise RuntimeError(
                "LGANRelationalGNN received a batch without symbol node types. "
                f"x_dict_keys={sorted(x_dict.keys())}."
            )

        cache = None
        if self.rel_layer_mode == "batched_cached":
            if extra is None:
                extra = {}
            cache = extra.setdefault("rel_batched_cache", {})

        # Phase 1: standard positional symbol->relation updates.
        standard_edges = {
            edge_type: edge_index
            for edge_type, edge_index in edge_index_dict.items()
            if str(edge_type[1]).isdigit()
        }
        if self.rel_layer_mode == "batched_cached":
            atom_msgs = self.symbols_to_relations_mp(x_dict, standard_edges, cache=cache)
        else:
            atom_msgs = self.symbols_to_relations_mp(x_dict, standard_edges)
        x_dict.update(atom_msgs)

        # Phase 2: relation-pair embedding pooling.
        relation_pair_x = self._pool_relation_embeddings(x_dict)
        if not relation_pair_x:
            raise RuntimeError(
                "LGANRelationalGNN could not build relation-pair embeddings. "
                f"available_node_types={sorted(x_dict.keys())} "
                f"relation_types={sorted(self._relation_type_ids)}."
            )

        # Phase 3: RR exchange (relation->relation).
        rr_edges = self._filter_directional_edges(
            edge_index_dict,
            edge_label=self.lgan_rr_edge_pos,
            relation_to_relation=True,
        )
        if self.rel_layer_mode == "batched_cached":
            rr_msgs_dict = self.rr_relations_to_relations_mp(
                relation_pair_x, rr_edges, cache=cache
            )
        else:
            rr_msgs_dict = self.rr_relations_to_relations_mp(relation_pair_x, rr_edges)
        for rel_type, rel_prev in relation_pair_x.items():
            rr_msg = rr_msgs_dict.get(rel_type)
            if rr_msg is not None:
                relation_pair_x[rel_type] = rel_prev + rr_msg

        # Phase 4: AGGR_t over TN edges (relation->symbol).
        tn_edges = self._filter_directional_edges(
            edge_index_dict,
            edge_label=self.lgan_tn_edge_pos,
            relation_to_relation=False,
        )
        symbol_aggr_x = dict(relation_pair_x)
        for symbol_ntype_id in symbol_ntype_ids:
            symbol_aggr_x[symbol_ntype_id] = x_dict[symbol_ntype_id]
        if self.rel_layer_mode == "batched_cached":
            tn_msgs_dict = self.tn_relations_to_symbols_mp(symbol_aggr_x, tn_edges, cache=cache)
        else:
            tn_msgs_dict = self.tn_relations_to_symbols_mp(symbol_aggr_x, tn_edges)

        # Phase 5: AGGR_n over NN edges (relation->symbol).
        nn_edges = self._filter_directional_edges(
            edge_index_dict,
            edge_label=self.lgan_nn_edge_pos,
            relation_to_relation=False,
        )
        if self.rel_layer_mode == "batched_cached":
            nn_msgs_dict = self.nn_relations_to_symbols_mp(symbol_aggr_x, nn_edges, cache=cache)
        else:
            nn_msgs_dict = self.nn_relations_to_symbols_mp(symbol_aggr_x, nn_edges)

        # Phase 6: fuse and residual-update symbols.
        for symbol_ntype_id in symbol_ntype_ids:
            prev_emb = x_dict[symbol_ntype_id]
            tn_msgs = tn_msgs_dict.get(symbol_ntype_id, torch.zeros_like(prev_emb))
            nn_msgs = nn_msgs_dict.get(symbol_ntype_id, torch.zeros_like(prev_emb))
            updated_obj_emb = self.fusion_updater(
                torch.cat([prev_emb, tn_msgs, nn_msgs], dim=1)
            )
            x_dict[symbol_ntype_id] = prev_emb + updated_obj_emb

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        batch_dict: Optional[Dict[str, Tensor]] = None,
        info_dict: Optional[Dict[str, Tensor]] = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        self._ensure_required_edge_families(edge_index_dict)
        out, out_batch = super().forward(x_dict, edge_index_dict, batch_dict, info_dict)
        if not out or all(t.numel() == 0 for t in out.values()):
            raise RuntimeError(
                "LGANRelationalGNN produced no symbol outputs for the batch. "
                f"input_node_types={sorted(x_dict.keys())} "
                f"input_edge_labels={sorted({str(k[1]) for k in edge_index_dict.keys()})}."
            )
        return out, out_batch
