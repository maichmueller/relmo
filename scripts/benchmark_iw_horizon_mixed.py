#!/usr/bin/env python3
"""Benchmark flat IW-horizon workloads on mixed IPC train splits.

This harness is intentionally aligned with the relmo + mifrost flat horizon lane
used by BE-IW-style lookahead policies:

1. sample states from `data/pddl_domains/*-ipc/train`
2. generate IW(1) lookaheads for each sampled state
3. materialize `mifrost.TransitionDAG` inputs
4. encode with `mifrost.FlatHorizonEncoder`
5. run `relmo.models.FlatRelationalGNN`

The main comparison is eager execution versus relmo's custom CUDA kernel lane,
with throughput reported both as raw latency and work-normalized candidate /
relation-argument throughput.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import mifrost  # type: ignore
import pymimir as mm  # type: ignore
import torch
from pymimir.advanced.search import (  # type: ignore
    BeamNoveltyMode,
    GoalCountLayerOrderingStrategy,
    InOrderLayerOrderingStrategy,
    RandomizedLayerOrderingStrategy,
    ReverseOrderLayerOrderingStrategy,
)

from relmo.models import (
    FlatExecutionPolicy,
    FlatRelationalGNN,
    PostNormTwoLayerPointwiseRelationMLP,
    PreNormTwoLayerPointwiseRelationMLP,
    RelationProgram,
    TwoLayerPointwiseRelationMLP,
)


@dataclass(frozen=True)
class SampledState:
    problem_name: str
    rollout_depth: int
    state: Any
    goal_condition: Any


@dataclass
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass
class DomainWorkloadStats:
    num_states: int
    num_problems: int
    rollout_depth_mean: float
    rollout_depth_max: int
    total_candidates: int
    mean_candidates: float
    max_candidates: int
    num_entities: int
    num_objects: int
    num_targets: int
    relation_instances: int
    relation_args: int


class IWLookaheadTransition:
    """Minimal IW macro-action wrapper for horizon DAG registration."""

    def __init__(
        self,
        start_state: Any,
        final_state: Any,
        actions: Sequence[Any],
    ) -> None:
        if start_state.get_problem() != final_state.get_problem():
            raise ValueError("start_state and final_state must belong to one problem.")
        self.start_state = start_state
        self.final_state = final_state
        self.actions = list(actions)
        self.problem = start_state.get_problem()
        self._effect: list[Any] | None = None

    @property
    def depth(self) -> int:
        return len(self.actions)

    def get_effect(self) -> list[Any]:
        if self._effect is None:
            start_atoms = set(self.start_state.get_atoms(ignore_static=True))
            final_atoms = set(self.final_state.get_atoms(ignore_static=True))
            add_atoms = final_atoms - start_atoms
            delete_atoms = start_atoms - final_atoms
            effect: list[Any] = []
            effect.extend(
                mm.GroundLiteral.new(atom, True, self.problem) for atom in add_atoms
            )
            effect.extend(
                mm.GroundLiteral.new(atom, False, self.problem) for atom in delete_atoms
            )
            self._effect = effect
        return self._effect


CandidateBatch = tuple[list[IWLookaheadTransition], list[tuple[int, int]]]
ActionBatch = list[Any]


def _resolve_device(device: str) -> torch.device:
    query = device.strip().lower()
    if query == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = torch.device(device)
    if out.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device but torch.cuda.is_available() is False.")
    return out


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _normalized_aggregation(name: str) -> str:
    normalized = name.strip().lower()
    if normalized == "smax":
        return "logsumexp"
    if normalized == "add":
        return "sum"
    if normalized == "hmax":
        return "max"
    return normalized


def _resolve_layer_ordering_strategy(name: str | None, problem: Any):
    if name is None:
        return None
    if name == "in_order":
        return InOrderLayerOrderingStrategy()
    if name == "reverse_order":
        return ReverseOrderLayerOrderingStrategy()
    if name == "randomized":
        return RandomizedLayerOrderingStrategy()
    if name == "goal_count_more":
        return GoalCountLayerOrderingStrategy(problem._advanced_problem, True)
    if name == "goal_count_less":
        return GoalCountLayerOrderingStrategy(problem._advanced_problem, False)
    raise ValueError(f"Unknown IW layer ordering strategy: {name!r}.")


def _resolve_beam_novelty_mode(name: str) -> BeamNoveltyMode:
    normalized = name.strip().lower()
    if normalized == "all_tested":
        return BeamNoveltyMode.ALL_TESTED
    if normalized == "survivors_only":
        return BeamNoveltyMode.SURVIVORS_ONLY
    raise ValueError(f"Unknown IW beam novelty mode: {name!r}.")


def _goal_literals(goal_condition: Any) -> list[Any]:
    return list(goal_condition.get_literals())


def _advanced_state_for_dag(state: Any):
    return state._advanced_state


def _discover_train_dirs(
    pddl_root: pathlib.Path,
    domains_arg: str,
) -> list[tuple[str, pathlib.Path]]:
    if domains_arg.strip().lower() == "auto":
        domain_names = sorted(
            path.name
            for path in pddl_root.iterdir()
            if path.is_dir()
            and path.name.endswith("-ipc")
            and (path / "train" / "domain.pddl").is_file()
        )
    else:
        domain_names = [part.strip() for part in domains_arg.split(",") if part.strip()]
    if not domain_names:
        raise RuntimeError(f"No benchmark domains found under {pddl_root}.")
    out: list[tuple[str, pathlib.Path]] = []
    for domain_name in domain_names:
        train_dir = (pddl_root / domain_name / "train").resolve()
        if not (train_dir / "domain.pddl").is_file():
            raise FileNotFoundError(f"train/domain.pddl not found for {domain_name}: {train_dir}")
        out.append((domain_name, train_dir))
    return out


def _load_problem_paths(train_dir: pathlib.Path) -> list[pathlib.Path]:
    problems = sorted(
        path for path in train_dir.glob("*.pddl") if path.name != "domain.pddl"
    )
    if not problems:
        raise RuntimeError(f"No training problems found in {train_dir}.")
    return problems


def _sample_states(
    domain: Any,
    problem_paths: Sequence[pathlib.Path],
    *,
    batch_size: int,
    max_rollout_depth: int,
    seed: int,
) -> list[SampledState]:
    rng = random.Random(seed)
    path_pool = list(problem_paths)
    rng.shuffle(path_pool)
    sampled: list[SampledState] = []
    problems: dict[pathlib.Path, Any] = {}
    path_index = 0

    while len(sampled) < int(batch_size):
        if path_index >= len(path_pool):
            rng.shuffle(path_pool)
            path_index = 0
        problem_path = path_pool[path_index]
        path_index += 1
        if problem_path not in problems:
            problems[problem_path] = mm.Problem(domain, str(problem_path), mode="lifted")
        problem = problems[problem_path]
        state = problem.get_initial_state()
        realized_depth = 0
        target_depth = rng.randint(0, int(max_rollout_depth))
        for _ in range(target_depth):
            actions = list(state.generate_applicable_actions())
            if not actions:
                break
            state = rng.choice(actions).apply(state)
            realized_depth += 1
        sampled.append(
            SampledState(
                problem_name=problem_path.stem,
                rollout_depth=realized_depth,
                state=state,
                goal_condition=problem.get_goal_condition(),
            )
        )
    return sampled


def _generate_iw_actions(
    start_state: Any,
    *,
    include_start: bool,
    projective_iw1: bool,
    typed_projection: bool,
    keep_depth_one_novel: bool,
    keep_goal_nonunary_atoms: bool,
    layer_ordering_strategy_name: str | None,
    max_next_layer_states: int,
    beam_width: int,
    beam_novelty_mode: str,
    max_depth: int,
    relaxed_survivors_only_beam: bool,
    randomize_equal_score_ties: bool,
    equal_score_tie_seed: int | None,
    num_threads: int,
    chunk_size: int,
) -> tuple[list[IWLookaheadTransition], list[tuple[int, int]]]:
    predecessors: dict[Any, tuple[Any, Any]] = {}
    distances: dict[Any, int] = {start_state: 0}

    def add_transition(
        current_state: Any,
        action: Any,
        _cost: float,
        successor_state: Any,
    ) -> None:
        current_distance = distances[current_state]
        successor_distance = current_distance + 1
        if (successor_state not in distances) or (
            successor_distance < distances[successor_state]
        ):
            distances[successor_state] = successor_distance
            predecessors[successor_state] = (action, current_state)

    search_kwargs = {
        "layer_ordering_strategy": _resolve_layer_ordering_strategy(
            layer_ordering_strategy_name,
            start_state.get_problem(),
        ),
        "max_next_layer_states": int(max_next_layer_states),
        "beam_width": int(beam_width),
        "beam_novelty_mode": _resolve_beam_novelty_mode(beam_novelty_mode),
        "max_depth": int(max_depth),
        "relaxed_survivors_only_beam": bool(relaxed_survivors_only_beam),
        "randomize_equal_score_ties": bool(randomize_equal_score_ties),
        "equal_score_tie_seed": equal_score_tie_seed,
        "num_threads": int(num_threads),
        "chunk_size": int(chunk_size),
        "on_generate_new_state": add_transition,
    }

    if projective_iw1:
        mm.projective_iw(
            start_state.get_problem(),
            start_state,
            typed_projection=bool(typed_projection),
            keep_depth_one_novel=bool(keep_depth_one_novel),
            keep_goal_nonunary_atoms=bool(keep_goal_nonunary_atoms),
            **search_kwargs,
        )
    else:
        mm.iw(start_state.get_problem(), start_state, 1, **search_kwargs)

    state_indices: dict[Any, int] = {start_state: 0} if include_start else {}
    transitions: list[IWLookaheadTransition] = []
    for final_state in predecessors.keys():
        actions: list[Any] = []
        cursor = final_state
        while cursor in predecessors:
            action, cursor = predecessors[cursor]
            actions.append(action)
        actions.reverse()
        state_indices[final_state] = len(state_indices)
        transitions.append(IWLookaheadTransition(start_state, final_state, actions))

    edges: list[tuple[int, int]] = []
    for state, (_action, predecessor_state) in predecessors.items():
        if (predecessor_state != start_state) or include_start:
            edges.append((state_indices[predecessor_state], state_indices[state]))
    return transitions, edges


def _generate_applicable_actions(start_state: Any) -> ActionBatch:
    return list(start_state.generate_applicable_actions())


def _build_iw_horizon_dag(
    root_state: Any,
    transitions: Sequence[IWLookaheadTransition],
    edges: Sequence[tuple[int, int]],
):
    dag = mifrost.TransitionDAG(_advanced_state_for_dag(root_state))
    parent_by_child: dict[int, int] = {}
    for parent_idx, child_idx in edges:
        if int(child_idx) in parent_by_child:
            raise RuntimeError(
                "Expected one parent per IW child, got multiple parents for "
                f"candidate index {child_idx}."
            )
        parent_by_child[int(child_idx)] = int(parent_idx)

    for candidate_idx, transition in enumerate(transitions):
        parent_idx = parent_by_child.get(candidate_idx)
        parent_state = (
            root_state if parent_idx is None else transitions[parent_idx].final_state
        )
        dag.register_transition(
            _advanced_state_for_dag(parent_state),
            _advanced_state_for_dag(transition.final_state),
            candidate_id=int(candidate_idx),
            delta_literals=transition.get_effect(),
        )
    return dag


def _build_horizon_encoder(
    domain: Any,
    *,
    transition_mode: str,
    enable_parent_relation: bool,
):
    return mifrost.FlatHorizonEncoder(
        domain,
        transition_mode=transition_mode,
        enable_parent_relation=bool(enable_parent_relation),
        enable_sibling_relation=False,
        enable_cousin_relation=False,
        root_policy=mifrost.RootPolicy.exclude,
        ignore_actions=True,
        ignore_zero_arity_relations=False,
        goal_derivations={
            mifrost.GoalDerivation.satisfied,
            mifrost.GoalDerivation.unsatisfied,
            mifrost.GoalDerivation.added_satisfied,
            mifrost.GoalDerivation.added_unsatisfied,
        },
        max_goal_level=0,
    )


def _build_action_encoder(domain: Any):
    return mifrost.FlatRelationEncoder(
        domain,
        target_sources=["action"],
    )


def _build_execution_policy(mode: str) -> FlatExecutionPolicy:
    normalized = mode.strip().lower()
    if normalized == "auto":
        return FlatExecutionPolicy(
            relation_kernels="auto",
            program_kernels="auto",
            relation_gather="auto",
        )
    if normalized == "eager":
        return FlatExecutionPolicy(
            relation_kernels="off",
            program_kernels="off",
            relation_gather="off",
        )
    raise ValueError(f"Unsupported execution mode: {mode!r}.")


def _make_relation_module(
    *,
    width: int,
    family: str,
    hidden: int | None,
) -> torch.nn.Module:
    block_hidden = int(hidden) if hidden is not None else int(width)
    normalized = family.strip().lower()
    if normalized == "mlp":
        return TwoLayerPointwiseRelationMLP(
            width=width,
            hidden=block_hidden,
            activation="mish",
        )
    if normalized == "prenorm_mlp":
        return PreNormTwoLayerPointwiseRelationMLP(
            width=width,
            hidden=block_hidden,
            activation="mish",
            norm="rmsnorm",
        )
    if normalized == "postnorm_mlp":
        return PostNormTwoLayerPointwiseRelationMLP(
            width=width,
            hidden=block_hidden,
            activation="mish",
            norm="layernorm",
        )
    if normalized == "program":
        return RelationProgram(
            TwoLayerPointwiseRelationMLP(
                width=width,
                hidden=block_hidden,
                activation="mish",
            ),
            TwoLayerPointwiseRelationMLP(
                width=width,
                hidden=block_hidden,
                activation="mish",
            ),
        )
    raise ValueError(f"Unsupported relation block family: {family!r}.")


def _build_model(
    *,
    encoder: Any,
    embedding_size: int,
    num_layers: int,
    aggregation: str,
    relation_block_family: str,
    relation_hidden: int | None,
    execution_mode: str,
    device: torch.device,
) -> torch.nn.Module:
    relations = {
        str(name): int(arity)
        for name, arity in zip(
            tuple(encoder.relation_names),
            tuple(encoder.relation_arities),
            strict=True,
        )
    }
    relation_modules = {
        name: _make_relation_module(
            width=int(arity) * int(embedding_size),
            family=relation_block_family,
            hidden=relation_hidden,
        )
        for name, arity in relations.items()
    }
    return FlatRelationalGNN(
        embedding_size=int(embedding_size),
        num_layers=int(num_layers),
        relations=relations,
        aggregation=_normalized_aggregation(aggregation),
        relation_modules=relation_modules,
        execution_policy=_build_execution_policy(execution_mode),
    ).to(device)


def _stats(times_ms: Sequence[float]) -> TimingStats:
    return TimingStats(
        mean_ms=float(statistics.fmean(times_ms)),
        median_ms=float(statistics.median(times_ms)),
        min_ms=float(min(times_ms)),
        max_ms=float(max(times_ms)),
    )


def _sum_embeddings_by_batch(
    embeddings: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    num_graphs = int(batch.max().item()) + 1 if int(batch.numel()) else 0
    summed = torch.zeros(
        (num_graphs, embeddings.shape[1]),
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    if num_graphs > 0:
        summed.index_add_(0, batch.long(), embeddings)
    return summed


def _score_action_targets(output: Any) -> torch.Tensor:
    object_embeddings = output.object
    object_batch = output.object_batch
    target_embeddings = output.target
    target_batch = output.target_batch
    if (
        object_embeddings is None
        or object_batch is None
        or target_embeddings is None
        or target_batch is None
    ):
        raise RuntimeError(
            "Expected object/target embeddings and batch indices from FlatRelationalGNN."
        )
    if int(target_embeddings.numel()) == 0:
        return torch.empty(
            0,
            dtype=object_embeddings.dtype,
            device=object_embeddings.device,
        )
    aggregated_objects = _sum_embeddings_by_batch(object_embeddings, object_batch)
    repeated_objects = aggregated_objects.index_select(0, target_batch.long())
    return (target_embeddings * repeated_objects).sum(dim=1)


def _validate_horizon_target_order(
    encoding: Any,
    *,
    expected_counts: Sequence[int],
) -> None:
    target_sizes = getattr(encoding, "target_sizes", None)
    target_candidate_ids = getattr(encoding, "target_candidate_ids", None)
    if target_sizes is None or target_candidate_ids is None:
        raise RuntimeError("Expected target_sizes and target_candidate_ids on horizon encoding.")
    if int(target_sizes.numel()) != len(expected_counts):
        raise RuntimeError(
            "Mismatch between encoded target_sizes and expected graph count: "
            f"{int(target_sizes.numel())} vs {len(expected_counts)}."
        )
    cursor = 0
    for graph_index, expected_count in enumerate(expected_counts):
        graph_size = int(target_sizes[graph_index].item())
        if graph_size != int(expected_count):
            raise RuntimeError(
                "Mismatch between target_sizes and candidate count for graph "
                f"{graph_index}: {graph_size} vs {expected_count}."
            )
        graph_ids = target_candidate_ids[cursor : cursor + expected_count]
        expected_ids = torch.arange(
            expected_count,
            device=graph_ids.device,
            dtype=graph_ids.dtype,
        )
        if not torch.equal(graph_ids, expected_ids):
            raise RuntimeError(
                "Horizon target rows are expected to preserve candidate order, "
                f"but graph {graph_index} emitted {graph_ids.tolist()!r}."
            )
        cursor += expected_count
    if cursor != int(target_candidate_ids.numel()):
        raise RuntimeError(
            "Unused target_candidate_ids remained after validation: "
            f"{cursor} vs {int(target_candidate_ids.numel())}."
        )


def _validate_action_target_sizes(
    encoding: Any,
    *,
    expected_counts: Sequence[int],
) -> None:
    target_sizes = getattr(encoding, "target_sizes", None)
    if target_sizes is None:
        return
    if int(target_sizes.numel()) != len(expected_counts):
        raise RuntimeError(
            "Mismatch between encoded target_sizes and expected graph count: "
            f"{int(target_sizes.numel())} vs {len(expected_counts)}."
        )
    for graph_index, expected_count in enumerate(expected_counts):
        graph_size = int(target_sizes[graph_index].item())
        if graph_size != int(expected_count):
            raise RuntimeError(
                "Mismatch between action target_sizes and applicable action count for graph "
                f"{graph_index}: {graph_size} vs {expected_count}."
            )


def _build_encoding_stats(encoding: Any) -> tuple[int, int, int, int, int]:
    relation_counts = getattr(encoding, "relation_counts", None)
    relation_args = getattr(encoding, "relation_args", None)
    x = getattr(encoding, "x", None)
    object_indices = getattr(encoding, "object_indices", None)
    target_candidate_ids = getattr(encoding, "target_candidate_ids", None)
    num_entities = int(x.shape[0]) if torch.is_tensor(x) else 0
    num_objects = int(object_indices.numel()) if torch.is_tensor(object_indices) else 0
    target_sizes = getattr(encoding, "target_sizes", None)
    target_batch = getattr(encoding, "target_batch", None)
    if torch.is_tensor(target_candidate_ids):
        num_targets = int(target_candidate_ids.numel())
    elif torch.is_tensor(target_sizes):
        num_targets = int(target_sizes.sum().item())
    elif torch.is_tensor(target_batch):
        num_targets = int(target_batch.numel())
    else:
        num_targets = 0
    relation_instances = int(relation_counts.sum().item()) if torch.is_tensor(relation_counts) else 0
    relation_args_count = int(relation_args.numel()) if torch.is_tensor(relation_args) else 0
    return num_entities, num_objects, num_targets, relation_instances, relation_args_count


def _benchmark_shared_candidate_generation(
    sampled_states: Sequence[SampledState],
    *,
    candidate_source: str,
    iw_kwargs: Mapping[str, Any],
    warmup: int,
    iterations: int,
) -> tuple[TimingStats, list[Any]]:
    times_ms: list[float] = []
    reference: list[CandidateBatch] | None = None
    for iteration_index in range(int(warmup) + int(iterations)):
        start = time.perf_counter()
        if candidate_source == "iw":
            current = [
                _generate_iw_actions(sample.state, **iw_kwargs)
                for sample in sampled_states
            ]
        elif candidate_source in {"actions", "applicable"}:
            current = [
                _generate_applicable_actions(sample.state)
                for sample in sampled_states
            ]
        else:
            raise ValueError(f"Unsupported candidate_source: {candidate_source!r}.")
        end = time.perf_counter()
        if iteration_index >= int(warmup):
            times_ms.append((end - start) * 1_000.0)
        if reference is None:
            reference = current
    if reference is None:
        raise RuntimeError("Failed to build reference candidate batches.")
    return _stats(times_ms), reference


def _benchmark_domain_mode(
    *,
    device: torch.device,
    encoder: Any,
    model: torch.nn.Module,
    sampled_states: Sequence[SampledState],
    candidate_source: str,
    candidate_batches: Sequence[Any],
    warmup: int,
    iterations: int,
    backward: bool,
    input_mode: str,
) -> tuple[dict[str, TimingStats], DomainWorkloadStats, str, str | None]:
    roots = [sample.state for sample in sampled_states]
    goals = [_goal_literals(sample.goal_condition) for sample in sampled_states]
    normalized_candidate_source = candidate_source.strip().lower()
    if normalized_candidate_source == "iw":
        expected_counts = [len(transitions) for transitions, _edges in candidate_batches]
    elif normalized_candidate_source in {"actions", "applicable"}:
        expected_counts = [len(actions) for actions in candidate_batches]
    else:
        raise ValueError(f"Unsupported candidate_source: {candidate_source!r}.")

    dag_times: list[float] = []
    encode_times: list[float] = []
    input_prepare_times: list[float] = []
    to_device_times: list[float] = []
    forward_times: list[float] = []
    backward_times: list[float] = []

    if normalized_candidate_source == "iw":
        inspect_dags = [
            _build_iw_horizon_dag(root, transitions, edges)
            for root, (transitions, edges) in zip(roots, candidate_batches, strict=True)
        ]
        inspect_encoding = encoder.encode_batch(
            roots,
            dags=inspect_dags,
            goals=goals,
            target_sources=[mifrost.TargetSource.states],
        )
        _validate_horizon_target_order(inspect_encoding, expected_counts=expected_counts)
    else:
        inspect_encoding = encoder.encode_batch(
            states=roots,
            goals=goals,
            actions=candidate_batches,
        )
        _validate_action_target_sizes(inspect_encoding, expected_counts=expected_counts)
    (
        num_entities,
        num_objects,
        num_targets,
        relation_instances,
        relation_args_count,
    ) = _build_encoding_stats(inspect_encoding)
    resolved_input_mode = input_mode.strip().lower()
    native_input_error: str | None = None
    if resolved_input_mode not in {"auto", "native", "pyg"}:
        raise ValueError(f"Unsupported input_mode: {input_mode!r}.")
    if resolved_input_mode in {"auto", "native"}:
        try:
            with torch.no_grad():
                _ = model(inspect_encoding.to(device))
            resolved_input_mode = "native"
        except Exception as exc:
            native_input_error = f"{type(exc).__name__}: {exc}"
            if resolved_input_mode == "native":
                raise
            resolved_input_mode = "pyg"
    if resolved_input_mode == "pyg":
        with torch.no_grad():
            _ = model(inspect_encoding.as_pyg(as_batch=True).to(device))
    rollout_depths = [sample.rollout_depth for sample in sampled_states]
    candidate_counts = expected_counts
    workload_stats = DomainWorkloadStats(
        num_states=len(sampled_states),
        num_problems=len({sample.problem_name for sample in sampled_states}),
        rollout_depth_mean=float(statistics.fmean(rollout_depths)),
        rollout_depth_max=max(rollout_depths) if rollout_depths else 0,
        total_candidates=sum(candidate_counts),
        mean_candidates=float(statistics.fmean(candidate_counts)) if candidate_counts else 0.0,
        max_candidates=max(candidate_counts) if candidate_counts else 0,
        num_entities=num_entities,
        num_objects=num_objects,
        num_targets=num_targets,
        relation_instances=relation_instances,
        relation_args=relation_args_count,
    )

    for iteration_index in range(int(warmup) + int(iterations)):
        model.zero_grad(set_to_none=True)
        _sync_if_needed(device)

        t0 = time.perf_counter()
        if normalized_candidate_source == "iw":
            dags = [
                _build_iw_horizon_dag(root, transitions, edges)
                for root, (transitions, edges) in zip(roots, candidate_batches, strict=True)
            ]
        else:
            dags = None
        _sync_if_needed(device)
        t1 = time.perf_counter()

        if normalized_candidate_source == "iw":
            encoding = encoder.encode_batch(
                roots,
                dags=dags,
                goals=goals,
                target_sources=[mifrost.TargetSource.states],
            )
        else:
            encoding = encoder.encode_batch(
                states=roots,
                goals=goals,
                actions=candidate_batches,
            )
        _sync_if_needed(device)
        t2 = time.perf_counter()

        if resolved_input_mode == "pyg":
            model_input = encoding.as_pyg(as_batch=True)
        else:
            model_input = encoding
        _sync_if_needed(device)
        t3 = time.perf_counter()

        model_input = model_input.to(device)
        _sync_if_needed(device)
        t4 = time.perf_counter()

        output = model(model_input)
        scores = _score_action_targets(output)
        loss = scores.sum()
        _sync_if_needed(device)
        t5 = time.perf_counter()

        if bool(backward):
            loss.backward()
            _sync_if_needed(device)
            t6 = time.perf_counter()
        else:
            t6 = t5

        if iteration_index >= int(warmup):
            dag_times.append((t1 - t0) * 1_000.0)
            encode_times.append((t2 - t1) * 1_000.0)
            input_prepare_times.append((t3 - t2) * 1_000.0)
            to_device_times.append((t4 - t3) * 1_000.0)
            forward_times.append((t5 - t4) * 1_000.0)
            backward_times.append((t6 - t5) * 1_000.0)

    timings = {
        "dag_build": _stats(dag_times),
        "encode": _stats(encode_times),
        "input_prepare": _stats(input_prepare_times),
        "encoded_to_device": _stats(to_device_times),
        "forward": _stats(forward_times),
    }
    if bool(backward):
        timings["backward"] = _stats(backward_times)
    return timings, workload_stats, resolved_input_mode, native_input_error


def _throughput(count: int, total_ms: float) -> float:
    if total_ms <= 0.0:
        return 0.0
    return float(count) / (float(total_ms) / 1_000.0)


def _stage_total_ms(timings: Mapping[str, TimingStats]) -> float:
    return float(sum(stats.mean_ms for stats in timings.values()))


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["candidate_source"]),
            str(row["relation_block_family"]),
            str(row["execution_mode"]),
        )
        grouped.setdefault(key, []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (candidate_source, family, mode), group in sorted(grouped.items()):
        total_candidates = sum(int(row["workload"]["total_candidates"]) for row in group)
        total_relation_args = sum(int(row["workload"]["relation_args"]) for row in group)
        mean_total_ms = float(statistics.fmean(float(row["total_mean_ms"]) for row in group))
        mean_end_to_end_ms = float(
            statistics.fmean(float(row["end_to_end_mean_ms"]) for row in group)
        )
        summed_total_ms = float(sum(float(row["total_mean_ms"]) for row in group))
        summed_end_to_end_ms = float(sum(float(row["end_to_end_mean_ms"]) for row in group))
        aggregates.append(
            {
                "candidate_source": candidate_source,
                "relation_block_family": family,
                "execution_mode": mode,
                "domains": [str(row["domain"]) for row in group],
                "mean_total_ms": mean_total_ms,
                "mean_end_to_end_ms": mean_end_to_end_ms,
                "aggregate_candidates_per_s": _throughput(total_candidates, summed_total_ms),
                "aggregate_candidates_per_s_end_to_end": _throughput(
                    total_candidates,
                    summed_end_to_end_ms,
                ),
                "aggregate_relation_args_per_s": _throughput(
                    total_relation_args,
                    summed_total_ms,
                ),
            }
        )
    return aggregates


def _print_row(prefix: str, row: Mapping[str, Any]) -> None:
    mode_note = f"[{row['input_mode']}]"
    print(
        f"{prefix:<20} {mode_note:<7} "
        f"total={float(row['total_mean_ms']):8.3f} ms "
        f"e2e={float(row['end_to_end_mean_ms']):8.3f} ms "
        f"cand/s={float(row['candidates_per_s']):10.1f} "
        f"cand/s(e2e)={float(row['candidates_per_s_end_to_end']):10.1f}"
    )
    native_input_error = row.get("native_input_error")
    if native_input_error:
        print(f"{'':<20} native_fallback={native_input_error}")


def _parse_args() -> argparse.Namespace:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark relmo FlatHorizonEncoder + FlatRelationalGNN workloads "
            "on mixed IPC train splits with IW lookaheads."
        )
    )
    parser.add_argument(
        "--pddl-root",
        type=str,
        default=str(repo_root / "data" / "pddl_domains"),
        help="Root containing *-ipc domain directories.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="auto",
        help="Comma-separated domain list or 'auto' for every *-ipc/train split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string or 'auto'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of sampled states per domain batch.",
    )
    parser.add_argument(
        "--max-rollout-depth",
        type=int,
        default=4,
        help="Maximum random rollout depth when sampling train states.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per measurement lane.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Timed iterations per measurement lane.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=122276,
        help="Base random seed.",
    )
    parser.add_argument(
        "--embedding-size",
        type=int,
        default=32,
        help="Embedding size for FlatRelationalGNN.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=10,
        help="Number of message-passing layers.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="smax",
        help="Aggregation name; smax/add/hmax are normalized for relmo.",
    )
    parser.add_argument(
        "--execution-modes",
        type=str,
        default="auto,eager",
        help="Comma-separated execution modes to compare.",
    )
    parser.add_argument(
        "--candidate-sources",
        type=str,
        default="iw,actions",
        help="Comma-separated candidate sources: iw and/or actions (applicable is accepted as an alias).",
    )
    parser.add_argument(
        "--relation-block-families",
        type=str,
        default="mlp,prenorm_mlp,postnorm_mlp,program",
        help="Comma-separated relation block families to benchmark.",
    )
    parser.add_argument(
        "--relation-hidden",
        type=int,
        default=None,
        help="Override hidden width for relation blocks.",
    )
    parser.add_argument(
        "--transition-mode",
        choices=("delta", "full"),
        default="delta",
        help="Transition payload emitted by FlatHorizonEncoder.",
    )
    parser.add_argument(
        "--input-mode",
        choices=("auto", "native", "pyg"),
        default="auto",
        help="Use native mifrost batches, PyG batches, or auto-fallback to PyG when native is broken.",
    )
    parser.add_argument(
        "--disable-parent-relation",
        action="store_true",
        help="Disable the horizon parent relation. Default keeps IW tree topology.",
    )
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="Skip backward timing.",
    )
    parser.add_argument("--projective-iw", action="store_true")
    parser.add_argument("--typed-projection", action="store_true")
    parser.add_argument(
        "--keep-depth-one-novel",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--keep-goal-nonunary-atoms", action="store_true")
    parser.add_argument("--iw-layer-ordering-strategy", type=str, default=None)
    parser.add_argument("--iw-max-next-layer-states", type=int, default=-1)
    parser.add_argument("--iw-beam-width", type=int, default=-1)
    parser.add_argument("--iw-beam-novelty-mode", type=str, default="all_tested")
    parser.add_argument("--iw-max-depth", type=int, default=-1)
    parser.add_argument("--iw-relaxed-survivors-only-beam", action="store_true")
    parser.add_argument("--iw-randomize-equal-score-ties", action="store_true")
    parser.add_argument("--iw-equal-score-tie-seed", type=int, default=None)
    parser.add_argument("--iw-num-threads", type=int, default=0)
    parser.add_argument("--iw-chunk-size", type=int, default=-1)
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path for JSON results.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    device = _resolve_device(str(args.device))
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(int(args.seed))
    torch.manual_seed(int(args.seed))

    pddl_root = pathlib.Path(args.pddl_root).expanduser().resolve()
    train_dirs = _discover_train_dirs(pddl_root, str(args.domains))
    execution_modes = [part.strip() for part in str(args.execution_modes).split(",") if part.strip()]
    candidate_sources = []
    for part in str(args.candidate_sources).split(","):
        normalized = part.strip().lower()
        if not normalized:
            continue
        if normalized == "applicable":
            normalized = "actions"
        candidate_sources.append(normalized)
    relation_block_families = [
        part.strip()
        for part in str(args.relation_block_families).split(",")
        if part.strip()
    ]
    if not execution_modes:
        raise RuntimeError("No execution modes requested.")
    if not candidate_sources:
        raise RuntimeError("No candidate sources requested.")
    if not relation_block_families:
        raise RuntimeError("No relation block families requested.")

    iw_kwargs = {
        "include_start": False,
        "projective_iw1": bool(args.projective_iw),
        "typed_projection": bool(args.typed_projection),
        "keep_depth_one_novel": bool(args.keep_depth_one_novel),
        "keep_goal_nonunary_atoms": bool(args.keep_goal_nonunary_atoms),
        "layer_ordering_strategy_name": args.iw_layer_ordering_strategy,
        "max_next_layer_states": int(args.iw_max_next_layer_states),
        "beam_width": int(args.iw_beam_width),
        "beam_novelty_mode": str(args.iw_beam_novelty_mode),
        "max_depth": int(args.iw_max_depth),
        "relaxed_survivors_only_beam": bool(args.iw_relaxed_survivors_only_beam),
        "randomize_equal_score_ties": bool(args.iw_randomize_equal_score_ties),
        "equal_score_tie_seed": args.iw_equal_score_tie_seed,
        "num_threads": int(args.iw_num_threads),
        "chunk_size": int(args.iw_chunk_size),
    }

    all_rows: list[dict[str, Any]] = []
    print(f"device={device} pddl_root={pddl_root}")
    print(
        f"batch_size={int(args.batch_size)} max_rollout_depth={int(args.max_rollout_depth)} "
        f"warmup={int(args.warmup)} iterations={int(args.iterations)} backward={not bool(args.forward_only)}"
    )
    print(
        f"candidate_sources={candidate_sources} execution_modes={execution_modes} "
        f"relation_block_families={relation_block_families} "
        f"transition_mode={args.transition_mode} input_mode={args.input_mode} "
        f"parent_relation={not bool(args.disable_parent_relation)}"
    )

    for domain_index, (domain_name, train_dir) in enumerate(train_dirs):
        domain_path = train_dir / "domain.pddl"
        domain = mm.Domain(str(domain_path))
        problem_paths = _load_problem_paths(train_dir)
        sampled_states = _sample_states(
            domain,
            problem_paths,
            batch_size=int(args.batch_size),
            max_rollout_depth=int(args.max_rollout_depth),
            seed=int(args.seed) + domain_index,
        )
        print()
        print(f"[{domain_name}] train_dir={train_dir}")

        encoders_by_source: dict[str, Any] = {}
        if "iw" in candidate_sources:
            encoders_by_source["iw"] = _build_horizon_encoder(
                domain,
                transition_mode=str(args.transition_mode),
                enable_parent_relation=not bool(args.disable_parent_relation),
            )
        if "actions" in candidate_sources:
            encoders_by_source["actions"] = _build_action_encoder(domain)

        for candidate_source in candidate_sources:
            encoder = encoders_by_source[candidate_source]
            candidate_timing, candidate_batches = _benchmark_shared_candidate_generation(
                sampled_states,
                candidate_source=candidate_source,
                iw_kwargs=iw_kwargs,
                warmup=int(args.warmup),
                iterations=int(args.iterations),
            )
            print(
                f"  {candidate_source}_generate mean={candidate_timing.mean_ms:.3f} ms "
                f"median={candidate_timing.median_ms:.3f} ms"
            )
            for relation_block_family in relation_block_families:
                for execution_mode in execution_modes:
                    torch.manual_seed(int(args.seed))
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(int(args.seed))
                    model = _build_model(
                        encoder=encoder,
                        embedding_size=int(args.embedding_size),
                        num_layers=int(args.num_layers),
                        aggregation=str(args.aggregation),
                        relation_block_family=relation_block_family,
                        relation_hidden=args.relation_hidden,
                        execution_mode=execution_mode,
                        device=device,
                    )
                    (
                        timings,
                        workload,
                        resolved_input_mode,
                        native_input_error,
                    ) = _benchmark_domain_mode(
                        device=device,
                        encoder=encoder,
                        model=model,
                        sampled_states=sampled_states,
                        candidate_source=candidate_source,
                        candidate_batches=candidate_batches,
                        warmup=int(args.warmup),
                        iterations=int(args.iterations),
                        backward=not bool(args.forward_only),
                        input_mode=str(args.input_mode),
                    )
                    total_mean_ms = _stage_total_ms(timings)
                    end_to_end_mean_ms = float(candidate_timing.mean_ms + total_mean_ms)
                    row = {
                        "domain": domain_name,
                        "train_dir": str(train_dir),
                        "candidate_source": candidate_source,
                        "relation_block_family": relation_block_family,
                        "execution_mode": execution_mode,
                        "input_mode": resolved_input_mode,
                        "native_input_error": native_input_error,
                        "candidate_generate_ms": asdict(candidate_timing),
                        "timings_ms": {name: asdict(stats) for name, stats in timings.items()},
                        "workload": asdict(workload),
                        "total_mean_ms": total_mean_ms,
                        "end_to_end_mean_ms": end_to_end_mean_ms,
                        "candidates_per_s": _throughput(workload.total_candidates, total_mean_ms),
                        "candidates_per_s_end_to_end": _throughput(
                            workload.total_candidates,
                            end_to_end_mean_ms,
                        ),
                        "relation_args_per_s": _throughput(workload.relation_args, total_mean_ms),
                    }
                    all_rows.append(row)
                    _print_row(
                        f"  {candidate_source}/{relation_block_family}/{execution_mode}",
                        row,
                    )

    aggregates = _aggregate_rows(all_rows)
    print()
    print("Aggregate")
    for aggregate in aggregates:
        print(
            f"  {aggregate['candidate_source']}/{aggregate['relation_block_family']}/{aggregate['execution_mode']:<10} "
            f"mean_total={aggregate['mean_total_ms']:8.3f} ms "
            f"mean_e2e={aggregate['mean_end_to_end_ms']:8.3f} ms "
            f"cand/s={aggregate['aggregate_candidates_per_s']:10.1f} "
            f"cand/s(e2e)={aggregate['aggregate_candidates_per_s_end_to_end']:10.1f}"
        )

    aggregate_by_family: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for aggregate in aggregates:
        aggregate_by_family.setdefault(str(aggregate["candidate_source"]), {}).setdefault(
            str(aggregate["relation_block_family"]), {}
        )[str(aggregate["execution_mode"])] = aggregate
    print()
    print("Kernel Verdict")
    for candidate_source in candidate_sources:
        for family in relation_block_families:
            eager = aggregate_by_family.get(candidate_source, {}).get(family, {}).get("eager")
            auto = aggregate_by_family.get(candidate_source, {}).get(family, {}).get("auto")
            if eager is None or auto is None:
                continue
            total_ratio = float(eager["mean_total_ms"]) / float(auto["mean_total_ms"])
            e2e_ratio = float(eager["mean_end_to_end_ms"]) / float(auto["mean_end_to_end_ms"])
            print(
                f"  {candidate_source}/{family:<14} auto_vs_eager_total={total_ratio:6.3f}x "
                f"auto_vs_eager_e2e={e2e_ratio:6.3f}x"
            )

    payload = {
        "config": {
            "pddl_root": str(pddl_root),
            "domains": [domain_name for domain_name, _train_dir in train_dirs],
            "device": str(device),
            "batch_size": int(args.batch_size),
            "max_rollout_depth": int(args.max_rollout_depth),
            "warmup": int(args.warmup),
            "iterations": int(args.iterations),
            "seed": int(args.seed),
            "embedding_size": int(args.embedding_size),
            "num_layers": int(args.num_layers),
            "aggregation": str(args.aggregation),
            "candidate_sources": candidate_sources,
            "execution_modes": execution_modes,
            "relation_block_families": relation_block_families,
            "relation_hidden": args.relation_hidden,
            "transition_mode": str(args.transition_mode),
            "input_mode": str(args.input_mode),
            "parent_relation": not bool(args.disable_parent_relation),
            "backward": not bool(args.forward_only),
            "iw_kwargs": iw_kwargs,
        },
        "rows": all_rows,
        "aggregates": aggregates,
    }
    if args.json_out:
        out_path = pathlib.Path(args.json_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"json_out={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
