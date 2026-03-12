from __future__ import annotations

from pathlib import Path
from typing import Mapping

import mifrost
import pymimir
import torch

from relm.models import FlatExecutionPolicy, TwoLayerPointwiseRelationMLP, RelationProgram


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAIN_PATH = REPO_ROOT / "data" / "pddl_domains" / "blocks" / "domain.pddl"
DEFAULT_PROBLEM_PATH = REPO_ROOT / "data" / "pddl_domains" / "blocks" / "blocks_b-7_v-1.pddl"


def load_problem(
    domain_path: Path = DEFAULT_DOMAIN_PATH,
    problem_path: Path = DEFAULT_PROBLEM_PATH,
):
    domain = pymimir.Domain(domain_path)
    problem = pymimir.Problem(domain, problem_path, mode="lifted")
    state = problem.get_initial_state()
    goals = list(problem.get_goal_condition().get_literals())
    actions = list(state.generate_applicable_actions())
    return domain, problem, state, goals, actions


def build_relations(batch) -> dict[str, int]:
    return {
        name: int(arity)
        for name, arity in zip(batch.relation_names, batch.relation_arities)
    }


def build_typed_relation_modules(
    relations: Mapping[str, int],
    *,
    embedding_size: int = 32,
    activation: str = "silu",
):
    modules = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(64, width)
        modules[name] = TwoLayerPointwiseRelationMLP(
            width=width,
            hidden=hidden,
            activation=activation,
        )
    return modules


def build_program_relation_modules(
    relations: Mapping[str, int],
    *,
    embedding_size: int = 32,
    activation: str = "silu",
):
    modules = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(64, width)
        modules[name] = RelationProgram(
            TwoLayerPointwiseRelationMLP(width=width, hidden=hidden, activation=activation),
            TwoLayerPointwiseRelationMLP(width=width, hidden=hidden, activation=activation),
        )
    return modules


def build_eager_fallback_modules(
    relations: Mapping[str, int],
    *,
    embedding_size: int = 32,
):
    modules = {}
    for name, arity in relations.items():
        width = int(arity) * embedding_size
        hidden = max(64, width)
        modules[name] = torch.nn.Sequential(
            torch.nn.Linear(width, hidden),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden, width),
        )
    return modules


EAGER_POLICY = FlatExecutionPolicy(
    relation_kernels="off",
    program_kernels="off",
    relation_gather="off",
)

KERNEL_POLICY = FlatExecutionPolicy(
    relation_kernels="auto",
    program_kernels="auto",
    relation_gather="auto",
)


def print_output(label: str, output) -> None:
    print(f"== {label} ==")
    print("entity:", tuple(output.entity.shape))
    print("object:", None if output.object is None else tuple(output.object.shape))
    print(
        "target_entity:",
        None if output.target_entity is None else tuple(output.target_entity.shape),
    )
    print("target:", None if output.target is None else tuple(output.target.shape))
