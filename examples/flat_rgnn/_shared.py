from __future__ import annotations

from pathlib import Path

import mifrost
import pymimir

from relmo.models import FlatExecutionPolicy
from relmo.models.builders import (
    EAGER_POLICY,
    RelationProgram,
    TwoLayerPointwiseRelationMLP,
    build_eager_fallback_modules,
    build_program_relation_modules,
    build_relations,
    build_typed_relation_modules,
)


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
