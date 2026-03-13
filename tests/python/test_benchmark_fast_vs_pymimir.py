from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.benchmark_fast_vs_pymimir import (
    _build_flat_relation_instance_counts,
    _build_flat_relm_inputs_from_flat_data,
    _build_pymimir_relation_instance_counts,
    _build_states,
    _compare_mapped_relation_values,
    _extract_pymimir_relation_arities,
    _make_pymimir_model,
    _select_relm_parity_relations,
)


def _load_blocks_problem():
    pymimir = pytest.importorskip("pymimir")
    root = Path(__file__).resolve().parents[2] / "data" / "pddl_domains" / "blocks"
    domain = pymimir.Domain(root / "domain.pddl")
    problem_files = sorted(
        path for path in root.glob("*.pddl") if path.name != "domain.pddl"
    )
    if not problem_files:
        pytest.skip("no Blocks problem files available in test fixture root")
    problem = pymimir.Problem(domain, problem_files[0], mode="lifted")
    return domain, problem


def _native_goal_sat_relm_encoder(domain):
    try:
        import mifrost  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent editable rebuild path
        pytest.skip(f"mifrost unavailable in this test environment: {exc}")
    try:
        flat_relation_encoder = mifrost.FlatRelationEncoder
    except Exception as exc:  # pragma: no cover - env-dependent optional wrapper path
        pytest.skip(f"mifrost FlatRelationEncoder wrapper unavailable: {exc}")
    return flat_relation_encoder(
        domain,
        goal_satisfaction_derivations={
            mifrost.GoalSatisfaction.satisfied,
            mifrost.GoalSatisfaction.unsatisfied,
        },
    )


def _mapped_count_comparison_for_state(state) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    pymimir = pytest.importorskip("pymimir")
    pytest.importorskip("pymimir_rgnn")
    from pymimir_rgnn.encoders import get_input_from_encoders

    domain, problem = _load_blocks_problem()
    goals = list(problem.get_goal_condition().get_literals())
    goal_condition = problem.get_goal_condition()

    relm_encoder = _native_goal_sat_relm_encoder(domain)
    relm_data = relm_encoder.encode_pyg(state, goals=goals)
    relm_relation_dict = {
        str(name): int(arity)
        for name, arity in zip(relm_data.relation_names, relm_data.relation_arities)
    }

    pymimir_model, input_spec = _make_pymimir_model(
        domain=domain,
        embedding_size=8,
        num_layers=1,
        aggr="sum",
        include_goal=True,
        normalize_updates=False,
    )
    pymimir_arities = _extract_pymimir_relation_arities(
        pymimir_model=pymimir_model,
        embedding_size=8,
    )

    selected_relmo_relations, selection_meta = _select_relm_parity_relations(
        relm_relation_dict=relm_relation_dict,
        pymimir_relation_arities=pymimir_arities,
    )
    relm_payload = _build_flat_relm_inputs_from_flat_data(
        data=relm_data,
        keep_relations=selected_relmo_relations.keys(),
    )
    relm_counts = _build_flat_relation_instance_counts(flat_payload=relm_payload)

    pymimir_encoded = get_input_from_encoders(
        [(state, goal_condition)],
        input_spec,
        pymimir_model.get_device(),
    )
    pymimir_counts = _build_pymimir_relation_instance_counts(
        pymimir_relations=pymimir_encoded.flattened_relations,
        pymimir_relation_arities=pymimir_arities,
    )
    rows, mismatches = _compare_mapped_relation_values(
        mappings=selection_meta["mappings"],
        relm_values=relm_counts,
        pymimir_values=pymimir_counts,
    )
    return rows, mismatches, selection_meta


def test_native_goal_sat_unsat_mapping_matches_pymimir_on_initial_state() -> None:
    _domain, problem = _load_blocks_problem()
    rows, mismatches, meta = _mapped_count_comparison_for_state(problem.get_initial_state())

    assert not mismatches
    on_false = next(row for row in rows if row["pymimir_relation"] == "relation_on_goal_false")
    on_true = next(row for row in rows if row["pymimir_relation"] == "relation_on_goal_true")
    assert on_false["relm_relation"] == "[+]on[g][unsat]"
    assert on_true["relm_relation"] == "[+]on[g][sat]"
    assert meta["selected_count"] == 13


def test_native_goal_sat_unsat_mapping_matches_pymimir_on_partially_satisfied_state() -> None:
    try:
        _domain, problem, states = _build_states(
            pddl_root=str(Path(__file__).resolve().parents[2] / "data" / "pddl_domains"),
            domain_case="blocks",
            problem_case="probBLOCKS-4-0",
            max_states=16,
            seed=7,
        )
    except Exception as exc:  # pragma: no cover - env-dependent pymimir state-space issue
        pytest.skip(f"pymimir state-space sampling unavailable in this environment: {exc}")

    rows: list[dict[str, object]] | None = None
    mismatches: list[dict[str, object]] | None = None
    for state in states:
        candidate_rows, candidate_mismatches, _meta = _mapped_count_comparison_for_state(state)
        on_true = next(
            row for row in candidate_rows if row["pymimir_relation"] == "relation_on_goal_true"
        )
        if int(on_true["pymimir_value"]) > 0:
            rows = candidate_rows
            mismatches = candidate_mismatches
            break

    assert rows is not None, "Expected at least one sampled state with satisfied on-goal tuples."
    assert mismatches is not None
    assert not mismatches

    on_false = next(row for row in rows if row["pymimir_relation"] == "relation_on_goal_false")
    on_true = next(row for row in rows if row["pymimir_relation"] == "relation_on_goal_true")
    assert int(on_false["pymimir_value"]) == int(on_false["relm_value"])
    assert int(on_true["pymimir_value"]) == int(on_true["relm_value"])
    assert int(on_true["pymimir_value"]) > 0


def test_build_flat_relm_inputs_from_native_batchencoding_matches_pyg_batch() -> None:
    try:
        import mifrost  # type: ignore
    except Exception as exc:  # pragma: no cover - env-dependent editable rebuild path
        pytest.skip(f"mifrost unavailable in this test environment: {exc}")
    try:
        flat_relation_encoder = mifrost.FlatRelationEncoder
    except Exception as exc:  # pragma: no cover - env-dependent optional wrapper path
        pytest.skip(f"mifrost FlatRelationEncoder wrapper unavailable: {exc}")
    domain, problem = _load_blocks_problem()
    goals = list(problem.get_goal_condition().get_literals())
    state = problem.get_initial_state()
    encoder = flat_relation_encoder(
        domain,
        target_sources=["goal"],
        goal_satisfaction_derivations={
            mifrost.GoalSatisfaction.satisfied,
            mifrost.GoalSatisfaction.unsatisfied,
        },
    )
    native_batch = encoder.encode_batch(states=[state, state], goals=goals)
    pyg_batch = native_batch.as_pyg(as_batch=True)

    payload_native = _build_flat_relm_inputs_from_flat_data(data=native_batch)
    payload_pyg = _build_flat_relm_inputs_from_flat_data(data=pyg_batch)

    assert torch.equal(payload_native["x"], payload_pyg["x"])
    assert torch.equal(payload_native["relation_counts"], payload_pyg["relation_counts"])
    assert torch.equal(payload_native["relation_args"], payload_pyg["relation_args"])
    assert torch.equal(payload_native["relation_arities"], payload_pyg["relation_arities"])
    assert payload_native["relation_names"] == payload_pyg["relation_names"]
    assert torch.equal(payload_native["node_sizes"], payload_pyg["node_sizes"])
    assert torch.equal(payload_native["object_indices"], payload_pyg["object_indices"])
    assert torch.equal(
        payload_native["target_entity_indices"],
        payload_pyg["target_entity_indices"],
    )
    assert torch.equal(payload_native["target_positions"], payload_pyg["target_positions"])
