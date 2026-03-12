from __future__ import annotations

import mifrost

from relm.models import (
    CentralizedFlatRelationalGNN,
    FlatLGANRelationalGNN,
    FlatRelationalGNN,
)

from _shared import EAGER_POLICY, build_relations, build_typed_relation_modules, load_problem, print_output


def main() -> None:
    domain, problem, state_a, goals, actions = load_problem()
    state_b = actions[0].apply(state_a)
    successor_a = actions[0].apply(state_a)
    successor_b = actions[0].apply(state_b)

    # 1. plain state + goals
    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal"])
    single = encoder.encode(state_a, goals=goals)
    batch = encoder.encode_batch(states=[state_a, state_b], goals=[goals, goals])
    relations = build_relations(single)
    modules = build_typed_relation_modules(relations)
    plain_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=modules,
        execution_policy=EAGER_POLICY,
    )
    print_output("plain/single", plain_model(single))
    print_output("plain/batch", plain_model(batch))

    # 2. actions co-encoded
    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal", "action"])
    single = encoder.encode(state_a, goals=goals, actions=actions[:2])
    batch = encoder.encode_batch(
        states=[state_a, state_b],
        goals=[goals, goals],
        actions=[list(state_a.generate_applicable_actions())[:1], list(state_b.generate_applicable_actions())[:2]],
    )
    relations = build_relations(single)
    modules = build_typed_relation_modules(relations)
    action_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=modules,
        execution_policy=EAGER_POLICY,
    )
    print_output("actions/single", action_model(single))
    print_output("actions/batch", action_model(batch))

    # 3. subgoals co-encoded
    encoder = mifrost.FlatRelationEncoder(
        domain,
        max_goal_level=1,
        target_sources=["goal", "subgoal"],
    )
    single = encoder.encode(state_a, goals=goals, subgoal_layers=[[goals[0]]])
    batch = encoder.encode_batch(
        states=[state_a, state_b],
        goals=[goals, goals],
        subgoal_layers=[[[goals[0]]], [[goals[0]]]],
    )
    relations = build_relations(single)
    modules = build_typed_relation_modules(relations)
    subgoal_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=modules,
        execution_policy=EAGER_POLICY,
    )
    print_output("subgoals/single", subgoal_model(single))
    print_output("subgoals/batch", subgoal_model(batch))

    # 4. history co-encoded
    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal", "history"])
    single = encoder.encode(
        state_a,
        goals=goals,
        history_subgoals=[(-1, [goals[0]])],
    )
    batch = encoder.encode_batch(
        states=[state_a, state_b],
        goals=[goals, goals],
        history_subgoals=[
            [(-1, [goals[0]])],
            [(-1, [goals[0]]), (-2, [goals[0]])],
        ],
    )
    relations = build_relations(single)
    modules = build_typed_relation_modules(relations)
    history_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=modules,
        execution_policy=EAGER_POLICY,
    )
    print_output("history/single", history_model(single))
    print_output("history/batch", history_model(batch))

    # 5. centralized flat RGNN on the same action-aware encoding
    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal", "action"])
    batch = encoder.encode_batch(
        states=[state_a, state_b],
        goals=[goals, goals],
        actions=[list(state_a.generate_applicable_actions())[:1], list(state_b.generate_applicable_actions())[:2]],
    )
    relations = build_relations(batch)
    centralized = CentralizedFlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        central_conditioning="film",
        execution_policy=EAGER_POLICY,
    )
    print_output("centralized/batch", centralized(batch))

    # 6. flat LGAN with explicit goal anchors
    encoder = mifrost.FlatRelationEncoder(
        domain,
        include_lgan_edges=True,
        lgan_anchor_sources=["goal"],
        target_sources=["goal"],
    )
    single = encoder.encode(state_a, goals=goals)
    batch = encoder.encode_batch(states=[state_a, state_b], goals=[goals, goals])
    relations = build_relations(single)
    modules = build_typed_relation_modules(relations)
    lgan_model = FlatLGANRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=modules,
        execution_policy=EAGER_POLICY,
    )
    print_output("lgan/single", lgan_model(single))
    print_output("lgan/batch", lgan_model(batch))

    # 7. current state + goals + successor state
    encoder = mifrost.FlatTransitionEncoder(domain)
    single = encoder.encode(current=state_a, goals=goals, successor=successor_a)
    batch = encoder.encode_batch(
        states=[state_a, state_b],
        goals=[goals, goals],
        successors=[successor_a, successor_b],
    )
    relations = build_relations(single)
    modules = build_typed_relation_modules(relations)
    transition_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=modules,
        execution_policy=EAGER_POLICY,
    )
    print_output("transition/single", transition_model(single))
    print_output("transition/batch", transition_model(batch))


if __name__ == "__main__":
    main()
