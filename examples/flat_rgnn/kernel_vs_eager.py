from __future__ import annotations

import mifrost

from relmo.models import FlatRelationalGNN

from _shared import (
    EAGER_POLICY,
    KERNEL_POLICY,
    build_eager_fallback_modules,
    build_program_relation_modules,
    build_relations,
    build_typed_relation_modules,
    load_problem,
    print_output,
)


def main() -> None:
    domain, problem, state, goals, actions = load_problem()
    state_b = actions[0].apply(state)

    encoder = mifrost.FlatRelationEncoder(domain, target_sources=["goal", "action"])
    batch = encoder.encode_batch(
        states=[state, state_b],
        goals=[goals, goals],
        actions=[list(state.generate_applicable_actions())[:1], list(state_b.generate_applicable_actions())[:2]],
    )
    relations = build_relations(batch)

    # 1. Typed block modules: eligible for exact relation kernels.
    typed_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=build_typed_relation_modules(relations),
        execution_policy=KERNEL_POLICY,
    )
    print_output("typed-blocks/kernel-policy", typed_model(batch))

    # 2. Arbitrary torch.nn.Module graph: valid, but eager fallback only.
    eager_fallback_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=build_eager_fallback_modules(relations),
        execution_policy=KERNEL_POLICY,
    )
    print_output("custom-sequential/kernel-policy-falls-back", eager_fallback_model(batch))

    # 3. RelationProgram: explicit composed program surface.
    program_model = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=build_program_relation_modules(relations),
        execution_policy=KERNEL_POLICY,
    )
    print_output("relation-program/kernel-policy", program_model(batch))

    # 4. Fully eager reference configuration.
    eager_reference = FlatRelationalGNN(
        embedding_size=32,
        num_layers=4,
        relations=relations,
        aggregation="sum",
        relation_modules=build_typed_relation_modules(relations),
        execution_policy=EAGER_POLICY,
    )
    print_output("typed-blocks/eager-policy", eager_reference(batch))


if __name__ == "__main__":
    main()
