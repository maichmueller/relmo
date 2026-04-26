# RelationalGNN Optimization Plan

Date: 2026-03-05  
Scope: `relm` decentralized relational message passing (`modular`, `batched_cached`, `fast_fused`) compared against `pymimir-rgnn`.

## 0. Benchmark Parity Guardrails (Must-Have)

All performance claims in this document are valid only if benchmark parity is enforced:

1. Same relation set in both models.
2. Same arity per mapped relation.
3. Same per-relation MLP linear topology.
4. Same encoded input semantics (`state + goal` for this benchmark track).
5. Same aggregation mode, embedding size, layer count, device, and precision settings.

Operational requirement:
- Benchmark harness should fail fast when parity constraints are violated.
- JSON outputs should include explicit parity diagnostics (mapping, shape checks, encoded-count checks).

## 1. Baseline Snapshot (Reference Point)

Primary reference artifacts:
- `docs/profile_relm_vs_pymimir_causal_desk01.json`
- `docs/benchmark_relm_vs_pymimir_rgnn_pddl_matrix_desk01.json`
- `docs/benchmark_relm_vs_pymimir_parity_desk01.json`
- `docs/benchmark_matrix_pddl_small_full.json`
- `docs/profile_relm_fast_fused_vs_baselines_pddl_blocks_medium_desk01.json`

Current observed gap on `blocks/medium`, `embedding=32`, `layers=6`, `n_states=16`:

- Relm vs pymimir forward:
  - `relm_python_rgnn`: `8.405 ms`
  - `pymimir_rgnn_full`: `3.860 ms`
- Relm vs pymimir backward:
  - `relm_python_rgnn`: `18.699 ms`
  - `pymimir_rgnn_full`: `12.771 ms`
- Encoding once:
  - relm mifrost encode: `2.554 ms`
  - pymimir encode: `1.161 ms`

Forward phase-level breakdown (compute-only profile):
- Relm:
  - `symbols_to_relations_mp.forward`: `0.680 ms/call`
  - `relations_to_symbols_mp.forward`: `0.256 ms/call`
  - `embedding_updater.forward`: `0.063 ms/call`
- Pymimir:
  - `message.forward`: `0.343 ms/call`
  - `aggregation.forward`: `0.024 ms/call`
  - `update.forward`: `0.061 ms/call`

Forward operator pressure (compute-only):
- Relm:
  - `cudaLaunchKernel` calls: `3930`
  - `cat` calls: `534`
  - `mm_family` calls: `828`
- Pymimir:
  - `cudaLaunchKernel` calls: `1644`
  - `cat` calls: `108`
  - `mm_family` calls: `432`

Interpretation:
- The updater block is not the bottleneck.
- Main gap is orchestration/routing overhead (especially fanin path and graph fragmentation), not expressivity.

## 2. Optimization Objectives

1. Reduce decentralized forward latency toward pymimir-like routing efficiency.
2. Reduce backward overhead without regressing numerical parity.
3. Keep fallback correctness and feature coverage (label-mode, flexible routing) intact.
4. Make performance stable across `sum`, `mean`, and `logsumexp`.

## 3. Success Criteria

Primary success criteria (decentralized, `blocks/medium`, same benchmark harness):
- Forward target: at least `25%` reduction from current Python baseline.
- Backward target: at least `20%` reduction from current Python baseline.
- Kernel launch count reduction: `>= 30%`.
- `cat` call reduction: `>= 40%`.

Safety criteria:
- Max forward absolute diff vs baseline path remains within expected floating tolerance.
- Max gradient absolute diff remains within expected floating tolerance.
- No regressions in existing parity/ops tests.

## 4. Work Plan (Prioritized)

### O1. Split hot non-label fanin path from label-mode fanin

Target files:
- `src/relm/models/hetero_mp/batched.py`

Change:
- Extract a dedicated non-label `BatchedFanIn` implementation with fewer runtime branches.
- Keep label-mode behavior in a separate class/module.

Motivation:
- Current `BatchedFanInMP.forward` handles many plan variants and mode checks in a single code path.
- Default relational workloads mostly use non-label mode; extra branching adds overhead each layer/destination.

Expected impact:
- Lower Python dispatch and conditional overhead.
- Better maintainability for further kernel integration.

Verification required:
- Numerical parity between old/new non-label path on representative batches.
- Label-mode parity tests unchanged.
- Profile comparison: reduced CPU self-time inside fanin forward.

---

### O2. Make flattened relation-slot handoff canonical (fanout -> fanin)

Target files:
- `src/relm/models/hetero_mp/batched.py`

Change:
- Standardize on `rel_flat_all` as the primary intermediate for fanin consumption.
- Minimize per-predicate rematerialization and repeated concatenation.

Motivation:
- Rebuilding per-predicate tensors and repeated `cat/index_select` increases overhead.
- Pymimir is closer to a single flat relation/index aggregation model.

Expected impact:
- Lower `cat` count and lower fanin phase cost.
- Better alignment with custom fanin-reduce kernels.

Verification required:
- Forward/backward parity against previous batched path.
- Operator profile: reduced `cat` and `index_select` calls.
- Benchmark matrix check for decentralized `sum/mean/logsumexp`.

---

### O3. Persist grouped-MLP parameter stacks across forwards

Target files:
- `src/relm/models/hetero_mp/batched.py`
- `src/relm/models/hetero_mp/fast_fused.py`

Change:
- Cache grouped linear weight/bias stacks beyond one forward pass (invalidated only when needed).

Motivation:
- Repeated `torch.stack` of the same module parameters is overhead with no modeling benefit.
- Current caching is per-forward in critical places.

Expected impact:
- Reduced CPU overhead in grouped MLP execution.
- More stable step-to-step latency.

Verification required:
- Ensure correctness under training (autograd safety) and eval.
- Verify cache invalidation behavior when module parameters change.
- Profile: reduced stack/copy/self CPU costs.

---

### O4. Remove concat-heavy grouping in modular fanout/fanin

Target files:
- `src/relm/models/hetero_mp/fanout.py`
- `src/relm/models/hetero_mp/fanin.py`

Change:
- Replace repeated `torch.cat` accumulation with preallocated writes where possible.

Motivation:
- `cat` frequency is much higher than pymimir and contributes measurable overhead.

Expected impact:
- Lower allocator pressure and fewer copy kernels.

Verification required:
- Functional parity in modular mode.
- Operator profile: fewer `cat` calls and lower copy overhead.

---

### O5. Replace PyG `SimpleConv`/`SelectMP` routing in hot modular path

Target files:
- `src/relm/models/hetero_mp/fanout.py`
- `src/relm/models/hetero_mp/fanin.py`

Change:
- Implement direct tensor-index routing (`index_select`/`index_add` style) in modular path.

Motivation:
- Per-edge-type message-passing abstraction introduces many small launches/dispatches.

Expected impact:
- Reduced kernel launch count and lower Python overhead.

Verification required:
- Parity for strict and non-strict type filtering.
- Phase profile: reduced fanout/fanin per-call time in modular mode.

---

### O6. Promote `fast_fused` for decentralized defaults after parity gates

Target files:
- `src/relm/models/relational_gnn.py`
- `src/relm/models/hetero_mp/fast_fused.py`

Change:
- Shift default decentralized runtime to `fast_fused` once parity/robustness gates pass.

Motivation:
- Existing profile already shows fewer launches and lower routing overhead in fused mode.

Expected impact:
- Immediate practical speedup without waiting for deeper refactors to land.

Verification required:
- End-to-end parity on all supported aggregations.
- Benchmark history update with before/after on common workloads.

---

### O7. Stabilize and default-enable batched fanin reduce kernel where safe

Target files:
- `src/relm/models/hetero_mp/_ops_env.py`
- `src/relm/models/hetero_mp/batched.py`

Change:
- Improve stability and autograd behavior of experimental fanin reduce path.
- Enable by preset for matching device/dtype once validated.

Motivation:
- Custom reduction can remove costly Python-side aggregation steps.

Expected impact:
- Lower fanin cost and lower launch count.
- Potentially strongest speed lever after O1/O2.

Verification required:
- Gradient parity (strict) in `sum` and `logsumexp` modes.
- A/B with environment flags on representative training loops.
- No regressions in fallback behavior.

---

### O8. Add cross-forward routing-plan cache keyed by topology signature

Target files:
- `src/relm/models/hetero_mp/batched.py`
- `src/relm/models/hetero_mp/fast_fused.py`

Change:
- Reuse routing plans across forward calls when graph topology repeats.

Motivation:
- Current caching is mostly per-forward; repeated workloads can reuse structure.

Expected impact:
- Reduced CPU routing prep time, especially in training loops with repeated structure.

Verification required:
- Correct invalidation when topology changes.
- Memory growth checks (bounded cache policy).
- Long-run benchmark with repeated workloads.

---

### O9. Expand compile strategy to include relational layer orchestration

Target files:
- `src/relm/models/relational_gnn.py`

Change:
- Move beyond updater-only compile; compile larger layer orchestration when safe.

Motivation:
- Python control flow remains significant in layer loop.

Expected impact:
- Reduced Python overhead for stable shapes/topologies.

Verification required:
- Compile/eager parity.
- Fallback behavior when compile fails.
- Throughput comparison with and without compile.

---

### O10. Add performance gates to avoid silent regressions

Target files:
- Benchmark/profiling scripts and CI glue.

Change:
- Track and gate:
  - forward/backward latency
  - kernel launch count
  - `cat` count
  - fanout/fanin phase timings

Motivation:
- Multiple optimization paths have shown tradeoffs (forward gains with backward regressions).

Expected impact:
- Better iteration safety and reproducibility.

Verification required:
- CI job producing machine-readable artifacts.
- Threshold definitions with clear pass/fail policy.

## 5. Verification Protocol (Unified)

For each optimization PR:

1. Correctness:
   - Unit tests in existing suite.
   - Explicit parity script with forward and gradient diffs.
2. Micro performance:
   - Operator profile (grouped stats).
   - Phase profile (fanout/fanin/update timings).
3. End-to-end:
   - Benchmark matrix on `blocks/medium` for `sum`, `mean`, `logsumexp`.
4. Record:
   - Append results to benchmark history docs.
   - Note environment flags used (must be explicit).

## 6. Risks and Expected Tradeoffs

- Aggressive fused/custom-kernel paths can improve forward while harming backward if autograd paths are not carefully controlled.
- Route-plan caching can produce stale-plan bugs if invalidation is incomplete.
- Compile strategies can be workload-sensitive and need robust fallback behavior.
- Label-mode and strict-filter correctness must remain intact while optimizing default paths.

## 7. Tracking Template

Use this table for each milestone:

| ID | Status | Owner | Branch | Fwd delta | Bwd delta | Kernel launches delta | Cat calls delta | Parity status | Notes |
|---|---|---|---|---:|---:|---:|---:|---|---|
| O1 | planned | - | - | - | - | - | - | - | - |
| O2 | planned | - | - | - | - | - | - | - | - |
| O3 | planned | - | - | - | - | - | - | - | - |
| O4 | planned | - | - | - | - | - | - | - | - |
| O5 | planned | - | - | - | - | - | - | - | - |
| O6 | planned | - | - | - | - | - | - | - | - |
| O7 | planned | - | - | - | - | - | - | - | - |
| O8 | planned | - | - | - | - | - | - | - | - |
| O9 | planned | - | - | - | - | - | - | - | - |
| O10 | planned | - | - | - | - | - | - | - | - |

## 8. Immediate Next Execution Slice

Recommended first implementation batch:
1. O1 (non-label fanin split)
2. O2 (canonical flattened handoff)
3. O3 (persistent grouped parameter stacks)

Reason:
- These directly attack the measured hotspots (fanin/routing + repeated tensor assembly) while keeping architectural risk moderate.
