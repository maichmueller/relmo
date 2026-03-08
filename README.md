# relm

Standalone package for relational-model runtime pieces (custom C++/CUDA ops plus Python fallbacks).

## What is in this repo now

- `relm.ops.mp`: Python loader/wrappers for relational message-passing ops.
- `relm.models`: RelationalGNN and CentralizedRelationalGNN model stack (self-contained, no `torchrl` etc. dependency).
- Non-stable ABI operator library target: `relm_mp_ops`.
- CPU and optional CUDA kernels for:
  - `fanout_scatter`
  - `fanin_reduce` (`sum` and `logsumexp`)
- Compatibility guards and feature flags.
- Parity-focused tests and micro-benchmark script.

## Build (editable)

```bash
python -m pip install -e . --no-build-isolation
```

For CUDA builds, ensure `nvcc` is visible, for example:

```bash
CUDACXX=/usr/local/cuda/bin/nvcc python -m pip install -e . --no-build-isolation
```

## Quick check

```bash
PYTHONPATH=src python -m pytest tests/python/test_mp_ops.py -q -p no:cacheprovider
```

## Benchmark modes

```bash
PYTHONPATH=src python scripts/benchmark_mp_ops.py \
  --import-mode source \
  --device cuda \
  --timing-modes end_to_end,kernel_only \
  --timer auto \
  --require-custom
```

- `--timing-modes end_to_end,kernel_only` benchmarks both full training-step style and kernel-focused paths.
- `--timer auto` uses CUDA events on CUDA devices and wall clock otherwise.
- `RELM_MP_FANOUT_BWD_KERNEL=auto|1d|2d` can force the fanout backward CUDA kernel variant for A/B tuning.
- `RELM_MP_FANIN_LSE_FWD=auto|atomic|segmented` selects the logsumexp forward CUDA strategy (`auto` enables segmented reduction only on larger workloads).

### Model workload benchmark

```bash
PYTHONPATH=src python scripts/benchmark_relational_models.py \
  --workload synthetic \
  --device cuda \
  --embedding-size 32 \
  --num-layer 10 \
  --rounds 20 \
  --warmup 5
```

- `--workload synthetic` benchmarks on generated hetero batches without requiring `mifrost`.
- `--workload pddl --pddl-root /path/to/pddl_instances` benchmarks on real PDDL state-space batches via `pymimir` + `mifrost` (no rgnet import required).
- `--workload rgnet --rgnet-root ~/GitHub/rgnet` benchmarks on rgnet problem batches (requires `mifrost`, rgnet test assets, and optional `numpy`).
- `--model-kinds decentralized,centralized` runs both models for direct comparison.
- `RELM_MODELS_MP_FANIN=1` enables model-side `relm.ops.mp` fanin kernels for centralized fused paths (default: on).
- `RELM_MODELS_MP_FANIN_FUSED=1` enables fanin kernels in `CentralFusedLayerMP` (default: on).
- Decentralized batched MP C++ flags remain as legacy scaffolding only; the maintained path is the regular PyG execution.
- `RELM_MODELS_MP_FANOUT=0` enables fanout kernel path in centralized fused MP (default: off).

### Definitive variant matrix benchmark (single source of truth)

```bash
PYTHONPATH=src python scripts/benchmark_relational_matrix.py \
  --workload pddl \
  --pddl-root /path/to/pddl_instances \
  --domain-case blocks \
  --problem-case medium \
  --max-states 50 \
  --device cuda \
  --embedding-size 32 \
  --num-layer 10 \
  --rounds 20 \
  --warmup 5 \
  --repeats 3 \
  --aggregations sum,logsumexp,mean \
  --variants python_only,optimized_full \
  --json-out docs/benchmark_matrix_latest.json \
  --csv-out docs/benchmark_matrix_latest.csv
```

- `python_only`: disables model-side custom mp kernels.
- `optimized_full`: enables the retained custom mp kernel lane for centralized/decentralized fanin/fanout.
- Output artifacts are written to JSON + CSV for stable history and comparison automation.

Benchmark snapshots are tracked in [docs/BENCHMARK_HISTORY.md](docs/BENCHMARK_HISTORY.md).

## Full handoff and implementation plan

See [docs/RELMP_PORT_HANDOFF.md](docs/RELMP_PORT_HANDOFF.md) for:

- exact ported files and architecture,
- compatibility/build policy,
- testing/benchmark workflow,
- phased execution plan for the next agent.
