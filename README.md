# relmo

Standalone package for relational-model runtime pieces (custom C++/CUDA ops plus Python fallbacks).

Distribution name: `relmo`
Import namespace: `relmo`

## What is in this repo now

- `relmo.ops.mp`: Python loader/wrappers for relational message-passing ops.
- `relmo.models`: RelationalGNN and CentralizedRelationalGNN model stack (self-contained, no `torchrl` etc. dependency).
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

## Install

Pure fallback wheel:

```bash
pip install relmo
```

Native wheels follow the PyG-style `find-links` pattern and are keyed by the installed Torch lane:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda or 'cpu')"
pip install relmo -f https://<wheel-host>/relmo-cu126.html --no-deps
```

The current wheel strategy is:

- build once per CUDA lane against the oldest supported Torch for that lane
- validate newer Torch versions in CI before claiming support
- keep `RELM_MP_TORCH_VERSION_POLICY=strict` available if you want the old exact-match behavior
- build the current `cu126` and `cu128` Linux wheels in public `manylinux-cuda` images

The minimum supported Torch version is currently `2.8`, which is the oldest version we have compiled and tested successfully so far.
The `--no-deps` keeps your chosen Torch runtime in place when installing a lane-specific native wheel.

The published package name and Python import namespace are both `relmo`.

## Flat model docs

The supported flat-model user surface is documented in [docs/flat_rgnn_usage_examples.md](docs/flat_rgnn_usage_examples.md) and centers on `relmo.models.flat` plus `relmo.models.builders`.

Kernel experiments that were evaluated and not kept are intentionally preserved as engineering history in [docs/flat_lgan_plan.md](docs/flat_lgan_plan.md) and [docs/manual_fused_program_kernel_plan.md](docs/manual_fused_program_kernel_plan.md), so future work does not repeat already-failed paths blindly.

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
- `RELM_MODELS_MP_FANIN=1` enables model-side `relmo.ops.mp` fanin kernels for centralized fused paths (default: on).
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

## Wheel CI

Wheel build and release policy lives in [docs/WHEEL_CI_STRATEGY.md](docs/WHEEL_CI_STRATEGY.md).

## Full handoff and implementation plan

See [docs/RELMP_PORT_HANDOFF.md](docs/RELMP_PORT_HANDOFF.md) for:

- exact ported files and architecture,
- compatibility/build policy,
- testing/benchmark workflow,
- phased execution plan for the next agent.
