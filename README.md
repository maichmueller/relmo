# relm

Standalone package for relational-model runtime pieces (custom C++/CUDA ops plus Python fallbacks).

## What is in this repo now

- `relm.ops.mp`: Python loader/wrappers for relational message-passing ops.
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

## Full handoff and implementation plan

See [docs/RELMP_PORT_HANDOFF.md](docs/RELMP_PORT_HANDOFF.md) for:

- exact ported files and architecture,
- compatibility/build policy,
- testing/benchmark workflow,
- phased execution plan for the next agent.
