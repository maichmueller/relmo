# relm

Standalone package for relational-model runtime pieces (custom C++/CUDA ops plus Python fallbacks).

## What is in this repo now

- `relm.ops.relmp`: Python loader/wrappers for relational message-passing ops.
- Stable-ABI operator library target: `relm_relmp_ops`.
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
PYTHONPATH=src python -m pytest tests/python/test_relmp_ops.py -q -p no:cacheprovider
```

## Full handoff and implementation plan

See [docs/RELMP_PORT_HANDOFF.md](docs/RELMP_PORT_HANDOFF.md) for:

- exact ported files and architecture,
- compatibility/build policy,
- testing/benchmark workflow,
- phased execution plan for the next agent.
