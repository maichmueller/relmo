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
- `RELM_MODELS_MP_FANIN=1` enables model-side use of `relm.ops.mp` fanin kernels (default: on).
- `RELM_MODELS_MP_FANIN_FUSED=1` enables fanin kernels in `CentralFusedLayerMP` (default: on).
- `RELM_MODELS_MP_FANIN_BATCHED=1` enables fanin kernels in `BatchedFanInMP` (default: on).
- `RELM_MODELS_MP_GROUPED_MLP=1` enables grouped execution for compatible per-relation residual MLPs in `BatchedFanOutMP` (default: on, with automatic fallback).
- `RELM_MODELS_MP_FANOUT=0` enables fanout kernel path in batched/fused MPs (default: off).

### Grouped MLP Interface

`RELM_MODELS_MP_GROUPED_MLP=1` can accelerate custom per-relation modules when they expose:

```python
from relm.models import GroupedMLPSpec

def relm_grouped_mlp_spec(self) -> GroupedMLPSpec | dict | None:
    return GroupedMLPSpec(
        linears=[self.lin1, self.lin2, self.lin3],  # torch.nn.Linear list
        ops=[  # execution order
            ("linear", 0),
            ("pointwise", self.act1),  # one of: Identity/ReLU/Mish/GELU/SiLU/Tanh/ELU/LeakyReLU
            ("linear", 1),
            ("pointwise", self.act2),
            ("linear", 2),
        ],
        truncated_dim=self.out_dim,  # optional residual truncation behavior
        truncate_right=False,        # optional
        signature=("my-mlp-v1",),    # optional grouping key (must be hashable)
    )
```

Returning a raw `dict` with the same keys is still supported for backward compatibility.
If this method is missing, returns `None`, or the module is incompatible, relm automatically falls back to per-relation execution.

Benchmark snapshots are tracked in [docs/BENCHMARK_HISTORY.md](docs/BENCHMARK_HISTORY.md).

## Full handoff and implementation plan

See [docs/RELMP_PORT_HANDOFF.md](docs/RELMP_PORT_HANDOFF.md) for:

- exact ported files and architecture,
- compatibility/build policy,
- testing/benchmark workflow,
- phased execution plan for the next agent.
