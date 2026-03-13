from __future__ import annotations

import os

import torch

from relmo.ops import mp


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _fallback_check() -> None:
    os.environ["RELM_MP_ENABLE"] = "0"
    os.environ["RELM_MP_FALLBACK"] = "python"

    x_cat = torch.randn(4, 3, dtype=torch.float32)
    src_global_idx = torch.tensor([0, 2, 1], dtype=torch.int64)
    flat_dst = torch.tensor([2, 0, 1], dtype=torch.int64)
    out = mp.fanout_scatter(x_cat, src_global_idx, flat_dst, out_rows=4)
    ref = x_cat.new_zeros((4, 3))
    ref.index_copy_(0, flat_dst, x_cat.index_select(0, src_global_idx))
    if not torch.allclose(out, ref):
        raise SystemExit("Python fallback smoke check failed.")


def _custom_check() -> None:
    os.environ["RELM_MP_ENABLE"] = "1"
    os.environ["RELM_MP_FALLBACK"] = "error"

    if not mp.available():
        raise SystemExit("Expected custom ops, but relmo.ops.mp.available() is false.")
    info = mp.assert_runtime_compat()
    print(f"custom ops build info: {info}")

    x_cat = torch.randn(5, 2, dtype=torch.float32)
    src_global_idx = torch.tensor([0, 3, 1], dtype=torch.int64)
    flat_dst = torch.tensor([2, 0, 1], dtype=torch.int64)
    out = mp.fanout_scatter(x_cat, src_global_idx, flat_dst, out_rows=4)
    ref = x_cat.new_zeros((4, 2))
    ref.index_copy_(0, flat_dst, x_cat.index_select(0, src_global_idx))
    if not torch.allclose(out, ref):
        raise SystemExit("Custom op smoke check failed.")


def main() -> int:
    expect_custom = _bool_env("RELM_EXPECT_CUSTOM_OPS", True)
    print(
        "torch runtime:",
        {"version": str(torch.__version__), "cuda": getattr(torch.version, "cuda", None)},
    )
    _fallback_check()
    if expect_custom:
        _custom_check()
    elif mp.available():
        raise SystemExit("Pure wheel unexpectedly exposed native custom ops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
