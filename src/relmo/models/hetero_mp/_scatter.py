from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor

from ._ops_env import relmo_mp_ops as _relmo_mp_ops_backend
from ._tensor_utils import _cat_or_single

relm_mp_ops = _relmo_mp_ops_backend


class _RelmoOpsProxy:
    def __getattr__(self, name: str) -> Any:
        return getattr(relm_mp_ops, name)


relmo_mp_ops = _RelmoOpsProxy()


def _fanout_scatter_multi_src(
    *,
    by_src: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    x_dict: Mapping[str, Tensor],
    out_rows: int,
) -> torch.Tensor:
    x_parts: list[torch.Tensor] = []
    src_parts: list[torch.Tensor] = []
    flat_parts: list[torch.Tensor] = []
    for src, (flat_dst, src_idx) in by_src.items():
        if src not in x_dict:
            raise KeyError(f"Missing src node type {src!r} in x_dict.")
        x_src = x_dict[src]
        x_parts.append(x_src)
        src_parts.append(src_idx)
        flat_parts.append(flat_dst)
    x_cat, src_global, flat_dst = relmo_mp_ops.fanout_pack_multi(
        x_parts,
        src_parts,
        flat_parts,
    )
    return relmo_mp_ops.fanout_scatter(
        x_cat, src_global, flat_dst, int(out_rows)
    )


def _build_fanout_scatter_plan(
    *,
    by_src: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    x_dict: Mapping[str, Tensor],
) -> dict[str, Any] | None:
    if len(by_src) == 1:
        src = next(iter(by_src.keys()))
        if src in x_dict:
            x_src = x_dict[src]
            flat_dst, src_idx = by_src[src]
            rows = int(x_src.size(0))
            return {
                "single_src": src,
                "single_rows": rows,
                "x_rows": rows,
                "flat_dst": flat_dst,
                "src_global": src_idx,
                "src_order": (src,),
                "src_rows": (rows,),
            }

    src_order: list[str] = []
    flat_parts: list[torch.Tensor] = []
    src_global_parts: list[torch.Tensor] = []
    src_rows: list[int] = []
    offset = 0
    for src in sorted(by_src.keys()):
        if src not in x_dict:
            continue
        x_src = x_dict[src]
        rows = int(x_src.size(0))
        flat_dst, src_idx = by_src[src]
        src_order.append(src)
        src_rows.append(rows)
        flat_parts.append(flat_dst)
        src_global_parts.append(src_idx + int(offset))
        offset += rows
    if not src_order:
        return None
    return {
        "src_order": tuple(src_order),
        "src_rows": tuple(src_rows),
        "x_rows": int(offset),
        "flat_dst": _cat_or_single(flat_parts, dim=0),
        "src_global": _cat_or_single(src_global_parts, dim=0),
    }


def _fanout_scatter_from_plan(
    *,
    plan: Mapping[str, Any],
    x_dict: Mapping[str, Tensor],
    out_rows: int,
) -> torch.Tensor:
    single_src = plan.get("single_src")
    if single_src is not None:
        src = str(single_src)
        if src not in x_dict:
            raise KeyError(f"Missing src node type {src!r} in x_dict.")
        x_src = x_dict[src]
        expected_rows = int(plan["single_rows"])
        if int(x_src.size(0)) != expected_rows:
            raise ValueError(
                f"Source size changed for {src!r}: expected {expected_rows}, got {int(x_src.size(0))}."
            )
        return relmo_mp_ops.fanout_scatter(  # type: ignore[union-attr]
            x_src, plan["src_global"], plan["flat_dst"], int(out_rows)
        )

    src_order = plan["src_order"]
    src_rows = plan["src_rows"]
    x_parts: list[torch.Tensor] = []
    for src, expected_rows in zip(src_order, src_rows):
        if src not in x_dict:
            raise KeyError(f"Missing src node type {src!r} in x_dict.")
        x_src = x_dict[src]
        if int(x_src.size(0)) != int(expected_rows):
            raise ValueError(
                f"Source size changed for {src!r}: expected {expected_rows}, got {int(x_src.size(0))}."
            )
        x_parts.append(x_src)
    x_cat = _cat_or_single(x_parts, dim=0)
    if int(x_cat.size(0)) != int(plan["x_rows"]):
        raise ValueError(
            f"Fanout plan x_rows mismatch: expected {plan['x_rows']}, got {int(x_cat.size(0))}."
        )
    return relmo_mp_ops.fanout_scatter(  # type: ignore[union-attr]
        x_cat, plan["src_global"], plan["flat_dst"], int(out_rows)
    )
