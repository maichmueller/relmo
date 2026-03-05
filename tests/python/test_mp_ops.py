import pytest
import torch

from relm.ops import mp


def _fanout_ref(
    x_cat: torch.Tensor,
    src_global_idx: torch.Tensor,
    flat_dst: torch.Tensor,
    out_rows: int,
) -> torch.Tensor:
    out = x_cat.new_zeros((int(out_rows), int(x_cat.size(-1))))
    if src_global_idx.numel() == 0 or int(out_rows) == 0:
        return out
    vals = x_cat.index_select(0, src_global_idx)
    out.index_copy_(0, flat_dst, vals)
    return out


def _fanin_ref(
    rel_flat: torch.Tensor,
    flat_src: torch.Tensor,
    dst_idx: torch.Tensor,
    dim_size: int,
    mode: int,
) -> torch.Tensor:
    emb = int(rel_flat.size(-1))
    if mode == 0:
        out = rel_flat.new_zeros((int(dim_size), emb))
        if flat_src.numel() == 0 or int(dim_size) == 0:
            return out
        vals = rel_flat.index_select(0, flat_src)
        out.index_add_(0, dst_idx, vals)
        return out
    if mode == 1:
        out = rel_flat.new_full((int(dim_size), emb), float("-inf"))
        if flat_src.numel() == 0 or int(dim_size) == 0:
            return out
        vals = rel_flat.index_select(0, flat_src)
        index = dst_idx.view(-1, 1).expand(-1, emb)
        amax = rel_flat.new_full((int(dim_size), emb), float("-inf"))
        amax.scatter_reduce_(0, index, vals, reduce="amax", include_self=True)
        offsets = amax.index_select(0, dst_idx)
        exps = (vals - offsets).exp()
        exps_sum = rel_flat.new_zeros((int(dim_size), emb))
        exps_sum.scatter_add_(0, index, exps)
        return exps_sum.log() + amax
    raise ValueError(mode)


def _fanout_pack_ref(
    x_parts: list[torch.Tensor],
    src_idx_parts: list[torch.Tensor],
    flat_dst_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cat_parts = []
    src_global_parts = []
    dst_parts = []
    offset = 0
    for x, src_idx, flat_dst in zip(x_parts, src_idx_parts, flat_dst_parts):
        x_cat_parts.append(x)
        src_global_parts.append(src_idx + int(offset))
        dst_parts.append(flat_dst)
        offset += int(x.size(0))
    x_cat = x_cat_parts[0] if len(x_cat_parts) == 1 else torch.cat(x_cat_parts, dim=0)
    src_global = (
        src_global_parts[0]
        if len(src_global_parts) == 1
        else torch.cat(src_global_parts, dim=0)
    )
    flat_dst = dst_parts[0] if len(dst_parts) == 1 else torch.cat(dst_parts, dim=0)
    return x_cat, src_global, flat_dst


def _fanin_pack_ref(
    rel_parts: list[torch.Tensor],
    flat_src_parts: list[torch.Tensor],
    dst_idx_parts: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_cat_parts = []
    src_parts = []
    dst_parts = []
    offset = 0
    for rel, flat_src, dst_idx in zip(rel_parts, flat_src_parts, dst_idx_parts):
        rel_cat_parts.append(rel)
        src_parts.append(flat_src + int(offset))
        dst_parts.append(dst_idx)
        offset += int(rel.size(0))
    rel_cat = rel_cat_parts[0] if len(rel_cat_parts) == 1 else torch.cat(rel_cat_parts, dim=0)
    flat_src = src_parts[0] if len(src_parts) == 1 else torch.cat(src_parts, dim=0)
    dst_idx = dst_parts[0] if len(dst_parts) == 1 else torch.cat(dst_parts, dim=0)
    return rel_cat, flat_src, dst_idx


def _fanout_pack_edges_ref(
    x_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    src_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    slot_offset_parts: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cat = x_parts[0] if len(x_parts) == 1 else torch.cat(x_parts, dim=0)
    offsets: list[int] = []
    offset = 0
    for x in x_parts:
        offsets.append(offset)
        offset += int(x.size(0))
    src_global_parts = []
    flat_dst_parts = []
    for edge_src, edge_dst, src_part, arity, pos, slot_offset in zip(
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    ):
        src_global_parts.append(edge_src + int(offsets[int(src_part)]))
        flat_dst_parts.append(int(slot_offset) + edge_dst * int(arity) + int(pos))
    src_global = (
        src_global_parts[0]
        if len(src_global_parts) == 1
        else torch.cat(src_global_parts, dim=0)
    )
    flat_dst = (
        flat_dst_parts[0] if len(flat_dst_parts) == 1 else torch.cat(flat_dst_parts, dim=0)
    )
    return x_cat, src_global, flat_dst


def _fanin_pack_edges_ref(
    rel_parts: list[torch.Tensor],
    edge_src_parts: list[torch.Tensor],
    edge_dst_parts: list[torch.Tensor],
    rel_part_ids: list[int],
    arity_parts: list[int],
    pos_parts: list[int],
    mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_cat = rel_parts[0] if len(rel_parts) == 1 else torch.cat(rel_parts, dim=0)
    offsets: list[int] = []
    offset = 0
    for rel in rel_parts:
        offsets.append(offset)
        offset += int(rel.size(0))
    flat_src_parts = []
    dst_parts = []
    for edge_src, edge_dst, rel_part, arity, pos in zip(
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
    ):
        if int(mode) == 1:
            local_src = edge_src
        else:
            local_src = edge_src * int(arity) + int(pos)
        flat_src_parts.append(local_src + int(offsets[int(rel_part)]))
        dst_parts.append(edge_dst)
    flat_src = (
        flat_src_parts[0] if len(flat_src_parts) == 1 else torch.cat(flat_src_parts, dim=0)
    )
    dst_idx = dst_parts[0] if len(dst_parts) == 1 else torch.cat(dst_parts, dim=0)
    return rel_cat, flat_src, dst_idx


@pytest.mark.parametrize("mode", [0, 1])
def test_python_fallback_path(mode: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELM_MP_ENABLE", "0")
    monkeypatch.setenv("RELM_MP_FALLBACK", "python")

    x_cat = torch.randn(6, 3, dtype=torch.float32)
    src_global_idx = torch.tensor([0, 4, 1], dtype=torch.int64)
    flat_dst = torch.tensor([3, 0, 1], dtype=torch.int64)
    out = mp.fanout_scatter(x_cat, src_global_idx, flat_dst, out_rows=5)
    ref = _fanout_ref(x_cat, src_global_idx, flat_dst, out_rows=5)
    assert torch.allclose(out, ref)

    rel_flat = torch.randn(8, 3, dtype=torch.float32)
    flat_src = torch.tensor([0, 2, 4, 1], dtype=torch.int64)
    dst_idx = torch.tensor([2, 0, 1, 0], dtype=torch.int64)
    out = mp.fanin_reduce(rel_flat, flat_src, dst_idx, dim_size=4, mode=mode)
    ref = _fanin_ref(rel_flat, flat_src, dst_idx, dim_size=4, mode=mode)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanout_scatter_parity(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    x_cat = torch.randn(7, 5, device=device, dtype=torch.float32, requires_grad=True)
    src_global_idx = torch.tensor([0, 3, 5, 2, 3], dtype=torch.int64, device=device)
    flat_dst = torch.tensor([4, 1, 0, 2, 3], dtype=torch.int64, device=device)

    out = mp.fanout_scatter(x_cat, src_global_idx, flat_dst, out_rows=6)
    ref = _fanout_ref(x_cat, src_global_idx, flat_dst, out_rows=6)
    assert torch.allclose(out, ref)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanout_scatter_grad_parity(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    x0 = torch.randn(8, 4, device=device, dtype=torch.float32, requires_grad=True)
    x1 = x0.detach().clone().requires_grad_(True)
    src_global_idx = torch.tensor([0, 5, 2, 6], dtype=torch.int64, device=device)
    flat_dst = torch.tensor([1, 0, 3, 2], dtype=torch.int64, device=device)

    out = mp.fanout_scatter(x0, src_global_idx, flat_dst, out_rows=5)
    ref = _fanout_ref(x1, src_global_idx, flat_dst, out_rows=5)
    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(x0.grad, x1.grad, atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanin_reduce_parity(
    mode: int, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    rel_flat = torch.randn(
        10, 4, device=device, dtype=torch.float32, requires_grad=True
    )
    flat_src = torch.tensor([0, 2, 7, 5, 2, 1], dtype=torch.int64, device=device)
    dst_idx = torch.tensor([1, 0, 2, 2, 1, 0], dtype=torch.int64, device=device)

    out = mp.fanin_reduce(rel_flat, flat_src, dst_idx, dim_size=4, mode=mode)
    ref = _fanin_ref(rel_flat, flat_src, dst_idx, dim_size=4, mode=mode)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5, equal_nan=True)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanin_reduce_grad_parity(
    mode: int, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    rel_flat0 = torch.randn(
        12, 3, device=device, dtype=torch.float32, requires_grad=True
    )
    rel_flat1 = rel_flat0.detach().clone().requires_grad_(True)
    flat_src = torch.tensor([0, 1, 8, 4, 9, 10], dtype=torch.int64, device=device)
    dst_idx = torch.tensor([2, 0, 1, 1, 0, 2], dtype=torch.int64, device=device)

    out = mp.fanin_reduce(rel_flat0, flat_src, dst_idx, dim_size=5, mode=mode)
    ref = _fanin_ref(rel_flat1, flat_src, dst_idx, dim_size=5, mode=mode)

    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(
        rel_flat0.grad, rel_flat1.grad, atol=1e-6, rtol=1e-5, equal_nan=True
    )


def test_fanout_scatter_noncontiguous_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    base = torch.randn(4, 6, dtype=torch.float32, requires_grad=True)
    x_cat = base.transpose(0, 1)  # non-contiguous [6, 4]
    src_global_idx = torch.tensor([0, 4, 2, 5], dtype=torch.int64)
    flat_dst = torch.tensor([2, 1, 0, 3], dtype=torch.int64)

    out = mp.fanout_scatter(x_cat, src_global_idx, flat_dst, out_rows=5)
    ref = _fanout_ref(x_cat, src_global_idx, flat_dst, out_rows=5)
    assert torch.allclose(out, ref)


def test_fanout_scatter_gradcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    x_cat = torch.randn(6, 3, dtype=torch.double, requires_grad=True)
    src_global_idx = torch.tensor([0, 5, 2, 1], dtype=torch.int64)
    flat_dst = torch.tensor([1, 0, 3, 2], dtype=torch.int64)

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return mp.fanout_scatter(inp, src_global_idx, flat_dst, out_rows=5)

    assert torch.autograd.gradcheck(fn, (x_cat,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_fanin_reduce_sum_gradcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    rel_flat = torch.randn(8, 4, dtype=torch.double, requires_grad=True)
    flat_src = torch.tensor([0, 2, 5, 7, 1], dtype=torch.int64)
    dst_idx = torch.tensor([1, 0, 3, 2, 0], dtype=torch.int64)

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return mp.fanin_reduce(inp, flat_src, dst_idx, dim_size=5, mode=0)

    assert torch.autograd.gradcheck(fn, (rel_flat,), eps=1e-6, atol=1e-4, rtol=1e-3)


def test_fanin_reduce_logsumexp_gradcheck(monkeypatch: pytest.MonkeyPatch) -> None:
    if not mp.available():
        pytest.skip("mp custom ops are not available in this build.")
    monkeypatch.setenv("RELM_MP_ENABLE", "1")
    monkeypatch.setenv("RELM_MP_FALLBACK", "error")

    rel_flat = torch.randn(8, 4, dtype=torch.double, requires_grad=True)
    flat_src = torch.tensor([0, 2, 5, 7, 1, 3], dtype=torch.int64)
    dst_idx = torch.tensor([1, 0, 3, 2, 0, 1], dtype=torch.int64)

    def fn(inp: torch.Tensor) -> torch.Tensor:
        return mp.fanin_reduce(inp, flat_src, dst_idx, dim_size=4, mode=1)

    assert torch.autograd.gradcheck(fn, (rel_flat,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanout_pack_multi_parity_and_grad(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mp.available() and hasattr(torch.ops.relm_mp, "fanout_pack_multi"):
        monkeypatch.setenv("RELM_MP_ENABLE", "1")
        monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    else:
        monkeypatch.setenv("RELM_MP_ENABLE", "0")
        monkeypatch.setenv("RELM_MP_FALLBACK", "python")

    x_a = torch.randn(5, 3, device=device, dtype=torch.float32)
    x_b = torch.randn(4, 3, device=device, dtype=torch.float32)
    x_a_custom = x_a.detach().clone().requires_grad_(True)
    x_b_custom = x_b.detach().clone().requires_grad_(True)
    x_a_ref = x_a.detach().clone().requires_grad_(True)
    x_b_ref = x_b.detach().clone().requires_grad_(True)
    src_a = torch.tensor([0, 2, 4], device=device, dtype=torch.int64)
    src_b = torch.tensor([1, 3], device=device, dtype=torch.int64)
    dst_a = torch.tensor([3, 1, 5], device=device, dtype=torch.int64)
    dst_b = torch.tensor([0, 4], device=device, dtype=torch.int64)

    x_cat, src_global, flat_dst = mp.fanout_pack_multi(
        [x_a_custom, x_b_custom], [src_a, src_b], [dst_a, dst_b]
    )
    x_ref, src_ref, dst_ref = _fanout_pack_ref(
        [x_a_ref, x_b_ref], [src_a, src_b], [dst_a, dst_b]
    )
    assert torch.allclose(x_cat, x_ref)
    assert torch.equal(src_global, src_ref)
    assert torch.equal(flat_dst, dst_ref)

    out = mp.fanout_scatter(x_cat, src_global, flat_dst, out_rows=6)
    ref = _fanout_ref(x_ref, src_ref, dst_ref, out_rows=6)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(x_a_custom.grad, x_a_ref.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(x_b_custom.grad, x_b_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanin_pack_multi_parity_and_grad(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mp.available() and hasattr(torch.ops.relm_mp, "fanin_pack_multi"):
        monkeypatch.setenv("RELM_MP_ENABLE", "1")
        monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    else:
        monkeypatch.setenv("RELM_MP_ENABLE", "0")
        monkeypatch.setenv("RELM_MP_FALLBACK", "python")

    rel_a = torch.randn(6, 4, device=device, dtype=torch.float32)
    rel_b = torch.randn(3, 4, device=device, dtype=torch.float32)
    rel_a_custom = rel_a.detach().clone().requires_grad_(True)
    rel_b_custom = rel_b.detach().clone().requires_grad_(True)
    rel_a_ref = rel_a.detach().clone().requires_grad_(True)
    rel_b_ref = rel_b.detach().clone().requires_grad_(True)
    src_a = torch.tensor([0, 2, 5], device=device, dtype=torch.int64)
    src_b = torch.tensor([1, 2], device=device, dtype=torch.int64)
    dst_a = torch.tensor([1, 0, 2], device=device, dtype=torch.int64)
    dst_b = torch.tensor([2, 1], device=device, dtype=torch.int64)

    rel_cat, flat_src, dst_idx = mp.fanin_pack_multi(
        [rel_a_custom, rel_b_custom], [src_a, src_b], [dst_a, dst_b]
    )
    rel_ref, src_ref, dst_ref = _fanin_pack_ref(
        [rel_a_ref, rel_b_ref], [src_a, src_b], [dst_a, dst_b]
    )
    assert torch.allclose(rel_cat, rel_ref)
    assert torch.equal(flat_src, src_ref)
    assert torch.equal(dst_idx, dst_ref)

    out = mp.fanin_reduce(rel_cat, flat_src, dst_idx, dim_size=4, mode=0)
    ref = _fanin_ref(rel_ref, src_ref, dst_ref, dim_size=4, mode=0)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5, equal_nan=True)

    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(
        rel_a_custom.grad, rel_a_ref.grad, atol=1e-6, rtol=1e-5, equal_nan=True
    )
    assert torch.allclose(
        rel_b_custom.grad, rel_b_ref.grad, atol=1e-6, rtol=1e-5, equal_nan=True
    )


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanout_pack_from_edges_parity_and_grad(
    device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mp.available() and hasattr(torch.ops.relm_mp, "fanout_pack_from_edges"):
        monkeypatch.setenv("RELM_MP_ENABLE", "1")
        monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    else:
        monkeypatch.setenv("RELM_MP_ENABLE", "0")
        monkeypatch.setenv("RELM_MP_FALLBACK", "python")

    x_a = torch.randn(5, 3, device=device, dtype=torch.float32)
    x_b = torch.randn(4, 3, device=device, dtype=torch.float32)
    x_a_custom = x_a.detach().clone().requires_grad_(True)
    x_b_custom = x_b.detach().clone().requires_grad_(True)
    x_a_ref = x_a.detach().clone().requires_grad_(True)
    x_b_ref = x_b.detach().clone().requires_grad_(True)

    edge_src_parts = [
        torch.tensor([0, 2, 4], device=device, dtype=torch.int64),
        torch.tensor([1, 3], device=device, dtype=torch.int64),
        torch.tensor([2, 0], device=device, dtype=torch.int64),
    ]
    edge_dst_parts = [
        torch.tensor([1, 0, 2], device=device, dtype=torch.int64),
        torch.tensor([3, 1], device=device, dtype=torch.int64),
        torch.tensor([0, 2], device=device, dtype=torch.int64),
    ]
    src_part_ids = [0, 1, 0]
    arity_parts = [2, 3, 2]
    pos_parts = [0, 1, 1]
    slot_offset_parts = [0, 8, 0]

    x_cat, src_global, flat_dst = mp.fanout_pack_from_edges(
        [x_a_custom, x_b_custom],
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    )
    x_ref, src_ref, dst_ref = _fanout_pack_edges_ref(
        [x_a_ref, x_b_ref],
        edge_src_parts,
        edge_dst_parts,
        src_part_ids,
        arity_parts,
        pos_parts,
        slot_offset_parts,
    )
    assert torch.allclose(x_cat, x_ref)
    assert torch.equal(src_global, src_ref)
    assert torch.equal(flat_dst, dst_ref)

    out = mp.fanout_scatter(x_cat, src_global, flat_dst, out_rows=20)
    ref = _fanout_ref(x_ref, src_ref, dst_ref, out_rows=20)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5)

    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(x_a_custom.grad, x_a_ref.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(x_b_custom.grad, x_b_ref.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("mode", [0, 1])
@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_fanin_pack_from_edges_parity_and_grad(
    mode: int, device: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if mp.available() and hasattr(torch.ops.relm_mp, "fanin_pack_from_edges"):
        monkeypatch.setenv("RELM_MP_ENABLE", "1")
        monkeypatch.setenv("RELM_MP_FALLBACK", "error")
    else:
        monkeypatch.setenv("RELM_MP_ENABLE", "0")
        monkeypatch.setenv("RELM_MP_FALLBACK", "python")

    rel_a = torch.randn(6, 4, device=device, dtype=torch.float32)
    rel_b = torch.randn(3, 4, device=device, dtype=torch.float32)
    rel_a_custom = rel_a.detach().clone().requires_grad_(True)
    rel_b_custom = rel_b.detach().clone().requires_grad_(True)
    rel_a_ref = rel_a.detach().clone().requires_grad_(True)
    rel_b_ref = rel_b.detach().clone().requires_grad_(True)

    if mode == 1:
        edge_src_parts = [
            torch.tensor([0, 2, 1], device=device, dtype=torch.int64),
            torch.tensor([1, 2], device=device, dtype=torch.int64),
        ]
    else:
        edge_src_parts = [
            torch.tensor([0, 2, 1], device=device, dtype=torch.int64),
            torch.tensor([0, 0], device=device, dtype=torch.int64),
        ]
    edge_dst_parts = [
        torch.tensor([1, 0, 2], device=device, dtype=torch.int64),
        torch.tensor([2, 1], device=device, dtype=torch.int64),
    ]
    rel_part_ids = [0, 1]
    if mode == 1:
        arity_parts = [1, 1]
        pos_parts = [0, 0]
    else:
        arity_parts = [2, 3]
        pos_parts = [1, 2]

    rel_cat, flat_src, dst_idx = mp.fanin_pack_from_edges(
        [rel_a_custom, rel_b_custom],
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
        mode,
    )
    rel_ref, src_ref, dst_ref = _fanin_pack_edges_ref(
        [rel_a_ref, rel_b_ref],
        edge_src_parts,
        edge_dst_parts,
        rel_part_ids,
        arity_parts,
        pos_parts,
        mode,
    )
    assert torch.allclose(rel_cat, rel_ref)
    assert torch.equal(flat_src, src_ref)
    assert torch.equal(dst_idx, dst_ref)

    out = mp.fanin_reduce(rel_cat, flat_src, dst_idx, dim_size=4, mode=0)
    ref = _fanin_ref(rel_ref, src_ref, dst_ref, dim_size=4, mode=0)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-5, equal_nan=True)

    out.sum().backward()
    ref.sum().backward()
    assert torch.allclose(
        rel_a_custom.grad, rel_a_ref.grad, atol=1e-6, rtol=1e-5, equal_nan=True
    )
    assert torch.allclose(
        rel_b_custom.grad, rel_b_ref.grad, atol=1e-6, rtol=1e-5, equal_nan=True
    )
