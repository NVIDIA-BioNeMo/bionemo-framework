from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch._dynamo as dynamo


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tri_mul_ext"))

from tri_mul_ext import tri_mul_fused, tri_mul_xbdnn_cublas  # noqa: E402


def _rel_error(actual: torch.Tensor, ref: torch.Tensor) -> float:
    diff = (actual.float() - ref.float()).norm()
    base = ref.float().norm().clamp_min(1e-12)
    return float((diff / base).item())


def _assert_no_graph_breaks(fn, *args) -> None:
    dynamo.reset()
    explain = dynamo.explain(fn)(*args)
    assert explain.graph_break_count == 0
    assert explain.graph_count == 1


def test_tri_mul_fused_matches_reference_kdim2():
    if not torch.cuda.is_available():
        return
    a = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    ref = torch.einsum("bikd,bjkd->bijd", a.float(), b.float()).to(torch.bfloat16)
    out = tri_mul_fused(a, b, k_dim=2, out_dtype=torch.bfloat16)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_tri_mul_fused_matches_reference_kdim1():
    if not torch.cuda.is_available():
        return
    a = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((2, 8, 8, 4), device="cuda", dtype=torch.bfloat16)
    ref = torch.einsum("bkid,bkjd->bijd", a.float(), b.float()).to(torch.bfloat16)
    out = tri_mul_fused(a, b, k_dim=1, out_dtype=torch.bfloat16)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)


def test_tri_mul_fused_compiles_without_graph_breaks():
    if not torch.cuda.is_available():
        return

    def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return tri_mul_fused(a, b, k_dim=2, out_dtype=torch.bfloat16)

    a = torch.randn((1, 32, 32, 32), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((1, 32, 32, 32), device="cuda", dtype=torch.bfloat16)
    _assert_no_graph_breaks(fn, a, b)


def test_tri_mul_xbdnn_cublas_matches_reference():
    if not torch.cuda.is_available():
        return
    x = torch.randn((2, 8, 8, 128), device="cuda", dtype=torch.bfloat16)
    x_bdnn = x.permute(0, 3, 1, 2).contiguous()
    x1, x2, x3, x4 = x.chunk(4, dim=-1)
    ref1 = torch.einsum("bikd,bjkd->bijd", x1.float(), x2.float())
    ref2 = torch.einsum("bkid,bkjd->bijd", x3.float(), x4.float())
    ref = torch.cat([ref1, ref2], dim=-1).to(torch.bfloat16)
    out = tri_mul_xbdnn_cublas(x_bdnn, out_dtype=torch.bfloat16)
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
    assert out.stride() == (8 * 8 * 64, 8, 1, 8 * 8)


def test_tri_mul_xbdnn_cublas_backward_populates_input_grad():
    if not torch.cuda.is_available():
        return
    x = torch.randn((1, 128, 32, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = tri_mul_xbdnn_cublas(x, out_dtype=torch.bfloat16)
    loss = out.float().square().mean()
    loss.backward()
    assert out.requires_grad
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max().item() > 0


def test_tri_mul_xbdnn_cublas_compiles_without_graph_breaks():
    if not torch.cuda.is_available():
        return

    def fn(x_bdnn: torch.Tensor) -> torch.Tensor:
        return tri_mul_xbdnn_cublas(x_bdnn, out_dtype=torch.bfloat16)

    x = torch.randn((1, 128, 32, 32), device="cuda", dtype=torch.bfloat16)
    _assert_no_graph_breaks(fn, x)


def test_tri_mul_xbdnn_cublas_3b_shape_backward_matches_reference():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(42)
    x = torch.randn((2, 128, 256, 256), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = tri_mul_xbdnn_cublas(x, out_dtype=torch.bfloat16)
    loss = out.float().square().mean()
    loss.backward()

    x_ref = x.detach().clone().requires_grad_(True)
    a1, b1, a2, b2 = torch.chunk(x_ref, 4, dim=1)
    batch, d_chunk, n, _ = a1.shape
    ref1 = torch.bmm(a1.reshape(batch * d_chunk, n, n), b1.reshape(batch * d_chunk, n, n).transpose(1, 2))
    ref2 = torch.bmm(a2.reshape(batch * d_chunk, n, n).transpose(1, 2), b2.reshape(batch * d_chunk, n, n))
    ref = torch.cat(
        [ref1.reshape(batch, d_chunk, n, n), ref2.reshape(batch, d_chunk, n, n)],
        dim=1,
    ).permute(0, 2, 3, 1)
    ref_loss = ref.float().square().mean()
    ref_loss.backward()

    out_max_abs = float((out.float() - ref.float()).abs().max().item())
    grad_max_abs = float((x.grad.float() - x_ref.grad.float()).abs().max().item())
    grad_rel = _rel_error(x.grad, x_ref.grad)

    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()
    assert out_max_abs < 1e-2
    assert grad_max_abs < 5e-2
    assert grad_rel < 5e-3
