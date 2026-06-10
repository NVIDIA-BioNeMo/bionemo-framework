import sys
from pathlib import Path

import pytest
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from flash_trimul import flash_trimul, flash_trimul_reference
from miniformer_te import TriangularUpdateTE
from quantization import ComponentPrecisionConfig


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_trimul_matches_reference_forward_and_backward():
    torch.manual_seed(0)
    proj = torch.randn(2, 64, 64, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    gate = torch.randn_like(proj, requires_grad=True)
    mask = torch.randint(0, 2, (2, 64, 64), device=DEVICE, dtype=torch.int32).to(torch.bfloat16)

    proj_ref = proj.detach().clone().requires_grad_(True)
    gate_ref = gate.detach().clone().requires_grad_(True)

    out = flash_trimul(proj, gate, mask)
    ref = flash_trimul_reference(proj_ref, gate_ref, mask)
    assert out.shape == ref.shape == (2, 64, 64, 64)
    assert torch.allclose(out.float(), ref.float(), atol=2e-1, rtol=2e-1)

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    assert torch.allclose(proj.grad.float(), proj_ref.grad.float(), atol=2e-1, rtol=2e-1)
    assert torch.allclose(gate.grad.float(), gate_ref.grad.float(), atol=2e-1, rtol=2e-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash_trimul_only_saves_proj_gate_and_mask():
    proj = torch.randn(1, 16, 16, 128, device=DEVICE, dtype=torch.float32, requires_grad=True)
    gate = torch.randn_like(proj, requires_grad=True)
    mask = torch.ones(1, 16, 16, device=DEVICE, dtype=torch.float32)

    out = flash_trimul(proj, gate, mask)
    saved = out.grad_fn.saved_tensors
    assert [tuple(t.shape) for t in saved] == [(1, 16, 16, 128), (1, 16, 16, 128), (1, 16, 16)]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_triangular_update_flash_trimul_matches_bmm():
    torch.manual_seed(0)
    cp_ref = ComponentPrecisionConfig(tri_einsum="off", tri_impl="bmm")
    cp_fused = ComponentPrecisionConfig(tri_einsum="off", tri_impl="flash_trimul")
    ref = TriangularUpdateTE(dim=128, component_precision=cp_ref, params_dtype=torch.float32).to(DEVICE)
    fused = TriangularUpdateTE(dim=128, component_precision=cp_fused, params_dtype=torch.float32).to(DEVICE)
    fused.load_state_dict(ref.state_dict())

    x = torch.randn(1, 32, 32, 128, device=DEVICE, dtype=torch.float32, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    mask = torch.ones(1, 32, 32, device=DEVICE, dtype=torch.float32)

    y_ref = ref(x_ref, mask)
    y_fused = fused(x, mask)
    assert y_ref.shape == y_fused.shape == (1, 32, 32, 128)
    assert torch.allclose(y_fused, y_ref, atol=5e-3, rtol=5e-3)

    grad = torch.randn_like(y_ref)
    y_ref.backward(grad)
    y_fused.backward(grad)
    assert torch.allclose(x.grad, x_ref.grad, atol=5e-3, rtol=5e-3)
