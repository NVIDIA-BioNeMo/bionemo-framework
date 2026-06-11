# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Correctness tests for the Triton sparse decoder kernels.

GPU-gated: these skip cleanly on CPU-only machines (Triton kernels need CUDA), so
the suite stays green everywhere and is validated on a GPU box.

Oracle = the dense, autograd-differentiable reference in sae.kernels.reference.
The decoder-weight gradient kernel uses atomic adds (nondeterministic FP
accumulation), so gradient comparisons use tolerances, not exact equality.
"""

import pytest
import torch
from sae.kernels import HAS_TRITON, TritonDecoderAutograd, reference_decode


pytestmark = pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()),
    reason="Triton sparse decoder kernels require CUDA + Triton",
)

DEVICE = "cuda"

# The Triton kernels accumulate in true fp32, but cuBLAS fp32 matmuls use TF32 by
# default on Ampere+/Hopper (~1e-2 error), which would make the dense reference the
# *less* accurate side. Disable TF32 so the reference is exact fp32 for comparison.
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _random_topk(a, n, k, d, dtype, seed=0):
    """Build random (indices, values, decoder_weight) with unique indices per row."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    # Unique top-k indices per row (argsort of random scores, take first k).
    scores = torch.rand(a, n, generator=g, device=DEVICE)
    indices = scores.argsort(dim=-1)[:, :k].contiguous().to(torch.int64)
    values = torch.rand(a, k, generator=g, device=DEVICE, dtype=torch.float32).to(dtype).contiguous()
    weight = torch.randn(d, n, generator=g, device=DEVICE, dtype=torch.float32).to(dtype)
    return indices, values, weight


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("k", [1, 32])
def test_forward_matches_reference(dtype, k):
    idx, vals, w = _random_topk(a=64, n=4096, k=k, d=256, dtype=dtype)
    out = TritonDecoderAutograd.apply(idx, vals, w)
    ref = reference_decode(idx, vals, w)
    atol, rtol = (1e-3, 1e-3) if dtype == torch.float32 else (5e-2, 5e-2)
    torch.testing.assert_close(out.float(), ref.float(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_matches_reference(dtype):
    idx, vals0, w0 = _random_topk(a=128, n=8192, k=32, d=256, dtype=dtype)
    grad_seed = torch.randn(128, 256, device=DEVICE, dtype=dtype).contiguous()

    # Triton path
    vals_t = vals0.clone().requires_grad_(True)
    w_t = w0.clone().requires_grad_(True)
    (TritonDecoderAutograd.apply(idx, vals_t, w_t) * grad_seed).sum().backward()

    # Dense reference path
    vals_r = vals0.clone().requires_grad_(True)
    w_r = w0.clone().requires_grad_(True)
    (reference_decode(idx, vals_r, w_r) * grad_seed).sum().backward()

    # fp32 is the strict correctness gate. bf16 grads are inherently coarse: a
    # magnitude-~20 dot product has bf16 ulp ~0.06-0.12, so a few-percent relative
    # tolerance with a matching atol is the right bar (the kernel itself accumulates
    # in fp32 and matches the fp64 truth — verified separately).
    atol, rtol = (2e-3, 2e-3) if dtype == torch.float32 else (3e-1, 3e-2)
    torch.testing.assert_close(vals_t.grad.float(), vals_r.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(w_t.grad.float(), w_r.grad.float(), atol=atol, rtol=rtol)


def test_topksae_dense_vs_triton_parity():
    """End-to-end: dense and triton TopKSAE give matching loss + param grads."""
    from sae.architectures import TopKSAE

    torch.manual_seed(0)
    x = torch.randn(256, 128, device=DEVICE)

    def build(impl):
        torch.manual_seed(123)
        sae = TopKSAE(input_dim=128, hidden_dim=1024, top_k=16, normalize_input=True, decoder_impl=impl)
        return sae.to(DEVICE)

    dense = build("dense")
    triton = build("triton")
    triton.load_state_dict(dense.state_dict())  # identical weights

    ld = dense.loss(x)
    lt = triton.loss(x)
    torch.testing.assert_close(lt["total"], ld["total"], atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(lt["mse"], ld["mse"], atol=1e-3, rtol=1e-3)

    ld["total"].backward()
    lt["total"].backward()
    for (n_d, p_d), (n_t, p_t) in zip(dense.named_parameters(), triton.named_parameters()):
        assert n_d == n_t
        if p_d.grad is None and p_t.grad is None:
            continue
        torch.testing.assert_close(p_t.grad, p_d.grad, atol=2e-3, rtol=2e-3, msg=f"grad mismatch: {n_d}")


def test_topksae_parity_with_auxk():
    """Parity including the auxk dead-latent path (codes=None path in triton)."""
    from sae.architectures import TopKSAE

    torch.manual_seed(1)
    x = torch.randn(256, 64, device=DEVICE)

    def build(impl):
        torch.manual_seed(7)
        sae = TopKSAE(
            input_dim=64,
            hidden_dim=512,
            top_k=8,
            normalize_input=True,
            auxk=32,
            dead_tokens_threshold=0,
            decoder_impl=impl,  # threshold 0 -> many "dead" -> exercises auxk
        )
        return sae.to(DEVICE)

    dense, triton = build("dense"), build("triton")
    triton.load_state_dict(dense.state_dict())
    # Prime dead-latent stats identically with one step.
    dense.loss(x)["total"].backward()
    triton.loss(x)["total"].backward()

    ld, lt = dense.loss(x), triton.loss(x)
    torch.testing.assert_close(lt["total"], ld["total"], atol=2e-3, rtol=2e-3)
    assert "aux" in lt and "aux" in ld
