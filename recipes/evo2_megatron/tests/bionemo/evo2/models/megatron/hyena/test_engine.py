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

import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from bionemo.evo2.models.megatron.hyena import engine
from bionemo.evo2.models.megatron.hyena.subquadratic_safety import ensure_subquadratic_ops_supported


def test_fftconv_func_is_prefix_invariant_when_filter_is_longer_than_input():
    """Short-input FFT convolution should match the prefix of a longer-input convolution."""
    torch.manual_seed(1234)
    batch_size = 2
    hidden_size = 4
    short_len = 5
    long_len = 128
    filter_len = 128

    u_long = torch.randn(batch_size, hidden_size, long_len)
    u_short = u_long[..., :short_len].contiguous()
    k = torch.randn(hidden_size, 1, filter_len)
    d = torch.randn(hidden_size)

    short_out = engine.fftconv_func(u=u_short, k=k, D=d)
    long_out = engine.fftconv_func(u=u_long, k=k, D=d)[..., :short_len]

    torch.testing.assert_close(short_out, long_out, rtol=1e-5, atol=1e-5)


def test_parallel_iir_is_prefix_invariant_when_filter_is_longer_than_input():
    """The IIR prefill convolution should not circularly alias short prefixes."""
    torch.manual_seed(1234)
    batch_size = 2
    hidden_size = 4
    short_len = 5
    long_len = 128
    filter_len = 128

    z_long = torch.randn(batch_size, 3 * hidden_size, long_len)
    z_short = z_long[..., :short_len].contiguous()
    h = torch.randn(hidden_size, filter_len)
    d = torch.randn(hidden_size)

    short_out, _ = engine.parallel_iir(
        z_pre=z_short,
        h=h,
        D=d,
        L=short_len,
        poles=None,
        t=None,
        hidden_size=hidden_size,
        compute_state=False,
    )
    long_out, _ = engine.parallel_iir(
        z_pre=z_long,
        h=h,
        D=d,
        L=long_len,
        poles=None,
        t=None,
        hidden_size=hidden_size,
        compute_state=False,
    )

    torch.testing.assert_close(short_out, long_out[:, :short_len], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("use_subquadratic_ops", [False, True], ids=["torch", "subq"])
def test_parallel_fir_short_cuda_path_matches_torch_depthwise_conv1d(use_subquadratic_ops):
    """Short FIR prefill should match F.conv1d or fail before returning bad subq output."""
    if not torch.cuda.is_available():
        pytest.skip("short FIR CUDA path requires CUDA")
    if use_subquadratic_ops:
        try:
            ensure_subquadratic_ops_supported()
        except RuntimeError as e:
            pytest.xfail(str(e))

    torch.manual_seed(1234)
    batch_size = 2
    seq_len = 17
    hidden_size = 8
    kernel_size = 7
    device = torch.device("cuda")

    u = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight = torch.randn(hidden_size, 1, kernel_size, device=device)
    bias = torch.randn(hidden_size, device=device)

    actual, state = engine.parallel_fir(
        u=u,
        weight=weight,
        bias=bias,
        L=seq_len,
        gated_bias=True,
        fir_length=kernel_size,
        compute_state=True,
        use_subquadratic_ops=use_subquadratic_ops,
    )

    u_bdl = u.transpose(1, 2).contiguous()
    expected = F.conv1d(
        u_bdl.float(),
        weight.float(),
        bias=None,
        stride=1,
        padding=kernel_size - 1,
        groups=hidden_size,
    )[..., :seq_len]
    expected = expected.to(u.dtype) + bias[None, :, None] * u_bdl

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(state, u_bdl[..., -(kernel_size - 1) :])


@pytest.mark.parametrize("flip_filter", [False, True], ids=["noflip", "flip"])
@pytest.mark.parametrize("gated_bias", [False, True], ids=["plainbias", "gatedbias"])
def test_step_fir_block_matches_per_token_loop(flip_filter, gated_bias):
    """The vectorized block step_fir (multi-token chunk) equals looping the single-token step_fir.

    Guards the chunked-prefill FIR fast path: a [B, L, D] block must thread the FIR ring identically
    to stepping the L tokens one at a time (the proven decode primitive), so block and loop agree on
    both the per-position output and the final ring state.
    """
    torch.manual_seed(0)
    batch, channels, kernel_size, seq_len = 2, 8, 5, 7  # ring (cache) size = kernel_size - 1
    weight = torch.randn(channels, 1, kernel_size)
    bias = torch.randn(channels)
    u = torch.randn(batch, seq_len, channels)
    ring0 = torch.randn(batch, channels, kernel_size - 1)

    ring = ring0.clone()
    ys = []
    for t in range(seq_len):
        y_t, ring = engine.step_fir(
            u=u[:, t].clone(),
            fir_state=ring,
            weight=weight.clone(),
            bias=bias,
            gated_bias=gated_bias,
            flip_filter=flip_filter,
        )
        ys.append(y_t)
    y_loop = torch.stack(ys, dim=1)  # [B, L, D]

    ring_block = ring0.clone()
    y_block, ring_block = engine.step_fir(
        u=u.clone(),
        fir_state=ring_block,
        weight=weight.clone(),
        bias=bias,
        gated_bias=gated_bias,
        flip_filter=flip_filter,
    )

    torch.testing.assert_close(y_block, y_loop, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(ring_block, ring, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("act_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_step_iir_block_matches_per_token_loop(act_dtype):
    """The vectorized block step_iir (multi-token chunk) equals looping the single-token step_iir.

    Guards the chunked-prefill IIR fast path: a [B, d, L] block must thread the (real-pole) modal
    state identically to stepping the L tokens one at a time, agreeing on both output and final state.
    The bf16 case mirrors real inference (bf16 activations + residues/D, fp32 state + log-poles) and
    guards the einsum dtype-mismatch a fp32-only test misses (the recurrence must run in fp32).
    """
    torch.manual_seed(0)
    batch, channels, order, seq_len = 2, 8, 4, 7
    log_poles = -(torch.rand(channels, order, 1) * 2.0 + 0.02)  # fp32; exp(.) gives real stable poles in (0, 1)
    # Activations + residues/D arrive in the activation dtype at inference; the IIR state stays fp32.
    residues = torch.randn(channels, order).to(act_dtype)
    decay_bias = torch.randn(channels).to(act_dtype)
    x1 = torch.randn(batch, channels, seq_len).to(act_dtype)
    x2 = torch.randn(batch, channels, seq_len).to(act_dtype)
    v = torch.randn(batch, channels, seq_len).to(act_dtype)
    state0 = torch.randn(batch, channels, order)  # fp32 persistent state

    state = state0.clone()
    ys = []
    for t in range(seq_len):
        y_t, state = engine.step_iir(
            x2=x2[:, :, t].clone(),
            x1=x1[:, :, t].clone(),
            v=v[:, :, t].clone(),
            D=decay_bias,
            residues=residues.clone(),
            poles=log_poles.clone(),
            iir_state=state,
        )
        ys.append(y_t)
    y_loop = torch.stack(ys, dim=-1)  # [B, d, L]

    state_block = state0.clone()
    y_block, state_block = engine.step_iir(
        x2=x2.clone(),
        x1=x1.clone(),
        v=v.clone(),
        D=decay_bias,
        residues=residues.clone(),
        poles=log_poles.clone(),
        iir_state=state_block,
    )

    rtol, atol = (2e-2, 2e-2) if act_dtype == torch.bfloat16 else (1e-4, 1e-4)
    torch.testing.assert_close(y_block, y_loop, rtol=rtol, atol=atol)
    torch.testing.assert_close(state_block, state, rtol=rtol, atol=atol)
