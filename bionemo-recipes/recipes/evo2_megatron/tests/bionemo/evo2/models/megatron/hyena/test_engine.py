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

    torch.manual_seed(1234)
    batch_size = 2
    seq_len = 17
    hidden_size = 8
    kernel_size = 7
    device = torch.device("cuda")

    u = torch.randn(batch_size, seq_len, hidden_size, device=device)
    weight = torch.randn(hidden_size, 1, kernel_size, device=device)
    bias = torch.randn(hidden_size, device=device)

    try:
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
    except RuntimeError as e:
        if use_subquadratic_ops and "failed a CUDA self-test" in str(e):
            pytest.xfail(str(e))
        raise

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
