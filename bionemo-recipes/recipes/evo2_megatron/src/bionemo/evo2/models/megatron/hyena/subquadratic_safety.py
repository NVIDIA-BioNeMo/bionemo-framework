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

from functools import lru_cache

import torch
import torch.nn.functional as F  # noqa: N812


def _raise_subquadratic_self_test_error(op_name: str, detail: str) -> None:
    raise RuntimeError(
        f"subquadratic_ops_torch.{op_name} failed a CUDA self-test ({detail}). "
        "This often happens with CUDA_ERROR_UNSUPPORTED_PTX_VERSION or unsupported GPU/toolchain "
        "combinations. Refusing to run this subquadratic kernel because it can otherwise return "
        "invalid outputs without raising."
    )


def _assert_close_or_raise(op_name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.cuda.synchronize(actual.device)
    if not torch.isfinite(actual).all():
        _raise_subquadratic_self_test_error(op_name, "non-finite output")

    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-4):
        max_diff = (actual.float() - expected.float()).abs().max().item()
        rel = (
            (actual.float() - expected.float()).pow(2).sum().sqrt() / (expected.float().pow(2).sum().sqrt() + 1e-30)
        ).item()
        _raise_subquadratic_self_test_error(op_name, f"max_diff={max_diff:.6g}, rel={rel:.6g}")


@lru_cache(maxsize=None)
def ensure_subquadratic_ops_supported(device_index: int | None = None) -> None:
    """Validate all subquadratic_ops_torch CUDA kernels used by Evo2."""
    ensure_subquadratic_causal_conv1d_supported(device_index)
    ensure_subquadratic_fft_causal_conv1d_supported(device_index)
    ensure_subquadratic_b2b_causal_conv1d_supported(device_index)


@lru_cache(maxsize=None)
def ensure_subquadratic_causal_conv1d_supported(device_index: int | None = None) -> None:
    """Validate subquadratic_ops_torch.causal_conv1d before using it for model data."""
    if not torch.cuda.is_available():
        return

    device_index = torch.cuda.current_device() if device_index is None else device_index
    device = torch.device("cuda", device_index)

    from subquadratic_ops_torch.causal_conv1d import causal_conv1d as subq_causal_conv1d

    batch_size = 1
    hidden_size = 4
    seq_len = 8
    kernel_size = 3
    pad_size = kernel_size - 1

    u = torch.linspace(-1.0, 1.0, steps=batch_size * hidden_size * seq_len, device=device).reshape(
        batch_size, hidden_size, seq_len
    )
    weight = torch.linspace(-0.5, 0.5, steps=hidden_size * kernel_size, device=device).reshape(
        hidden_size, kernel_size
    )

    expected = F.conv1d(
        u,
        weight.unsqueeze(1),
        bias=None,
        stride=1,
        padding=pad_size,
        groups=hidden_size,
    )[..., :seq_len]
    actual = subq_causal_conv1d(F.pad(u, (pad_size, 0)), weight)[..., pad_size:]
    _assert_close_or_raise("causal_conv1d", actual, expected)


@lru_cache(maxsize=None)
def ensure_subquadratic_fft_causal_conv1d_supported(device_index: int | None = None) -> None:
    """Validate subquadratic_ops_torch.fft_causal_conv1d before using it for model data."""
    if not torch.cuda.is_available():
        return

    device_index = torch.cuda.current_device() if device_index is None else device_index
    device = torch.device("cuda", device_index)

    from subquadratic_ops_torch.fft_causal_conv1d import fft_causal_conv1d as subq_fft_causal_conv1d

    batch_size = 1
    hidden_size = 4
    seq_len = 8
    kernel_size = 5

    u = torch.linspace(-1.0, 1.0, steps=batch_size * hidden_size * seq_len, device=device).reshape(
        batch_size, hidden_size, seq_len
    )
    weight = torch.linspace(-0.5, 0.5, steps=hidden_size * kernel_size, device=device).reshape(
        hidden_size, kernel_size
    )

    expected = F.conv1d(
        u,
        weight.flip(-1).unsqueeze(1),
        bias=None,
        stride=1,
        padding=kernel_size - 1,
        groups=hidden_size,
    )[..., :seq_len]
    actual = subq_fft_causal_conv1d(u, weight)
    _assert_close_or_raise("fft_causal_conv1d", actual, expected)


@lru_cache(maxsize=None)
def ensure_subquadratic_b2b_causal_conv1d_supported(device_index: int | None = None) -> None:
    """Validate subquadratic_ops_torch.b2b_causal_conv1d before using it for model data."""
    if not torch.cuda.is_available():
        return

    device_index = torch.cuda.current_device() if device_index is None else device_index
    device = torch.device("cuda", device_index)

    from subquadratic_ops_torch.b2b_causal_conv1d import b2b_causal_conv1d as subq_b2b_causal_conv1d

    batch_size = 1
    hidden_size = 2
    seq_len = 10
    proj_kernel_size = 3
    mixer_kernel_size = 7

    x = torch.linspace(-1.0, 1.0, steps=batch_size * 3 * hidden_size * seq_len, device=device).reshape(
        batch_size, 3 * hidden_size, seq_len
    )
    proj_weight = torch.linspace(-0.5, 0.5, steps=3 * hidden_size * proj_kernel_size, device=device).reshape(
        3 * hidden_size, proj_kernel_size
    )
    mixer_weight = torch.linspace(-0.25, 0.25, steps=hidden_size * mixer_kernel_size, device=device).reshape(
        hidden_size, mixer_kernel_size
    )
    bias = torch.linspace(-0.1, 0.1, steps=hidden_size, device=device)

    actual = subq_b2b_causal_conv1d(x, proj_weight, mixer_weight, bias)

    projected = F.conv1d(
        F.pad(x, (proj_kernel_size - 1, 0)),
        proj_weight.flip(-1).unsqueeze(1),
        groups=3 * hidden_size,
    )
    x1, x2, v = projected[:, ::3], projected[:, 1::3], projected[:, 2::3]
    z = x2 * v
    mixed = F.conv1d(
        F.pad(z, (mixer_kernel_size - 1, 0)),
        mixer_weight.flip(-1).unsqueeze(1),
        groups=hidden_size,
    )
    expected = x1 * (mixed + bias[None, :, None] * z)
    _assert_close_or_raise("b2b_causal_conv1d", actual, expected)
