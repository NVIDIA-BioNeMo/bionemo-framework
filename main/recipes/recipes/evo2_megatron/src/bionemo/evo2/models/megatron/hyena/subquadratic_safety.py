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

import warnings
from functools import lru_cache

import torch
import torch.nn.functional as F  # noqa: N812


try:
    import pynvml
except ImportError:  # pragma: no cover - pynvml ships as a subquadratic_ops_torch dependency
    pynvml = None


@lru_cache(maxsize=None)
def _host_driver_cuda_version() -> int | None:
    """Max CUDA version the host NVIDIA driver supports, as an NVML int (e.g. 13020 for 13.2).

    This is the CUDA ceiling reported by ``nvidia-smi`` ("CUDA Version"). Returns None if it
    cannot be determined (pynvml missing, or no driver visible inside the container).
    """
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        try:
            return int(pynvml.nvmlSystemGetCudaDriverVersion())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


@lru_cache(maxsize=None)
def _host_driver_version() -> str | None:
    """Host NVIDIA driver build string (e.g. '595.71.05'), or None if it cannot be determined."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        try:
            version = pynvml.nvmlSystemGetDriverVersion()
            return version.decode() if isinstance(version, bytes) else version
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


@lru_cache(maxsize=None)
def _image_cuda_version() -> int | None:
    """CUDA toolkit version this image/PyTorch build targets, as an NVML int (e.g. 13020 for 13.2).

    The prebuilt subquadratic_ops_torch kernels shipped in the image are compiled for this CUDA
    version. Derived from ``torch.version.cuda`` ("the CUDA install in the image"). Returns None if
    it cannot be parsed.
    """
    cuda = torch.version.cuda
    if cuda:
        try:
            major, _, minor = cuda.partition(".")
            return int(major) * 1000 + (int(minor) if minor else 0) * 10
        except ValueError:
            pass
    try:
        return int(torch._C._cuda_getCompiledVersion())
    except Exception:
        return None


def _cuda_version_to_str(version: int | None) -> str:
    """Format an NVML/CUDA integer version (e.g. 13020) as 'major.minor' (e.g. '13.2')."""
    if version is None:
        return "unknown"
    return f"{version // 1000}.{(version % 1000) // 10}"


def _driver_image_cuda_diagnostic() -> str:
    """Diagnose a host-driver vs image-CUDA mismatch and explain how to fix it.

    subquadratic_ops_torch ships *prebuilt* CUDA kernels (SASS cubins), not runtime-compiled PTX.
    A host driver that only supports an older CUDA version than the image cannot load those cubins,
    so the kernels silently fail to launch and return zeros/garbage -- frequently surfacing as a
    misleading ``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` even though no PTX compilation is involved.
    """
    driver_cuda = _host_driver_cuda_version()
    image_cuda = _image_cuda_version()
    driver_str = _cuda_version_to_str(driver_cuda)
    image_str = _cuda_version_to_str(image_cuda)
    summary = (
        f"Host NVIDIA driver {_host_driver_version() or 'unknown'} supports up to CUDA {driver_str}; "
        f"the CUDA toolkit in this image is {image_str}."
    )

    if driver_cuda is not None and image_cuda is not None and driver_cuda < image_cuda:
        return (
            f"{summary} ROOT CAUSE: the host driver is too old for this image. subquadratic_ops_torch "
            f"ships prebuilt CUDA kernels compiled for CUDA {image_str}, and a driver that only supports "
            f"CUDA {driver_str} cannot load them -- they then silently return zeros/garbage (often "
            f"surfacing as the misleading 'CUDA_ERROR_UNSUPPORTED_PTX_VERSION', even though no runtime PTX "
            f"compilation is involved). FIX: upgrade the HOST NVIDIA driver to one supporting CUDA "
            f">= {image_str} (for CUDA 13.2 use driver r595 or newer). The driver lives on the HOST, not "
            f"inside the container, so rebuilding or changing the image will NOT help; instead run on a "
            f"host whose `nvidia-smi` 'CUDA Version' is >= {image_str}, or use an image whose CUDA is "
            f"<= the host driver's CUDA."
        )
    if driver_cuda is not None and image_cuda is not None:
        return (
            f"{summary} The host driver supports CUDA {driver_str} >= image CUDA {image_str}, so a "
            f"driver/toolkit version mismatch is unlikely to be the cause; verify this GPU's compute "
            f"capability is among subquadratic_ops_torch's prebuilt architectures."
        )
    return (
        f"{summary} Could not fully determine the driver and/or image CUDA versions to diagnose further "
        f"-- check that the NVIDIA driver is visible inside the container and that pynvml is installed."
    )


def warn_if_host_driver_too_old() -> None:
    """Warn (without raising) if the host driver is older than the image's CUDA toolkit.

    Advisory only -- the per-op CUDA self-tests below are the authoritative correctness gate (CUDA
    minor-version compatibility means a slightly older driver can still work for some architectures).
    This surfaces the most common root cause early, with an actionable message, since a too-old host
    driver makes the prebuilt subquadratic_ops_torch kernels silently return invalid output.
    """
    driver_cuda = _host_driver_cuda_version()
    image_cuda = _image_cuda_version()
    if driver_cuda is not None and image_cuda is not None and driver_cuda < image_cuda:
        warnings.warn(f"[subquadratic_ops_torch] {_driver_image_cuda_diagnostic()}", stacklevel=2)


def _raise_subquadratic_self_test_error(op_name: str, detail: str) -> None:
    raise RuntimeError(
        f"subquadratic_ops_torch.{op_name} failed a CUDA self-test ({detail}). "
        "This usually means the prebuilt subquadratic_ops_torch CUDA kernels could not be loaded or "
        "launched on this GPU/driver, so they returned invalid output without raising. "
        f"{_driver_image_cuda_diagnostic()}"
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
    warn_if_host_driver_too_old()
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

    # subquadratic_ops_torch.b2b_causal_conv1d uses the weight[-1] == current-tap convention
    # (same as its causal_conv1d), so the reference convs must NOT flip the weights. Flipping
    # them produces a spurious ~0.6 relative mismatch that wrongly trips the self-test.
    projected = F.conv1d(
        F.pad(x, (proj_kernel_size - 1, 0)),
        proj_weight.unsqueeze(1),
        groups=3 * hidden_size,
    )
    x1, x2, v = projected[:, ::3], projected[:, 1::3], projected[:, 2::3]
    z = x2 * v
    mixed = F.conv1d(
        F.pad(z, (mixer_kernel_size - 1, 0)),
        mixer_weight.unsqueeze(1),
        groups=hidden_size,
    )
    expected = x1 * (mixed + bias[None, :, None] * z)
    _assert_close_or_raise("b2b_causal_conv1d", actual, expected)
