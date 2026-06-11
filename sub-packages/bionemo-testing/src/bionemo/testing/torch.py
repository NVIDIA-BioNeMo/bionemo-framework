# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import numpy as np
import torch


def check_fp8_support(device_id: int = 0) -> tuple[bool, str, str]:
    """Check if FP8 is supported on the current GPU.

    FP8 requires compute capability 8.9+ (Ada Lovelace/Hopper architecture or newer).
    """
    if not torch.cuda.is_available():
        return False, "0.0", "CUDA not available"
    device_props = torch.cuda.get_device_properties(device_id)
    compute_capability = f"{device_props.major}.{device_props.minor}"
    device_name = device_props.name
    # FP8 is supported on compute capability 8.9+ (Ada Lovelace/Hopper architecture)
    is_supported = (device_props.major > 8) or (device_props.major == 8 and device_props.minor >= 9)
    return is_supported, compute_capability, f"Device: {device_name}, Compute Capability: {compute_capability}"


def recursive_detach(x):
    """Detach all tensors in a nested structure."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    elif isinstance(x, (list, tuple)):
        return type(x)(recursive_detach(item) for item in x)
    elif isinstance(x, dict):
        return {key: recursive_detach(value) for key, value in x.items()}
    else:
        return x


def recursive_assert_approx_equal(x, y, atol=1e-4, rtol=1e-4):
    """Assert that all tensors in a nested structure are approximately equal."""
    if isinstance(x, torch.Tensor):
        torch.testing.assert_close(x, y, atol=atol, rtol=rtol)
    elif isinstance(x, np.ndarray):
        np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), f"Length mismatch: {len(x)} vs {len(y)}"
        for x_item, y_item in zip(x, y):
            recursive_assert_approx_equal(x_item, y_item, atol=atol, rtol=rtol)
    elif isinstance(x, dict):
        assert x.keys() == y.keys()
        for key in x:
            recursive_assert_approx_equal(x[key], y[key], atol=atol, rtol=rtol)
    else:
        assert x == y


def get_device_and_memory_allocated() -> str:
    """Get the current device index, name, and memory usage."""
    current_device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device_index)
    message = f"""
        current device index: {current_device_index}
        current device uuid: {props.uuid}
        current device name: {props.name}
        memory, total on device: {torch.cuda.mem_get_info()[1] / 1024**3:.3f} GB
        memory, available on device: {torch.cuda.mem_get_info()[0] / 1024**3:.3f} GB
        memory allocated for tensors etc: {torch.cuda.memory_allocated() / 1024**3:.3f} GB
        max memory reserved for tensors etc: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB
        """
    return message
