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

import pytest

from bionemo.evo2.models.megatron.hyena import subquadratic_safety as sqs


def test_cuda_version_to_str():
    """NVML integer versions format as 'major.minor', and None as 'unknown'."""
    assert sqs._cuda_version_to_str(13020) == "13.2"
    assert sqs._cuda_version_to_str(12080) == "12.8"
    assert sqs._cuda_version_to_str(None) == "unknown"


def _patch_versions(monkeypatch, driver: int | None, image: int | None, build: str | None = "590.48.01"):
    monkeypatch.setattr(sqs, "_host_driver_cuda_version", lambda: driver)
    monkeypatch.setattr(sqs, "_image_cuda_version", lambda: image)
    monkeypatch.setattr(sqs, "_host_driver_version", lambda: build)


def test_diagnostic_flags_old_driver(monkeypatch):
    """A driver older than the image CUDA yields a root-cause + actionable host-driver fix."""
    _patch_versions(monkeypatch, driver=13010, image=13020)
    msg = sqs._driver_image_cuda_diagnostic()
    # Names the versions, the cause, an actionable fix, and that the driver is a HOST concern.
    assert "13.1" in msg and "13.2" in msg
    assert "ROOT CAUSE" in msg
    assert "FIX:" in msg and "595" in msg
    assert "HOST" in msg


def test_diagnostic_ok_when_driver_new_enough(monkeypatch):
    """A driver at least as new as the image CUDA reports no mismatch."""
    _patch_versions(monkeypatch, driver=13020, image=13020, build="595.71.05")
    assert "unlikely to be the cause" in sqs._driver_image_cuda_diagnostic()


def test_diagnostic_handles_unknown_versions(monkeypatch):
    """Undeterminable versions degrade gracefully instead of crashing."""
    _patch_versions(monkeypatch, driver=None, image=None, build=None)
    msg = sqs._driver_image_cuda_diagnostic()
    assert "Could not fully determine" in msg
    assert "unknown" in msg


def test_warn_fires_on_old_driver(monkeypatch):
    """warn_if_host_driver_too_old emits a warning when the host driver is too old."""
    _patch_versions(monkeypatch, driver=13010, image=13020)
    with pytest.warns(UserWarning, match="ROOT CAUSE"):
        sqs.warn_if_host_driver_too_old()


def test_warn_silent_when_driver_new_enough(monkeypatch):
    """No warning is emitted when the host driver supports the image CUDA."""
    _patch_versions(monkeypatch, driver=13020, image=13020)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning would raise and fail the test
        sqs.warn_if_host_driver_too_old()


def test_self_test_error_includes_fix(monkeypatch):
    """A tripped kernel self-test surfaces the driver/image diagnostic and fix."""
    _patch_versions(monkeypatch, driver=13010, image=13020)
    with pytest.raises(RuntimeError, match="FIX:"):
        sqs._raise_subquadratic_self_test_error("causal_conv1d", "max_diff=1.0")
