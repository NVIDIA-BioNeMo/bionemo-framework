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

"""Integration checks for native extensions built by ``.ci_build.sh``."""

import importlib.util
import subprocess
import sys

import pytest


def test_causal_conv1d_extension_imports_with_active_torch():
    """The CUDA extension must be compiled for the active system Torch ABI."""
    if importlib.util.find_spec("causal_conv1d") is None:
        pytest.skip("causal_conv1d is not installed in this environment")
    result = subprocess.run(
        [sys.executable, "-c", "import causal_conv1d"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
