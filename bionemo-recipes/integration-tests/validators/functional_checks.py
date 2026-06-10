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

"""Functional validation - runs generated code if GPU is available."""

import subprocess
import sys
from pathlib import Path

import pytest


def run_generated_tests(test_dir: Path, timeout: int = 120) -> None:
    """Run pytest in the generated test directory.

    Requires GPU. Skips if no GPU available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("No GPU available for functional tests")
    except ImportError:
        pytest.skip("torch not available for functional tests")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", str(test_dir)],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(test_dir),
    )
    assert result.returncode == 0, f"Generated tests failed:\n{result.stdout}\n{result.stderr}"
