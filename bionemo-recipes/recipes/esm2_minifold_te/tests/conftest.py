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

"""Pytest fixtures for ESM2-MiniFold TE tests."""

import sys
from pathlib import Path

import pytest
import torch


# Add recipe root to path so we can import recipe modules
sys.path.insert(0, str(Path(__file__).parent.parent))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


@pytest.fixture(autouse=True)
def set_seed():
    """Set seed before each test for reproducibility."""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


@pytest.fixture
def device():
    return DEVICE
