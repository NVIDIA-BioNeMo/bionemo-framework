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

"""Tests for the recipe package metadata."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


RECIPE_ROOT = Path(__file__).parents[3]


def test_report_runtime_declares_tabulate_dependency():
    """Installed report commands must include pandas' Markdown-table backend."""
    config = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())

    assert any(Requirement(dependency).name == "tabulate" for dependency in config["project"]["dependencies"])
