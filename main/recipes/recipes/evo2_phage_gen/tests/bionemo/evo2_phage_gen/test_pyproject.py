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


def _requirements_by_name(requirements: list[str]) -> dict[str, Requirement]:
    """Parse a requirement collection into a normalized name mapping."""
    parsed = (Requirement(requirement) for requirement in requirements)
    return {requirement.name.lower(): requirement for requirement in parsed}


def test_report_runtime_declares_tabulate_dependency():
    """Installed report commands must include pandas' Markdown-table backend."""
    config = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())

    assert any(Requirement(dependency).name == "tabulate" for dependency in config["project"]["dependencies"])


def test_runtime_dependencies_exclude_reported_vulnerable_ray_and_pillow_versions():
    """Both runtime metadata surfaces must select patched Ray and Pillow releases."""
    config = tomllib.loads((RECIPE_ROOT / "pyproject.toml").read_text())
    nemo_rl_metadata = next(
        metadata for metadata in config["tool"]["uv"]["dependency-metadata"] if metadata["name"] == "nemo-rl"
    )
    dependency_sets = [
        _requirements_by_name(config["project"]["dependencies"]),
        _requirements_by_name(nemo_rl_metadata["requires-dist"]),
    ]

    for dependencies in dependency_sets:
        assert not dependencies["ray"].specifier.contains("2.55.1")
        assert dependencies["ray"].specifier.contains("2.56.0")
        assert not dependencies["pillow"].specifier.contains("12.2.0")
        assert dependencies["pillow"].specifier.contains("12.3.0")


def test_security_constraints_reject_reported_versions_and_allow_fixed_versions():
    """Transitive dependencies must not resolve to scanner-reported vulnerable releases."""
    constraints = _requirements_by_name(
        [
            line
            for line in (RECIPE_ROOT / "security_constraints.txt").read_text().splitlines()
            if line and not line.startswith("#")
        ]
    )
    vulnerable_and_fixed = {
        "black": ("25.1.0", "26.3.1"),
        "bleach": ("6.2.0", "6.4.0"),
        "cryptography": ("42.0.8", "46.0.6"),
        "gitpython": ("3.1.46", "3.1.59"),
        "jupyter-server": ("2.16.0", "2.20.0"),
        "jupyterlab": ("4.4.5", "4.5.10"),
        "mistune": ("3.1.3", "3.3.4"),
        "nbconvert": ("7.16.6", "7.17.1"),
        "notebook": ("7.4.5", "7.5.6"),
        "pygments": ("2.19.2", "2.20.0"),
        "python-multipart": ("0.0.20", "0.0.22"),
        "transformers": ("5.0.0", "5.8.1"),
        "urllib3": ("2.6.3", "2.7.0"),
    }

    assert vulnerable_and_fixed.keys() <= constraints.keys()
    for name, (vulnerable, fixed) in vulnerable_and_fixed.items():
        assert not constraints[name].specifier.contains(vulnerable), name
        assert constraints[name].specifier.contains(fixed), name
