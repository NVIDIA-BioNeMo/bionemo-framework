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

"""Tests for the repository-wide skills CI job."""

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[1]
SKILLS_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "unit-tests-skills.yml"
RECIPES_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "unit-tests-recipes.yml"


def _workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_skills_ci_runs_root_suite() -> None:
    """The root skills suite should run from a complete checkout."""
    workflow = _workflow(SKILLS_WORKFLOW_PATH)
    job = workflow["jobs"]["skills-tests"]
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout repository")
    commands = "\n".join(step.get("run", "") for step in job["steps"])

    assert "sparse-checkout" not in checkout.get("with", {})
    assert "pytest -v skills_tests" in commands


def test_skills_ci_paths() -> None:
    """Only repository-level skill edits should select the skills job."""
    workflow = _workflow(SKILLS_WORKFLOW_PATH)
    triggers = workflow.get("on", workflow.get(True))
    patterns = triggers["push"]["paths"]

    for root_glob in (
        "skills/**",
        "skills_tests/**",
        ".agents/**",
        ".github/workflows/unit-tests-skills.yml",
    ):
        assert root_glob in patterns
    for excluded_glob in (
        ".claude-plugin/**",
        ".codex-plugin/**",
        "**/skills/**",
        "**/.agents/**",
        "**/.claude-plugin/**",
        "**/.codex-plugin/**",
        ".github/workflows/unit-tests-recipes.yml",
    ):
        assert excluded_glob not in patterns


def test_recipe_ci_stays_recipe_only() -> None:
    """Recipe CI should not own the repository-level skills job."""
    workflow = _workflow(RECIPES_WORKFLOW_PATH)
    changed_dirs = workflow["jobs"]["changed-dirs"]

    assert "skills-tests" not in workflow["jobs"]
    assert "skills_changed" not in changed_dirs["outputs"]
    assert all(step.get("id") != "changed-skills" for step in changed_dirs["steps"])
    assert "skills-tests" not in workflow["jobs"]["verify-recipe-tests"]["needs"]

    steps = workflow["jobs"]["unit-tests"]["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout repository")
    commands = "\n".join(step.get("run", "") for step in steps)

    assert checkout["with"]["sparse-checkout"] == "${{ matrix.recipe.dir }}"
    assert "git sparse-checkout" not in commands
