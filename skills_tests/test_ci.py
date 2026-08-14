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
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "unit-tests-recipes.yml"


def _workflow() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_ci_has_skills_job() -> None:
    """The root skills suite should run from a complete checkout."""
    workflow = _workflow()
    job = workflow["jobs"]["skills-tests"]
    checkout = next(step for step in job["steps"] if step.get("name") == "Checkout repository")
    commands = "\n".join(step.get("run", "") for step in job["steps"])

    assert "sparse-checkout" not in checkout.get("with", {})
    assert "pytest -v skills_tests" in commands
    assert "needs.changed-dirs.outputs.skills_changed == 'true'" in job["if"]
    assert "skills-tests" in workflow["jobs"]["verify-recipe-tests"]["needs"]


def test_ci_skill_globs() -> None:
    """Only repository-level skill edits should select the skills job."""
    workflow = _workflow()
    changed_dirs = workflow["jobs"]["changed-dirs"]
    changed_step = next(step for step in changed_dirs["steps"] if step.get("id") == "changed-skills")
    patterns = changed_step["with"]["files"].splitlines()

    assert changed_dirs["outputs"]["skills_changed"] == "${{ steps.changed-skills.outputs.any_changed }}"
    for root_glob in (
        "skills/**",
        "skills_tests/**",
        ".agents/**",
        ".claude-plugin/**",
        ".codex-plugin/**",
    ):
        assert root_glob in patterns
    for recipe_glob in (
        "**/skills/**",
        "**/.agents/**",
        "**/.claude-plugin/**",
        "**/.codex-plugin/**",
        ".github/workflows/unit-tests-recipes.yml",
    ):
        assert recipe_glob not in patterns


def test_ci_recipe_checkout() -> None:
    """Recipe jobs should retain recipe-only sparse checkouts."""
    workflow = _workflow()
    steps = workflow["jobs"]["unit-tests"]["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout repository")
    commands = "\n".join(step.get("run", "") for step in steps)

    assert checkout["with"]["sparse-checkout"] == "${{ matrix.recipe.dir }}"
    assert "git sparse-checkout" not in commands
