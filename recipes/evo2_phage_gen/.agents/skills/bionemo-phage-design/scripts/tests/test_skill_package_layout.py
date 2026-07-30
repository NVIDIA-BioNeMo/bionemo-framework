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

import json
from pathlib import Path


REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
ROOT_AGENT_DIR = REPO_ROOT / ".agents"
RECIPE_ROOT = REPO_ROOT / "recipes" / "evo2_phage_gen"
RECIPE_AGENT_DIR = RECIPE_ROOT / ".agents"
EXPECTED_RECIPE_SKILLS = {
    "bionemo-phage-design",
    "bionemo-phage-design-adapt-execution",
    "bionemo-phage-design-calibrate-rl-sampling",
    "bionemo-phage-design-collect-genomes",
    "bionemo-phage-design-generate-and-screen",
    "bionemo-phage-design-implement-rl-objectives",
    "bionemo-phage-design-operate-mbridge-sft",
    "bionemo-phage-design-operate-nemo-rl",
    "bionemo-phage-design-plan-rl-objectives",
    "bionemo-phage-design-prepare-sft",
    "bionemo-phage-design-publish-stage-artifacts",
    "bionemo-phage-design-research-evidence",
}


def _skill_names(agent_dir: Path) -> set[str]:
    return {path.name for path in (agent_dir / "skills").iterdir() if (path / "SKILL.md").is_file()}


def _plugin(agent_dir: Path) -> dict:
    return json.loads((agent_dir / ".codex-plugin" / "plugin.json").read_text())


def test_root_agent_package_contains_only_portable_entry_skill() -> None:
    assert _skill_names(ROOT_AGENT_DIR) == {"bionemo-phage-generation"}
    assert _plugin(ROOT_AGENT_DIR)["name"] == "bionemo-phage-generation"
    assert _plugin(ROOT_AGENT_DIR)["skills"] == "./skills/"


def test_recipe_agent_package_owns_implementation_skills() -> None:
    assert _skill_names(RECIPE_AGENT_DIR) == EXPECTED_RECIPE_SKILLS
    assert _plugin(RECIPE_AGENT_DIR)["name"] == "bionemo-phage-design"
    assert (RECIPE_ROOT / "CLAUDE.md").is_symlink()
    assert (RECIPE_ROOT / "CLAUDE.md").readlink() == Path("AGENTS.md")
