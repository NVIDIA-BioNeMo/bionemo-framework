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

"""Tests for the release-facing BioNeMo phage generation skill."""

import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).parents[2]
ROOT_SKILLS = REPO_ROOT / "skills"


def _frontmatter(path: Path) -> dict:
    parts = path.read_text(encoding="utf-8").split("---", maxsplit=2)
    assert len(parts) == 3 and not parts[0].strip(), path
    metadata = yaml.safe_load(parts[1])
    assert isinstance(metadata, dict), path
    return metadata


def test_root_skill_layout() -> None:
    """The compatibility alias should expose the canonical root skill directory."""
    skill_names = {path.name for path in ROOT_SKILLS.iterdir() if (path / "SKILL.md").is_file()}
    alias = REPO_ROOT / ".agents" / "skills"

    assert "bionemo-phage-generation" in skill_names
    assert alias.is_symlink()
    assert alias.readlink() == Path("../skills")
    assert alias.resolve() == ROOT_SKILLS.resolve()


def test_portable_handoff() -> None:
    """The portable skill should retain its complete-checkout handoff contract."""
    skill_root = ROOT_SKILLS / "bionemo-phage-generation"
    skill = (skill_root / "SKILL.md").read_text(encoding="utf-8")
    for marker in (
        "VERSION >= 2.4",
        "bionemo-phage-design/SKILL.md",
        "design-scope-and-viability.md",
        "ema-2025-draft-phage-therapy-quality-guideline.md",
        ".codex-plugin/plugin.json",
        "https://github.com/NVIDIA-BioNeMo/bionemo-recipes",
        "canonical default revision",
        "origin/jstjohn/evo2_phage_gen",
        "separate clean checkout",
        "absolute checkout root",
        "absolute recipe root",
        "original request",
        "Codex",
        "$bionemo-phage-design",
        "--plugin-dir .",
        "/evo2-phage-gen:bionemo-phage-design",
        "missing or integrity-failed skill",
        "plugin's `skills` root",
        "fixed required sibling allowlist",
        "unexpected child skills",
        "required instruction or plugin files",
        "paths and SHA-256",
        "integrity check fails",
    ):
        assert marker in skill

    evals_text = json.dumps(json.loads((skill_root / "evals" / "evals.json").read_text(encoding="utf-8")))
    for marker in (
        "VERSION == 2.4",
        "no recipe-local controller",
        "aggregation",
        "absolute-root Codex",
        "absolute-root Claude",
        "original request",
        "fixed required sibling allowlist",
        "integrity-failed skill",
    ):
        assert marker in evals_text


def test_entry_description() -> None:
    """The release-facing entrypoint should use likely user discovery language."""
    skill_path = ROOT_SKILLS / "bionemo-phage-generation" / "SKILL.md"
    description = str(_frontmatter(skill_path)["description"]).lower()

    for marker in ("bacteriophage genome", "phage therapy", "antibiotic-resistant infections"):
        assert marker in description


def test_root_card_license() -> None:
    """The release-facing skill card should link its declared license."""
    card = (ROOT_SKILLS / "bionemo-phage-generation" / "skill-card.md").read_text(encoding="utf-8")

    assert "[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0.txt)" in card
