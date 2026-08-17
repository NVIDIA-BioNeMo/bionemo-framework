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


def test_entry_description() -> None:
    """The release-facing entrypoint should use likely user discovery language."""
    skill_path = ROOT_SKILLS / "bionemo-phage-generation" / "SKILL.md"
    description = str(_frontmatter(skill_path)["description"]).lower()

    for marker in ("bacteriophage genome", "phage therapy", "antibiotic-resistant infections"):
        assert marker in description
