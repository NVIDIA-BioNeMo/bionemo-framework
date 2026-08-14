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

"""Repository-wide skill discovery and metadata tests."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).parents[1]


def _tracked_skills() -> list[Path]:
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sorted(
        REPO_ROOT / path for path in map(Path, tracked) if path.name == "SKILL.md" and path.parts[0] == "skills"
    )


def _frontmatter(path: Path) -> dict:
    parts = path.read_text(encoding="utf-8").split("---", maxsplit=2)
    assert len(parts) == 3 and not parts[0].strip(), path
    metadata = yaml.safe_load(parts[1])
    assert isinstance(metadata, dict), path
    return metadata


@pytest.mark.parametrize("skill_path", _tracked_skills(), ids=lambda path: str(path.relative_to(REPO_ROOT)))
def test_skill_frontmatter(skill_path: Path) -> None:
    """Every tracked skill should have matching, discoverable metadata."""
    metadata = _frontmatter(skill_path)
    name = metadata.get("name")

    assert name == skill_path.parent.name
    assert re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
    assert len(name) <= 64
    assert metadata.get("description")
