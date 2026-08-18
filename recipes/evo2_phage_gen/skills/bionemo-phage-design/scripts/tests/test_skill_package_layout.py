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
import re
from pathlib import Path


def _find_repo_root(path: Path) -> Path:
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            return parent
    raise RuntimeError(f"could not locate repository root containing .git above {path}")


REPO_ROOT = _find_repo_root(Path(__file__))
RECIPE_ROOT = REPO_ROOT / "recipes" / "evo2_phage_gen"
SKILLS_ROOT = RECIPE_ROOT / "skills"


def _skill_dirs() -> list[Path]:
    return sorted(path for path in SKILLS_ROOT.iterdir() if (path / "SKILL.md").is_file())


def test_recipe_skill_package_layout() -> None:
    alias = RECIPE_ROOT / ".agents" / "skills"
    assert alias.is_symlink()
    assert alias.readlink() == Path("../skills")
    assert alias.resolve() == SKILLS_ROOT.resolve()

    skill_dirs = _skill_dirs()
    assert skill_dirs
    assert (SKILLS_ROOT / "bionemo-phage-design" / "SKILL.md").is_file()

    for skill_dir in skill_dirs:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"name: {skill_dir.name}\n" in text
        assert "\ndescription: " in text
        eval_path = skill_dir / "evals" / "evals.json"
        assert eval_path.is_file()
        suite = json.loads(eval_path.read_text(encoding="utf-8"))
        assert suite["skill_name"] == skill_dir.name
        cases = suite["evals"]
        assert isinstance(cases, list)
        assert cases


def test_recipe_plugin_manifests_point_at_local_skills() -> None:
    expected_names = {"codex": "bionemo-phage-design", "claude": "evo2-phage-gen"}
    for agent, expected_name in expected_names.items():
        manifest = json.loads((RECIPE_ROOT / f".{agent}-plugin" / "plugin.json").read_text(encoding="utf-8"))
        assert manifest["name"] == expected_name
        skill_paths = manifest["skills"] if isinstance(manifest["skills"], list) else [manifest["skills"]]
        assert [path.rstrip("/") for path in skill_paths] == ["./skills"]


def test_local_markdown_links_resolve() -> None:
    markdown_link = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
    sources = [RECIPE_ROOT / "README.md", *SKILLS_ROOT.rglob("*.md")]
    missing: list[str] = []

    for source in sources:
        for raw_target in markdown_link.findall(source.read_text(encoding="utf-8")):
            target_text = raw_target.split("#", maxsplit=1)[0]
            if not target_text or "://" in target_text or target_text.startswith("mailto:"):
                continue
            target = (source.parent / target_text).resolve()
            if not target.is_relative_to(RECIPE_ROOT.resolve()) or not target.exists():
                missing.append(f"{source.relative_to(RECIPE_ROOT)} -> {raw_target}")

    assert not missing, "\n".join(missing)


def test_retired_prompt_terms() -> None:
    prompt_surface = "\n".join(path.read_text(encoding="utf-8") for path in SKILLS_ROOT.rglob("*.md")).lower()
    for retired in (
        "superpowers",
        "test-driven development",
        "red/green",
        "dependency_graph",
        "action-traceability",
        "lineage-contract",
        "monitoring-contract",
        "implementation-contract",
    ):
        assert retired not in prompt_surface
