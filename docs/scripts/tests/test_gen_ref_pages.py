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

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


try:
    import mkdocs_gen_files  # noqa: F401
except ModuleNotFoundError:
    sys.modules["mkdocs_gen_files"] = ModuleType("mkdocs_gen_files")

gen_ref_pages = importlib.import_module("docs.scripts.gen_ref_pages")


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        (
            "skills/bionemo-phage-design/references/historical-evidence.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            "skills/bionemo-phage-design/references/historical-evidence.md",
        ),
        (
            "skills/bionemo-phage-design/SKILL.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            "skills/bionemo-phage-design/SKILL.md",
        ),
        (
            "skills/bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            "skills/bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md",
        ),
        (
            "skills/bionemo-phage-design/assets/VALIDATION.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            "skills/bionemo-phage-design/assets/VALIDATION.md",
        ),
        (
            "skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design/",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/tree/main/recipes/demo/"
            "skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design",
        ),
    ],
)
def test_unpublished_recipe_skill_links_target_github(target: str, expected: str, tmp_path: Path) -> None:
    root = tmp_path
    source = root / "recipes" / "demo" / "README.md"
    target_path = source.parent / target
    if target.endswith("/"):
        target_path.mkdir(parents=True)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.touch()
    source.write_text(f"[reference]({target})", encoding="utf-8")

    rewritten = gen_ref_pages._rewrite_relative_links(
        source,
        dest=Path("main/recipes/recipes/demo/index.md"),
        root=root,
        text=source.read_text(encoding="utf-8"),
    )

    assert rewritten == f"[reference]({expected})"


def test_example_readme_and_script_copy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "examples"
    source_dir.mkdir()
    readme = source_dir / "README.md"
    script = source_dir / "run.sh"
    readme.write_text("[run](run.sh)\n", encoding="utf-8")
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    text_copies: list[tuple[Path, Path]] = []
    binary_copies: list[tuple[Path, Path]] = []

    monkeypatch.setattr(
        gen_ref_pages,
        "copy_text_file",
        lambda source, dest, root, log: text_copies.append((source, dest)),
    )
    monkeypatch.setattr(
        gen_ref_pages,
        "copy_binary_file",
        lambda source, dest, log: binary_copies.append((source, dest)),
    )
    monkeypatch.setattr(
        gen_ref_pages,
        "write_directory_index",
        lambda *args: pytest.fail("README.md should supply the directory index"),
    )

    copied_docs = gen_ref_pages.copy_docs_from_dir(
        source_dir,
        Path("main/examples/demo/examples"),
        tmp_path,
        "copied",
    )

    assert text_copies == [(readme, Path("main/examples/demo/examples/index.md"))]
    assert binary_copies == [(script, Path("main/examples/demo/examples/run.sh"))]
    assert copied_docs == [Path("main/examples/demo/examples/index.md")]


def test_example_support_file_is_copied_next_to_generated_readme(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Linked example configuration must exist beside its generated documentation."""
    source_dir = tmp_path / "examples"
    source_dir.mkdir()
    (source_dir / "README.md").write_text(
        "[configuration](default-settings.yaml)\n",
        encoding="utf-8",
    )
    settings = source_dir / "default-settings.yaml"
    settings.write_text("temperature: 1.0\n", encoding="utf-8")
    generated_root = tmp_path / "generated"

    def open_generated(path: Path, mode: str):
        destination = generated_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination.open(mode)

    generated_files = ModuleType("mkdocs_gen_files")
    generated_files.open = open_generated
    generated_files.set_edit_path = lambda *args: None
    monkeypatch.setattr(gen_ref_pages, "mkdocs_gen_files", generated_files)

    copied_docs = gen_ref_pages.copy_docs_from_dir(
        source_dir,
        Path("main/examples/demo/examples"),
        tmp_path,
        "copied",
    )

    generated_settings = generated_root / "main/examples/demo/examples/default-settings.yaml"
    assert generated_settings.read_text(encoding="utf-8") == "temperature: 1.0\n"
    assert copied_docs == [Path("main/examples/demo/examples/index.md")]


def test_run_results_are_not_doc_support() -> None:
    assert not gen_ref_pages._should_copy_support_file(Path("results/run-001/records.json"))
