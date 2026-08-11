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
            ".agents/skills/bionemo-phage-design/references/historical-evidence.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            ".agents/skills/bionemo-phage-design/references/historical-evidence.md",
        ),
        (
            ".agents/skills/bionemo-phage-design/SKILL.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            ".agents/skills/bionemo-phage-design/SKILL.md",
        ),
        (
            ".agents/skills/bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            ".agents/skills/bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md",
        ),
        (
            ".agents/skills/bionemo-phage-design/assets/VALIDATION.md",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main/recipes/demo/"
            ".agents/skills/bionemo-phage-design/assets/VALIDATION.md",
        ),
        (
            ".agents/skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design/",
            "https://github.com/NVIDIA-BioNeMo/bionemo-framework/tree/main/recipes/demo/"
            ".agents/skills/bionemo-phage-design/assets/literature/king-2025-generative-phage-design",
        ),
    ],
)
def test_hidden_recipe_links_target_github(target: str, expected: str, tmp_path: Path) -> None:
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
