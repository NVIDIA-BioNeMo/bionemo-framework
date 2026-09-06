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

"""Tests for the recipe-local Dockerfile and build context."""

import re
from pathlib import Path


RECIPE_ROOT = Path(__file__).parents[3]


def test_recipe_dockerfile_builds_from_the_recipe_directory():
    """The documented recipe-local build must copy the recipe into its final workdir."""
    assert not (RECIPE_ROOT / "Dockerfile.dockerignore").exists()

    direct_patterns = set((RECIPE_ROOT / ".dockerignore").read_text().splitlines())
    assert {
        "results",
        "data/*",
        "!data/.gitignore",
        "!data/phage_prompts.jsonl",
    } <= direct_patterns

    dockerfile = (RECIPE_ROOT / "Dockerfile").read_text()
    assert "WORKDIR /workspace/bionemo/recipes/evo2_phage_gen\nCOPY . .\n" in dockerfile
    assert dockerfile.count("WORKDIR ") == 1


def test_recipe_dockerfile_installs_pinned_uv_before_ci_build():
    """The recipe image must provide uv before invoking the CI build script."""
    dockerfile = (RECIPE_ROOT / "Dockerfile").read_text()
    uv_copy = re.search(
        r"^COPY --from=ghcr\.io/astral-sh/uv:([^\s]+) /uv /uvx /bin/$",
        dockerfile,
        flags=re.MULTILINE,
    )

    assert uv_copy is not None
    assert uv_copy.group(1) != "latest"
    assert uv_copy.start() < dockerfile.index("./.ci_build.sh")
