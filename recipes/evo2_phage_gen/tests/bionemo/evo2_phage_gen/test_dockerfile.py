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
import shutil
import subprocess
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


def test_ci_env_preloads_the_recipe_cudnn_runtime_idempotently(tmp_path: Path):
    """Ray workers must load the recipe cuDNN even when the image already loaded an older copy."""
    environment_script = tmp_path / ".ci_test_env.sh"
    shutil.copyfile(RECIPE_ROOT / ".ci_test_env.sh", environment_script)
    virtual_environment = tmp_path / ".venv"
    (virtual_environment / "bin").mkdir(parents=True)
    (virtual_environment / "bin" / "activate").write_text(f'export VIRTUAL_ENV="{virtual_environment}"\n')
    cudnn_directory = virtual_environment / "lib/python3.12/site-packages/nvidia/cudnn/lib"
    cudnn_directory.mkdir(parents=True)
    cudnn_library = cudnn_directory / "libcudnn.so.9"
    cudnn_library.touch()

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; source "$1"; printf "%s\\n%s\\n" "$LD_PRELOAD" "$LD_LIBRARY_PATH"',
            "_",
            str(environment_script),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "LD_PRELOAD": "/image/libcudnn.so.9",
            "LD_LIBRARY_PATH": "/image/cudnn",
        },
    )

    preload, library_path = result.stdout.splitlines()
    preload_paths = preload.split(":")
    library_paths = library_path.split(":")
    assert preload_paths[0] == str(cudnn_library)
    assert preload_paths.count(str(cudnn_library)) == 1
    assert "/image/libcudnn.so.9" in preload_paths
    assert library_paths[0] == str(cudnn_directory)
    assert library_paths.count(str(cudnn_directory)) == 1
    assert "/image/cudnn" in library_paths
