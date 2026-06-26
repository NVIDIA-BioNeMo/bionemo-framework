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

"""Tests for ``bionemo.evo2_phage_gen.nemo_rl_patches``."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

from bionemo.evo2_phage_gen import nemo_rl_patches


def test_apply_nemo_rl_patch_applies_against_installed_package_root(tmp_path: Path, monkeypatch) -> None:
    """The patch command should run from site-packages, not require a source checkout path."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text("diff --git a/nemo_rl/example.py b/nemo_rl/example.py\n")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd))
        return subprocess.CompletedProcess(["patch", *args], 0, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)

    assert result == f"patch applied to {source_root}"
    assert calls == [
        (["--dry-run", "-p1", "-i", str(patch_file)], source_root),
        (["-p1", "-i", str(patch_file)], source_root),
    ]


def test_apply_nemo_rl_patch_reports_already_applied(tmp_path: Path, monkeypatch) -> None:
    """Reverse dry-run success means the patch is already present."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text("diff --git a/nemo_rl/example.py b/nemo_rl/example.py\n")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return_code = 0 if "-R" in args else 1
        return subprocess.CompletedProcess(["patch", *args], return_code, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)

    assert result == f"patch already applied to {source_root}"


def test_patch_nemo_rl_packaging_metadata_includes_subpackages(tmp_path: Path) -> None:
    """The repair helper should make upstream NeMo-RL package all submodules."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '\n'.join(
            [
                "[tool.setuptools]",
                'packages = ["nemo_rl"]',
                "",
                "[project]",
                'requires-python = ">=3.13.13,<3.14"',
            ]
        )
    )

    nemo_rl_patches._patch_nemo_rl_packaging_metadata(tmp_path)

    patched = pyproject.read_text()
    assert 'packages = { find = { include = ["nemo_rl*"] } }' in patched
    assert 'requires-python = ">=3.10"' in patched
