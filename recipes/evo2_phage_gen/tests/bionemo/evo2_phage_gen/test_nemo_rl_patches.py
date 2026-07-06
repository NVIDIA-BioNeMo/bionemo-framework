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
import shutil
import subprocess
import types
from pathlib import Path

import pytest

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
        (["--batch", "--dry-run", "-p1", "-i", str(patch_file)], source_root),
        (["--batch", "-p1", "-i", str(patch_file)], source_root),
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
        "\n".join(
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


def test_find_cached_nemo_rl_source_rejects_dirty_checkout(tmp_path: Path, monkeypatch) -> None:
    """Repair installs must not reuse uv checkouts that already contain local patch drift."""
    checkout = tmp_path / "cache" / "git-v0" / "checkouts" / "repo" / "rev"
    (checkout / "nemo_rl" / "algorithms").mkdir(parents=True)
    (checkout / "nemo_rl" / "algorithms" / "grpo.py").write_text("")
    (checkout / "pyproject.toml").write_text("")

    monkeypatch.setattr(nemo_rl_patches, "_uv_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(nemo_rl_patches, "_git_head", lambda path: "abc123")
    monkeypatch.setattr(nemo_rl_patches, "_git_worktree_is_clean", lambda path: False)

    assert nemo_rl_patches._find_cached_nemo_rl_source("abc123") is None


def test_run_patch_uses_real_batch_dry_run(tmp_path: Path) -> None:
    """The maintained check path should use a real noninteractive patch dry-run."""
    source_file = tmp_path / "nemo_rl" / "example.py"
    source_file.parent.mkdir()
    source_file.write_text("old\n")
    patch_file = tmp_path / "change.patch"
    patch_file.write_text(
        "\n".join(
            [
                "diff --git a/nemo_rl/example.py b/nemo_rl/example.py",
                "index 1111111..2222222 100644",
                "--- a/nemo_rl/example.py",
                "+++ b/nemo_rl/example.py",
                "@@ -1 +1 @@",
                "-old",
                "+new",
                "",
            ]
        )
    )

    result = nemo_rl_patches._run_patch(["--batch", "--dry-run", "-p1", "-i", str(patch_file)], cwd=tmp_path)

    assert result.returncode == 0
    assert "--forward" in result.args
    assert source_file.read_text() == "old\n"


def test_apply_nemo_rl_patch_is_forward_only_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Rerunning the patcher on an already-patched runtime must not reverse the patch."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    source_file = package_dir / "example.py"
    source_file.write_text("old\n")
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text(
        "\n".join(
            [
                "diff --git a/nemo_rl/example.py b/nemo_rl/example.py",
                "index 1111111..2222222 100644",
                "--- a/nemo_rl/example.py",
                "+++ b/nemo_rl/example.py",
                "@@ -1 +1 @@",
                "-old",
                "+new",
                "",
            ]
        )
    )
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)
    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)

    first_result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)
    second_result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)

    assert first_result == f"patch applied to {source_root}"
    assert second_result == f"patch already applied to {source_root}"
    assert source_file.read_text() == "new\n"


def test_patch_sha256_reports_patch_content_hash(tmp_path: Path) -> None:
    """The launcher should be able to log the exact maintained patch content."""
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("patch contents\n")

    assert nemo_rl_patches.patch_sha256(patch_file) == (
        "3e21aed045526cbe401bb21136236cf0b768acfb13d71101e953f78792549fa1"
    )


def test_maintained_patch_applies_to_pinned_nemo_rl_source(tmp_path: Path) -> None:
    """The actual maintained patch should apply against the pinned clean NeMo-RL source."""
    _, revision = nemo_rl_patches._nemo_rl_source_pin()
    source_root = nemo_rl_patches._find_cached_nemo_rl_source(revision)
    if source_root is None:
        pytest.skip("Pinned NeMo-RL source is not present in the uv git cache")

    build_root = tmp_path / "nemo-rl-source"
    shutil.copytree(source_root / "nemo_rl", build_root / "nemo_rl")
    result = nemo_rl_patches._run_patch(
        ["--batch", "--dry-run", "-p1", "-i", str(nemo_rl_patches.DEFAULT_PATCH)],
        cwd=build_root,
    )

    assert result.returncode == 0, result.stdout


def test_maintained_patch_calls_adapter_generate_worker() -> None:
    """The NeMo worker patch should use the Evo2 adapter's worker-side entry point."""
    patch_text = nemo_rl_patches.DEFAULT_PATCH.read_text()

    assert 'getattr(adapter, "generate_worker", None)' in patch_text
    assert "return generate_worker(self, data=data, greedy=greedy)" in patch_text


def test_assert_nemo_rl_patch_runtime_requires_reverse_patch_match(tmp_path: Path, monkeypatch) -> None:
    """Runtime verification should prove the installed package matches the maintained patch."""
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text("diff --git a/nemo_rl/example.py b/nemo_rl/example.py\n")
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(nemo_rl_patches, "assert_nemo_rl_patch_symbols", lambda: None)

    calls: list[tuple[list[str], Path]] = []

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd))
        return subprocess.CompletedProcess(["patch", *args], 0, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    nemo_rl_patches.assert_nemo_rl_patch_runtime(patch_file)

    assert calls == [(["--batch", "--dry-run", "-R", "-p1", "-i", str(patch_file.resolve())], source_root)]


def test_assert_nemo_rl_patch_symbols_accepts_expected_runtime_symbols(monkeypatch) -> None:
    """Startup should accept a runtime with all expected patched symbols."""
    rollouts = types.SimpleNamespace(collect_environment_metrics=object())
    megatron_generation = types.SimpleNamespace(_load_generation_adapter=object())
    generation_mixin_cls = type(
        "MegatronGenerationMixin",
        (),
        {"_load_generation_adapter": object(), "generate_with_adapter": object()},
    )
    megatron_worker = types.SimpleNamespace(MegatronGenerationMixin=generation_mixin_cls)
    megatron_setup = types.SimpleNamespace(
        _apply_target_allowlist_prefixes=object(),
        NoRefitMegatronBridge=object(),
        _uses_colocated_megatron_generation=object(),
    )
    modules = {
        "nemo_rl.experience.rollouts": rollouts,
        "nemo_rl.models.generation.megatron.megatron_generation": megatron_generation,
        "nemo_rl.models.generation.megatron.megatron_worker": megatron_worker,
        "nemo_rl.models.megatron.setup": megatron_setup,
    }

    monkeypatch.setattr(nemo_rl_patches.importlib, "import_module", lambda name: modules[name])

    nemo_rl_patches.assert_nemo_rl_patch_symbols()
