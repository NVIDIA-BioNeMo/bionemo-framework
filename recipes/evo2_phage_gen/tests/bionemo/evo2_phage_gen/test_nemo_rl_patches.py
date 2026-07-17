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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_patched_nemo_rl_prompt_digest_matches_portable_vector() -> None:
    from nemo_rl.models.generation.interfaces import (
        generation_prompt_token_ids_bytes,
        generation_prompt_token_ids_sha256,
    )

    expected_hex = (
        "67656e65726174696f6e2e70726f6d70745f746f6b656e5f6964732e763100"
        "0000000000000001000000000000002b"
    )
    expected_sha256 = "8fcfb284618fdd1c28d8a7022eee50831e44986fac86e48b396800bf5ba2c93b"

    _require(generation_prompt_token_ids_bytes([43]).hex() == expected_hex, "prompt digest bytes drifted")
    _require(
        generation_prompt_token_ids_sha256([43]) == expected_sha256,
        "prompt digest SHA256 drifted",
    )
    for invalid in ([True], [-1], [2**63], type("ListSubclass", (list,), {})([43])):
        with pytest.raises((TypeError, ValueError), match="prompt token IDs"):
            generation_prompt_token_ids_sha256(invalid)


def test_production_evo2_generation_worker_imports_with_maintained_patch() -> None:
    """The recipe worker must depend only on symbols shipped by the maintained patch."""
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    from bionemo.evo2.vllm.nemo_generation_worker import Evo2NemoRlGenerationWorkerImpl

    _require(
        issubclass(Evo2NemoRlGenerationWorkerImpl, VllmGenerationWorkerImpl),
        "the production Evo2 worker must remain a thin NeMo-RL vLLM worker subclass",
    )


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


def test_apply_nemo_rl_patch_applies_default_series_in_order(tmp_path: Path, monkeypatch) -> None:
    """The default CLI path should apply both maintained patches in declared order."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    patch_paths = (tmp_path / "policy.patch", tmp_path / "vllm.patch")
    for path in patch_paths:
        path.write_text(f"diff --git a/{path.stem} b/{path.stem}\n")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)
    calls: list[list[str]] = []

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(nemo_rl_patches, "DEFAULT_PATCHES", patch_paths)

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        _require(cwd == source_root, "patch series used the wrong source root")
        calls.append(args)
        return subprocess.CompletedProcess(["patch", *args], 0, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    result = nemo_rl_patches.apply_nemo_rl_patch()

    _require(
        result == f"patch applied to {source_root}\npatch applied to {source_root}",
        "default series result drifted",
    )
    _require(
        [Path(args[-1]).name for args in calls] == ["policy.patch", "policy.patch", "vllm.patch", "vllm.patch"],
        "default patch order drifted",
    )


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


def test_vllm_patch_excludes_request_timing_telemetry() -> None:
    """Stable phase timers, not optional vLLM request metrics, own timing evidence."""
    patch_text = (nemo_rl_patches.RECIPE_ROOT / "patches" / "nemo-rl-evo2-vllm.patch").read_text()

    forbidden = (
        "generation_first_token_latency_s",
        "generation_decode_s",
        "vLLM request timing metrics are missing or inconsistent",
        "metrics.first_token_latency",
        "metrics.last_token_ts",
    )
    _require(
        all(value not in patch_text for value in forbidden),
        "diagnostic request timing telemetry entered the dependency patch",
    )


def test_maintained_patches_apply_to_pinned_nemo_rl_source(tmp_path: Path) -> None:
    """The maintained series should apply in order against the pinned clean source."""
    _, revision = nemo_rl_patches._nemo_rl_source_pin()
    source_root = nemo_rl_patches._find_cached_nemo_rl_source(revision)
    if source_root is None:
        pytest.skip("Pinned NeMo-RL source is not present in the uv git cache")

    build_root = tmp_path / "nemo-rl-source"
    shutil.copytree(source_root / "nemo_rl", build_root / "nemo_rl")
    for patch_path in nemo_rl_patches.DEFAULT_PATCHES:
        result = nemo_rl_patches._run_patch(
            ["--batch", "-p1", "-i", str(patch_path)],
            cwd=build_root,
        )
        _require(result.returncode == 0, result.stdout)


def test_maintained_patch_inventory_is_narrow_and_does_not_patch_vllm_core() -> None:
    """Only recipe-required NeMo-RL seams belong in the dependency patches."""
    paths = set()
    for patch_path in nemo_rl_patches.DEFAULT_PATCHES:
        for line in patch_path.read_text().splitlines():
            if line.startswith("diff --git a/"):
                paths.add(line.split()[2].removeprefix("a/"))

    _require(
        paths
        == {
            "nemo_rl/distributed/worker_groups.py",
            "nemo_rl/models/generation/interfaces.py",
            "nemo_rl/models/generation/vllm/config.py",
            "nemo_rl/models/generation/vllm/vllm_generation.py",
            "nemo_rl/models/generation/vllm/vllm_worker.py",
            "nemo_rl/models/megatron/setup.py",
            "nemo_rl/models/policy/lm_policy.py",
            "nemo_rl/models/policy/workers/megatron_policy_worker.py",
        },
        f"dependency patch inventory drifted: {sorted(paths)}",
    )
    _require(
        all(not path.startswith("vllm/") for path in paths),
        "the recipe patch must not modify upstream vLLM core",
    )


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
    megatron_setup = types.SimpleNamespace(
        _apply_target_allowlist_prefixes=object(),
        NoRefitMegatronBridge=object(),
        _uses_colocated_megatron_generation=object(),
    )
    modules = {
        "nemo_rl.models.generation.interfaces": types.SimpleNamespace(
            generation_prompt_token_ids_sha256=object()
        ),
        "nemo_rl.models.megatron.setup": megatron_setup,
        "nemo_rl.models.generation.vllm.config": types.SimpleNamespace(VllmActorExecutionConfig=object()),
        "nemo_rl.models.generation.vllm.vllm_generation": types.SimpleNamespace(_request_seeds_for_dp_stream=object()),
    }

    monkeypatch.setattr(nemo_rl_patches.importlib, "import_module", lambda name: modules[name])

    nemo_rl_patches.assert_nemo_rl_patch_symbols()
