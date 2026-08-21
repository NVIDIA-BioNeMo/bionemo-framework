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

"""Focused tests for NeMo-RL source setup."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bionemo.evo2_phage_gen import nemo_rl_setup


def _cached_source() -> Path | None:
    _, revision = nemo_rl_setup._configured_source()
    return nemo_rl_setup._find_cached_source(revision)


def test_patch_applies(tmp_path: Path) -> None:
    source = _cached_source()
    if source is None:
        pytest.skip("configured NeMo-RL source is not cached")
    build = nemo_rl_setup._copy_build_source(source, tmp_path / "build")
    nemo_rl_setup.apply_source_patch(build, check_only=True)


def test_patch_owns_packaging_changes(tmp_path: Path) -> None:
    source = _cached_source()
    if source is None:
        pytest.skip("configured NeMo-RL source is not cached")
    build = nemo_rl_setup._copy_build_source(source, tmp_path / "build")
    nemo_rl_setup.apply_source_patch(build)
    pyproject = (build / "pyproject.toml").read_text()
    assert 'packages = { find = { include = ["nemo_rl*"] } }' in pyproject
    assert 'requires-python = ">=3.10"' in pyproject


def test_patch_uses_standard_bridge_config_loader(tmp_path: Path) -> None:
    source = _cached_source()
    if source is None:
        pytest.skip("configured NeMo-RL source is not cached")
    build = nemo_rl_setup._copy_build_source(source, tmp_path / "build")
    nemo_rl_setup.apply_source_patch(build)
    setup_source = (build / "nemo_rl" / "models" / "megatron" / "setup.py").read_text()

    assert "cfg_from_pretrained = ConfigContainer.from_yaml(" in setup_source
    assert "_apply_target_allowlist_prefixes(config)" in setup_source
    assert "load_model_config(pretrained_path)" not in setup_source
    assert "_reset_model_runtime_state" not in setup_source
    assert "read_run_config(pretrained_run_config)" not in setup_source


def test_environment_metrics_receive_one_task_namespace(tmp_path: Path) -> None:
    source = _cached_source()
    if source is None:
        pytest.skip("configured NeMo-RL source is not cached")
    build = nemo_rl_setup._copy_build_source(source, tmp_path / "build")
    nemo_rl_setup.apply_source_patch(build)
    rollout_source = (build / "nemo_rl" / "experience" / "rollouts.py").read_text()

    assert 'key.startswith("__timing__/")' in rollout_source
    assert 'metric_key = key if key.startswith("__timing__/") else f"{task_name}/{key}"' in rollout_source


def test_setup_patches_before_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    (source / "nemo_rl" / "algorithms").mkdir(parents=True)
    (source / "nemo_rl" / "algorithms" / "grpo.py").write_text("")
    (source / "pyproject.toml").write_text("[project]\nname='nemo-rl'\n")
    patch = tmp_path / "recipe.patch"
    patch.write_text("patch")
    events: list[str] = []

    monkeypatch.setattr(nemo_rl_setup, "_runtime_is_complete", lambda: False)
    monkeypatch.setattr(nemo_rl_setup, "_configured_source", lambda: ("unused", "revision"))
    monkeypatch.setattr(nemo_rl_setup, "_find_cached_source", lambda revision: source)
    monkeypatch.setattr(
        nemo_rl_setup,
        "apply_source_patch",
        lambda build, selected_patch, check_only=False: events.append("patch") or "ok",
    )
    monkeypatch.setattr(nemo_rl_setup, "assert_nemo_rl_runtime", lambda: events.append("verify"))

    def run(command, **kwargs):
        assert (Path(command[-1]) / "nemo_rl").is_dir()
        events.append("install")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(nemo_rl_setup.subprocess, "run", run)
    result = nemo_rl_setup.setup_nemo_rl(patch_path=patch, force_reinstall=True)

    assert events == ["patch", "install", "verify"]
    assert result == "installed NeMo-RL revision with Evo2 support"


def test_runtime_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    def init_ray(log_dir=None, *, include_dashboard=True, num_cpus=None):
        return None

    monkeypatch.setattr(nemo_rl_setup, "_runtime_is_complete", lambda: True)

    def import_module(name):
        if name.endswith(".grpo"):
            return SimpleNamespace(split_environment_timing_metrics=lambda metrics: (metrics, {}))
        if name.endswith(".datasets.utils"):
            return SimpleNamespace(resolve_external_dataset_class=lambda name: name)
        return SimpleNamespace(init_ray=init_ray)

    monkeypatch.setattr(nemo_rl_setup.importlib, "import_module", import_module)
    nemo_rl_setup.assert_nemo_rl_runtime()


def test_runtime_capabilities_require_external_dataset_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale install must fail before configs use a dotted recipe dataset."""

    def init_ray(log_dir=None, *, include_dashboard=True, num_cpus=None):
        return None

    def import_module(name):
        if name.endswith(".grpo"):
            return SimpleNamespace(split_environment_timing_metrics=lambda metrics: (metrics, {}))
        if name.endswith(".datasets.utils"):
            return SimpleNamespace()
        return SimpleNamespace(init_ray=init_ray)

    monkeypatch.setattr(nemo_rl_setup, "_runtime_is_complete", lambda: True)
    monkeypatch.setattr(nemo_rl_setup.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="external recipe datasets"):
        nemo_rl_setup.assert_nemo_rl_runtime()
