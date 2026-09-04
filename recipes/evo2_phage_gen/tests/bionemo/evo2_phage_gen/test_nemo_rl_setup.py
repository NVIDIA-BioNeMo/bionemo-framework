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
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bionemo.evo2_phage_gen import nemo_rl_setup


class _GenerationWorkerMixin:
    def _generation_adapter_requires_persistent_model_storage(self):
        return False

    def _generation_adapter_model_refit_complete(self):
        return None


class _StorageOnlyGenerationWorkerMixin:
    def _generation_adapter_requires_persistent_model_storage(self):
        return False


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


def test_policy_replay_keeps_sampled_action(tmp_path: Path) -> None:
    """Replay keeps a sampled token finite if recomputed logits move it outside top-k."""
    source = _cached_source()
    if source is None:
        pytest.skip("configured NeMo-RL source is not cached")
    build = nemo_rl_setup._copy_build_source(source, tmp_path / "build")
    nemo_rl_setup.apply_source_patch(build)

    script = """
import torch

from nemo_rl.algorithms.logits_sampling_utils import apply_top_k_top_p

sampled = torch.tensor([5])
for top_k, top_p in ((5, 1.0), (None, 0.9), (5, 0.999)):
    logits = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], requires_grad=True)
    unconditioned, ordinary_mask = apply_top_k_top_p(logits, top_k=top_k, top_p=top_p, chunk_size=1)
    assert torch.isneginf(unconditioned.gather(-1, sampled[:, None])).all()

    filtered, keep_mask = apply_top_k_top_p(
        logits,
        top_k=top_k,
        top_p=top_p,
        chunk_size=1,
        target_token_ids=sampled,
    )
    selected_logprob = torch.log_softmax(filtered, dim=-1).gather(-1, sampled[:, None]).sum()
    assert torch.isfinite(selected_logprob)
    assert keep_mask.gather(-1, sampled[:, None]).all()
    assert keep_mask.sum() == ordinary_mask.sum() + 1
    (-selected_logprob).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.gather(-1, sampled[:, None]).ne(0).all()

logits = torch.tensor([[6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
supported = torch.tensor([0])
ordinary_logits, ordinary_mask = apply_top_k_top_p(logits, top_k=5, top_p=0.999, chunk_size=1)
replay_logits, replay_mask = apply_top_k_top_p(
    logits,
    top_k=5,
    top_p=0.999,
    chunk_size=1,
    target_token_ids=supported,
)
assert torch.equal(ordinary_logits, replay_logits)
assert torch.equal(ordinary_mask, replay_mask)
"""
    subprocess.run([sys.executable, "-c", script], cwd=build, check=True)

    model_utils = (build / "nemo_rl" / "distributed" / "model_utils.py").read_text()
    assert model_utils.count("target_token_ids=target_local") == 3
    assert "target_token_ids=next_tokens" in model_utils

    worker = (build / "nemo_rl" / "models" / "policy" / "workers" / "megatron_policy_worker.py").read_text()
    assert "self._generation_adapter_requires_persistent_model_storage()" in worker
    assert 'self.model, "cpu", move_params=not preserve_model_storage' in worker
    assert "self._generation_adapter_model_refit_complete()" in worker


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
        if name.endswith(".logits_sampling_utils"):
            return SimpleNamespace(
                apply_top_k_top_p=lambda logits, top_k, top_p, chunk_size=None, target_token_ids=None: logits
            )
        if name.endswith(".megatron_worker"):
            return SimpleNamespace(MegatronGenerationMixin=_GenerationWorkerMixin)
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
        if name.endswith(".logits_sampling_utils"):
            return SimpleNamespace(
                apply_top_k_top_p=lambda logits, top_k, top_p, chunk_size=None, target_token_ids=None: logits
            )
        if name.endswith(".megatron_worker"):
            return SimpleNamespace(MegatronGenerationMixin=_GenerationWorkerMixin)
        return SimpleNamespace(init_ray=init_ray)

    monkeypatch.setattr(nemo_rl_setup, "_runtime_is_complete", lambda: True)
    monkeypatch.setattr(nemo_rl_setup.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="external recipe datasets"):
        nemo_rl_setup.assert_nemo_rl_runtime()


def test_runtime_requires_sampled_action_support(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale install fails before filtered policy replay can erase sampled actions."""

    def init_ray(log_dir=None, *, include_dashboard=True, num_cpus=None):
        return None

    def import_module(name):
        if name.endswith(".grpo"):
            return SimpleNamespace(split_environment_timing_metrics=lambda metrics: (metrics, {}))
        if name.endswith(".datasets.utils"):
            return SimpleNamespace(resolve_external_dataset_class=lambda name: name)
        if name.endswith(".logits_sampling_utils"):
            return SimpleNamespace(apply_top_k_top_p=lambda logits, top_k, top_p, chunk_size=None: logits)
        if name.endswith(".megatron_worker"):
            return SimpleNamespace(MegatronGenerationMixin=_GenerationWorkerMixin)
        return SimpleNamespace(init_ray=init_ray)

    monkeypatch.setattr(nemo_rl_setup, "_runtime_is_complete", lambda: True)
    monkeypatch.setattr(nemo_rl_setup.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match="sampled actions in filtered log-probability support"):
        nemo_rl_setup.assert_nemo_rl_runtime()


@pytest.mark.parametrize(
    ("generation_mixin", "expected_error"),
    [
        (SimpleNamespace, "preserve CUDA-graph model storage"),
        (_StorageOnlyGenerationWorkerMixin, "refresh quantized CUDA graphs"),
    ],
)
def test_runtime_requires_graph_storage_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    generation_mixin,
    expected_error,
) -> None:
    """A stale install must expose both graph-storage lifecycle hooks."""

    def init_ray(log_dir=None, *, include_dashboard=True, num_cpus=None):
        return None

    def import_module(name):
        if name.endswith(".grpo"):
            return SimpleNamespace(split_environment_timing_metrics=lambda metrics: (metrics, {}))
        if name.endswith(".datasets.utils"):
            return SimpleNamespace(resolve_external_dataset_class=lambda name: name)
        if name.endswith(".logits_sampling_utils"):
            return SimpleNamespace(
                apply_top_k_top_p=lambda logits, top_k, top_p, chunk_size=None, target_token_ids=None: logits
            )
        if name.endswith(".megatron_worker"):
            return SimpleNamespace(MegatronGenerationMixin=generation_mixin)
        return SimpleNamespace(init_ray=init_ray)

    monkeypatch.setattr(nemo_rl_setup, "_runtime_is_complete", lambda: True)
    monkeypatch.setattr(nemo_rl_setup.importlib, "import_module", import_module)

    with pytest.raises(RuntimeError, match=expected_error):
        nemo_rl_setup.assert_nemo_rl_runtime()
