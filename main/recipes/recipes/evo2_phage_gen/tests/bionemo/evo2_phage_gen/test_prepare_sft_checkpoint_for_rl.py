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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen import prepare_sft_checkpoint_for_rl


def _write_source(root: Path) -> Path:
    iteration = root / "iter_0005200"
    iteration.mkdir(parents=True)
    (iteration / ".metadata").write_bytes(b"source metadata")
    (iteration / "__0_0.distcp").write_bytes(b"source checkpoint payload")
    (iteration / "common.pt").write_bytes(b"common")
    (iteration / "train_state.pt").write_bytes(b"iteration training state")
    (root / "latest_train_state.pt").write_bytes(b"latest training state")
    (root / "latest_checkpointed_iteration.txt").write_text("5200\n")
    (iteration / "run_config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {
                    "_target_": "example.Model",
                    "no_sync_func": {"_target_": "runtime.no_sync"},
                    "grad_sync_func": {"_target_": "runtime.grad_sync"},
                    "param_sync_func": {"_target_": "runtime.param_sync"},
                    "finalize_model_grads_func": {"_target_": "runtime.finalize"},
                    "grad_scale_func": {"_target_": "runtime.grad_scale"},
                    "timers": {"_target_": "megatron.core.timers.Timers"},
                    "hidden_size": 4096,
                },
                "optimizer": {
                    "_target_": "example.Optimizer",
                    "timers": {"_target_": "megatron.core.timers.Timers"},
                    "lr": 1e-5,
                },
                "checkpoint": {"load_optim": True, "load_rng": True},
            },
            sort_keys=False,
        )
    )
    return iteration


def test_prepare_builds_reusable_rl_checkpoint(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_iteration = _write_source(source_root)
    output = tmp_path / "sft-checkpoint"
    calls: list[tuple[Path, Path, bool]] = []

    def fake_remove_optimizer(source: Path, destination: Path, *, preserve_model_object_state: bool = False) -> Path:
        calls.append((source, destination, preserve_model_object_state))
        target = destination / source.name
        shutil.copytree(source, target)
        (destination / "latest_checkpointed_iteration.txt").write_text("5200\n")
        (destination / "latest_train_state.pt").write_bytes(b"latest training state")
        return destination

    monkeypatch.setattr(prepare_sft_checkpoint_for_rl, "remove_optimizer", fake_remove_optimizer)

    prepared = prepare_sft_checkpoint_for_rl.prepare_sft_checkpoint_for_rl(source_root, output)

    assert prepared == output / "iter_0005200"
    assert len(calls) == 1
    assert calls[0][0] == source_iteration.resolve()
    assert calls[0][2] is True
    assert (prepared / "__0_0.distcp").read_bytes() == b"source checkpoint payload"
    assert not (prepared / "train_state.pt").exists()
    assert not (output / "latest_train_state.pt").exists()
    config = yaml.safe_load((prepared / "run_config.yaml").read_text())
    assert config["model"]["hidden_size"] == 4096
    assert config["optimizer"]["lr"] == 1e-5
    assert config["checkpoint"] == {"load_optim": True, "load_rng": True}
    for field in prepare_sft_checkpoint_for_rl.MODEL_RUNTIME_FIELDS:
        assert config["model"][field] is None
    for field in prepare_sft_checkpoint_for_rl.OPTIMIZER_RUNTIME_FIELDS:
        assert config["optimizer"][field] is None

    manifest = json.loads((output / "preparation-manifest.json").read_text())
    assert manifest["state"] == "succeeded"
    assert manifest["source_sft_checkpoint"] == str(source_iteration.resolve())
    assert manifest["prepared_sft_checkpoint"] == str(prepared.resolve())
    assert manifest["copy_mode"] == "model-only-dcp-rewrite"
    assert manifest["source_sft_checkpoint_tree_unchanged"] is True
    assert manifest["sanitized_model_runtime_fields"] == list(prepare_sft_checkpoint_for_rl.MODEL_RUNTIME_FIELDS)
    assert manifest["sanitized_optimizer_runtime_fields"] == list(
        prepare_sft_checkpoint_for_rl.OPTIMIZER_RUNTIME_FIELDS
    )
    assert manifest["stripped_dcp_key_prefixes"] == ["optimizer.", "opt_param_scheduler.", "rng_state"]

    assert prepare_sft_checkpoint_for_rl.prepare_sft_checkpoint_for_rl(source_root, output) == prepared
    assert len(calls) == 1


def test_existing_prepared_checkpoint_rejects_changed_source(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    source_iteration = _write_source(source_root)
    output = tmp_path / "sft-checkpoint"

    def fake_remove_optimizer(source: Path, destination: Path, *, preserve_model_object_state: bool = False) -> Path:
        shutil.copytree(source, destination / source.name)
        return destination

    monkeypatch.setattr(prepare_sft_checkpoint_for_rl, "remove_optimizer", fake_remove_optimizer)
    prepare_sft_checkpoint_for_rl.prepare_sft_checkpoint_for_rl(source_root, output)
    (source_iteration / "run_config.yaml").write_text("model: {}\noptimizer: {}\n")

    try:
        prepare_sft_checkpoint_for_rl.prepare_sft_checkpoint_for_rl(source_root, output)
    except FileExistsError as error:
        assert "does not match the selected source checkpoint" in str(error)
    else:
        raise AssertionError("changed source checkpoint reused a stale prepared checkpoint")


def test_schema_one_preparation_is_rebuilt(tmp_path: Path, monkeypatch) -> None:
    """A rerun replaces payloads made before model object state was preserved."""
    source_root = tmp_path / "source"
    _write_source(source_root)
    output = tmp_path / "sft-checkpoint"
    calls = 0

    def fake_remove_optimizer(source: Path, destination: Path, *, preserve_model_object_state: bool = False) -> Path:
        nonlocal calls
        calls += 1
        shutil.copytree(source, destination / source.name)
        return destination

    monkeypatch.setattr(prepare_sft_checkpoint_for_rl, "remove_optimizer", fake_remove_optimizer)
    prepare_sft_checkpoint_for_rl.prepare_sft_checkpoint_for_rl(source_root, output)
    legacy_manifest_path = output / "preparation-manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text())
    legacy_manifest["schema_version"] = 1
    legacy_manifest.pop("model_object_state_preserved", None)
    legacy_manifest_path.write_text(json.dumps(legacy_manifest))

    prepared = prepare_sft_checkpoint_for_rl.prepare_sft_checkpoint_for_rl(source_root, output)

    assert calls == 2
    assert prepared == output / "iter_0005200"
    current_manifest = json.loads(legacy_manifest_path.read_text())
    assert current_manifest["schema_version"] == 2
    assert current_manifest["model_object_state_preserved"] is True
