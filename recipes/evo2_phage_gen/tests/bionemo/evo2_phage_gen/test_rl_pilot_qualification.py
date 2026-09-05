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

import json
from pathlib import Path

import pytest
import torch
import yaml

from bionemo.evo2_phage_gen.rl_pilot_qualification import validate_rl_pilot


def _write_checkpoint(root: Path, *, step: int = 3, save_optimizer: bool = True) -> Path:
    checkpoint = root / f"step_{step}"
    iteration = checkpoint / "policy/weights/iter_0000000"
    iteration.mkdir(parents=True)
    (iteration / ".metadata").write_text("metadata\n")
    common_state = {"iteration": step}
    if save_optimizer:
        common_state["optimizer"] = {"state": "present"}
    torch.save(common_state, iteration / "common.pt")
    (checkpoint / "training_info.json").write_text(
        json.dumps({"total_steps": step, "current_step": step, "consumed_samples": step * 16}) + "\n"
    )
    (checkpoint / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpointing": {"save_optimizer": save_optimizer},
                "policy": {"generation": {"mcore_generation_config": {"kv_cache_management_mode": "offload"}}},
            }
        )
    )
    (checkpoint / "train_dataloader.pt").write_bytes(b"state")
    return checkpoint


def _write_logs(tmp_path: Path, checkpoint: Path, *, optimizer_reinitialized: bool = False) -> tuple[Path, Path]:
    runner_log = tmp_path / "runner.log"
    runner_log.write_text(
        "Evo2 native cache release verified: mode=offload tensor_state_allocated=false "
        "allocated_before_mib=28404 allocated_after_mib=12465\n"
    )
    reload_log = tmp_path / "reload.log"
    reload_lines = [
        f"successfully loaded checkpoint from {checkpoint / 'policy/weights/iter_0000000'} "
        "[ t 0/1, p 0/1 ] at iteration 0\n"
    ]
    if optimizer_reinitialized:
        reload_lines.append(
            f"Optimizer state not found at {checkpoint / 'policy/optimizer'}; Optimizer will be freshly initialized.\n"
        )
    reload_log.write_text("".join(reload_lines))
    return runner_log, reload_log


def test_validates_full_state_checkpoint_reload_and_observed_cache_release(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint)

    summary = validate_rl_pilot(
        checkpoint_root=root,
        expected_step=3,
        runner_log=runner_log,
        reload_log=reload_log,
    )

    assert summary == {
        "checkpoint": str(checkpoint.resolve()),
        "restored_total_steps": 3,
        "optimizer_state_saved": True,
        "optimizer_reinitialized": False,
        "cache_release_observations": 1,
        "maximum_observed_cache_release_mib": 15939,
    }


def test_accepts_rich_wrapped_checkpoint_path_in_reload_log(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint)
    weights = str(checkpoint / "policy/weights/iter_0000000")
    split = len(weights) // 2
    reload_log.write_text(
        "\x1b[36m(MegatronPolicyWorker pid=7)\x1b[0m successfully loaded checkpoint from\n"
        f"{weights[:split]}\n{weights[split:]} [ t 0/1, p 0/1 ] at iteration 0\n"
    )

    summary = validate_rl_pilot(
        checkpoint_root=root,
        expected_step=3,
        runner_log=runner_log,
        reload_log=reload_log,
    )

    assert summary["optimizer_state_saved"] is True
    assert summary["optimizer_reinitialized"] is False


def test_validates_explicit_model_only_fallback(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root, save_optimizer=False)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint, optimizer_reinitialized=True)

    summary = validate_rl_pilot(
        checkpoint_root=root,
        expected_step=3,
        runner_log=runner_log,
        reload_log=reload_log,
    )

    assert summary["optimizer_state_saved"] is False
    assert summary["optimizer_reinitialized"] is True


def test_rejects_configured_offload_without_finish_generation_release_evidence(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint)
    runner_log.write_text("Final config: kv_cache_management_mode=offload\n")

    with pytest.raises(ValueError, match="no verified finish_generation cache release"):
        validate_rl_pilot(
            checkpoint_root=root,
            expected_step=3,
            runner_log=runner_log,
            reload_log=reload_log,
        )


def test_accepts_rank_matched_finish_generation_allocation_drop(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint)
    runner_log.write_text(
        "[GPU Rank 3] finish_generation START | alloc=28404MiB private=0MiB\n"
        "[GPU Rank 3] finish_generation END | alloc=12465MiB private=0MiB\n"
    )

    summary = validate_rl_pilot(
        checkpoint_root=root,
        expected_step=3,
        runner_log=runner_log,
        reload_log=reload_log,
    )

    assert summary["cache_release_observations"] == 1
    assert summary["maximum_observed_cache_release_mib"] == 15939


def test_rejects_reload_that_restarts_training_instead_of_restoring_step(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint)
    reload_log.write_text(reload_log.read_text() + "========================= Step 1/3 =========================\n")

    with pytest.raises(ValueError, match="reload process restarted training"):
        validate_rl_pilot(
            checkpoint_root=root,
            expected_step=3,
            runner_log=runner_log,
            reload_log=reload_log,
        )


def test_rejects_full_state_config_without_optimizer_payload(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    torch.save({"iteration": 3}, checkpoint / "policy/weights/iter_0000000/common.pt")
    runner_log, reload_log = _write_logs(tmp_path, checkpoint)

    with pytest.raises(ValueError, match="full-state checkpoint has no"):
        validate_rl_pilot(
            checkpoint_root=root,
            expected_step=3,
            runner_log=runner_log,
            reload_log=reload_log,
        )


def test_rejects_full_state_reload_that_reinitializes_optimizer(tmp_path: Path) -> None:
    root = tmp_path / "checkpoints"
    checkpoint = _write_checkpoint(root)
    runner_log, reload_log = _write_logs(tmp_path, checkpoint, optimizer_reinitialized=True)

    with pytest.raises(ValueError, match="discarded the saved optimizer"):
        validate_rl_pilot(
            checkpoint_root=root,
            expected_step=3,
            runner_log=runner_log,
            reload_log=reload_log,
        )
