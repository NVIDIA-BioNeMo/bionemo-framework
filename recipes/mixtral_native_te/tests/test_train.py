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

"""L0 sanity training test for Mixtral FSDP2 + EP recipe."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires a GPU")

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)

requires_sm100 = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] == 10),
    reason="fused_grouped_mlp expert path requires compute capability 10.x; 12.0 support is pending",
)


def _compose_config(recipe_path: Path, tmp_path: Path, overrides: list[str] | None = None):
    """Compose L0_sanity with standard test overrides."""
    base = [
        f"checkpoint.ckpt_dir={tmp_path}",
        f"+wandb.dir={tmp_path}",
        "checkpoint.resume_from_checkpoint=false",
        "wandb.mode=disabled",
    ]
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        return compose(config_name="L0_sanity", overrides=base + list(overrides or []))


def _assert_loss_valid(loss: float | None, label: str = "") -> None:
    tag = f" ({label})" if label else ""
    assert loss is not None, f"Loss is None{tag}"
    loss_val = float(loss)
    assert not torch.isnan(torch.tensor(loss_val)), f"Loss is NaN{tag}"
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}{tag}"


def _run_torchrun(worker: str, tmp_dir: str, nproc: int = 2) -> subprocess.CompletedProcess[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        str(Path(__file__).resolve()),
        worker,
        tmp_dir,
    ]
    env = os.environ.copy()
    env["WANDB_DISABLED"] = "true"
    env["WANDB_MODE"] = "disabled"
    env["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    env["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        cwd=str(Path(__file__).parent.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        env=env,
    )


def _run_l0_sanity_worker(
    tmp_dir: str,
    *,
    num_train_steps: int = 10,
    config_overrides: list[str] | None = None,
    result_filename: str = "l0_sanity_result.json",
) -> None:
    import train_fsdp2_ep
    from distributed_config import DistributedConfig

    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    recipe_root = Path(__file__).parent.parent
    captured: list[tuple[int, float]] = []
    original_log_step = train_fsdp2_ep.PerfLogger.log_step

    def _capture_log_step(self, step, grad_norm, lr):
        if (
            step > 0
            and step % self.logging_frequency == 0
            and self.grad_acc_step_count > 0
            and self._dist_config.is_main_process()
        ):
            avg_loss = (self.running_loss / self.grad_acc_step_count).item()
            captured.append((step, avg_loss))
        return original_log_step(self, step, grad_norm, lr)

    train_fsdp2_ep.PerfLogger.log_step = _capture_log_step

    cfg = _compose_config(
        recipe_root,
        Path(tmp_dir),
        overrides=[f"num_train_steps={num_train_steps}", *(config_overrides or [])],
    )
    min_loss = train_fsdp2_ep.main(cfg)

    dist_config = DistributedConfig()
    if dist_config.is_main_process():
        result = {
            "min_loss": min_loss,
            "losses": captured,
        }
        out_path = Path(tmp_dir) / result_filename
        out_path.write_text(json.dumps(result))
        print(f"L0 sanity min_loss={min_loss}, steps={len(captured)}")


def _worker_l0_sanity(tmp_dir: str) -> None:
    _run_l0_sanity_worker(tmp_dir)


def _worker_l0_sanity_bf16_grouped_linear(tmp_dir: str) -> None:
    _run_l0_sanity_worker(
        tmp_dir,
        config_overrides=[
            "expert_ffn_mode=grouped_linear",
            "fp8_config.enabled=false",
            "fp8_config.quantized_model_init_kwargs.enabled=false",
            "init_from_pretrained=null",
        ],
        result_filename="l0_sanity_bf16_grouped_linear_result.json",
    )


def _worker_single_gpu_bf16_checkpoint_save(tmp_dir: str) -> None:
    _run_l0_sanity_worker(
        tmp_dir,
        num_train_steps=4,
        config_overrides=[
            "parallelism.dp_size=1",
            "parallelism.ep_size=1",
            "expert_ffn_mode=grouped_linear",
            "fp8_config.enabled=false",
            "fp8_config.quantized_model_init_kwargs.enabled=false",
            "init_from_pretrained=null",
            "checkpoint.resume_from_checkpoint=false",
            "checkpoint.save_every_n_steps=2",
            "checkpoint.save_final_model=false",
            "checkpoint.async_save=false",
            "dataset.num_workers=0",
        ],
        result_filename="single_gpu_bf16_checkpoint_save_result.json",
    )


def _worker_single_gpu_bf16_checkpoint_resume(tmp_dir: str) -> None:
    _run_l0_sanity_worker(
        tmp_dir,
        num_train_steps=5,
        config_overrides=[
            "parallelism.dp_size=1",
            "parallelism.ep_size=1",
            "expert_ffn_mode=grouped_linear",
            "fp8_config.enabled=false",
            "fp8_config.quantized_model_init_kwargs.enabled=false",
            "init_from_pretrained=null",
            "checkpoint.resume_from_checkpoint=true",
            "checkpoint.save_every_n_steps=0",
            "checkpoint.save_final_model=false",
            "checkpoint.async_save=false",
            "dataset.num_workers=0",
        ],
        result_filename="single_gpu_bf16_checkpoint_resume_result.json",
    )


def _assert_l0_sanity_result(result: subprocess.CompletedProcess[str], result_path: Path, label: str) -> None:
    if result.returncode != 0:
        print(result.stdout)
        pytest.fail(f"{label} failed with exit code {result.returncode}")

    assert result_path.exists(), f"Worker did not write {result_path.name}"
    payload = json.loads(result_path.read_text())
    losses = payload["losses"]
    min_loss = payload["min_loss"]

    _assert_loss_valid(min_loss, label)
    assert len(losses) >= 2, f"Expected at least 2 logged losses, got {losses}"

    first_loss = losses[0][1]
    last_loss = losses[-1][1]
    _assert_loss_valid(first_loss, "first step")
    _assert_loss_valid(last_loss, "last step")
    assert last_loss < first_loss, (
        f"Loss did not decrease: first={first_loss:.4f}, last={last_loss:.4f}, trajectory={losses}"
    )
    assert min_loss <= last_loss, f"min_loss {min_loss} should be <= last_loss {last_loss}"


@requires_multi_gpu
@requires_sm100
def test_l0_sanity_training_decreases_loss(tmp_path, recipe_path):
    """L0 EP=2 training: finite loss that decreases over the run."""
    result = _run_torchrun("l0_sanity", str(tmp_path))
    _assert_l0_sanity_result(result, tmp_path / "l0_sanity_result.json", "L0 fused GroupedMLP sanity training")


@requires_multi_gpu
def test_l0_sanity_bf16_grouped_linear_decreases_loss(tmp_path):
    """Randomly initialized tiny Mixtral trains in BF16 with grouped-linear experts."""
    result = _run_torchrun("l0_sanity_bf16_grouped_linear", str(tmp_path))
    _assert_l0_sanity_result(
        result,
        tmp_path / "l0_sanity_bf16_grouped_linear_result.json",
        "L0 BF16 grouped-linear sanity training",
    )


@requires_cuda
def test_single_gpu_bf16_training_checkpoint_resume(tmp_path):
    """Tiny BF16 training saves a checkpoint and resumes in a fresh one-GPU process."""
    phase1 = _run_torchrun("single_gpu_bf16_checkpoint_save", str(tmp_path), nproc=1)
    if phase1.returncode != 0:
        print(phase1.stdout)
        pytest.fail(f"Single-GPU BF16 checkpoint-save phase failed with exit code {phase1.returncode}")

    checkpoint = tmp_path / "train_fsdp2_ep" / "step_2"
    assert checkpoint.is_dir(), f"Missing checkpoint directory {checkpoint}"

    phase2 = _run_torchrun("single_gpu_bf16_checkpoint_resume", str(tmp_path), nproc=1)
    if phase2.returncode != 0:
        print(phase2.stdout)
        pytest.fail(f"Single-GPU BF16 checkpoint-resume phase failed with exit code {phase2.returncode}")

    result_path = tmp_path / "single_gpu_bf16_checkpoint_resume_result.json"
    assert result_path.exists(), f"Worker did not write {result_path.name}"
    payload = json.loads(result_path.read_text())
    _assert_loss_valid(payload["min_loss"], "single-GPU BF16 checkpoint resume")

    losses = payload["losses"]
    assert [step for step, _ in losses] == [3, 4], f"Expected resumed steps [3, 4], got {losses}"
    for step, loss in losses:
        _assert_loss_valid(loss, f"resumed step {step}")


if __name__ == "__main__":
    workers = {
        "l0_sanity": _worker_l0_sanity,
        "l0_sanity_bf16_grouped_linear": _worker_l0_sanity_bf16_grouped_linear,
        "single_gpu_bf16_checkpoint_save": _worker_single_gpu_bf16_checkpoint_save,
        "single_gpu_bf16_checkpoint_resume": _worker_single_gpu_bf16_checkpoint_resume,
    }
    workers[sys.argv[1]](sys.argv[2])
