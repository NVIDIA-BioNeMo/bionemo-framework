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

"""Distributed checkpoint stop-go tests for the OpenGenome2 Mixtral native TE recipe."""

import gc
import os
import socket
import subprocess

import pytest
import torch
from hydra import compose, initialize_config_dir
from train_fsdp2 import main as main_fsdp2


os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"


def _reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


os.environ["MASTER_PORT"] = str(_reserve_port())

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def _compose_config(recipe_path, tmp_path, overrides):
    ckpt_dir = str(tmp_path / "ckpt")
    base = [
        f"checkpoint.ckpt_dir={ckpt_dir}",
        f"+wandb.dir={tmp_path}",
        "dataset.use_stateful_dataloader=true",
    ]
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        return compose(config_name="L0_sanity", overrides=base + list(overrides))


def _assert_loss_valid(loss):
    assert loss is not None
    loss_val = float(loss)
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"


def _assert_checkpoint_step(ckpt_subdir, step, num_ranks):
    step_dir = os.path.join(ckpt_subdir, f"step_{step}")
    assert os.path.isdir(step_dir), f"Step {step} directory not found: {step_dir}"
    files = os.listdir(step_dir)
    # FSDP2 DCP checkpoints save as .distcp files with a .metadata index,
    # not the older model_rank_*/optimizer_rank_* format.
    distcp_files = [f for f in files if f.endswith(".distcp")]
    has_metadata = ".metadata" in files
    assert has_metadata, f"Missing .metadata in {step_dir}: {files}"
    assert len(distcp_files) >= num_ranks, f"Expected at least {num_ranks} .distcp files in {step_dir}: {files}"
    dataloader_files = [f for f in files if "dataloader" in f]
    assert len(dataloader_files) >= num_ranks, (
        f"Expected dataloader files for {num_ranks} ranks in {step_dir}: {files}"
    )


def test_checkpoint_save_and_load_single_process_fsdp2_ep1(recipe_path, tokenizer_path, tmp_path):
    """Single-process FSDP2 checkpoint save and resume with EP=1."""
    common = [
        "checkpoint.save_every_n_steps=5",
        "checkpoint.async_save=false",
        f"dataset.tokenizer_name_or_path={tokenizer_path}",
        "expert_parallel_size=1",
    ]

    cfg1 = _compose_config(
        recipe_path,
        tmp_path,
        ["num_train_steps=10", "checkpoint.resume_from_checkpoint=false", *common],
    )
    loss1 = main_fsdp2(cfg1)
    gc.collect()
    torch.cuda.empty_cache()

    ckpt_subdir = os.path.join(str(tmp_path / "ckpt"), "train_fsdp2")
    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=1)

    cfg2 = _compose_config(
        recipe_path,
        tmp_path,
        ["num_train_steps=15", "checkpoint.resume_from_checkpoint=true", *common],
    )
    loss2 = main_fsdp2(cfg2)
    gc.collect()
    torch.cuda.empty_cache()

    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=1)
    _assert_checkpoint_step(ckpt_subdir, 10, num_ranks=1)
    _assert_loss_valid(loss1)
    _assert_loss_valid(loss2)


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_fsdp2_ep2(recipe_path, tokenizer_path, tmp_path):
    """Multi-GPU FSDP2 checkpoint save and resume with EP=2 via torchrun."""
    ckpt_dir = str(tmp_path / "ckpt")
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MASTER_PORT"] = str(_reserve_port())
    env["PATH"] = f"/usr/local/cuda/bin:{env['PATH']}"
    env["CPATH"] = f"/usr/local/cuda/include:{env.get('CPATH', '')}".rstrip(":")
    env["BIONEMO_DISABLE_TORCH_COMPILE_HELPERS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = "WARN"
    env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

    train_script = recipe_path / "train_fsdp2.py"
    common = [
        f"checkpoint.ckpt_dir={ckpt_dir}",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.async_save=false",
        "dataset.use_stateful_dataloader=true",
        f"dataset.tokenizer_name_or_path={tokenizer_path}",
        "expert_parallel_size=2",
    ]
    base_cmd = ["torchrun", "--standalone", "--nproc_per_node=2", str(train_script)]

    result1 = subprocess.run(
        [*base_cmd, "num_train_steps=10", "checkpoint.resume_from_checkpoint=false", *common],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result1.returncode == 0, f"Phase 1 failed: {result1.stderr}"

    ckpt_subdir = os.path.join(ckpt_dir, "train_fsdp2")
    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=2)

    result2 = subprocess.run(
        [*base_cmd, "num_train_steps=15", "checkpoint.resume_from_checkpoint=true", *common],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    _assert_checkpoint_step(ckpt_subdir, 5, num_ranks=2)
    _assert_checkpoint_step(ckpt_subdir, 10, num_ranks=2)
