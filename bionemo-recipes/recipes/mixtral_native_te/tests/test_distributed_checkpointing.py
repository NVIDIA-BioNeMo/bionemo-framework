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

"""Distributed checkpoint stop-go tests for the Mixtral native TE recipe."""

import gc
import os
import socket
import subprocess
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from transformers import PreTrainedTokenizerFast


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


def _create_local_tokenizer(tmp_path: Path) -> str:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "[UNK]": 0,
                "[PAD]": 1,
                "[BOS]": 2,
                "[EOS]": 3,
                "hello": 4,
                "world": 5,
                "mixtral": 6,
                "token": 7,
                "checkpoint": 8,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast_tokenizer.save_pretrained(tokenizer_dir)
    return str(tokenizer_dir)


def _assert_loss_valid(loss):
    assert loss is not None
    loss_val = float(loss)
    assert torch.isfinite(torch.tensor(loss_val)), f"Loss is not finite: {loss_val}"


def _assert_checkpoint_step(ckpt_subdir, step, num_ranks, is_ddp, use_distributed_checkpoint=False):
    step_dir = os.path.join(ckpt_subdir, f"step_{step}")
    assert os.path.isdir(step_dir), f"Step {step} directory not found: {step_dir}"
    files = os.listdir(step_dir)
    if is_ddp and not use_distributed_checkpoint:
        assert "checkpoint.pt" in files, f"Missing checkpoint.pt in {step_dir}: {files}"
    if use_distributed_checkpoint:
        model_files = [f for f in files if f.startswith("model_rank_")]
        optimizer_files = [f for f in files if f.startswith("optimizer_rank_")]
        assert len(model_files) >= num_ranks, f"Expected model files for {num_ranks} ranks in {step_dir}: {files}"
        assert len(optimizer_files) >= num_ranks, (
            f"Expected optimizer files for {num_ranks} ranks in {step_dir}: {files}"
        )
        assert "metadata.pt" in files, f"Missing metadata.pt in {step_dir}: {files}"
    dataloader_files = [f for f in files if "dataloader" in f]
    assert len(dataloader_files) >= num_ranks, (
        f"Expected dataloader files for {num_ranks} ranks in {step_dir}: {files}"
    )


def _run_single_process_checkpoint_test(recipe_path, tmp_path, main_fn, ckpt_subdir_name, extra_overrides, is_ddp):
    tokenizer_path = _create_local_tokenizer(tmp_path)
    expert_parallel_size = int(
        next(o.split("=", 1)[1] for o in extra_overrides if o.startswith("expert_parallel_size="))
    )
    use_distributed_checkpoint = is_ddp and expert_parallel_size > 1
    common = [
        "checkpoint.save_every_n_steps=5",
        "checkpoint.async_save=false",
        f"dataset.tokenizer_name_or_path={tokenizer_path}",
        *extra_overrides,
    ]

    cfg1 = _compose_config(
        recipe_path,
        tmp_path,
        ["num_train_steps=10", "checkpoint.resume_from_checkpoint=false", *common],
    )
    loss1 = main_fn(cfg1)
    gc.collect()
    torch.cuda.empty_cache()

    ckpt_subdir = os.path.join(str(tmp_path / "ckpt"), ckpt_subdir_name)
    _assert_checkpoint_step(
        ckpt_subdir, 5, num_ranks=1, is_ddp=is_ddp, use_distributed_checkpoint=use_distributed_checkpoint
    )

    cfg2 = _compose_config(
        recipe_path,
        tmp_path,
        ["num_train_steps=15", "checkpoint.resume_from_checkpoint=true", *common],
    )
    loss2 = main_fn(cfg2)
    gc.collect()
    torch.cuda.empty_cache()

    _assert_checkpoint_step(
        ckpt_subdir, 5, num_ranks=1, is_ddp=is_ddp, use_distributed_checkpoint=use_distributed_checkpoint
    )
    _assert_checkpoint_step(
        ckpt_subdir, 10, num_ranks=1, is_ddp=is_ddp, use_distributed_checkpoint=use_distributed_checkpoint
    )
    _assert_loss_valid(loss1)
    _assert_loss_valid(loss2)


def _run_multi_process_checkpoint_test(
    recipe_path, tmp_path, train_script_name, ckpt_subdir_name, extra_overrides, is_ddp
):
    ckpt_dir = str(tmp_path / "ckpt")
    tokenizer_path = _create_local_tokenizer(tmp_path)
    expert_parallel_size = int(
        next(o.split("=", 1)[1] for o in extra_overrides if o.startswith("expert_parallel_size="))
    )
    use_distributed_checkpoint = is_ddp and expert_parallel_size > 1
    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MASTER_PORT"] = str(_reserve_port())
    env["PATH"] = f"/usr/local/cuda/bin:{env['PATH']}"
    env["CPATH"] = f"/usr/local/cuda/include:{env.get('CPATH', '')}".rstrip(":")
    env["BIONEMO_DISABLE_TORCH_COMPILE_HELPERS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["NCCL_DEBUG"] = "WARN"
    env["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"
    train_script = recipe_path / train_script_name
    common = [
        f"checkpoint.ckpt_dir={ckpt_dir}",
        "checkpoint.save_every_n_steps=5",
        "checkpoint.async_save=false",
        "dataset.use_stateful_dataloader=true",
        f"dataset.tokenizer_name_or_path={tokenizer_path}",
        *extra_overrides,
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

    ckpt_subdir = os.path.join(ckpt_dir, ckpt_subdir_name)
    _assert_checkpoint_step(
        ckpt_subdir, 5, num_ranks=2, is_ddp=is_ddp, use_distributed_checkpoint=use_distributed_checkpoint
    )

    result2 = subprocess.run(
        [*base_cmd, "num_train_steps=15", "checkpoint.resume_from_checkpoint=true", *common],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result2.returncode == 0, f"Phase 2 failed: {result2.stderr}"

    _assert_checkpoint_step(
        ckpt_subdir, 5, num_ranks=2, is_ddp=is_ddp, use_distributed_checkpoint=use_distributed_checkpoint
    )
    _assert_checkpoint_step(
        ckpt_subdir, 10, num_ranks=2, is_ddp=is_ddp, use_distributed_checkpoint=use_distributed_checkpoint
    )


def test_checkpoint_save_and_load_single_process_ddp_ep1(recipe_path, tmp_path):
    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_ddp,
        ckpt_subdir_name="train_ddp",
        extra_overrides=["expert_parallel_size=1"],
        is_ddp=True,
    )


def test_checkpoint_save_and_load_single_process_fsdp2_ep1(recipe_path, tmp_path):
    _run_single_process_checkpoint_test(
        recipe_path,
        tmp_path,
        main_fsdp2,
        ckpt_subdir_name="train_fsdp2",
        extra_overrides=["expert_parallel_size=1"],
        is_ddp=False,
    )


@requires_multi_gpu
@pytest.mark.xfail(
    reason=(
        "DDP stop-go checkpointing with expert_parallel_size > 1 is currently unsupported in this recipe. "
        "Resume drops EP expert weights from the saved model state; use the FSDP2 recipe for EP save/resume."
    ),
    strict=False,
)
def test_checkpoint_save_and_load_two_processes_ddp_ep2(recipe_path, tmp_path):
    _run_multi_process_checkpoint_test(
        recipe_path,
        tmp_path,
        "train_ddp.py",
        ckpt_subdir_name="train_ddp",
        extra_overrides=["expert_parallel_size=2"],
        is_ddp=True,
    )


@requires_multi_gpu
def test_checkpoint_save_and_load_two_processes_fsdp2_ep2(recipe_path, tmp_path):
    _run_multi_process_checkpoint_test(
        recipe_path,
        tmp_path,
        "train_fsdp2.py",
        ckpt_subdir_name="train_fsdp2",
        extra_overrides=["expert_parallel_size=2"],
        is_ddp=False,
    )
