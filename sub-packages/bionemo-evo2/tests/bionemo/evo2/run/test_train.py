# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
import argparse
import io
import os
import shlex
from contextlib import redirect_stderr, redirect_stdout
from typing import Tuple

import pytest
import torch
from nemo import lightning as nl
from transformer_engine.pytorch.fp8 import check_fp8_support

from bionemo.evo2.run.train import parse_args, train
from bionemo.testing.assert_optimizer_grads_match import assert_optimizer_states_match
from bionemo.testing.lightning import extract_global_steps_from_log
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.subprocess_utils import run_command_in_subprocess

from .common import small_training_cmd, small_training_finetune_cmd


fp8_available, reason_for_no_fp8 = check_fp8_support()


def run_train_with_std_redirect(args: argparse.Namespace) -> Tuple[str, nl.Trainer]:
    """Run a function with output capture."""
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        with distributed_model_parallel_state():
            trainer: nl.Trainer = train(args)

    train_stdout = stdout_buf.getvalue()
    train_stderr = stderr_buf.getvalue()
    print("Captured STDOUT:\n", train_stdout)
    print("Captured STDERR:\n", train_stderr)
    return train_stdout, trainer


def distributed_training_cmd(
    path,
    max_steps,
    val_check,
    num_devices,
    dp,
    tp,
    cp,
    pp,
    dataset_dir=None,
    training_config=None,
    additional_args: str = "",
):
    """Create distributed training command with specified parallelism settings.

    Args:
        path: Result directory path
        max_steps: Maximum training steps
        val_check: Validation check interval
        num_devices: Total number of devices
        dp: Data parallel size
        tp: Tensor parallel size
        cp: Context parallel size
        pp: Pipeline parallel size
        dataset_dir: Path to preprocessed dataset directory (if None, uses --mock-data)
        training_config: Path to training data config YAML file (required if dataset_dir is provided)
        additional_args: Additional command line arguments
    """
    micro_batch_size = 1 if dp == 2 else 2

    # Use real dataset if provided, otherwise fall back to mock data
    if dataset_dir and training_config:
        data_args = f"-d {training_config} --dataset-dir {dataset_dir}"
    else:
        data_args = "--mock-data"

    cmd = (
        f"train_evo2 {data_args} --result-dir {path} --devices {num_devices} "
        f"--tensor-parallel-size {tp} --pipeline-model-parallel-size {pp} --context-parallel-size {cp} "
        "--model-size 7b --num-layers 4 --hybrid-override-pattern SDH* --limit-val-batches 1 "
        "--no-activation-checkpointing --add-bias-output --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batches 1 "
        f"--seq-length 64 --hidden-dropout 0.0 --attention-dropout 0.0 --seed 42 --workers 0 "
        f"--micro-batch-size {micro_batch_size} --global-batch-size 2 "
        f"--adam-beta1 0 --adam-beta2 0 {additional_args}"
    )
    return cmd


def small_training_mamba_cmd(path, max_steps, val_check, devices: int = 1, additional_args: str = ""):
    cmd = (
        f"train_evo2 --mock-data --result-dir {path} --devices {devices} "
        "--model-size hybrid_mamba_8b --num-layers 2 --hybrid-override-pattern M- --limit-val-batches 1 "
        "--no-activation-checkpointing --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batches 1 "
        f"--seq-length 8 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args}"
    )
    return cmd


def small_training_mamba_finetune_cmd(
    path, max_steps, val_check, prev_ckpt, devices: int = 1, additional_args: str = ""
):
    cmd = (
        f"train_evo2 --mock-data --result-dir {path} --devices {devices} "
        "--model-size hybrid_mamba_8b --num-layers 2 --hybrid-override-pattern M- --limit-val-batches 1 "
        "--no-activation-checkpointing --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batches 1 "
        f"--seq-length 16 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args} --ckpt-dir {prev_ckpt}"
    )
    return cmd


def small_training_llama_cmd(path, max_steps, val_check, devices: int = 1, additional_args: str = ""):
    cmd = (
        f"train_evo2 --no-fp32-residual-connection --mock-data --result-dir {path} --devices {devices} "
        "--model-size 8B --num-layers 2 --limit-val-batches 1 "
        "--no-activation-checkpointing --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batches 1 "
        f"--seq-length 8 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args}"
    )
    return cmd


def small_training_llama_finetune_cmd(
    path, max_steps, val_check, prev_ckpt, devices: int = 1, additional_args: str = ""
):
    cmd = (
        f"train_evo2 --no-fp32-residual-connection --mock-data --result-dir {path} --devices {devices} "
        "--model-size 8B --num-layers 2 --limit-val-batches 1 "
        "--no-activation-checkpointing --create-tensorboard-logger --create-tflops-callback "
        f"--max-steps {max_steps} --warmup-steps 1 --val-check-interval {val_check} --limit-val-batches 1 "
        f"--seq-length 16 --hidden-dropout 0.1 --attention-dropout 0.1 {additional_args} --ckpt-dir {prev_ckpt}"
    )
    return cmd


@pytest.mark.timeout(512)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_finetune_runs(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    num_steps = 2
    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path / "pretrain", max_steps=num_steps, val_check=num_steps)
    stdout_pretrain: str = run_command_in_subprocess(command=command, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" not in stdout_pretrain

    log_dir = tmp_path / "pretrain" / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    tensorboard_dir = log_dir / "dev"

    # Check if logs dir exists
    assert log_dir.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir}"
    assert len(matching_subfolders) == 1, "Only one checkpoint subfolder should be found."
    command_finetune = small_training_finetune_cmd(
        tmp_path / "finetune", max_steps=num_steps, val_check=num_steps, prev_ckpt=matching_subfolders[0]
    )
    stdout_finetune: str = run_command_in_subprocess(command=command_finetune, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout_finetune

    log_dir_ft = tmp_path / "finetune" / "evo2"
    checkpoints_dir_ft = log_dir_ft / "checkpoints"
    tensorboard_dir_ft = log_dir_ft / "dev"

    # Check if logs dir exists
    assert log_dir_ft.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir_ft.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders_ft = [
        p for p in checkpoints_dir_ft.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders_ft, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir_ft}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir_ft.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir_ft.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir_ft}"

    assert len(matching_subfolders_ft) == 1, "Only one checkpoint subfolder should be found."


@pytest.mark.timeout(512)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_mamba_finetune_runs(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    num_steps = 2
    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_mamba_cmd(tmp_path / "pretrain", max_steps=num_steps, val_check=num_steps)
    stdout_pretrain: str = run_command_in_subprocess(command=command, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" not in stdout_pretrain

    log_dir = tmp_path / "pretrain" / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    tensorboard_dir = log_dir / "dev"

    # Check if logs dir exists
    assert log_dir.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir}"

    assert len(matching_subfolders) == 1, "Only one checkpoint subfolder should be found."
    command_finetune = small_training_mamba_finetune_cmd(
        tmp_path / "finetune", max_steps=num_steps, val_check=num_steps, prev_ckpt=matching_subfolders[0]
    )
    stdout_finetune: str = run_command_in_subprocess(command=command_finetune, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout_finetune

    log_dir_ft = tmp_path / "finetune" / "evo2"
    checkpoints_dir_ft = log_dir_ft / "checkpoints"
    tensorboard_dir_ft = log_dir_ft / "dev"

    # Check if logs dir exists
    assert log_dir_ft.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir_ft.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders_ft = [
        p for p in checkpoints_dir_ft.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders_ft, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir_ft}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir_ft.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir_ft.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir_ft}"

    assert len(matching_subfolders_ft) == 1, "Only one checkpoint subfolder should be found."


@pytest.mark.timeout(512)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_llama_finetune_runs(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory using Llama model.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    num_steps = 2
    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_llama_cmd(tmp_path / "pretrain", max_steps=num_steps, val_check=num_steps)
    stdout_pretrain: str = run_command_in_subprocess(command=command, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" not in stdout_pretrain

    log_dir = tmp_path / "pretrain" / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    tensorboard_dir = log_dir / "dev"

    # Check if logs dir exists
    assert log_dir.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir}"

    assert len(matching_subfolders) == 1, "Only one checkpoint subfolder should be found."
    command_finetune = small_training_llama_finetune_cmd(
        tmp_path / "finetune", max_steps=num_steps, val_check=num_steps, prev_ckpt=matching_subfolders[0]
    )
    stdout_finetune: str = run_command_in_subprocess(command=command_finetune, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout_finetune

    log_dir_ft = tmp_path / "finetune" / "evo2"
    checkpoints_dir_ft = log_dir_ft / "checkpoints"
    tensorboard_dir_ft = log_dir_ft / "dev"

    # Check if logs dir exists
    assert log_dir_ft.exists(), "Logs folder should exist."
    # Check if checkpoints dir exists
    assert checkpoints_dir_ft.exists(), "Checkpoints folder does not exist."

    expected_checkpoint_suffix = f"{num_steps}.0-last"
    matching_subfolders_ft = [
        p for p in checkpoints_dir_ft.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders_ft, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir_ft}."
    )

    # Check if directory with tensorboard logs exists
    assert tensorboard_dir_ft.exists(), "TensorBoard logs folder does not exist."
    # Recursively search for files with tensorboard logger
    event_files = list(tensorboard_dir_ft.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {tensorboard_dir_ft}"

    assert len(matching_subfolders_ft) == 1, "Only one checkpoint subfolder should be found."


@pytest.mark.timeout(256)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_stops(tmp_path):
    """
    This test runs the `train_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.
    """
    max_steps = 500000
    early_stop_steps = 4
    val_check = 2
    additional_args = f"--early-stop-on-step {early_stop_steps}"
    # Expected location of logs and checkpoints
    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"

    assert not log_dir.exists(), "Logs folder shouldn't exist yet."

    # Note: The command assumes that `train_evo2` is in your PATH.
    command = small_training_cmd(tmp_path, max_steps=max_steps, val_check=val_check, additional_args=additional_args)
    command_parts_no_program = shlex.split(command)[1:]
    args = parse_args(args=command_parts_no_program)
    train_stdout, trainer = run_train_with_std_redirect(args)

    assert f"Training epoch 0, iteration 0/{early_stop_steps - 1}" in train_stdout
    # Extract and validate global steps
    global_steps = extract_global_steps_from_log(train_stdout)
    assert global_steps[0] == 0
    assert global_steps[-1] == (early_stop_steps - 1)
    assert trainer.global_step == early_stop_steps
    assert len(global_steps) == early_stop_steps

    expected_checkpoint_suffix = f"{early_stop_steps}.0-last"
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    assert "reduced_train_loss" in trainer.logged_metrics  # validation logging on by default
    assert "TFLOPS_per_GPU" in trainer.logged_metrics  # ensuring that tflops logger can be added
    assert "train_step_timing in s" in trainer.logged_metrics


@pytest.mark.parametrize(
    "additional_args",
    [
        pytest.param("", id="no_fp8"),
        pytest.param(
            "--fp8",
            marks=[
                pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8),
            ],
            id="fp8",
        ),
    ],
)
@pytest.mark.timeout(512)  # Optional: fail if the test takes too long.
@pytest.mark.slow
def test_train_evo2_stop_at_max_steps_and_continue(tmp_path, additional_args):
    max_steps_first_run = 4
    max_steps_second_run = 6
    val_check_interval = 2
    # Expected location of logs and checkpoints
    log_dir = tmp_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"

    command_first_run = small_training_cmd(
        tmp_path, max_steps_first_run, val_check_interval, additional_args=additional_args
    )

    # The first training command to finish at max_steps_first_run
    stdout_first_run = run_command_in_subprocess(command=command_first_run, path=str(tmp_path))

    assert f"Training epoch 0, iteration 0/{max_steps_first_run - 1}" in stdout_first_run
    # Extract and validate global steps
    global_steps_first_run = extract_global_steps_from_log(stdout_first_run)

    assert global_steps_first_run[0] == 0
    assert global_steps_first_run[-1] == max_steps_first_run - 1
    assert len(global_steps_first_run) == max_steps_first_run

    expected_checkpoint_first_run_suffix = f"{max_steps_first_run}.0-last"
    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."
    # Check if any ckpt subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_first_run_suffix in p.name)
    ]
    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_first_run_suffix}' found in {checkpoints_dir}."
    )

    # The second training command to continue from max_steps_first_run and finish at max_steps_second_run
    command_second_run = small_training_cmd(
        tmp_path, max_steps_second_run, val_check_interval, additional_args=additional_args
    )
    stdout_second_run = run_command_in_subprocess(command=command_second_run, path=str(tmp_path))
    global_steps_second_run = extract_global_steps_from_log(stdout_second_run)

    assert global_steps_second_run[0] == max_steps_first_run
    assert global_steps_second_run[-1] == max_steps_second_run - 1
    assert len(global_steps_second_run) == max_steps_second_run - max_steps_first_run

    expected_checkpoint_second_run_suffix = f"{max_steps_second_run}.0-last"
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_second_run_suffix in p.name)
    ]
    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_second_run_suffix}' found in {checkpoints_dir}."
    )


@pytest.fixture(scope="session")
def dataset_config(request):
    """Get dataset directory and training config from command line options or environment variables.

    Users can provide dataset paths via:
    - Command line: pytest --dataset-dir=/path/to/data --training-config=/path/to/config.yaml
    - Environment: DATASET_DIR=/path/to/data TRAINING_CONFIG=/path/to/config.yaml pytest

    If not provided, tests will fall back to --mock-data.
    """
    # Try to get from pytest command line options first
    dataset_dir = request.config.getoption("--dataset-dir", default=None)
    training_config = request.config.getoption("--training-config", default=None)

    # Fall back to environment variables
    if not dataset_dir:
        dataset_dir = os.environ.get("DATASET_DIR")
    if not training_config:
        training_config = os.environ.get("TRAINING_CONFIG")

    return {"dataset_dir": dataset_dir, "training_config": training_config}


@pytest.fixture(scope="session")
def initial_checkpoint():
    """Load the initial checkpoint for distributed training tests."""
    from bionemo.core.data.load import load

    return load("evo2/7b-8k:1.0")


@pytest.fixture(scope="session")
def base_checkpoint(tmp_path_factory, initial_checkpoint, dataset_config):
    """Create a base checkpoint by training one step with no parallelism.

    This fixture is session-scoped, so it creates the checkpoint once and reuses it
    across all parametrized test cases, significantly improving test performance.
    """
    num_steps = 1
    tmp_path = tmp_path_factory.mktemp("base_checkpoint_session")
    base_path = tmp_path / "base_training"

    # Create command with the initial checkpoint and dataset (if provided)
    cmd = distributed_training_cmd(
        path=base_path,
        max_steps=num_steps,
        val_check=num_steps,
        num_devices=1,
        dp=1,
        tp=1,
        cp=1,
        pp=1,
        dataset_dir=dataset_config["dataset_dir"],
        training_config=dataset_config["training_config"],
        additional_args=f"--ckpt-dir {initial_checkpoint}",
    )

    # Run training
    stdout = run_command_in_subprocess(command=cmd, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout

    # Find the resulting checkpoint
    log_dir = base_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    # Lightning uses 0-indexed step counting, so after max_steps=1, we're at step 0
    expected_checkpoint_suffix = "step=0"

    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert len(matching_subfolders) == 1, "Expected exactly one checkpoint subfolder"
    return matching_subfolders[0]


@pytest.mark.parametrize(
    "dp,cp,tp,pp",
    [
        pytest.param(2, 1, 1, 1, id="data_parallel"),
        pytest.param(1, 2, 1, 1, id="context_parallel"),
        pytest.param(1, 1, 2, 1, id="tensor_parallel"),
        pytest.param(1, 1, 1, 2, id="pipeline_parallel"),
    ],
)
@pytest.mark.timeout(900)
@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires at least 2 GPUs")
def test_distributed_training_gradient_equivalence(
    tmp_path, initial_checkpoint, base_checkpoint, dataset_config, dp, cp, tp, pp
):
    """Test that gradients are equivalent across different distributed training strategies."""
    # NOTE: Megatron Core is changing its distributed checkpoint format soon. This test needs to be updated after release 0.14.
    num_steps = 1

    # Calculate total devices needed
    num_devices = dp * cp * tp * pp
    assert num_devices == 2, (
        f"Test is designed for 2 GPUs but got {num_devices} for dp={dp}, cp={cp}, tp={tp}, pp={pp}"
    )

    # Create parallel training checkpoint
    parallel_path = tmp_path / f"parallel_dp{dp}_cp{cp}_tp{tp}_pp{pp}"

    cmd = distributed_training_cmd(
        path=parallel_path,
        max_steps=num_steps,
        val_check=num_steps,
        num_devices=num_devices,
        dp=dp,
        tp=tp,
        cp=cp,
        pp=pp,
        dataset_dir=dataset_config["dataset_dir"],
        training_config=dataset_config["training_config"],
        additional_args=f"--ckpt-dir {initial_checkpoint}",
    )

    # Run distributed training
    stdout = run_command_in_subprocess(command=cmd, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout

    # Find the resulting checkpoint
    log_dir = parallel_path / "evo2"
    checkpoints_dir = log_dir / "checkpoints"
    # Lightning uses 0-indexed step counting, so after max_steps=1, we're at step 0
    expected_checkpoint_suffix = "step=0"

    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert len(matching_subfolders) == 1, "Expected exactly one checkpoint subfolder"
    parallel_checkpoint = matching_subfolders[0]

    # Compare gradients/optimizer states between base and parallel distributed training
    print(f"Base checkpoint: {base_checkpoint}")
    print(f"Parallel checkpoint (dp={dp}, cp={cp}, tp={tp}, pp={pp}): {parallel_checkpoint}")

    # Ensure both checkpoints exist before comparison
    assert base_checkpoint.exists(), "Base checkpoint should exist"
    assert parallel_checkpoint.exists(), "Parallel checkpoint should exist"

    # Use the custom gradient comparison logic to verify optimizer states match
    # This implements theorem 5.3 of https://www.arxiv.org/pdf/2506.09280 for gradient equivalence
    checkpoint_dirs = [str(base_checkpoint / "weights"), str(parallel_checkpoint / "weights")]
    assert_optimizer_states_match(checkpoint_dirs)
