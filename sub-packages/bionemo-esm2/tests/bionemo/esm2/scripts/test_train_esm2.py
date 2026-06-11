# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import io
import os
import shlex
import sqlite3
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict

import pandas as pd
import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.esm2.scripts.train_esm2 import get_parser, main
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.datamodule_utils import parse_kwargs_to_arglist
from bionemo.testing.lightning import extract_global_steps_from_log
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.subprocess_utils import run_command_in_subprocess


def run_train_with_std_redirect(
    train_database_path,
    valid_database_path,
    parquet_train_val_inputs,
    result_dir,
    num_steps,
    val_check_interval,
    create_checkpoint_callback,
    create_tensorboard_logger,
    create_tflops_callback,
    resume_if_exists,
    wandb_project,
) -> str:
    """
    Run a function with output capture.
    """
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
        train_small_esm2(
            train_database_path,
            valid_database_path,
            parquet_train_val_inputs,
            result_dir,
            num_steps,
            val_check_interval,
            create_checkpoint_callback,
            create_tensorboard_logger,
            create_tflops_callback,
            resume_if_exists,
            wandb_project,
        )

    train_stdout = stdout_buf.getvalue()
    train_stderr = stderr_buf.getvalue()
    print("Captured STDOUT:\n", train_stdout)
    print("Captured STDERR:\n", train_stderr)
    return train_stdout


def train_small_esm2_args(
    train_database_path,
    valid_database_path,
    parquet_train_val_inputs,
    result_dir,
    num_steps,
    val_check_interval,
    create_checkpoint_callback,
    create_tensorboard_logger,
    create_tflops_callback,
    resume_if_exists,
    wandb_project=None,
    limit_val_batches=1,
    experiment_name="esm2",
) -> dict:
    train_cluster_path, valid_cluster_path = parquet_train_val_inputs
    # Extract arguments from the given function call
    args_dict = {
        "train_cluster_path": train_cluster_path,
        "train_database_path": train_database_path,
        "valid_cluster_path": valid_cluster_path,
        "valid_database_path": valid_database_path,
        "num_nodes": 1,
        "devices": 1,
        "min_seq_length": 128,
        "max_seq_length": 128,
        "result_dir": result_dir,
        "experiment_name": experiment_name,
        "wandb_project": wandb_project,
        "wandb_offline": True,
        "wandb_anonymous": True,
        "num_steps": num_steps,
        "warmup_steps": 1,
        "limit_val_batches": limit_val_batches,
        "val_check_interval": val_check_interval,
        "log_every_n_steps": 1,
        "num_dataset_workers": 1,
        "lr": 1e-4,
        "micro_batch_size": 2,
        "accumulate_grad_batches": 1,
        "precision": "bf16-mixed",
        "resume_if_exists": resume_if_exists,
        "create_tensorboard_logger": create_tensorboard_logger,
        "create_tflops_callback": create_tflops_callback,
        "create_checkpoint_callback": create_checkpoint_callback,
        "num_layers": 2,
        "num_attention_heads": 2,
        "hidden_size": 4,
        "ffn_hidden_size": 4 * 4,
        "scheduler_num_steps": None,
        "biobert_spec_option": BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec.value,
    }
    return args_dict


def train_small_esm2_cmd(
    train_database_path,
    valid_database_path,
    parquet_train_val_inputs,
    result_dir,
    num_steps,
    val_check_interval,
    create_checkpoint_callback,
    create_tensorboard_logger,
    create_tflops_callback,
    resume_if_exists,
    wandb_project=None,
    limit_val_batches=1,
    experiment_name="esm2",
) -> str:
    args = train_small_esm2_args(
        train_database_path,
        valid_database_path,
        parquet_train_val_inputs,
        result_dir,
        num_steps,
        val_check_interval,
        create_checkpoint_callback,
        create_tensorboard_logger,
        create_tflops_callback,
        resume_if_exists,
        wandb_project,
        limit_val_batches,
        experiment_name,
    )

    def get_command_line_args(arg_name, arg_value) -> str:
        if arg_name == "create_checkpoint_callback":
            if not arg_value:
                return "--disable-checkpointing"
            else:
                return ""
        if arg_name in ["wandb_project", "scheduler_num_steps"] and arg_value is None:
            return ""

        if arg_name in [
            "wandb_offline",
            "wandb_anonymous",
            "create_tensorboard_logger",
            "resume_if_exists",
            "create_tflops_callback",
        ]:
            if arg_value:
                return f"--{arg_name.replace('_', '-')}"
            else:
                return ""

        if arg_name == "devices":
            return f"--num-gpus={arg_value}"

        arg_str = f"--{arg_name.replace('_', '-')}={arg_value}"
        return arg_str

    cmd = f"train_esm2 {' '.join(get_command_line_args(arg_name, arg_value) for arg_name, arg_value in args.items())}"
    return cmd


def train_small_esm2(
    train_database_path,
    valid_database_path,
    parquet_train_val_inputs,
    result_dir,
    num_steps,
    val_check_interval,
    create_checkpoint_callback,
    create_tensorboard_logger,
    create_tflops_callback,
    resume_if_exists,
    wandb_project=None,
    limit_val_batches=1,
    experiment_name="esm2",
):
    args = train_small_esm2_args(
        train_database_path,
        valid_database_path,
        parquet_train_val_inputs,
        result_dir,
        num_steps,
        val_check_interval,
        create_checkpoint_callback,
        create_tensorboard_logger,
        create_tflops_callback,
        resume_if_exists,
        wandb_project,
        limit_val_batches,
        experiment_name,
    )
    with distributed_model_parallel_state():
        trainer = main(**args)
    return trainer


@pytest.fixture
def dummy_protein_dataset(tmp_path):
    """Create a mock protein dataset."""
    db_file = tmp_path / "protein_dataset.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE protein (
            id TEXT PRIMARY KEY,
            sequence TEXT
        )
    """
    )

    proteins = [
        ("UniRef90_A", "ACDEFGHIKLMNPQRSTVWY"),
        ("UniRef90_B", "DEFGHIKLMNPQRSTVWYAC"),
        ("UniRef90_C", "MGHIKLMNPQRSTVWYACDE"),
        ("UniRef50_A", "MKTVRQERLKSIVRI"),
        ("UniRef50_B", "MRILERSKEPVSGAQLA"),
    ]
    cursor.executemany("INSERT INTO protein VALUES (?, ?)", proteins)

    conn.commit()
    conn.close()

    return db_file


@pytest.fixture
def dummy_parquet_train_val_inputs(tmp_path):
    """Create a mock protein train and val cluster parquet."""
    train_cluster_path = tmp_path / "train_clusters.parquet"
    train_clusters = pd.DataFrame(
        {
            "ur90_id": [["UniRef90_A"], ["UniRef90_B", "UniRef90_C"]],
        }
    )
    train_clusters.to_parquet(train_cluster_path)

    valid_cluster_path = tmp_path / "valid_clusters.parquet"
    valid_clusters = pd.DataFrame(
        {
            "ur50_id": ["UniRef50_A", "UniRef50_B", "UniRef50_A", "UniRef50_B"],  # 2 IDs more than confest
        }
    )
    valid_clusters.to_parquet(valid_cluster_path)
    return train_cluster_path, valid_cluster_path


@pytest.mark.parametrize("create_checkpoint_callback", [True, False])
def test_main_runs(tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs, create_checkpoint_callback):
    val_check_interval = 2
    num_steps = 4
    experiment_name = "esm2"
    trainer = train_small_esm2(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=num_steps,
        val_check_interval=val_check_interval,
        create_checkpoint_callback=create_checkpoint_callback,
        create_tensorboard_logger=True,
        create_tflops_callback=True,
        resume_if_exists=False,
        wandb_project=None,
        experiment_name=experiment_name,
    )
    experiment_dir = tmp_path / experiment_name
    assert experiment_dir.exists(), "Could not find experiment directory."
    assert experiment_dir.is_dir(), "Experiment directory is supposed to be a directory."
    log_dir = experiment_dir / "dev"
    assert log_dir.exists(), "Directory with logs does not exist"

    children = list(experiment_dir.iterdir())
    # ["checkpoints", "dev"] since wandb is disabled. Offline mode was causing troubles
    expected_children = 2 if create_checkpoint_callback else 1
    assert len(children) == expected_children, (
        f"Expected {expected_children} child in the experiment directory, found {children}."
    )

    if create_checkpoint_callback:
        checkpoints_dir = experiment_dir / "checkpoints"
        assert checkpoints_dir.exists(), "Checkpoints directory does not exist."
        # check if correct checkpoint was saved
        expected_checkpoint_suffix = f"step={num_steps - 1}"
        matching_subfolders = [
            p
            for p in checkpoints_dir.iterdir()
            if p.is_dir() and (expected_checkpoint_suffix in p.name and "last" in p.name)
        ]
        assert matching_subfolders, (
            f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
        )

    assert (log_dir / "nemo_log_globalrank-0_localrank-0.txt").is_file(), "Could not find experiment log."

    # Recursively search for files from tensorboard logger
    event_files = list(log_dir.rglob("events.out.tfevents*"))
    assert event_files, f"No TensorBoard event files found under {log_dir}"
    assert "val_ppl" in trainer.logged_metrics  # validation logging on by default
    assert "TFLOPS_per_GPU" in trainer.logged_metrics  # ensuring that tflops logger can be added
    assert "train_step_timing in s" in trainer.logged_metrics


@pytest.mark.slow
def test_main_stop_at_num_steps_and_continue(tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs):
    max_steps_first_run = 4
    max_steps_second_run = 6
    val_check_interval = 2
    # Expected location of logs and checkpoints
    experiment_name = "esm2"
    log_dir = tmp_path / experiment_name
    checkpoints_dir = log_dir / "checkpoints"

    command_first_run = train_small_esm2_cmd(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=max_steps_first_run,
        val_check_interval=val_check_interval,
        create_checkpoint_callback=True,
        create_tensorboard_logger=True,
        create_tflops_callback=True,
        resume_if_exists=True,
        wandb_project=None,
        experiment_name=experiment_name,
    )

    # The first training command to finish at max_steps_first_run
    # stdout_first_run, stderr_first_run, returncode_first_run
    #
    stdout_first_run = run_command_in_subprocess(command=command_first_run, path=str(tmp_path))

    assert f"Training epoch 0, iteration 0/{max_steps_first_run - 1}" in stdout_first_run
    # Extract and validate global steps
    global_steps_first_run = extract_global_steps_from_log(stdout_first_run)

    assert global_steps_first_run[0] == 0
    assert global_steps_first_run[-1] == max_steps_first_run - 1
    assert len(global_steps_first_run) == max_steps_first_run

    # Check if checkpoints dir exists
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."
    # Check if any ckpt subfolder ends with the expected suffix
    expected_checkpoint_first_run_suffix = f"step={max_steps_first_run - 1}"
    matching_subfolders = [
        p
        for p in checkpoints_dir.iterdir()
        if p.is_dir() and (expected_checkpoint_first_run_suffix in p.name and "last" in p.name)
    ]
    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_first_run_suffix}' found in {checkpoints_dir}."
    )

    # The second training command to continue from max_steps_first_run and finish at max_steps_second_run
    command_second_run = train_small_esm2_cmd(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=max_steps_second_run,
        val_check_interval=val_check_interval,
        create_checkpoint_callback=True,
        create_tensorboard_logger=False,
        create_tflops_callback=False,
        resume_if_exists=True,
        wandb_project=None,
        experiment_name=experiment_name,
    )
    stdout_second_run = run_command_in_subprocess(command=command_second_run, path=str(tmp_path))

    # Verify the model can continue training from the checkpoint without errors

    global_steps_second_run = extract_global_steps_from_log(stdout_second_run)

    assert global_steps_second_run[0] == max_steps_first_run
    assert global_steps_second_run[-1] == max_steps_second_run - 1
    assert len(global_steps_second_run) == max_steps_second_run - max_steps_first_run

    expected_checkpoint_second_run_suffix = f"step={max_steps_second_run - 1}"
    matching_subfolders = [
        p
        for p in checkpoints_dir.iterdir()
        if p.is_dir() and (expected_checkpoint_second_run_suffix in p.name and "last" in p.name)
    ]
    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_second_run_suffix}' found in {checkpoints_dir}."
    )


@pytest.mark.parametrize("limit_val_batches", [0.0, 1.0, 4, None])
def test_val_dataloader_in_main_runs_with_limit_val_batches(
    tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs, limit_val_batches
):
    # TODO: pydantic.
    """Ensures doesn't run out of validation samples whenever updating limit_val_batches logic.

    Args:
        monkeypatch: (MonkeyPatch): Monkey patch for environment variables.
        tmpdir (str): Temporary directory.
        dummy_protein_dataset (str): Path to dummy protein dataset.
        dummy_parquet_train_val_inputs (tuple[str, str]): Tuple of dummy protein train and val cluster parquet paths.
        limit_val_batches (Union[int, float, None]): Limit validation batches. None implies 1.0 as in PTL.
    """
    train_small_esm2(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=4,
        val_check_interval=2,
        create_checkpoint_callback=False,
        create_tensorboard_logger=False,
        create_tflops_callback=False,
        resume_if_exists=False,
        wandb_project=None,
        limit_val_batches=limit_val_batches,
    )


@pytest.mark.skip("duplicate with argparse, model and data unittests")
def test_pretrain_cli(tmp_path, dummy_protein_dataset, dummy_parquet_train_val_inputs):
    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    open_port = find_free_network_port()
    # NOTE: if you need to change the following command, please update the README.md example.
    cmd_str = train_small_esm2_cmd(
        train_database_path=dummy_protein_dataset,
        valid_database_path=dummy_protein_dataset,
        parquet_train_val_inputs=dummy_parquet_train_val_inputs,
        result_dir=tmp_path,
        num_steps=4,
        val_check_interval=2,
        create_checkpoint_callback=True,
        create_tensorboard_logger=False,
        create_tflops_callback=False,
        resume_if_exists=False,
        wandb_project=None,
        experiment_name="esm2",
    )
    experiment_dir = tmp_path / "esm2"
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        env=env,
        capture_output=True,
    )
    assert result.returncode == 0, f"Pretrain script failed: {cmd_str}"
    assert experiment_dir.exists(), "Could not find the experiment directory."


@pytest.fixture(scope="function")
def required_args_reference() -> Dict[str, str]:
    """
    This fixture provides a dictionary of required arguments for the pretraining script.

    It includes the following keys:
    - train_cluster_path: The path to the training cluster parquet file.
    - train_database_path: The path to the training database file.
    - valid_cluster_path: The path to the validation cluster parquet file.
    - valid_database_path: The path to the validation database file.

    The values for these keys are placeholders and should be replaced with actual file paths.

    Returns:
        A dictionary with the required arguments for the pretraining script.
    """
    return {
        "train_cluster_path": "path/to/train_cluster.parquet",
        "train_database_path": "path/to/train.db",
        "valid_cluster_path": "path/to/valid_cluster.parquet",
        "valid_database_path": "path/to/valid.db",
    }


# TODO(@sichu) add test on dataset/datamodule on invalid path
def test_required_train_cluster_path(required_args_reference):
    """
    Test train_cluster_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("train_cluster_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


def test_required_train_database_path(required_args_reference):
    """
    Test train_database_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("train_database_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


def test_required_valid_cluster_path(required_args_reference):
    """
    Test valid_cluster_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("valid_cluster_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


def test_required_valid_database_path(required_args_reference):
    """
    Test valid_database_path is required.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference.pop("valid_database_path")
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(arglist)


#### test expected behavior on parser ####
@pytest.mark.parametrize("limit_val_batches", [0.1, 0.5, 1.0])
def test_limit_val_batches_is_float(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as a float.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
        limit_val_batches (float): The value of limit_val_batches.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    parser.parse_args(arglist)


@pytest.mark.parametrize("limit_val_batches", ["0.1", "0.5", "1.0"])
def test_limit_val_batches_is_float_string(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as a string of float.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
        limit_val_batches (float): The value of limit_val_batches.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    parser.parse_args(arglist)


@pytest.mark.parametrize("limit_val_batches", [None, "None"])
def test_limit_val_batches_is_none(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as none.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    args = parser.parse_args(arglist)
    assert args.limit_val_batches is None


@pytest.mark.parametrize("limit_val_batches", [1, 2])
def test_limit_val_batches_is_int(required_args_reference, limit_val_batches):
    """
    Test whether limit_val_batches can be parsed as integer.

    Args:
        required_args_reference (Dict[str, str]): A dictionary with the required arguments for the pretraining script.
        limit_val_batches (int): The value of limit_val_batches.
    """
    required_args_reference["limit_val_batches"] = limit_val_batches
    arglist = parse_kwargs_to_arglist(required_args_reference)
    parser = get_parser()
    parser.parse_args(arglist)
