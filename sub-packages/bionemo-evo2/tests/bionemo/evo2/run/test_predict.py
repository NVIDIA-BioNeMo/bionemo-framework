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

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import lightning as pl
import pytest
import torch
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.core.data.load import load
from bionemo.evo2.run.predict import predict
from bionemo.llm.lightning import batch_collator
from bionemo.testing.data.fasta import ALU_SEQUENCE, create_fasta_file
from bionemo.testing.megatron_parallel_state_utils import clean_parallel_state_context
from bionemo.testing.subprocess_utils import run_command_in_subprocess
from bionemo.testing.torch import check_fp8_support

from .common import predict_cmd, small_training_finetune_cmd


def is_a6000_gpu() -> bool:
    # Check if any of the visible GPUs is an A6000
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        if "A6000" in device_name:
            return True
    return False


@pytest.fixture(scope="module")
def checkpoint_1b_8k_bf16_path() -> Path:
    try:
        checkpoint_path = load("evo2/1b-8k-bf16:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    yield checkpoint_path


@pytest.fixture(scope="module")
def checkpoint_7b_1m_path() -> Path:
    try:
        checkpoint_path = load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    yield checkpoint_path


def test_predict_does_not_instantiate_optimizer(tmp_path: Path, checkpoint_1b_8k_bf16_path: Path):
    output_dir = tmp_path / "test_output"
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        1,
        sequence_lengths=[512],
        repeating_dna_pattern=ALU_SEQUENCE,
    )

    class AssertNoOptimizerCallback(pl.Callback):
        def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            assert not trainer.optimizers, (
                f"Optimizer should not be instantiated for prediction, got {trainer.optimizers}"
            )
            trainer_model_opt = getattr(trainer.model, "optim", None)
            assert trainer_model_opt is None or not trainer_model_opt.state_dict(), (
                f"Model optimizer found, got {trainer_model_opt} with state {trainer_model_opt.state_dict()}"
            )

    with clean_parallel_state_context():
        predict(
            fasta_path=fasta_file_path,
            ckpt_dir=str(checkpoint_1b_8k_bf16_path),
            output_dir=output_dir,
            tensor_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            num_nodes=1,
            devices=1,
            model_size="1b",
            ckpt_format="torch_dist",
            fp8=False,
            full_fp8=False,
            work_dir=tmp_path,
            micro_batch_size=1,
            output_log_prob_seqs=True,
            log_prob_collapse_option="mean",
            write_interval="epoch",
            prepend_bos=False,
            no_sequence_parallel=False,
            hybrid_override_pattern="SDH*",
            num_layers=4,
            seq_len_interpolation_factor=None,
            files_per_subdir=None,
            lora_checkpoint_path=None,
            extra_callbacks=[
                AssertNoOptimizerCallback(),
            ],  # use this for making testing the loop easier.
        )


@pytest.mark.parametrize(
    "ddp,pp,wi",
    [
        pytest.param(1, 1, "epoch", id="ddp=1,pp=1,wi=epoch"),
        pytest.param(2, 1, "epoch", id="ddp=2,pp=1,wi=epoch"),
        pytest.param(2, 1, "batch", id="ddp=2,pp=1,wi=batch"),
        pytest.param(
            1,
            2,
            "epoch",
            id="ddp=1,pp=2,wi=epoch",
            marks=pytest.mark.skip("Pipeline parallelism test currently hangs."),
        ),
    ],
)
def test_predict_evo2_runs(
    tmp_path,
    ddp: int,
    pp: int,
    wi: str,
    checkpoint_1b_8k_bf16_path: Path,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] = [3149, 3140, 1024, 3148, 3147],
):
    """
    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    Since it's the full output this does not support CP, so we only test with TP=1. We also want coverage of the
        case where the sequence lengths are different and not necessarily divisible by CP.
    """
    world_size = ddp * pp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is less than the number of GPUs {torch.cuda.device_count()}")
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )
    # Create a mock data directory.
    # a local copy of the environment
    env = dict(**os.environ)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    # Build the command string.
    # Note: The command assumes that `train_evo2` is in your PATH.
    output_dir = tmp_path / "test_output"
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --no-python "
        f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} --model-size 1b "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--pipeline-model-parallel-size {pp} --num-nodes 1 --devices {world_size}"
    )

    # Run the command in a subshell, using the temporary directory as the current working directory.
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    result = subprocess.run(
        command,
        shell=True,  # Use the shell to interpret wildcards (e.g. SDH*)
        cwd=tmp_path,  # Run in the temporary directory
        capture_output=True,  # Capture stdout and stderr for debugging
        env=env,  # Pass in the env where we override the master port.
        text=True,  # Decode output as text
    )

    # For debugging purposes, print the output if the test fails.
    if result.returncode != 0:
        sys.stderr.write("STDOUT:\n" + result.stdout + "\n")
        sys.stderr.write("STDERR:\n" + result.stderr + "\n")

    # Assert that the command completed successfully.
    assert result.returncode == 0, "train_evo2 command failed."

    # Assert that the output directory was created.
    pred_files = glob.glob(os.path.join(output_dir, "predictions__rank_*.pt"))
    if wi == "batch":
        assert len(pred_files) == 2, f"Expected 2 prediction file (for this test), got {len(pred_files)}"
    else:
        assert len(pred_files) == ddp, f"Expected {ddp} prediction file (for this test), got {len(pred_files)}"
    with open(output_dir / "seq_idx_map.json", "r") as f:
        seq_idx_map = json.load(
            f
        )  # This gives us the mapping from the sequence names to the indices in the predictions.
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim_key_defaults={"token_logits": 0},
        seq_dim_key_defaults={"token_logits": 1},
    )
    assert isinstance(preds, dict)
    assert "token_logits" in preds
    assert "pad_mask" in preds
    assert "seq_idx" in preds

    assert len(preds["token_logits"]) == len(preds["pad_mask"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences
    for original_idx, pad_mask, token_logits in zip(preds["seq_idx"], preds["pad_mask"], preds["token_logits"]):
        # seq_idx is not sorted necessarily, so use the saved "seq_idx" to determine the original order.
        expected_len = target_sequence_lengths[original_idx]
        assert pad_mask.sum() == expected_len
        assert token_logits.shape == (max(target_sequence_lengths), 512)


@pytest.fixture(scope="module")
def baseline_predictions_7b_1m_results(
    checkpoint_7b_1m_path: Path,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] = [2048, 2048, 2048, 2048, 2048],
) -> dict[int, float]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        fasta_file_path = tmp_path / "test.fasta"
        create_fasta_file(
            fasta_file_path,
            num_sequences,
            sequence_lengths=target_sequence_lengths,
            repeating_dna_pattern=ALU_SEQUENCE,
        )
        output_dir = tmp_path / "test_output"
        command = (
            f"torchrun --nproc_per_node 1 --nnodes 1 --no-python "
            f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {checkpoint_7b_1m_path} "
            f"--num-layers 4 --hybrid-override-pattern SDH* "  # subset of layers for testing
            # FIXME changing batch size from 3 to 1 required dropping rel=1e-6 to rel=1e-3
            #  even when model parallelism is not used. This should be investigated.
            f"--micro-batch-size 3 "
            f"--output-dir {output_dir} --model-size 7b_arc_longcontext "
            f"--num-nodes 1 --write-interval epoch "
            "--output-log-prob-seqs --log-prob-collapse-option sum"
        )
        # Create a mock data directory.
        # a local copy of the environment
        env = dict(**os.environ)
        open_port = find_free_network_port()
        env["MASTER_PORT"] = str(open_port)
        result = subprocess.run(
            command,
            shell=True,  # Use the shell to interpret wildcards (e.g. SDH*)
            cwd=tmp_path,  # Run in the temporary directory
            capture_output=True,  # Capture stdout and stderr for debugging
            env=env,  # Pass in the env where we override the master port.
            text=True,  # Decode output as text
        )
        assert result.returncode == 0, "predict_evo2 command failed."
        # Assert that the output directory was created.
        pred_files = glob.glob(os.path.join(output_dir, "predictions__rank_*.pt"))
        preds = [torch.load(pf) for pf in pred_files]
        preds = batch_collator(
            [p for p in preds if p is not None],
        )
        yield dict(zip([i.item() for i in preds["seq_idx"]], [p.item() for p in preds["log_probs_seqs"]]))


@pytest.mark.parametrize(
    "ddp,cp,pp,tp,fp8,wi",
    [
        pytest.param(1, 1, 1, 1, False, "epoch", id="ddp=1,cp=1,pp=1,tp=1,fp8=False,wi=epoch"),
        pytest.param(2, 1, 1, 1, False, "epoch", id="ddp=2,cp=1,pp=1,tp=1,fp8=False,wi=epoch"),
        pytest.param(
            2, 1, 1, 1, False, "batch", id="ddp=2,cp=1,pp=1,tp=1,fp8=False,wi=batch"
        ),  # simulate a large prediction run with dp parallelism
        pytest.param(1, 2, 1, 1, False, "epoch", id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=epoch"),
        pytest.param(1, 2, 1, 1, False, "batch", id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=batch"),
        pytest.param(
            1,
            1,
            2,
            1,
            False,
            "epoch",
            id="ddp=1,cp=1,pp=2,tp=1,fp8=False,wi=epoch",
            marks=pytest.mark.skip("Pipeline parallelism test currently hangs."),
        ),
        pytest.param(
            1, 1, 1, 2, True, "epoch", id="ddp=1,cp=1,pp=1,tp=2,fp8=True,wi=epoch"
        ),  # Cover case where FP8 was not supported with TP=2
        pytest.param(1, 1, 1, 2, False, "epoch", id="ddp=1,cp=1,pp=1,tp=2,fp8=False,wi=epoch"),
    ],
    ids=lambda x: f"ddp={x[0]},cp={x[1]},pp={x[2]},tp={x[3]},fp8={x[4]},wi={x[5]}",
)
def test_predict_evo2_equivalent_with_log_probs(
    tmp_path,
    ddp: int,
    cp: int,
    pp: int,
    tp: int,
    fp8: bool,
    wi: str,
    checkpoint_7b_1m_path: Path,
    baseline_predictions_7b_1m_results: dict[int, float],
    num_sequences: int = 5,
    target_sequence_lengths: list[int] = [2048, 2048, 2048, 2048, 2048],
):
    """
    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    For this test, we want coverage of CP, so we make sure sequence lengths are all the same and divisible by CP.

    The other thing this test does is check that the log probabilities are equivalent to the baseline predictions
     without model parallelism.
    """

    world_size = ddp * cp * pp * tp
    mp_size = cp * pp * tp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is less than the number of GPUs {torch.cuda.device_count()}")
    is_fp8_supported, _, _ = check_fp8_support(torch.cuda.current_device())
    if not is_fp8_supported and fp8:
        pytest.skip("FP8 is not supported on this GPU.")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )
    # Create a mock data directory.
    # a local copy of the environment
    env = dict(**os.environ)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    fp8_option = "--fp8" if fp8 else ""
    # Build the command string.
    # Note: The command assumes that `train_evo2` is in your PATH.
    output_dir = tmp_path / "test_output"
    command = (
        f"torchrun --nproc_per_node {world_size} --nnodes 1 --no-python "
        f"predict_evo2 --fasta {fasta_file_path} --ckpt-dir {checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--num-layers 4 --hybrid-override-pattern SDH* "  # subset of layers for testing
        f"--output-dir {output_dir} --model-size 7b_arc_longcontext --tensor-parallel-size {tp} {fp8_option} "
        f"--pipeline-model-parallel-size {pp} --context-parallel-size {cp} --num-nodes 1 --devices {world_size} "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    # Run the command in a subshell, using the temporary directory as the current working directory.
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    result = subprocess.run(
        command,
        shell=True,  # Use the shell to interpret wildcards (e.g. SDH*)
        cwd=tmp_path,  # Run in the temporary directory
        capture_output=True,  # Capture stdout and stderr for debugging
        env=env,  # Pass in the env where we override the master port.
        text=True,  # Decode output as text
    )

    # For debugging purposes, print the output if the test fails.
    if result.returncode != 0:
        sys.stderr.write("STDOUT:\n" + result.stdout + "\n")
        sys.stderr.write("STDERR:\n" + result.stderr + "\n")

    # Assert that the command completed successfully.
    assert result.returncode == 0, "train_evo2 command failed."

    # Assert that the output directory was created.
    pred_files = glob.glob(os.path.join(output_dir, "predictions__rank_*.pt"))
    if wi == "batch":
        assert len(pred_files) == 2, f"Expected 2 prediction file (for this test), got {len(pred_files)}"
    else:
        assert len(pred_files) == ddp, f"Expected {ddp} prediction file (for this test), got {len(pred_files)}"
    with open(output_dir / "seq_idx_map.json", "r") as f:
        seq_idx_map = json.load(
            f
        )  # This gives us the mapping from the sequence names to the indices in the predictions.
    preds = [torch.load(pf) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
    )
    assert isinstance(preds, dict)
    assert "log_probs_seqs" in preds
    assert "seq_idx" in preds
    assert len(preds["log_probs_seqs"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences
    for original_idx, log_probs in zip(preds["seq_idx"], preds["log_probs_seqs"]):
        if mp_size > 1 and not fp8:
            # FIXME changing batch size so it doesn't match also required dropping rel=1e-6 to rel=1e-3.
            #  This should be investigated.
            rel = 1e-3
        elif fp8:
            # NOTE: This is hand-tuned on a b300 to pass for now as of 9/10/2025.
            rel = 1e-2
        else:
            rel = 1e-6
        assert log_probs.item() == pytest.approx(baseline_predictions_7b_1m_results[original_idx.item()], rel=rel)


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_different_results_with_without_peft(tmp_path):
    try:
        base_model_checkpoint_path = load("evo2/1b-8k:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    num_steps = 2

    result_dir = tmp_path / "lora_finetune"

    # Note: The command assumes that `train_evo2` is in your PATH.
    command_finetune = small_training_finetune_cmd(
        result_dir,
        max_steps=num_steps,
        val_check=num_steps,
        prev_ckpt=base_model_checkpoint_path,
        create_tflops_callback=False,
        additional_args="--lora-finetune",
    )
    stdout_finetune: str = run_command_in_subprocess(command=command_finetune, path=str(tmp_path))
    assert "Restoring model weights from RestoreConfig(path='" in stdout_finetune
    assert "Loading adapters from" not in stdout_finetune

    # Check if checkpoints dir exists
    checkpoints_dir = result_dir / "evo2" / "checkpoints"
    assert checkpoints_dir.exists(), "Checkpoints folder does not exist."

    # Create a sample FASTA file to run predictions
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_file_path, 3, sequence_lengths=[32, 65, 129], repeating_dna_pattern=ALU_SEQUENCE)

    result_dir_original = tmp_path / "results_original"
    cmd_predict = predict_cmd(base_model_checkpoint_path, result_dir_original, fasta_file_path)
    stdout_predict: str = run_command_in_subprocess(command=cmd_predict, path=str(tmp_path))

    # Assert that the output directory was created.
    pred_files_original = glob.glob(str(result_dir_original / "predictions__rank_*.pt"))
    assert len(pred_files_original) == 1, f"Expected 1 prediction file (for this test), got {len(pred_files_original)}"

    # Find the checkpoint dir generated by finetuning
    expected_checkpoint_suffix = f"{num_steps}.0-last"
    # Check if any subfolder ends with the expected suffix
    matching_subfolders = [
        p for p in checkpoints_dir.iterdir() if p.is_dir() and (expected_checkpoint_suffix in p.name)
    ]

    assert matching_subfolders, (
        f"No checkpoint subfolder ending with '{expected_checkpoint_suffix}' found in {checkpoints_dir}."
    )

    result_dir_peft = tmp_path / "results_peft"
    additional_args = f"--lora-checkpoint-path {matching_subfolders[0]}"
    cmd_predict = predict_cmd(base_model_checkpoint_path, result_dir_peft, fasta_file_path, additional_args)
    stdout_predict: str = run_command_in_subprocess(command=cmd_predict, path=str(tmp_path))
    assert "Loading adapters from" in stdout_predict

    pred_files_peft = glob.glob(str(result_dir_peft / "predictions__rank_*.pt"))
    assert len(pred_files_peft) == 1, f"Expected 1 prediction file (for this test), got {len(pred_files_peft)}"

    results_original = torch.load(f"{result_dir_original}/predictions__rank_0__dp_rank_0.pt")
    results_peft = torch.load(f"{result_dir_peft}/predictions__rank_0__dp_rank_0.pt")

    seq_idx_original = results_original["seq_idx"]
    seq_idx_peft = results_peft["seq_idx"]
    assert torch.equal(seq_idx_original, seq_idx_peft), f"Tensors differ: {seq_idx_original} vs {seq_idx_peft}"

    logits_original = results_original["token_logits"]
    logits_peft = results_peft["token_logits"]
    assert (logits_original != logits_peft).any()
    assert logits_original.shape == logits_peft.shape, (
        f"Shapes don't match: {logits_original.shape} vs {logits_peft.shape}"
    )
