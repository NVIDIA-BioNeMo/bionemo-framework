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

"""Tests for Eden prediction (inference) workflow using Megatron Bridge."""

import copy
import glob
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.eden.data.test_utils.create_fasta_file import ALU_SEQUENCE, create_fasta_file
from bionemo.eden.run.predict import batch_collator

from ..utils import find_free_network_port, is_a6000_gpu


# Do this at collection time before we run any tests.
PRETEST_ENV = copy.deepcopy(os.environ)


# =============================================================================
# Eden (Llama) prediction tests
# =============================================================================


@pytest.fixture(scope="module")
def mbridge_eden_checkpoint_path(mbridge_eden_checkpoint) -> Path:
    """Module-scoped alias for the session-scoped Eden checkpoint."""
    return mbridge_eden_checkpoint


@pytest.mark.slow
def test_predict_eden_runs(
    tmp_path,
    mbridge_eden_checkpoint_path: Path,
    num_sequences: int = 3,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that predict_eden works correctly with an Eden (Llama) mbridge checkpoint.

    This exercises the full Eden prediction pipeline: loading a GPT/Llama model from
    the mbridge checkpoint run_config, running a forward pass, and writing predictions.
    """
    if target_sequence_lengths is None:
        target_sequence_lengths = [64, 64, 64]

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    output_dir = tmp_path / "eden_test_output"
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"-m bionemo.eden.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_eden_checkpoint_path} "
        f"--output-dir {output_dir} "
        f"--micro-batch-size 3 --write-interval epoch "
        f"--num-nodes 1 --devices 1"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(cmd_parts, check=False, cwd=tmp_path, capture_output=True, env=env, text=True)

    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    assert result.returncode == 0, f"Eden predict command failed with code {result.returncode}"

    pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
    assert len(pred_files) == 1, f"Expected 1 prediction file, got {len(pred_files)}"

    seq_idx_map_path = output_dir / "seq_idx_map.json"
    assert seq_idx_map_path.exists(), f"seq_idx_map.json not found at {seq_idx_map_path}"

    with open(seq_idx_map_path) as f:
        seq_idx_map = json.load(f)

    preds = [torch.load(pf, weights_only=True) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    assert isinstance(preds, dict)
    assert "token_logits" in preds
    assert "pad_mask" in preds
    assert "seq_idx" in preds

    assert len(preds["token_logits"]) == len(preds["pad_mask"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences

    for original_idx, pad_mask, token_logits in zip(preds["seq_idx"], preds["pad_mask"], preds["token_logits"]):
        expected_len = target_sequence_lengths[original_idx]
        assert pad_mask.sum() == expected_len
        # Vocab size is 256 for the default nucleotide tokenizer (padded to make_vocab_size_divisible_by)
        assert token_logits.shape[-1] == 256


@pytest.mark.slow
def test_predict_eden_log_probs(
    tmp_path,
    mbridge_eden_checkpoint_path: Path,
    num_sequences: int = 3,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that Eden prediction with log probability output works correctly."""
    if target_sequence_lengths is None:
        target_sequence_lengths = [64, 64, 64]

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    output_dir = tmp_path / "eden_logprobs_output"
    open_port = find_free_network_port()
    command = (
        f"torchrun --nproc_per_node 1 --nnodes 1 --master_port {open_port} "
        f"-m bionemo.eden.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_eden_checkpoint_path} "
        f"--output-dir {output_dir} "
        f"--micro-batch-size 3 --write-interval epoch "
        f"--num-nodes 1 --devices 1 "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(cmd_parts, check=False, cwd=tmp_path, capture_output=True, env=env, text=True)

    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    assert result.returncode == 0, f"Eden predict (log probs) command failed with code {result.returncode}"

    pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
    assert len(pred_files) == 1, f"Expected 1 prediction file, got {len(pred_files)}"

    preds = torch.load(pred_files[0], weights_only=True)
    assert isinstance(preds, dict)
    assert "log_probs_seqs" in preds
    assert "seq_idx" in preds
    assert len(preds["log_probs_seqs"]) == num_sequences
