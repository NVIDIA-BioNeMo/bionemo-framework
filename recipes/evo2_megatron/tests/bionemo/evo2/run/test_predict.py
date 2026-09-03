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

"""Tests for Evo2 prediction (inference) workflow using Megatron Bridge."""

import copy
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from bionemo.common.data.load import load as bionemo_load
from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.data.test_utils.create_fasta_file import ALU_SEQUENCE, create_fasta_file
from bionemo.evo2.run import predict as predict_module
from bionemo.evo2.run.predict import (
    _length_bucketed_batches,
    _packing_collate_fn,
    _predict_step,
    _unpack_packed_tensor,
    batch_collator,
)
from bionemo.evo2.run.predict import parse_args as parse_predict_args
from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

from ..utils import check_fp8_support, is_a6000_gpu


# Do this at collection time before we run any tests.
PRETEST_ENV = copy.deepcopy(os.environ)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], None), (["--context-parallel-comm-type", "p2p"], "p2p"), (["--context-parallel-comm-type", "a2a"], "a2a")],
)
def test_predict_context_parallel_comm_type_cli(monkeypatch, extra_args, expected):
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict", "--fasta", "/tmp/input.fasta", "--ckpt-dir", "/tmp/ckpt", *extra_args],
    )

    assert parse_predict_args().context_parallel_comm_type == expected


@pytest.mark.parametrize(
    ("extra_args", "expected_policy", "legacy_disabled"),
    [
        pytest.param([], "auto", False, id="default-auto"),
        pytest.param(["--sequence-parallel-policy", "on"], "on", False, id="explicit-on"),
        pytest.param(["--sequence-parallel-policy", "off"], "off", False, id="explicit-off"),
        pytest.param(["--no-sequence-parallel"], "auto", True, id="legacy-off-alias"),
    ],
)
def test_predict_sequence_parallel_policy_cli(monkeypatch, extra_args, expected_policy, legacy_disabled):
    """Removing either the tri-state option or legacy alias breaks the prediction CLI contract."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["predict", "--fasta", "/tmp/input.fasta", "--ckpt-dir", "/tmp/ckpt", *extra_args],
    )

    args = parse_predict_args()

    assert args.sequence_parallel_policy == expected_policy
    assert args.no_sequence_parallel is legacy_disabled


def test_predict_pp_fails_early() -> None:
    """Prediction has no pipeline schedule, so PP must fail before reading the checkpoint."""
    with pytest.raises(ValueError, match="Pipeline parallelism > 1 is not currently supported"):
        predict_module.predict(
            fasta_path=Path("/does/not/exist.fasta"),
            ckpt_dir=Path("/does/not/exist.ckpt"),
            pipeline_model_parallel_size=2,
        )


@pytest.mark.parametrize(
    ("embedding_layer", "expected_num_layers"),
    [(-25, 1), (-2, 24), (-1, 25), (0, 1), (5, 6), (24, 25)],
)
def test_resolve_embedding_layer(embedding_layer: int, expected_num_layers: int) -> None:
    assert predict_module._resolve_embedding_layer(embedding_layer, 25) == expected_num_layers


@pytest.mark.parametrize("embedding_layer", [-26, 25, 100])
def test_resolve_embedding_layer_rejects_out_of_range(embedding_layer: int) -> None:
    with pytest.raises(ValueError, match=rf"Invalid embedding_layer={embedding_layer}"):
        predict_module._resolve_embedding_layer(embedding_layer, 25)


def test_resolve_embedding_layer_rejects_log_probs() -> None:
    with pytest.raises(ValueError, match="Cannot use --output-log-prob-seqs with --embedding-layer"):
        predict_module._resolve_embedding_layer(-1, 25, output_log_prob_seqs=True)


def test_predict_step_gathers_sequence_parallel_embeddings():
    local_embeddings = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    full_embeddings = torch.arange(12, dtype=torch.float32).reshape(4, 1, 3)
    tp_group = object()
    batch = {
        "tokens": torch.zeros((1, 4), dtype=torch.long),
        "position_ids": torch.arange(4).reshape(1, 4),
        "loss_mask": torch.ones((1, 4), dtype=torch.bool),
        "seq_idx": torch.tensor([0]),
    }
    model = MagicMock(return_value=local_embeddings)
    model.module.config.sequence_parallel = True

    with (
        patch("bionemo.evo2.run.predict.parallel_state.is_pipeline_last_stage", return_value=True),
        patch("bionemo.evo2.run.predict.parallel_state.get_tensor_model_parallel_world_size", return_value=2),
        patch("bionemo.evo2.run.predict.parallel_state.get_tensor_model_parallel_group", return_value=tp_group),
        patch(
            "bionemo.evo2.run.predict.gather_from_sequence_parallel_region",
            create=True,
            return_value=full_embeddings,
        ) as gather_tp,
        patch("bionemo.evo2.run.predict._gather_along_cp_dim", side_effect=lambda value, **_: value) as gather_cp,
    ):
        result = _predict_step(model, batch, output_embeddings=True)

    gather_tp.assert_called_once_with(local_embeddings, tensor_parallel_output_grad=False, group=tp_group)
    gather_cp.assert_any_call(full_embeddings, seq_dim=0)
    torch.testing.assert_close(result["hidden_embeddings"], full_embeddings.transpose(0, 1))


def test_packed_embedding_step_gathers_and_unpacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ragged packed embeddings must retain the legacy padded endpoint schema under TP."""
    batch = _packing_collate_fn(
        [_prediction_sample(2, [10, 11, 12]), _prediction_sample(7, [20])],
        pad_token_id=0,
    )
    local_embeddings = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
    full_embeddings = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    tp_group = object()
    model = MagicMock(return_value=local_embeddings)
    model.module.config.sequence_parallel = True

    monkeypatch.setattr(predict_module.parallel_state, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(predict_module.parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(predict_module.parallel_state, "get_tensor_model_parallel_group", lambda: tp_group)
    gather_tp = MagicMock(return_value=full_embeddings)
    monkeypatch.setattr(predict_module, "gather_from_sequence_parallel_region", gather_tp)
    monkeypatch.setattr(predict_module, "_gather_along_cp_dim", lambda tensor, **_: tensor)

    result = _predict_step(model, batch, output_embeddings=True)

    assert model.call_args.kwargs["packed_seq_params"] is not None
    gather_tp.assert_called_once_with(local_embeddings, tensor_parallel_output_grad=False, group=tp_group)
    assert result["hidden_embeddings"].shape == (2, 3, 2)
    torch.testing.assert_close(result["hidden_embeddings"][0], full_embeddings[:3, 0])
    torch.testing.assert_close(result["hidden_embeddings"][1, 0], full_embeddings[3, 0])
    torch.testing.assert_close(result["hidden_embeddings"][1, 1:], torch.zeros(2, 2))
    torch.testing.assert_close(result["pad_mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]))
    torch.testing.assert_close(result["tokens"], torch.tensor([[10, 11, 12], [20, 0, 0]]))
    torch.testing.assert_close(result["seq_idx"], torch.tensor([2, 7]))


def test_global_fp8_all_layers_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["predict", "--fasta", "/tmp/in.fa", "--ckpt-dir", "/tmp/ckpt"])
    defaults = predict_module.parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--fasta",
            "/tmp/in.fa",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--mixed-precision-recipe",
            "bf16_with_fp8_current_scaling_mixed",
            "--fp8-all-layers",
        ],
    )
    fp8 = predict_module.parse_args()

    assert defaults.fp8_all_layers is False
    assert fp8.fp8_all_layers is True


def test_length_bucketed_batches_split_extreme_skew_once_before_model_layers() -> None:
    lengths = [200_000, 10, 10, 10, 10, 10, 300_000]

    batches = _length_bucketed_batches(
        lengths,
        max_records=6,
        max_tokens=250_000,
        bucket_by_length=True,
        data_parallel_rank=0,
        data_parallel_size=1,
    )

    assert batches == [[6], [0], [1, 2, 3, 4, 5]]
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert all(sum(lengths[index] for index in batch) <= 250_000 or len(batch) == 1 for batch in batches)


def test_length_bucketed_batches_reports_empty_record_indices() -> None:
    with pytest.raises(ValueError, match=r"record indices \[1, 3\].*--no-sequence-packing"):
        _length_bucketed_batches(
            [8, 0, 4, -1],
            max_records=4,
            max_tokens=32,
            bucket_by_length=True,
            data_parallel_rank=0,
            data_parallel_size=1,
        )


def _prediction_sample(index: int, tokens: list[int]) -> dict[str, torch.Tensor]:
    length = len(tokens)
    return {
        "tokens": torch.tensor(tokens, dtype=torch.long),
        "position_ids": torch.arange(length, dtype=torch.long),
        "seq_idx": torch.tensor(index, dtype=torch.long),
        "loss_mask": torch.ones(length, dtype=torch.long),
    }


def test_packing_collate_flattens_without_batch_max_padding() -> None:
    batch = [_prediction_sample(4, [10, 11, 12]), _prediction_sample(9, [20])]

    packed = _packing_collate_fn(batch, pad_token_id=0, min_length=2)

    torch.testing.assert_close(packed["tokens"], torch.tensor([[10, 11, 12, 20, 0]]))
    torch.testing.assert_close(packed["position_ids"], torch.tensor([[0, 1, 2, 0, 1]]))
    torch.testing.assert_close(packed["loss_mask"], torch.tensor([[1, 1, 1, 1, 0]]))
    torch.testing.assert_close(packed["seq_idx"], torch.tensor([4, 9]))
    torch.testing.assert_close(packed["cu_seqlens"], torch.tensor([0, 3, 5], dtype=torch.int32))
    torch.testing.assert_close(packed["packed_sequence_ids"], torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32))
    assert packed["packed_max_seqlen"] == 3
    assert packed["packed_pad_token_id"] == 0


def test_unpack_packed_tensor_materializes_endpoint_padding_once() -> None:
    flat = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    sequence_ids = torch.tensor([0, 0, 0, 1, 1])
    local_positions = torch.tensor([0, 1, 2, 0, 1])

    unpacked = _unpack_packed_tensor(
        flat,
        sequence_ids=sequence_ids,
        local_positions=local_positions,
        batch_size=2,
        max_seqlen=3,
        pad_value=-1,
    )

    assert unpacked.shape == (2, 3, 2)
    torch.testing.assert_close(unpacked[0], flat[0, :3])
    torch.testing.assert_close(unpacked[1, :2], flat[0, 3:])
    torch.testing.assert_close(unpacked[1, 2], torch.full((2,), -1.0))


def test_predict_step_sends_one_packed_call_and_restores_output_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    batch = _packing_collate_fn(
        [_prediction_sample(2, [10, 11, 12]), _prediction_sample(7, [20])],
        pad_token_id=0,
    )

    class RecordingModel:
        packed_seq_params = None

        def __call__(self, *, input_ids, position_ids, attention_mask, packed_seq_params=None):
            self.packed_seq_params = packed_seq_params
            total_tokens = input_ids.shape[1]
            return torch.arange(total_tokens * 3, dtype=torch.float32).reshape(1, total_tokens, 3)

    model = RecordingModel()
    monkeypatch.setattr(predict_module.parallel_state, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(predict_module.parallel_state, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(predict_module, "_gather_along_last_dim", lambda tensor, group: tensor)
    monkeypatch.setattr(predict_module, "_gather_along_cp_dim", lambda tensor, **_: tensor)

    result = predict_module._predict_step(model, batch)

    assert model.packed_seq_params is not None
    torch.testing.assert_close(model.packed_seq_params.cu_seqlens_q, batch["cu_seqlens"])
    assert result["token_logits"].shape == (2, 3, 3)
    torch.testing.assert_close(result["pad_mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]))
    torch.testing.assert_close(result["tokens"], torch.tensor([[10, 11, 12], [20, 0, 0]]))
    torch.testing.assert_close(result["seq_idx"], torch.tensor([2, 7]))


@pytest.mark.parametrize("collapse_option", ["sum", "mean"])
def test_packed_collapsed_log_probs_stay_flat(monkeypatch: pytest.MonkeyPatch, collapse_option: str) -> None:
    batch = _packing_collate_fn(
        [_prediction_sample(2, [10, 11, 12]), _prediction_sample(7, [20, 21])],
        pad_token_id=0,
    )
    logits = torch.randn(1, 5, 32, generator=torch.Generator().manual_seed(11))
    # A tempting cross-boundary label must never contribute to either sequence.
    logits[0, 2, 20] = 100.0

    class RecordingModel:
        def __call__(self, **_kwargs):
            return logits

    monkeypatch.setattr(predict_module.parallel_state, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(predict_module.parallel_state, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(predict_module, "_gather_along_last_dim", lambda tensor, group: tensor)
    monkeypatch.setattr(predict_module, "_gather_along_cp_dim", lambda tensor, **_: tensor)
    monkeypatch.setattr(
        predict_module,
        "_unpack_packed_tensor",
        lambda *_args, **_kwargs: pytest.fail("collapsed packed scoring must not materialize rectangular output"),
    )

    result = predict_module._predict_step(
        RecordingModel(),
        batch,
        output_log_prob_seqs=True,
        log_prob_collapse_option=collapse_option,
    )

    references = []
    for start, end in ((0, 3), (3, 5)):
        reference = predict_module._compute_log_probs(
            logits=logits[:, start:end],
            tokens=batch["tokens"][:, start:end],
            loss_mask=batch["loss_mask"][:, start:end],
            seq_idx=batch["seq_idx"][(0 if start == 0 else 1) : (1 if start == 0 else 2)],
            collapse_option=collapse_option,
            context_parallel_size=1,
        )
        references.append(reference["log_probs_seqs"])

    torch.testing.assert_close(result["log_probs_seqs"], torch.cat(references))
    torch.testing.assert_close(result["seq_idx"], batch["seq_idx"])


def _xfail_if_unsupported_subquadratic_ops(result: subprocess.CompletedProcess, use_subquadratic_ops: bool) -> None:
    if use_subquadratic_ops and "failed a CUDA self-test" in result.stderr:
        pytest.xfail("subquadratic_ops_torch CUDA kernels are unsupported in this environment")


@pytest.fixture(scope="module")
def mbridge_checkpoint_1b_8k_bf16_path(mbridge_checkpoint_1b_8k_bf16) -> Path:
    """Module-scoped alias for the session-scoped 1b-8k-bf16 checkpoint.

    The actual checkpoint conversion is done once per session in conftest.py via
    the mbridge_checkpoint_1b_8k_bf16 fixture, and shared across all test files.

    Returns:
        Path to the MBridge checkpoint iteration directory (e.g., .../iter_0000001)
    """
    return mbridge_checkpoint_1b_8k_bf16


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
            marks=pytest.mark.skip("Prediction intentionally rejects pipeline parallelism greater than one."),
        ),
    ],
)
@pytest.mark.slow
def test_predict_evo2_runs(
    tmp_path,
    ddp: int,
    pp: int,
    wi: str,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that the predict_evo2 command runs successfully with MBridge checkpoints.

    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    Since it's the full output this does not support CP, so we only test with TP=1. We also want coverage of the
        case where the sequence lengths are different and not necessarily divisible by CP.
    """
    if target_sequence_lengths is None:
        target_sequence_lengths = [3149, 3140, 1024, 3148, 3147]

    world_size = ddp * pp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {torch.cuda.device_count()}")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    # Create a local copy of the environment
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    # Build the command string
    output_dir = tmp_path / "test_output"
    command = (
        f"torchrun --standalone --nproc_per_node {world_size} --nnodes 1 "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--pipeline-model-parallel-size {pp} --num-nodes 1 --devices {world_size}"
    )

    # Run the command in a subshell
    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # For debugging purposes, print the output if the test fails
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    # Assert that the command completed successfully
    assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"
    combined_output = result.stdout + result.stderr
    assert "Prediction sequence packing: enabled" in combined_output
    assert "Packed prediction schedule:" in combined_output

    # Assert that the output directory was created and contains predictions
    # With DDP, each DP rank produces its own file with dp_rank in the filename
    # File naming convention:
    #   Batch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt
    #   Epoch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    if wi == "batch":
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*__batch_*.pt")))
        # With batch write interval, we expect multiple files (batches * dp_ranks)
        assert len(pred_files) >= ddp, f"Expected at least {ddp} prediction files, got {len(pred_files)}"
    else:
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
        # With epoch write interval, we expect one file per DP rank
        assert len(pred_files) == ddp, f"Expected {ddp} prediction files (one per DP rank), got {len(pred_files)}"

    # Check sequence index map exists
    seq_idx_map_path = output_dir / "seq_idx_map.json"
    assert seq_idx_map_path.exists(), f"seq_idx_map.json not found at {seq_idx_map_path}"

    with open(seq_idx_map_path) as f:
        seq_idx_map = json.load(f)

    # Load and collate predictions
    # Note: predict.py outputs are all batch-first (batch_dim=0), seq-second (seq_dim=1)
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
        # seq_idx is not sorted necessarily, so use the saved "seq_idx" to determine the original order
        expected_len = target_sequence_lengths[original_idx]
        assert pad_mask.sum() == expected_len
        # Vocab size should be 512 for the nucleotide tokenizer
        assert token_logits.shape[-1] == 512


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_1m_path(tmp_path_factory) -> Path:
    """Create or load a MBridge checkpoint for 7b-1m model testing."""
    try:
        nemo2_checkpoint_path = bionemo_load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    # Create a temporary directory for the MBridge checkpoint
    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b")
    # Note: run_nemo2_to_mbridge uses full model config from model_size
    # For testing we use the full 7b model but with shorter sequences
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_checkpoint_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="evo2_7b",
        seq_length=8192,  # Use shorter seq length for tests
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.fixture(scope="module")
def baseline_predictions_7b_1m_results(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path_factory,
    num_sequences: int = 5,
    target_sequence_lengths: list[int] | None = None,
) -> dict[int, float]:
    """Generate baseline predictions for 7b-1m model comparison."""
    if target_sequence_lengths is None:
        target_sequence_lengths = [2048, 2048, 2048, 2048, 2048]

    tmp_path = tmp_path_factory.mktemp("baseline_preds")
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        num_sequences,
        sequence_lengths=target_sequence_lengths,
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = tmp_path / "test_output"
    command = (
        "torchrun --standalone --nproc_per_node 1 --nnodes 1 "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 "
        f"--output-dir {output_dir} "
        f"--num-nodes 1 --write-interval epoch "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    env = copy.deepcopy(PRETEST_ENV)
    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )
    assert result.returncode == 0, f"predict_evo2 command failed: {result.stderr}"

    # Use the updated glob pattern matching the new naming convention
    # Epoch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    pred_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    preds = [torch.load(pf, weights_only=True) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    return dict(zip([i.item() for i in preds["seq_idx"]], [p.item() for p in preds["log_probs_seqs"]]))


@pytest.fixture(scope="module")
def subquadratic_predictions_7b_1m_results(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path_factory,
    num_sequences: int = 5,
) -> dict[int, float]:
    """Generate the TP=1 baseline for the accelerated subquadratic kernel family.

    Projection/mixer B2B fusion and the other accelerated kernels change BF16
    accumulation order relative to PyTorch's FFT/einsum implementations. Keep
    topology assertions tight within this production kernel family instead of
    weakening their tolerance against a different implementation.
    """
    target_sequence_lengths = [2048] * num_sequences
    tmp_path = tmp_path_factory.mktemp("subquadratic_baseline_preds")
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        num_sequences,
        sequence_lengths=target_sequence_lengths,
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = tmp_path / "test_output"
    command = (
        "torchrun --standalone --nproc_per_node 1 --nnodes 1 "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --output-dir {output_dir} --num-nodes 1 --write-interval epoch "
        "--use-subquadratic-ops --output-log-prob-seqs --log-prob-collapse-option sum"
    )
    result = subprocess.run(
        shlex.split(command),
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=copy.deepcopy(PRETEST_ENV),
        text=True,
    )
    _xfail_if_unsupported_subquadratic_ops(result, use_subquadratic_ops=True)
    assert result.returncode == 0, f"Subquadratic baseline prediction failed: {result.stderr}"

    pred_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    preds = batch_collator(
        [torch.load(path, weights_only=True) for path in pred_files],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    return dict(zip([i.item() for i in preds["seq_idx"]], [p.item() for p in preds["log_probs_seqs"]]))


def _run_rectangular_prediction_baseline(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path: Path,
    *,
    use_subquadratic_ops: bool,
    num_sequences: int = 5,
) -> dict[int, float]:
    """Generate a layout- and kernel-family-matched baseline for CP prediction.

    CP currently uses rectangular batches even though packed prediction is the CLI default.
    Comparing CP with the default packed baseline would conflate topology accuracy with the
    different BF16 accumulation order of the segmented kernels.
    """
    target_sequence_lengths = [2048] * num_sequences
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        num_sequences,
        sequence_lengths=target_sequence_lengths,
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = tmp_path / "test_output"
    subquadratic_option = "--use-subquadratic-ops" if use_subquadratic_ops else ""
    command = (
        "torchrun --standalone --nproc_per_node 1 --nnodes 1 "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --output-dir {output_dir} --num-nodes 1 --write-interval epoch "
        f"--no-sequence-packing {subquadratic_option} "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )
    run = subprocess.run(
        shlex.split(command),
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=copy.deepcopy(PRETEST_ENV),
        text=True,
    )
    _xfail_if_unsupported_subquadratic_ops(run, use_subquadratic_ops)
    assert run.returncode == 0, f"Rectangular baseline prediction failed: {run.stderr}"

    pred_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    preds = batch_collator(
        [torch.load(path, weights_only=True) for path in pred_files],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    return dict(zip([i.item() for i in preds["seq_idx"]], [p.item() for p in preds["log_probs_seqs"]]))


@pytest.fixture(scope="module")
def rectangular_predictions_7b_1m_results(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path_factory,
) -> dict[int, float]:
    return _run_rectangular_prediction_baseline(
        mbridge_checkpoint_7b_1m_path,
        tmp_path_factory.mktemp("rectangular_baseline_preds"),
        use_subquadratic_ops=False,
    )


@pytest.fixture(scope="module")
def rectangular_subquadratic_predictions_7b_1m_results(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path_factory,
) -> dict[int, float]:
    return _run_rectangular_prediction_baseline(
        mbridge_checkpoint_7b_1m_path,
        tmp_path_factory.mktemp("rectangular_subquadratic_baseline_preds"),
        use_subquadratic_ops=True,
    )


@pytest.fixture(scope="module")
def tp_reference_predictions_7b_1m_results(
    mbridge_checkpoint_7b_1m_path: Path,
    tmp_path_factory,
    num_sequences: int = 5,
) -> dict[int, float]:
    """Generate a BF16 TP=1 baseline with the slow test-only sharding oracle.

    This is intentionally separate from the normal TE baseline. The oracle makes
    BF16 tensor-parallel topology a mathematical-layout test instead of conflating it
    with topology-dependent GEMM accumulation. FP8 uses the production BF16 numerical
    reference because current scaling is performed independently on the actual TP shards.
    """
    target_sequence_lengths = [2048] * num_sequences
    tmp_path = tmp_path_factory.mktemp("tp_reference_baseline_preds")
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        num_sequences,
        sequence_lengths=target_sequence_lengths,
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    launcher = Path(__file__).with_name("tp_reference_predict.py")
    output_dir = tmp_path / "bf16"
    command = (
        "torchrun --standalone --nproc_per_node 1 --nnodes 1 "
        f"{launcher} --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --output-dir {output_dir} --num-nodes 1 --write-interval epoch "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )
    result = subprocess.run(
        shlex.split(command),
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=copy.deepcopy(PRETEST_ENV),
        text=True,
    )
    assert result.returncode == 0, f"TP reference prediction failed: {result.stderr}"

    pred_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    preds = batch_collator(
        [torch.load(path, weights_only=True) for path in pred_files],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    return dict(zip([i.item() for i in preds["seq_idx"]], [p.item() for p in preds["log_probs_seqs"]]))


@pytest.mark.parametrize(
    "ddp,cp,pp,tp,fp8,wi,use_subquadratic_ops,context_parallel_comm_type",
    [
        pytest.param(1, 1, 1, 1, False, "epoch", False, None, id="ddp=1,cp=1,pp=1,tp=1,fp8=False,wi=epoch,subq=False"),
        pytest.param(2, 1, 1, 1, False, "epoch", False, None, id="ddp=2,cp=1,pp=1,tp=1,fp8=False,wi=epoch,subq=False"),
        pytest.param(
            2, 1, 1, 1, False, "batch", False, None, id="ddp=2,cp=1,pp=1,tp=1,fp8=False,wi=batch,subq=False"
        ),  # simulate a large prediction run with dp parallelism
        pytest.param(
            1,
            2,
            1,
            1,
            False,
            "epoch",
            False,
            None,
            id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=epoch,subq=False,cpcomm=p2p-default",
        ),
        pytest.param(
            1,
            2,
            1,
            1,
            False,
            "epoch",
            False,
            "a2a",
            id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=epoch,subq=False,cpcomm=a2a",
        ),
        pytest.param(
            1,
            2,
            1,
            1,
            False,
            "batch",
            False,
            None,
            id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=batch,subq=False,cpcomm=p2p-default",
        ),
        pytest.param(1, 1, 1, 1, False, "epoch", True, None, id="ddp=1,cp=1,pp=1,tp=1,fp8=False,wi=epoch,subq=True"),
        pytest.param(
            1,
            2,
            1,
            1,
            False,
            "epoch",
            True,
            None,
            id="ddp=1,cp=2,pp=1,tp=1,fp8=False,wi=epoch,subq=True,cpcomm=p2p-default",
        ),
        pytest.param(
            1,
            1,
            2,
            1,
            False,
            "epoch",
            False,
            None,
            id="ddp=1,cp=1,pp=2,tp=1,fp8=False,wi=epoch,subq=False",
            marks=pytest.mark.skip("Prediction intentionally rejects pipeline parallelism greater than one."),
        ),
        pytest.param(
            1, 1, 1, 2, True, "epoch", False, None, id="ddp=1,cp=1,pp=1,tp=2,fp8=True,wi=epoch,subq=False"
        ),  # Cover case where FP8 was not supported with TP=2
        pytest.param(1, 1, 1, 2, False, "epoch", False, None, id="ddp=1,cp=1,pp=1,tp=2,fp8=False,wi=epoch,subq=False"),
        pytest.param(1, 1, 1, 8, False, "epoch", False, None, id="ddp=1,cp=1,pp=1,tp=8,fp8=False,wi=epoch,subq=False"),
        pytest.param(
            1, 1, 1, 8, True, "epoch", False, None, id="ddp=1,cp=1,pp=1,tp=8,fp8=True,wi=epoch,subq=False"
        ),  # Cover TP=8 with FP8
    ],
)
@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip 7b-1m checkpoint tests in CI due to disk space")
def test_predict_evo2_equivalent_with_log_probs(
    request: pytest.FixtureRequest,
    tmp_path,
    ddp: int,
    cp: int,
    pp: int,
    tp: int,
    fp8: bool,
    wi: str,
    use_subquadratic_ops: bool,
    context_parallel_comm_type: str | None,
    mbridge_checkpoint_7b_1m_path: Path,
    baseline_predictions_7b_1m_results: dict[int, float],
    num_sequences: int = 5,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that predict_evo2 produces equivalent log probabilities with different parallelism settings.

    This test runs the `predict_evo2` command with mock data in a temporary directory.
    It uses the temporary directory provided by pytest as the working directory.
    The command is run in a subshell, and we assert that it returns an exit code of 0.

    For this test, we want coverage of CP, so we make sure sequence lengths are all the same and divisible by CP.

    CP/DDP behavior is compared with the matching production kernel-family baseline.
    BF16 TP layout is compared with a deliberately slow, test-only full-logical-tensor
    oracle so the assertion is independent of topology-dependent accumulation order.
    FP8 TP uses production TE and is compared with BF16 because current scaling is
    intentionally local to the real TP shards and therefore topology-dependent.
    """
    if target_sequence_lengths is None:
        target_sequence_lengths = [2048, 2048, 2048, 2048, 2048]

    world_size = ddp * cp * pp * tp
    mp_size = cp * pp * tp
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {torch.cuda.device_count()}")
    is_fp8_supported, _, _ = check_fp8_support(torch.cuda.current_device())
    if not is_fp8_supported and fp8:
        pytest.skip("FP8 is not supported on this GPU.")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    # Create a local copy of the environment
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
        env["NCCL_P2P_DISABLE"] = "1"

    fp8_option = "--mixed-precision-recipe bf16_with_fp8_current_scaling_mixed --fp8-all-layers" if fp8 else ""
    subquadratic_ops_option = "--use-subquadratic-ops" if use_subquadratic_ops else ""
    context_parallel_comm_type_option = (
        f"--context-parallel-comm-type {context_parallel_comm_type}" if context_parallel_comm_type else ""
    )
    output_dir = tmp_path / "test_output"
    # BF16 TP correctness uses the slow test-only oracle. FP8 must use the normal TE
    # implementation so its per-shard current scaling is tested rather than emulated.
    launcher = (
        str(Path(__file__).with_name("tp_reference_predict.py"))
        if tp > 1 and not fp8
        else "-m bionemo.evo2.run.predict"
    )
    command = (
        f"torchrun --standalone --nproc_per_node {world_size} --nnodes 1 "
        f"{launcher} --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--micro-batch-size 3 --write-interval {wi} "
        f"--output-dir {output_dir} --tensor-parallel-size {tp} {fp8_option} {subquadratic_ops_option} "
        f"--pipeline-model-parallel-size {pp} --context-parallel-size {cp} {context_parallel_comm_type_option} "
        f"--num-nodes 1 --devices {world_size} "
        "--output-log-prob-seqs --log-prob-collapse-option sum"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # For debugging purposes, print the output if the test fails
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    # Assert that the command completed successfully
    assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"
    combined_output = result.stdout + result.stderr
    if cp > 1:
        assert "Sequence-packed prediction currently falls back to rectangular batches for CP>1" in combined_output
        assert "Prediction sequence packing: disabled" in combined_output
    else:
        assert "Prediction sequence packing: enabled" in combined_output

    # Assert that the output directory was created
    # With DDP, each DP rank produces its own file with dp_rank in the filename
    # File naming convention:
    #   Batch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}__batch_{batch_idx}.pt
    #   Epoch mode: predictions__rank_{global_rank}__dp_rank_{dp_rank}.pt
    if wi == "batch":
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*__batch_*.pt")))
        # With batch write interval, we expect multiple files (batches * dp_ranks)
        assert len(pred_files) >= ddp, f"Expected at least {ddp} prediction files, got {len(pred_files)}"
    else:
        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
        # With epoch write interval, we expect one file per DP rank
        assert len(pred_files) == ddp, f"Expected {ddp} prediction files (one per DP rank), got {len(pred_files)}"

    with open(output_dir / "seq_idx_map.json") as f:
        seq_idx_map = json.load(f)

    # Load and collate predictions from all DP ranks
    preds = [torch.load(pf, weights_only=True) for pf in pred_files]
    preds = batch_collator(
        [p for p in preds if p is not None],
        batch_dim=0,
        seq_dim=1,
        batch_dim_key_defaults={},
        seq_dim_key_defaults={},
    )
    assert isinstance(preds, dict)
    assert "log_probs_seqs" in preds
    assert "seq_idx" in preds
    assert len(preds["log_probs_seqs"]) == len(preds["seq_idx"]) == num_sequences
    assert len(seq_idx_map) == num_sequences

    if tp > 1 and fp8:
        expected_predictions = baseline_predictions_7b_1m_results
    elif tp > 1:
        expected_predictions = request.getfixturevalue("tp_reference_predictions_7b_1m_results")
    elif cp > 1 and not fp8:
        fixture_name = (
            "rectangular_subquadratic_predictions_7b_1m_results"
            if use_subquadratic_ops
            else "rectangular_predictions_7b_1m_results"
        )
        expected_predictions = request.getfixturevalue(fixture_name)
    elif use_subquadratic_ops:
        expected_predictions = request.getfixturevalue("subquadratic_predictions_7b_1m_results")
    else:
        expected_predictions = baseline_predictions_7b_1m_results
    for original_idx, log_probs in zip(preds["seq_idx"], preds["log_probs_seqs"]):
        if tp > 1 and not fp8:
            # The test-only full-logical-tensor GEMMs remain within 2e-6 across the
            # extra BF16 casts introduced by TP=8 sequence-parallel sharding.
            rel = 2e-6
        elif tp > 1 and fp8:
            # Current scaling is applied to each production TP shard. The measured
            # worst case against BF16 is ~2.02%, so retain a narrow 2.5% envelope.
            rel = 2.5e-2
        elif cp > 1 and not fp8:
            if context_parallel_comm_type == "a2a":
                # A2A evaluates every attention head over the full sequence and
                # matches the non-parallel reduction order.
                rel = 1e-6
            else:
                # TE's P2P ring evaluates attention in chunks, changing BF16
                # accumulation order. Main historically allowed 2e-3; the current
                # measured worst case is ~2.05e-3, so retain a narrow 2.5e-3 bound.
                rel = 2.5e-3
        elif mp_size > 1 and not fp8:
            # Pipeline-parallel prediction is currently skipped above; retain its
            # historical bound until that independent path is enabled and audited.
            rel = 2e-3
        elif ddp > 1:
            # Packed DDP repartitions sequences into different segmented calls. The
            # resulting BF16 FFT accumulation differs by about 1.1e-6 relative while
            # remaining far below a token-level accuracy threshold.
            rel = 2e-6
        elif use_subquadratic_ops:
            # A single-rank run must reproduce its matching kernel-family baseline.
            rel = 1e-6
        elif fp8:
            # FP8 + TP can have 1 to 2% log-prob drift vs baseline; use 2% relative tolerance.
            rel = 2e-2
        else:
            # Independent packed BF16 CUDA processes vary by up to about 1.5e-6
            # relative while preserving the same per-token predictions.
            rel = 2e-6
        assert log_probs.item() == pytest.approx(expected_predictions[original_idx.item()], rel=rel)


def _run_segmented_parallel_predict_probe(
    *,
    checkpoint_path: Path,
    work_dir: Path,
    tensor_parallel_size: int,
    fp8: bool = False,
) -> tuple[dict[int, float], list[dict]]:
    """Run one unequal-length packed call and collect actual kernel use per TP rank."""
    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(f"Packed prediction probe needs {tensor_parallel_size} GPUs, found {torch.cuda.device_count()}")
    fasta_path = work_dir / "ragged.fasta"
    create_fasta_file(
        fasta_path,
        3,
        sequence_lengths=[130, 190, 250],
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = work_dir / "predictions"
    probe_dir = work_dir / "kernel-proof"
    launcher = Path(__file__).with_name("packed_parallel_probe.py")
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(tensor_parallel_size),
        "--nnodes",
        "1",
        str(launcher),
        "predict",
        "--fasta",
        str(fasta_path),
        "--ckpt-dir",
        str(checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--micro-batch-size",
        "3",
        "--packed-token-budget",
        "1000",
        "--write-interval",
        "epoch",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--num-nodes",
        "1",
        "--devices",
        str(tensor_parallel_size),
        "--output-log-prob-seqs",
        "--log-prob-collapse-option",
        "sum",
    ]
    if fp8:
        command.extend(
            [
                "--mixed-precision-recipe",
                "bf16_with_fp8_current_scaling_mixed",
                "--fp8-all-layers",
            ]
        )
    env = copy.deepcopy(PRETEST_ENV)
    env["EVO2_PACKED_PROBE_DIR"] = str(probe_dir)
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900, env=env)
    assert result.returncode == 0, (
        f"Packed parallel predict probe failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    prediction_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    assert len(prediction_files) == 1
    predictions = torch.load(prediction_files[0], weights_only=True)
    values = {
        int(sequence_index): float(log_probability)
        for sequence_index, log_probability in zip(predictions["seq_idx"], predictions["log_probs_seqs"])
    }
    proofs = [json.loads((probe_dir / f"rank-{rank}.json").read_text()) for rank in range(tensor_parallel_size)]
    assert {proof["rank"] for proof in proofs} == set(range(tensor_parallel_size))
    assert all(proof["calls"] > 0 for proof in proofs)
    assert all(proof["max_segments"] == 3 for proof in proofs)
    assert {operator for proof in proofs for operator in proof["operator_counts"]} == {
        "hyena",
        "hyena_medium_conv",
        "hyena_short_conv",
    }
    return values, proofs


def _run_embedding_probe(
    *,
    checkpoint_path: Path,
    work_dir: Path,
    tensor_parallel_size: int,
    packed: bool,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Extract layer-two embeddings and optionally prove segmented execution on every rank."""
    if torch.cuda.device_count() < tensor_parallel_size:
        pytest.skip(f"Embedding probe needs {tensor_parallel_size} GPUs, found {torch.cuda.device_count()}")
    fasta_path = work_dir / "ragged.fasta"
    create_fasta_file(
        fasta_path,
        3,
        sequence_lengths=[130, 190, 250],
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = work_dir / "predictions"
    probe_dir = work_dir / "kernel-proof"
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(tensor_parallel_size),
        "--nnodes",
        "1",
    ]
    if packed:
        command.extend([str(Path(__file__).with_name("packed_parallel_probe.py")), "predict"])
    else:
        command.extend(["-m", "bionemo.evo2.run.predict"])
    command.extend(
        [
            "--fasta",
            str(fasta_path),
            "--ckpt-dir",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--micro-batch-size",
            "3",
            "--packed-token-budget",
            "1000",
            "--write-interval",
            "epoch",
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--num-nodes",
            "1",
            "--devices",
            str(tensor_parallel_size),
            "--embedding-layer",
            "2",
        ]
    )
    if not packed:
        command.append("--no-sequence-packing")
    env = copy.deepcopy(PRETEST_ENV)
    env["EVO2_PACKED_PROBE_DIR"] = str(probe_dir)
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900, env=env)
    assert result.returncode == 0, f"Embedding probe failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    combined_output = result.stdout + result.stderr
    expected_layout = "enabled" if packed else "disabled"
    assert f"Prediction sequence packing: {expected_layout}" in combined_output
    if tensor_parallel_size > 1:
        assert "Prediction sequence parallelism: enabled (policy: auto)" in combined_output

    prediction_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    assert len(prediction_files) == 1
    predictions = torch.load(prediction_files[0], weights_only=True)
    proofs = []
    if packed:
        proofs = [json.loads((probe_dir / f"rank-{rank}.json").read_text()) for rank in range(tensor_parallel_size)]
        assert {proof["rank"] for proof in proofs} == set(range(tensor_parallel_size))
        assert all(proof["calls"] == 3 for proof in proofs)
        assert all(proof["max_segments"] == 3 for proof in proofs)
        assert {operator for proof in proofs for operator in proof["operator_counts"]} == {
            "hyena",
            "hyena_medium_conv",
            "hyena_short_conv",
        }
    return predictions, proofs


@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip 7b-1m checkpoint tests in CI due to disk space")
def test_packed_embedding_tp2(mbridge_checkpoint_7b_1m_path, tmp_path_factory) -> None:
    """Ragged layer output must match rectangular and TP=1 while all packed operators run."""
    rectangular, _ = _run_embedding_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("embedding-rectangular-tp1-7b"),
        tensor_parallel_size=1,
        packed=False,
    )
    packed_tp1, _ = _run_embedding_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("embedding-packed-tp1-7b"),
        tensor_parallel_size=1,
        packed=True,
    )
    packed_tp2, _ = _run_embedding_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("embedding-packed-tp2-7b"),
        tensor_parallel_size=2,
        packed=True,
    )

    lengths = [130, 190, 250]
    outputs = (rectangular, packed_tp1, packed_tp2)
    for predictions in outputs:
        assert set(predictions) == {"hidden_embeddings", "pad_mask", "seq_idx", "tokens"}
        assert predictions["hidden_embeddings"].shape == (3, 250, 4096)
        assert predictions["pad_mask"].shape == predictions["tokens"].shape == (3, 250)
        for row, sequence_index in enumerate(predictions["seq_idx"].tolist()):
            length = lengths[sequence_index]
            assert predictions["pad_mask"][row].sum().item() == length
            assert torch.isfinite(predictions["hidden_embeddings"][row, :length]).all()
    for predictions in (packed_tp1, packed_tp2):
        for row, sequence_index in enumerate(predictions["seq_idx"].tolist()):
            assert torch.count_nonzero(predictions["hidden_embeddings"][row, lengths[sequence_index] :]) == 0

    def unpadded(predictions: dict[str, torch.Tensor], sequence_index: int) -> torch.Tensor:
        row = (predictions["seq_idx"] == sequence_index).nonzero(as_tuple=True)[0].item()
        return predictions["hidden_embeddings"][row, : lengths[sequence_index]].float()

    def relative_error(left: torch.Tensor, right: torch.Tensor) -> float:
        return float((left - right).norm() / right.norm().clamp_min(1e-30))

    for sequence_index in range(3):
        rectangular_embedding = unpadded(rectangular, sequence_index)
        packed_embedding = unpadded(packed_tp1, sequence_index)
        parallel_embedding = unpadded(packed_tp2, sequence_index)
        assert relative_error(packed_embedding, rectangular_embedding) < 1e-4
        assert relative_error(parallel_embedding, packed_embedding) < 5e-3


@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip 7b-1m checkpoint tests in CI due to disk space")
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8-all-layers"])
def test_segmented_packed_prediction_executes_on_every_tp_rank(
    mbridge_checkpoint_7b_1m_path,
    tmp_path_factory,
    fp8: bool,
) -> None:
    """Production TP cannot pass by silently selecting rectangular or padding-oracle code."""
    baseline, _ = _run_segmented_parallel_predict_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("packed-predict-baseline-7b"),
        tensor_parallel_size=1,
        fp8=fp8,
    )
    parallel, _ = _run_segmented_parallel_predict_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("packed-predict-tp2-7b"),
        tensor_parallel_size=2,
        fp8=fp8,
    )

    assert set(parallel) == set(baseline)
    for sequence_index in baseline:
        # Per-shard current scaling is topology-dependent; the measured ragged FP8
        # maximum is 2.83%, while the broken SP padding path exceeded 1,100%.
        assert parallel[sequence_index] == pytest.approx(baseline[sequence_index], rel=3.5e-2 if fp8 else 2e-2)


@pytest.mark.parametrize(
    "tp,fp8",
    [
        pytest.param(2, False, id="tp=2,fp8=False"),
        pytest.param(2, True, id="tp=2,fp8=True"),
        pytest.param(8, False, id="tp=8,fp8=False"),
        pytest.param(8, True, id="tp=8,fp8=True"),
    ],
)
@pytest.mark.slow
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip 7b-1m checkpoint tests in CI due to disk space")
def test_predict_standard_te_tensor_parallel_smoke(
    tmp_path,
    tp: int,
    fp8: bool,
    mbridge_checkpoint_7b_1m_path: Path,
) -> None:
    """Keep the production TE TP path fast while the separate oracle checks its layout."""
    if tp > torch.cuda.device_count():
        pytest.skip(f"TP size {tp} is greater than the number of available GPUs")
    is_fp8_supported, _, _ = check_fp8_support(torch.cuda.current_device())
    if fp8 and not is_fp8_supported:
        pytest.skip("FP8 is not supported on this GPU.")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path,
        1,
        sequence_lengths=[256],
        repeating_dna_pattern=ALU_SEQUENCE,
    )
    output_dir = tmp_path / "test_output"
    fp8_option = "--mixed-precision-recipe bf16_with_fp8_current_scaling_mixed" if fp8 else ""
    command = (
        f"torchrun --standalone --nproc_per_node {tp} --nnodes 1 -m bionemo.evo2.run.predict "
        f"--fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_7b_1m_path} "
        f"--output-dir {output_dir} --tensor-parallel-size {tp} --num-nodes 1 --devices {tp} "
        f"{fp8_option} --output-log-prob-seqs --log-prob-collapse-option sum"
    )
    result = subprocess.run(
        shlex.split(command),
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=copy.deepcopy(PRETEST_ENV),
        text=True,
    )
    assert result.returncode == 0, f"Standard TE TP prediction failed: {result.stderr}"

    pred_files = glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt"))
    assert len(pred_files) == 1
    predictions = torch.load(pred_files[0], weights_only=True)["log_probs_seqs"]
    assert predictions.shape == (1,)
    assert torch.isfinite(predictions).all()


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_different_results_with_without_peft(tmp_path, mbridge_checkpoint_1b_8k_bf16_path, lora_finetune_checkpoint):
    """Predict on base vs. LoRA ckpt and assert logits differ."""
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_file_path, 3, sequence_lengths=[32, 65, 129], repeating_dna_pattern=ALU_SEQUENCE)

    def _run_predict(ckpt: Path, output_dir: Path) -> None:
        cmd = (
            "torchrun --standalone --nproc_per_node 1 --nnodes 1 "
            f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {ckpt} "
            f"--output-dir {output_dir} --micro-batch-size 3 --write-interval epoch "
            f"--pipeline-model-parallel-size 1 --num-nodes 1 --devices 1"
        )
        r = subprocess.run(shlex.split(cmd), check=False, cwd=tmp_path, capture_output=True, text=True, env=env)
        assert r.returncode == 0, f"predict_evo2 failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"

    out_base = tmp_path / "out_base"
    out_lora = tmp_path / "out_lora"
    _run_predict(mbridge_checkpoint_1b_8k_bf16_path, out_base)
    _run_predict(lora_finetune_checkpoint, out_lora)

    base_files = glob.glob(str(out_base / "predictions__rank_*__dp_rank_*.pt"))
    lora_files = glob.glob(str(out_lora / "predictions__rank_*__dp_rank_*.pt"))
    assert len(base_files) == 1 and len(lora_files) == 1

    base = torch.load(base_files[0], weights_only=False)
    lora = torch.load(lora_files[0], weights_only=False)
    assert torch.equal(base["seq_idx"], lora["seq_idx"])
    assert base["token_logits"].shape == lora["token_logits"].shape
    assert (base["token_logits"] != lora["token_logits"]).any(), "LoRA adapter had no effect on logits"


@pytest.mark.parametrize(
    "embedding_layer,expected_num_layers",
    [
        pytest.param(
            -1,
            25,
            id="embedding_layer=-1_expects_25_layers",
            marks=pytest.mark.skipif(
                bool(os.environ.get("CI")), reason="Full-depth embeddings run in prefix parity test"
            ),
        ),
        pytest.param(
            -2,
            24,
            id="embedding_layer=-2_expects_24_layers",
            marks=pytest.mark.skipif(
                bool(os.environ.get("CI")), reason="Interior negative index is covered outside CI"
            ),
        ),
        pytest.param(0, 1, id="embedding_layer=0_expects_1_layer"),
        pytest.param(
            5,
            6,
            id="embedding_layer=5_expects_6_layers",
            marks=pytest.mark.skipif(
                bool(os.environ.get("CI")), reason="Interior positive index is covered outside CI"
            ),
        ),
    ],
)
@pytest.mark.slow
def test_predict_evo2_embedding_extraction(
    tmp_path,
    embedding_layer: int,
    expected_num_layers: int,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
    num_sequences: int = 3,
    target_sequence_lengths: list[int] | None = None,
):
    """Test that embedding extraction produces outputs with expected shapes and keys.

    This test verifies:
    1. The model is initialized with the correct number of layers (logged and verified)
    2. Output contains 'hidden_embeddings' key instead of 'token_logits'
    3. Embeddings have expected shape [B, S, H] where H is hidden dimension
    4. Other expected keys (pad_mask, seq_idx, tokens) are present

    The 1b model has 25 layers, so:
    - embedding_layer=-1 -> 25 layers (last layer)
    - embedding_layer=-2 -> 24 layers (second-to-last)
    - embedding_layer=0 -> 1 layer (first layer only)
    - embedding_layer=5 -> 6 layers (layers 0-5)
    """
    original_num_layers = 25  # 1b model has 25 layers

    if target_sequence_lengths is None:
        target_sequence_lengths = [64, 96, 128]

    world_size = 1
    if world_size > torch.cuda.device_count():
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {torch.cuda.device_count()}")

    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(
        fasta_file_path, num_sequences, sequence_lengths=target_sequence_lengths, repeating_dna_pattern=ALU_SEQUENCE
    )

    # Create a local copy of the environment
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"

    output_dir = tmp_path / "test_output"
    command = (
        f"torchrun --standalone --nproc_per_node {world_size} --nnodes 1 "
        f"-m bionemo.evo2.run.predict --fasta {fasta_file_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
        f"--output-dir {output_dir} "
        f"--micro-batch-size 2 --write-interval epoch "
        f"--embedding-layer {embedding_layer}"
    )

    cmd_parts = shlex.split(command)
    result = subprocess.run(
        cmd_parts,
        check=False,
        cwd=tmp_path,
        capture_output=True,
        env=env,
        text=True,
    )

    # For debugging purposes, print the output if the test fails
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)

    # Assert that the command completed successfully
    assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"

    # Combine stdout and stderr for log checking
    combined_output = result.stdout + result.stderr

    # Verify logging about model layers is present and extract the layer count
    assert "Model initialized with" in combined_output, "Expected logging about model layer count"
    assert "Embedding extraction" in combined_output, "Expected logging about embedding extraction mode"

    # Parse and verify the actual number of layers from the log
    # Look for pattern: "Model initialized with N layers"
    layer_match = re.search(r"Model initialized with (\d+) layers", combined_output)
    assert layer_match is not None, "Could not parse 'Model initialized with N layers' from output"
    actual_num_layers = int(layer_match.group(1))
    assert actual_num_layers == expected_num_layers, (
        f"Expected model to have {expected_num_layers} layers for embedding_layer={embedding_layer}, "
        f"but got {actual_num_layers} layers"
    )

    # Verify the embedding extraction log shows correct layer info
    # Look for pattern: "using N of M layers"
    extraction_match = re.search(r"using (\d+) of (\d+) layers", combined_output)
    assert extraction_match is not None, "Could not parse 'using N of M layers' from output"
    layers_used = int(extraction_match.group(1))
    layers_original = int(extraction_match.group(2))
    assert layers_used == expected_num_layers, (
        f"Expected 'using {expected_num_layers}' layers, but log shows 'using {layers_used}'"
    )
    assert layers_original == original_num_layers, (
        f"Expected original model to have {original_num_layers} layers, but log shows {layers_original}"
    )

    # Load predictions
    pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
    assert len(pred_files) == 1, f"Expected 1 prediction file, got {len(pred_files)}"

    preds = torch.load(pred_files[0], weights_only=True)
    assert isinstance(preds, dict)

    # Verify expected keys for embedding extraction
    assert "hidden_embeddings" in preds, "Expected 'hidden_embeddings' key in embedding extraction mode"
    assert "token_logits" not in preds, "Should not have 'token_logits' in embedding extraction mode"
    assert "pad_mask" in preds, "Expected 'pad_mask' key"
    assert "seq_idx" in preds, "Expected 'seq_idx' key"
    assert "tokens" in preds, "Expected 'tokens' key"

    # Verify shapes
    hidden_embeddings = preds["hidden_embeddings"]
    pad_mask = preds["pad_mask"]
    tokens = preds["tokens"]

    # hidden_embeddings should be [B, S, H] where H is hidden dimension (1920 for 1b model)
    assert len(hidden_embeddings.shape) == 3, f"Expected 3D tensor, got shape {hidden_embeddings.shape}"
    batch_size, seq_len, hidden_dim = hidden_embeddings.shape

    assert batch_size == num_sequences, f"Expected batch size {num_sequences}, got {batch_size}"
    # Sequence length should match padded length
    max_seq_len = max(target_sequence_lengths)
    assert seq_len == max_seq_len, f"Expected seq_len {max_seq_len}, got {seq_len}"
    # Hidden dim should be 1920 for 1b model
    assert hidden_dim == 1920, f"Expected hidden_dim 1920 for 1b model, got {hidden_dim}"

    # Verify pad_mask and tokens have matching shapes
    assert pad_mask.shape == (batch_size, seq_len), f"pad_mask shape mismatch: {pad_mask.shape}"
    assert tokens.shape == (batch_size, seq_len), f"tokens shape mismatch: {tokens.shape}"

    # Verify seq_idx has correct count
    assert len(preds["seq_idx"]) == num_sequences, f"Expected {num_sequences} seq_idx entries"

    # Check sequence index map exists
    seq_idx_map_path = output_dir / "seq_idx_map.json"
    assert seq_idx_map_path.exists(), f"seq_idx_map.json not found at {seq_idx_map_path}"

    with open(seq_idx_map_path) as f:
        seq_idx_map = json.load(f)
    assert len(seq_idx_map) == num_sequences


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_predict_evo2_short_embedding_is_prefix_invariant_across_batch_padding(
    tmp_path,
    mbridge_checkpoint_1b_8k_bf16_path: Path,
):
    """A short sequence should embed the same alone or padded in a longer batch."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Embedding prediction test requires a GPU")

    short_sequence = "ACGTACGTAA"
    padding_sequence = (ALU_SEQUENCE * (256 // len(ALU_SEQUENCE) + 1))[:256]

    def _write_fasta(fasta_path: Path, records: dict[str, str]) -> None:
        fasta_path.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in records.items()))

    def _run_predict(fasta_path: Path, output_dir: Path) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        command = (
            "torchrun --standalone --nproc_per_node 1 --nnodes 1 "
            f"-m bionemo.evo2.run.predict --fasta {fasta_path} --ckpt-dir {mbridge_checkpoint_1b_8k_bf16_path} "
            f"--output-dir {output_dir} --micro-batch-size 2 --write-interval epoch --embedding-layer -1"
        )
        result = subprocess.run(
            shlex.split(command),
            check=False,
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print("STDOUT:\n" + result.stdout)
            print("STDERR:\n" + result.stderr)
        assert result.returncode == 0, f"predict_evo2 command failed with code {result.returncode}"

        pred_files = sorted(glob.glob(str(output_dir / "predictions__rank_*__dp_rank_*.pt")))
        assert len(pred_files) == 1, f"Expected 1 prediction file, got {len(pred_files)}"
        with open(output_dir / "seq_idx_map.json") as f:
            seq_idx_map = json.load(f)
        return torch.load(pred_files[0], weights_only=True), seq_idx_map

    def _unpadded_dna_embeddings(
        preds: dict[str, torch.Tensor],
        seq_idx_map: dict[str, int],
        seqid: str,
        dna_length: int,
    ) -> torch.Tensor:
        matches = (preds["seq_idx"] == seq_idx_map[seqid]).nonzero(as_tuple=True)[0]
        assert matches.numel() == 1
        row = matches.item()
        assert preds["pad_mask"][row].sum().item() == dna_length
        return preds["hidden_embeddings"][row, :dna_length].to(torch.float32)

    def _relative_frobenius_error(left: torch.Tensor, right: torch.Tensor) -> float:
        numerator = (left - right).float().pow(2).sum().sqrt()
        denominator = right.float().pow(2).sum().sqrt()
        return float(numerator / (denominator + 1e-30))

    def _assert_prefix_embeddings_close(left: torch.Tensor, right: torch.Tensor) -> None:
        rel_error = _relative_frobenius_error(left, right)
        bound = 4.0 * (1.03**33) * float(torch.finfo(torch.bfloat16).eps)
        if rel_error <= bound:
            return

        rel_shuffled_hidden = _relative_frobenius_error(left, torch.roll(right, shifts=-1, dims=-1))
        rel_shuffled_sequence = _relative_frobenius_error(left, torch.roll(right, shifts=-1, dims=0))
        max_abs_diff = (left - right).abs().max().item()
        raise AssertionError(
            "Prefix embeddings exceeded bf16 relative-norm tolerance: "
            f"rel={rel_error}, bound={bound}, rel_shuffled_hidden={rel_shuffled_hidden}, "
            f"rel_shuffled_sequence={rel_shuffled_sequence}, max_abs_diff={max_abs_diff}"
        )

    alone_fasta = tmp_path / "short_alone.fasta"
    padded_fasta = tmp_path / "short_padded.fasta"
    _write_fasta(alone_fasta, {"short": short_sequence})
    _write_fasta(padded_fasta, {"short": short_sequence, "padding": padding_sequence})
    alone_preds, alone_seq_idx_map = _run_predict(alone_fasta, tmp_path / "alone_output")
    padded_preds, padded_seq_idx_map = _run_predict(padded_fasta, tmp_path / "padded_output")
    assert alone_preds["hidden_embeddings"].shape[1] == len(short_sequence)
    assert padded_preds["hidden_embeddings"].shape[1] == len(padding_sequence)

    alone_embeddings = _unpadded_dna_embeddings(alone_preds, alone_seq_idx_map, "short", len(short_sequence))
    padded_embeddings = _unpadded_dna_embeddings(padded_preds, padded_seq_idx_map, "short", len(short_sequence))

    _assert_prefix_embeddings_close(alone_embeddings, padded_embeddings)


def test_load_model_to_layer_requires_layer():
    """`full=False` needs a layer; the guard fails fast before any checkpoint I/O (CPU)."""
    from bionemo.evo2.run.predict import load_model_to_layer

    with pytest.raises(ValueError, match="layer is required"):
        load_model_to_layer("/nonexistent/ckpt", layer=None, full=False)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU to load Evo2")
def test_load_model_to_layer_truncated(mbridge_checkpoint_path):
    """Truncated load returns a usable (model, tokenizer) for hidden-state extraction."""
    from bionemo.evo2.run.predict import load_model_to_layer

    model, tokenizer = load_model_to_layer(mbridge_checkpoint_path, layer=2, full=False)
    assert model is not None
    assert tokenizer.vocab_size > 0
