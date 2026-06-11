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

from pathlib import Path
from unittest import mock

import pytest
from datasets import Dataset

from bionemo.amplify.train_amplify import main
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.llm.model.biobert.model import BiobertSpecOption
from bionemo.testing import megatron_parallel_state_utils


@pytest.fixture
def mock_hf_dataset():
    """Create a mock HuggingFace dataset with protein sequences."""
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "DEFGHIKLMNPQRSTVWYAC",
        "MGHIKLMNPQRSTVWYACDE",
        "MKTVRQERLKSIVRI",
        "MRILERSKEPVSGAQLA",
    ]
    return Dataset.from_dict({"sequence": sequences})


@pytest.mark.parametrize("create_checkpoint_callback", [True, False])
def test_train_amplify_runs_small_model(tmpdir, monkeypatch, mock_hf_dataset, create_checkpoint_callback):
    """Test that train_amplify can run a small model for a few steps."""
    # Set up the test directory
    result_dir = Path(tmpdir) / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Mock the HuggingFace dataset loading
    mock_dataset = mock.MagicMock(return_value=mock_hf_dataset)
    monkeypatch.setattr("bionemo.amplify.train_amplify.hf_load_dataset", mock_dataset)

    # Run the training with minimal settings
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        trainer = main(
            num_nodes=1,
            devices=1,
            min_seq_length=64,
            max_seq_length=64,
            result_dir=result_dir,
            num_steps=3,
            warmup_steps=1,
            decay_steps=2,
            limit_val_batches=1,
            val_check_interval=1,
            log_every_n_steps=1,
            num_dataset_workers=1,
            biobert_spec_option=BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec,
            lr=1e-4,
            micro_batch_size=2,
            accumulate_grad_batches=1,
            experiment_name="test_amplify",
            resume_if_exists=False,
            precision="bf16-mixed",
            wandb_offline=True,
            wandb_project=None,
            create_tensorboard_logger=False,
            create_checkpoint_callback=create_checkpoint_callback,  # Use the parameter
            # Small model configuration
            num_layers=2,
            hidden_size=8,
            num_attention_heads=2,
            ffn_hidden_size=32,
            random_mask_strategy=RandomMaskStrategy.ALL_TOKENS,
        )

    # Verify the training completed successfully
    experiment_dir = result_dir / "test_amplify"
    assert experiment_dir.exists(), "Could not find test experiment directory"
    assert experiment_dir.is_dir(), "Test experiment directory is supposed to be a directory"

    # Check for the run directory
    log_dir = experiment_dir / "dev"
    assert log_dir.exists(), "Directory with logs does not exist"

    # Check expected number of children based on checkpoint creation
    children = list(experiment_dir.iterdir())
    expected_children = 2 if create_checkpoint_callback else 1
    assert len(children) == expected_children, (
        f"Expected {expected_children} children in experiment directory, found {children}"
    )

    # Check for checkpoints if they should exist
    if create_checkpoint_callback:
        checkpoints_dir = experiment_dir / "checkpoints"
        assert checkpoints_dir.exists(), "Checkpoints directory does not exist"

        # Check if correct checkpoint was saved
        expected_checkpoint_suffix = "step=2"  # Since you run 3 steps (0,1,2)
        matching_checkpoints = [p for p in checkpoints_dir.iterdir() if expected_checkpoint_suffix in p.name]
        assert matching_checkpoints, (
            f"No checkpoint file with '{expected_checkpoint_suffix}' found in {checkpoints_dir}"
        )

    # Check for logs
    assert (log_dir / "nemo_log_globalrank-0_localrank-0.txt").is_file(), "Could not find experiment log"

    # Verify trainer completed steps
    assert trainer.global_step == 3, f"Expected 3 training steps, got {trainer.global_step}"
