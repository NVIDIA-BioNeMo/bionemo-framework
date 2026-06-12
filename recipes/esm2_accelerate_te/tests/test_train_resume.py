# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random
import re
import shutil
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

import train


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires a GPU",
)


@requires_gpu
def test_train_can_resume_from_checkpoint(monkeypatch, tmp_path: Path):
    """Test that train.py runs successfully with sanity config and creates expected outputs."""

    # Get the recipe directory
    recipe_dir = Path(train.__file__).parent

    # Set required environment variables for distributed training
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", f"{random.randint(20000, 40000)}")
    monkeypatch.setenv("WANDB_MODE", "disabled")

    with initialize_config_dir(config_dir=str(recipe_dir / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"trainer.output_dir={tmp_path}",
                "stop_after_n_steps=4",
                "trainer.do_eval=False",
                "trainer.save_steps=2",
                f"hydra.run.dir={tmp_path}/outputs",
                "trainer.dataloader_num_workers=0",
            ],
        )

    train.main(sanity_config)

    output_dir = tmp_path

    # Check that the output directory exists
    assert output_dir.exists(), f"Output directory {output_dir} does not exist"

    # Check for checkpoint directories
    checkpoint_dirs = [d for d in output_dir.iterdir() if d.is_dir() and re.match(r"checkpoint-\d+", d.name)]
    assert len(checkpoint_dirs) == 2, (
        f"Expected 2 checkpoint directories, found {len(checkpoint_dirs)}: {[d.name for d in checkpoint_dirs]}"
    )

    # Check for the final model checkpoint
    final_checkpoint = output_dir / "checkpoint-last"
    assert final_checkpoint.exists(), f"Final checkpoint directory {final_checkpoint} does not exist"

    # Verify the final checkpoint contains model files
    model_files = list(final_checkpoint.glob("*.safetensors"))
    config_files = list(final_checkpoint.glob("*.json"))

    assert len(model_files) > 0, f"No model files found in {final_checkpoint}"
    assert len(config_files) > 0, f"No config files found in {final_checkpoint}"

    # Check that training metrics were saved
    train_metrics_file = output_dir / "train_results.json"
    assert train_metrics_file.exists(), f"Training metrics file {train_metrics_file} does not exist"

    ## Remove last two checkpoints and re-train

    # Remove the checkpoint-10 and checkpoint-last directories
    checkpoint_4 = tmp_path / "checkpoint-4"
    checkpoint_last = tmp_path / "checkpoint-last"
    if checkpoint_4.exists():
        shutil.rmtree(checkpoint_4)
    if checkpoint_last.exists():
        shutil.rmtree(checkpoint_last)

    assert (tmp_path / "checkpoint-2").exists(), f"Checkpoint-2 directory {tmp_path / 'checkpoint-2'} does not exist."

    # Re-train
    train.main(sanity_config)

    # Check for checkpoint directories
    checkpoint_dirs = [d for d in output_dir.iterdir() if d.is_dir() and re.match(r"checkpoint-\d+", d.name)]
    assert len(checkpoint_dirs) == 2, (
        f"Expected 2 checkpoint directories, found {len(checkpoint_dirs)}: {[d.name for d in checkpoint_dirs]}"
    )

    # Check for the final model checkpoint
    final_checkpoint = output_dir / "checkpoint-last"
    assert final_checkpoint.exists(), f"Final checkpoint directory {final_checkpoint} does not exist"

    # Verify the final checkpoint contains model files
    model_files = list(final_checkpoint.glob("*.safetensors"))
    config_files = list(final_checkpoint.glob("*.json"))

    assert len(model_files) > 0, f"No model files found in {final_checkpoint}"
    assert len(config_files) > 0, f"No config files found in {final_checkpoint}"

    # Check that training metrics were saved
    train_metrics_file = output_dir / "train_results.json"
    assert train_metrics_file.exists(), f"Training metrics file {train_metrics_file} does not exist"
