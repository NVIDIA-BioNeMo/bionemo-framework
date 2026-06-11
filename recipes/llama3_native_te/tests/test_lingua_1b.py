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

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.optim import AdamW

from scheduler import get_cosine_annealing_schedule_with_warmup


def test_lingua_1b_optimizer_golden_values(recipe_path):
    """Test that optimizer and scheduler match golden values from a lingua 1B NeMo 2.0 run."""
    optimizer_golden_values = [
        (3879, 0.0023268),
        (16370, 0.0026946743005784087),
        (31387, 0.0015953843038901037),
        (56763, 0.000025585942891050332),
    ]  # From https://gitlab-master.nvidia.com/dl/JoC/nemo-ci/-/issues/1050

    # Load the config from L2_lingua_1b.yaml
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(config_name="L2_lingua_1b")

    # Create a dummy model with parameters for the optimizer
    model = torch.nn.Linear(10, 1)

    # Create optimizer using the same logic as train_fsdp2.py
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(config.adamw_kwargs, resolve=True))  # type: ignore

    # Verify optimizer parameters match config
    assert optimizer.param_groups[0]["lr"] == config.adamw_kwargs.lr

    # Betas may be stored as list or tuple, so compare values
    assert list(optimizer.param_groups[0]["betas"]) == list(config.adamw_kwargs.betas)
    assert optimizer.param_groups[0]["eps"] == config.adamw_kwargs.eps
    assert optimizer.param_groups[0]["weight_decay"] == config.adamw_kwargs.weight_decay

    # Create scheduler using the same logic as train_fsdp2.py
    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **config.lr_scheduler_kwargs)

    # Step through the scheduler and verify learning rates match golden values
    # Note: In PyTorch 1.1.0+, optimizer.step() should be called before scheduler.step()
    # For this test, we call optimizer.step() first to avoid warnings and match training behavior
    current_step = 0
    for target_step, expected_lr in optimizer_golden_values:
        # Step scheduler to target step (calling optimizer.step() first to match PyTorch best practices)
        while current_step < target_step:
            optimizer.step()
            scheduler.step()
            current_step += 1

        # Get the current learning rate
        actual_lr = optimizer.param_groups[0]["lr"]

        # Verify learning rate matches golden value (with tolerance for floating point precision)
        torch.testing.assert_close(
            actual_lr,
            expected_lr,
            atol=1e-8,
            rtol=1e-3,
            msg=lambda x: f"Learning rate mismatch at step {target_step}: {x}",
        )
