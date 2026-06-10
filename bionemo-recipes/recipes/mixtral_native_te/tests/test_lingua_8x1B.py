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

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scheduler import get_cosine_annealing_schedule_with_warmup
from torch.optim import AdamW


def test_lingua_8x1b_optimizer_golden_values(recipe_path):
    """Test that optimizer and scheduler match the recipe configuration."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        config = compose(config_name="L2_lingua_8x1B")

    model = torch.nn.Linear(10, 1)
    optimizer = AdamW(model.parameters(), **OmegaConf.to_container(config.adamw_kwargs, resolve=True))  # type: ignore[arg-type]

    assert optimizer.param_groups[0]["lr"] == config.adamw_kwargs.lr
    assert list(optimizer.param_groups[0]["betas"]) == list(config.adamw_kwargs.betas)
    assert optimizer.param_groups[0]["eps"] == config.adamw_kwargs.eps
    assert optimizer.param_groups[0]["weight_decay"] == config.adamw_kwargs.weight_decay

    scheduler = get_cosine_annealing_schedule_with_warmup(optimizer, **config.lr_scheduler_kwargs)

    for _ in range(3):
        optimizer.step()
        scheduler.step()

    assert optimizer.param_groups[0]["lr"] > 0
