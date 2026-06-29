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

import random

import pytest
import torch
from hydra import compose, initialize_config_dir
from train_fsdp2 import main as main_fsdp2


@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def test_sanity_convergence_fsdp2(tmp_path, recipe_path):
    """Test that CodonFM converges with FSDP2 on synthetic data."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_no_meta_device(tmp_path, recipe_path):
    """Test CodonFM without meta device initialization."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                "use_meta_device=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_thd(tmp_path, monkeypatch, recipe_path):
    """Test CodonFM with THD sequence packing."""
    if torch.cuda.get_device_capability() == (12, 0):
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_grad_acc(tmp_path, recipe_path):
    """Test CodonFM FSDP2 training with gradient accumulation."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_thd_grad_acc(tmp_path, monkeypatch, recipe_path):
    """Test CodonFM with THD sequence packing and gradient accumulation."""
    if torch.cuda.get_device_capability() == (12, 0):
        monkeypatch.setenv("NVTE_FUSED_ATTN", "0")

    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_fp8(tmp_path, recipe_path):
    """Test CodonFM with FP8 enabled for all layers."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"


def test_sanity_convergence_fsdp2_fp32_master_weights(tmp_path, recipe_path):
    """Test CodonFM with FP32 master weights."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb_init_args.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_fp32_master_weights=true",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    assert final_loss < 5.0, f"Final loss {final_loss} is too high"
