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

import gc
import random

import pytest
import torch
from hydra import compose, initialize_config_dir
from train_ddp import main as main_ddp
from train_fsdp2 import main as main_fsdp2
from transformer_engine.pytorch.fp8 import check_fp8_support


# TODO(@jomitchell): Delete once https://nvbugspro.nvidia.com/bug/5458694 is fixed.
requires_datacenter_hardware = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not any(
        gpu_name in torch.cuda.get_device_name(0).upper() for gpu_name in ["H100", "H200", "B100", "B200", "B300"]
    ),
    reason="Test requires datacenter hardware (H100, H200, B100, B200, B300)",
)

_fp8_support_result = check_fp8_support() if torch.cuda.is_available() else (False, "CUDA not available")
requires_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not _fp8_support_result[0],
    reason=f"Test requires FP8 support: {_fp8_support_result[1]}",
)


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def test_sanity_convergence_fsdp2_te_bshd(tmp_path, recipe_path, local_tokenizer_path):
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "config_kwargs.attn_input_format=bshd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"


def test_sanity_convergence_fsdp2_te_thd(tmp_path, recipe_path, local_tokenizer_path):
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "config_kwargs.self_attn_mask_type=padding_causal",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"


def test_sanity_convergence_fsdp2_te_bshd_grad_acc(tmp_path, recipe_path, local_tokenizer_path):
    """Test FSDP2 training with gradient accumulation."""
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "config_kwargs.attn_input_format=bshd",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    # Grad accumulation halves effective optimizer steps, so convergence is weaker
    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"


def test_sanity_convergence_ddp_te(tmp_path, recipe_path, local_tokenizer_path):
    """Test that DDP training converges on sanity-scale data."""
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "config_kwargs.attn_input_format=bshd",
            ],
        )

    final_loss = main_ddp(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"


def test_sanity_convergence_ddp_te_grad_acc(tmp_path, recipe_path, local_tokenizer_path):
    """Test DDP training with gradient accumulation."""
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "config_kwargs.attn_input_format=bshd",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_ddp(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"


def test_sanity_convergence_fsdp2_hf(tmp_path, recipe_path, local_tokenizer_path):
    """Test that FSDP2 training converges with HuggingFace (non-TE) model."""
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "use_te=false",
                "use_meta_device=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"


@requires_fp8
@requires_datacenter_hardware
def test_sanity_convergence_fsdp2_te_fp8(tmp_path, recipe_path, local_tokenizer_path, fp_recipe):
    """Test FSDP2 training with FP8 enabled using parametrized FP8 recipes."""
    tokenizer_path = local_tokenizer_path
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                f"dataset.tokenizer_name_or_path={tokenizer_path}",
                "checkpoint.resume_from_checkpoint=false",
                "num_train_steps=40",
                "config_kwargs.attn_input_format=bshd",
                "fp8_config.enabled=true",
                *fp_recipe,
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    _cleanup()

    assert final_loss < 8.5, f"Final loss {final_loss} is too high, expected < 8.5"
