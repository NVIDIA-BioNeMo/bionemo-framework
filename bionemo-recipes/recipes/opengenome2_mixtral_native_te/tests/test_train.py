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
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM
from optimizer import get_parameter_groups_with_weight_decay
from train_fsdp2 import main as main_fsdp2


@pytest.fixture(autouse=True)
def set_seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


def test_sanity_convergence_fsdp2_te_bshd(tmp_path, recipe_path):
    """Test that FSDP2 training converges with BSHD format."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "config_kwargs.attn_input_format=bshd",
                "use_sequence_packing=false",
                "num_train_steps=80",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_sanity_convergence_fsdp2_te_thd(tmp_path, recipe_path):
    """Test that FSDP2 training converges with THD format."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "config_kwargs.self_attn_mask_type=padding_causal",
                "dataset.max_seq_length=256",
                "num_train_steps=80",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_sanity_convergence_fsdp2_te_thd_grad_acc(tmp_path, recipe_path):
    """Test FSDP2 training with THD format and gradient accumulation."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "config_kwargs.self_attn_mask_type=padding_causal",
                "grad_acc_steps=2",
                "num_train_steps=40",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_train_fsdp2_fp32_master_weights_thd(tmp_path, recipe_path):
    """Test FSDP2 convergence with FP32 master weights and THD sequence packing."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_fp32_master_weights=true",
                "fp8_config.enabled=false",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "config_kwargs.self_attn_mask_type=padding_causal",
                "num_train_steps=40",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def _create_tiny_config(**overrides) -> NVMixtralConfig:
    """Create a small Mixtral config for fast grouping tests."""
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
    )
    kwargs.update(overrides)
    return NVMixtralConfig(**kwargs)


def test_weight_decay_grouping():
    """Test that weight decay grouping correctly separates decay and no-decay params."""
    model = NVMixtralForCausalLM(_create_tiny_config())

    param_groups = get_parameter_groups_with_weight_decay(model, weight_decay=0.1)
    decay_group = param_groups[0]
    no_decay_group = param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0
    assert len(decay_group["params"]) > 0
    assert len(no_decay_group["params"]) > 0

    no_decay_set = {id(p) for p in no_decay_group["params"]}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias"):
            assert id(param) in no_decay_set, f"1D/bias param '{name}' should be in no-decay group"


def test_weight_decay_skip_embeddings():
    """Test that skip_embeddings=True excludes embedding weights from weight decay."""
    model = NVMixtralForCausalLM(_create_tiny_config())

    param_groups = get_parameter_groups_with_weight_decay(model, weight_decay=0.1, skip_embeddings=True)
    no_decay_set = {id(p) for p in param_groups[1]["params"]}

    for name, param in model.named_parameters():
        if "embed" in name.lower() and param.requires_grad:
            assert id(param) in no_decay_set, f"Embedding param '{name}' should be in no-decay group"
