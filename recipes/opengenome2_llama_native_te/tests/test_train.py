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

import gc
import math
import random

import pytest
import torch
from hydra import compose, initialize_config_dir

from opengenome_modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM
from optimizer import get_parameter_groups_with_weight_decay
from train_fsdp2 import main as main_fsdp2
from train_fsdp2_cp import main as main_fsdp2_cp


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
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_thd(tmp_path, recipe_path):
    """Test that FSDP2 training converges with THD format."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "config_kwargs.attn_input_format=thd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_convergence_fsdp2_te_bshd_grad_acc(tmp_path, recipe_path):
    """Test FSDP2 training with BSHD format and gradient accumulation."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "config_kwargs.attn_input_format=bshd",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


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
                "dataset.max_seq_length=1024",
                "grad_acc_steps=2",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_fsdp2_with_sequence_packing(tmp_path, recipe_path):
    """Test FSDP2 training with THD sequence packing."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
                "dataset.max_seq_length=1024",
                "num_train_steps=10",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"


def test_train_fsdp2_fp8_bshd(tmp_path, recipe_path):
    """Test FSDP2 training with FP8 enabled in BSHD format."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "+dataset.pad_sequences_to_be_divisible_by=16",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_train_fsdp2_fp8_thd(tmp_path, recipe_path):
    """Test FSDP2 training with FP8 enabled in THD format."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "fp8_config.enabled=true",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


# ============================================================================
# Weight initialization tests
# ============================================================================


def _create_tiny_config(**overrides) -> NVLlamaConfig:
    """Create a small NVLlamaConfig for fast init testing."""
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=256,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        initializer_range=0.02,
        attn_input_format="bshd",
    )
    kwargs.update(overrides)
    return NVLlamaConfig(**kwargs)


def test_megatron_scaled_init():
    """Test that Megatron-style scaled init applies reduced std to proj/fc2.

    With use_megatron_scaled_init=True, attention proj and MLP fc2 weights should
    be initialized with std = initializer_range / sqrt(2 * num_layers), while
    QKV/fc1 weights should use the regular initializer_range (0.02).
    """
    std = 0.02
    num_layers = 4
    expected_output_std = std / math.sqrt(2.0 * num_layers)

    config = _create_tiny_config(use_megatron_scaled_init=True)
    model = NVLlamaForCausalLM(config)

    # Collect std values from all layers
    proj_stds = []
    fc2_stds = []
    qkv_stds = []
    fc1_stds = []

    for layer in model.model.layers:
        if hasattr(layer.self_attention, "proj") and layer.self_attention.proj.weight is not None:
            proj_stds.append(layer.self_attention.proj.weight.float().std().item())
        if hasattr(layer, "layernorm_mlp") and hasattr(layer.layernorm_mlp, "fc1_weight"):
            fc1_stds.append(layer.layernorm_mlp.fc1_weight.float().std().item())
        if hasattr(layer, "layernorm_mlp") and hasattr(layer.layernorm_mlp, "fc2_weight"):
            fc2_stds.append(layer.layernorm_mlp.fc2_weight.float().std().item())
        if hasattr(layer.self_attention, "layernorm_qkv"):
            qkv_stds.append(layer.self_attention.layernorm_qkv.weight.float().std().item())

    # proj and fc2 should have scaled std (much smaller than 0.02)
    for i, proj_std in enumerate(proj_stds):
        assert abs(proj_std - expected_output_std) < 0.005, (
            f"Layer {i} proj std={proj_std:.6f}, expected ~{expected_output_std:.6f}"
        )
    for i, fc2_std in enumerate(fc2_stds):
        assert abs(fc2_std - expected_output_std) < 0.005, (
            f"Layer {i} fc2 std={fc2_std:.6f}, expected ~{expected_output_std:.6f}"
        )

    # QKV and fc1 should have regular std (0.02)
    for i, qkv_std in enumerate(qkv_stds):
        assert abs(qkv_std - std) < 0.005, f"Layer {i} QKV std={qkv_std:.6f}, expected ~{std}"
    for i, fc1_std in enumerate(fc1_stds):
        assert abs(fc1_std - std) < 0.005, f"Layer {i} fc1 std={fc1_std:.6f}, expected ~{std}"


def test_regular_init_without_scaled():
    """Test that without scaled init, proj/fc2 use the same std as QKV/fc1."""
    std = 0.02

    config = _create_tiny_config(use_megatron_scaled_init=False)
    model = NVLlamaForCausalLM(config)

    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.self_attention, "proj") and layer.self_attention.proj.weight is not None:
            proj_std = layer.self_attention.proj.weight.float().std().item()
            assert abs(proj_std - std) < 0.005, f"Layer {i} proj std={proj_std:.6f}, expected ~{std}"
        if hasattr(layer, "layernorm_mlp") and hasattr(layer.layernorm_mlp, "fc2_weight"):
            fc2_std = layer.layernorm_mlp.fc2_weight.float().std().item()
            assert abs(fc2_std - std) < 0.005, f"Layer {i} fc2 std={fc2_std:.6f}, expected ~{std}"


def test_spike_no_more_embedding_init():
    """Test that Spike-No-More embedding init uses std=1.0 for embeddings.

    When embedding_init_std=1.0, embed_tokens should have a much larger std than
    the default 0.02, while all other weights should still use initializer_range.
    """
    config = _create_tiny_config(embedding_init_std=1.0)
    model = NVLlamaForCausalLM(config)

    emb_std = model.model.embed_tokens.weight.float().std().item()
    assert abs(emb_std - 1.0) < 0.15, f"Embedding std={emb_std:.4f}, expected ~1.0"

    # QKV should still use regular std
    layer = model.model.layers[0]
    if hasattr(layer.self_attention, "layernorm_qkv"):
        qkv_std = layer.self_attention.layernorm_qkv.weight.float().std().item()
        assert abs(qkv_std - 0.02) < 0.005, f"QKV std={qkv_std:.6f}, expected ~0.02"


def test_default_embedding_init():
    """Test that without embedding_init_std, embeddings use initializer_range (0.02)."""
    config = _create_tiny_config(embedding_init_std=None)
    model = NVLlamaForCausalLM(config)

    emb_std = model.model.embed_tokens.weight.float().std().item()
    assert abs(emb_std - 0.02) < 0.005, f"Embedding std={emb_std:.4f}, expected ~0.02"


def test_weight_decay_grouping():
    """Test that weight decay grouping correctly separates decay/no-decay params.

    Megatron convention: skip weight decay on bias terms and 1D params (LayerNorm/RMSNorm).
    """
    config = _create_tiny_config()
    model = NVLlamaForCausalLM(config)

    param_groups = get_parameter_groups_with_weight_decay(model, weight_decay=0.1)
    decay_group = param_groups[0]
    no_decay_group = param_groups[1]

    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    # Both groups should have parameters
    assert len(decay_group["params"]) > 0, "Decay group should not be empty"
    assert len(no_decay_group["params"]) > 0, "No-decay group should not be empty"

    # All 1D params (LayerNorm/RMSNorm weights) should be in no-decay group
    no_decay_set = {id(p) for p in no_decay_group["params"]}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias"):
            assert id(param) in no_decay_set, f"1D/bias param '{name}' should be in no-decay group"


def test_weight_decay_skip_embeddings():
    """Test that skip_embeddings=True excludes embedding weights from weight decay."""
    config = _create_tiny_config()
    model = NVLlamaForCausalLM(config)

    param_groups = get_parameter_groups_with_weight_decay(model, weight_decay=0.1, skip_embeddings=True)
    no_decay_set = {id(p) for p in param_groups[1]["params"]}

    for name, param in model.named_parameters():
        if "embed" in name.lower() and param.requires_grad:
            assert id(param) in no_decay_set, f"Embedding param '{name}' should be in no-decay group"


def test_scaled_init_with_spike_no_more():
    """Test that scaled init and Spike-No-More can be combined correctly.

    This is the full OG2 production configuration:
    - Embeddings: std=1.0 (Spike-No-More)
    - QKV, fc1: std=0.02 (regular)
    - proj, fc2: std=0.02/sqrt(2*4) ~= 0.00707 (Megatron scaled)
    """
    std = 0.02
    num_layers = 4
    expected_output_std = std / math.sqrt(2.0 * num_layers)

    config = _create_tiny_config(
        use_megatron_scaled_init=True,
        embedding_init_std=1.0,
    )
    model = NVLlamaForCausalLM(config)

    # Embedding should be ~1.0
    emb_std = model.model.embed_tokens.weight.float().std().item()
    assert abs(emb_std - 1.0) < 0.15, f"Embedding std={emb_std:.4f}, expected ~1.0"

    # proj/fc2 should be scaled
    layer = model.model.layers[0]
    if hasattr(layer.self_attention, "proj") and layer.self_attention.proj.weight is not None:
        proj_std = layer.self_attention.proj.weight.float().std().item()
        assert abs(proj_std - expected_output_std) < 0.005, (
            f"proj std={proj_std:.6f}, expected ~{expected_output_std:.6f}"
        )

    # QKV should be regular
    if hasattr(layer.self_attention, "layernorm_qkv"):
        qkv_std = layer.self_attention.layernorm_qkv.weight.float().std().item()
        assert abs(qkv_std - std) < 0.005, f"QKV std={qkv_std:.6f}, expected ~{std}"


def test_train_fsdp2_fp32_master_weights_thd(tmp_path, recipe_path):
    """Test FSDP2 convergence with FP32 master weights and THD sequence packing.

    Uses MixedPrecisionPolicy(cast_forward_inputs=False) which prevents FSDP from
    blanket-casting RoPE embeddings (computed in FP32 via torch.autocast(enabled=False))
    down to BF16.
    """
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "checkpoint.resume_from_checkpoint=false",
                "use_fp32_master_weights=true",
                "use_torch_compile=false",
                "fp8_config.enabled=false",
                "use_sequence_packing=true",
                "config_kwargs.attn_input_format=thd",
            ],
        )

    final_loss = main_fsdp2(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert final_loss < 8.0, f"Final loss {final_loss} is too high, expected < 8.0"


def test_sanity_fsdp2_cp(tmp_path, recipe_path):
    """Test FSDP2 with context parallelism on a single GPU (cp_size=1)."""
    with initialize_config_dir(config_dir=str(recipe_path / "hydra_config"), version_base="1.2"):
        sanity_config = compose(
            config_name="L0_sanity_cp",
            overrides=[
                f"+wandb.dir={tmp_path}",
                f"checkpoint.ckpt_dir={tmp_path}",
                "num_train_steps=10",
                "checkpoint.resume_from_checkpoint=false",
            ],
        )

    final_loss = main_fsdp2_cp(sanity_config)
    gc.collect()
    torch.cuda.empty_cache()

    assert torch.isfinite(torch.tensor(final_loss)), f"Final loss {final_loss} is not finite"
