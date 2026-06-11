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

"""Conversion utilities between HuggingFace Qwen2 and TransformerEngine formats."""

import inspect

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

import state
from modeling_qwen2_te import NVQwen2Config, NVQwen2ForCausalLM


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.layernorm_mlp.layer_norm_weight",
    "model.layers.*.mlp.down_proj.weight": "model.layers.*.layernorm_mlp.fc2_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Reverse mapping from TE to HF format by reversing the original mapping
reverse_mapping = {v: k for k, v in mapping.items()}


def _merge_qkv_bias(ctx: state.TransformCTX, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Merge separate q, k, v biases into interleave-concatenated qkv bias."""
    target_config = ctx.target.config

    head_num = target_config.num_attention_heads
    num_query_groups = target_config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    head_size = target_config.hidden_size // head_num

    q = q.view(head_num, head_size)
    k = k.view(num_query_groups, head_size)
    v = v.view(num_query_groups, head_size)

    qkv_bias_l = []
    for i in range(num_query_groups):
        qkv_bias_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :])
        qkv_bias_l.append(k[i : i + 1, :])
        qkv_bias_l.append(v[i : i + 1, :])
    qkv_bias = torch.cat(qkv_bias_l)

    return qkv_bias.reshape(-1)


def _split_qkv_bias(ctx: state.TransformCTX, qkv_bias: torch.Tensor):
    """Split interleave-concatenated qkv bias into separate q, k, v biases."""
    target_config = ctx.target.config

    head_num = target_config.num_attention_heads
    num_query_groups = target_config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    head_size = target_config.hidden_size // head_num
    qkv_total_dim = head_num + 2 * num_query_groups

    qkv_bias = qkv_bias.reshape(qkv_total_dim, head_size)
    q_slice = torch.cat(
        [
            torch.arange(
                (heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group, device=qkv_bias.device
            )
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2), device=qkv_bias.device)
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2), device=qkv_bias.device)

    q_bias = qkv_bias[q_slice].reshape(-1)
    k_bias = qkv_bias[k_slice].reshape(-1)
    v_bias = qkv_bias[v_slice].reshape(-1)

    return q_bias, k_bias, v_bias


def _zero_bias_from_weight(ctx: state.TransformCTX, weight: torch.Tensor):
    """Create a zero bias with dimension matching the weight's first axis."""
    return torch.zeros(weight.shape[0], device=weight.device, dtype=weight.dtype)


def _zero_fc1_bias(ctx: state.TransformCTX, gate: torch.Tensor, up: torch.Tensor):
    """Create a zero fc1 bias for the merged gate+up projection."""
    return torch.zeros(gate.shape[0] + up.shape[0], device=gate.device, dtype=gate.dtype)


def convert_qwen2_hf_to_te(model_hf: Qwen2ForCausalLM, **config_kwargs) -> NVQwen2ForCausalLM:
    """Convert a Hugging Face Qwen2 model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVQwen2Config.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    config_dict = model_hf.config.to_dict()
    # Ensure layer_types is consistent with num_hidden_layers (from_pretrained can leave stale layer_types)
    if len(config_dict.get("layer_types", [])) != config_dict.get("num_hidden_layers", 0):
        config_dict["layer_types"] = config_dict["layer_types"][: config_dict["num_hidden_layers"]]
    te_config = NVQwen2Config(**config_dict, **config_kwargs)
    with torch.device("meta"):
        model_te = NVQwen2ForCausalLM(te_config)

    if model_hf.config.tie_word_embeddings:
        state_dict_ignored_entries = ["lm_head.weight"]
    else:
        state_dict_ignored_entries = []

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [
            # Merge Q/K/V weights into fused QKV
            state.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="model.layers.*.self_attention.layernorm_qkv.weight",
                fn=state.TransformFns.merge_qkv,
            ),
            # Merge Q/K/V biases into fused QKV bias
            state.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                target_key="model.layers.*.self_attention.layernorm_qkv.bias",
                fn=_merge_qkv_bias,
            ),
            # Merge gate/up projections into fc1
            state.state_transform(
                source_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                target_key="model.layers.*.layernorm_mlp.fc1_weight",
                fn=state.TransformFns.merge_fc1,
            ),
            # TE bias=True creates biases for all linear layers, but Qwen2 only has bias on QKV.
            # Initialize the extra TE biases (output projection, MLP) to zero.
            state.state_transform(
                source_key="model.layers.*.self_attn.o_proj.weight",
                target_key="model.layers.*.self_attention.proj.bias",
                fn=_zero_bias_from_weight,
            ),
            state.state_transform(
                source_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                target_key="model.layers.*.layernorm_mlp.fc1_bias",
                fn=_zero_fc1_bias,
            ),
            state.state_transform(
                source_key="model.layers.*.mlp.down_proj.weight",
                target_key="model.layers.*.layernorm_mlp.fc2_bias",
                fn=_zero_bias_from_weight,
            ),
        ],
        state_dict_ignored_entries=state_dict_ignored_entries,
    )

    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()

    return output_model


def convert_qwen2_te_to_hf(model_te: NVQwen2ForCausalLM, **config_kwargs) -> Qwen2ForCausalLM:
    """Convert a Transformer Engine Qwen2 model to a Hugging Face model.

    Args:
        model_te (nn.Module): The Transformer Engine model.
        **config_kwargs: Additional configuration kwargs to be passed to Qwen2Config.

    Returns:
        nn.Module: The Hugging Face model.
    """
    # Filter out keys from model_te.config that are not valid Qwen2Config attributes
    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(Qwen2Config.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    # Ensure layer_types is consistent with num_hidden_layers
    if len(filtered_config.get("layer_types", [])) != filtered_config.get("num_hidden_layers", 0):
        filtered_config["layer_types"] = filtered_config["layer_types"][: filtered_config["num_hidden_layers"]]
    hf_config = Qwen2Config(**filtered_config, **config_kwargs)

    with torch.device("meta"):
        model_hf = Qwen2ForCausalLM(hf_config)

    if model_hf.config.tie_word_embeddings:
        state_dict_ignored_entries = model_hf._tied_weights_keys
    else:
        state_dict_ignored_entries = []

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [
            # Split fused QKV weight into separate Q/K/V
            state.state_transform(
                source_key="model.layers.*.self_attention.layernorm_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=state.TransformFns.split_qkv,
            ),
            # Split fused QKV bias into separate Q/K/V biases
            state.state_transform(
                source_key="model.layers.*.self_attention.layernorm_qkv.bias",
                target_key=(
                    "model.layers.*.self_attn.q_proj.bias",
                    "model.layers.*.self_attn.k_proj.bias",
                    "model.layers.*.self_attn.v_proj.bias",
                ),
                fn=_split_qkv_bias,
            ),
            # Split fc1 into gate/up projections
            state.state_transform(
                source_key="model.layers.*.layernorm_mlp.fc1_weight",
                target_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                fn=state.TransformFns.split_fc1,
            ),
        ],
        state_dict_ignored_entries=state_dict_ignored_entries,
    )

    output_model.model.rotary_emb.inv_freq = model_te.model.rotary_emb.inv_freq.clone()

    if model_hf.config.tie_word_embeddings:
        output_model.tie_weights()

    return output_model
