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

"""Conversion utilities between HuggingFace Qwen3 and TransformerEngine formats."""

import inspect

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

import state
from modeling_qwen3_te import NVQwen3Config, NVQwen3ForCausalLM


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.self_attn.q_norm.weight": "model.layers.*.self_attention.q_norm.weight",
    "model.layers.*.self_attn.k_norm.weight": "model.layers.*.self_attention.k_norm.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.layernorm_mlp.layer_norm_weight",
    "model.layers.*.mlp.down_proj.weight": "model.layers.*.layernorm_mlp.fc2_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

# Reverse mapping from TE to HF format by reversing the original mapping
reverse_mapping = {v: k for k, v in mapping.items()}


def _merge_qkv(ctx: state.TransformCTX, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """Merge q, k, v to interleave-concatenated qkv.

    This version uses config.head_dim instead of hidden_size // num_attention_heads,
    which is necessary for Qwen3 where head_dim is independently configured.
    """
    target_config = ctx.target.config

    head_num = target_config.num_attention_heads
    num_query_groups = target_config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    hidden_size = target_config.hidden_size
    head_size = target_config.head_dim

    old_tensor_shape = q.size()
    new_q_tensor_shape = (head_num, head_size, *old_tensor_shape[1:])
    new_kv_tensor_shape = (num_query_groups, head_size, *old_tensor_shape[1:])

    q = q.view(*new_q_tensor_shape)
    k = k.view(*new_kv_tensor_shape)
    v = v.view(*new_kv_tensor_shape)

    qkv_weights_l = []
    for i in range(num_query_groups):
        qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
        qkv_weights_l.append(k[i : i + 1, :, :])
        qkv_weights_l.append(v[i : i + 1, :, :])
    qkv_weights = torch.cat(qkv_weights_l)
    assert qkv_weights.ndim == 3, qkv_weights.shape
    assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
    assert qkv_weights.shape[1] == head_size, qkv_weights.shape
    assert qkv_weights.shape[2] == old_tensor_shape[1], qkv_weights.shape

    qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])

    return qkv_weights


def _split_qkv(ctx: state.TransformCTX, linear_qkv: torch.Tensor):
    """Split interleave-concatenated qkv to q, k, v.

    This version uses config.head_dim instead of hidden_size // num_attention_heads,
    which is necessary for Qwen3 where head_dim is independently configured.
    """
    target_config = ctx.target.config

    head_num = target_config.num_attention_heads
    num_query_groups = target_config.num_key_value_heads
    heads_per_group = head_num // num_query_groups
    head_size = target_config.head_dim
    qkv_total_dim = head_num + 2 * num_query_groups

    linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
    hidden_size = linear_qkv.size(-1)
    q_slice = torch.cat(
        [
            torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
            for i in range(num_query_groups)
        ]
    )
    k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
    v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

    q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
    k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
    v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

    return q_proj, k_proj, v_proj


def convert_qwen3_hf_to_te(model_hf: Qwen3ForCausalLM, **config_kwargs) -> NVQwen3ForCausalLM:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVQwen3Config.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    te_config = NVQwen3Config(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVQwen3ForCausalLM(te_config)

    if model_hf.config.tie_word_embeddings:
        state_dict_ignored_entries = ["lm_head.weight"]
    else:
        state_dict_ignored_entries = []

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [
            state.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="model.layers.*.self_attention.layernorm_qkv.weight",
                fn=_merge_qkv,
            ),
            state.state_transform(
                source_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                target_key="model.layers.*.layernorm_mlp.fc1_weight",
                fn=state.TransformFns.merge_fc1,
            ),
        ],
        state_dict_ignored_entries=state_dict_ignored_entries,
    )

    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()

    return output_model


def convert_qwen3_te_to_hf(model_te: NVQwen3ForCausalLM, **config_kwargs) -> Qwen3ForCausalLM:
    """Convert a Transformer Engine model to a Hugging Face model.

    Args:
        model_te (nn.Module): The Transformer Engine model.
        **config_kwargs: Additional configuration kwargs to be passed to Qwen3Config.

    Returns:
        nn.Module: The Hugging Face model.
    """
    # Filter out keys from model_te.config that are not valid Qwen3Config attributes
    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(Qwen3Config.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = Qwen3Config(**filtered_config, **config_kwargs)

    with torch.device("meta"):
        model_hf = Qwen3ForCausalLM(hf_config)

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [
            state.state_transform(
                source_key="model.layers.*.self_attention.layernorm_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=_split_qkv,
            ),
            state.state_transform(
                source_key="model.layers.*.layernorm_mlp.fc1_weight",
                target_key=(
                    "model.layers.*.mlp.gate_proj.weight",
                    "model.layers.*.mlp.up_proj.weight",
                ),
                fn=state.TransformFns.split_fc1,
            ),
        ],
        state_dict_ignored_entries=model_hf._tied_weights_keys,
    )

    output_model.model.rotary_emb.inv_freq = model_te.model.rotary_emb.inv_freq.clone()
    output_model.tie_weights()

    return output_model
