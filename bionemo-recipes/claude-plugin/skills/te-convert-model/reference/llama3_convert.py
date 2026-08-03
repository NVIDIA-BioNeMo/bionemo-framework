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

"""Reference: Llama3 HF<->TE Conversion (Decoder/Causal LM pattern).

This file demonstrates decoder model conversion with:
- GQA (Group Query Attention) -> fused QKV with TransformFns.merge_qkv
- SwiGLU FFN (gate+up projections) -> fused fc1 with TransformFns.merge_fc1
- Tied word embeddings handling
- Rotary embedding preservation
"""

import inspect

# NOTE: For decoder models, the mapping is simpler because TE's TransformerLayer
# handles more of the fusion internally. The key difference from encoder models:
# - Q/K/V are fused using TransformFns.merge_qkv (handles GQA properly)
# - Gate+Up projections are fused using TransformFns.merge_fc1
import state
import torch


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    # NOTE: input_layernorm -> layernorm_qkv.layer_norm_weight (fused into attention)
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    # Output projection stays as-is
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    # NOTE: post_attention_layernorm -> layernorm_mlp.layer_norm_weight (fused into MLP)
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.layernorm_mlp.layer_norm_weight",
    # down_proj -> fc2 (second linear in MLP)
    "model.layers.*.mlp.down_proj.weight": "model.layers.*.layernorm_mlp.fc2_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

reverse_mapping = {v: k for k, v in mapping.items()}


def convert_llama_hf_to_te(model_hf, **config_kwargs):
    """Convert HuggingFace Llama to TransformerEngine.

    NOTE: Key differences from encoder conversion:
    - Uses TransformFns.merge_qkv for GQA-aware Q/K/V fusion
    - Uses TransformFns.merge_fc1 for gate/up projection fusion
    - Handles tied word embeddings (skip lm_head.weight if tied)
    - Copies rotary_emb.inv_freq separately
    """
    from modeling_llama_te import NVLlamaConfig, NVLlamaForCausalLM

    te_config = NVLlamaConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVLlamaForCausalLM(te_config)

    # NOTE: Handle tied embeddings - if tied, skip lm_head.weight in target
    state_dict_ignored_entries = ["lm_head.weight"] if model_hf.config.tie_word_embeddings else []

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [
            # NOTE: TransformFns.merge_qkv handles GQA automatically.
            # It interleaves Q heads with their corresponding K/V heads.
            state.state_transform(
                source_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                target_key="model.layers.*.self_attention.layernorm_qkv.weight",
                fn=state.TransformFns.merge_qkv,
            ),
            # NOTE: For SwiGLU, gate and up projections are concatenated into fc1.
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

    # NOTE: Rotary embeddings are not part of state_dict, copy manually
    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()
    return output_model


def convert_llama_te_to_hf(model_te, **config_kwargs):
    """Convert TE Llama back to HuggingFace format."""
    from transformers import LlamaConfig, LlamaForCausalLM

    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(LlamaConfig.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = LlamaConfig(**filtered_config, **config_kwargs)

    with torch.device("meta"):
        model_hf = LlamaForCausalLM(hf_config)

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [
            # NOTE: split_qkv reverses merge_qkv, handling GQA head grouping
            state.state_transform(
                source_key="model.layers.*.self_attention.layernorm_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=state.TransformFns.split_qkv,
            ),
            # NOTE: split_fc1 reverses merge_fc1 via torch.chunk(2)
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
