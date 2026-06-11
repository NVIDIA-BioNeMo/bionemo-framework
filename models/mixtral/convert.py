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

"""Conversion utilities between HuggingFace Mixtral and TransformerEngine formats."""

import inspect

import torch
from transformers import MixtralConfig, MixtralForCausalLM

import state
from modeling_mixtral_te import NVMixtralConfig, NVMixtralForCausalLM


mapping = {
    "model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.layers.*.input_layernorm.weight": "model.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "model.layers.*.self_attn.o_proj.weight": "model.layers.*.self_attention.proj.weight",
    "model.layers.*.post_attention_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
    "model.layers.*.mlp.gate.weight": "model.layers.*.mlp.gate.weight",
    "model.layers.*.mlp.experts.gate_up_proj": "model.layers.*.mlp.experts_gate_up_weight",
    "model.layers.*.mlp.experts.down_proj": "model.layers.*.mlp.experts_down_weight",
    "model.norm.weight": "model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

reverse_mapping = {v: k for k, v in mapping.items()}


def convert_mixtral_hf_to_te(model_hf: MixtralForCausalLM, **config_kwargs) -> NVMixtralForCausalLM:
    """Convert a Hugging Face Mixtral model to a Transformer Engine model.

    Args:
        model_hf: The Hugging Face Mixtral model.
        **config_kwargs: Additional configuration kwargs to be passed to NVMixtralConfig.

    Returns:
        The Transformer Engine Mixtral model.
    """
    te_config = NVMixtralConfig(**model_hf.config.to_dict(), **config_kwargs)
    with torch.device("meta"):
        model_te = NVMixtralForCausalLM(te_config)

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
                fn=state.TransformFns.merge_qkv,
            ),
        ],
    )

    output_model.model.rotary_emb.inv_freq = model_hf.model.rotary_emb.inv_freq.clone()

    return output_model


def convert_mixtral_te_to_hf(model_te: NVMixtralForCausalLM, **config_kwargs) -> MixtralForCausalLM:
    """Convert a Transformer Engine Mixtral model to a Hugging Face model.

    Args:
        model_te: The Transformer Engine Mixtral model.
        **config_kwargs: Additional configuration kwargs to be passed to MixtralConfig.

    Returns:
        The Hugging Face Mixtral model.
    """
    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(MixtralConfig.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = MixtralConfig(**filtered_config, **config_kwargs)

    with torch.device("meta"):
        model_hf = MixtralForCausalLM(hf_config)

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
                fn=state.TransformFns.split_qkv,
            ),
        ],
    )

    output_model.model.rotary_emb.inv_freq = model_te.model.rotary_emb.inv_freq.clone()
    output_model.tie_weights()

    return output_model
