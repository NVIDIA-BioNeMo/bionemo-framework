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

"""Reference: ESM2 HF<->TE Conversion (Encoder/BERT-like pattern).

This file demonstrates how to convert a HuggingFace encoder model to TransformerEngine.
Key patterns:
- Mapping dict with wildcard layer indices
- QKV packing: separate Q/K/V -> fused interleaved QKV
- Embedding padding for FP8 compatibility
- Bidirectional conversion (HF->TE and TE->HF)
"""

import inspect

# NOTE: These imports are relative - adjust for your project structure
import state
import torch
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM
from torch import nn


# NOTE: The mapping dict renames HF state dict keys to TE keys.
# Use "*" as wildcard for layer indices (e.g., layer.0, layer.1, ...).
# TE uses different naming: layernorm_qkv (fused LN+QKV), layernorm_mlp (fused LN+MLP)
mapping = {
    # Attention output projection
    "esm.encoder.layer.*.attention.output.dense.weight": "esm.encoder.layers.*.self_attention.proj.weight",
    "esm.encoder.layer.*.attention.output.dense.bias": "esm.encoder.layers.*.self_attention.proj.bias",
    # Attention LayerNorm -> fused into TE's layernorm_qkv
    "esm.encoder.layer.*.attention.LayerNorm.weight": "esm.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "esm.encoder.layer.*.attention.LayerNorm.bias": "esm.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_bias",
    # FFN intermediate -> TE's fc1 (first linear in MLP)
    "esm.encoder.layer.*.intermediate.dense.weight": "esm.encoder.layers.*.layernorm_mlp.fc1_weight",
    "esm.encoder.layer.*.intermediate.dense.bias": "esm.encoder.layers.*.layernorm_mlp.fc1_bias",
    # FFN output -> TE's fc2 (second linear in MLP)
    "esm.encoder.layer.*.output.dense.weight": "esm.encoder.layers.*.layernorm_mlp.fc2_weight",
    "esm.encoder.layer.*.output.dense.bias": "esm.encoder.layers.*.layernorm_mlp.fc2_bias",
    # FFN LayerNorm -> fused into TE's layernorm_mlp
    "esm.encoder.layer.*.LayerNorm.weight": "esm.encoder.layers.*.layernorm_mlp.layer_norm_weight",
    "esm.encoder.layer.*.LayerNorm.bias": "esm.encoder.layers.*.layernorm_mlp.layer_norm_bias",
    # Post-encoder LayerNorm (not fused)
    "esm.encoder.emb_layer_norm_after.weight": "esm.encoder.emb_layer_norm_after.weight",
    "esm.encoder.emb_layer_norm_after.bias": "esm.encoder.emb_layer_norm_after.bias",
    # LM head
    "lm_head.dense.weight": "lm_head.dense.weight",
    "lm_head.dense.bias": "lm_head.dense.bias",
    "lm_head.layer_norm.weight": "lm_head.decoder.layer_norm_weight",
    "lm_head.layer_norm.bias": "lm_head.decoder.layer_norm_bias",
}

reverse_mapping = {v: k for k, v in mapping.items()}


def convert_esm_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert HuggingFace ESM2 to TransformerEngine format.

    NOTE: The pattern is:
    1. Create NV config from HF config (pass through all existing fields + add TE fields)
    2. Create empty TE model on meta device (avoids GPU memory for large models)
    3. Apply transforms to copy and reshape weights
    """
    from accelerate import init_empty_weights

    te_config = NVEsmConfig(**model_hf.config.to_dict(), **config_kwargs)
    with init_empty_weights():
        model_te = NVEsmForMaskedLM(te_config)

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [
            _pack_qkv_weight,  # Merge Q/K/V weights into fused QKV
            _pack_qkv_bias,  # Merge Q/K/V biases into fused QKV
            _pad_embeddings,  # Pad embedding matrix for FP8
            _pad_decoder_weights,  # Pad LM head weights
            _pad_bias,  # Pad LM head bias (with -inf for padding positions)
        ],
    )
    return output_model


def convert_esm_te_to_hf(model_te: nn.Module, **config_kwargs) -> nn.Module:
    """Convert TE format back to HuggingFace format.

    NOTE: Filter out TE-specific config keys that aren't valid for the original config class.
    """
    from accelerate import init_empty_weights
    from transformers import EsmConfig, EsmForMaskedLM

    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(EsmConfig.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = EsmConfig(**filtered_config, **config_kwargs)

    with init_empty_weights():
        model_hf = EsmForMaskedLM(hf_config)

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [_unpack_qkv_weight, _unpack_qkv_bias, _unpad_embeddings, _unpad_decoder_weights, _unpad_bias],
        state_dict_ignored_entries=["lm_head.decoder.weight"],  # Tied weight
    )
    output_model.post_init()
    return output_model


# NOTE: QKV packing for MHA (Multi-Head Attention) uses interleaved format.
# For each head, Q/K/V weights are interleaved: [h0_q, h0_k, h0_v, h1_q, h1_k, h1_v, ...]
# This is required when qkv_weight_interleaved=True in TE.
@state.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="esm.encoder.layers.*.self_attention.layernorm_qkv.weight",
)
def _pack_qkv_weight(ctx, query, key, value):
    """Pack separate Q, K, V weights into interleaved QKV format."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    num_heads = ctx.target.config.num_attention_heads
    concat_weights = concat_weights.view(3, num_heads, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    return concat_weights.view(*input_shape)


@state.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="esm.encoder.layers.*.self_attention.layernorm_qkv.bias",
)
def _pack_qkv_bias(ctx, query, key, value):
    """Pack separate Q, K, V biases into interleaved QKV format."""
    concat_biases = torch.cat((query, key, value), dim=0)
    input_shape = concat_biases.size()
    num_heads = ctx.target.config.num_attention_heads
    concat_biases = concat_biases.view(3, num_heads, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    return concat_biases.view(*input_shape)


# NOTE: Reverse transforms for TE->HF conversion
@state.state_transform(
    source_key="esm.encoder.layers.*.self_attention.layernorm_qkv.weight",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
)
def _unpack_qkv_weight(ctx, qkv_weight):
    """Unpack fused QKV weights back to separate Q, K, V."""
    num_heads = ctx.source.config.num_attention_heads
    total_rows, input_dim = qkv_weight.size()
    head_dim = total_rows // (3 * num_heads)
    qkv_weight = qkv_weight.view(num_heads, 3, head_dim, input_dim).transpose(0, 1).contiguous()
    return (
        qkv_weight[0].reshape(-1, input_dim),
        qkv_weight[1].reshape(-1, input_dim),
        qkv_weight[2].reshape(-1, input_dim),
    )


@state.state_transform(
    source_key="esm.encoder.layers.*.self_attention.layernorm_qkv.bias",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
)
def _unpack_qkv_bias(ctx, qkv_bias):
    """Unpack fused QKV biases back to separate Q, K, V."""
    num_heads = ctx.source.config.num_attention_heads
    total_size = qkv_bias.size(0)
    head_dim = total_size // (3 * num_heads)
    qkv_bias = qkv_bias.view(num_heads, 3, head_dim).transpose(0, 1).contiguous()
    return qkv_bias[0].reshape(-1), qkv_bias[1].reshape(-1), qkv_bias[2].reshape(-1)


# NOTE: Embedding padding -- pad vocab to padded_vocab_size for FP8 compatibility.
# For embeddings: pad with zeros. For bias: pad with -inf (so softmax ignores padding).
def _pad_weights(ctx, source_embed):
    target_dim = ctx.target.config.padded_vocab_size
    num_padding = target_dim - source_embed.size(0)
    padding = torch.zeros(num_padding, source_embed.size(1), dtype=source_embed.dtype, device=source_embed.device)
    return torch.cat((source_embed, padding), dim=0)


def _unpad_weights(ctx, padded_embed):
    return padded_embed[: ctx.target.config.vocab_size]


_pad_embeddings = state.state_transform(
    "esm.embeddings.word_embeddings.weight", "esm.embeddings.word_embeddings.weight"
)(_pad_weights)
_pad_decoder_weights = state.state_transform("lm_head.decoder.weight", "lm_head.decoder.weight")(_pad_weights)
_unpad_embeddings = state.state_transform(
    "esm.embeddings.word_embeddings.weight", "esm.embeddings.word_embeddings.weight"
)(_unpad_weights)
_unpad_decoder_weights = state.state_transform("lm_head.decoder.weight", "lm_head.decoder.weight")(_unpad_weights)


@state.state_transform(source_key="lm_head.bias", target_key="lm_head.decoder.bias")
def _pad_bias(ctx, source_bias):
    """Pad bias with -inf so padded positions produce ~0 probability after softmax."""
    target_dim = ctx.target.config.padded_vocab_size
    output_bias = torch.finfo(source_bias.dtype).min * torch.ones(
        target_dim, dtype=source_bias.dtype, device=source_bias.device
    )
    output_bias[: source_bias.size(0)] = source_bias
    return output_bias


@state.state_transform(source_key="lm_head.decoder.bias", target_key="lm_head.bias")
def _unpad_bias(ctx, padded_bias):
    return padded_bias[: ctx.target.config.vocab_size]
