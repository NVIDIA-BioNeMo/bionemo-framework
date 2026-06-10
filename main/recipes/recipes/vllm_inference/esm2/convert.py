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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: bionemo-recipes/models/esm2/convert.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---

"""Conversion utilities between HuggingFace ESM2 and TransformerEngine formats."""

import inspect

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import EsmConfig, EsmForMaskedLM

import state
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM


mapping = {
    "esm.encoder.layer.*.attention.output.dense.weight": "model.encoder.layers.*.self_attention.proj.weight",
    "esm.encoder.layer.*.attention.output.dense.bias": "model.encoder.layers.*.self_attention.proj.bias",
    "esm.encoder.layer.*.attention.LayerNorm.weight": "model.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_weight",
    "esm.encoder.layer.*.attention.LayerNorm.bias": "model.encoder.layers.*.self_attention.layernorm_qkv.layer_norm_bias",
    "esm.encoder.layer.*.intermediate.dense.weight": "model.encoder.layers.*.layernorm_mlp.fc1_weight",
    "esm.encoder.layer.*.intermediate.dense.bias": "model.encoder.layers.*.layernorm_mlp.fc1_bias",
    "esm.encoder.layer.*.output.dense.weight": "model.encoder.layers.*.layernorm_mlp.fc2_weight",
    "esm.encoder.layer.*.output.dense.bias": "model.encoder.layers.*.layernorm_mlp.fc2_bias",
    "esm.encoder.layer.*.LayerNorm.weight": "model.encoder.layers.*.layernorm_mlp.layer_norm_weight",
    "esm.encoder.layer.*.LayerNorm.bias": "model.encoder.layers.*.layernorm_mlp.layer_norm_bias",
    "esm.encoder.emb_layer_norm_after.weight": "model.encoder.emb_layer_norm_after.weight",
    "esm.encoder.emb_layer_norm_after.bias": "model.encoder.emb_layer_norm_after.bias",
    "lm_head.dense.weight": "lm_head.dense.weight",
    "lm_head.dense.bias": "lm_head.dense.bias",
    "lm_head.layer_norm.weight": "lm_head.decoder.layer_norm_weight",
    "lm_head.layer_norm.bias": "lm_head.decoder.layer_norm_bias",
}

# Reverse mapping from TE to HF format by reversing the original mapping
reverse_mapping = {v: k for k, v in mapping.items()}


def convert_esm_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to NVEsmConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    # TODO (peter): this is super similar method to the AMPLIFY one, maybe we can abstract or keep simlar naming? models/amplify/src/amplify/state_dict_convert.py:convert_amplify_hf_to_te
    te_config = NVEsmConfig(**model_hf.config.to_dict(), **config_kwargs)
    with init_empty_weights():
        model_te = NVEsmForMaskedLM(te_config)

    output_model = state.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [
            _pack_qkv_weight,
            _pack_qkv_bias,
            _pad_embeddings,
            _pad_decoder_weights,
            _pad_bias,
        ],
    )

    return output_model


def convert_esm_te_to_hf(model_te: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Transformer Engine model back to the original HuggingFace Facebook ESM-2 format.

    This function converts from the NVIDIA Transformer Engine (TE) format back to the
    weight format compatible with the original facebook/esm2_* series of checkpoints.
    The TE model is also a HuggingFace model, but this conversion ensures compatibility
    with the original Facebook ESM-2 model architecture and weight format hosted on Hugging Face.

    Args:
        model_te (nn.Module): The Transformer Engine model.
        **config_kwargs: Additional configuration kwargs to be passed to EsmConfig.

    Returns:
        nn.Module: The Hugging Face model in original Facebook ESM-2 format hosted on Hugging Face.
    """
    # Convert TE config to HF config, filtering out TE-specific keys
    te_config_dict = model_te.config.to_dict()
    valid_keys = set(inspect.signature(EsmConfig.__init__).parameters)
    filtered_config = {k: v for k, v in te_config_dict.items() if k in valid_keys}
    hf_config = EsmConfig(**filtered_config, **config_kwargs)

    with init_empty_weights():
        model_hf = EsmForMaskedLM(hf_config)

        # Remove contact_head since it's not present in TE models
        if hasattr(model_hf.esm, "contact_head"):
            delattr(model_hf.esm, "contact_head")

    output_model = state.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [_unpack_qkv_weight, _unpack_qkv_bias, _unpad_embeddings, _unpad_decoder_weights, _unpad_bias],
        state_dict_ignored_entries=[
            "lm_head.decoder.weight",
            "esm.contact_head.regression.weight",
            "esm.contact_head.regression.bias",
        ],
    )

    output_model.post_init()

    # Note: contact_head parameters are not preserved in TE models
    # They are lost during HF -> TE conversion and cannot be recovered
    # The converted model will not have the original contact_head weights

    return output_model


@state.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="model.encoder.layers.*.self_attention.layernorm_qkv.weight",
)
def _pack_qkv_weight(ctx: state.TransformCTX, query, key, value):
    """Pack separate Q, K, V weight tensors into a single interleaved QKV weight tensor."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    num_heads = ctx.target.config.num_attention_heads
    # transpose weights
    # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
    # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
    concat_weights = concat_weights.view(3, num_heads, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    concat_weights = concat_weights.view(*input_shape)
    return concat_weights


@state.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="model.encoder.layers.*.self_attention.layernorm_qkv.bias",
)
def _pack_qkv_bias(ctx: state.TransformCTX, query, key, value):
    """Pack separate Q, K, V bias tensors into a single interleaved QKV bias tensor."""
    concat_biases = torch.cat((query, key, value), dim=0)
    input_shape = concat_biases.size()
    num_heads = ctx.target.config.num_attention_heads
    # transpose biases
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    concat_biases = concat_biases.view(3, num_heads, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    concat_biases = concat_biases.view(*input_shape)
    return concat_biases


@state.state_transform(
    source_key="model.encoder.layers.*.self_attention.layernorm_qkv.weight",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
)
def _unpack_qkv_weight(ctx: state.TransformCTX, qkv_weight):
    """Unpack fused QKV weights into separate [hidden_size, input_dim] tensors for query/key/value."""
    num_heads = ctx.source.config.num_attention_heads
    total_rows, input_dim = qkv_weight.size()  # size: [num_heads * 3 *head_dim, input_dim]
    assert total_rows % (3 * num_heads) == 0, (
        f"QKV weight rows {total_rows} not divisible by 3*num_heads {3 * num_heads}"
    )
    head_dim = total_rows // (3 * num_heads)

    qkv_weight = (
        qkv_weight.view(num_heads, 3, head_dim, input_dim).transpose(0, 1).contiguous()
    )  # size: [3, num_heads, head_dim, input_dim]
    query, key, value = qkv_weight[0], qkv_weight[1], qkv_weight[2]  # size: [num_heads, head_dim, input_dim]

    query = query.reshape(-1, input_dim)  # size: [num_heads * head_dim, input_dim]
    key = key.reshape(-1, input_dim)  # size: [num_heads * head_dim, input_dim]
    value = value.reshape(-1, input_dim)  # size: [num_heads * head_dim, input_dim]

    return query, key, value


@state.state_transform(
    source_key="model.encoder.layers.*.self_attention.layernorm_qkv.bias",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
)
def _unpack_qkv_bias(ctx: state.TransformCTX, qkv_bias):
    """Unpack fused QKV biases into separate [hidden_size] tensors for query/key/value."""
    num_heads = ctx.source.config.num_attention_heads
    total_size = qkv_bias.size(0)  # size: [num_heads * 3 * head_dim]
    assert total_size % (3 * num_heads) == 0, (
        f"QKV bias size {total_size} not divisible by 3*num_heads {3 * num_heads}"
    )
    head_dim = total_size // (3 * num_heads)

    qkv_bias = qkv_bias.view(num_heads, 3, head_dim).transpose(0, 1).contiguous()  # size: [3, num_heads, head_dim]
    query, key, value = qkv_bias[0], qkv_bias[1], qkv_bias[2]  # size: [num_heads, head_dim]

    query = query.reshape(-1)  # size: [num_heads * head_dim]
    key = key.reshape(-1)  # size: [num_heads * head_dim]
    value = value.reshape(-1)  # size: [num_heads * head_dim]

    return query, key, value


def _unpad_weights(ctx: state.TransformCTX, padded_embed):
    """Remove padding from the embedding layer to get back to the original dimension."""
    target_embedding_dimension = ctx.target.config.vocab_size
    return padded_embed[:target_embedding_dimension]


def _pad_weights(ctx: state.TransformCTX, source_embed):
    """Pad the embedding layer to the new input dimension."""
    target_embedding_dimension = ctx.target.config.padded_vocab_size
    hf_embedding_dimension = source_embed.size(0)
    num_padding_rows = target_embedding_dimension - hf_embedding_dimension
    padding_rows = torch.zeros(
        num_padding_rows, source_embed.size(1), dtype=source_embed.dtype, device=source_embed.device
    )
    return torch.cat((source_embed, padding_rows), dim=0)


_pad_embeddings = state.state_transform(
    source_key="esm.embeddings.word_embeddings.weight",
    target_key="model.embeddings.word_embeddings.weight",
)(_pad_weights)

_pad_decoder_weights = state.state_transform(
    source_key="lm_head.decoder.weight",
    target_key="lm_head.decoder.weight",
)(_pad_weights)

_unpad_embeddings = state.state_transform(
    source_key="model.embeddings.word_embeddings.weight",
    target_key="esm.embeddings.word_embeddings.weight",
)(_unpad_weights)

_unpad_decoder_weights = state.state_transform(
    source_key="lm_head.decoder.weight",
    target_key="lm_head.decoder.weight",
)(_unpad_weights)


@state.state_transform(
    source_key="lm_head.bias",
    target_key="lm_head.decoder.bias",
)
def _pad_bias(ctx: state.TransformCTX, source_bias):
    """Pad the embedding layer to the new input dimension."""
    target_embedding_dimension = ctx.target.config.padded_vocab_size
    hf_embedding_dimension = source_bias.size(0)
    output_bias = torch.finfo(source_bias.dtype).min * torch.ones(
        target_embedding_dimension, dtype=source_bias.dtype, device=source_bias.device
    )
    output_bias[:hf_embedding_dimension] = source_bias
    return output_bias


@state.state_transform(
    source_key="lm_head.decoder.bias",
    target_key="lm_head.bias",
)
def _unpad_bias(ctx: state.TransformCTX, padded_bias):
    """Remove padding from the bias to get back to the original dimension."""
    target_embedding_dimension = ctx.target.config.vocab_size
    return padded_bias[:target_embedding_dimension]
