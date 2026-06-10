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

"""Conversion utilities for Geneformer to Transformer Engine format.

Adapted from ESM2 conversion pattern using NeMo's io.apply_transforms.
"""

import torch
from accelerate import init_empty_weights
from nemo.lightning import io
from torch import nn
from transformers import BertConfig
from transformers import BertForMaskedLM as HFBertForMaskedLM

from geneformer import BertForMaskedLM, TEBertConfig


# Mapping from HF BERT format to TE format
mapping = {
    "bert.embeddings.word_embeddings.weight": "bert.embeddings.word_embeddings.weight",
    "bert.embeddings.position_embeddings.weight": "bert.embeddings.position_embeddings.weight",
    "bert.embeddings.token_type_embeddings.weight": "bert.embeddings.token_type_embeddings.weight",
    "bert.embeddings.LayerNorm.weight": "bert.embeddings.LayerNorm.weight",
    "bert.embeddings.LayerNorm.bias": "bert.embeddings.LayerNorm.bias",
    # Attention output parameters
    "bert.encoder.layer.*.attention.output.dense.weight": "bert.encoder.layer.*.self_attention.proj.weight",
    "bert.encoder.layer.*.attention.output.dense.bias": "bert.encoder.layer.*.self_attention.proj.bias",
    # Attention LayerNorm
    "bert.encoder.layer.*.attention.output.LayerNorm.weight": "bert.encoder.layer.*.layernorm.weight",
    "bert.encoder.layer.*.attention.output.LayerNorm.bias": "bert.encoder.layer.*.layernorm.bias",
    # MLP parameters (custom TEBertLayer)
    "bert.encoder.layer.*.intermediate.dense.weight": "bert.encoder.layer.*.layernorm_mlp.fc1.weight",
    "bert.encoder.layer.*.intermediate.dense.bias": "bert.encoder.layer.*.layernorm_mlp.fc1.bias",
    "bert.encoder.layer.*.output.dense.weight": "bert.encoder.layer.*.layernorm_mlp.fc2.weight",
    "bert.encoder.layer.*.output.dense.bias": "bert.encoder.layer.*.layernorm_mlp.fc2.bias",
    "bert.encoder.layer.*.output.LayerNorm.weight": "bert.encoder.layer.*.layernorm_mlp.layer_norm.weight",
    "bert.encoder.layer.*.output.LayerNorm.bias": "bert.encoder.layer.*.layernorm_mlp.layer_norm.bias",
    # Classification head parameters
    "cls.predictions.bias": "cls.predictions.bias",
    "cls.predictions.decoder.weight": "cls.predictions.decoder.weight",
    "cls.predictions.decoder.bias": "cls.predictions.decoder.bias",
    "cls.predictions.transform.dense.weight": "cls.predictions.transform.dense.weight",
    "cls.predictions.transform.dense.bias": "cls.predictions.transform.dense.bias",
    "cls.predictions.transform.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
    "cls.predictions.transform.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
}

# Reverse mapping from TE to HF format
reverse_mapping = {v: k for k, v in mapping.items()}


def convert_geneformer_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face Geneformer model to Transformer Engine format.

    This function maps weights from the original BERT-based Geneformer model
    to the TE-optimized version, following the ESM2 conversion pattern.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to TEBertConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    te_config = TEBertConfig(**model_hf.config.to_dict(), **config_kwargs)
    te_config.use_te_layers = True  # Enable TE layers
    te_config.fuse_qkv_params = True  # Enable fused QKV parameters

    with init_empty_weights():
        model_te = BertForMaskedLM(te_config)

    output_model = io.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [_pack_qkv_weight, _pack_qkv_bias],  # Use transforms for fused QKV parameters
        state_dict_ignored_entries=["cls.predictions.decoder.weight", "cls.predictions.decoder.bias"],
    )

    output_model.tie_weights()

    return output_model


def convert_geneformer_te_to_hf(model_te: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Transformer Engine Geneformer model back to Hugging Face format.

    This function converts from the NVIDIA Transformer Engine (TE) format back to the
    weight format compatible with the original BERT-based Geneformer checkpoints.

    Args:
        model_te (nn.Module): The Transformer Engine model.
        **config_kwargs: Additional configuration kwargs to be passed to BertConfig.

    Returns:
        nn.Module: The Hugging Face model in original BERT format.
    """
    # Convert TE config to HF config
    hf_config_dict = model_te.config.to_dict()

    # Remove TE-specific config options
    te_specific_keys = [
        "qkv_weight_interleaved",
        "encoder_activation",
        "attn_input_format",
        "fuse_qkv_params",
        "micro_batch_size",
        "max_seq_length",
        "use_te_layers",
    ]
    for key in te_specific_keys:
        hf_config_dict.pop(key, None)

    hf_config = BertConfig(**hf_config_dict, **config_kwargs)

    with init_empty_weights():
        model_hf = HFBertForMaskedLM(hf_config)

    # Create a list of _extra_state entries to ignore
    extra_state_entries = []
    for i in range(model_te.config.num_hidden_layers):
        extra_state_entries.extend(
            [
                # Custom TEBertLayer with individual TE components
                f"bert.encoder.layer.{i}.layernorm_mlp.fc1._extra_state",
                f"bert.encoder.layer.{i}.layernorm_mlp.fc2._extra_state",
                f"bert.encoder.layer.{i}.self_attention.core_attention._extra_state",
                f"bert.encoder.layer.{i}.self_attention.qkv._extra_state",
                f"bert.encoder.layer.{i}.self_attention.proj._extra_state",
            ]
        )

    output_model = io.apply_transforms(
        model_te,
        model_hf,
        reverse_mapping,
        [_unpack_qkv_weight, _unpack_qkv_bias],
        state_dict_ignored_entries=[
            "cls.predictions.decoder.weight",
            "cls.predictions.decoder.bias",
            *extra_state_entries,
        ],
    )

    output_model.tie_weights()

    return output_model


@io.state_transform(
    source_key=(
        "bert.encoder.layer.*.attention.self.query.weight",
        "bert.encoder.layer.*.attention.self.key.weight",
        "bert.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="bert.encoder.layer.*.self_attention.qkv.weight",
)
def _pack_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pack the QKV weights into fused TE format."""
    concat_weights = torch.cat((query, key, value), dim=0)  # [3 * num_heads * head_dim, input_dim]
    input_shape = concat_weights.size()
    np = ctx.target.config.num_attention_heads
    concat_weights = concat_weights.view(3, np, -1, query.size()[-1])  # [3, num_heads, head_dim, input_dim]
    concat_weights = concat_weights.transpose(0, 1).contiguous()  # [num_heads, 3, head_dim, input_dim]
    concat_weights = concat_weights.view(*input_shape)  # [3 * num_heads * head_dim, input_dim]

    return concat_weights


@io.state_transform(
    source_key=(
        "bert.encoder.layer.*.attention.self.query.bias",
        "bert.encoder.layer.*.attention.self.key.bias",
        "bert.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="bert.encoder.layer.*.self_attention.qkv.bias",
)
def _pack_qkv_bias(ctx: io.TransformCTX, query, key, value):
    """Pack the QKV biases into fused TE format."""
    # Input shapes: query, key, value each have shape [num_heads * head_dim]
    # Example: [768] for 12 heads with 64-dim heads

    # Step 1: Concatenate Q, K, V biases along dimension 0
    # Shape: [3 * num_heads * head_dim] = [2304]
    concat_biases = torch.cat((query, key, value), dim=0)

    # Store original shape for final reshaping
    input_shape = concat_biases.size()  # [2304]

    # Get number of attention heads
    np = ctx.target.config.num_attention_heads  # 12

    # Step 2: Reshape to separate Q, K, V biases and organize by heads
    # Shape: [3, num_heads, head_dim] = [3, 12, 64]
    concat_biases = concat_biases.view(3, np, -1)

    # Step 3: Transpose to interleave Q, K, V biases for each head
    # Shape: [num_heads, 3, head_dim] = [12, 3, 64]
    concat_biases = concat_biases.transpose(0, 1).contiguous()

    # Step 4: Reshape back to original shape but with interleaved QKV biases
    # Shape: [3 * num_heads * head_dim] = [2304]
    concat_biases = concat_biases.view(*input_shape)

    return concat_biases


@io.state_transform(
    source_key="bert.encoder.layer.*.self_attention.qkv.weight",
    target_key=(
        "bert.encoder.layer.*.attention.self.query.weight",
        "bert.encoder.layer.*.attention.self.key.weight",
        "bert.encoder.layer.*.attention.self.value.weight",
    ),
)
def _unpack_qkv_weight(ctx: io.TransformCTX, qkv_weight):
    """Unpack the fused QKV weight into separate query, key, and value weights."""
    np = ctx.source.config.num_attention_heads

    # Reverse the packing transformation
    # First, reshape to separate the interleaved Q, K, V
    # [attention head size * num_splits_model_parallel * #attention heads]
    # --> [num_splits_model_parallel * attention head size * #attention heads]
    qkv_weight = qkv_weight.view(np, 3, -1, qkv_weight.size()[-1])  # Output:[num_heads, 3, head_dim, vocab_size]
    qkv_weight = qkv_weight.transpose(0, 1).contiguous()  # Output:[3, num_heads, head_dim, vocab_size]

    # Split into Q, K, V directly from the transposed tensor
    # qkv_weight shape: [3, num_heads, head_dim, input_dim]
    query = qkv_weight[0]  # [num_heads, head_dim, input_dim]
    key = qkv_weight[1]  # [num_heads, head_dim, input_dim]
    value = qkv_weight[2]  # [num_heads, head_dim, input_dim]

    # Reshape to match HF format: [total_head_dim, input_dim]
    query = query.view(-1, query.size()[-1])  # [num_heads * head_dim, input_dim]
    key = key.view(-1, key.size()[-1])  # [num_heads * head_dim, input_dim]
    value = value.view(-1, value.size()[-1])  # [num_heads * head_dim, input_dim]

    return query, key, value


@io.state_transform(
    source_key="bert.encoder.layer.*.self_attention.qkv.bias",
    target_key=(
        "bert.encoder.layer.*.attention.self.query.bias",
        "bert.encoder.layer.*.attention.self.key.bias",
        "bert.encoder.layer.*.attention.self.value.bias",
    ),
)
def _unpack_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    """Unpack the fused QKV bias into separate query, key, and value biases."""
    np = ctx.source.config.num_attention_heads

    # Reverse the packing transformation
    # First, reshape to separate the interleaved Q, K, V
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    qkv_bias = qkv_bias.view(np, 3, -1)
    qkv_bias = qkv_bias.transpose(0, 1).contiguous()

    # Split into Q, K, V directly from the transposed tensor
    # qkv_bias shape: [3, num_heads, head_dim]
    query = qkv_bias[0]  # [num_heads, head_dim]
    key = qkv_bias[1]  # [num_heads, head_dim]
    value = qkv_bias[2]  # [num_heads, head_dim]

    # Reshape to match HF format: [total_head_dim]
    query = query.view(-1)  # [num_heads * head_dim]
    key = key.view(-1)  # [num_heads * head_dim]
    value = value.view(-1)  # [num_heads * head_dim]

    return query, key, value
