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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
from accelerate import init_empty_weights
from torch import nn

import amplify.state as io
from amplify.amplify_te import AMPLIFYConfig, AMPLIFYForMaskedLM


mapping = {
    "transformer_encoder.*.wo.weight": "amplify.transformer_encoder.*.self_attention.proj.weight",
    "transformer_encoder.*.ffn.w12.weight": "amplify.transformer_encoder.*.layernorm_mlp.fc1_weight",
    "transformer_encoder.*.ffn_norm.weight": "amplify.transformer_encoder.*.layernorm_mlp.layer_norm_weight",
    "transformer_encoder.*.ffn.w3.weight": "amplify.transformer_encoder.*.layernorm_mlp.fc2_weight",
    "transformer_encoder.*.attention_norm.weight": "amplify.transformer_encoder.*.self_attention.layernorm_qkv.layer_norm_weight",
    "layer_norm_2.weight": "decoder.layer_norm_weight",
}


def convert_amplify_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert a Hugging Face model to a Transformer Engine model.

    Args:
        model_hf (nn.Module): The Hugging Face model.
        **config_kwargs: Additional configuration kwargs to be passed to AMPLIFYConfig.

    Returns:
        nn.Module: The Transformer Engine model.
    """
    te_config = AMPLIFYConfig(**model_hf.config.to_dict(), **config_kwargs)
    with init_empty_weights():
        model_te = AMPLIFYForMaskedLM(te_config, dtype=te_config.dtype)

    output_model = io.apply_transforms(
        model_hf,
        model_te,
        mapping,
        [_pack_qkv_weight, _pad_embeddings, _pad_decoder_weights, _pad_bias],
    )

    output_model.tie_weights()

    return output_model


@io.state_transform(
    source_key=(
        "transformer_encoder.*.q.weight",
        "transformer_encoder.*.k.weight",
        "transformer_encoder.*.v.weight",
    ),
    target_key="amplify.transformer_encoder.*.self_attention.layernorm_qkv.weight",
)
def _pack_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    np = ctx.target.config.num_attention_heads
    # transpose weights
    # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
    # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
    concat_weights = concat_weights.view(3, np, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    concat_weights = concat_weights.view(*input_shape)
    return concat_weights


def _pad_weights(ctx: io.TransformCTX, source_embed):
    """Pad the embedding layer to the new input dimension."""
    target_embedding_dimension = ctx.target.config.padded_vocab_size
    hf_embedding_dimension = source_embed.size(0)
    num_padding_rows = target_embedding_dimension - hf_embedding_dimension
    padding_rows = torch.zeros(num_padding_rows, source_embed.size(1))
    return torch.cat((source_embed, padding_rows), dim=0)


_pad_embeddings = io.state_transform(
    source_key="encoder.weight",
    target_key="amplify.encoder.weight",
)(_pad_weights)

_pad_decoder_weights = io.state_transform(
    source_key="decoder.weight",
    target_key="decoder.weight",
)(_pad_weights)


@io.state_transform(
    source_key="decoder.bias",
    target_key="decoder.bias",
)
def _pad_bias(ctx: io.TransformCTX, source_bias):
    """Pad the embedding layer to the new input dimension."""
    target_embedding_dimension = ctx.target.config.padded_vocab_size
    hf_embedding_dimension = source_bias.size(0)
    output_bias = torch.finfo(source_bias.dtype).min * torch.ones(
        target_embedding_dimension, dtype=source_bias.dtype, device=source_bias.device
    )
    output_bias[:hf_embedding_dimension] = source_bias
    return output_bias
