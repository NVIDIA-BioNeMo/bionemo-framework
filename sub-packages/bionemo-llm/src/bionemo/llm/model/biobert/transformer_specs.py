# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from enum import Enum
from typing import Optional, Sequence, Type

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.bert import bert_layer_specs
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import spec_utils
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from torch.nn import Module

from bionemo.llm.model.layers import ESM2QueryScaling, TELayerNorm


__all__: Sequence[str] = (
    "BiobertSpecOption",
    "get_biobert_spec",
)


class BiobertSpecOption(str, Enum):
    """Options for the BiobertSpec. The spec defines the architecture of the transformer (BERT) block in the biobert model.
    This is a `str, Enum` type so that argparse can use the string names as choices.
    """  # noqa: D205

    bert_layer_local_spec = "bert_layer_local_spec"
    bert_layer_local_spec_with_qk_ln = "bert_layer_local_spec_with_qk_ln"
    bert_layer_with_transformer_engine_spec = "bert_layer_with_transformer_engine_spec"
    bert_layer_with_transformer_engine_and_qk_ln_spec = "bert_layer_with_transformer_engine_and_qk_ln_spec"
    # ESM2 spec
    esm2_bert_layer_local_spec = "esm2_bert_layer_local_spec"
    esm2_bert_layer_with_transformer_engine_spec = "esm2_bert_layer_with_transformer_engine_spec"
    amplify_with_transformer_engine_spec = "amplify_with_transformer_engine_spec"


def get_biobert_spec(  # noqa: D417
    biobert_spec_option: BiobertSpecOption,
    qk_layernorm: bool = False,
    core_attention: Optional[Type[Module]] = None,
) -> spec_utils.ModuleSpec:
    """Get the spec for the Biobert model.

    Args:
        model_type (ModelType): The model type.
        spec_option (BiobertSpecOption): The spec option.

    Returns:
        TransformerConfig: The Biobert spec.
    """
    #
    # BEGIN define several specs that are a function of `qk_layernorm`
    #

    match biobert_spec_option:
        case BiobertSpecOption.bert_layer_local_spec:
            return bert_layer_specs.bert_layer_local_spec

        case BiobertSpecOption.bert_layer_local_spec_with_qk_ln:
            # Use this spec for an implementation using only modules in megatron core

            if core_attention is None:
                core_attention = DotProductAttention

            bert_layer_local_spec_with_qk_ln = spec_utils.ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=FusedLayerNorm,
                    self_attention=spec_utils.ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=core_attention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=FusedLayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=FusedLayerNorm,
                    mlp=spec_utils.ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=ColumnParallelLinear,
                            linear_fc2=RowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                        "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
                    },
                ),
            )
            return bert_layer_local_spec_with_qk_ln

        case BiobertSpecOption.bert_layer_with_transformer_engine_spec:
            return bert_layer_specs.bert_layer_with_transformer_engine_spec

        case BiobertSpecOption.bert_layer_with_transformer_engine_and_qk_ln_spec:
            if core_attention is None:
                core_attention = TEDotProductAttention

            bert_layer_with_transformer_engine_and_qk_ln_spec = spec_utils.ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=spec_utils.ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=core_attention,
                            linear_proj=TERowParallelLinear,
                            q_layernorm=TELayerNorm if qk_layernorm else IdentityOp,
                            k_layernorm=TELayerNorm if qk_layernorm else IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    mlp=spec_utils.ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear,
                            linear_fc2=TERowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            )
            return bert_layer_with_transformer_engine_and_qk_ln_spec

        case BiobertSpecOption.esm2_bert_layer_local_spec:
            if core_attention is None:
                raise ValueError(f"Must supply core_attention with {BiobertSpecOption.esm2_bert_layer_local_spec} !")

            esm2_bert_layer_local_spec = spec_utils.ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=FusedLayerNorm,
                    self_attention=spec_utils.ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=ColumnParallelLinear,
                            core_attention=core_attention,
                            linear_proj=RowParallelLinear,
                            q_layernorm=ESM2QueryScaling,
                            k_layernorm=IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    pre_mlp_layernorm=FusedLayerNorm,
                    mlp=spec_utils.ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=ColumnParallelLinear,
                            linear_fc2=RowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                    sharded_state_dict_keys_map={
                        "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                        "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
                    },
                ),
            )
            return esm2_bert_layer_local_spec

        case BiobertSpecOption.esm2_bert_layer_with_transformer_engine_spec:
            if core_attention is None:
                core_attention = TEDotProductAttention

            esm2_bert_layer_local_spec = spec_utils.ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=spec_utils.ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=core_attention,
                            linear_proj=TERowParallelLinear,
                            q_layernorm=ESM2QueryScaling,
                            k_layernorm=IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    mlp=spec_utils.ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear,
                            linear_fc2=TERowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            )
            return esm2_bert_layer_local_spec

        case BiobertSpecOption.amplify_with_transformer_engine_spec:
            if core_attention is None:
                core_attention = TEDotProductAttention

            amplify_with_transformer_engine_spec = spec_utils.ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=spec_utils.ModuleSpec(
                        module=SelfAttention,
                        params={"attn_mask_type": AttnMaskType.padding},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=TELayerNormColumnParallelLinear,
                            core_attention=core_attention,
                            linear_proj=TERowParallelLinear,
                            q_layernorm=IdentityOp,
                            k_layernorm=IdentityOp,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                    mlp=spec_utils.ModuleSpec(
                        module=MLP,
                        submodules=MLPSubmodules(
                            linear_fc1=TELayerNormColumnParallelLinear,
                            linear_fc2=TERowParallelLinear,
                        ),
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            )
            return amplify_with_transformer_engine_spec

        case _:
            raise NotImplementedError(f"Spec option {biobert_spec_option} not implemented")
