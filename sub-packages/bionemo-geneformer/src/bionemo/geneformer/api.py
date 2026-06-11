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

from dataclasses import dataclass
from typing import Callable, Sequence, Type

from torch.nn import functional as F

from bionemo.geneformer.model.finetune_token_regressor import FineTuneSeqLenBioBertConfig
from bionemo.llm.api import MegatronLossType
from bionemo.llm.model.biobert.model import BioBertConfig, MegatronBioBertModel, PositionEmbeddingKinds
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.model.loss import BERTMLMLossWithReduction
from bionemo.llm.utils import iomixin_utils as iom


__all__: Sequence[str] = (
    "FineTuneSeqLenBioBertConfig",
    "GeneformerConfig",
    "GeneformerModel",
)

GeneformerModel = MegatronBioBertModel


@dataclass
class GeneformerConfig(BioBertConfig[GeneformerModel, MegatronLossType], iom.IOMixinWithGettersSetters):
    """A geneformer config.

    The geneformer config overrides the parent config, and adds a leaf-level iomixin, please do not inherit from this
    directly, as your parameters will likely be reset to this method's parameters silently.
    """

    num_layers: int = 6
    hidden_size: int = 256
    ffn_hidden_size: int = 512
    num_attention_heads: int = 4
    seq_length: int = 2048
    fp32_residual_connection: bool = False
    # Dropout
    attention_dropout: float = 0.1  # NeMo1 hard-coded, differs from publication of ReLU
    hidden_dropout: float = 0.02
    init_method_std: float = 0.02
    apply_query_key_layer_scaling: bool = False
    make_vocab_size_divisible_by: int = 128
    fp16_lm_cross_entropy: bool = False
    layernorm_zero_centered_gamma: bool = False
    layernorm_epsilon: float = 1.0e-12
    activation_func: Callable = F.gelu  # NeMo1 hard-coded, differes from publication of ReLU
    qk_layernorm: bool = False
    apply_residual_connection_post_layernorm: bool = False  # False is new default, True was BERT pub.
    share_embeddings_and_output_weights: bool = True
    # FUSION SETTINGS
    parallel_output: bool = True
    bias_dropout_fusion: bool = True
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    get_attention_mask_from_fusion: bool = True

    position_embedding_type: PositionEmbeddingKinds = "learned_absolute"
    biobert_spec_option: BiobertSpecOption = BiobertSpecOption.bert_layer_with_transformer_engine_spec
    qk_layernorm: bool = False

    enable_autocast: bool = False
    model_cls: Type[GeneformerModel] = GeneformerModel
    loss_reduction_class: Type[MegatronLossType] = BERTMLMLossWithReduction
