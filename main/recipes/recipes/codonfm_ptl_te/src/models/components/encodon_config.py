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


class EnCodonConfig:
    """Configuration class for EnCodon model.

    This class handles the configuration for the EnCodon model, including model architecture
    parameters and training settings.
    """

    model_type = "encodon"

    def __init__(  # noqa: D417
        self,
        vocab_size: int = 69,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 3,
        classifier_dropout: float = 0.1,
        gamma_init: float = 0.1,
        model_type: str = "custom",
        rotary_theta: float = 1e5,
        use_return_dict: bool = False,
        max_position_embeddings: int = 2048,
        attn_input_format: str = "bshd",
    ):
        """Initialize EnCodonConfig.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            num_hidden_layers: Number of hidden layers.
            num_attention_heads: Number of attention heads.
            intermediate_size: Size of the intermediate layer in the transformer.
            hidden_act: Activation function for the hidden layer.
            hidden_dropout_prob: Dropout probability for hidden layers.
            attention_probs_dropout_prob: Dropout probability for attention layers.
            initializer_range: Range for weight initialization.
            layer_norm_eps: Epsilon for layer normalization.
            pad_token_id: ID of the padding token.
            classifier_dropout: Dropout probability for classifier.
            gamma_init: Initial value for gamma.
            rotary_theta: Theta value for rotary embeddings.
            max_position_embeddings: Maximum position embeddings.
            attn_input_format: Attention input format ("bshd" or "thd").

        """
        # Validate parameters
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        if hidden_act not in ["gelu", "relu", "silu", "gelu_new"]:
            raise ValueError(f"hidden_act must be one of: gelu, relu, silu, gelu_new, got {hidden_act}")

        self.model_type = model_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.classifier_dropout = classifier_dropout
        self.gamma_init = gamma_init
        self.rotary_theta = rotary_theta
        self.use_return_dict = use_return_dict
        self.max_position_embeddings = max_position_embeddings
        self.attn_input_format = attn_input_format

    def to_dict(self):  # noqa: D102
        return self.__dict__.copy()
