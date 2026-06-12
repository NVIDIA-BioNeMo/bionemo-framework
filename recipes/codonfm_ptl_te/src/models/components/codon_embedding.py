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


from typing import Optional

import torch
import torch.nn as nn


class CodonEmbedding(nn.Module):
    """Codon Embedding layer.

    This module takes codon IDs as input and returns their embeddings.
    It includes an embedding layer, layer normalization, and dropout.
    """

    def __init__(self, config):
        """Initializes the CodonEmbedding module.

        Args:
            config: A configuration object with the following attributes:
                - vocab_size (int): The size of the vocabulary.
                - hidden_size (int): The size of the hidden embeddings.
                - pad_token_id (int): The ID of the padding token.
                - layer_norm_eps (float): The epsilon value for layer normalization.
                - hidden_dropout_prob (float): The dropout probability.
        """
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.post_ln = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Forward pass for the CodonEmbedding module.

        Args:
            input_ids (Optional[torch.LongTensor]): A tensor of shape (batch_size, sequence_length)
                containing the input codon IDs.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, sequence_length, hidden_size)
                representing the embeddings of the input codons.
        """
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.post_ln(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
