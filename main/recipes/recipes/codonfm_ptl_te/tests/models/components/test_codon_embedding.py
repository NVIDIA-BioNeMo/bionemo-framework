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


import pytest
import torch

from src.models.components.codon_embedding import CodonEmbedding
from src.models.components.encodon_config import EnCodonConfig


@pytest.fixture
def config():
    return EnCodonConfig(
        vocab_size=100, hidden_size=768, pad_token_id=3, layer_norm_eps=1e-12, hidden_dropout_prob=0.1
    )


class TestCodonEmbedding:
    def test_init(self, config):
        embedding_layer = CodonEmbedding(config)
        assert isinstance(embedding_layer.word_embeddings, torch.nn.Embedding)
        assert embedding_layer.word_embeddings.num_embeddings == config.vocab_size
        assert embedding_layer.word_embeddings.embedding_dim == config.hidden_size
        assert isinstance(embedding_layer.post_ln, torch.nn.LayerNorm)
        assert isinstance(embedding_layer.dropout, torch.nn.Dropout)

    def test_forward_pass(self, config):
        embedding_layer = CodonEmbedding(config)

        batch_size = 4
        seq_length = 20
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))

        output_embeddings = embedding_layer(input_ids)

        assert output_embeddings.shape == (batch_size, seq_length, config.hidden_size)
        assert output_embeddings.dtype == torch.float32

    def test_padding_idx(self, config):
        embedding_layer = CodonEmbedding(config)
        # Set dropout to 0 to check padding
        embedding_layer.dropout.p = 0.0

        input_ids = torch.tensor([[1, 2, config.pad_token_id, 4]], dtype=torch.long)

        # Manually get embedding for non-padded tokens
        expected_output = embedding_layer.post_ln(embedding_layer.word_embeddings(input_ids))  # noqa: F841

        # Pad token embedding should be zero before LayerNorm
        # After LayerNorm, it might not be exactly zero, but let's test the word_embedding part
        word_embeddings = embedding_layer.word_embeddings(input_ids)
        assert torch.all(word_embeddings[0, 2] == 0)
