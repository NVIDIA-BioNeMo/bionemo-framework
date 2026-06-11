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


import pytest
import torch

from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.embedding import ESM2_MASK_RATIO_TRAIN, ESM2Embedding
from bionemo.llm.lightning import get_dtype_device
from bionemo.testing import megatron_parallel_state_utils


@pytest.fixture(scope="module")
def tokenizer() -> BioNeMoESMTokenizer:
    yield get_tokenizer()


@pytest.fixture(scope="module")
def embedding(tokenizer) -> ESM2Embedding:
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        config = ESM2Config(seq_length=20, hidden_size=128)
        model = config.configure_model(tokenizer)
        yield model.embedding


def test_init(embedding, tokenizer):
    assert isinstance(embedding, ESM2Embedding)
    assert embedding.token_dropout is True
    assert embedding.use_attention_mask is True
    assert embedding.mask_token_id == tokenizer.mask_token_id


def test_forward(embedding):
    _, device = get_dtype_device(embedding)
    vocab_size = embedding.vocab_size
    max_sequence_length = embedding.max_sequence_length

    input_ids = torch.randint(0, vocab_size, (2, 10), device=device)  # [b, s]
    position_ids = torch.randint(0, max_sequence_length, (2, 10), device=device)  # [b, s]
    attention_mask = torch.randint(0, 2, (2, 10), device=device, dtype=torch.bool)  # [b, s, s]
    output = embedding(input_ids, position_ids, attention_mask=attention_mask)
    assert output.shape == (10, 2, 128)  # [s, b, h]


def test_apply_esm2_customization(embedding):
    # Create mock input tensors
    batch_size = 2
    sequence_length = 5
    hidden_size = embedding.config.hidden_size
    mask_token_id = embedding.mask_token_id

    input_ids = torch.tensor([[1, 2, 3, mask_token_id, 5], [6, 7, 8, 9, mask_token_id]])  # (b, s)
    attention_mask = torch.tensor([[1, 0, 1, 1, 1], [1, 1, 1, 0, 1]], dtype=torch.bool)  # (b, s, s)
    word_embeddings = torch.randn(batch_size, sequence_length, hidden_size)  # (b, s, h)

    # Call the _apply_esm2_customization function
    output_embeddings, embeddings_mask = embedding._apply_esm2_customization(
        word_embeddings, input_ids, attention_mask
    )

    # Check the output shapes
    assert output_embeddings.shape == (batch_size, sequence_length, hidden_size)
    assert embeddings_mask.shape == (batch_size, sequence_length)

    # Check the token dropout and attention masking logic
    assert torch.allclose(output_embeddings[0, 3, :], torch.zeros_like(output_embeddings[0, 3, :]))
    assert torch.allclose(output_embeddings[1, 4, :], torch.zeros_like(output_embeddings[1, 4, :]))
    assert torch.allclose(embeddings_mask[0, 1], torch.zeros_like(embeddings_mask[0, 1]))
    assert torch.allclose(embeddings_mask[1, 3], torch.zeros_like(embeddings_mask[1, 3]))

    # Check the mask ratio calculation
    mask_ratio_observed = (input_ids == mask_token_id).sum(-1).float() / attention_mask.sum(-1)
    assert torch.allclose(mask_ratio_observed, torch.tensor([0.25, 0.25]))

    # Check the word embeddings scaling
    scale_factor = (1 - ESM2_MASK_RATIO_TRAIN) / (1 - mask_ratio_observed)[:, None, None]
    word_embeddings = word_embeddings.masked_fill((input_ids == mask_token_id).unsqueeze(-1), 0.0)
    assert torch.allclose(output_embeddings, word_embeddings * scale_factor * embeddings_mask.unsqueeze(-1))

    # Check the attention masking
    assert torch.equal(embeddings_mask, attention_mask)
