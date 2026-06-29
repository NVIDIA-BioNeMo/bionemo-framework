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


from unittest.mock import patch

import pytest
import torch

from src.models.components.mha import MultiHeadAttention


@pytest.fixture
def mha_module():
    return MultiHeadAttention(embed_dim=128, num_heads=8, dropout=0.1)


class TestMultiHeadAttention:
    def test_init(self, mha_module):
        assert mha_module.num_heads == 8
        assert mha_module.dropout_rate == 0.1
        assert isinstance(mha_module.query, torch.nn.Linear)
        assert mha_module.rotary_emb.inv_freq.shape == (8,)  # 128 / 8 / 2 = 8

    def test_forward_pass_shape(self, mha_module):
        batch_size = 2
        seq_len = 16  # Must be divisible by 8
        embed_dim = 128
        device = torch.device("cuda")

        # Move module to GPU
        mha_module = mha_module.to(device)

        x = torch.randn(batch_size, seq_len, embed_dim).to(device)
        # a simple padding mask
        attention_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool).to(device)

        output = mha_module(x, attention_mask)

        assert output.shape == (batch_size, seq_len, embed_dim)

    def test_assertion_errors(self, mha_module):
        # embed_dim not divisible by num_heads
        with pytest.raises(AssertionError):
            MultiHeadAttention(embed_dim=128, num_heads=10)

        # attention_mask shape
        x = torch.randn(2, 16, 128)
        attention_mask = torch.ones(2, 1, 1, 15)  # not divisible by 8
        with pytest.raises(AssertionError, match="must be divisible by 8"):
            mha_module(x, attention_mask)

    @patch("src.models.components.mha.apply_rotary_pos_emb")
    def test_rotary_embedding_application(self, mock_apply_rope, mha_module):
        batch_size = 2
        seq_len = 16
        embed_dim = 128
        x = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool)

        # The mock should return the tensor passed to it to not break the chain
        mock_apply_rope.side_effect = lambda t, cos, sin: t

        # Actually call the forward pass
        mha_module(x, attention_mask)

        # Check that apply_rotary_pos_emb was called twice (for q and k)
        assert mock_apply_rope.call_count == 2

        # Check the inputs to the first call (query)
        q_call_args = mock_apply_rope.call_args_list[0][0]
        assert q_call_args[0].shape == (batch_size, seq_len, mha_module.num_heads, embed_dim // mha_module.num_heads)

        # Check the inputs to the second call (key)
        k_call_args = mock_apply_rope.call_args_list[1][0]
        assert k_call_args[0].shape == (batch_size, seq_len, mha_module.num_heads, embed_dim // mha_module.num_heads)

    def test_forward_pass_end_to_end(self):
        """
        Test the forward pass of the MultiHeadAttention module end-to-end on a GPU.
        This test is skipped if CUDA is not available.
        """
        batch_size = 2
        seq_len = 16
        embed_dim = 128
        num_heads = 8
        device = torch.device("cuda")

        mha_module = MultiHeadAttention(embed_dim, num_heads).to(device)

        x = torch.randn(batch_size, seq_len, embed_dim).to(device)
        attention_mask = torch.ones(batch_size, 1, 1, seq_len, dtype=torch.bool).to(device)

        output = mha_module(x, attention_mask)

        assert output.shape == (batch_size, seq_len, embed_dim)
