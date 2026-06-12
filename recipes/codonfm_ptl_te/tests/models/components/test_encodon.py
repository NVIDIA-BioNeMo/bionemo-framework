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

from src.models.components.cross_attention import CrossAttention
from src.models.components.encodon import EnCodon
from src.models.components.encodon_config import EnCodonConfig


@pytest.fixture
def config():
    return EnCodonConfig(
        vocab_size=100,
        hidden_size=128,
        num_attention_heads=8,
        num_hidden_layers=2,  # Keep it small for testing
        intermediate_size=512,
    )


@pytest.fixture
def encodon_model(config):
    return EnCodon(config)


class TestEnCodon:
    def test_init(self, encodon_model, config):
        assert isinstance(encodon_model.embeddings, torch.nn.Module)
        assert len(encodon_model.layers) == config.num_hidden_layers
        assert isinstance(encodon_model.cls, torch.nn.Sequential)

    def test_init_weights(self, encodon_model):
        # A simple check to see if weights are not all zeros or ones after init
        # except for specific biases/weights
        for name, param in encodon_model.named_parameters():
            if "LayerNorm" in name and "weight" in name:
                assert torch.all(param == 1.0)
            elif "bias" in name:
                assert torch.all(param == 0.0)
            else:
                assert not torch.all(param == 0.0)

    def test_get_extended_attention_mask(self, encodon_model):
        # Test 2D mask
        mask_2d = torch.tensor([[1, 1, 0]], dtype=torch.float32)
        extended_2d = encodon_model._get_extended_attention_mask(mask_2d, mask_2d.shape, mask_2d.device, mask_2d.dtype)
        assert extended_2d.shape == (1, 1, 1, 3)
        assert extended_2d[0, 0, 0, 2] < -1000  # Masked position
        assert extended_2d[0, 0, 0, 0] == 0  # Unmasked position

        # Test 3D mask
        mask_3d = torch.ones(1, 3, 3)
        extended_3d = encodon_model._get_extended_attention_mask(mask_3d, mask_3d.shape, mask_3d.device, mask_3d.dtype)
        assert extended_3d.shape == (1, 1, 3, 3)

    @patch("src.models.components.encodon.EncoderLayer", autospec=True)
    def test_forward_pass(self, mock_encoder_layer_class, config):
        # Mock the layer to just return the input
        mock_layer_instance = mock_encoder_layer_class.return_value
        mock_layer_instance.side_effect = lambda hidden_states, attention_mask: hidden_states

        model = EnCodon(config)

        batch_size = 2
        seq_len = 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        output = model(input_ids, attention_mask)

        assert isinstance(output.logits, torch.Tensor)
        assert output.logits.shape == (batch_size, seq_len, config.vocab_size)
        assert output.last_hidden_state.shape == (batch_size, seq_len, config.hidden_size)

        # Check that layers were called
        assert mock_layer_instance.call_count == config.num_hidden_layers

    @patch("src.models.components.encodon.EncoderLayer", autospec=True)
    def test_forward_return_hidden_states(self, mock_encoder_layer_class, config):
        mock_layer_instance = mock_encoder_layer_class.return_value
        mock_layer_instance.side_effect = lambda hidden_states, attention_mask: hidden_states + 0.1

        model = EnCodon(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        attention_mask = torch.ones(2, 16)

        output = model(input_ids, attention_mask, return_hidden_states=True)

        assert output.all_hidden_states is not None
        assert len(output.all_hidden_states) == config.num_hidden_layers
        assert output.all_hidden_states[0].shape == (2, 16, config.hidden_size)
        # check that they are different due to the side_effect
        assert not torch.equal(output.all_hidden_states[0], output.all_hidden_states[1])


class TestCrossAttention:
    @pytest.mark.parametrize("n_out", [1, 3])
    def test_cross_attention_forward_shapes_and_grad(self, n_out):
        torch.manual_seed(0)
        hidden_dim = 32
        batch = 2
        seq_len = 5

        layer = CrossAttention(hidden_dim=hidden_dim, n_out=n_out, num_heads=4, dropout=0.0)
        layer.train(False)

        query_input = torch.randn(batch, hidden_dim, requires_grad=True)
        key_value_input = torch.randn(batch, seq_len, hidden_dim, requires_grad=True)

        out = layer(query_input, key_value_input, attention_mask=None)
        assert out.shape == (batch, n_out)

        loss = out.sum()
        loss.backward()
        assert query_input.grad is not None
        assert key_value_input.grad is not None

    def test_cross_attention_with_mask_shapes(self):
        torch.manual_seed(0)
        hidden_dim = 32
        batch = 3
        seq_len = 7

        layer = CrossAttention(hidden_dim=hidden_dim, n_out=2, num_heads=4, dropout=0.0)
        layer.eval()

        query_input = torch.randn(batch, hidden_dim)
        key_value_input = torch.randn(batch, seq_len, hidden_dim)
        attention_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.bool,
        )

        out = layer(query_input, key_value_input, attention_mask=attention_mask)
        assert out.shape == (batch, 2)
