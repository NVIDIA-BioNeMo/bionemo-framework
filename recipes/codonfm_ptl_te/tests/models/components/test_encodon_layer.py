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

from src.models.components.encodon_config import EnCodonConfig
from src.models.components.encodon_layer import EncoderLayer


@pytest.fixture(scope="module")
def device():
    """Fixture to use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def config():
    return EnCodonConfig(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=512,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )


@pytest.fixture
def encoder_layer(config, device):
    return EncoderLayer(config).to(device)


class TestEncoderLayer:
    def test_init(self, encoder_layer, config):
        assert isinstance(encoder_layer.attention, torch.nn.Module)
        assert isinstance(encoder_layer.pre_attn_layer_norm, torch.nn.LayerNorm)
        assert isinstance(encoder_layer.intermediate_dense, torch.nn.Linear)
        assert encoder_layer.intermediate_dense.out_features == config.intermediate_size
        assert encoder_layer.output_dense.in_features == config.intermediate_size

    def test_forward_pass_shape(self, encoder_layer, config, device):
        with patch.object(encoder_layer, "attention", autospec=True) as mock_mha_instance:
            mock_mha_instance.return_value = torch.randn(2, 16, config.hidden_size).to(device)
            batch_size = 2
            seq_len = 16
            hidden_size = config.hidden_size

            hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(device)
            attention_mask = torch.ones(batch_size, 1, 1, seq_len).to(device)

            output = encoder_layer(hidden_states, attention_mask)

            assert output.shape == (batch_size, seq_len, hidden_size)

            # Verify MHA was called correctly
            mock_mha_instance.assert_called_once()
            args, kwargs = mock_mha_instance.call_args
            assert args[0].shape == hidden_states.shape  # attn_input
            assert torch.equal(kwargs["attention_mask"], attention_mask)  # attention_mask

    def test_ffn_block(self, encoder_layer, config, device):
        # We can test the FFN block somewhat in isolation by passing a tensor through it

        # Let's test the logic inside the forward pass directly.
        # Create a dummy input for the FFN part
        ffn_input = torch.randn(2, 16, config.hidden_size).to(device)

        # Manually trace the FFN part of the forward pass
        ff_hidden_states = encoder_layer.pre_ffn_layer_norm(ffn_input)
        ff_hidden_states = encoder_layer.intermediate_dense(ff_hidden_states)
        assert ff_hidden_states.shape[-1] == config.intermediate_size

        ff_hidden_states = encoder_layer.intermediate_act_fn(ff_hidden_states)
        ff_hidden_states = encoder_layer.post_ffn_layer_norm(ff_hidden_states)
        ff_hidden_states = encoder_layer.output_dense(ff_hidden_states)
        assert ff_hidden_states.shape[-1] == config.hidden_size

        ff_hidden_states = encoder_layer.ffn_dropout(ff_hidden_states)

        layer_output = ff_hidden_states + ffn_input
        assert layer_output.shape == ffn_input.shape
