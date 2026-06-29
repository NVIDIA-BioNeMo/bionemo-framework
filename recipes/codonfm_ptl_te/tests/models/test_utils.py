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


from functools import partial

import pytest
import torch
from torch import nn
from torch.optim import Adam

from src.models.components.encodon_config import EnCodonConfig
from src.models.utils import (
    _get_cosine_schedule_with_warmup_lr_lambda,
    construct_pretrained_config,
    get_cosine_schedule_with_warmup,
    get_decay_parameter_names,
)


class TestConstructPretrainedConfig:
    def test_with_dict(self):
        config_dict = {"hidden_size": 256, "num_hidden_layers": 6, "num_attention_heads": 32}
        encodon_config = construct_pretrained_config(config_dict, EnCodonConfig)

        assert isinstance(encodon_config, EnCodonConfig)
        assert encodon_config.hidden_size == 256
        assert encodon_config.num_hidden_layers == 6
        assert encodon_config.num_attention_heads == 32
        assert encodon_config.vocab_size == 69  # default value

    def test_with_object(self):
        class MockConfig:
            hidden_size = 144
            # missing other attributes

        mock_config = MockConfig()
        encodon_config = construct_pretrained_config(mock_config, EnCodonConfig)

        assert encodon_config.hidden_size == 144
        assert encodon_config.num_attention_heads == 12  # default value

    def test_value_error(self):
        config_dict = {"hidden_size": 128, "num_attention_heads": 12}
        with pytest.raises(ValueError):
            construct_pretrained_config(config_dict, EnCodonConfig)


class TestCosineScheduler:
    def test_lr_lambda_function(self):
        lr_lambda = partial(
            _get_cosine_schedule_with_warmup_lr_lambda,
            num_warmup_steps=100,
            num_training_steps=1000,
            num_cycles=0.5,
        )

        # Warmup phase
        assert lr_lambda(50) == pytest.approx(0.5)
        # End of warmup
        assert lr_lambda(100) == pytest.approx(1.0)
        # Mid-training
        # progress = (550 - 100) / (1000 - 100) = 450 / 900 = 0.5
        # 0.5 * (1 + cos(pi * 0.5 * 2 * 0.5)) = 0.5 * (1 + cos(pi/2)) = 0.5
        assert lr_lambda(550) == pytest.approx(0.5)
        # End of training
        assert lr_lambda(1000) == 0.0

    def test_get_scheduler(self):
        model = nn.Linear(10, 2)
        optimizer = Adam(model.parameters(), lr=0.1)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 100, 1000)

        assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


class TestGetDecayParameterNames:
    def test_get_decay_parameter_names(self):
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(10, 10)
                self.layer1 = nn.Linear(10, 10)
                self.norm = nn.LayerNorm(10)
                self.bias_param = nn.Parameter(torch.zeros(10))

        model = MockModel()

        decay_params = get_decay_parameter_names(model)

        assert "layer1.weight" in decay_params
        assert "embedding.weight" not in decay_params
        assert "norm.weight" not in decay_params
        assert "norm.bias" not in decay_params
        assert "layer1.bias" not in decay_params
        # The logic adds all top-level parameters not matching disallowed names
        # 'bias_param' does not contain 'bias' as a full word but as a substring,
        # so it depends on the matching logic. The current logic uses 'in'.
        # Let's check the code: `any(forbidden in k.lower() for forbidden in disallowed_layer_names)`
        # 'bias' is in 'bias_param'.
        assert "bias_param" not in decay_params
