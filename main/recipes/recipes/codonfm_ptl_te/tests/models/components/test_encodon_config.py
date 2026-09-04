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

from src.models.components.encodon_config import EnCodonConfig


class TestEnCodonConfig:
    def test_default_initialization(self):
        config = EnCodonConfig()
        assert config.vocab_size == 69
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 12
        assert config.hidden_act == "gelu"

    def test_custom_initialization(self):
        config = EnCodonConfig(vocab_size=100, hidden_size=512, num_attention_heads=8, hidden_act="relu")
        assert config.vocab_size == 100
        assert config.hidden_size == 512
        assert config.num_attention_heads == 8
        assert config.hidden_act == "relu"

    def test_hidden_size_validation(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            EnCodonConfig(hidden_size=768, num_attention_heads=10)

    def test_hidden_act_validation(self):
        with pytest.raises(ValueError, match="must be one of"):
            EnCodonConfig(hidden_act="invalid_act")

    def test_to_dict(self):
        config = EnCodonConfig(vocab_size=50)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["vocab_size"] == 50
