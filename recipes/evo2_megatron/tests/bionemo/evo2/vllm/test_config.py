# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from bionemo.evo2.vllm.config import Evo2Config


PATTERN_1B = "SDH*SDHSDH*SDHSDH*SDHSDH*"
PATTERN_7B = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*"


@pytest.mark.parametrize(
    ("pattern", "num_layers", "hidden_size", "num_attention_heads"),
    [
        (PATTERN_1B, 25, 1920, 15),
        (PATTERN_7B, 32, 4096, 32),
    ],
)
def test_layer_pattern(pattern, num_layers, hidden_size, num_attention_heads):
    config = Evo2Config(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        hybrid_override_pattern=pattern,
    )

    assert config.operator_types == tuple(pattern)
    assert config.layers_block_type == ["attention" if symbol == "*" else "mamba" for symbol in pattern]
    assert config.head_dim == hidden_size // num_attention_heads


def test_default_config_matches_evo2_7b():
    config = Evo2Config()

    assert config.hidden_size == 4096
    assert config.num_hidden_layers == 32
    assert config.intermediate_size == 11008
    assert config.num_attention_heads == 32
    assert config.vocab_size == 512
    assert config.hybrid_override_pattern == PATTERN_7B
    assert config.num_groups_hyena == 4096
    assert config.num_groups_hyena_medium == 256
    assert config.num_groups_hyena_short == 256
    assert config.hidden_act == "gelu"
    assert config.gelu_approximate == "none"
    assert config.gated_linear_unit is True
    assert config.remove_activation_post_first_layer is True


def test_tp2_state_shapes_are_uniform():
    config = Evo2Config(num_hidden_layers=4, hybrid_override_pattern="SDH*")

    assert config.local_state_shapes(2) == ((6144, 2), (2048, 127))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_hidden_layers": 3, "hybrid_override_pattern": "SDH*"}, "length"),
        ({"num_hidden_layers": 4, "hybrid_override_pattern": "SDX*"}, "symbols"),
        ({"hidden_size": 10, "num_attention_heads": 3}, "attention heads"),
    ],
)
def test_invalid_model_shapes_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Evo2Config(**kwargs)


@pytest.mark.parametrize("tp_size", [0, -1, 3])
def test_invalid_tensor_parallel_sizes_are_rejected(tp_size):
    config = Evo2Config()

    with pytest.raises(ValueError, match="tensor parallel"):
        config.local_state_shapes(tp_size)


def test_tensor_parallel_size_must_partition_attention_heads():
    config = Evo2Config(
        hidden_size=12,
        num_attention_heads=3,
        num_hidden_layers=1,
        hybrid_override_pattern="*",
    )

    with pytest.raises(ValueError, match="attention heads"):
        config.local_state_shapes(2)


def test_config_round_trip(tmp_path):
    config = Evo2Config(
        max_position_embeddings=10240,
        seq_len_interpolation_factor=128.0,
        use_short_conv_bias=False,
        eos_token_id=0,
        pad_token_id=1,
    )

    config.save_pretrained(tmp_path)
    loaded = Evo2Config.from_pretrained(tmp_path)

    assert loaded.max_position_embeddings == 10240
    assert loaded.seq_len_interpolation_factor == 128.0
    assert loaded.use_short_conv_bias is False
    assert loaded.operator_types == tuple(PATTERN_7B)
    assert loaded.layers_block_type == config.layers_block_type
    assert loaded.architectures == ["Evo2ForCausalLM"]
    assert loaded.tie_word_embeddings is True
    assert loaded.eos_token_id == 0
    assert loaded.pad_token_id == 1
    assert loaded.hidden_act == "gelu"
    assert loaded.gelu_approximate == "none"
    assert loaded.gated_linear_unit is True
    assert loaded.remove_activation_post_first_layer is True
