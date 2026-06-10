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

import sys
from pathlib import Path

import pytest
import torch


sys.path.append(Path(__file__).parent.parent.as_posix())

from modeling_codonfm_te import MODEL_PRESETS, CodonFMConfig, CodonFMForMaskedLM


@pytest.fixture
def default_config():
    return CodonFMConfig()


@pytest.fixture
def small_config():
    return CodonFMConfig(**MODEL_PRESETS["encodon_200k"])


class TestCodonFMConfig:
    def test_defaults(self, default_config):
        assert default_config.vocab_size == 69
        assert default_config.hidden_size == 768
        assert default_config.num_hidden_layers == 12
        assert default_config.num_attention_heads == 12
        assert default_config.intermediate_size == 3072
        assert default_config.pad_token_id == 3
        assert default_config.mask_token_id == 4
        assert default_config.layer_norm_eps == 1e-12

    def test_validation_hidden_size(self):
        with pytest.raises(ValueError, match="hidden_size.*divisible"):
            CodonFMConfig(hidden_size=100, num_attention_heads=12)

    def test_validation_activation(self):
        with pytest.raises(ValueError, match="hidden_act"):
            CodonFMConfig(hidden_act="invalid")

    def test_json_roundtrip(self, tmp_path, default_config):
        path = tmp_path / "config.json"
        default_config.to_json_file(str(path))
        loaded = CodonFMConfig.from_json_file(str(path))
        assert loaded.vocab_size == default_config.vocab_size
        assert loaded.hidden_size == default_config.hidden_size


class TestModelPresets:
    def test_all_presets_exist(self):
        expected = {"encodon_200k", "encodon_80m", "encodon_600m", "encodon_1b", "encodon_5b"}
        assert set(MODEL_PRESETS.keys()) == expected

    @pytest.mark.parametrize("preset_name", list(MODEL_PRESETS.keys()))
    def test_preset_creates_valid_config(self, preset_name):
        config = CodonFMConfig(**MODEL_PRESETS[preset_name])
        assert config.hidden_size % config.num_attention_heads == 0

    def test_encodon_200k_values(self):
        config = CodonFMConfig(**MODEL_PRESETS["encodon_200k"])
        assert config.hidden_size == 128
        assert config.intermediate_size == 512
        assert config.num_attention_heads == 4
        assert config.num_hidden_layers == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
class TestCodonFMForMaskedLM:
    @pytest.mark.parametrize("preset_name", list(MODEL_PRESETS.keys()))
    def test_forward_all_presets(self, preset_name):
        config = CodonFMConfig(**MODEL_PRESETS[preset_name])
        model = CodonFMForMaskedLM(config).cuda()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_labels(self, small_config):
        model = CodonFMForMaskedLM(small_config).cuda()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
        labels = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")
        labels[labels == 0] = -100  # Mask some labels

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        assert outputs.loss is not None
        assert outputs.loss.dim() == 0  # scalar
        assert outputs.loss.item() > 0

    def test_forward_without_labels(self, small_config):
        model = CodonFMForMaskedLM(small_config).cuda()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        assert outputs.loss is None
        assert outputs.logits is not None

    def test_embedding_shapes(self, small_config):
        model = CodonFMForMaskedLM(small_config).cuda()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, small_config.vocab_size, (batch_size, seq_len), device="cuda")

        embeddings = model.embeddings(input_ids)
        assert embeddings.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_loss_decreases(self, small_config):
        model = CodonFMForMaskedLM(small_config).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        batch_size, seq_len = 4, 32
        input_ids = torch.randint(5, small_config.vocab_size, (batch_size, seq_len), device="cuda")
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long, device="cuda")
        labels = input_ids.clone()

        initial_loss = None
        for _ in range(20):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            if initial_loss is None:
                initial_loss = outputs.loss.item()
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        final_loss = outputs.loss.item()
        assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"
