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

"""Tests for ESM2 model using the common test library.

This file provides comprehensive tests for the ESM2 model including:
- Meta device initialization tests
- Golden value tests against HuggingFace reference models
- Conversion tests (HF ↔ TE)
- FP8 tests
- Model-specific tests

Most tests are inherited from the common test library to reduce duplication.
"""

from typing import Callable, Dict, List, Literal, Type
from unittest.mock import MagicMock

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from transformers.models.esm.modeling_esm import EsmForMaskedLM

from collator import DataCollatorWithFlattening
from convert import (
    _pack_qkv_bias,
    _pack_qkv_weight,
    _pad_bias,
    _pad_weights,
    convert_esm_hf_to_te,
    convert_esm_te_to_hf,
    mapping,
)
from modeling_esm_te import NVEsmConfig, NVEsmForMaskedLM
from tests.common import BaseModelTest, TestTolerances


class TestESM2Model(BaseModelTest):
    """Model tester for ESM2.

    This class provides ESM2-specific configuration for the common test suite.
    """

    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the ESM2 TE model class."""
        return NVEsmForMaskedLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the ESM2 config class."""
        return NVEsmConfig

    def get_upstream_model_id(self) -> str:
        """Return the upstream HuggingFace model ID."""
        return "facebook/esm2_t6_8M_UR50D"

    def get_upstream_model_revision(self) -> str:
        """Return the specific revision for the upstream model."""
        return "c731040f"

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return the upstream HuggingFace model class."""
        return EsmForMaskedLM

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers."""
        return list(model.model.encoder.layers)  # type: ignore

    def get_reference_model_no_weights(self) -> PreTrainedModel:
        """For checkpoint conversion tests to pass, we need to remove the unused contact head."""
        model = super().get_reference_model_no_weights()
        del model.esm.contact_head
        return model

    def get_test_input_data(
        self,
        format: Literal["bshd", "thd"] = "bshd",
        pad_to_multiple_of: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data (protein sequences)."""

        tokenizer = self.get_tokenizer()

        # Use real protein sequences
        test_proteins = [
            "MLSATEKLSDYISSLFASVSIINSISTEDLFFLKLTCQTFSKDSEEYKAAYRILRGVQRGKVQIIEEALVS",
            "MFVFFAGTLVNQDTLNFRDQLNINVVGTVRGIAQDASKYLEYAIDSV",
            "MAATGSLILSDEEQAELIALAVRIVLACAGGSQNKELAAQLGVIETTVGEWRRRFAQNRVEGLRDEARPGAPSDDQ",
            "MSAVLSAVASDDWTAFAKLVHPYVHWTADGITTRGRTRVMARLSGHDGVKPASSYELRDGQVYRWTS",
            "MSDPAAEPPADTSGIAWRKSSYSGPNGNCVELAQISGDHVGIRNSRDLHGSVLTCTRAEFAALLCDIKAGRFDSLIL",
        ]

        # Tokenize
        tokenized = [tokenizer(p, truncation=True, max_length=1024) for p in test_proteins]

        # Use data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=pad_to_multiple_of if format == "bshd" else None,
            seed=42,
        )

        if format == "thd":
            data_collator = DataCollatorWithFlattening(
                collator=data_collator,
                pad_sequences_to_be_divisible_by=pad_to_multiple_of,
            )

        batch = data_collator(tokenized)

        # Move to device
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def get_hf_to_te_converter(self) -> Callable:
        """Return the HF to TE conversion function."""
        return convert_esm_hf_to_te

    def get_te_to_hf_converter(self) -> Callable:
        """Return the TE to HF conversion function."""
        return convert_esm_te_to_hf

    def get_tolerances(self) -> TestTolerances:
        """Return ESM2-specific test tolerances."""
        return TestTolerances(
            golden_value_loss_atol=2e-2,
            golden_value_loss_rtol=1e-2,
            golden_value_logits_atol=2.0,  # Higher tolerance needed after transformers PR#40370
            golden_value_logits_rtol=1e-4,
            cp_loss_atol=0.1,
            cp_loss_rtol=0.05,
        )

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the ESM2 tokenizer."""
        return AutoTokenizer.from_pretrained("esm_fast_tokenizer")

    # ==================== ESM2-Specific Tests ====================

    def test_convert_state_dict_explicit_check(self):
        """Test detailed state dict conversion and mapping."""

        input_data = self.get_test_input_data()
        model_hf = self.get_reference_model()
        model_te = self.get_converted_te_model()

        model_hf.to("cuda")
        model_te.to("cuda")
        input_data = {k: v.to("cuda") for k, v in input_data.items()}

        with torch.no_grad():
            outputs = model_te(**input_data)
            assert outputs.loss

        te_state_dict_keys = {
            k for k in model_te.state_dict().keys() if not k.endswith("_extra_state") and not k.endswith("inv_freq")
        }

        # Check standard mapping
        for k, v in mapping.items():
            if "*" in k:
                for i in range(model_hf.config.num_hidden_layers):
                    k_sub = k.replace("*", str(i))
                    v_sub = v.replace("*", str(i))
                    torch.testing.assert_close(
                        model_te.state_dict()[v_sub],
                        model_hf.state_dict()[k_sub],
                        msg=lambda x: f"{k} {i} is not close: {x}",
                    )
                    te_state_dict_keys.remove(v_sub)
            else:
                torch.testing.assert_close(
                    model_te.state_dict()[v],
                    model_hf.state_dict()[k],
                    msg=lambda x: f"{k} is not close: {x}",
                )
                te_state_dict_keys.remove(v)

        # Check packed QKV weights
        for i in range(model_hf.config.num_hidden_layers):
            k = f"model.encoder.layers.{i}.self_attention.layernorm_qkv.weight"
            v = [
                f"esm.encoder.layer.{i}.attention.self.query.weight",
                f"esm.encoder.layer.{i}.attention.self.key.weight",
                f"esm.encoder.layer.{i}.attention.self.value.weight",
            ]

            ctx_mock = MagicMock()
            ctx_mock.target.config.num_attention_heads = model_hf.config.num_attention_heads

            packed_weight = _pack_qkv_weight.transform(
                ctx_mock,
                model_hf.state_dict()[v[0]],
                model_hf.state_dict()[v[1]],
                model_hf.state_dict()[v[2]],
            )

            torch.testing.assert_close(packed_weight, model_te.state_dict()[k])
            te_state_dict_keys.remove(k)

        # Check packed QKV biases
        for i in range(model_hf.config.num_hidden_layers):
            k = f"model.encoder.layers.{i}.self_attention.layernorm_qkv.bias"
            v = [
                f"esm.encoder.layer.{i}.attention.self.query.bias",
                f"esm.encoder.layer.{i}.attention.self.key.bias",
                f"esm.encoder.layer.{i}.attention.self.value.bias",
            ]

            ctx_mock = MagicMock()
            ctx_mock.target.config.num_attention_heads = model_hf.config.num_attention_heads

            packed_weight = _pack_qkv_bias.transform(
                ctx_mock,
                model_hf.state_dict()[v[0]],
                model_hf.state_dict()[v[1]],
                model_hf.state_dict()[v[2]],
            )

            torch.testing.assert_close(packed_weight, model_te.state_dict()[k])
            te_state_dict_keys.remove(k)

        # Check padded embeddings and LM head
        ctx_mock = MagicMock()
        ctx_mock.target.config.padded_vocab_size = model_te.config.padded_vocab_size

        torch.testing.assert_close(
            _pad_weights(ctx_mock, model_hf.state_dict()["esm.embeddings.word_embeddings.weight"]),
            model_te.state_dict()["model.embeddings.word_embeddings.weight"],
        )
        torch.testing.assert_close(
            _pad_weights(ctx_mock, model_hf.state_dict()["lm_head.decoder.weight"]),
            model_te.state_dict()["lm_head.decoder.weight"],
        )
        torch.testing.assert_close(
            _pad_bias.transform(ctx_mock, model_hf.state_dict()["lm_head.bias"]),
            model_te.state_dict()["lm_head.decoder.bias"],
        )

        te_state_dict_keys.remove("model.embeddings.word_embeddings.weight")
        te_state_dict_keys.remove("lm_head.decoder.weight")
        te_state_dict_keys.remove("lm_head.decoder.bias")

        assert len(te_state_dict_keys) == 0

        # Check that the tied weights are the same
        assert (
            model_hf.state_dict()["esm.embeddings.word_embeddings.weight"].data_ptr()
            == model_hf.state_dict()["lm_head.decoder.weight"].data_ptr()
        )

        assert (
            model_te.state_dict()["model.embeddings.word_embeddings.weight"].data_ptr()
            == model_te.state_dict()["lm_head.decoder.weight"].data_ptr()
        )

    def create_inference_params(self, config, batch_size=1, max_seq_len=256, num_beams=1):
        """These are unused for non-autoregressive models."""
        pass
