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

"""Tests for Qwen2 model.

This file provides comprehensive tests for the Qwen2 model including:
- Common tests from the test library (meta device init, golden values, conversion, FP8)
- Qwen2-specific tests (inference, generation with KV-cache)
"""

import os
from typing import Callable, Dict, List, Literal, Type

import pytest
import torch
import transformers
from torch import nn
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from collator import DataCollatorWithFlattening
from convert_qwen2 import convert_qwen2_hf_to_te, convert_qwen2_te_to_hf
from modeling_qwen2_te import HFInferenceParams, NVQwen2Config, NVQwen2ForCausalLM
from tests.common import BaseModelTest, TestTolerances


class TestQwen2Model(BaseModelTest):
    """Model tester for Qwen2.

    This class provides Qwen2-specific configuration for the common test suite.
    """

    is_autoregressive = True

    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the Qwen2 TE model class."""
        return NVQwen2ForCausalLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the Qwen2 config class."""
        return NVQwen2Config

    def get_upstream_model_id(self) -> str:
        """Return the upstream HuggingFace model ID."""
        return "Qwen/Qwen2.5-0.5B"

    def get_upstream_model_revision(self) -> str:
        """Return the specific revision for the upstream model."""
        return "060db64"

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the Qwen2 tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.get_upstream_model_id(), revision=self.get_upstream_model_revision()
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return the upstream HuggingFace model class."""
        return transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM

    def create_test_config(self, **kwargs) -> PretrainedConfig:
        # Limit the number of hidden layers to 2 for faster tests.
        return super().create_test_config(num_hidden_layers=2, **kwargs)

    def get_reference_model(
        self, dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "flash_attention_2"
    ) -> PreTrainedModel:
        """Return the reference HuggingFace model."""
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping Qwen2 reference model test in CI, requires Qwen2.5-0.5B download ~1GB")
        return super().get_reference_model(dtype=dtype, attn_implementation=attn_implementation)

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers."""
        return list(model.model.layers)  # type: ignore

    def get_test_input_data(
        self, format: Literal["bshd", "thd"] = "bshd", pad_to_multiple_of: int | None = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data (text sequences)."""
        tokenizer = self.get_tokenizer()
        test_texts = [
            "Unless required by applicable law or agreed to in writing, software distributed under the License.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            mlm=False,
        )

        if format == "thd":
            data_collator = DataCollatorWithFlattening(
                collator=data_collator,
                pad_sequences_to_be_divisible_by=pad_to_multiple_of,
                separator_id=-100,
            )

        batch = data_collator([tokenizer(text) for text in test_texts])
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def get_hf_to_te_converter(self) -> Callable:
        """Return the HF to TE conversion function."""
        return convert_qwen2_hf_to_te

    def get_te_to_hf_converter(self) -> Callable:
        """Return the TE to HF conversion function."""
        return convert_qwen2_te_to_hf

    def get_tolerances(self) -> TestTolerances:
        """Return Qwen2-specific test tolerances."""
        return TestTolerances(
            golden_value_loss_atol=0.05,
            golden_value_loss_rtol=0.02,
            golden_value_logits_atol=2.0,
            golden_value_logits_rtol=0.01,
            cp_loss_atol=0.5,
            cp_loss_rtol=0.25,
        )

    # ==================== Qwen2 Overrides ====================

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_quantized_model_init_forward_and_backward(self, fp8_recipe, input_format, tie_word_embeddings):
        """Test FP8 forward and backward pass with both tied and untied word embeddings."""
        super().test_quantized_model_init_forward_and_backward(
            fp8_recipe, input_format, tie_word_embeddings=tie_word_embeddings
        )

    # ==================== Qwen2-Specific Overrides ====================

    def create_inference_params(self, config, batch_size=1, max_seq_len=256, num_beams=1):
        """Create HFInferenceParams for the given config.

        Uses hidden_size // num_attention_heads for head_dim since Qwen2 does not
        independently configure head_dim.
        """
        head_dim = config.hidden_size // config.num_attention_heads
        past_key_values = HFInferenceParams(
            max_batch_size=batch_size * num_beams,
            max_sequence_length=max_seq_len,
            num_heads_kv=config.num_key_value_heads,
            head_dim_k=head_dim,
            dtype=torch.bfloat16,
            qkv_format="thd",
            max_ctx_len=max_seq_len,
        )
        for layer_number in range(1, config.num_hidden_layers + 1):
            past_key_values.allocate_memory(layer_number)
        return past_key_values
