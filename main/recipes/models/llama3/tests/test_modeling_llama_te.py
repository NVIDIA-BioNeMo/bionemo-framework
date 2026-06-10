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

"""Tests for LLaMA3 model.

This file provides comprehensive tests for the LLaMA3 model including:
- Common tests from the test library (meta device init, golden values, conversion, FP8)
- LLaMA-specific tests (inference, generation, THD inputs, etc.)
"""

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
from convert import convert_llama_hf_to_te, convert_llama_te_to_hf
from modeling_llama_te import HFInferenceParams, NVLlamaConfig, NVLlamaForCausalLM
from tests.common import BaseModelTest, TestTolerances


class TestLlama3Model(BaseModelTest):
    """Model tester for LLaMA3.

    This class provides LLaMA3-specific configuration for the common test suite.
    """

    is_autoregressive = True

    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the LLaMA3 TE model class."""
        return NVLlamaForCausalLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the LLaMA3 config class."""
        return NVLlamaConfig

    def get_upstream_model_id(self) -> str:
        """Return the upstream HuggingFace model ID."""
        # Use smaller 1B model for testing
        return "meta-llama/Llama-3.2-1B-Instruct"

    def get_upstream_model_revision(self) -> str:
        """Return the specific revision for the upstream model."""
        return "9213176"

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the LLaMA3 tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.get_upstream_model_id())
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return the upstream HuggingFace model class."""
        return transformers.models.llama.modeling_llama.LlamaForCausalLM

    def create_test_config(self, **kwargs) -> PretrainedConfig:
        # Limit the number of hidden layers to 2 for faster tests.
        return super().create_test_config(num_hidden_layers=2, **kwargs)

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers."""
        return list(model.model.layers)  # type: ignore

    def get_test_input_data(
        self, format: Literal["bshd", "thd"] = "bshd", pad_to_multiple_of: int | None = None
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data (text sequences)."""
        tokenizer = self.get_tokenizer()
        # Use text sequences
        test_texts = [
            "Unless required by applicable law or agreed to in writing, software distributed under the License.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        # Set pad token if not already set
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

        # Move to device
        batch = data_collator([tokenizer(text) for text in test_texts])
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def get_hf_to_te_converter(self) -> Callable:
        """Return the HF to TE conversion function."""
        return convert_llama_hf_to_te

    def get_te_to_hf_converter(self) -> Callable:
        """Return the TE to HF conversion function."""
        return convert_llama_te_to_hf

    def get_tolerances(self) -> TestTolerances:
        """Return LLaMA3-specific test tolerances."""
        return TestTolerances(
            golden_value_loss_atol=5e-3,
            golden_value_loss_rtol=0.01,
            golden_value_logits_atol=1.5,
            golden_value_logits_rtol=0.01,
            # Higher CP tolerances due to causal LM boundary effects
            cp_loss_atol=0.5,
            cp_loss_rtol=0.25,
        )

    # ==================== LLaMA3-Specific Overrides ====================

    def create_inference_params(self, config, batch_size=1, max_seq_len=256, num_beams=1):
        """Create HFInferenceParams for the given config."""
        past_key_values = HFInferenceParams(
            max_batch_size=batch_size * num_beams,
            max_sequence_length=max_seq_len,
            num_heads_kv=config.num_key_value_heads,
            head_dim_k=config.hidden_size // config.num_attention_heads,
            dtype=torch.bfloat16,
            qkv_format="thd",
            max_ctx_len=max_seq_len,
        )
        for layer_number in range(1, config.num_hidden_layers + 1):
            past_key_values.allocate_memory(layer_number)
        return past_key_values

    def test_golden_values(self, input_format):  # pyright: ignore[reportIncompatibleMethodOverride]
        """For llama3, we can test both the dynamic sequence packing and native bshd attention formats."""
        model_hf = self.get_reference_model(dtype=torch.bfloat16)
        model_te = self.get_converted_te_model(attn_input_format=input_format, dtype=torch.bfloat16)

        # Prepare input data
        input_data = self.get_test_input_data("bshd")

        # Run forward pass
        with torch.no_grad():
            te_outputs = model_te(**input_data)
            hf_outputs = model_hf(**input_data)

        # Compare outputs
        self.compare_outputs(
            te_outputs,
            hf_outputs,
            input_data,
            compare_loss=True,
            compare_logits=True,
            compare_hidden_states=False,
        )

    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_quantized_model_init_forward_and_backward(self, fp8_recipe, input_format, tie_word_embeddings):  # pyright: ignore[reportIncompatibleMethodOverride]
        """There was a weird bug in BIO-217 on tied weights with quantized model init, so we test both cases."""
        super().test_quantized_model_init_forward_and_backward(
            fp8_recipe, input_format, tie_word_embeddings=tie_word_embeddings
        )
