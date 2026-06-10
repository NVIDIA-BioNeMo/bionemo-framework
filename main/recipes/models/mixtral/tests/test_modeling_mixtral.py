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

"""Tests for the Mixtral MoE model.

This file provides comprehensive tests for the Mixtral model including:
- Common tests from the test library (meta device init, golden values, conversion, FP8)
- Mixtral-specific tests
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
from convert import convert_mixtral_hf_to_te, convert_mixtral_te_to_hf
from modeling_mixtral_te import HFInferenceParams, NVMixtralConfig, NVMixtralForCausalLM
from tests.common import BaseModelTest, TestTolerances


class TestMixtralModel(BaseModelTest):
    """Model tester for Mixtral.

    This class provides Mixtral-specific configuration for the common test suite.
    """

    is_autoregressive = True

    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the Mixtral TE model class."""
        return NVMixtralForCausalLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the Mixtral config class."""
        return NVMixtralConfig

    def get_upstream_model_id(self) -> str:
        """Return the upstream HuggingFace model ID."""
        return "NeuralNovel/Mini-Mixtral-v0.2"

    def get_upstream_model_revision(self) -> str:
        """Return the specific revision for the upstream model."""
        return "2fb530d"

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return the Mixtral tokenizer."""
        tokenizer = AutoTokenizer.from_pretrained(self.get_upstream_model_id())
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # TE only supports right-padding for BSHD inputs, left-padding (Mixtral default) causes issues with RoPE and
        # attention calculations.
        tokenizer.padding_side = "right"
        return tokenizer

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return the upstream HuggingFace model class."""
        return transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM

    def create_test_config(self, **kwargs) -> PretrainedConfig:
        # Limit the number of hidden layers to 2 for faster tests.
        return super().create_test_config(num_hidden_layers=2, **kwargs)

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers."""
        return list(model.model.layers)  # type: ignore

    def get_reference_model(
        self, dtype: torch.dtype = torch.bfloat16, attn_implementation: str = "flash_attention_2"
    ) -> PreTrainedModel:
        """Return the reference HuggingFace model."""
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping Mixtral reference model test in CI, requires Mini-Mixtral download ~25GB")
        return super().get_reference_model(dtype=dtype, attn_implementation=attn_implementation)

    def get_reference_model_no_weights(self, **kwargs) -> PreTrainedModel:
        # Limit the number of hidden layers to 2 for faster tests.
        return super().get_reference_model_no_weights(num_hidden_layers=2, **kwargs)

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
        return convert_mixtral_hf_to_te

    def get_te_to_hf_converter(self) -> Callable:
        """Return the TE to HF conversion function."""
        return convert_mixtral_te_to_hf

    def get_tolerances(self) -> TestTolerances:
        """Return Mixtral-specific test tolerances."""
        return TestTolerances(
            golden_value_loss_atol=5e-3,
            golden_value_loss_rtol=0.01,
            golden_value_logits_atol=1.5,
            golden_value_logits_rtol=0.01,
            cp_loss_atol=0.5,
            cp_loss_rtol=0.25,
        )

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


# ---------------------------------------------------------------------------
# Single-GPU tests for the AllToAll dispatch/combine code path
#
# By initialising a single-rank NCCL process group and setting an EP group on
# the model, we force the AllToAllTokenDispatcher to take the all-to-all path
# (differentiable all-to-all, expert sort/unsort, etc.) even on one GPU.
# This ensures CI coverage of that code without requiring multiple GPUs.
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def dist_process_group():
    """Initialize and tear down a single-rank NCCL process group."""
    if torch.distributed.is_initialized():
        pytest.skip("Distributed already initialized")
    torch.cuda.set_device(0)
    store = torch.distributed.HashStore()
    torch.distributed.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
    yield
    torch.distributed.destroy_process_group()


def _small_config():
    """Create a small Mixtral config for single-GPU EP tests."""
    return NVMixtralConfig(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        vocab_size=1000,
        attn_input_format="bshd",
        self_attn_mask_type="causal",
        router_jitter_noise=0.0,
    )


def _dummy_batch(vocab_size, device="cuda"):
    """Create a deterministic dummy batch."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (2, 32), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@requires_cuda
def test_alltoall_forward_matches_local(dist_process_group):
    """All-to-all code path produces the same output as the local-only path."""
    from torch.distributed.tensor.device_mesh import DeviceMesh

    config = _small_config()
    batch = _dummy_batch(config.vocab_size)

    # Reference: local-only path (no EP group set)
    torch.manual_seed(0)
    model_local = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    model_local.eval()
    with torch.no_grad():
        out_local = model_local(**batch)

    # Test: all-to-all path (EP group set on a single-rank mesh)
    torch.manual_seed(0)
    model_ep = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    ep_mesh = DeviceMesh("cuda", [0])
    model_ep.model.set_ep_groups(ep_mesh.get_group(), ep_mesh)
    model_ep.eval()
    with torch.no_grad():
        out_ep = model_ep(**batch)

    torch.testing.assert_close(out_ep.logits, out_local.logits, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_ep.loss, out_local.loss, atol=1e-5, rtol=1e-5)


@requires_cuda
def test_alltoall_backward_all_params_have_gradients(dist_process_group):
    """All trainable parameters receive gradients through the all-to-all code path."""
    from torch.distributed.tensor.device_mesh import DeviceMesh

    config = _small_config()
    batch = _dummy_batch(config.vocab_size)

    torch.manual_seed(0)
    model = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    ep_mesh = DeviceMesh("cuda", [0])
    model.model.set_ep_groups(ep_mesh.get_group(), ep_mesh)

    outputs = model(**batch)
    outputs.loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient with all-to-all code path"


@requires_cuda
def test_alltoall_backward_gradients_match_local(dist_process_group):
    """All-to-all backward produces the same gradients as the local-only path."""
    from torch.distributed.tensor.device_mesh import DeviceMesh

    config = _small_config()
    batch = _dummy_batch(config.vocab_size)

    # Reference: local-only path
    torch.manual_seed(0)
    model_local = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    model_local(**batch).loss.backward()
    ref_grads = {name: p.grad.detach().clone() for name, p in model_local.named_parameters() if p.grad is not None}

    # Test: all-to-all path
    torch.manual_seed(0)
    model_ep = NVMixtralForCausalLM(config).to(dtype=torch.bfloat16, device="cuda")
    ep_mesh = DeviceMesh("cuda", [0])
    model_ep.model.set_ep_groups(ep_mesh.get_group(), ep_mesh)
    model_ep(**batch).loss.backward()

    for name, param in model_ep.named_parameters():
        if param.requires_grad:
            g = param.grad
            if hasattr(g, "full_tensor"):
                g = g.full_tensor()
            assert name in ref_grads, f"Unexpected gradient for {name}"
            torch.testing.assert_close(
                g,
                ref_grads[name],
                atol=1e-5,
                rtol=1e-5,
                msg=f"All-to-all gradient mismatch for {name}",
            )
