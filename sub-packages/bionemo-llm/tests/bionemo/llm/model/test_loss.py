# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import torch
import torch.nn.functional as F
from megatron.core.models.common.language_module import language_module
from megatron.core.transformer import transformer_config
from nemo.lightning import megatron_parallel

from bionemo.llm.model import loss as bionemo_loss
from bionemo.llm.model.loss import unreduced_token_loss_fn
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.lightning import get_random_microbatch


def test_loss_equivalency_nemo_vs_pytorch():
    # Setup no grad and megatron distributed contexts for the test
    with torch.no_grad(), megatron_parallel_state_utils.distributed_model_parallel_state():
        # Define the batch size, sequence length, and number of tokens
        batch_size = 2
        sequence_length = 5
        num_tokens = 31

        # Generate random logits (batch_size x sequence_length x num_tokens) with
        #   mean 0 and standard deviation 10
        logits = torch.randn(batch_size, sequence_length, num_tokens, dtype=torch.float32).cuda() * 10

        # Generate target sequences (batch_size x sequence_length) with random integers
        target = torch.randint(0, num_tokens, (batch_size, sequence_length), dtype=torch.long).cuda()

        # Generate a loss mask (batch_size x sequence_length) with random 0s and 1s
        loss_mask = torch.randint(0, 2, (batch_size, sequence_length), dtype=bool).cuda()

        ####################
        # Base case: Calculate the cross-entropy loss of masked tokens using the vanilla pytorch function.
        expected_loss = F.cross_entropy(logits[loss_mask], target[loss_mask], reduction="mean")

        ####################
        # Part 1) get the loss using NeMo/Megatron's default strategy of
        #  a. computing the first part of the loss  inside of the forward pass of the model
        #     (through a call to `compute_language_model_loss`)
        #  b. passing this through the forward of MaskedTokenLossReduction, which is executed
        #     in parallel across GPUs and owns reducing within a parllel group.
        #  c. A final reduction across parallel groups through a call to `reduce`
        dummy_model = language_module.LanguageModule(
            config=transformer_config.TransformerConfig(
                num_layers=1,
                hidden_size=64,
                ffn_hidden_size=128,
                num_attention_heads=1,
                kv_channels=None,
            )
        )
        # Transpose the logits from (batch_size x sequence_length x num_tokens) to (sequence_length x batch_size x num_tokens)
        #  since this is what `compute_language_model_loss` expects.
        unreduced_megatron_loss = dummy_model.compute_language_model_loss(target, logits.transpose(0, 1).contiguous())
        nemo_default_loss_fn = megatron_parallel.MaskedTokenLossReduction()
        batch_megatron = {
            "loss_mask": loss_mask,
        }
        forward_nemo_loss = nemo_default_loss_fn.forward(
            batch=batch_megatron,
            forward_out=unreduced_megatron_loss,  # wants the loss directly
        )
        final_nemo_loss = nemo_default_loss_fn.reduce([forward_nemo_loss[2]])

        # First check, nemo+megatron loss
        torch.testing.assert_close(expected_loss, final_nemo_loss)


def test_loss_equivalency_bionemo_vs_pytorch():
    # Setup no grad and megatron distributed contexts for the test
    with torch.no_grad(), megatron_parallel_state_utils.distributed_model_parallel_state():
        # Define the batch size, sequence length, and number of tokens
        batch_size = 2
        sequence_length = 5
        num_tokens = 31

        # Generate random logits (batch_size x sequence_length x num_tokens) with
        #   mean 0 and standard deviation 10
        logits = torch.randn(batch_size, sequence_length, num_tokens, dtype=torch.float32).cuda() * 10

        # Generate target sequences (batch_size x sequence_length) with random integers
        target = torch.randint(0, num_tokens, (batch_size, sequence_length), dtype=torch.long).cuda()

        # Generate a loss mask (batch_size x sequence_length) with random 0s and 1s
        loss_mask = torch.randint(0, 2, (batch_size, sequence_length), dtype=bool).cuda()

        ####################
        # Base case: Calculate the cross-entropy loss of masked tokens using the vanilla pytorch function.
        expected_loss = F.cross_entropy(logits[loss_mask], target[loss_mask], reduction="mean")
        ####################
        # Part 2) get the loss using BioNeMo's default strategy of
        #  a. passing model logits through the forward of MaskedTokenLossReduction, which is executed
        #     in parallel across GPUs and owns reducing within a parllel group. This combines parts a and b of the
        #     NeMo/Megatron strategy into a single step, and doesn't expect the model to compute loss in forward.
        #  b. A final reduction across parallel groups through a call to `reduce`
        # Second, check bionemo loss where model outputs logits
        bionemo_loss_fn = bionemo_loss.BERTMLMLossWithReduction()
        bionemo_model_output = {
            "token_logits": logits.transpose(
                0, 1
            ).contiguous(),  # bionemo loss function also wants logits s,b like nemo.
        }
        bionemo_batch = {
            "loss_mask": loss_mask,
            "labels": target,
        }
        forward_bionemo_loss = bionemo_loss_fn.forward(
            batch=bionemo_batch,
            forward_out=bionemo_model_output,
        )
        final_bionemo_loss = bionemo_loss_fn.reduce([forward_bionemo_loss[2]])
        torch.testing.assert_close(expected_loss, final_bionemo_loss)


def test_vocab_parallel_cross_entropy_golden_value(seed: int = 42):
    """Test tensor_parallel.vocab_parallel_cross_entropy"""
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed=seed):
        # setup test input
        microbatch_size, max_sequence_length, vocab_size = 1, 1024, 2
        microbatch_outputs = [get_random_microbatch(microbatch_size, max_sequence_length, vocab_size, seed=seed)]

        # 1. torch.nn.functional
        loss = torch.nn.functional.cross_entropy(
            input=microbatch_outputs[0]["forward_out"]["token_logits"].transpose(0, 1).contiguous().reshape(1024, 2),
            target=microbatch_outputs[0]["batch"]["labels"].flatten(),
            reduction="none",
            ignore_index=-100,
        )

        # 2. tensor_parallel.vocab_parallel_cross_entropy
        unreduced_token_loss = unreduced_token_loss_fn(
            logits=microbatch_outputs[0]["forward_out"]["token_logits"],
            labels=microbatch_outputs[0]["batch"]["labels"],
        )

        torch.testing.assert_close(
            unreduced_token_loss.flatten(),
            loss,
        )


@pytest.mark.xfail(reason="tensor_parallel.vocab_parallel_cross_entropy modifies input token_logits")
def test_vocab_parallel_cross_entropy_inplace_operation(seed: int = 42):
    """Test inplace operation on input in tensor_parallel.vocab_parallel_cross_entropy"""
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed=seed):
        # setup test input
        microbatch_size, max_sequence_length, vocab_size = 1, 1024, 2
        microbatch_outputs = [get_random_microbatch(microbatch_size, max_sequence_length, vocab_size, seed=seed)]

        token_logits_clone = microbatch_outputs[0]["forward_out"]["token_logits"].clone()
        labels_clone = microbatch_outputs[0]["batch"]["labels"].clone()

        _ = unreduced_token_loss_fn(
            logits=microbatch_outputs[0]["forward_out"]["token_logits"],
            labels=microbatch_outputs[0]["batch"]["labels"],
        )

        torch.testing.assert_allclose(microbatch_outputs[0]["batch"]["labels"], labels_clone)  # pass
        torch.testing.assert_allclose(
            microbatch_outputs[0]["forward_out"]["token_logits"], token_logits_clone
        )  # xfail
