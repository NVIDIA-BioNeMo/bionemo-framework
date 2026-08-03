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

"""Sanity tests for SimpleLlamaForCausalLM."""

import torch
from config import SimpleLlamaConfig
from modeling_simple_llama import SimpleLlamaForCausalLM


def test_forward_pass():
    config = SimpleLlamaConfig()
    model = SimpleLlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    output = model(input_ids=input_ids)
    assert output.logits.shape == (2, 16, config.vocab_size)


def test_backward_pass():
    config = SimpleLlamaConfig()
    model = SimpleLlamaForCausalLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    labels = input_ids.clone()
    output = model(input_ids=input_ids, labels=labels)
    output.loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_loss_decreases():
    torch.manual_seed(42)
    config = SimpleLlamaConfig()
    model = SimpleLlamaForCausalLM(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fixed batch to overfit on
    input_ids = torch.randint(5, config.vocab_size, (4, 32))
    labels = input_ids.clone()

    losses = []
    for _ in range(20):
        output = model(input_ids=input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(output.loss.item())

    assert losses[-1] < losses[0]


def test_gqa_dimensions():
    config = SimpleLlamaConfig(num_attention_heads=4, num_key_value_heads=2)
    model = SimpleLlamaForCausalLM(config)
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.out_features == 4 * (config.hidden_size // 4)
    assert attn.k_proj.out_features == 2 * (config.hidden_size // 4)
    assert attn.v_proj.out_features == 2 * (config.hidden_size // 4)
