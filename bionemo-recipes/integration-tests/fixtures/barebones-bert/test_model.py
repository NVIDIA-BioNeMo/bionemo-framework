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

import torch
from config import SimpleBertConfig
from modeling_simple_bert import SimpleBertForMaskedLM


def _make_model():
    config = SimpleBertConfig()
    return SimpleBertForMaskedLM(config)


def _make_inputs(config, batch_size=4, seq_len=32):
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))
    labels = torch.full_like(input_ids, -100)
    mask_pos = torch.rand(batch_size, seq_len) < 0.15
    labels[mask_pos] = input_ids[mask_pos]
    input_ids[mask_pos] = config.mask_token_id
    return input_ids, labels


def test_forward_pass():
    model = _make_model()
    input_ids, labels = _make_inputs(model.config)
    output = model(input_ids, labels=labels)
    assert output.logits.shape == (4, 32, model.config.vocab_size)
    assert output.loss is not None
    assert output.loss.item() > 0


def test_backward_pass():
    model = _make_model()
    input_ids, labels = _make_inputs(model.config)
    output = model(input_ids, labels=labels)
    output.loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_loss_decreases():
    model = _make_model()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids, labels = _make_inputs(model.config)

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        output.loss.backward()
        optimizer.step()
        losses.append(output.loss.item())

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
