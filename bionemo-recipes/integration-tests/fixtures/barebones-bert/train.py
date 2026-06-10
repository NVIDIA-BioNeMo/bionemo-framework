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


def main():
    """Train a minimal BERT model for a few steps to verify it works."""
    config = SimpleBertConfig()
    model = SimpleBertForMaskedLM(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Random training data
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_len))

    # Create labels: mask ~15% of tokens, rest are -100 (ignored)
    labels = torch.full_like(input_ids, -100)
    mask_positions = torch.rand(batch_size, seq_len) < 0.15
    labels[mask_positions] = input_ids[mask_positions]
    input_ids[mask_positions] = config.mask_token_id

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        output = model(input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Step {step}: loss={loss.item():.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("Training complete. Loss decreased successfully.")


if __name__ == "__main__":
    main()
