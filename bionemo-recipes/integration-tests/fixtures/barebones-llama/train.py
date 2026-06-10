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

"""Basic training script for SimpleLlamaForCausalLM."""

import torch
from config import SimpleLlamaConfig
from modeling_simple_llama import SimpleLlamaForCausalLM


def main():
    """Train a minimal Llama model for a few steps to verify it works."""
    torch.manual_seed(42)
    config = SimpleLlamaConfig()
    model = SimpleLlamaForCausalLM(config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Fixed batch to overfit on — proves the model can learn
    input_ids = torch.randint(5, config.vocab_size, (4, 32))
    labels = input_ids.clone()
    labels[:, :5] = -100

    losses = []
    for step in range(20):
        output = model(input_ids=input_ids, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        print(f"Step {step}: loss={loss.item():.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    print("Training complete!")


if __name__ == "__main__":
    main()
