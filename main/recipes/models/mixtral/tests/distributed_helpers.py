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

"""Shared test utilities for distributed (EP/FSDP) tests."""

import os
from dataclasses import dataclass, field

import torch

from modeling_mixtral_te import NVMixtralConfig


def create_small_mixtral_config(**overrides) -> NVMixtralConfig:
    """Create a small Mixtral config suitable for testing."""
    defaults = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 128,
        "vocab_size": 1000,
        "attn_input_format": "bshd",
        "self_attn_mask_type": "causal",
        "router_jitter_noise": 0.0,
    }
    defaults.update(overrides)
    return NVMixtralConfig(**defaults)


def get_dummy_batch(vocab_size: int, seq_len: int = 32, batch_size: int = 2, device: str = "cuda"):
    """Create a simple dummy batch for testing."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass(frozen=True)
class DistributedConfig:
    """Distributed environment configuration."""

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))
    _master_addr: str = field(default_factory=lambda: os.environ.setdefault("MASTER_ADDR", "localhost"))
    _master_port: str = field(default_factory=lambda: os.environ.setdefault("MASTER_PORT", "12355"))

    def is_main_process(self) -> bool:
        """Return True if this is the global rank 0 process."""
        return self.rank == 0
