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

"""Eden (Llama 3.1 variant) model providers for the Eden recipe.

Eden models are Llama 3.1 architecture variants
by BCR (Bio-Computing Research). They inherit from Megatron Bridge's
Llama31ModelProvider and override architecture-specific defaults.
"""

from dataclasses import dataclass
from typing import Type

from megatron.bridge.models.llama.llama_provider import Llama31ModelProvider


@dataclass
class EdenModelProvider(Llama31ModelProvider):
    """Eden base provider (~8B params, Llama 3.1 architecture).

    Inherits from Llama31ModelProvider which sets rope_scaling=True and
    rope_scaling_factor=8.0. Eden overrides seq_length to 8192 (training
    context) and sets the standard Eden defaults.
    """

    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32
    num_query_groups: int = 8
    init_method_std: float = 0.02
    make_vocab_size_divisible_by: int = 8
    share_embeddings_and_output_weights: bool = False
    params_dtype: type = None  # Will be set to torch.bfloat16 at runtime


@dataclass
class Eden100MModelProvider(EdenModelProvider):
    """Eden ~100M provider (Llama 3.1 architecture)."""

    num_layers: int = 12
    hidden_size: int = 512
    ffn_hidden_size: int = 2048
    num_attention_heads: int = 8
    num_query_groups: int = 2


@dataclass
class Eden300MModelProvider(EdenModelProvider):
    """Eden ~300M provider (Llama 3.1 architecture)."""

    num_layers: int = 20
    hidden_size: int = 1024
    ffn_hidden_size: int = 4096
    num_attention_heads: int = 16
    num_query_groups: int = 4


@dataclass
class Eden1BModelProvider(EdenModelProvider):
    """Eden ~1B provider (Llama 3.1 architecture)."""

    num_layers: int = 16
    hidden_size: int = 2048
    ffn_hidden_size: int = 8192
    num_attention_heads: int = 16
    num_query_groups: int = 8


@dataclass
class Eden11BModelProvider(EdenModelProvider):
    """Eden ~11B provider (Llama 3.1 architecture)."""

    num_layers: int = 36
    hidden_size: int = 5120
    ffn_hidden_size: int = 13824
    num_attention_heads: int = 40
    num_query_groups: int = 8


@dataclass
class Eden18BModelProvider(EdenModelProvider):
    """Eden ~18B provider (Llama 3.1 architecture)."""

    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 48
    num_query_groups: int = 8


@dataclass
class Eden21BModelProvider(EdenModelProvider):
    """Eden ~21B provider (Llama 3.1 architecture)."""

    num_layers: int = 42
    hidden_size: int = 7168
    ffn_hidden_size: int = 19456
    num_attention_heads: int = 56
    num_query_groups: int = 8


@dataclass
class Eden24BModelProvider(EdenModelProvider):
    """Eden ~24B provider (Llama 3.1 architecture, 32K context)."""

    seq_length: int = 32768
    num_layers: int = 46
    hidden_size: int = 6144
    ffn_hidden_size: int = 23296
    num_attention_heads: int = 48
    num_query_groups: int = 8


@dataclass
class Eden27BModelProvider(EdenModelProvider):
    """Eden ~27B provider (Llama 3.1 architecture, 32K context)."""

    seq_length: int = 32768
    num_layers: int = 46
    hidden_size: int = 6656
    ffn_hidden_size: int = 23296
    num_attention_heads: int = 52
    num_query_groups: int = 8


@dataclass
class Eden28BModelProvider(EdenModelProvider):
    """Eden ~28B provider (Llama 3.1 architecture)."""

    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 26368
    num_attention_heads: int = 48
    num_query_groups: int = 8


@dataclass
class Eden35BModelProvider(EdenModelProvider):
    """Eden ~35B provider (Llama 3.1 architecture)."""

    num_layers: int = 64
    hidden_size: int = 7168
    ffn_hidden_size: int = 20480
    num_attention_heads: int = 56
    num_query_groups: int = 8


def patch_eden_tokenizer(tokenizer):
    """Patch the byte-level tokenizer to use Eden-specific special token IDs.

    Eden training uses BOS=1, EOS=2, SEP=3, PAD=0 instead of the defaults.

    Note: Not called automatically by predict/train/infer — those load tokenizer
    assets from the checkpoint which already contain the correct IDs.  This
    utility is provided for callers that construct a fresh tokenizer outside
    the normal checkpoint flow.
    """
    bos_id, eos_id, sep_id, pad_id = 1, 2, 3, 0
    tokenizer._bos_id = bos_id
    tokenizer._eos_id = eos_id
    tokenizer._sep_id = sep_id
    tokenizer._pad_id = pad_id


EDEN_MODEL_OPTIONS: dict[str, Type[EdenModelProvider]] = {
    "eden_100m": Eden100MModelProvider,
    "eden_300m": Eden300MModelProvider,
    "eden_1b": Eden1BModelProvider,
    "eden_7b": EdenModelProvider,
    "eden_11b": Eden11BModelProvider,
    "eden_18b": Eden18BModelProvider,
    "eden_21b": Eden21BModelProvider,
    "eden_24b": Eden24BModelProvider,
    "eden_27b": Eden27BModelProvider,
    "eden_28b": Eden28BModelProvider,
    "eden_35b": Eden35BModelProvider,
}


__all__ = [
    "EDEN_MODEL_OPTIONS",
    "Eden1BModelProvider",
    "Eden11BModelProvider",
    "Eden18BModelProvider",
    "Eden21BModelProvider",
    "Eden24BModelProvider",
    "Eden27BModelProvider",
    "Eden28BModelProvider",
    "Eden35BModelProvider",
    "Eden100MModelProvider",
    "Eden300MModelProvider",
    "EdenModelProvider",
    "patch_eden_tokenizer",
]
