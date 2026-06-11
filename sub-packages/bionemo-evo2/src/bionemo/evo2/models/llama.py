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


from dataclasses import dataclass
from typing import Optional

import torch
from nemo.collections import llm
from nemo.collections.llm.gpt.model.llama import HFLlamaImporter, LlamaModel
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import io
from nemo.lightning.pytorch.utils import dtype_from_hf


@dataclass
class EdenConfig(llm.Llama31Config8B):
    """Eden-flavoured Llama-3.1 ~8B (keeps all Eden behaviors). Inherits from the llama 3.1 config for proper handling of RoPE when converting checkpoints."""

    rotary_base: int = 500_000
    seq_length: int = 8192
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32

    scale_factor: int = 1
    low_freq_factor: int = 1
    high_freq_factor: int = 4
    old_context_len: int = 8192
    init_method_std: float = 0.02
    embedding_init_method_std: Optional[float] = None


@dataclass
class Eden11BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~14B (keeps all Eden behaviors)."""

    # If you want long context like Eden-long, bump this; else inherit 8192.
    seq_length: int = 8192  # or remove this line to keep 8192

    # ~14B sizing (head_dim ≈ 128)
    num_layers: int = 36
    hidden_size: int = 5120
    ffn_hidden_size: int = 13824
    num_attention_heads: int = 40
    num_query_groups: int = 8  # GQA (inherited value is also fine if already 8)


@dataclass
class Eden18BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~18B (keeps all Eden behaviors)."""

    # If you want long context like Eden-long, bump this; else inherit 8192.
    seq_length: int = 8192  # or remove this line to keep 8192

    # ~18B sizing (head_dim ≈ 128)
    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 16384
    num_attention_heads: int = 48
    num_query_groups: int = 8  # GQA (inherited value is also fine if already 8)
    old_context_len: int = 8192  # or remove this line to keep 8192


@dataclass
class Eden21BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~21B (keeps all Eden behaviors)."""

    seq_length: int = 8192

    # ~21B sizing (head_dim = 128)
    num_layers: int = 42  # 42 layers for 21B target
    hidden_size: int = 7168  # 56 * 128 = 7168 for exact head_dim
    ffn_hidden_size: int = 19456  # ~2.7x hidden_size
    num_attention_heads: int = 56  # Divisible by 8
    num_query_groups: int = 8  # GQA
    old_context_len: int = 8192


@dataclass
class Eden24BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~8B (keeps all Eden behaviors)."""

    # If you want long context like Eden-long, bump this; else inherit 8192.
    seq_length: int = 32768  # or remove this line to keep 8192

    # ~8B sizing (head_dim ≈ 128)
    num_layers: int = 46
    hidden_size: int = 6144
    ffn_hidden_size: int = 23296
    num_attention_heads: int = 48
    num_query_groups: int = 8  # GQA (inherited value is also fine if already 8)
    old_context_len: int = 8192


@dataclass
class Eden27BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~8B (keeps all Eden behaviors)."""

    # If you want long context like Eden-long, bump this; else inherit 8192.
    seq_length: int = 32768  # or remove this line to keep 8192

    # ~8B sizing (head_dim ≈ 128)
    num_layers: int = 46
    hidden_size: int = 6656
    ffn_hidden_size: int = 23296
    num_attention_heads: int = 52
    num_query_groups: int = 8  # GQA (inherited value is also fine if already 8)
    old_context_len: int = 8192


@dataclass
class Eden28BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~28B (keeps all Eden behaviors)."""

    # If you want long context like Eden-long, bump this; else inherit 8192.
    seq_length: int = 8192  # or remove this line to keep 8192

    # ~8B sizing (head_dim ≈ 128)
    num_layers: int = 48
    hidden_size: int = 6144
    ffn_hidden_size: int = 26368
    num_attention_heads: int = 48
    num_query_groups: int = 8  # GQA (inherited value is also fine if already 8)
    old_context_len: int = 8192  # or remove this line to keep 8192


@dataclass
class Eden35BConfig(EdenConfig):
    """Eden-flavoured Llama-3.1 ~35B (keeps all Eden behaviors)."""

    seq_length: int = 8192

    # ~35B sizing (head_dim ≈ 128)
    num_layers: int = 64
    hidden_size: int = 7168
    ffn_hidden_size: int = 20480
    num_attention_heads: int = 56
    num_query_groups: int = 8  # GQA
    old_context_len: int = 8192


@io.model_importer(LlamaModel, "hf")
class HFEdenLlamaImporter(HFLlamaImporter):
    """Importer for Eden-flavoured Llama models which just overrides the tokenizer and config classes from NeMo."""

    @property
    def config(self) -> EdenConfig:
        """Create a NeMo LlamaConfig from the HF model config.

        Translates the HF configuration parameters to the equivalent NeMo
        configuration.

        Returns:
            LlamaConfig: NeMo configuration for Llama models
        """
        from transformers import AutoConfig, GenerationConfig

        source = AutoConfig.from_pretrained(str(self))
        try:
            generation_config = GenerationConfig.from_pretrained(str(self))
        except Exception:
            generation_config = None

        def make_vocab_size_divisible_by(vocab_size):
            base = 128
            while vocab_size % base != 0:
                base //= 2
            return base

        cls = EdenConfig
        scale_factor = source.rope_scaling.get("factor", 8.0) if source.rope_scaling is not None else 8.0

        args = {}

        output = cls(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=(
                source.intermediate_size
                if not getattr(source, "intermediate_size_mlp", None)
                else source.intermediate_size_mlp
            ),
            num_attention_heads=source.num_attention_heads,
            init_method_std=source.initializer_range,
            layernorm_epsilon=source.rms_norm_eps,
            num_query_groups=source.num_key_value_heads,
            seq_length=source.max_position_embeddings,
            rotary_base=source.rope_theta,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=make_vocab_size_divisible_by(source.vocab_size),
            share_embeddings_and_output_weights=getattr(source, "tie_word_embeddings", False),
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
            generation_config=generation_config,
            vocab_size=source.vocab_size,
            kv_channels=getattr(source, "head_dim", None),
            scale_factor=scale_factor,
            **args,
        )

        return output

    @property
    def tokenizer(self):
        """Override the tokenizer to use the Eden-flavoured tokenizer."""
        from bionemo.evo2.run.utils import patch_eden_tokenizer  # avoid circular import

        tokenizer = get_nmt_tokenizer("byte-level")
        patch_eden_tokenizer(tokenizer)
        return tokenizer


LLAMA_MODEL_OPTIONS = {
    "8B": lambda **kwargs: llm.Llama3Config8B(**kwargs),
    "7B": lambda **kwargs: EdenConfig(**kwargs),
    "11B": lambda **kwargs: Eden11BConfig(**kwargs),
    "18B": lambda **kwargs: Eden18BConfig(**kwargs),
    "21B": lambda **kwargs: Eden21BConfig(**kwargs),
    "24B": lambda **kwargs: Eden24BConfig(**kwargs),
    "27B": lambda **kwargs: Eden27BConfig(**kwargs),
    "28B": lambda **kwargs: Eden28BConfig(**kwargs),
    "35B": lambda **kwargs: Eden35BConfig(**kwargs),
}
