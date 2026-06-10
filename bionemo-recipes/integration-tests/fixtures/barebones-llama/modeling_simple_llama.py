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

"""Minimal Llama-style causal LM with Group Query Attention.

This is a vanilla PyTorch implementation with NO TransformerEngine.
It uses separate Q/K/V projections and SwiGLU FFN, which are the key
patterns that TE conversion needs to handle (Q/K/V fusion, gate/up fusion).
"""

from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SimpleLlamaConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size, eps=1e-5):  # noqa: D107
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):  # noqa: D102
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(x.dtype)


def apply_rotary_pos_emb(x, cos, sin):  # noqa: D103
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1).flatten(-2)
    return x * cos + rotated * sin


class SimpleLlamaAttention(nn.Module):
    """GQA attention with separate Q/K/V projections."""

    def __init__(self, config):  # noqa: D107
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # NOTE: Separate projections — TE conversion will fuse these
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin):  # noqa: D102
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # GQA: repeat K/V heads
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn)


class SimpleLlamaMLP(nn.Module):
    """SwiGLU FFN with separate gate/up projections."""

    def __init__(self, config):  # noqa: D107
        super().__init__()
        # NOTE: Separate gate and up projections — TE conversion will fuse these
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):  # noqa: D102
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleLlamaLayer(nn.Module):
    """Single Llama transformer layer with pre-norm and residual connections."""

    def __init__(self, config):  # noqa: D107
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SimpleLlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SimpleLlamaMLP(config)

    def forward(self, hidden_states, cos, sin):  # noqa: D102
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class SimpleLlamaModel(PreTrainedModel):
    """Llama base model with token embeddings and transformer layers."""

    config_class = SimpleLlamaConfig

    def __init__(self, config):  # noqa: D107
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([SimpleLlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Precompute rotary embeddings — repeat each freq for the interleaved (even/odd) layout
        head_dim = config.hidden_size // config.num_attention_heads
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(config.max_position_embeddings).float()
        emb = torch.outer(t, freqs)
        # Repeat each frequency so shape is (seq, head_dim) matching x[..., ::2]/x[..., 1::2] interleaving
        cos = emb.cos().repeat_interleave(2, dim=-1)
        sin = emb.sin().repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", cos[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", sin[None, None, :, :], persistent=False)
        self.post_init()

    def forward(self, input_ids, **kwargs):  # noqa: D102
        hidden_states = self.embed_tokens(input_ids)
        seq_len = input_ids.size(1)
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)

        return self.norm(hidden_states)


class SimpleLlamaForCausalLM(PreTrainedModel):
    """Llama causal language model with tied embeddings."""

    config_class = SimpleLlamaConfig
    _tied_weights_keys: ClassVar[dict[str, str]] = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):  # noqa: D107
        super().__init__(config)
        self.model = SimpleLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):  # noqa: D102
        return self.model.embed_tokens

    def get_output_embeddings(self):  # noqa: D102
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):  # noqa: D102
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(loss=loss, logits=logits)
