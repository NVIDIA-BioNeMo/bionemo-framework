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


from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import transformer_engine.pytorch
from torch.nn import Module
from transformer_engine.pytorch import TransformerLayer as TETransformerLayer
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from transformers.activations import ACT2FN

from .codon_embedding import CodonEmbedding
from .encodon_te_layer import EncodonTELayer


@dataclass
class EnCodonOutput:
    """Base class for EnCodon model's outputs."""

    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[torch.FloatTensor] = None


class EnCodonTE(nn.Module):
    """EnCodon is a transformer-based model for encoding codon sequences.

    It consists of a codon embedding layer, a stack of transformer encoder layers,
    and a prediction head.
    """

    def __init__(self, config):
        """Initializes the EnCodon model.

        Args:
            config: A configuration object containing model hyperparameters.
        """
        super().__init__()
        self.config = config
        self.embeddings = CodonEmbedding(config)
        config.max_seq_length = 2048
        config.encoder_activation = "gelu"
        config.padded_vocab_size = None
        config.fuse_qkv_params = True
        config.qkv_weight_interleaved = True
        config.self_attn_mask_type = "padding"
        config.rotary_pos_interleaved = False
        if os.getenv("CODON_FM_TE_IMPL", "exact") == "exact":
            print("BRUNO: Using exact implementation")
        else:
            print("BRUNO: Using TE implementation")

        self.layers = nn.ModuleList(
            [
                (EncodonTELayer if os.getenv("CODON_FM_TE_IMPL", "exact") == "exact" else TETransformerLayer)(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    layernorm_epsilon=config.layer_norm_eps,
                    hidden_dropout=config.hidden_dropout_prob,
                    attention_dropout=config.attention_probs_dropout_prob,
                    qkv_weight_interleaved=config.qkv_weight_interleaved,
                    rotary_pos_interleaved=config.rotary_pos_interleaved,
                    layer_number=i + 1,
                    layer_type="encoder",
                    device="cpu",
                    self_attn_mask_type=config.self_attn_mask_type,
                    activation=config.encoder_activation,
                    attn_input_format=config.attn_input_format,
                    seq_length=config.max_seq_length,
                    num_gqa_groups=config.num_attention_heads,
                    fuse_qkv_params=config.fuse_qkv_params,
                    window_size=(-1, -1),
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.cls = nn.Sequential(
            transformer_engine.pytorch.Linear(config.hidden_size, config.hidden_size, device="cpu"),
            ACT2FN[config.hidden_act],
            transformer_engine.pytorch.LayerNormLinear(
                config.hidden_size,
                config.padded_vocab_size if config.padded_vocab_size is not None else config.vocab_size,
                bias=True,
                eps=config.layer_norm_eps,
                device="cpu",
            ),
        )
        self.rotary_embeddings = RotaryPositionEmbedding(config.hidden_size // config.num_attention_heads)
        self.te_rope_emb = None
        self._init_weights()

    def reset_cls_parameters(self):
        """Resets the parameters of the classification head."""
        for module in self.cls.modules():
            if isinstance(module, (nn.Linear, transformer_engine.pytorch.Linear)):
                # We don't use the name-based scaling for the classification head
                gain = self.config.initializer_range * math.sqrt(math.log(2 * self.config.num_hidden_layers))
                if getattr(module, "weight", None) is not None:
                    nn.init.xavier_normal_(module.weight, gain=gain)
                    if module.bias is not None:
                        module.bias.data.zero_()
                if getattr(module, "query_weight", None) is not None:
                    nn.init.xavier_normal_(module.query_weight, gain=gain)
                    if module.query_bias is not None:
                        module.query_bias.data.zero_()
                if getattr(module, "key_weight", None) is not None:
                    nn.init.xavier_normal_(module.key_weight, gain=gain)
                    if module.key_bias is not None:
                        module.key_bias.data.zero_()
                if getattr(module, "value_weight", None) is not None:
                    nn.init.xavier_normal_(module.value_weight, gain=gain)
                    if module.value_bias is not None:
                        module.value_bias.data.zero_()
            elif isinstance(module, (nn.LayerNorm, transformer_engine.pytorch.LayerNorm)):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, transformer_engine.pytorch.LayerNormLinear):
                module.layer_norm_weight.data.fill_(1.0)
                if module.layer_norm_bias is not None:
                    module.layer_norm_bias.data.zero_()

    def _init_weights(self):
        """Initializes the weights of the model using the MAGNETO initialization scheme."""
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, transformer_engine.pytorch.Linear)):
                is_qk = "query" in name or "key" in name
                # This scaling factor is part of a custom initialization strategy.
                # It may be derived from experimental results or a specific theoretical motivation
                # to stabilize training for this model architecture.
                scale_factor = math.sqrt(math.log(2 * self.config.num_hidden_layers))
                scale_value = self.config.initializer_range * scale_factor
                gain = 1.0 if is_qk else scale_value
                if getattr(module, "weight", None) is not None:
                    nn.init.xavier_normal_(module.weight, gain=gain)
                    if module.bias is not None:
                        module.bias.data.zero_()
                if getattr(module, "query_weight", None) is not None:
                    nn.init.xavier_normal_(module.query_weight, gain=gain)
                    if module.query_bias is not None:
                        module.query_bias.data.zero_()
                if getattr(module, "key_weight", None) is not None:
                    nn.init.xavier_normal_(module.key_weight, gain=gain)
                    if module.key_bias is not None:
                        module.key_bias.data.zero_()
                if getattr(module, "value_weight", None) is not None:
                    nn.init.xavier_normal_(module.value_weight, gain=gain)
                    if module.value_bias is not None:
                        module.value_bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[self.config.pad_token_id].zero_()
            elif isinstance(module, (nn.LayerNorm, transformer_engine.pytorch.LayerNorm)):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

            if isinstance(module, transformer_engine.pytorch.LayerNormLinear):
                module.layer_norm_weight.data.fill_(1.0)
                if module.layer_norm_bias is not None:
                    module.layer_norm_bias.data.zero_()

    def get_input_embeddings(self):  # noqa: D102
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):  # noqa: D102
        self.embeddings.word_embeddings = value

    def _get_extended_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: tuple[int],
        device: torch.device,
        dtype: torch.float,
    ) -> torch.Tensor:
        """Creates a broadcastable attention mask from a 2D or 3D input mask."""
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )
        extended_attention_mask = extended_attention_mask.to(dtype=dtype, device=device)
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        extract_embeddings_only: bool = False,
        **kwargs,
    ) -> EnCodonOutput:
        """Performs the forward pass of the EnCodon model.

        Args:
            input_ids: Tensor of input token ids.
            attention_mask: Tensor indicating which tokens to attend to.
            return_hidden_states: Whether to return the hidden states.
            extract_embeddings_only: Whether to extract the embeddings only.
            **kwargs: Additional keyword arguments.

        Returns:
            An `EnCodonOutput` object containing the logits and the last hidden state.
        """
        hidden_states = self.embeddings(input_ids=input_ids)
        input_shape = hidden_states.size()[:-1]

        all_hidden_states = [] if return_hidden_states else None
        te_rope_emb = None
        extended_attention_mask = None

        if self.config.attn_input_format == "bshd":
            extended_attention_mask: torch.Tensor = self._get_extended_attention_mask(
                attention_mask, input_shape, device=input_ids.device, dtype=next(self.parameters()).dtype
            )
            extended_attention_mask = extended_attention_mask < -1
            attention_mask = extended_attention_mask
        with torch.autocast(device_type="cuda", enabled=False):
            if self.config.attn_input_format == "bshd":
                te_rope_emb = self.rotary_embeddings(max_seq_len=hidden_states.shape[1])
            elif self.config.attn_input_format == "thd":
                te_rope_emb = self.rotary_embeddings(max_seq_len=kwargs["max_length_q"])
        te_rope_emb = te_rope_emb.to(hidden_states.device, non_blocking=True)

        for i, layer_module in enumerate(self.layers):
            if self.config.attn_input_format == "bshd":
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    rotary_pos_emb=te_rope_emb,
                    core_attention_bias_type="no_bias",
                    core_attention_bias=None,
                    attention_mask=attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states=hidden_states,
                    rotary_pos_emb=te_rope_emb,
                    attention_mask=None,
                    cu_seqlens_q=kwargs.get("cu_seq_lens_q"),
                    cu_seqlens_kv=kwargs.get("cu_seq_lens_k"),
                    max_seqlen_q=kwargs.get("max_length_q"),
                    max_seqlen_kv=kwargs.get("max_length_k"),
                )
            hidden_states = layer_outputs

            if return_hidden_states:
                all_hidden_states.append(hidden_states)

        sequence_output = hidden_states
        prediction_scores = self.cls(sequence_output) if not extract_embeddings_only else None

        return EnCodonOutput(
            logits=prediction_scores,
            last_hidden_state=hidden_states,
            all_hidden_states=all_hidden_states,
        )

    def extract_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, return_hidden_states: bool = False
    ) -> EnCodonOutput:
        """Extracts the embeddings from the model."""
        return self.forward(
            input_ids, attention_mask, return_hidden_states=return_hidden_states, extract_embeddings_only=True
        )

    def get_codon_embeddings(self) -> Module:
        """Returns the codon embedding module."""
        return self.get_input_embeddings()
