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
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import Module

from .codon_embedding import CodonEmbedding
from .encodon_layer import EncoderLayer


@dataclass
class EnCodonOutput:
    """Base class for EnCodon model's outputs."""

    logits: torch.FloatTensor = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[torch.FloatTensor] = None


class EnCodon(nn.Module):
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
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            nn.Linear(config.hidden_size, config.vocab_size),
        )
        self._init_weights()

    def reset_cls_parameters(self):
        """Resets the parameters of the classification head."""
        for module in self.cls.modules():
            if isinstance(module, nn.Linear):
                # We don't use the name-based scaling for the classification head
                gain = self.config.initializer_range * math.sqrt(math.log(2 * self.config.num_hidden_layers))
                nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def _init_weights(self):
        """Initializes the weights of the model using the MAGNETO initialization scheme."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                is_qk = "query" in name or "key" in name
                # This scaling factor is part of a custom initialization strategy.
                # It may be derived from experimental results or a specific theoretical motivation
                # to stabilize training for this model architecture.
                scale_factor = math.sqrt(math.log(2 * self.config.num_hidden_layers))
                scale_value = self.config.initializer_range * scale_factor
                gain = 1.0 if is_qk else scale_value
                nn.init.xavier_normal_(module.weight, gain=gain)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[self.config.pad_token_id].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

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
        """Creates a broadcastable attention mask from a 2D or 3D input mask.
        The resulting mask is suitable for use in self-attention mechanisms where
        it can be added to the attention scores.

        - If `attention_mask` is 2D (batch_size, seq_length), it's expanded to
          (batch_size, 1, 1, seq_length).
        - If `attention_mask` is 3D (batch_size, seq_length, seq_length), it's expanded to
          (batch_size, 1, seq_length, seq_length).

        The mask values are inverted (1s become 0s, 0s become a large negative number),
        so that masked positions have a large negative value and non-masked positions are 0.
        """  # noqa: D205
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

    def forward(  # noqa: D417
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        extract_embeddings_only: bool = False,
        **kwargs,
    ) -> EnCodonOutput:
        """Performs the forward pass of the EnCodon model.

        Args:
            input_ids: Tensor of input token ids. Shape (batch_size, sequence_length).
            attention_mask: Tensor indicating which tokens to attend to.
                            Shape (batch_size, sequence_length).

        Returns:
            An `EnCodonOutput` object containing the logits and the last hidden state.
        """
        hidden_states = self.embeddings(input_ids=input_ids)
        input_shape = hidden_states.size()[:-1]

        extended_attention_mask: torch.Tensor = self._get_extended_attention_mask(
            attention_mask, input_shape, device=input_ids.device, dtype=next(self.parameters()).dtype
        )
        all_hidden_states = [] if return_hidden_states else None
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(
                hidden_states,
                extended_attention_mask,
            )
            # print(f"after layer {i}")
            # print(layer_outputs[0][0][0])
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
