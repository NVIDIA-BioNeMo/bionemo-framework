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
import torch.nn as nn
from config import SimpleBertConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


class SimpleBertModel(PreTrainedModel):
    """Minimal BERT encoder using nn.TransformerEncoder."""

    config_class = SimpleBertConfig

    def __init__(self, config: SimpleBertConfig):  # noqa: D107
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

    def forward(self, input_ids, attention_mask=None):  # noqa: D102
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        embeddings = self.token_embeddings(input_ids) + self.position_embeddings(position_ids)
        embeddings = self.embedding_layer_norm(embeddings)
        embeddings = self.embedding_dropout(embeddings)

        # Convert attention_mask from (batch, seq) to (batch, seq) bool mask for TransformerEncoder
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0  # True = ignore

        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        return hidden_states


class SimpleBertForMaskedLM(PreTrainedModel):
    """Minimal BERT for masked language modeling."""

    config_class = SimpleBertConfig

    def __init__(self, config: SimpleBertConfig):  # noqa: D107
        super().__init__(config)
        self.bert = SimpleBertModel(config)

        # LM head: Linear -> GELU -> LayerNorm -> Linear
        self.lm_head_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head_act = nn.GELU()
        self.lm_head_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head_proj = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):  # noqa: D102
        hidden_states = self.bert(input_ids, attention_mask=attention_mask)

        # LM head
        x = self.lm_head_dense(hidden_states)
        x = self.lm_head_act(x)
        x = self.lm_head_norm(x)
        logits = self.lm_head_proj(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return MaskedLMOutput(loss=loss, logits=logits)
