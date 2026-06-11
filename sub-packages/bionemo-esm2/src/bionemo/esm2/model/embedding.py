# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import Literal, Optional, Sequence, Tuple

import torch
from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig
from torch import Tensor


__all__: Sequence[str] = (
    "ESM2_MASK_RATIO_TRAIN",
    "ESM2Embedding",
)

ESM2_MASK_RATIO_TRAIN = 0.15 * 0.8  # Hardcoded as the ratio used in all ESM model training runs


class ESM2Embedding(LanguageModelEmbedding):
    """ESM2 Embedding with custom logic for attention masking and token dropout."""

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal["learned_absolute", "rope"] = "rope",
        num_tokentypes: int = 0,
        # ESM2 NEW ARGS
        token_dropout: bool = True,
        use_attention_mask: bool = True,
        mask_token_id: Optional[int] = None,
    ) -> None:
        """Initialize the ESM2 Embedding module."""
        super().__init__(
            config=config,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            position_embedding_type=position_embedding_type,
            num_tokentypes=num_tokentypes,
        )
        self.token_dropout = token_dropout
        self.use_attention_mask = use_attention_mask
        self.mask_token_id = mask_token_id

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the embedding weights."""
        return self.word_embeddings.weight.dtype

    def _apply_esm2_customization(
        self, word_embeddings: Tensor, input_ids: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor | None]:
        """ESM2 customization for attention masking and token dropout.

        Args:
            word_embeddings (Tensor[float]): The input tokens. Shape: [b, s, h]
            input_ids (Tensor[int]): The input tokens. Shape: [b, s]
            attention_mask (Tensor[bool]): attention mask. Shape: [b, s]

        Returns:
            Tuple[Tensor, Tensor]: (Updated embeddings, embedding mask) Shape: ([b, s, h], [b, s])
        """
        embeddings_mask = None
        if attention_mask is not None and (self.token_dropout or self.use_attention_mask):
            embeddings_mask = attention_mask

        if embeddings_mask is not None and self.token_dropout:
            word_embeddings = word_embeddings.masked_fill((input_ids == self.mask_token_id).unsqueeze(-1), 0.0)
            src_lengths = embeddings_mask.sum(-1)
            mask_ratio_observed = (input_ids == self.mask_token_id).sum(-1).to(self.dtype) / src_lengths

            scale_factor = (1 - ESM2_MASK_RATIO_TRAIN) / (1 - mask_ratio_observed)[:, None, None]
            word_embeddings = (word_embeddings * scale_factor).to(word_embeddings.dtype)
        if embeddings_mask is not None and self.use_attention_mask:
            word_embeddings = (word_embeddings * embeddings_mask.unsqueeze(-1)).to(word_embeddings.dtype)
        return word_embeddings, embeddings_mask

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        tokentype_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens. Shape: [b, s]
            position_ids (Tensor): The position id's used to calculate position embeddings. Shape: [b, s]
            tokentype_ids (Tensor, optional): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None
            attention_mask (Tensor): attention mask. Shape: [b, s]

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)  # [b, s, h]

        # ESM2 Customization
        word_embeddings, embeddings_mask = self._apply_esm2_customization(word_embeddings, input_ids, attention_mask)

        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        # ESM2 Customization: include attention masking from ESM2
        if embeddings_mask is not None and self.use_attention_mask:
            embeddings = (embeddings * embeddings_mask.unsqueeze(-1)).to(embeddings.dtype)

        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            if self.tokentype_embeddings is None:
                raise ValueError("tokentype_embedding is needed to process tokentype_ids")
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            embeddings = tensor_parallel.scatter_to_sequence_parallel_region(embeddings)
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings
