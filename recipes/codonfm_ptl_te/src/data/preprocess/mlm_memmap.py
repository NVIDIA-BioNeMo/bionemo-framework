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


from typing import Any, Dict, List

import numpy as np

from src.data.metadata import MetadataFields


def process_item(  # noqa: D417
    tokenizer: Any,
    sequence_tokens: np.ndarray,
    context_length: int,
    mlm_probability: float = 0.15,
    mask_replace_prob: float = 0.8,
    random_replace_prob: float = 0.1,
    ignore_index: int = -100,
    codon_weights: np.array = None,
) -> Dict[str, List[int]]:
    """Process an item from the dataset.

    Args:
        tokenizer (`Any`):
            The tokenizer to use for tokenizing the input sequence.
        sequence_tokens (`np.ndarray`):
            The input sequence to process.
        context_length (`int`):
            The length of the context to use.
        mlm_probability (`float`, *optional*, defaults to 0.15):
            The probability with which to (randomly) mask tokens in the input, when `mlm` is set to `True`.
        mask_replace_prob (`float`, *optional*, defaults to 0.8):
            The probability with which masked tokens are replaced by the tokenizer's mask token (e.g., `[MASK]`).
            Defaults to 0.8, meaning 80% of the masked tokens will be replaced with `[MASK]`.
            Only works when `mlm` is set to `True`.
        random_replace_prob (`float`, *optional*, defaults to 0.1):
            The probability with which masked tokens are replaced by random tokens from the tokenizer's vocabulary.
            Defaults to 0.1, meaning 10% of the masked tokens will be replaced with random tokens. The remaining
            masked tokens (1 - mask_replace_prob - random_replace_prob) are left unchanged.
            Only works when `mlm` is set to `True`.
    """
    input_sequence_toks = np.asarray(sequence_tokens)[: (context_length - 2)]

    # - add CLS token
    input_sequence_toks = np.insert(input_sequence_toks, 0, tokenizer.cls_token_id)

    # - add SEP token
    input_sequence_toks = np.append(input_sequence_toks, tokenizer.sep_token_id)

    masked_input_sequence_toks = input_sequence_toks.copy()
    # - masking logic
    if mlm_probability > 0.0 and mlm_probability <= 1.0:
        prob_input_sequence = np.ones_like(input_sequence_toks) * mlm_probability
        if codon_weights is not None:
            pos_weight = codon_weights[input_sequence_toks]
            pos_weight[1:-1] = pos_weight[1:-1] / pos_weight[1:-1].mean()
            prob_input_sequence = prob_input_sequence * pos_weight
            prob_input_sequence = np.clip(prob_input_sequence, 0.05, 0.4)
        mask_indices = np.random.binomial(1, prob_input_sequence).astype(bool)  # noqa: NPY002
        mask_indices[0] = False  # - avoid masking the CLS token
        mask_indices[-1] = False  # - avoid masking the SEP token
    else:
        mask_indices = np.zeros_like(input_sequence_toks).astype(bool)

    if mask_replace_prob > 0.0:
        indices_replaced = (
            np.random.binomial(1, np.ones_like(mask_indices) * mask_replace_prob).astype(bool) & mask_indices  # noqa: NPY002
        )
        masked_input_sequence_toks[indices_replaced] = tokenizer.mask_token_id

    if random_replace_prob > 0.0:
        indices_random = np.random.binomial(  # noqa: NPY002
            1,
            (np.ones_like(mask_indices).astype(bool) & mask_indices & ~indices_replaced)
            * (random_replace_prob / (1 - mask_replace_prob)),
        ).astype(bool)
        valid_tokens = np.setdiff1d(
            np.arange(tokenizer.vocab_size),
            [
                tokenizer.cls_token_id,
                tokenizer.sep_token_id,
                tokenizer.pad_token_id,
                tokenizer.mask_token_id,
                tokenizer.unk_token_id,
            ],
        )

        masked_input_sequence_toks[indices_random] = np.random.choice(valid_tokens, size=indices_random.sum())  # noqa: NPY002

    attention_mask = np.zeros_like(input_sequence_toks)
    mask = np.zeros_like(input_sequence_toks)
    mask[mask_indices] = 1
    # - padding
    padding_length = context_length - len(input_sequence_toks)
    if padding_length > 0:
        input_sequence_toks = np.pad(input_sequence_toks, (0, padding_length), constant_values=tokenizer.pad_token_id)
        masked_input_sequence_toks = np.pad(
            masked_input_sequence_toks, (0, padding_length), constant_values=tokenizer.pad_token_id
        )
        mask = np.pad(mask, (0, padding_length), constant_values=0)
        attention_mask = np.pad(attention_mask, (0, padding_length), constant_values=1)

    attention_mask = ~(attention_mask.astype(bool))
    mask = mask.astype(bool)
    input_sequence_toks[~mask] = ignore_index
    return {
        MetadataFields.INPUT_IDS: masked_input_sequence_toks,
        MetadataFields.LABELS: input_sequence_toks,
        MetadataFields.ATTENTION_MASK: attention_mask,
        MetadataFields.INPUT_MASK: mask,
    }
