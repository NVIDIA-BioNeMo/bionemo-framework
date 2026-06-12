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


import numpy as np

from src.data.metadata import MetadataFields


def _construct_sentence(
    ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, mask_mutation, use_alt
):
    assert not (mask_mutation and use_alt), "Cannot mask mutation and use alt sequence at the same time"
    input_sequence_toks = tokenizer.tokenize(ref_seq)
    input_sequence_toks = tokenizer.convert_tokens_to_ids(input_sequence_toks)
    input_sequence_toks = np.asarray(input_sequence_toks)[: (context_length - 2)]

    # - add CLS token
    input_sequence_toks = np.insert(input_sequence_toks, 0, tokenizer.cls_token_id)

    # - add SEP token
    input_sequence_toks = np.append(input_sequence_toks, tokenizer.sep_token_id)

    ref_codon_toks = tokenizer.convert_tokens_to_ids(ref_codon)
    alt_codon_toks = tokenizer.convert_tokens_to_ids(alt_codon)

    mutation_token_idx = int(codon_position + 1)
    if 0 < mutation_token_idx < len(input_sequence_toks) - 1:
        if mask_mutation:
            # - mask the mutation token
            input_sequence_toks[mutation_token_idx] = tokenizer.mask_token_id
        if use_alt:
            input_sequence_toks[mutation_token_idx] = alt_codon_toks
    else:
        raise ValueError(
            f"Mutation token index {mutation_token_idx} is out of bounds for input sequence of length {len(input_sequence_toks)}"
        )

    attention_mask = np.ones(context_length, dtype=np.int64)
    attention_mask[len(input_sequence_toks) :] = 0
    input_sequence_toks = np.pad(
        input_sequence_toks,
        (0, max(0, context_length - len(input_sequence_toks))),
        mode="constant",
        constant_values=tokenizer.pad_token_id,
    )
    input_sequence_toks = input_sequence_toks[:context_length]
    return input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx


def mlm_process_item(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, mask_mutation=True):  # noqa: D103
    input_sequence_toks, ref_codon_toks, alt_codon_toks, attention_mask, mutation_token_idx = _construct_sentence(
        ref_seq,
        codon_position,
        ref_codon,
        alt_codon,
        context_length,
        tokenizer,
        mask_mutation=mask_mutation,
        use_alt=False,
    )

    return {
        MetadataFields.INPUT_IDS: np.asarray(input_sequence_toks, dtype=np.int64),
        MetadataFields.REF_CODON_TOKS: np.asarray(ref_codon_toks, dtype=np.int64),
        MetadataFields.ALT_CODON_TOKS: np.asarray(alt_codon_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
        MetadataFields.MUTATION_TOKEN_IDX: np.asarray([mutation_token_idx], dtype=np.int64),
    }


def likelihood_process_item(ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer):
    """Processes a sequence for likelihood prediction.
    The input sequence is the reference sequence with the alternative codon used at the mutation site.
    """  # noqa: D205
    input_sequence_toks, _, _, attention_mask, _ = _construct_sentence(
        ref_seq, codon_position, ref_codon, alt_codon, context_length, tokenizer, mask_mutation=False, use_alt=True
    )

    return {
        MetadataFields.INPUT_IDS: np.asarray(input_sequence_toks, dtype=np.int64),
        MetadataFields.ATTENTION_MASK: np.asarray(attention_mask, dtype=np.int64),
    }
