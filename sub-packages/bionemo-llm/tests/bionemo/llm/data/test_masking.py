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


import pytest
import torch

from bionemo.llm.data.masking import BertMaskConfig, add_cls_and_eos_tokens, apply_bert_pretraining_mask


class TestTokenizer:
    @property
    def mask_token_id(self):
        return 32

    @property
    def all_special_ids(self):
        return [0, 32]


def test_bert_mask_config_raises_with_invalid_probabilities():
    with pytest.raises(ValueError):
        BertMaskConfig(tokenizer=1, random_tokens=range(2, 4), mask_token_prob=0.9, random_token_prob=0.2)


def test_apply_bert_pretraining_mask():
    # fmt: off
    tokenized_sequence = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    random_seed = 123

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        random_seed,
        mask_config=BertMaskConfig(tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )

    # Check the unmasked tokens are unchanged.
    assert torch.allclose(masked_sequence[~loss_mask], tokenized_sequence[~loss_mask])

    # Make sure the output labels are correct.
    assert torch.allclose(labels[loss_mask], tokenized_sequence[loss_mask])

    values, _ = torch.mode(masked_sequence[loss_mask])
    assert values.item() == 32


def test_apply_bert_pretraining_mask_no_mask_token():
    # fmt: off
    tokenized_sequence = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    random_seed = 123

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        random_seed,
        mask_config=BertMaskConfig(mask_token_prob=0.0, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )

    # Check the unmasked tokens are unchanged.
    assert torch.allclose(masked_sequence[~loss_mask], tokenized_sequence[~loss_mask])

    # Make sure the output labels are correct.
    assert torch.allclose(labels[loss_mask], tokenized_sequence[loss_mask])

    # Make sure no mask tokens are in the output sequence
    assert torch.all(masked_sequence != 32)


def test_apply_bert_pretraining_mask_changing_mask_prob():
    # fmt: off
    tokenized_sequence = torch.tensor(
        [20, 15, 11,  7, 10, 16,  9, 10,  4, 15,  8, 12,  7, 10, 12,  4,  9,
         10,  8, 15,  9, 14,  7,  8,  6,  5, 16,  4,  5,  9,  9,  4,  8,  7,
          8, 10, 16,  7, 12,  7, 16, 13, 12,  5, 19,  4, 10,  8,  4,  6, 19,
         17, 12,  7,  5, 11, 14, 10,  6, 19,  7,  4,  5,  6,  6])
    # fmt: on

    random_seed = 123

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        random_seed,
        mask_config=BertMaskConfig(mask_prob=0.0, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )

    # All mask values should be False.
    assert torch.all(~loss_mask)


def test_apply_bert_pretraining_mask_converges_to_correct_probability():
    sequence = torch.ones(100_000, dtype=torch.long)
    random_seed = 123

    masked_sequence, _, loss_mask = apply_bert_pretraining_mask(
        sequence,
        random_seed,
        mask_config=BertMaskConfig(
            tokenizer=TestTokenizer(),
            random_tokens=range(3, 5),
            mask_prob=0.5,
            mask_token_prob=0.25,
            random_token_prob=0.12,
        ),
    )

    # Check that overall masking probability is correct.
    assert pytest.approx(loss_mask.float().mean(), abs=0.01) == 0.5

    # Check that the distribution of masked tokens is correct.
    assert pytest.approx((masked_sequence == 32).float().mean(), abs=0.01) == 0.5 * 0.25

    # Check that the distribution of random tokens is correct.
    assert (
        pytest.approx(torch.logical_or(masked_sequence == 3, masked_sequence == 4).float().mean(), abs=0.01)
        == 0.5 * 0.12
    )

    # Check that the distribution of unmasked tokens is correct.
    assert pytest.approx((masked_sequence[loss_mask] == 1).float().mean(), abs=0.01) == 1.0 - (0.25 + 0.12)


def test_apply_bert_pretraining_mask_is_reproducible_with_same_seed():
    torch.manual_seed(42)
    tokenized_sequence = torch.randint(0, 100, (1000,))

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        123,
        mask_config=BertMaskConfig(mask_prob=0.5, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )

    for _ in range(10):
        new_seq, new_labels, new_mask = apply_bert_pretraining_mask(
            tokenized_sequence,
            123,
            mask_config=BertMaskConfig(mask_prob=0.5, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
        )

        assert torch.allclose(masked_sequence, new_seq)
        assert torch.allclose(labels, new_labels)
        assert torch.allclose(loss_mask, new_mask)


def test_apply_bert_pretraining_mask_changes_with_new_seed():
    torch.manual_seed(42)
    tokenized_sequence = torch.randint(0, 100, (1000,))

    # Apply the function
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        123,
        mask_config=BertMaskConfig(mask_prob=0.5, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )

    new_seq, new_labels, new_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        321,
        mask_config=BertMaskConfig(mask_prob=0.5, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )

    assert not torch.allclose(masked_sequence, new_seq)
    assert not torch.allclose(labels, new_labels)
    assert not torch.allclose(loss_mask, new_mask)


def test_apply_bert_pretraining_mask_doesnt_mask_special_tokens():
    tokenized_sequence = torch.zeros(1000, dtype=torch.long)
    masked_sequence, labels, loss_mask = apply_bert_pretraining_mask(
        tokenized_sequence,
        123,
        mask_config=BertMaskConfig(mask_prob=0.5, tokenizer=TestTokenizer(), random_tokens=range(4, 24)),
    )
    assert torch.all(masked_sequence == 0)
    assert torch.all(labels == -100)
    assert torch.all(~loss_mask)


def test_add_cls_and_eos_tokens_both_tokens():
    sequence = torch.tensor([1, 2, 3])
    loss_mask = torch.tensor([False, True, False])
    labels = torch.tensor([-1, 2, -1])

    augmented_sequence, augmented_labels, augmented_loss_mask = add_cls_and_eos_tokens(
        sequence, labels, loss_mask, cls_token=0, eos_token=4
    )

    assert len(augmented_sequence) == len(sequence) + 2
    assert augmented_sequence[0] == 0
    assert torch.allclose(augmented_sequence[1:-1], sequence)
    assert augmented_sequence[-1] == 4

    assert len(augmented_loss_mask) == len(loss_mask) + 2
    assert not augmented_loss_mask[0]
    assert torch.allclose(augmented_loss_mask[1:-1], loss_mask)
    assert not augmented_loss_mask[-1]

    assert len(augmented_labels) == len(labels) + 2
    assert augmented_labels[0] == -1
    assert torch.allclose(augmented_labels[1:-1], labels)
    assert augmented_labels[-1] == -1


def test_add_cls_and_eos_tokens_only_cls():
    sequence = torch.tensor([1, 2, 3])
    loss_mask = torch.tensor([False, True, False])
    labels = torch.tensor([-1, 2, -1])

    augmented_sequence, augmented_labels, augmented_loss_mask = add_cls_and_eos_tokens(
        sequence, labels, loss_mask, cls_token=0, eos_token=None
    )

    assert len(augmented_sequence) == len(sequence) + 1
    assert augmented_sequence[0] == 0
    assert torch.allclose(augmented_sequence[1:], sequence)

    assert len(augmented_loss_mask) == len(loss_mask) + 1
    assert not augmented_loss_mask[0]
    assert torch.allclose(augmented_loss_mask[1:], loss_mask)

    assert len(augmented_labels) == len(labels) + 1
    assert augmented_labels[0] == -1
    assert torch.allclose(augmented_labels[1:], labels)


def test_add_cls_and_eos_tokens_only_bos():
    sequence = torch.tensor([1, 2, 3])
    loss_mask = torch.tensor([False, True, False])
    labels = torch.tensor([-1, 2, -1])

    augmented_sequence, augmented_labels, augmented_loss_mask = add_cls_and_eos_tokens(
        sequence, labels, loss_mask, cls_token=None, eos_token=4
    )

    assert len(augmented_sequence) == len(sequence) + 1
    assert torch.allclose(augmented_sequence[:-1], sequence)
    assert augmented_sequence[-1] == 4

    assert len(augmented_loss_mask) == len(loss_mask) + 1
    assert torch.allclose(augmented_loss_mask[:-1], loss_mask)
    assert not augmented_loss_mask[-1]

    assert len(augmented_labels) == len(labels) + 1
    assert torch.allclose(augmented_labels[:-1], labels)
    assert augmented_labels[-1] == -1
