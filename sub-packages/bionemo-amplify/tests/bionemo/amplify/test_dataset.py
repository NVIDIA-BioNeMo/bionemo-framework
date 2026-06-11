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

import datasets
import pytest
import torch
from torch.utils.data import Dataset as TorchDataset

from bionemo.amplify import tokenizer
from bionemo.amplify.dataset import AMPLIFYMaskedResidueDataset
from bionemo.core.data.multi_epoch_dataset import EpochIndex
from bionemo.esm2.data.dataset import RandomMaskStrategy


@pytest.fixture
def dummy_hf_dataset():
    """Create a mock HuggingFace dataset with protein sequences."""
    sequences = [
        "ACDEFGHIKLMNPQRSTVWY",
        "DEFGHIKLMNPQRSTVWYAC",
        "MGHIKLMNPQRSTVWYACDE",
        "MKTVRQERLKSIVRI",
        "MRILERSKEPVSGAQLA",
    ]
    return datasets.Dataset.from_dict({"sequence": sequences})


@pytest.fixture
def amplify_tokenizer():
    """Return the AMPLIFY tokenizer."""
    return tokenizer.BioNeMoAMPLIFYTokenizer()


def test_amplify_masked_residue_dataset_init(dummy_hf_dataset, amplify_tokenizer):
    """Test initialization of AMPLIFYMaskedResidueDataset."""
    dataset = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    assert isinstance(dataset, TorchDataset)
    assert len(dataset) == len(dummy_hf_dataset)


def test_amplify_masked_residue_dataset_getitem_has_expected_structure(dummy_hf_dataset, amplify_tokenizer):
    """Test that __getitem__ returns a sample with the expected structure."""
    dataset = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    sample = dataset[EpochIndex(0, 0)]

    # Check that all required fields are present
    assert "text" in sample
    assert "types" in sample
    assert "attention_mask" in sample
    assert "labels" in sample
    assert "loss_mask" in sample
    assert "is_random" in sample

    # Check that all fields are tensors
    assert isinstance(sample["text"], torch.Tensor)
    assert isinstance(sample["types"], torch.Tensor)
    assert isinstance(sample["attention_mask"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
    assert isinstance(sample["loss_mask"], torch.Tensor)
    assert isinstance(sample["is_random"], torch.Tensor)

    # Check that all tensors have the same length
    seq_len = len(sample["text"])
    assert len(sample["types"]) == seq_len
    assert len(sample["attention_mask"]) == seq_len
    assert len(sample["labels"]) == seq_len
    assert len(sample["loss_mask"]) == seq_len
    assert len(sample["is_random"]) == seq_len


def test_amplify_masked_residue_dataset_getitem_match_for_identical_seeds(dummy_hf_dataset, amplify_tokenizer):
    """Test that samples are identical for the same seed."""
    dataset1 = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=123,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    dataset2 = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=123,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    # Check that the datasets are equal
    for epoch in range(3):
        for i in range(len(dataset1)):
            sample1 = dataset1[EpochIndex(epoch, i)]
            sample2 = dataset2[EpochIndex(epoch, i)]

            for key in sample1:
                torch.testing.assert_close(sample1[key], sample2[key])


def test_amplify_masked_residue_dataset_getitem_is_deterministic(dummy_hf_dataset, amplify_tokenizer):
    """Test that __getitem__ is deterministic for the same index."""
    dataset = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=123,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    sample1 = dataset[EpochIndex(5, 1)]

    for _ in range(10):
        sample2 = dataset[EpochIndex(5, 1)]
        for key in sample1:
            torch.testing.assert_close(sample1[key], sample2[key])


def test_amplify_masked_residue_dataset_getitem_differs_with_different_seeds(dummy_hf_dataset, amplify_tokenizer):
    """Test that samples differ with different seeds."""
    dataset1 = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=123,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    dataset2 = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=321,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    for epoch in range(3):
        for i in range(len(dataset1)):
            sample1 = dataset1[EpochIndex(epoch, i)]
            sample2 = dataset2[EpochIndex(epoch, i)]
            assert not torch.equal(sample1["text"], sample2["text"])


def test_amplify_masked_residue_dataset_max_seq_length(dummy_hf_dataset, amplify_tokenizer):
    """Test that sequences are properly truncated to max_seq_length."""
    max_seq_length = 10
    dataset = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=123,
        max_seq_length=max_seq_length,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.AMINO_ACIDS_ONLY,
        tokenizer=amplify_tokenizer,
    )

    sample = dataset[EpochIndex(0, 0)]
    assert len(sample["text"]) <= max_seq_length


def test_amplify_masked_residue_dataset_random_mask_strategy(dummy_hf_dataset, amplify_tokenizer):
    """Test that random mask strategy affects token replacement."""
    dataset = AMPLIFYMaskedResidueDataset(
        hf_dataset=dummy_hf_dataset,
        seed=123,
        max_seq_length=512,
        mask_prob=0.15,
        mask_token_prob=0.8,
        mask_random_prob=0.1,
        random_mask_strategy=RandomMaskStrategy.ALL_TOKENS,
        tokenizer=amplify_tokenizer,
    )

    sample = dataset[EpochIndex(0, 0)]
    # Check that masked tokens can be any token (not just amino acids)
    masked_indices = sample["loss_mask"].nonzero().squeeze()
    masked_tokens = sample["text"][masked_indices]
    assert torch.all(masked_tokens == amplify_tokenizer.mask_token_id)


@pytest.mark.skip(
    reason="This test is slow and requires a real HuggingFace dataset, it's mainly here to demo the functionality "
    "and as a fast test when the dataset is available locally."
)
def test_amplify_with_real_hf_dataset(amplify_tokenizer):
    """Test that the AMPLIFYMaskedResidueDataset can be used with a real HuggingFace dataset."""
    dataset = AMPLIFYMaskedResidueDataset(
        datasets.load_dataset("chandar-lab/UR100P", split="test"),  # type: ignore
        tokenizer=amplify_tokenizer,
    )
    assert len(dataset) > 0
