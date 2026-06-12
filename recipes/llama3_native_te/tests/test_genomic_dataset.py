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

import pytest
import torch
from transformers.data.data_collator import DataCollatorForLanguageModeling

from collator import DataCollatorWithFlattening
from genomic_dataset import GenomicDataCollator


@pytest.fixture
def tokenizer(tokenizer_path):
    """Load the nucleotide tokenizer."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


# Tests for GenomicDataCollator
def test_collator_basic(tokenizer):
    """Test basic collator functionality."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [65, 67, 71, 84]}]
    batch = collator(features)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape[0] == 1


def test_collator_uppercases(tokenizer):
    """Test that collator uppercases labels while keeping inputs mixed case."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [97, 67, 103, 116]}]  # "aCgt"
    batch = collator(features)

    # Verify inputs unchanged (still mixed case)
    input_ids = batch["input_ids"]
    assert input_ids[0, 0].item() == 97, "Input 'a' (97) should stay lowercase"
    assert input_ids[0, 2].item() == 103, "Input 'g' (103) should stay lowercase"

    # Verify labels uppercased
    # Parent doesn't shift: labels = [97, 67, 103, 116] (same as input_ids)
    # Our uppercase: [97, 67, 103, 116] → [65, 67, 71, 84]
    #                 a   C   g    t   →   A   C   G   T
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84]])  # All uppercase
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_uppercases_sequence_packing(tokenizer):
    """Test that collator uppercases labels while keeping inputs mixed case."""
    base_mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    base = DataCollatorWithFlattening(collator=base_mlm_collator, separator_id=-100)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=False,
    )

    features = [{"input_ids": [97, 67, 103, 116]}]  # "aCgt"
    batch = collator(features)

    # Verify inputs unchanged (still mixed case)
    input_ids = batch["input_ids"]
    assert input_ids[0, 0].item() == 97, "Input 'a' (97) should stay lowercase"
    assert input_ids[0, 2].item() == 103, "Input 'g' (103) should stay lowercase"

    # Verify labels uppercased
    # Parent doesn't shift: labels = [97, 67, 103, 116] (same as input_ids)
    # Our uppercase: [97, 67, 103, 116] → [65, 67, 71, 84]
    #                 a   C   g    t   →   A   C   G   T
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84]])  # All uppercase: A, C, G, T
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_masks_degenerate(tokenizer):
    """Test that collator masks degenerate bases (N, R, Y, etc.)."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=False,
        mask_degenerate_bases=True,
    )

    features = [{"input_ids": [65, 67, 71, 84, 78]}]  # "ACGTN" (N is degenerate)
    batch = collator(features)

    # Parent creates labels = input_ids (no shift): [65, 67, 71, 84, 78]
    # Our degenerate masking: 78 (N) → -100
    # Expected: [65, 67, 71, 84, -100]
    #           A   C   G   T   MASKED
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84, -100]])
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_combined(tokenizer):
    """Test collator with both uppercase and degenerate masking."""
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=True,
    )

    features = [{"input_ids": [97, 67, 103, 84, 78]}]  # "aCgTN" (mixed case + degenerate)
    batch = collator(features)

    # Verify inputs unchanged (still mixed case)
    input_ids = batch["input_ids"]
    assert input_ids[0, 0].item() == 97, "Input 'a' should stay lowercase"
    assert input_ids[0, 2].item() == 103, "Input 'g' should stay lowercase"

    # Verify labels after BOTH uppercase AND degenerate masking
    # Parent: labels = input_ids = [97, 67, 103, 84, 78]
    # Step 1 uppercase: [97, 67, 103, 84, 78] → [65, 67, 71, 84, 78]
    #                   a   C   g    T   N   →   A   C   G   T   N
    # Step 2 mask degenerate: [65, 67, 71, 84, 78] → [65, 67, 71, 84, -100]
    #                         A   C   G   T   N   →   A   C   G   T   MASKED
    # Expected: [65, 67, 71, 84, -100]
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, 71, 84, -100]])
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"


def test_collator_handles_lowercase_degenerate(tokenizer):
    """Test that lowercase degenerate bases are handled correctly (uppercase then mask).

    Tests the order of operations: lowercase 'n' should be uppercased to 'N',
    then masked to -100.
    """
    base = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    collator = GenomicDataCollator(
        base_collator=base,
        uppercase_labels=True,
        mask_degenerate_bases=True,
    )

    # Input has lowercase degenerate: "ACn" (n=110)
    features = [{"input_ids": [65, 67, 110]}]
    batch = collator(features)

    # Verify exact output:
    # Parent: labels = input_ids = [65, 67, 110]
    # Step 1 uppercase: [65, 67, 110] → [65, 67, 78]
    #                   A   C   n    →   A   C   N (110→78)
    # Step 2 mask degenerate: [65, 67, 78] → [65, 67, -100]
    #                         A   C   N   →   A   C   MASKED
    # Expected: [65, 67, -100]
    labels = batch["labels"]
    expected_labels = torch.tensor([[65, 67, -100]])
    assert torch.equal(labels, expected_labels), f"Expected {expected_labels}, got {labels}"
