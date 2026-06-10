# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Unit tests for ASCII nucleotide tokenizer.
"""

import pytest
import torch
from pathlib import Path
from transformers import AutoTokenizer


@pytest.fixture
def tokenizer():
    """Load the ASCII nucleotide tokenizer."""
    tokenizer_path = Path(__file__).parent.parent / "nucleotide_tokenizer"
    return AutoTokenizer.from_pretrained(str(tokenizer_path))


def test_tokenizer_special_token_ids(tokenizer):
    """Test that the tokenizer's special token IDs are correct (match NeMo)"""
    assert tokenizer.eos_token_id == 0
    assert tokenizer.pad_token_id == 1
    assert tokenizer.bos_token_id == 2
    assert tokenizer.unk_token_id == 3


def test_tokenizer_encode_simple_sequences(tokenizer):
    """Test encoding a simple repeated character sequences."""
    sequence = "AAAA"
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    
    # Expected: BOS + AAAA + EOS = [2, 65, 65, 65, 65, 0]
    expected = [2, 65, 65, 65, 65, 0]
    assert encoded == expected

    sequence = "C" 
    encoded = tokenizer.encode(sequence, add_special_tokens=True)

    # Expected: BOS + C + EOS = [2, 67, 0]
    expected = [2, 67, 0]
    assert encoded == expected
    
    sequence = "G" *20
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    expected = [2] + [71] * 20 + [0]
    assert encoded == expected


def test_tokenizer_encode_without_special_tokens(tokenizer):
    """Test encoding without BOS/EOS tokens."""
    sequence = "TTTT"
    encoded = tokenizer.encode(sequence, add_special_tokens=False)
    
    # Expected: just the Ts (T=84)
    expected = [84, 84, 84, 84]
    assert encoded == expected


def test_tokenizer_nucleotide_mappings(tokenizer):
    """Test each nucleotide maps to its ASCII value."""
    # A=65, T=84, C=67, G=71
    assert tokenizer.encode("A", add_special_tokens=False) == [65]
    assert tokenizer.encode("T", add_special_tokens=False) == [84]
    assert tokenizer.encode("C", add_special_tokens=False) == [67]
    assert tokenizer.encode("G", add_special_tokens=False) == [71]



def test_tokenizer_padding_to_longest(tokenizer):
    """Test padding pads to longest sequence in batch."""
    batch = tokenizer(["AAAA", "TTTTTTTT"], padding=True, add_special_tokens=True, return_tensors="pt")
    
    # AAAA → [2, 65, 65, 65, 65, 0] = 6 tokens
    # TTTTTTTT → [2, 84, 84, 84, 84, 84, 84, 84, 84, 0] = 10 tokens
    # Should pad to 10
    assert batch['input_ids'].shape == torch.Size([2, 10])
    
    # First sequence should have padding (PAD=1)
    assert batch['input_ids'][0, 6].item() == 1  # First padding position
    assert batch['input_ids'][0, 9].item() == 1  # Last padding position
    
    # Attention mask: 1 for real tokens, 0 for padding
    assert batch['attention_mask'][0, 5].item() == 1  # Last real token
    assert batch['attention_mask'][0, 6].item() == 0  # First padding


def test_tokenizer_attention_mask_correct(tokenizer):
    """Test attention mask is 1 for real tokens, 0 for padding."""
    batch = tokenizer(["GG", "GGGGGG"], padding=True, add_special_tokens=True, return_tensors="pt")
    
    # GG → 4 tokens (BOS + GG + EOS)
    # GGGGGG → 8 tokens (BOS + GGGGGG + EOS)
    # Padded to 8 tokens
    
    # First sequence: 4 real + 4 padding
    expected_mask_0 = [1, 1, 1, 1, 0, 0, 0, 0]
    assert batch['attention_mask'][0].tolist() == expected_mask_0
    
    # Second sequence: all real
    expected_mask_1 = [1, 1, 1, 1, 1, 1, 1, 1]
    assert batch['attention_mask'][1].tolist() == expected_mask_1


def test_tokenizer_mixed_nucleotides(tokenizer):
    """Test all standard nucleotides encode correctly."""
    sequence = "ATCGGTC"
    encoded = tokenizer.encode(sequence, add_special_tokens=False)
    
    # A=65, T=84, C=67, G=71
    # ATCGGTC = A, T, C, G, G, T, C
    expected = [65, 84, 67, 71, 71, 84, 67]
    assert encoded == expected


def test_tokenizer_special_nucleotides(tokenizer):
    """Test that sequences with ambiguity tokens (N, R, Y) encodes correctly."""
    sequence = "AANNNRY"
    encoded = tokenizer.encode(sequence, add_special_tokens=False)

    # A=65, N=78, R=82, Y=89
    expected = [65, 65, 78, 78, 78, 82, 89]
    assert encoded == expected

