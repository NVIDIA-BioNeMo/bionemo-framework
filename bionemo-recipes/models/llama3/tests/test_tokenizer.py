# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Unit tests for ASCII nucleotide tokenizer.
"""

from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer


@pytest.fixture(scope="session")
def tokenizer():
    """Load the ASCII nucleotide tokenizer."""
    tokenizer_path = Path(__file__).parent.parent / "nucleotide_fast_tokenizer"
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


def test_tokenizer_roundtrip_encode_decode(tokenizer):
    """Test that encoding and decoding produces the original sequence."""
    sequence = "ATCGATCG"
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    
    # Decoded may have spaces between tokens, so compare without spaces
    assert sequence == decoded.replace(" ", "")


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


def test_10kbp_sequence_creates_expected_window_count(tokenizer):
    """Test 10kbp sequence creates correct number of windows with seq_length=1000, stride=800.
    
    Verifies windowing math: 10000bp with seq_length=1000, stride=800.
    """
    sequence = "A" * 10000  # 10kbp
    
    result = tokenizer(
        sequence,
        max_length=1000,
        stride=800,       # 800 token overlap
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Hardcoded expectation based on input data:
    # 10000bp with 1000 token windows and 800 token stride
    # Step forward = 1000 - 800 = 200 tokens per window
    assert len(result["input_ids"]) == 47


def test_overlapping_windows_creates_more_samples(tokenizer):
    """Test overlapping stride creates more windows than less overlapping."""
    sequence = "ATCG" * 2500  # 10kbp
    
    result_more_overlap = tokenizer(
        sequence,
        max_length=1000,
        stride=800,  # 200 token step (80% overlap)
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    result_less_overlap = tokenizer(
        sequence,
        max_length=1000,
        stride=500,  # 500 token step (50% overlap)
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Hardcoded expectations
    assert len(result_more_overlap["input_ids"]) == 47  # With more overlap (smaller step)
    assert len(result_less_overlap["input_ids"]) == 20  # With less overlap (larger step)
    assert len(result_more_overlap["input_ids"]) > len(result_less_overlap["input_ids"])


def test_production_window_length_creates_expected_samples(tokenizer):
    """Test production settings (8192 window, 200 overlap) create correct number of windows."""
    sequence = "A" * 50000  # 50kbp sequence
    
    result = tokenizer(
        sequence,
        max_length=8192,
        stride=200,  # 200 token overlap
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Hardcoded expectation with production settings:
    # 50000bp with 8192 window and 200 stride (overlap)
    # Step forward = 8192 - 200 = 7992 tokens per window
    assert len(result["input_ids"]) == 7


def test_short_sequences_dont_overflow(tokenizer):
    """Test that short sequences (< max_length) don't create overflow windows."""
    sequence = "ATCG" * 100  # 400bp
    
    result = tokenizer(
        sequence,
        max_length=1000,
        stride=800,
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Sequence is shorter than max_length, should only create 1 window
    assert len(result["input_ids"]) == 1
    # Length should be 400bp + BOS + EOS = 402 tokens
    assert len(result["input_ids"][0]) == 402


def test_bos_eos_in_overlapping_windows(tokenizer):
    """Test that BOS/EOS tokens are added to every overlapping window.
    
    Verifies that when using return_overflowing_tokens with add_special_tokens=True,
    each window gets its own BOS and EOS tokens, treating each as an independent sequence.
    This matches the behavior needed for causal language modeling training.
    """
    # Use a short genomic sequence that will produce exactly 2 overlapping windows
    # With max_length=7 and stride=4, sequence of 8bp should give 2 windows
    sequence = "ATCGATCG"  # 8bp
    
    result = tokenizer(
        sequence,
        max_length=7,      # BOS + 5 content + EOS = 7 tokens total
        stride=4,          # Overlap of 4 tokens between windows
        truncation=True,
        return_overflowing_tokens=True,
        add_special_tokens=True,
    )
    
    # Should produce exactly 2 windows
    num_windows = len(result["input_ids"])
    assert num_windows >= 2, f"Should produce at least 2 overlapping windows, got {num_windows}"
    
    first_window = result["input_ids"][0]
    second_window = result["input_ids"][1]
    
    # Verify both windows have BOS at start and EOS at end
    assert first_window[0] == tokenizer.bos_token_id
    assert first_window[-1] == tokenizer.eos_token_id
    assert second_window[0] == tokenizer.bos_token_id
    assert second_window[-1] == tokenizer.eos_token_id
    
    # Verify windows are actually overlapping by checking they share some content
    first_content = set(first_window[1:-1])
    second_content = set(second_window[1:-1])
    assert len(first_content & second_content) > 0


