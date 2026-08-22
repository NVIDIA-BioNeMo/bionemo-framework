# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Unit tests for windowing logic
"""

import pytest
import torch
import sqlite3
import tempfile
from pathlib import Path
from transformers import AutoTokenizer

from sqlite_dataset import GenomicSequenceDataset


@pytest.fixture
def tokenizer():
    """Load the nucleotide tokenizer."""
    tokenizer_path = Path(__file__).parent.parent / "nucleotide_tokenizer"
    return AutoTokenizer.from_pretrained(str(tokenizer_path))


@pytest.fixture
def simple_database():
    """Create test database with repeated character sequences."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    db_path = temp_db.name
    temp_db.close()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE sequences (
            contig_id TEXT PRIMARY KEY,
            nt_sequence TEXT NOT NULL,
            length INTEGER NOT NULL
        )
    """)
    
    test_sequences = [
        ("seq_A", "A" * 10000, 10000),
        ("seq_T", "T" * 8000, 8000),
        ("seq_C", "C" * 5000, 5000),
        ("seq_G", "G" * 2000, 2000),
    ]
    
    cursor.executemany(
        "INSERT INTO sequences (contig_id, nt_sequence, length) VALUES (?, ?, ?)",
        test_sequences
    )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    Path(db_path).unlink()


def test_dataset_length_matches_expected_window_count(simple_database, tokenizer):
    """Test dataset creates correct total number of windows.
    
    Verifies windowing math: seq_A (10000bp) + seq_T (8000bp) + seq_C (5000bp) + seq_G (2000bp)
    with seq_length=1000, stride=800 should create exactly 29 windows.
    """
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=1000,
        tokenizer=tokenizer,
        stride=800,
        min_window_length=500,
        seed=42,
    )
    
    # Expectation based on input data:
    # seq_A (10000bp): 1 + (10000-1000)//800 = 12 windows
    # seq_T (8000bp):  1 + (8000-1000)//800  = 9 windows
    # seq_C (5000bp):  1 + (5000-1000)//800  = 6 windows
    # seq_G (2000bp):  1 + (2000-1000)//800  = 2 windows
    # Total: 12 + 9 + 6 + 2 = 29 windows
    assert len(dataset) == 29


def test_overlapping_windows_creates_more_samples(simple_database, tokenizer):
    """Test overlapping stride creates more windows than non-overlapping."""
    dataset_overlap = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=1000,
        tokenizer=tokenizer,
        stride=800,  # Overlap
        min_window_length=500,
        seed=42,
    )
    
    dataset_no_overlap = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=1000,
        tokenizer=tokenizer,
        stride=1000,  # No overlap
        min_window_length=500,
        seed=42,
    )
    
    assert len(dataset_overlap) == 29  # With overlap
    assert len(dataset_no_overlap) == 25  # Without overlap
    assert len(dataset_overlap) - len(dataset_no_overlap) == 4


def test_getitem_returns_tokenized_sequence_with_special_tokens(simple_database, tokenizer):
    """Test __getitem__ returns properly tokenized sequence."""
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=100,
        tokenizer=tokenizer,
        stride=50,
        min_window_length=50,
        seed=42,
    )
    
    sample = dataset[0]
    
    # Check structure
    assert isinstance(sample, dict)
    assert "input_ids" in sample
    assert isinstance(sample["input_ids"], torch.Tensor)
    
    # Check has BOS and EOS
    assert sample["input_ids"][0].item() == 2  # BOS
    assert sample["input_ids"][-1].item() == 0  # EOS


def test_production_window_length_creates_expected_samples(simple_database, tokenizer):
    """Test production settings (8192/7992) create correct number of windows."""
    dataset = GenomicSequenceDataset(
        database_path=simple_database,
        seq_length=8192,
        tokenizer=tokenizer,
        stride=7992,  # EVO2 default: 200bp overlap
        min_window_length=1000,
        seed=42,
    )
    
    # Hardcoded expectation with production settings:
    # seq_A (10000bp): 1 window (10000 < 8192 + 7992)
    # seq_T (8000bp):  0 windows (8000 < 8192, but >= min_window_length, so 1 window)
    # seq_C (5000bp):  0 windows (5000 < 8192, but >= min_window_length, so 1 window)  
    # seq_G (2000bp):  1 window (2000 >= min_window_length)
    # Total: 4 windows (one from each sequence)
    assert len(dataset) == 4



