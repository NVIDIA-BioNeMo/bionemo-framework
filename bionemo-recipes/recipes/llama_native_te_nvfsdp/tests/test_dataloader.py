# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Unit tests for genomic dataloader
"""

import pytest
import torch
import sqlite3
import tempfile
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoTokenizer

from sqlite_dataset import create_genomic_dataloader, GenomicSequenceDataset


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
        ("seq_A", "A" * 100, 100),
        ("seq_T", "T" * 80, 80),
        ("seq_C", "C" * 50, 50),
        ("seq_G", "G" * 20, 20),
    ]
    
    cursor.executemany(
        "INSERT INTO sequences (contig_id, nt_sequence, length) VALUES (?, ?, ?)",
        test_sequences
    )
    
    conn.commit()
    conn.close()
    
    yield db_path
    
    Path(db_path).unlink()


def test_dataset_loads_sample_sequence(tokenizer):
    """Test dataset retrieves sequences correctly from database.
    
    Uses single sequence so shuffling doesn't affect test (Jonathan's suggestion).
    Pattern: expected_sequence = [nucleotide_id] * seqlen
    """
    # Create minimal database with just 1 sequence
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    db_path = temp_db.name
    temp_db.close()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE sequences (contig_id TEXT PRIMARY KEY, nt_sequence TEXT NOT NULL, length INTEGER NOT NULL)")
    cursor.execute("INSERT INTO sequences VALUES (?, ?, ?)", ("seq_T", "T" * 10, 10))
    conn.commit()
    conn.close()
    
    try:
        dataset = GenomicSequenceDataset(
            database_path=db_path,
            seq_length=10,
            tokenizer=tokenizer,
            stride=5,
            min_window_length=5,
            seed=42,
        )
        
        # Only 1 sequence → 1 window → dataset[0] is predictable regardless of shuffle
        sample = dataset[0]
        nucleotides = sample["input_ids"][1:-1]  # Remove BOS/EOS
        
        # Should be all Ts (10 nucleotides)
        expected_sequence = [84] * 10  # All Ts
        received_sequence = nucleotides.tolist()
        
        assert received_sequence == expected_sequence
        
    finally:
        Path(db_path).unlink()


def test_dataloader_returns_expected_batch(tokenizer):
    """Test dataloader returns exact expected batch.
    """
    # Create minimal test database with exactly 2 sequences (one batch worth)
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".sqlite")
    db_path = temp_db.name
    temp_db.close()
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE sequences (contig_id TEXT PRIMARY KEY, nt_sequence TEXT NOT NULL, length INTEGER NOT NULL)")
    cursor.executemany("INSERT INTO sequences VALUES (?, ?, ?)", [
        ("seq_A", "A" * 5, 5),
    ])
    conn.commit()
    conn.close()
    
    try:
        args = DictConfig({
            "dataset": {
                "database_path": db_path,
                "seq_length": 5,
                "batch_size": 1,  # Just one sample per batch
                "num_workers": 0,
                "stride": 3,
                "min_window_length": 3,
                "seed": 42,
            }
        })
        
        train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
        returned_batch = next(train_iterator)
        
        # Hardcode expected batch (1 sequence, so the output will be deterministic)
        # seq_A: 5bp of As -> BOS + 5 As + EOS
        BOS = tokenizer.bos_token_id  # 2
        EOS = tokenizer.eos_token_id  # 0
        A = 65  # ASCII value of 'A'
        
        expected_batch = {
            "input_ids": torch.tensor([
                [BOS, A, A, A, A, A, EOS],  # All As
            ], dtype=torch.int64),
            "labels": torch.tensor([
                [BOS, A, A, A, A, A, EOS],  # Same as input_ids
            ], dtype=torch.int64),
            "attention_mask": torch.tensor([
                [1, 1, 1, 1, 1, 1, 1],  # All real tokens
            ], dtype=torch.int64),
        }
        
        assert torch.equal(returned_batch["input_ids"], expected_batch["input_ids"])
        assert torch.equal(returned_batch["labels"], expected_batch["labels"])
        assert torch.equal(returned_batch["attention_mask"], expected_batch["attention_mask"])
        
    finally:
        Path(db_path).unlink()


def test_attention_mask_aligns_with_labels(simple_database, tokenizer):
    """Test attention_mask correctly identifies real vs padded positions in labels.
    
    Where attention_mask=1: labels should contain real token IDs
    Where attention_mask=0: labels should contain ignore_index value
    """
    # Define the ignore value we expect for padding (HF's LanguageModeling Collator hardcoded default) 
    IGNORE_PAD_TOKEN = -100
    
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 80,  # Larger window to create mixed-length batch to properly test padding
            "batch_size": 2,
            "num_workers": 0,
            "stride": 60,
            "min_window_length": 10,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    batch = next(train_iterator)
    
    # Check first sequence
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    
    # Where attention_mask=1, labels should equal input_ids (NON PADDED tokens)
    real_positions = attention_mask == 1
    real_labels = labels[real_positions]
    real_input_ids = input_ids[real_positions]
    
    # Check labels match input_ids for real positions
    assert torch.all(real_labels == real_input_ids)
    
    # Check a few positions to ensure real tokens (not all IGNORE_INDEX), and no accidental MLM bugs
    assert real_labels[0].item() == 2  # BOS
    assert real_labels[1].item() in [65, 84, 67, 71]  # Nucleotide
    assert real_labels[-1].item() == 0  # EOS
    
    # Ensure NO real position has padding
    assert torch.all(real_labels != IGNORE_PAD_TOKEN)
    
    # Where attention_mask=0, labels should be the pad token
    padded_positions = attention_mask == 0
    if padded_positions.any():
        assert torch.all(labels[padded_positions] == IGNORE_PAD_TOKEN)


def test_epoch_length_equals_total_windows_divided_by_batch_size(simple_database, tokenizer):
    """Test epoch_len is correct for given configuration."""
    args = DictConfig({
        "dataset": {
            "database_path": simple_database,
            "seq_length": 10,
            "batch_size": 3,
            "num_workers": 0,
            "stride": 8,
            "min_window_length": 5,
            "seed": 42,
        }
    })
    
    train_iterator, epoch_len = create_genomic_dataloader(args, tokenizer)
    
    # Hardcoded: 29 windows total, batch_size=3, drop_last=True -> 9 batches
    assert epoch_len == 9
