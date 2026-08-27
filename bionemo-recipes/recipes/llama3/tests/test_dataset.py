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

import tempfile
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from transformers import AutoTokenizer

from dataset import create_bshd_dataloader, create_tokenized_dataset
from distributed_config import DistributedConfig


@pytest.fixture(scope="session")
def tokenizer_path():
    """Get the path to the nucleotide tokenizer."""
    return str(Path(__file__).parent.parent.parent.parent / "models" / "llama3" / "nucleotide_fast_tokenizer")


@pytest.fixture(scope="session")
def tokenizer(tokenizer_path):
    """Load the nucleotide tokenizer."""
    return AutoTokenizer.from_pretrained(tokenizer_path)


@pytest.fixture
def simple_parquet(tmp_path):
    """Create a simple Parquet file with multiple genomic sequences for testing batching."""
    parquet_path = tmp_path / "genomic_sequences.parquet"
    
    # Create multiple sequences of varying lengths for better batching tests
    sequences = [
        "A" * 1000,
        "T" * 1200,
        "C" * 800,
        "G" * 1500,
        "ATCG" * 300,
    ]
    
    table = pa.table({
        "sequence": sequences,
    })
    
    pq.write_table(table, parquet_path)
    return str(parquet_path)


def test_dataset_loads_and_tokenizes_sequence(tokenizer_path, tmp_path):
    """Test that dataset loads and tokenizes a sequence correctly with exact token verification.
    
    Uses single sequence so shuffling doesn't affect test (similar to SQLite test approach).
    Pattern: expected_sequence = [nucleotide_id] * seqlen
    """
    # Create a Parquet file with a single T sequence of known length
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "T" * 10  # Small, predictable sequence
    table = pa.table({"sequence": [sequence]})
    pq.write_table(table, parquet_path)
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }
    
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=20,  # Large enough to fit the sequence
        stride=10,
        buffer_size=10_000,
        use_lazy_tokenization=False,  # Eager to get predictable dataset
    )
    
    # Only 1 sequence → 1 window → dataset[0] is predictable regardless of shuffle
    sample = tokenized_dataset[0]
    assert "input_ids" in sample
    
    # Get nucleotides (remove BOS and EOS)
    tokens = sample["input_ids"]
    nucleotides = tokens[1:-1]
    
    # Verify exact expected sequence
    BOS = 2
    EOS = 0
    T = 84  # ASCII value of 'T'
    
    expected_sequence = [T] * 10  # All Ts
    received_sequence = nucleotides
    
    assert tokens[0] == BOS, f"First token should be BOS (2), got {tokens[0]}"
    assert tokens[-1] == EOS, f"Last token should be EOS (0), got {tokens[-1]}"
    assert received_sequence == expected_sequence, \
        f"Expected {expected_sequence}, got {received_sequence}"


def test_dataloader_returns_expected_batch(tokenizer_path, tmp_path):
    """Test dataloader returns exact expected batch with known input.
    
    Creates minimal test data with exactly one sequence to get deterministic output.
    Verifies exact token values match expected hardcoded batch.
    """
    # Create minimal test parquet with exactly 1 sequence
    parquet_path = tmp_path / "single_sequence.parquet"
    sequence = "A" * 5  # 5 As
    table = pa.table({"sequence": [sequence]})
    pq.write_table(table, parquet_path)
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=1,  # Just one sample per batch
        num_workers=0,
        max_seq_length=10,  # Large enough for 5bp sequence
        stride=5,
        use_lazy_tokenization=False,  # Eager for deterministic behavior
    )
    
    returned_batch = next(iter(dataloader))
    
    # Hardcode expected batch (1 sequence, deterministic output)
    # seq: 5bp of As -> BOS + 5 As + EOS
    BOS = 2
    EOS = 0
    A = 65  # ASCII value of 'A'
    
    expected_input_ids = torch.tensor([[BOS, A, A, A, A, A, EOS]], dtype=torch.long)
    expected_labels = torch.tensor([[BOS, A, A, A, A, A, EOS]], dtype=torch.long)  # CLM: labels = input_ids
    expected_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)  # All real tokens
    
    assert torch.equal(returned_batch["input_ids"], expected_input_ids), \
        f"Expected input_ids {expected_input_ids}, got {returned_batch['input_ids']}"
    assert torch.equal(returned_batch["labels"], expected_labels), \
        f"Expected labels {expected_labels}, got {returned_batch['labels']}"
    assert torch.equal(returned_batch["attention_mask"], expected_attention_mask), \
        f"Expected attention_mask {expected_attention_mask}, got {returned_batch['attention_mask']}"


def test_attention_mask_aligns_with_labels(tokenizer_path, simple_parquet):
    """Test attention_mask correctly identifies real vs padded positions in labels.
    
    Where attention_mask=1: labels should contain real token IDs (matching input_ids)
    Where attention_mask=0: labels should contain ignore_index value (-100)
    """
    # HuggingFace's DataCollatorForLanguageModeling uses -100 as ignore_index by default
    IGNORE_PAD_TOKEN = -100
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
    }
    
    # Use a moderate window size to ensure we get padding in batches
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
    )
    
    batch = next(iter(dataloader))
    
    # Check first sequence in batch
    attention_mask = batch["attention_mask"][0]
    labels = batch["labels"][0]
    input_ids = batch["input_ids"][0]
    
    # Where attention_mask=1, labels should equal input_ids (real tokens)
    real_positions = attention_mask == 1
    real_labels = labels[real_positions]
    real_input_ids = input_ids[real_positions]
    
    # For CLM (Causal Language Modeling), labels should match input_ids at real positions
    assert torch.all(real_labels == real_input_ids), "Labels should match input_ids at real token positions"
    
    # Verify specific token positions contain expected values
    assert real_labels[0].item() == 2, "First token should be BOS (2)"
    assert real_labels[-1].item() == 0, "Last real token should be EOS (0)"
    # Middle tokens should be nucleotides (A=65, T=84, C=67, G=71)
    if len(real_labels) > 2:
        middle_token = real_labels[1].item()
        assert middle_token in [65, 84, 67, 71], f"Nucleotide tokens should be A/T/C/G, got {middle_token}"
    
    # Ensure NO real position has the ignore padding value
    assert torch.all(real_labels != IGNORE_PAD_TOKEN), "Real tokens should not have IGNORE_PAD_TOKEN"
    
    # Where attention_mask=0, labels should be IGNORE_PAD_TOKEN (-100)
    padded_positions = attention_mask == 0
    if padded_positions.any():
        padded_labels = labels[padded_positions]
        assert torch.all(padded_labels == IGNORE_PAD_TOKEN), \
            f"Padded positions should have IGNORE_PAD_TOKEN (-100), got {padded_labels.unique()}"




def test_windowing_in_dataset_creates_multiple_samples(tokenizer_path, tmp_path):
    """Test that the dataset's windowing creates expected number of samples."""
    # Create a 3kbp sequence
    parquet_path = tmp_path / "genomic_sequences.parquet"
    sequence = "A" * 3000
    table = pa.table({"sequence": [sequence]})
    pq.write_table(table, parquet_path)
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }
    
    tokenized_dataset, tokenizer = create_tokenized_dataset(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        max_seq_length=1000,
        stride=800,  # 800 token overlap, so 200 token step
        buffer_size=10_000,
        use_lazy_tokenization=False,  # Use eager tokenization to expand windows
    )
    
    # Count samples
    num_samples = len(tokenized_dataset)
    
    # With 3000bp sequence, max_length=1000, stride=800 (800 overlap, 200 step)
    # Formula: ceil((3000+2 - 1000) / 200) + 1 = ceil(2002/200) + 1 = 11 + 1 = 12 windows
    assert num_samples == 12, f"Expected exactly 12 windows, got {num_samples}"


def test_lazy_tokenization_returns_batch(tokenizer_path, simple_parquet):
    """Test that lazy tokenization works and returns valid batches."""
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
        "streaming": False,
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        use_lazy_tokenization=True,
    )
    
    # Get a batch
    batch = next(iter(dataloader))
    
    # Verify batch is not None and has correct structure
    assert batch is not None
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert isinstance(batch["input_ids"], torch.Tensor)
    
    # With lazy tokenization and windowing, batch size can vary due to on-the-fly window expansion
    # Just verify we get at least one sample (lazy tokenization + windowing makes exact count unpredictable)
    assert batch["input_ids"].shape[0] >= 1, \
        f"Expected at least 1 sample in batch, got {batch['input_ids'].shape[0]}"


@pytest.mark.parametrize("streaming", [False, True])
def test_multiple_sequences_batch_correctly(tokenizer_path, simple_parquet, streaming):
    """Test that multiple sequences batch together correctly in both streaming and non-streaming modes.
    
    This test catches bugs that only appear with multi-row datasets vs single-row:
    - Batching/collation works with multiple sequences
    - Sequences in batch are different (not duplicated)
    - Padding aligns correctly across multiple sequences
    - All sequences are processed across batches
    - Works in both streaming=True and streaming=False modes
    """
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": simple_parquet,
        "split": "train",
        "streaming": streaming,
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=500,
        stride=100,
        buffer_size=10_000,  # Only used for streaming
        use_lazy_tokenization=False,
    )
    
    # Get first batch
    batch = next(iter(dataloader))
    
    # KEY TEST 1: Verify batch contains MULTIPLE sequences (not just 1)
    assert batch["input_ids"].shape[0] == 2, \
        f"Batch should contain 2 sequences, got {batch['input_ids'].shape[0]}"
    
    # KEY TEST 2: Verify sequences in batch are DIFFERENT (catch duplication bugs)
    seq1 = batch["input_ids"][0]
    seq2 = batch["input_ids"][1]
    assert not torch.equal(seq1, seq2), \
        "Sequences in batch should be different, not duplicates"
    
    # KEY TEST 3: Verify padding aligns across all tensors in batch
    batch_size, seq_length = batch["input_ids"].shape
    assert batch["attention_mask"].shape == (batch_size, seq_length)
    assert batch["labels"].shape == (batch_size, seq_length)
    
    # KEY TEST 4: Verify all sequences are processed (multiple batches produced)
    # With 5 sequences from simple_parquet (800-1500bp) and max_seq_length=500, 
    # windowing will create ~11+ windows total. With batch_size=2, expect ~5-6 batches.
    # We already consumed 1 batch, so should have at least 4 remaining batches.
    all_batches = list(dataloader)
    total_batches = len(all_batches) + 1  # +1 for first batch already consumed
    assert len(all_batches) >= 4, \
        f"Expected at least 4 remaining batches (5 total), got {len(all_batches)} remaining ({total_batches} total)"
    
    # KEY TEST 5: Verify subsequent batches also valid (not just first batch)
    if len(all_batches) > 0:
        second_batch = all_batches[0]
        # Check structure is consistent across batches
        assert "input_ids" in second_batch
        assert "attention_mask" in second_batch
        assert "labels" in second_batch
        # Verify it also has multiple sequences (could be different count due to windowing)
        assert second_batch["input_ids"].shape[0] >= 1, \
            f"Second batch should have at least 1 sequence, got {second_batch['input_ids'].shape[0]}"
        # Verify tensors align
        batch_size_2, seq_length_2 = second_batch["input_ids"].shape
        assert second_batch["attention_mask"].shape == (batch_size_2, seq_length_2)
        assert second_batch["labels"].shape == (batch_size_2, seq_length_2)


def test_batching_produces_correct_batch_size(tokenizer_path, tmp_path):
    """Test that batching combines multiple sequences correctly with exact batch counts.
    
    Creates 5 short sequences (no windowing) with micro_batch_size=2.
    Should produce exactly 3 batches with shapes: [2, 2, 1].
    """
    # Create 5 sequences that won't trigger windowing (all very short)
    parquet_path = tmp_path / "five_sequences.parquet"
    sequences = [
        "A" * 10,    # Seq 1
        "T" * 15,    # Seq 2
        "C" * 12,    # Seq 3
        "G" * 8,     # Seq 4
        "ATCG" * 3,  # Seq 5 (12bp)
    ]
    table = pa.table({"sequence": sequences})
    pq.write_table(table, parquet_path)
    
    distributed_config = DistributedConfig(rank=0, world_size=1)
    
    load_dataset_kwargs = {
        "path": "parquet",
        "data_files": str(parquet_path),
        "split": "train",
    }
    
    dataloader, _ = create_bshd_dataloader(
        distributed_config=distributed_config,
        tokenizer_path=tokenizer_path,
        load_dataset_kwargs=load_dataset_kwargs,
        micro_batch_size=2,
        num_workers=0,
        max_seq_length=50,  # Large enough - no windowing
        stride=10,
        use_lazy_tokenization=False,  # Use eager to ensure predictable batching
    )
    
    # Collect all batches
    batches = list(dataloader)
    
    # With 5 sequences and batch_size=2, expect exactly 3 batches: [2, 2, 1]
    assert len(batches) == 3, f"Expected exactly 3 batches from 5 sequences, got {len(batches)}"
    
    # Check each batch has correct shape
    assert batches[0]["input_ids"].shape[0] == 2, "Batch 0 should have 2 sequences"
    assert batches[1]["input_ids"].shape[0] == 2, "Batch 1 should have 2 sequences"
    assert batches[2]["input_ids"].shape[0] == 1, "Batch 2 should have 1 sequence (remainder)"
