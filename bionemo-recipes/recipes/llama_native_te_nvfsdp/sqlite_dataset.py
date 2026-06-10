# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""
Simple LLAMA3 Genomic Dataloader (no Lightning dependencies).

Implements meeting requirements:
1. First tiling, then tokenization  
2. Fixed window positions + randomized access (EVO2 style)
3. SQL backend for speed
4. EVO2-style overlap ratio (200bp default)
"""

import logging
import sqlite3
import random
import numpy as np
from typing import Dict, Optional, Tuple, Iterator
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class GenomicSequenceDataset(Dataset):
    """
    EVO2-style genomic dataset with proper tiling and randomization.
    
    Key features:
    1. First tiling (pre-compute window mappings)
    2. Then tokenization (ASCII nucleotide encoding)
    3. Fixed positions + shuffled access (like EVO2)
    """
    
    def __init__(
        self,
        database_path: str,
        seq_length: int = 8192,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        stride: int = 7992,  # EVO2 default: 200bp overlap
        min_window_length: int = 1000,
        seed: int = 42,
        rc_augmentation: bool = False,
    ):
        """
        Initialize genomic dataset with EVO2-style tiling.
        
        Args:
            database_path: Path to SQLite database with sequences
            seq_length: Window size (8192 for LLAMA3)
            tokenizer: HuggingFace tokenizer (use AutoTokenizer.from_pretrained)
            stride: Stride between windows (7992 = 8192-200 for EVO2 overlap)
            min_window_length: Minimum effective window length
            seed: Random seed for reproducible randomization
            rc_augmentation: Apply reverse complement augmentation
        """
        self.database_path = Path(database_path)
        self.seq_length = seq_length
        self.stride = stride
        self.min_window_length = min_window_length
        self.seed = seed
        self.rc_augmentation = rc_augmentation
        
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided. Use AutoTokenizer.from_pretrained('./nucleotide_tokenizer')")
        
        self.tokenizer = tokenizer
            
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Step 1: Load sequences from database
        self._load_sequences()
        
        # Step 2: First create window mappings (EVO2 style)
        self._create_window_mappings()
        
        # Step 3: Randomize lookups - shuffle window indices
        self._randomize_window_access()
        
        logger.info(f"Dataset ready: {len(self.window_mappings):,} windows with EVO2-style tiling")
    
    def _load_sequences(self):
        """Load sequence metadata from SQLite database."""
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
            
        with sqlite3.connect(str(self.database_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT contig_id, length FROM sequences ORDER BY contig_id")
            self.sequences = cursor.fetchall()
            
        logger.info(f"Loaded {len(self.sequences)} sequences from database")
    
    def _create_window_mappings(self):
        """
        Step 1: First tiling - create window mappings EVO2 style.
        
        Fixed window positions (no jitter) like EVO2.
        """
        self.window_mappings = []  # List of (seq_idx, contig_id, start_pos, effective_length)
        
        total_windows = 0
        for seq_idx, (contig_id, seq_length) in enumerate(self.sequences):
            if seq_length < self.min_window_length:
                continue
            
            # Calculate number of windows (EVO2 formula)
            if seq_length <= self.seq_length:
                # Short sequence: single window
                num_windows = 1
                positions = [0]
            else:
                # Long sequence: EVO2 stride-based windows
                num_windows = 1 + (seq_length - self.seq_length) // self.stride
                positions = [i * self.stride for i in range(num_windows)]
            
            # Create window mappings for this sequence (fixed positions)
            for i, start_pos in enumerate(positions):
                effective_length = min(self.seq_length, seq_length - start_pos)
                if effective_length >= self.min_window_length:
                    self.window_mappings.append((seq_idx, contig_id, start_pos, effective_length))
                    total_windows += 1
        
        logger.info(f"Created {total_windows:,} window mappings with stride={self.stride} (overlap={self.seq_length - self.stride}bp)")
    
    def _randomize_window_access(self):
        """
        Step 3: Randomize lookups - shuffle window indices.
        
        Uses fixed window positions (like EVO2) but randomizes access order.
        This ensures we don't just get position-0 windows from each sequence.
        """
        # Create shuffled indices for randomized access
        self.shuffled_indices = list(range(len(self.window_mappings)))
        random.shuffle(self.shuffled_indices)
        logger.info(f"Shuffled {len(self.shuffled_indices):,} window indices for randomized access")
    
    def __len__(self) -> int:
        """Return number of windows."""
        return len(self.window_mappings)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a windowed sequence sample.
        
        Process:
        1. Map shuffled index to actual window (randomized lookup)
        2. Retrieve sequence from SQL database (tiled position)
        3. Tokenize with ASCII encoding (then tokenization)
        """
        # Map shuffled index to actual window (randomized lookup)
        actual_idx = self.shuffled_indices[idx % len(self.shuffled_indices)]
        seq_idx, contig_id, start_pos, window_length = self.window_mappings[actual_idx]
        
        # Retrieve sequence from SQL database (speed requirement)
        with sqlite3.connect(str(self.database_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT substr(nt_sequence, ?, ?) FROM sequences WHERE contig_id = ?",
                (start_pos + 1, window_length, contig_id)  # SQLite is 1-indexed
            )
            result = cursor.fetchone()
            
        if result is None:
            raise ValueError(f"Sequence {contig_id} not found in database")
            
        sequence = result[0].upper()
        
        # Apply reverse complement augmentation if enabled
        if self.rc_augmentation and self.rng.rand() > 0.5:
            sequence = self._reverse_complement(sequence)
        
        # Step 2: Then tokenization - ASCII encoding (meeting requirement)
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=True)
        
        # For DataCollatorForLanguageModeling: return variable-length sequences
        # The collator will handle padding and label creation
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
        }
    
    def _reverse_complement(self, sequence: str) -> str:
        """Apply reverse complement transformation to DNA sequence."""
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
        return ''.join(complement.get(base, 'N') for base in reversed(sequence))


def infinite_dataloader(dataloader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    """Create an infinite dataloader that loops through epochs (simplified for testing).
    
    Args:
        dataloader: The DataLoader to loop through.
        
    Yields:
        Dict containing batch data.
    """
    epoch = 0
    while True:
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


def create_genomic_dataloader(args: DictConfig, tokenizer: PreTrainedTokenizerBase) -> Tuple[Iterator[Dict[str, torch.Tensor]], int]:
    """
    Create genomic dataloader compatible with Bruno's training script.
    
    This creates an infinite iterator matching Bruno's interface while
    implementing EVO2-style tiling and randomization.
    """
    # Create simple dataset for training loop
    dataset = GenomicSequenceDataset(
        database_path=args.dataset.database_path,
        seq_length=args.dataset.seq_length,
        tokenizer=tokenizer,
        stride=args.dataset.get("stride", args.dataset.seq_length - 200),  # EVO2 default
        min_window_length=args.dataset.get("min_window_length", 1000),
        seed=args.dataset.get("seed", 42),
        rc_augmentation=args.dataset.get("rc_augmentation", False),
    )
    
    # Create data collator for autoregressive (causal) language modeling
    # mlm=False means the model will handle label shifting internally (via TransformerEngine)
    # Note: We don't use pad_to_multiple_of because our sequences are already seq_length bp,
    # which become (seq_length + 2) tokens after adding BOS/EOS. Padding to longest in batch
    # is more efficient (typically all sequences are the same length = no padding needed).
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM - no masked language modeling
    )
    
    # Create simple dataloader for testing (no distributed dependencies)
    dataloader = DataLoader(
        dataset,
        batch_size=args.dataset.batch_size,
        shuffle=True,  # Shuffling for randomization during training
        num_workers=args.dataset.get("num_workers", 0),
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.dataset.get("num_workers", 0) > 0,
        collate_fn=data_collator,  # Use the language modeling collator
    )
    
    # Calculate epoch length  
    epoch_len = len(dataloader)
    
    logger.info(f"Created genomic dataloader: {len(dataset):,} windows, {epoch_len:,} batches per epoch")
    logger.info(f"Tiling: stride={dataset.stride}, overlap={dataset.seq_length - dataset.stride}bp")
    logger.info("Randomization: fixed_positions=True, shuffled_access=True (EVO2 style)")
    logger.info("Mode: Simplified for testing (no distributed training)")
    logger.info(f"Collator: DataCollatorForLanguageModeling(mlm=False) - model handles label shifting")
    
    # Create infinite iterator (matches Bruno's interface and other recipes)
    train_iterator = infinite_dataloader(dataloader)
    
    return train_iterator, epoch_len
