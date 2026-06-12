# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Dataset and dataloader utilities for CodonFM pretraining."""

import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from distributed_config import DistributedConfig
from tokenizer import CodonTokenizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler


BASES = "ACGT"


class SyntheticCodonDataset(Dataset):
    """Generates random codon sequences on-the-fly for testing."""

    def __init__(self, num_samples: int = 1000, min_codons: int = 30, max_codons: int = 200, seed: int = 42):
        """Initialize.

        Args:
            num_samples: Number of sequences to generate.
            min_codons: Minimum number of codons per sequence.
            max_codons: Maximum number of codons per sequence.
            seed: Random seed.
        """
        self.num_samples = num_samples
        self.min_codons = min_codons
        self.max_codons = max_codons
        self.rng = random.Random(seed)
        self.sequences = [self._generate_sequence() for _ in range(num_samples)]

    def _generate_sequence(self) -> str:
        num_codons = self.rng.randint(self.min_codons, self.max_codons)
        return "".join(self.rng.choice(BASES) for _ in range(num_codons * 3))

    def __len__(self) -> int:  # noqa: D105
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, str]:  # noqa: D105
        return {"sequence": self.sequences[idx]}


class ParquetCodonDataset(Dataset):
    """Dataset that reads codon sequences from a parquet file using memory-mapped Arrow arrays.

    Uses PyArrow memory mapping instead of loading into a pandas DataFrame,
    avoiding the pandas copy and letting the OS page data in/out as needed.
    """

    def __init__(self, path: str):
        """Initialize.

        Args:
            path: Path to the parquet file with a 'sequence' column.
        """
        self._table = pq.read_table(path, columns=["sequence"], memory_map=True)
        self._sequences = self._table.column("sequence")

    def __len__(self) -> int:  # noqa: D105
        return len(self._sequences)

    def __getitem__(self, idx: int) -> dict[str, str]:  # noqa: D105
        return {"sequence": self._sequences[idx].as_py()}


class MemmapCodonDataset(Dataset):
    """Dataset that reads codon sequences from memory-mapped chunk files.

    Reads the memmap format produced by the PTL recipe's ncbi_memmap_dataset_creator:
    a directory containing metadata.json, sequences_chunk*.mmap, and index_chunk*.mmap files.
    Long sequences are split into non-overlapping sliding windows of max_seq_length - 2 tokens
    (leaving room for CLS/SEP added by the collator). Returns decoded codon strings matching
    the same interface as ParquetCodonDataset.
    """

    def __init__(self, data_path: str, max_seq_length: int = 512):
        """Initialize.

        Args:
            data_path: Path to directory containing metadata.json and mmap chunk files.
            max_seq_length: Maximum sequence length (including special tokens). Windows are
                sized to max_seq_length - 2 to leave room for CLS/SEP.
        """
        self.data_path = Path(data_path)
        self.tokenizer = CodonTokenizer()
        self.window_size = max_seq_length - 2  # room for CLS/SEP

        with open(self.data_path / "metadata.json") as f:
            metadata = json.load(f)

        # Load mmap chunks
        self.sequences_mmaps = []
        self.indices_mmaps = []
        for chunk in metadata["chunks"]:
            seq_mmap = np.memmap(
                self.data_path / chunk["sequences"]["path"],
                dtype=chunk["sequences"]["dtype"],
                mode="r",
                shape=tuple(chunk["sequences"]["shape"]),
            )
            idx_mmap = np.memmap(
                self.data_path / chunk["index"]["path"],
                dtype=chunk["index"]["dtype"],
                mode="r",
                shape=tuple(chunk["index"]["shape"]),
            )
            self.sequences_mmaps.append(seq_mmap)
            self.indices_mmaps.append(idx_mmap)

        # Build or load cached global indices
        cache_path = self.data_path / "global_indices_cache.npy"
        if cache_path.exists():
            self.global_indices = np.load(cache_path)
        else:
            self.global_indices = self._build_global_indices()
            np.save(cache_path, self.global_indices)

    def _build_global_indices(self) -> np.ndarray:
        """Build sliding window indices over all sequences in all chunks.

        Returns:
            Array of shape (num_windows, 3) with columns [chunk_id, start_token_idx, end_token_idx].
        """
        indices = []
        for chunk_id, idx_mmap in enumerate(self.indices_mmaps):
            for seq_idx in range(len(idx_mmap)):
                seq_start, seq_end, _taxid = idx_mmap[seq_idx]
                seq_len = seq_end - seq_start
                if seq_len <= 0:
                    continue
                # Non-overlapping windows of window_size tokens
                num_windows = max(1, (seq_len + self.window_size - 1) // self.window_size)
                for win_idx in range(num_windows):
                    start = seq_start + win_idx * self.window_size
                    end = min(start + self.window_size, seq_end)
                    if end > start:
                        indices.append([chunk_id, start, end])
        return np.array(indices, dtype=np.int64)

    def __len__(self) -> int:  # noqa: D105
        return len(self.global_indices)

    def __getitem__(self, idx: int) -> dict[str, str]:  # noqa: D105
        chunk_id, start, end = self.global_indices[idx]
        token_ids = self.sequences_mmaps[chunk_id][start:end]
        sequence = self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        return {"sequence": sequence}


class CodonMLMCollator:
    """Collator that tokenizes sequences and applies MLM masking for BSHD format."""

    def __init__(
        self,
        tokenizer: CodonTokenizer,
        max_seq_length: int = 512,
        mlm_probability: float = 0.15,
        seed: int = 42,
    ):
        """Initialize.

        Args:
            tokenizer: CodonTokenizer instance.
            max_seq_length: Maximum sequence length (including special tokens).
            mlm_probability: Probability of masking a token.
            seed: Random seed for reproducible masking.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.rng = random.Random(seed)

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Collate a batch of sequences into MLM training inputs.

        Args:
            batch: List of dicts with 'sequence' key.

        Returns:
            Dict with input_ids, attention_mask, and labels tensors.
        """
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for sample in batch:
            ids = self.tokenizer.encode(sample["sequence"], add_special_tokens=True)
            # Truncate to max_seq_length, preserving trailing SEP token
            if len(ids) > self.max_seq_length:
                ids = [*ids[: self.max_seq_length - 1], self.tokenizer.sep_token_id]
            seq_len = len(ids)

            # Create attention mask and pad
            attn_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
            ids = ids + [self.tokenizer.pad_token_id] * (self.max_seq_length - seq_len)

            # Apply MLM masking
            labels = [-100] * self.max_seq_length
            for i in range(seq_len):
                # Skip special tokens (CLS at 0, SEP at end)
                if ids[i] in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id):
                    continue
                if self.rng.random() < self.mlm_probability:
                    labels[i] = ids[i]
                    r = self.rng.random()
                    if r < 0.8:
                        ids[i] = self.tokenizer.mask_token_id
                    elif r < 0.9:
                        # Random codon token (IDs 5 through 68)
                        ids[i] = self.rng.randint(5, self.tokenizer.vocab_size - 1)
                    # else: keep original (10% of the time)

            all_input_ids.append(ids)
            all_attention_masks.append(attn_mask)
            all_labels.append(labels)

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }


class CodonTHDCollator:
    """Collator for THD (packed sequence) format."""

    def __init__(
        self,
        tokenizer: CodonTokenizer,
        max_seq_length: int = 512,
        mlm_probability: float = 0.15,
        seed: int = 42,
        pad_to_multiple_of: int | None = None,
    ):
        """Initialize.

        Args:
            tokenizer: CodonTokenizer instance.
            max_seq_length: Maximum sequence length per sample.
            mlm_probability: Probability of masking a token.
            seed: Random seed for reproducible masking.
            pad_to_multiple_of: If set, pad total tokens to a multiple of this value.
                Required for FP8 (8), MXFP8 (16), or NVFP4 (32) with THD format.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.rng = random.Random(seed)
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        """Collate a batch into THD packed format.

        Args:
            batch: List of dicts with 'sequence' key.

        Returns:
            Dict with input_ids, labels (flattened), cu_seq_lens_q/k, max_length_q/k.
        """
        all_ids = []
        all_labels = []
        seq_lengths = []

        for sample in batch:
            ids = self.tokenizer.encode(sample["sequence"], add_special_tokens=True)
            # Truncate to max_seq_length, preserving trailing SEP token
            if len(ids) > self.max_seq_length:
                ids = [*ids[: self.max_seq_length - 1], self.tokenizer.sep_token_id]
            seq_len = len(ids)

            # Apply MLM masking
            labels = [-100] * seq_len
            for i in range(seq_len):
                if ids[i] in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id):
                    continue
                if self.rng.random() < self.mlm_probability:
                    labels[i] = ids[i]
                    r = self.rng.random()
                    if r < 0.8:
                        ids[i] = self.tokenizer.mask_token_id
                    elif r < 0.9:
                        ids[i] = self.rng.randint(5, self.tokenizer.vocab_size - 1)

            all_ids.extend(ids)
            all_labels.extend(labels)
            seq_lengths.append(seq_len)

        cu_seq_lens = torch.zeros(len(seq_lengths) + 1, dtype=torch.int32)
        cu_seq_lens[1:] = torch.cumsum(torch.tensor(seq_lengths, dtype=torch.int32), dim=0)

        input_ids = torch.tensor(all_ids, dtype=torch.long).unsqueeze(0)
        labels = torch.tensor(all_labels, dtype=torch.long).unsqueeze(0)
        max_length = max(seq_lengths)

        # Pad total tokens to a multiple of pad_to_multiple_of by appending mock padding sequences.
        # Required for FP8/FP4 with THD format where TE needs the first dim divisible by 8/16/32.
        if self.pad_to_multiple_of is not None:
            remainder = -input_ids.numel() % self.pad_to_multiple_of
            if remainder > 0:
                input_ids = torch.cat(
                    [input_ids, torch.full((1, remainder), self.tokenizer.pad_token_id, dtype=input_ids.dtype)], dim=1
                )
                labels = torch.cat([labels, torch.full((1, remainder), -100, dtype=labels.dtype)], dim=1)
                # Split padding into multiple sequences each <= max_seq_length to stay
                # within the RoPE position embedding range. A single oversized mock sequence
                # would cause TE's fused RoPE kernel to read out-of-bounds, producing NaN.
                pad_cu_lens = []
                offset = cu_seq_lens[-1].item()
                remaining = remainder
                while remaining > 0:
                    chunk = min(remaining, self.max_seq_length)
                    offset += chunk
                    pad_cu_lens.append(offset)
                    remaining -= chunk
                cu_seq_lens = torch.cat([cu_seq_lens, torch.tensor(pad_cu_lens, dtype=cu_seq_lens.dtype)])
                max_length = max(max_length, min(remainder, self.max_seq_length))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length,
        }


def _create_dataset(data_path: str, max_seq_length: int, seed: int) -> Dataset:
    """Create the appropriate dataset based on data_path format.

    Args:
        data_path: 'synthetic', path to a parquet file, or path to a memmap directory.
        max_seq_length: Maximum sequence length (used for memmap sliding windows).
        seed: Random seed.

    Returns:
        A Dataset instance.
    """
    if data_path == "synthetic":
        return SyntheticCodonDataset(num_samples=500, seed=seed)
    data_dir = Path(data_path)
    if data_dir.is_dir() and (data_dir / "metadata.json").exists():
        return MemmapCodonDataset(data_path, max_seq_length=max_seq_length)
    return ParquetCodonDataset(data_path)


def create_bshd_dataloader(
    dist_config: DistributedConfig,
    data_path: str,
    micro_batch_size: int = 2,
    max_seq_length: int = 512,
    mlm_probability: float = 0.15,
    num_workers: int = 1,
    seed: int = 42,
    pad_to_multiple_of: int | None = None,
) -> tuple[DataLoader, DistributedSampler]:
    """Create a BSHD-format dataloader.

    Args:
        dist_config: Distributed configuration.
        data_path: Path to parquet file or 'synthetic'.
        micro_batch_size: Batch size per GPU.
        max_seq_length: Maximum sequence length.
        mlm_probability: MLM masking probability.
        num_workers: Number of dataloader workers.
        seed: Random seed.
        pad_to_multiple_of: Unused in BSHD mode (only applies to THD).

    Returns:
        Tuple of (DataLoader, DistributedSampler).
    """
    tokenizer = CodonTokenizer()

    dataset = _create_dataset(data_path, max_seq_length, seed)

    sampler = DistributedSampler(
        dataset,
        rank=dist_config.rank,
        num_replicas=dist_config.world_size,
        seed=seed,
    )

    collator = CodonMLMCollator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, sampler


def create_thd_dataloader(
    dist_config: DistributedConfig,
    data_path: str,
    micro_batch_size: int = 2,
    max_seq_length: int = 512,
    mlm_probability: float = 0.15,
    num_workers: int = 1,
    seed: int = 42,
    pad_to_multiple_of: int | None = None,
) -> tuple[DataLoader, DistributedSampler]:
    """Create a THD-format (packed sequence) dataloader.

    Args:
        dist_config: Distributed configuration.
        data_path: Path to parquet file or 'synthetic'.
        micro_batch_size: Number of sequences to pack per batch.
        max_seq_length: Maximum sequence length per sample.
        mlm_probability: MLM masking probability.
        num_workers: Number of dataloader workers.
        seed: Random seed.
        pad_to_multiple_of: If set, pad total tokens to a multiple of this value. If None,
            defaults to micro_batch_size * max_seq_length for consistent tensor shapes
            (matching ESM2's approach). Set to 0 to disable padding.

    Returns:
        Tuple of (DataLoader, DistributedSampler).
    """
    tokenizer = CodonTokenizer()

    # Default pad_to_multiple_of to token_micro_batch_size for consistent tensor shapes,
    # matching ESM2's approach: pad_to_multiple_of = micro_batch_size * max_seq_length.
    # This guarantees divisibility by 8/16/32 for FP8/FP4 since max_seq_length (512) is already divisible.
    if pad_to_multiple_of is None:
        pad_to_multiple_of = micro_batch_size * max_seq_length
    elif pad_to_multiple_of == 0:
        pad_to_multiple_of = None

    dataset = _create_dataset(data_path, max_seq_length, seed)

    sampler = DistributedSampler(
        dataset,
        rank=dist_config.rank,
        num_replicas=dist_config.world_size,
        seed=seed,
    )

    collator = CodonTHDCollator(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        mlm_probability=mlm_probability,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=micro_batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataloader, sampler
