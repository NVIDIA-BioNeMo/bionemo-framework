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

import random

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
    ):
        """Initialize.

        Args:
            tokenizer: CodonTokenizer instance.
            max_seq_length: Maximum sequence length per sample.
            mlm_probability: Probability of masking a token.
            seed: Random seed for reproducible masking.
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mlm_probability = mlm_probability
        self.rng = random.Random(seed)

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

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long).unsqueeze(0),
            "labels": torch.tensor(all_labels, dtype=torch.long).unsqueeze(0),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max(seq_lengths),
            "max_length_k": max(seq_lengths),
        }


def create_bshd_dataloader(
    dist_config: DistributedConfig,
    data_path: str,
    micro_batch_size: int = 2,
    max_seq_length: int = 512,
    mlm_probability: float = 0.15,
    num_workers: int = 1,
    seed: int = 42,
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

    Returns:
        Tuple of (DataLoader, DistributedSampler).
    """
    tokenizer = CodonTokenizer()

    if data_path == "synthetic":
        dataset = SyntheticCodonDataset(num_samples=500, seed=seed)
    else:
        dataset = ParquetCodonDataset(data_path)

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

    Returns:
        Tuple of (DataLoader, DistributedSampler).
    """
    tokenizer = CodonTokenizer()

    if data_path == "synthetic":
        dataset = SyntheticCodonDataset(num_samples=500, seed=seed)
    else:
        dataset = ParquetCodonDataset(data_path)

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
