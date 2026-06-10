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

"""Dataset for protein structure prediction training.

Provides:
- SyntheticStructureDataset: generates random data for testing
- ParquetStructureDataset: loads from parquet (pre-processed Ca coords)
- MmcifStructureDataset: loads from mmCIF files on-the-fly via BioPython
- create_dataloader: factory function for any dataset type
"""

import logging
from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)
_TOKENIZER_FALLBACK = "facebook/esm2_t33_650M_UR50D"


def _load_esm_tokenizer(tokenizer_name: str):
    from transformers import EsmTokenizer

    try:
        return EsmTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    except Exception:
        try:
            return EsmTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            logger.warning("Falling back to cached ESM-2 tokenizer %s for %s", _TOKENIZER_FALLBACK, tokenizer_name)
            return EsmTokenizer.from_pretrained(_TOKENIZER_FALLBACK, local_files_only=True)


class SyntheticStructureDataset(Dataset):
    """Generates synthetic protein structure data for testing.

    Each sample contains:
        input_ids: Random ESM-2 token IDs (L,)
        attention_mask: All ones (L,)
        mask: All ones (L,) as float
        coords: Random Ca coordinates (L, 3)
    """

    def __init__(self, num_samples: int = 1000, max_seq_length: int = 128, seed: int = 42):
        self.num_samples = num_samples
        self.max_seq_length = max_seq_length
        self.rng = torch.Generator().manual_seed(seed)

        # ESM-2 special tokens: 0=cls, 1=pad, 2=eos, 3=unk, then 4-23 are AA tokens
        self.vocab_start = 4
        self.vocab_end = 24  # 20 amino acid tokens

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random sequence length (at least 16 residues)
        seq_len = torch.randint(16, self.max_seq_length - 2, (1,), generator=self.rng).item()

        # Generate random tokens: [CLS] + AA tokens + [EOS] + padding
        tokens = torch.randint(self.vocab_start, self.vocab_end, (seq_len,), generator=self.rng)
        input_ids = torch.zeros(self.max_seq_length, dtype=torch.long)
        input_ids[0] = 0  # CLS
        input_ids[1 : seq_len + 1] = tokens
        input_ids[seq_len + 1] = 2  # EOS

        # Attention mask: 1 for real tokens, 0 for padding
        attention_mask = torch.zeros(self.max_seq_length, dtype=torch.long)
        attention_mask[: seq_len + 2] = 1  # CLS + tokens + EOS

        # Residue mask: 1 for real residues (excludes special tokens)
        mask = torch.zeros(self.max_seq_length, dtype=torch.float)
        mask[: seq_len + 2] = 1.0

        # Random Ca coordinates (Angstroms)
        coords = torch.randn(self.max_seq_length, 3, generator=self.rng) * 10.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask,
            "coords": coords,
        }


class ParquetStructureDataset(Dataset):
    """Loads protein structures from a parquet file.

    Expected columns:
        sequence: str - amino acid sequence
        coords: list[list[float]] - Ca coordinates (N, 3)
        ca_mask: list[int] - optional, 1=valid Ca, 0=missing (defaults to all-1s)
    """

    def __init__(self, parquet_path: str, tokenizer, max_seq_length: int = 256):
        import pandas as pd

        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.has_ca_mask = "ca_mask" in self.df.columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sequence = row["sequence"]

        # Tokenize
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Mask is same as attention_mask but as float
        mask = attention_mask.float()

        # Coordinates: pad to max_seq_length
        import numpy as np

        # Parquet stores list-of-lists as numpy object array of arrays; np.stack handles this
        coords_raw = torch.from_numpy(np.stack(row["coords"]).astype(np.float32))
        coords = torch.zeros(self.max_seq_length, 3)
        seq_len = min(len(coords_raw), self.max_seq_length)
        coords[:seq_len] = coords_raw[:seq_len]

        # Zero out coords for residues with missing Ca atoms
        if self.has_ca_mask:
            ca_mask_list = row["ca_mask"]
            for i in range(min(len(ca_mask_list), self.max_seq_length)):
                if ca_mask_list[i] == 0:
                    coords[i] = 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask,
            "coords": coords,
        }


class MmcifStructureDataset(Dataset):
    """Loads protein structures directly from mmCIF files via BioPython.

    Parses each .cif file on-the-fly, extracts the amino acid sequence and Ca
    coordinates, tokenizes with ESM-2, and returns the standard batch format.
    """

    # 3-letter to 1-letter amino acid mapping
    AA_3TO1: ClassVar[dict[str, str]] = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
        "MSE": "M",
    }

    def __init__(
        self,
        cif_dir: str,
        tokenizer,
        max_seq_length: int = 256,
        pdb_ids: list[str] | None = None,
        min_residues: int = 50,
        max_residues: int = 300,
        min_ca_completeness: float = 0.9,
    ):
        from Bio.PDB.MMCIFParser import MMCIFParser

        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.min_residues = min_residues
        self.max_residues = max_residues
        self.min_ca_completeness = min_ca_completeness
        self.parser = MMCIFParser(QUIET=True)

        cif_path = Path(cif_dir)
        all_files = sorted(cif_path.glob("*.cif"))

        if pdb_ids is not None:
            # Preserve caller's ordering (e.g., to match parquet row order)
            file_by_id = {f.stem.upper(): f for f in all_files}
            self.files = [file_by_id[pid.upper()] for pid in pdb_ids if pid.upper() in file_by_id]
        else:
            self.files = all_files

        if not self.files:
            raise FileNotFoundError(f"No .cif files found in {cif_dir}")

        logger.info("MmcifStructureDataset: %d CIF files from %s", len(self.files), cif_dir)

    def __len__(self):
        return len(self.files)

    def _parse_cif(self, cif_path):
        """Parse mmCIF file and extract sequence + Ca coordinates.

        Uses the same filtering as prepare_pdb_dataset.py: min/max residues,
        Ca completeness threshold, and truncation to max_residues.

        Returns (sequence, ca_coords, ca_mask) or raises on failure.
        """
        pdb_id = cif_path.stem
        structure = self.parser.get_structure(pdb_id, str(cif_path))
        model = structure[0]

        for chain in model:
            residues = []
            for res in chain.get_residues():
                if res.id[0] != " ":
                    continue
                resname = res.get_resname().strip()
                if resname not in self.AA_3TO1:
                    continue
                residues.append(res)

            if len(residues) < self.min_residues:
                continue
            if len(residues) > self.max_residues:
                residues = residues[: self.max_residues]

            sequence = []
            coords = []
            ca_mask = []
            for res in residues:
                resname = res.get_resname().strip()
                sequence.append(self.AA_3TO1[resname])
                if "CA" in res:
                    ca = res["CA"].get_vector()
                    coords.append([float(ca[0]), float(ca[1]), float(ca[2])])
                    ca_mask.append(1)
                else:
                    coords.append([0.0, 0.0, 0.0])
                    ca_mask.append(0)

            completeness = sum(ca_mask) / len(ca_mask)
            if completeness < self.min_ca_completeness:
                continue

            return "".join(sequence), coords, ca_mask

        raise ValueError(f"No valid protein chain in {pdb_id}")

    def __getitem__(self, idx):
        try:
            sequence, ca_coords, ca_mask = self._parse_cif(self.files[idx])
        except Exception as e:
            logger.warning("Failed to parse %s: %s, falling back to index 0", self.files[idx].name, e)
            sequence, ca_coords, ca_mask = self._parse_cif(self.files[0])

        # Tokenize
        encoded = self.tokenizer(
            sequence,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        mask = attention_mask.float()

        # Coordinates: pad to max_seq_length
        coords_raw = torch.tensor(ca_coords, dtype=torch.float32)
        coords = torch.zeros(self.max_seq_length, 3)
        seq_len = min(len(coords_raw), self.max_seq_length)
        coords[:seq_len] = coords_raw[:seq_len]

        # Zero out missing Ca positions
        for i in range(seq_len):
            if ca_mask[i] == 0:
                coords[i] = 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "mask": mask,
            "coords": coords,
        }


def create_dataloader(
    dist_config: DistributedConfig,
    micro_batch_size: int = 2,
    max_seq_length: int = 128,
    num_workers: int = 0,
    dataset_type: str = "synthetic",
    parquet_path: str | None = None,
    tokenizer_name: str | None = None,
    num_samples: int = 1000,
    seed: int = 42,
    cif_dir: str | None = None,
    pdb_ids: list[str] | None = None,
    shuffle: bool = True,
    drop_last: bool = True,
    **kwargs,
):
    """Create a DataLoader for structure prediction training or evaluation.

    Args:
        dist_config: Distributed training configuration.
        micro_batch_size: Batch size per GPU.
        max_seq_length: Maximum sequence length.
        num_workers: Number of DataLoader workers.
        dataset_type: "synthetic", "parquet", or "mmcif".
        parquet_path: Path to parquet file (required if dataset_type="parquet").
        tokenizer_name: HuggingFace tokenizer name (required if dataset_type="parquet" or "mmcif").
        num_samples: Number of synthetic samples.
        cif_dir: Directory with .cif files (required if dataset_type="mmcif").
        pdb_ids: Optional list of PDB IDs to filter (for dataset_type="mmcif").
        shuffle: Whether to shuffle the data (False for eval).
        drop_last: Whether to drop the last incomplete batch (False for eval).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Tuple of (DataLoader, DistributedSampler).
    """
    if dataset_type == "synthetic":
        dataset = SyntheticStructureDataset(
            num_samples=num_samples,
            max_seq_length=max_seq_length,
            seed=seed,
        )
    elif dataset_type == "parquet":
        tokenizer = _load_esm_tokenizer(tokenizer_name)
        dataset = ParquetStructureDataset(
            parquet_path=parquet_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
    elif dataset_type == "mmcif":
        tokenizer = _load_esm_tokenizer(tokenizer_name)
        dataset = MmcifStructureDataset(
            cif_dir=cif_dir,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            pdb_ids=pdb_ids,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=dist_config.world_size,
        rank=dist_config.rank,
        shuffle=shuffle,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )

    return dataloader, sampler
