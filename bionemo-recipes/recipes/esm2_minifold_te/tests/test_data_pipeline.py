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

"""Tests for the PDB data pipeline.

Validates:
- mmCIF parsing correctness (BioPython)
- MmcifStructureDataset batch format
- ParquetStructureDataset batch format
- Equivalence between both dataset implementations

Requires network access to download test structures from RCSB PDB.
Run with: pytest tests/test_data_pipeline.py -v
"""

import sys
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
import pytest
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import MmcifStructureDataset, ParquetStructureDataset


# Test protein: 1CRN (crambin, 46 residues, very high resolution)
TEST_PDB_ID = "1CRN"
TEST_PDB_URL = f"https://files.rcsb.org/download/{TEST_PDB_ID}.cif"
TEST_SEQ_LENGTH = 46
MAX_SEQ_LENGTH = 64


@pytest.fixture(scope="session")
def tokenizer():
    """Load ESM-2 tokenizer (small model for speed)."""
    from transformers import EsmTokenizer

    return EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")


@pytest.fixture(scope="session")
def cif_dir(tmp_path_factory):
    """Download 1CRN.cif to a temp directory."""
    d = tmp_path_factory.mktemp("cif_files")
    cif_path = d / f"{TEST_PDB_ID}.cif"
    urlretrieve(TEST_PDB_URL, cif_path)
    return str(d)


@pytest.fixture(scope="session")
def parsed_data(cif_dir):
    """Parse 1CRN and return (sequence, coords, ca_mask)."""
    from Bio.PDB.MMCIFParser import MMCIFParser

    aa_3to1 = {
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
    }
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(TEST_PDB_ID, str(Path(cif_dir) / f"{TEST_PDB_ID}.cif"))
    model = structure[0]
    chain = next(iter(model))

    sequence, coords, ca_mask = [], [], []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        resname = res.get_resname().strip()
        if resname not in aa_3to1:
            continue
        sequence.append(aa_3to1[resname])
        if "CA" in res:
            ca = res["CA"].get_vector()
            coords.append([float(ca[0]), float(ca[1]), float(ca[2])])
            ca_mask.append(1)
        else:
            coords.append([0.0, 0.0, 0.0])
            ca_mask.append(0)

    return "".join(sequence), coords, ca_mask


@pytest.fixture(scope="session")
def parquet_path(parsed_data, tmp_path_factory):
    """Create a parquet file from parsed 1CRN data."""
    sequence, coords, ca_mask = parsed_data
    d = tmp_path_factory.mktemp("parquet")
    path = d / "test_structures.parquet"
    df = pd.DataFrame(
        [
            {
                "pdb_id": TEST_PDB_ID,
                "sequence": sequence,
                "coords": coords,
                "ca_mask": ca_mask,
                "num_residues": len(sequence),
            }
        ]
    )
    df.to_parquet(str(path), index=False)
    return str(path)


# ===========================================================================
# CIF Parsing
# ===========================================================================


class TestCifParsing:
    def test_sequence_length(self, parsed_data):
        sequence, coords, ca_mask = parsed_data
        assert len(sequence) == TEST_SEQ_LENGTH, f"Expected {TEST_SEQ_LENGTH} residues, got {len(sequence)}"

    def test_coords_count_matches_sequence(self, parsed_data):
        sequence, coords, ca_mask = parsed_data
        assert len(coords) == len(sequence)
        assert len(ca_mask) == len(sequence)

    def test_ca_coords_finite(self, parsed_data):
        _, coords, _ = parsed_data
        for i, c in enumerate(coords):
            assert all(abs(v) < 1e6 for v in c), f"Non-finite coords at residue {i}: {c}"

    def test_all_ca_present(self, parsed_data):
        """1CRN is a high-quality structure - all Ca should be present."""
        _, _, ca_mask = parsed_data
        assert all(m == 1 for m in ca_mask), "1CRN should have all Ca atoms resolved"

    def test_sequence_standard_amino_acids(self, parsed_data):
        sequence, _, _ = parsed_data
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        for i, aa in enumerate(sequence):
            assert aa in valid_aa, f"Non-standard amino acid '{aa}' at position {i}"

    def test_first_residue_is_threonine(self, parsed_data):
        """1CRN starts with Thr-Thr-Cys."""
        sequence, _, _ = parsed_data
        assert sequence[:3] == "TTC", f"Expected TTC, got {sequence[:3]}"


# ===========================================================================
# MmcifStructureDataset
# ===========================================================================


class TestMmcifStructureDataset:
    def test_batch_keys(self, cif_dir, tokenizer):
        ds = MmcifStructureDataset(cif_dir, tokenizer, max_seq_length=MAX_SEQ_LENGTH, min_residues=20)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "mask", "coords"}

    def test_batch_shapes(self, cif_dir, tokenizer):
        ds = MmcifStructureDataset(cif_dir, tokenizer, max_seq_length=MAX_SEQ_LENGTH, min_residues=20)
        sample = ds[0]
        assert sample["input_ids"].shape == (MAX_SEQ_LENGTH,)
        assert sample["attention_mask"].shape == (MAX_SEQ_LENGTH,)
        assert sample["mask"].shape == (MAX_SEQ_LENGTH,)
        assert sample["coords"].shape == (MAX_SEQ_LENGTH, 3)

    def test_batch_dtypes(self, cif_dir, tokenizer):
        ds = MmcifStructureDataset(cif_dir, tokenizer, max_seq_length=MAX_SEQ_LENGTH, min_residues=20)
        sample = ds[0]
        assert sample["input_ids"].dtype == torch.long
        assert sample["attention_mask"].dtype == torch.long
        assert sample["mask"].dtype == torch.float32
        assert sample["coords"].dtype == torch.float32

    def test_cls_eos_tokens(self, cif_dir, tokenizer):
        ds = MmcifStructureDataset(cif_dir, tokenizer, max_seq_length=MAX_SEQ_LENGTH, min_residues=20)
        sample = ds[0]
        assert sample["input_ids"][0].item() == 0, "First token should be CLS (0)"
        # Find EOS position
        real_len = sample["attention_mask"].sum().item()
        assert sample["input_ids"][int(real_len) - 1].item() == 2, "Last real token should be EOS (2)"

    def test_padding_is_zero(self, cif_dir, tokenizer):
        ds = MmcifStructureDataset(cif_dir, tokenizer, max_seq_length=MAX_SEQ_LENGTH, min_residues=20)
        sample = ds[0]
        real_len = sample["attention_mask"].sum().item()
        assert (sample["attention_mask"][int(real_len) :] == 0).all()
        assert (sample["coords"][TEST_SEQ_LENGTH:] == 0).all()


# ===========================================================================
# ParquetStructureDataset
# ===========================================================================


class TestParquetStructureDataset:
    def test_batch_keys(self, parquet_path, tokenizer):
        ds = ParquetStructureDataset(parquet_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
        sample = ds[0]
        assert set(sample.keys()) == {"input_ids", "attention_mask", "mask", "coords"}

    def test_batch_shapes(self, parquet_path, tokenizer):
        ds = ParquetStructureDataset(parquet_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
        sample = ds[0]
        assert sample["input_ids"].shape == (MAX_SEQ_LENGTH,)
        assert sample["coords"].shape == (MAX_SEQ_LENGTH, 3)

    def test_batch_dtypes(self, parquet_path, tokenizer):
        ds = ParquetStructureDataset(parquet_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
        sample = ds[0]
        assert sample["input_ids"].dtype == torch.long
        assert sample["coords"].dtype == torch.float32


# ===========================================================================
# Dataset Equivalence
# ===========================================================================


class TestDatasetEquivalence:
    """Both datasets must produce matching outputs for the same protein."""

    def _get_samples(self, cif_dir, parquet_path, tokenizer):
        ds_cif = MmcifStructureDataset(cif_dir, tokenizer, max_seq_length=MAX_SEQ_LENGTH, min_residues=20)
        ds_pq = ParquetStructureDataset(parquet_path, tokenizer, max_seq_length=MAX_SEQ_LENGTH)
        return ds_cif[0], ds_pq[0]

    def test_same_input_ids(self, cif_dir, parquet_path, tokenizer):
        s_cif, s_pq = self._get_samples(cif_dir, parquet_path, tokenizer)
        assert torch.equal(s_cif["input_ids"], s_pq["input_ids"]), "input_ids mismatch"

    def test_same_attention_mask(self, cif_dir, parquet_path, tokenizer):
        s_cif, s_pq = self._get_samples(cif_dir, parquet_path, tokenizer)
        assert torch.equal(s_cif["attention_mask"], s_pq["attention_mask"]), "attention_mask mismatch"

    def test_same_mask(self, cif_dir, parquet_path, tokenizer):
        s_cif, s_pq = self._get_samples(cif_dir, parquet_path, tokenizer)
        assert torch.equal(s_cif["mask"], s_pq["mask"]), "mask mismatch"

    def test_same_coords(self, cif_dir, parquet_path, tokenizer):
        s_cif, s_pq = self._get_samples(cif_dir, parquet_path, tokenizer)
        assert torch.allclose(s_cif["coords"], s_pq["coords"], atol=1e-4), (
            f"coords max diff: {(s_cif['coords'] - s_pq['coords']).abs().max().item()}"
        )

    def test_distogram_loss_equivalence(self, cif_dir, parquet_path, tokenizer):
        """Both datasets should produce the same distogram loss."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from train_fsdp2 import compute_distogram_loss

        s_cif, s_pq = self._get_samples(cif_dir, parquet_path, tokenizer)

        # Fake preds (same for both)
        torch.manual_seed(42)
        preds = torch.randn(1, MAX_SEQ_LENGTH, MAX_SEQ_LENGTH, 64)

        loss_cif = compute_distogram_loss(preds, s_cif["coords"].unsqueeze(0), s_cif["mask"].unsqueeze(0))
        loss_pq = compute_distogram_loss(preds, s_pq["coords"].unsqueeze(0), s_pq["mask"].unsqueeze(0))

        assert torch.allclose(loss_cif, loss_pq, atol=1e-4), (
            f"Loss mismatch: cif={loss_cif.item()}, pq={loss_pq.item()}"
        )
