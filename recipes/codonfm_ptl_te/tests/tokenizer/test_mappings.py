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


import pytest

from src.tokenizer.mappings import AA_TABLE, CODON_TABLE, CODON_TO_AA


class TestCodonTable:
    def test_rna_codons(self):
        rna_codons = CODON_TABLE["rna"]
        assert isinstance(rna_codons, dict)
        assert len(rna_codons) == 64
        assert rna_codons["AUG"] == "M"
        assert rna_codons["UAA"] == "*"
        assert "UUU" in rna_codons
        assert "TTT" not in rna_codons

    def test_dna_codons(self):
        dna_codons = CODON_TABLE["dna"]
        assert isinstance(dna_codons, dict)
        assert len(dna_codons) == 64
        assert dna_codons["ATG"] == "M"
        assert dna_codons["TAA"] == "*"
        assert "TTT" in dna_codons
        assert "UUU" not in dna_codons

    def test_invalid_type(self):
        with pytest.raises(AssertionError):
            CODON_TABLE["invalid"]


class TestAATable:
    def test_rna_aas(self):
        rna_aas = AA_TABLE["rna"]
        assert isinstance(rna_aas, dict)
        assert "M" in rna_aas
        assert "AUG" in rna_aas["M"]
        assert "UAA" not in rna_aas  # Stop codons are not keys
        assert "*" in rna_aas
        assert "UAA" in rna_aas["*"]
        assert "UAG" in rna_aas["*"]
        assert "UGA" in rna_aas["*"]
        assert "UUU" in rna_aas["F"]
        assert "UUC" in rna_aas["F"]

    def test_dna_aas(self):
        dna_aas = AA_TABLE["dna"]
        assert isinstance(dna_aas, dict)
        assert "M" in dna_aas
        assert "ATG" in dna_aas["M"]
        assert "TAA" not in dna_aas  # Stop codons are not keys
        assert "*" in dna_aas
        assert "TAA" in dna_aas["*"]
        assert "TAG" in dna_aas["*"]
        assert "TGA" in dna_aas["*"]
        assert "TTT" in dna_aas["F"]
        assert "TTC" in dna_aas["F"]

    def test_invalid_type(self):
        with pytest.raises(AssertionError):
            AA_TABLE["invalid"]


def test_mapping_consistency():
    # Check if all codons in CODON_TO_AA are present in CODON_TABLE['rna']
    for codon, aa in CODON_TO_AA.items():
        assert CODON_TABLE["rna"][codon] == aa

    # Check if all codons are present in AA_TABLE
    for aa, codons in AA_TABLE["rna"].items():
        for codon in codons:
            assert CODON_TABLE["rna"][codon] == aa

    for aa, codons in AA_TABLE["dna"].items():
        for codon in codons:
            assert CODON_TABLE["dna"][codon] == aa

    # Check DNA/RNA consistency
    assert set(c.replace("U", "T") for c in CODON_TABLE["rna"].keys()) == set(CODON_TABLE["dna"].keys())
    assert set(CODON_TABLE["rna"].values()) == set(CODON_TABLE["dna"].values())
