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

import sys
from pathlib import Path

import pytest


sys.path.append(Path(__file__).parent.parent.as_posix())

from tokenizer import CodonTokenizer


@pytest.fixture
def tokenizer():
    return CodonTokenizer(seq_type="dna")


class TestCodonTokenizer:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 69  # 5 special + 64 codons

    def test_special_token_ids(self, tokenizer):
        assert tokenizer.cls_token_id == 0
        assert tokenizer.sep_token_id == 1
        assert tokenizer.unk_token_id == 2
        assert tokenizer.pad_token_id == 3
        assert tokenizer.mask_token_id == 4

    def test_all_codons_present(self, tokenizer):
        assert len(tokenizer.codons) == 64
        # Check first and last codons
        assert "AAA" in tokenizer.codons
        assert "TTT" in tokenizer.codons

    def test_tokenize(self, tokenizer):
        tokens = tokenizer.tokenize("ATGCGA")
        assert tokens == ["ATG", "CGA"]

    def test_tokenize_incomplete_codon(self, tokenizer):
        tokens = tokenizer.tokenize("ATGCG")
        assert tokens == ["ATG"]  # Incomplete codon dropped

    def test_encode_with_special_tokens(self, tokenizer):
        ids = tokenizer.encode("ATGCGA", add_special_tokens=True)
        assert ids[0] == tokenizer.cls_token_id
        assert ids[-1] == tokenizer.sep_token_id
        assert len(ids) == 4  # CLS + ATG + CGA + SEP

    def test_encode_without_special_tokens(self, tokenizer):
        ids = tokenizer.encode("ATGCGA", add_special_tokens=False)
        assert len(ids) == 2  # ATG + CGA
        assert ids[0] == tokenizer.encoder["ATG"]
        assert ids[1] == tokenizer.encoder["CGA"]

    def test_decode_roundtrip(self, tokenizer):
        original = "ATGCGATTT"
        ids = tokenizer.encode(original, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == original

    def test_decode_with_special_tokens(self, tokenizer):
        ids = tokenizer.encode("ATGCGA", add_special_tokens=True)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == "ATGCGA"

    def test_case_insensitive(self, tokenizer):
        ids_upper = tokenizer.encode("ATGCGA", add_special_tokens=False)
        ids_lower = tokenizer.encode("atgcga", add_special_tokens=False)
        assert ids_upper == ids_lower

    def test_rna_tokenizer(self):
        rna_tok = CodonTokenizer(seq_type="rna")
        assert rna_tok.vocab_size == 69
        assert "AUG" in rna_tok.codons
        assert "TTT" not in rna_tok.codons
        assert "UUU" in rna_tok.codons

    def test_invalid_seq_type(self):
        with pytest.raises(ValueError, match="seq_type"):
            CodonTokenizer(seq_type="protein")

    def test_comparison_with_codonfm_ptl_te(self):
        """Verify that our tokenizer produces the same vocab as codonfm_ptl_te."""
        try:
            ptl_te_path = Path(__file__).parent.parent.parent / "codonfm_ptl_te"
            if not ptl_te_path.exists():
                pytest.skip("codonfm_ptl_te recipe not found")

            sys.path.insert(0, str(ptl_te_path))
            from src.tokenizer.tokenizer import Tokenizer as PTLTokenizer

            ptl_tokenizer = PTLTokenizer(seq_type="dna")
            our_tokenizer = CodonTokenizer(seq_type="dna")

            # Same vocab size
            assert our_tokenizer.vocab_size == ptl_tokenizer.vocab_size

            # Same codon list
            assert our_tokenizer.codons == ptl_tokenizer.codons

            # Same special token IDs
            assert our_tokenizer.cls_token_id == ptl_tokenizer.cls_token_id
            assert our_tokenizer.sep_token_id == ptl_tokenizer.sep_token_id
            assert our_tokenizer.pad_token_id == ptl_tokenizer.pad_token_id
            assert our_tokenizer.mask_token_id == ptl_tokenizer.mask_token_id

            # Same encoding for a sample sequence
            test_seq = "ATGCGATTTAAACCC"
            our_ids = our_tokenizer.encode(test_seq, add_special_tokens=False)
            ptl_ids = ptl_tokenizer.encode(test_seq)
            # PTL tokenizer may add special tokens by default, strip them
            # Compare just the codon token IDs
            ptl_codon_ids = [i for i in ptl_ids if i >= 5]  # Skip special tokens
            assert our_ids == ptl_codon_ids
        finally:
            # Clean up sys.path
            if str(ptl_te_path) in sys.path:
                sys.path.remove(str(ptl_te_path))
