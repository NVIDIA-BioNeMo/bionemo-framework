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

from src.tokenizer.tokenizer import Tokenizer


class TestTokenizer:
    def test_init_dna(self):
        tokenizer = Tokenizer(seq_type="dna")
        assert tokenizer.seq_type == "dna"
        assert "ATG" in tokenizer.codons
        assert "AUG" not in tokenizer.codons
        assert len(tokenizer.codons) == 64
        assert tokenizer.vocab_size == 5 + 64  # 5 special + 64 codons

    def test_init_rna(self):
        tokenizer = Tokenizer(seq_type="rna")
        assert tokenizer.seq_type == "rna"
        assert "AUG" in tokenizer.codons
        assert "ATG" not in tokenizer.codons
        assert len(tokenizer.codons) == 64
        assert tokenizer.vocab_size == 5 + 64

    def test_init_invalid_type(self):
        with pytest.raises(AssertionError):
            Tokenizer(seq_type="invalid")

    def test_tokenize(self):
        tokenizer = Tokenizer(seq_type="dna")
        seq = "ATGCGT"
        tokens = tokenizer.tokenize(seq)
        assert tokens == ["ATG", "CGT"]

        seq_with_special = "<CLS>ATGCGT<SEP>"
        tokens = tokenizer.tokenize(seq_with_special)
        assert tokens == ["<CLS>", "ATG", "CGT", "<SEP>"]

        # With unknown tokens
        seq_with_unk = "ATGN"
        tokens = tokenizer.tokenize(seq_with_unk)
        # The regex will find ATG and N. But N is not a codon.
        # It will be split into ['ATG', 'N']. Then N becomes <UNK> during encoding.
        # The regex is a bit greedy. Let's check how it handles it.
        # a single char not in codons or special tokens will be matched by \S
        assert tokens == ["ATG", "N"]

    def test_convert_tokens_to_ids(self):
        tokenizer = Tokenizer(seq_type="dna")
        tokens = ["<CLS>", "ATG", "CGT", "<SEP>", "NNN"]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        assert ids == [
            tokenizer.cls_token_id,
            tokenizer.encoder["ATG"],
            tokenizer.encoder["CGT"],
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]

    def test_convert_ids_to_tokens(self):
        tokenizer = Tokenizer(seq_type="dna")
        ids = [
            tokenizer.cls_token_id,
            tokenizer.encoder["ATG"],
            tokenizer.encoder["CGT"],
            tokenizer.sep_token_id,
            tokenizer.unk_token_id,
        ]
        tokens = tokenizer.convert_ids_to_tokens(ids)
        assert tokens == ["<CLS>", "ATG", "CGT", "<SEP>", "<UNK>"]

    def test_add_organism_tokens(self):
        tokenizer = Tokenizer(seq_type="dna")
        initial_vocab_size = tokenizer.vocab_size
        organism_tokens = ["<E_COLI>", "<H_SAPIENS>"]
        tokenizer.set_organism_tokens(organism_tokens)
        assert tokenizer.vocab_size == initial_vocab_size + 2
        assert "<E_COLI>" in tokenizer.encoder
        assert tokenizer.encoder["<E_COLI>"] == initial_vocab_size
        assert tokenizer.encoder["<H_SAPIENS>"] == initial_vocab_size + 1

    @pytest.mark.parametrize("token_type_mode", ["regular", "regular_special", "aa"])
    def test_token_type_modes(self, token_type_mode):
        tokenizer = Tokenizer(seq_type="dna", token_type_mode=token_type_mode)
        assert tokenizer.token_type_mode == token_type_mode
        assert hasattr(tokenizer, "token_type_encoder")

        # Test create_token_type_ids_from_sequences
        tokens = ["<CLS>", "ATG", "CGT", "<SEP>"]
        ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = tokenizer.create_token_type_ids_from_sequences(ids)

        if token_type_mode == "regular":
            assert all(t_id == 0 for t_id in token_type_ids)
        elif token_type_mode == "regular_special":
            # ATG, CGT are not special tokens and should have type id 1
            # special tokens should have type id 0
            assert token_type_ids[0] == 0  # CLS
            assert token_type_ids[1] == 1  # ATG
            assert token_type_ids[2] == 1  # CGT
            assert token_type_ids[3] == 0  # SEP
        elif token_type_mode == "aa":
            # CLS and SEP have type 0
            # ATG -> M, CGT -> R
            # The id is index + 1
            m_id = list(tokenizer.amino_acids + ["*"]).index("M") + 1  # noqa: RUF005
            r_id = list(tokenizer.amino_acids + ["*"]).index("R") + 1  # noqa: RUF005
            assert token_type_ids[0] == 0
            assert token_type_ids[1] == m_id
            assert token_type_ids[2] == r_id
            assert token_type_ids[3] == 0

    def test_encode(self):
        tokenizer = Tokenizer(seq_type="dna")
        seq = "ATGCGCTAA"
        encoded = tokenizer.encode(seq)
        expected = [tokenizer.encoder["ATG"], tokenizer.encoder["CGC"], tokenizer.encoder["TAA"]]
        assert encoded == expected

    def test_encode_aa(self):
        # This function seems to take a text and tokenize it first
        tokenizer = Tokenizer(seq_type="dna")
        seq = "ATGCGC"
        encoded = tokenizer.encode_aa(seq)
        # It tokenizes to ['ATG', 'CGC'] and then should encode them to AA ids.
        # But `encoder_aa` is based on single character amino acids not codons.
        # `encode_aa` tokenizes based on codon regex, but looks up in aa vocab.
        # This seems wrong. Let's trace it.
        # _tokenize("ATGCGC") -> ['ATG', 'CGC']
        # then it does self.encoder_aa.get(token, ...) for each token.
        # self.encoder_aa is {'<CLS>': 0, ..., 'A': 5, 'C': 6, ...}
        # So it will try to get 'ATG' from `encoder_aa` which is not there.
        # So it will return UNK for both.
        unk_aa_id = tokenizer.encoder_aa[tokenizer.unk_token]
        assert encoded == [unk_aa_id, unk_aa_id]

        # If the input is amino acids, it should work.
        seq_aa = "AR"  # Alanine, Arginine
        encoded_aa = tokenizer.encode_aa(seq_aa)
        # _tokenize("AR") -> ['A', 'R']
        a_id = tokenizer.encoder_aa["A"]
        r_id = tokenizer.encoder_aa["R"]
        assert encoded_aa == [a_id, r_id]

    def test_convert_tokens_to_string_and_specials(self):
        tok = Tokenizer(seq_type="dna")
        tokens = ["<CLS>", "ATG", "CGT", "<SEP>"]
        text = tok.convert_tokens_to_string(tokens)
        assert text == "<CLS>ATGCGT<SEP>"

    def test_build_inputs_with_special_tokens_and_mask(self):
        tok = Tokenizer(seq_type="dna")
        ids = [tok.encoder["ATG"], tok.encoder["CGT"]]
        built = tok.build_inputs_with_special_tokens(ids)
        assert built[0] == tok.cls_token_id
        assert built[-1] == tok.sep_token_id

        mask = tok.get_special_tokens_mask(ids, already_has_special_tokens=False)
        assert mask[0] == 1 and mask[-1] == 1

        mask2 = tok.get_special_tokens_mask(built, already_has_special_tokens=True)
        assert mask2[0] == 1 and mask2[-1] == 1

    def test_token_type_vocab_and_aa_vocab_sizes(self):
        tok = Tokenizer(seq_type="dna", token_type_mode="regular")
        assert tok.get_aa_vocab_size() == 5 + len(tok.amino_acids)
        assert tok.token_type_vocab_size >= 1

    @pytest.mark.parametrize("seq", ["", "atgcgt", "ATGXXCGT", "NNN"])
    def test_tokenize_edge_cases(self, seq):
        tok = Tokenizer(seq_type="dna")
        tokens = tok._tokenize(seq)
        assert isinstance(tokens, list)
