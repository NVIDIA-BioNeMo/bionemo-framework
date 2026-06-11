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

"""CPU unit tests for the shared FASTA reader (no torch / no GPU)."""

import gzip

from evo2_sae.fasta import read_fasta


def test_basic_multiline_and_header_token(tmp_path):
    """Header keeps only its first token; sequence lines are concatenated."""
    fa = tmp_path / "x.fa"
    fa.write_text(">chr1 some description\nACGT\nACGT\n>chr2\nTTTT\n")
    assert list(read_fasta(fa)) == [("chr1", "ACGTACGT"), ("chr2", "TTTT")]


def test_tokenless_header_gets_generated_id(tmp_path):
    """A bare ``>`` / ``"> "`` header must not IndexError — it gets a ``seq_<n>`` id."""
    fa = tmp_path / "x.fa"
    fa.write_text(">good\nAAAA\n> \nCCCC\n>\nGGGG\n")
    assert list(read_fasta(fa)) == [("good", "AAAA"), ("seq_1", "CCCC"), ("seq_2", "GGGG")]


def test_gzip_transparent(tmp_path):
    """A ``.gz`` path is decompressed transparently."""
    fa = tmp_path / "x.fa.gz"
    with gzip.open(fa, "wt") as f:
        f.write(">a\nACGT\n")
    assert list(read_fasta(fa)) == [("a", "ACGT")]


def test_empty_file(tmp_path):
    """An empty file yields nothing (no trailing phantom record)."""
    fa = tmp_path / "empty.fa"
    fa.write_text("")
    assert list(read_fasta(fa)) == []
