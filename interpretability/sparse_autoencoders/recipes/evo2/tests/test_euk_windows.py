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

"""CPU tests for euk_windows: GFF parsing, 1-based->0-based labeling, single-chromosome guard."""

import pytest
from evo2_sae.eval.probing.euk_windows import _label_window, build_windows, parse_gff, representative_tx


_GFF = (
    "##gff-version 3\n"
    "1\tt\tgene\t11\t40\t.\t+\t.\tID=gene:g1;biotype=protein_coding\n"
    "1\tt\tmRNA\t11\t40\t.\t+\t.\tID=transcript:t1;Parent=gene:g1;biotype=protein_coding\n"
    "1\tt\texon\t11\t20\t.\t+\t.\tParent=transcript:t1\n"
    "1\tt\texon\t31\t40\t.\t+\t.\tParent=transcript:t1\n"
    "1\tt\tCDS\t11\t20\t.\t+\t.\tParent=transcript:t1\n"
    # a non-coding gene that must be filtered out
    "1\tt\tgene\t100\t120\t.\t+\t.\tID=gene:g2;biotype=lncRNA\n"
    "1\tt\tmRNA\t100\t120\t.\t+\t.\tID=transcript:t2;Parent=gene:g2;biotype=lncRNA\n"
    "1\tt\texon\t100\t120\t.\t+\t.\tParent=transcript:t2\n"
)


def test_parse_gff_keeps_only_protein_coding(tmp_path):
    """parse_gff returns only protein_coding genes, with sorted exon/CDS coords (1-based, inclusive)."""
    gff = tmp_path / "g.gff3"
    gff.write_text(_GFF)
    genes = parse_gff(str(gff))
    assert set(genes) == {"g1"}  # the lncRNA gene is filtered out
    _, tx = representative_tx(genes["g1"])
    assert tx["exon"] == [(11, 20), (31, 40)] and tx["cds"] == [(11, 20)]


def test_label_window_converts_1based_inclusive_to_0based_halfopen():
    """A GFF interval s..e (1-based inclusive) labels 0-based positions s-1..e-1; intergenic/UTR
    fall out correctly."""
    chrom = "A" * 50
    gm = {
        "exons": [(11, 20)],
        "introns": [(21, 25)],
        "cds": [(11, 15)],
        "gi": 0,
        "exon_ids": [0],
        "intron_ids": [0],
        "span": (11, 25),
    }
    lab = _label_window(chrom, 0, 50, gm, 50)["labels"]
    assert list(lab["exon"].nonzero()[0]) == list(range(10, 20))  # 11..20 -> 10..19
    assert list(lab["intron"].nonzero()[0]) == list(range(20, 25))  # 21..25 -> 20..24
    assert list(lab["cds"].nonzero()[0]) == list(range(10, 15))  # 11..15 -> 10..14
    assert list(lab["utr"].nonzero()[0]) == list(range(15, 20))  # exon minus cds
    assert lab["intergenic"][:10].all() and lab["intergenic"][25:].all()  # outside the gene span
    assert not lab["intergenic"][10:25].any()


def test_build_windows_rejects_multi_record_fasta(tmp_path):
    """A multi-record FASTA is rejected (coords would be applied against a concatenated blob)."""
    fasta = tmp_path / "f.fa"
    fasta.write_text(">chr1\nACGTACGT\n>chr2\nACGTACGT\n")
    gff = tmp_path / "g.gff3"
    gff.write_text("##gff-version 3\n")
    with pytest.raises(ValueError, match="single-chromosome"):
        build_windows(str(fasta), str(gff))
