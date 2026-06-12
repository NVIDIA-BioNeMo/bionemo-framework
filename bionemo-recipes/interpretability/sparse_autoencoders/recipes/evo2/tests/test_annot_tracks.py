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

"""CPU unit tests for the generic interval-track loader (no model / no torch-CUDA)."""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from annot_tracks import label_windows, load_track, read_fasta_dict


def test_read_fasta_dict_uses_first_token(tmp_path):
    """seq_id is the first header token (so it matches BED/GFF chrom)."""
    fa = tmp_path / "g.fa"
    fa.write_text(">chr1 Homo sapiens\nACGT\nACGT\n>chr2\nTTTT\n")
    assert read_fasta_dict(fa) == {"chr1": "ACGTACGT", "chr2": "TTTT"}


def test_load_bed_is_half_open(tmp_path):
    """BED is 0-based half-open and used as-is."""
    bed = tmp_path / "t.bed"
    bed.write_text("chr1\t2\t5\tsiteA\nchr1\t10\t12\tsiteB\n")
    assert load_track(str(bed)) == {"chr1": [(2, 5), (10, 12)]}


def test_load_gff_converts_to_half_open_and_filters_type(tmp_path):
    """GFF 1-based inclusive -> 0-based half-open; feature_type filters column 3."""
    gff = tmp_path / "t.gff3"
    gff.write_text(
        "# comment\n"
        "chr1\tsrc\texon\t3\t5\t.\t+\t.\tID=e1\n"  # 1-based [3,5] -> [2,5)
        "chr1\tsrc\tCDS\t3\t5\t.\t+\t.\tID=c1\n"
    )
    assert load_track(str(gff), feature_type="exon") == {"chr1": [(2, 5)]}


def test_label_windows_mask_and_instance_ids(tmp_path):
    """Each interval is one instance; mask + instance id line up with the window positions."""
    seqs = {"chr1": "ACGT" * 25}  # 100 bp (above the 60 bp window floor)
    tracks = {"site": {"chr1": [(2, 5), (10, 12)]}}  # two instances
    windows, stats = label_windows(seqs, tracks, seq_len=100)
    assert len(windows) == 1
    w = windows[0]
    mask, inst = w["labels"]["site"], w["instances"]["site"]
    assert list(mask.nonzero()[0]) == [2, 3, 4, 10, 11]
    assert set(inst[mask].tolist()) == {0, 1}  # two distinct instances
    assert (inst[~mask] == -1).all()
    assert stats["n_inst"]["site"] == 2


def test_instance_id_stable_across_split_windows():
    """An interval spanning a window boundary keeps ONE global instance id (recall counts it once)."""
    seqs = {"chr1": "A" * 200}
    tracks = {"big": {"chr1": [(90, 110)]}}  # straddles the 0-100 / 100-200 boundary
    windows, stats = label_windows(seqs, tracks, seq_len=100)
    ids = set()
    for w in windows:
        inst = w["instances"]["big"]
        ids.update(int(x) for x in inst[inst >= 0])
    assert ids == {0}  # same id in both windows
    assert stats["n_inst"]["big"] == 1
