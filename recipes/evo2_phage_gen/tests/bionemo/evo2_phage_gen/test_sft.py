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

"""Tests for Microviridae SFT download and split preparation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen import sft


def _fasta(path: Path, sequences: list[str]) -> None:
    path.write_text("".join(f">seq-{index}\n+${sequence}\n" for index, sequence in enumerate(sequences, start=1)))


def test_download_uses_existing_file(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "source.fna"
    output.write_text(">one\nACGT\n")
    monkeypatch.setattr(
        sft.urllib.request,
        "urlretrieve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )
    assert sft._download("https://example.test/source.fna", output) == output


def test_circular_key_groups_equivalents() -> None:
    sequence = "AACCGT"
    assert sft._circular_key(sequence) == sft._circular_key(sequence[2:] + sequence[:2])
    assert sft._circular_key(sequence) == sft._circular_key(sft._reverse_complement(sequence))


def test_split_writes_reusable_configs(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.fna"
    sequences = [
        "AACCGT",
        "CCGTAA",
        "AAAACG",
        "AAAAGC",
        "AAACGC",
        "AAAGCC",
        "AACGCC",
        "AAGCCC",
        "ACGCCC",
        "AGCCCC",
    ]
    _fasta(source, sequences)
    cluster_tsv = tmp_path / "clusters.tsv"
    cluster_tsv.write_text("".join(f"record-{index:06d}\trecord-{index:06d}\n" for index in range(1, 11)))
    monkeypatch.setattr(sft, "_mmseqs_cluster", lambda *args, **kwargs: cluster_tsv)
    audits: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        sft,
        "_audit_pair",
        lambda query, target, output, **kwargs: audits.append((query, target)) or output.write_text(""),
    )
    monkeypatch.setattr(
        sft,
        "_run",
        lambda command: subprocess.CompletedProcess(command, 0, stdout="MMseqs Version: test"),
    )

    output = tmp_path / "split"
    summary = sft.prepare_cluster_split(
        source,
        output,
        mmseqs_bin="mmseqs",
        validation_count=2,
        test_count=2,
        seed=1234,
        min_seq_id=0.98,
        coverage=0.8,
        cov_mode=0,
        threads=2,
        tokenizer=Path("tokenizers/nucleotide_fast_tokenizer_512"),
        workers=1,
    )

    assert summary["split_counts"] == {"train": 6, "validation": 2, "test": 2}
    assert len(audits) == 3
    preprocess = yaml.safe_load((output / "preprocess.yaml").read_text())
    assert [entry["output_prefix"] for entry in preprocess] == [
        "clusterheldout_train",
        "clusterheldout_validation",
        "clusterheldout_test",
    ]
    dataset = yaml.safe_load((output / "training_dataset.yaml").read_text())
    assert dataset[1]["dataset_prefix"] == dataset[2]["dataset_prefix"]
    heldout_dataset = yaml.safe_load((output / "heldout_dataset.yaml").read_text())
    assert heldout_dataset[1]["dataset_prefix"] == heldout_dataset[2]["dataset_prefix"]
    assert "clusterheldout_test_" in heldout_dataset[2]["dataset_prefix"]
    for query, target in audits:
        assert query.name.endswith(".biological.fna")
        assert target.name.endswith(".biological.fna")
        assert "+$" not in query.read_text()
        assert "+$" not in target.read_text()
    split_text = {name: (output / "splits" / f"{name}.fna").read_text() for name in ("train", "validation", "test")}
    locations = [name for name, text in split_text.items() if ">seq-1\n" in text or ">seq-2\n" in text]
    assert len(locations) == 1
    assert ">seq-1\n" in split_text[locations[0]]
    assert ">seq-2\n" in split_text[locations[0]]


def test_split_rejects_unknown_sequence_symbols(tmp_path: Path) -> None:
    source = tmp_path / "source.fna"
    source.write_text(">bad\nACGTX\n")
    try:
        sft._read_fasta(source)
    except ValueError as error:
        assert "unsupported sequence characters" in str(error)
    else:
        raise AssertionError("invalid FASTA was accepted")
