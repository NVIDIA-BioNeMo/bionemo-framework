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

"""Download and prepare the Microviridae SFT data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import yaml


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ZENODO_DIR = RECIPE_ROOT / "data" / "external" / "zenodo"
DEFAULT_SFT_PROCESSED = DEFAULT_ZENODO_DIR / "microviridae_sft_training_data_processed.fna"
DEFAULT_SFT_RAW = DEFAULT_ZENODO_DIR / "microviridae_sft_training_data_raw.fna"
DEFAULT_SFT_PROCESSED_URL = (
    "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_processed.fna?download=1"
)
DEFAULT_SFT_RAW_URL = "https://zenodo.org/records/17101843/files/microviridae_sft_training_data_raw.fna?download=1"
CONDITIONING_PREFIXES = ("+!", "+#", "+$", "+^", "+~")


@dataclass(frozen=True)
class FastaRecord:
    """Represent one source FASTA record."""

    record_id: str
    header: str
    conditioned_sequence: str
    biological_sequence: str


class Components:
    """Track connected components while clustering related genomes."""

    def __init__(self, values: Iterable[str]) -> None:
        """Initialize one component per record."""
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        """Return the representative for a record."""
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        """Join the components containing two records."""
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _recipe_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(RECIPE_ROOT).as_posix()
    except ValueError:
        return str(path)


def _download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".download", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        urllib.request.urlretrieve(url, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


def prepare_sft_data(*, include_raw: bool = False, overwrite: bool = False) -> list[Path]:
    """Download the published Microviridae SFT data."""
    paths = [_download(DEFAULT_SFT_PROCESSED_URL, DEFAULT_SFT_PROCESSED, overwrite=overwrite)]
    if include_raw:
        paths.append(_download(DEFAULT_SFT_RAW_URL, DEFAULT_SFT_RAW, overwrite=overwrite))
    return paths


def _read_fasta(path: Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def finish() -> None:
        if header is None:
            return
        conditioned = "".join(chunks).replace(" ", "").upper()
        prefix = conditioned[:2] if conditioned[:2] in CONDITIONING_PREFIXES else ""
        biological = conditioned[len(prefix) :]
        if not biological or set(biological) - set("ACGTN"):
            raise ValueError(f"{path}: {header.split()[0]} is empty or has unsupported sequence characters")
        records.append(
            FastaRecord(
                record_id=f"record-{len(records) + 1:06d}",
                header=header,
                conditioned_sequence=conditioned,
                biological_sequence=biological,
            )
        )

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header = line[1:].strip()
                chunks = []
            elif header is None:
                raise ValueError(f"{path}: sequence before first FASTA header")
            else:
                chunks.append(line)
    finish()
    if not records:
        raise ValueError(f"{path}: no FASTA records")
    return records


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _minimum_rotation(sequence: str) -> str:
    """Return the lexicographically smallest circular rotation in linear time."""
    if not sequence:
        return sequence
    doubled = sequence + sequence
    left, right, offset = 0, 1, 0
    length = len(sequence)
    while left < length and right < length and offset < length:
        a = doubled[left + offset]
        b = doubled[right + offset]
        if a == b:
            offset += 1
            continue
        if a > b:
            left = left + offset + 1
            if left == right:
                left += 1
        else:
            right = right + offset + 1
            if left == right:
                right += 1
        offset = 0
    start = min(left, right)
    return doubled[start : start + length]


def _circular_key(sequence: str) -> str:
    return min(_minimum_rotation(sequence), _minimum_rotation(_reverse_complement(sequence)))


def _write_fasta(path: Path, records: Sequence[FastaRecord], *, biological: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            sequence = record.biological_sequence if biological else record.conditioned_sequence
            header = record.record_id if biological else record.header
            handle.write(f">{header}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def _mmseqs_cluster(
    records: Sequence[FastaRecord],
    work_dir: Path,
    *,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
) -> Path:
    input_fasta = work_dir / "unconditioned.fna"
    _write_fasta(input_fasta, records, biological=True)
    prefix = work_dir / "clusters"
    command = [
        mmseqs_bin,
        "easy-cluster",
        str(input_fasta),
        str(prefix),
        str(work_dir / "tmp-cluster"),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--cluster-mode",
        "0",
        "--threads",
        str(threads),
    ]
    result = _run(command)
    if result.returncode:
        raise RuntimeError(f"MMseqs clustering failed:\n{result.stdout}")
    cluster_tsv = prefix.with_name(prefix.name + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise RuntimeError(f"MMseqs did not write {cluster_tsv}")
    return cluster_tsv


def _load_components(records: Sequence[FastaRecord], cluster_tsv: Path) -> Components:
    record_ids = {record.record_id for record in records}
    components = Components(record_ids)
    for line in cluster_tsv.read_text().splitlines():
        if not line.strip():
            continue
        representative, member, *_ = line.split("\t")
        if representative not in record_ids or member not in record_ids:
            raise ValueError("MMseqs cluster output contains an unknown record")
        components.union(representative, member)

    exact_groups: dict[str, str] = {}
    for record in records:
        key = _circular_key(record.biological_sequence)
        previous = exact_groups.setdefault(key, record.record_id)
        components.union(previous, record.record_id)
    return components


def _groups(records: Sequence[FastaRecord], components: Components) -> dict[str, list[FastaRecord]]:
    result: dict[str, list[FastaRecord]] = {}
    for record in records:
        result.setdefault(components.find(record.record_id), []).append(record)
    return result


def _take_groups(
    candidates: list[tuple[str, list[FastaRecord]]],
    target_count: int,
) -> tuple[list[tuple[str, list[FastaRecord]]], list[tuple[str, list[FastaRecord]]]]:
    selected: list[tuple[str, list[FastaRecord]]] = []
    remaining = target_count
    rest: list[tuple[str, list[FastaRecord]]] = []
    for item in candidates:
        size = len(item[1])
        if size <= remaining:
            selected.append(item)
            remaining -= size
        else:
            rest.append(item)
    if remaining:
        raise ValueError(f"cannot assign whole clusters to reach {target_count} records")
    selected_ids = {item[0] for item in selected}
    rest.extend(item for item in candidates if item[0] not in selected_ids and item not in rest)
    return selected, rest


def _audit_pair(
    query: Path,
    target: Path,
    output: Path,
    *,
    mmseqs_bin: str,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
) -> None:
    command = [
        mmseqs_bin,
        "easy-search",
        str(query),
        str(target),
        str(output),
        str(output.parent / f"tmp-{output.stem}"),
        "--min-seq-id",
        str(min_seq_id),
        "-c",
        str(coverage),
        "--cov-mode",
        str(cov_mode),
        "--search-type",
        "3",
        "--threads",
        str(threads),
        "--format-output",
        "query,target,pident,qcov,tcov",
    ]
    result = _run(command)
    if result.returncode:
        raise RuntimeError(f"MMseqs leakage search failed:\n{result.stdout}")
    if output.is_file() and output.read_text().strip():
        raise RuntimeError(f"near-duplicate leakage found; inspect {output}")


def _preprocess_config(output_dir: Path, *, tokenizer: Path, seed: int, workers: int) -> list[dict[str, object]]:
    preprocessed = output_dir / "preprocessed"
    entries: list[dict[str, object]] = []
    for split, fractions in (
        ("train", (1.0, 0.0, 0.0)),
        ("validation", (0.0, 1.0, 0.0)),
        ("test", (0.0, 0.0, 1.0)),
    ):
        train, validation, test = fractions
        entries.append(
            {
                "datapaths": [str((output_dir / "splits" / f"{split}.fna").resolve())],
                "output_dir": str(preprocessed.resolve()),
                "output_prefix": f"clusterheldout_{split}",
                "train_split": train,
                "valid_split": validation,
                "test_split": test,
                "overwrite": False,
                "embed_reverse_complement": False,
                "random_reverse_complement": 0.0,
                "force_uppercase": False,
                "indexed_dataset_dtype": "uint8",
                "append_eod": True,
                "hf_tokenizer_model_path": str(tokenizer.resolve()),
                "workers": workers,
                "preproc_concurrency": 1000,
                "chunksize": 1,
                "drop_empty_sequences": True,
                "nnn_filter": False,
                "seed": seed,
            }
        )
    return entries


def _dataset_config(output_dir: Path, *, heldout_test: bool = False) -> list[dict[str, object]]:
    prefix = output_dir / "preprocessed"
    tokenizer_suffix = "nucleotide_fast_tokenizer_512"
    validation_prefix = prefix / f"clusterheldout_validation_{tokenizer_suffix}_val"
    test_prefix = prefix / f"clusterheldout_test_{tokenizer_suffix}_test"
    return [
        {
            "dataset_prefix": str((prefix / f"clusterheldout_train_{tokenizer_suffix}_train").resolve()),
            "dataset_weight": 1.0,
            "dataset_split": "train",
        },
        {
            "dataset_prefix": str((test_prefix if heldout_test else validation_prefix).resolve()),
            "dataset_weight": 1.0,
            "dataset_split": "validation",
        },
        {
            # The training config uses validation here because train_evo2 eagerly
            # constructs a test iterator. Use heldout_dataset.yaml only after
            # checkpoint selection for a zero-update held-out evaluation.
            "dataset_prefix": str((test_prefix if heldout_test else validation_prefix).resolve()),
            "dataset_weight": 1.0,
            "dataset_split": "test",
        },
    ]


def prepare_cluster_split(
    source_fasta: Path,
    output_dir: Path,
    *,
    mmseqs_bin: str,
    validation_count: int,
    test_count: int,
    seed: int,
    min_seq_id: float,
    coverage: float,
    cov_mode: int,
    threads: int,
    tokenizer: Path,
    workers: int,
    overwrite: bool = False,
) -> dict[str, object]:
    """Build leakage-controlled train, validation, and test splits."""
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{output_dir} is not empty; choose another output or pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_fasta(source_fasta)
    if len(records) <= validation_count + test_count:
        raise ValueError("the source corpus is too small for the requested held-out sets")

    work_dir = output_dir / "work"
    work_dir.mkdir()
    cluster_tsv = _mmseqs_cluster(
        records,
        work_dir,
        mmseqs_bin=mmseqs_bin,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
    )
    components = _load_components(records, cluster_tsv)
    grouped = sorted(_groups(records, components).items())
    random.Random(seed).shuffle(grouped)
    test_groups, remainder = _take_groups(grouped, test_count)
    validation_groups, train_groups = _take_groups(remainder, validation_count)

    split_groups = {"train": train_groups, "validation": validation_groups, "test": test_groups}
    split_records: dict[str, list[FastaRecord]] = {}
    for split, groups_for_split in split_groups.items():
        values = sorted(
            (record for _, group in groups_for_split for record in group),
            key=lambda record: record.record_id,
        )
        split_records[split] = values
        _write_fasta(output_dir / "splits" / f"{split}.fna", values)
        _write_fasta(work_dir / f"{split}.biological.fna", values, biological=True)

    audit_dir = output_dir / "leakage"
    audit_dir.mkdir()
    _audit_pair(
        work_dir / "validation.biological.fna",
        work_dir / "train.biological.fna",
        audit_dir / "validation-vs-train.tsv",
        mmseqs_bin=mmseqs_bin,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
    )
    _audit_pair(
        work_dir / "test.biological.fna",
        work_dir / "train.biological.fna",
        audit_dir / "test-vs-train.tsv",
        mmseqs_bin=mmseqs_bin,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
    )
    _audit_pair(
        work_dir / "test.biological.fna",
        work_dir / "validation.biological.fna",
        audit_dir / "test-vs-validation.tsv",
        mmseqs_bin=mmseqs_bin,
        min_seq_id=min_seq_id,
        coverage=coverage,
        cov_mode=cov_mode,
        threads=threads,
    )

    preprocess = _preprocess_config(output_dir, tokenizer=tokenizer, seed=seed, workers=workers)
    (output_dir / "preprocess.yaml").write_text(yaml.safe_dump(preprocess, sort_keys=False))
    dataset = _dataset_config(output_dir)
    (output_dir / "training_dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False))
    heldout_dataset = _dataset_config(output_dir, heldout_test=True)
    (output_dir / "heldout_dataset.yaml").write_text(yaml.safe_dump(heldout_dataset, sort_keys=False))

    version_result = _run([mmseqs_bin, "version"])
    summary = {
        "source_fasta": str(source_fasta.resolve()),
        "records": len(records),
        "components": len(grouped),
        "split_counts": {name: len(values) for name, values in split_records.items()},
        "split_components": {name: len(values) for name, values in split_groups.items()},
        "seed": seed,
        "mmseqs": {
            "executable": mmseqs_bin,
            "version": version_result.stdout.strip() if version_result.returncode == 0 else "unknown",
            "min_seq_id": min_seq_id,
            "coverage": coverage,
            "cov_mode": cov_mode,
        },
        "leakage_matches": 0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    timestamp = datetime.now(timezone.utc).isoformat()
    (output_dir / "RUNLOG.md").write_text(
        "# SFT split run log\n\n"
        f"- {timestamp}: prepared cluster-held-out split from {source_fasta.resolve()}.\n"
        f"- Counts: {summary['split_counts']}; components: {summary['split_components']}.\n"
        f"- MMseqs state: {summary['mmseqs']}; independent boundary matches: 0.\n"
    )
    return summary


def split_main(argv: Sequence[str] | None = None) -> None:
    """Prepare an SFT split from the command line."""
    parser = argparse.ArgumentParser(description="Prepare a cluster-held-out Microviridae SFT split")
    parser.add_argument("--source-fasta", type=Path, default=DEFAULT_SFT_PROCESSED)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mmseqs-bin", default="mmseqs")
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-seq-id", type=float, default=0.98)
    parser.add_argument("--coverage", type=float, default=0.8)
    parser.add_argument("--cov-mode", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--tokenizer", type=Path, default=RECIPE_ROOT / "tokenizers/nucleotide_fast_tokenizer_512")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    summary = prepare_cluster_split(
        args.source_fasta,
        args.output_dir,
        mmseqs_bin=args.mmseqs_bin,
        validation_count=args.validation_count,
        test_count=args.test_count,
        seed=args.seed,
        min_seq_id=args.min_seq_id,
        coverage=args.coverage,
        cov_mode=args.cov_mode,
        threads=args.threads,
        tokenizer=args.tokenizer,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> None:
    """Download the published SFT data from the command line."""
    parser = argparse.ArgumentParser(description="Download Zenodo Microviridae SFT FASTA files")
    parser.add_argument("--include-raw", action="store_true", help="Also download the raw SFT FASTA")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    for path in prepare_sft_data(include_raw=args.include_raw, overwrite=args.overwrite):
        print(_recipe_relative(path))


if __name__ == "__main__":
    main()
