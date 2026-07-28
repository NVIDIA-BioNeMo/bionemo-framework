# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Selected-SFT temperature and nucleotide-prefix calibration utilities."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from bionemo.evo2_phage_gen.generation import PHIX174_REFERENCE_START


def _format_temperature(value: float) -> str:
    text = f"{float(value):g}"
    return text if "." in text else f"{text}.0"


@dataclass(frozen=True)
class SweepCell:
    """One comparable temperature and nucleotide-prefix condition."""

    prefix_length: int
    temperature: float

    @property
    def key(self) -> str:
        return f"prefix{self.prefix_length}_temp{_format_temperature(self.temperature)}"


def build_sweep_cells(prefix_lengths: Sequence[int], temperatures: Sequence[float]) -> list[SweepCell]:
    """Return the deterministic temperature-major calibration grid."""
    if not prefix_lengths or not temperatures:
        raise ValueError("prefix_lengths and temperatures must be non-empty")
    if any(length < 0 for length in prefix_lengths):
        raise ValueError("prefix lengths must be non-negative")
    if any(temperature <= 0 for temperature in temperatures):
        raise ValueError("temperatures must be positive")
    return [
        SweepCell(prefix_length=int(prefix_length), temperature=float(temperature))
        for temperature in temperatures
        for prefix_length in prefix_lengths
    ]


def partition_gpu_groups(gpu_ids: Sequence[int], tensor_parallel_size: int) -> list[tuple[int, ...]]:
    """Partition all GPUs into independent fixed-size model-parallel replicas."""
    if tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be positive")
    if not gpu_ids or len(gpu_ids) % tensor_parallel_size:
        raise ValueError("GPU count must be non-zero and divisible by tensor_parallel_size")
    return [
        tuple(int(gpu) for gpu in gpu_ids[start : start + tensor_parallel_size])
        for start in range(0, len(gpu_ids), tensor_parallel_size)
    ]


def write_cell_prompts(
    path: Path,
    *,
    cell: SweepCell,
    reference_start: str,
    marker: str,
    num_prompts: int,
) -> Path:
    """Write one repeated prompt bank with stable cell-scoped IDs."""
    reference_start = reference_start.strip().upper()
    if num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if cell.prefix_length > len(reference_start):
        raise ValueError("prefix length exceeds reference_start")
    prompt = f"{marker}{reference_start[: cell.prefix_length]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({"id": f"{cell.key}_{index:04d}", "prompt": prompt}) + "\n"
            for index in range(num_prompts)
        )
    )
    return path


def build_inference_command(
    *,
    infer_script: Path,
    checkpoint: Path,
    prompt_file: Path,
    output_file: Path,
    cell: SweepCell,
    target_length: int,
    seed: int,
    tensor_parallel_size: int,
    master_port: int,
    prompt_batch_size: int,
    max_seq_length: int,
) -> list[str]:
    """Build one strict target-total-length Evo2 inference command."""
    max_new_tokens = target_length - cell.prefix_length
    if max_new_tokens <= 0:
        raise ValueError("target_length must exceed prefix length")
    return [
        "torchrun",
        "--nproc_per_node",
        str(tensor_parallel_size),
        "--nnodes",
        "1",
        "--master_port",
        str(master_port),
        str(infer_script),
        "--ckpt-dir",
        str(checkpoint),
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        str(max_new_tokens),
        "--temperature",
        _format_temperature(cell.temperature),
        "--top-k",
        "4",
        "--top-p",
        "1.0",
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--max-seq-length",
        str(max_seq_length),
        "--prompt-batch-size",
        str(prompt_batch_size),
        "--strict-generation",
        "--stream-output",
        "--output-file",
        str(output_file),
    ]


def validate_cell_output(output_file: Path, prompt_file: Path, *, expected_records: int) -> int:
    """Require an exact, unique prompt-to-generation mapping for one cell."""
    expected = [json.loads(line) for line in prompt_file.read_text().splitlines() if line.strip()]
    actual = [json.loads(line) for line in output_file.read_text().splitlines() if line.strip()]
    if len(expected) != expected_records or len(actual) != expected_records:
        raise ValueError(f"record count differs from expected {expected_records}")
    expected_by_id = {str(record["id"]): str(record["prompt"]) for record in expected}
    actual_by_id = {str(record["id"]): record for record in actual}
    if len(expected_by_id) != expected_records or len(actual_by_id) != expected_records:
        raise ValueError("prompt or output IDs are not unique")
    if set(actual_by_id) != set(expected_by_id):
        raise ValueError("output IDs differ from prompt IDs")
    for record_id, prompt in expected_by_id.items():
        record = actual_by_id[record_id]
        if str(record.get("prompt")) != prompt:
            raise ValueError(f"output prompt differs for {record_id}")
        if not str(record.get("completion", "")):
            raise ValueError(f"empty completion for {record_id}")
    return len(actual)


def materialize_sweep(
    *,
    run_root: Path,
    checkpoint: Path,
    prefix_lengths: Sequence[int],
    temperatures: Sequence[float],
    num_prompts: int,
    reference_start: str,
    marker: str,
    gpu_ids: Sequence[int],
    tensor_parallel_size: int,
    target_length: int,
    top_k: int,
    top_p: float,
    seed: int,
    prompt_batch_size: int,
    max_seq_length: int,
) -> dict:
    """Materialize an immutable sweep contract and all cell prompt banks."""
    cells = build_sweep_cells(prefix_lengths, temperatures)
    groups = partition_gpu_groups(gpu_ids, tensor_parallel_size)
    prompts_dir = run_root / "prompts"
    jsonl_dir = run_root / "jsonl"
    logs_dir = run_root / "logs"
    runtime_dir = run_root / "runtime"
    for path in (prompts_dir, jsonl_dir, logs_dir, runtime_dir):
        path.mkdir(parents=True, exist_ok=True)

    cell_rows = []
    for index, cell in enumerate(cells):
        prompt_file = prompts_dir / f"{cell.key}_{num_prompts}.jsonl"
        output_file = jsonl_dir / f"{cell.key}.jsonl"
        write_cell_prompts(
            prompt_file,
            cell=cell,
            reference_start=reference_start,
            marker=marker,
            num_prompts=num_prompts,
        )
        cell_rows.append(
            {
                "index": index,
                "key": cell.key,
                "prefix_length": cell.prefix_length,
                "temperature": _format_temperature(cell.temperature),
                "prompt_file": str(prompt_file.resolve()),
                "output_file": str(output_file.resolve()),
            }
        )

    cells_path = run_root / "cells.tsv"
    with cells_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=tuple(cell_rows[0]),
        )
        writer.writeheader()
        writer.writerows(cell_rows)

    contract = {
        "schema_version": 1,
        "state": "planned",
        "checkpoint": str(checkpoint.resolve()),
        "reference_start": reference_start,
        "marker": marker,
        "prefix_lengths": [int(value) for value in prefix_lengths],
        "temperatures": [float(value) for value in temperatures],
        "num_prompts_per_cell": int(num_prompts),
        "target_length": int(target_length),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "seed": int(seed),
        "prompt_batch_size": int(prompt_batch_size),
        "max_seq_length": int(max_seq_length),
        "topology": {
            "gpu_ids": [int(gpu) for gpu in gpu_ids],
            "tensor_parallel_size": int(tensor_parallel_size),
            "replicas": len(groups),
        },
        "cells": [cell.key for cell in cells],
        "cells_tsv": str(cells_path.resolve()),
    }
    contract_path = run_root / "sweep_contract.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text())
        if existing != contract:
            raise ValueError(f"existing sweep contract differs: {contract_path}")
    else:
        contract_path.write_text(json.dumps(contract, indent=2) + "\n")
    return contract


def validate_sweep(run_root: Path) -> dict:
    """Validate every contracted cell and return exact completion counts."""
    contract = json.loads((run_root / "sweep_contract.json").read_text())
    rows = list(csv.DictReader((run_root / "cells.tsv").open(), delimiter="\t"))
    expected = int(contract["num_prompts_per_cell"])
    counts = {}
    for row in rows:
        counts[row["key"]] = validate_cell_output(
            Path(row["output_file"]),
            Path(row["prompt_file"]),
            expected_records=expected,
        )
    return {"cells": len(rows), "records": sum(counts.values()), "counts": counts}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Selected-SFT phage sampling calibration")
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--run-root", type=Path, required=True)
    materialize.add_argument("--checkpoint", type=Path, required=True)
    materialize.add_argument("--prefix-lengths", type=int, nargs="+", required=True)
    materialize.add_argument("--temperatures", type=float, nargs="+", required=True)
    materialize.add_argument("--num-prompts", type=int, required=True)
    materialize.add_argument("--reference-start", default=PHIX174_REFERENCE_START)
    materialize.add_argument("--marker", default="+~")
    materialize.add_argument("--gpu-ids", type=int, nargs="+", required=True)
    materialize.add_argument("--tensor-parallel-size", type=int, required=True)
    materialize.add_argument("--target-length", type=int, default=6000)
    materialize.add_argument("--top-k", type=int, default=4)
    materialize.add_argument("--top-p", type=float, default=1.0)
    materialize.add_argument("--seed", type=int, default=7)
    materialize.add_argument("--prompt-batch-size", type=int, default=16)
    materialize.add_argument("--max-seq-length", type=int, default=10240)

    validate = subparsers.add_parser("validate-cell")
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--prompts", type=Path, required=True)
    validate.add_argument("--expected-records", type=int, required=True)

    validate_all = subparsers.add_parser("validate-all")
    validate_all.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "materialize":
        contract = materialize_sweep(
            run_root=args.run_root,
            checkpoint=args.checkpoint,
            prefix_lengths=args.prefix_lengths,
            temperatures=args.temperatures,
            num_prompts=args.num_prompts,
            reference_start=args.reference_start,
            marker=args.marker,
            gpu_ids=args.gpu_ids,
            tensor_parallel_size=args.tensor_parallel_size,
            target_length=args.target_length,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
            prompt_batch_size=args.prompt_batch_size,
            max_seq_length=args.max_seq_length,
        )
        print(json.dumps(contract, sort_keys=True))
    elif args.command == "validate-cell":
        print(validate_cell_output(args.output, args.prompts, expected_records=args.expected_records))
    else:
        print(json.dumps(validate_sweep(args.run_root), sort_keys=True))


if __name__ == "__main__":
    main()
