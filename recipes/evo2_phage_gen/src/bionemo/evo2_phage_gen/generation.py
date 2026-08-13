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

"""Utilities for paper-style Evo2 Microviridae generation replication."""

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from bionemo.evo2_phage_gen.qc import trim_at_first_eos


PHIX174_REFERENCE_START = "GAGTTTTATCGCTTCCATGACGCAGAAGTTAACACTTTCGGATATTTCTGATGAGTCGAA"
DEFAULT_PROMPT_LENGTHS = tuple(range(1, 12))
DEFAULT_PROMPT_PREFIX = "+~"
PAPER_USEFUL_RL_PROMPT_LENGTHS = (4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 10, 11)
PAPER_USEFUL_RL_VALIDATION_PROMPT_LENGTH = 10
PAPER_USEFUL_RL_VALIDATION_RECORDS = 96
DNA_ALPHABET = frozenset("ACGTacgt")
RECIPE_ROOT = Path(__file__).resolve().parents[3]


def phix174_prompts(
    reference_start: str = PHIX174_REFERENCE_START,
    prompt_lengths: Sequence[int] = DEFAULT_PROMPT_LENGTHS,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
) -> dict[int, str]:
    """Return paper-style PhiX174-start prompts keyed by nucleotide prompt length."""
    reference_start = reference_start.strip().upper()
    prompts: dict[int, str] = {}
    for prompt_len in prompt_lengths:
        if prompt_len < 0:
            raise ValueError(f"Prompt length must be non-negative, got {prompt_len}")
        if prompt_len > len(reference_start):
            raise ValueError(f"Prompt length {prompt_len} exceeds reference length {len(reference_start)}")
        prompts[int(prompt_len)] = f"{prompt_prefix}{reference_start[:prompt_len]}"
    return prompts


def _openai_prompt_record(prompt: str) -> dict[str, list[dict[str, str]]]:
    """Return the OpenAI-style prompt record consumed by the phage RL processor."""
    return {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": ""}]}


def write_openai_prompt_jsonl(path: Path, prompts: Sequence[str]) -> Path:
    """Write OpenAI-style user-message prompts to one JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for prompt in prompts:
            handle.write(json.dumps(_openai_prompt_record(prompt)) + "\n")
    return path


def ensure_paper_useful_rl_prompt_files(
    data_dir: Path | None = None,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Materialize deterministic paper-useful RL prompt JSONL files when absent."""
    data_dir = data_dir or RECIPE_ROOT / "data"
    prompts_by_length = phix174_prompts(prompt_lengths=sorted(set(PAPER_USEFUL_RL_PROMPT_LENGTHS)))
    train_path = data_dir / "phage_prompts_paper_useful_rl.jsonl"
    validation_path = data_dir / "phage_prompts_paper_useful_rl_validation_prompt10_96.jsonl"

    if overwrite or not train_path.exists():
        train_prompts = [prompts_by_length[prompt_len] for prompt_len in PAPER_USEFUL_RL_PROMPT_LENGTHS]
        write_openai_prompt_jsonl(train_path, train_prompts)
    if overwrite or not validation_path.exists():
        validation_prompt = prompts_by_length[PAPER_USEFUL_RL_VALIDATION_PROMPT_LENGTH]
        write_openai_prompt_jsonl(
            validation_path,
            [validation_prompt] * PAPER_USEFUL_RL_VALIDATION_RECORDS,
        )
    return {"train": train_path, "validation": validation_path}


def write_prompt_sweep_jsonl(
    output_dir: Path,
    *,
    reference_start: str = PHIX174_REFERENCE_START,
    prompt_lengths: Sequence[int] = DEFAULT_PROMPT_LENGTHS,
    prompt_prefix: str = DEFAULT_PROMPT_PREFIX,
    num_prompts: int = 1000,
    id_prefix: str = "phix174",
) -> list[Path]:
    """Write repeated prompt JSONL files for ``infer_evo2 --prompt-file``."""
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []
    for prompt_len, prompt in phix174_prompts(reference_start, prompt_lengths, prompt_prefix=prompt_prefix).items():
        output_path = output_dir / f"{id_prefix}_prompt{prompt_len}_{num_prompts}.jsonl"
        with output_path.open("w") as f:
            for idx in range(num_prompts):
                record = {"id": f"{id_prefix}_prompt{prompt_len}_{idx:04d}", "prompt": prompt}
                f.write(json.dumps(record) + "\n")
        written_paths.append(output_path)
    return written_paths


def _sequence_before_eos(sequence: str) -> str:
    """Return generated sequence content before textual EOS markers."""
    return trim_at_first_eos(str(sequence).replace("\n", "").strip())


def _prompt_nucleotides(prompt: str) -> str:
    """Keep only nucleotide bases from a prompt that may include SFT soft tokens."""
    return "".join(char for char in prompt if char in DNA_ALPHABET)


def _wrap_fasta_sequence(sequence: str, width: int = 80) -> str:
    """Wrap a FASTA sequence to a fixed line width."""
    return "\n".join(sequence[i : i + width] for i in range(0, len(sequence), width))


def infer_jsonl_to_fasta(
    jsonl_paths: Iterable[Path],
    output_fasta: Path,
    *,
    include_source_stem: bool = True,
) -> Path:
    """Convert ``infer_evo2`` JSONL records to FASTA by prepending prompt nucleotides to completion."""
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    with output_fasta.open("w") as fasta:
        for jsonl_path in sorted(Path(path) for path in jsonl_paths):
            with jsonl_path.open() as f:
                for line_idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    record_id = str(record.get("id") or f"{jsonl_path.stem}_{line_idx:06d}")
                    header = f"{record_id}|{jsonl_path.stem}" if include_source_stem else record_id
                    prompt = _prompt_nucleotides(_sequence_before_eos(record.get("prompt", "")))
                    completion = _sequence_before_eos(record.get("completion", ""))
                    sequence = f"{prompt}{completion}".upper()
                    fasta.write(f">{header}\n{_wrap_fasta_sequence(sequence)}\n")
    return output_fasta


def _resolve_jsonl_inputs(input_jsonl: Sequence[Path], input_dir: Path | None, glob_pattern: str) -> list[Path]:
    """Resolve explicit JSONL files plus optional directory glob inputs."""
    paths = [Path(path) for path in input_jsonl]
    if input_dir is not None:
        paths.extend(sorted(Path(input_dir).glob(glob_pattern)))
    if not paths:
        raise ValueError("Provide at least one --input-jsonl or --input-dir")
    return paths


def main() -> None:
    """CLI entry point for generation replication helpers."""
    parser = argparse.ArgumentParser(description="Evo2 phage generation replication helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    rl_prompt_parser = subparsers.add_parser(
        "prepare-rl-prompts",
        help="Materialize deterministic paper-useful RL prompt JSONL files",
    )
    rl_prompt_parser.add_argument("--data-dir", type=Path, default=RECIPE_ROOT / "data")
    rl_prompt_parser.add_argument("--overwrite", action="store_true")

    prompt_parser = subparsers.add_parser("write-prompts", help="Write PhiX174 prompt sweep JSONL files")
    prompt_parser.add_argument("--output-dir", type=Path, required=True)
    prompt_parser.add_argument("--reference-start", type=str, default=PHIX174_REFERENCE_START)
    prompt_parser.add_argument("--prompt-lengths", type=int, nargs="+", default=list(DEFAULT_PROMPT_LENGTHS))
    prompt_parser.add_argument("--prompt-prefix", type=str, default=DEFAULT_PROMPT_PREFIX)
    prompt_parser.add_argument("--num-prompts", type=int, default=1000)
    prompt_parser.add_argument("--id-prefix", type=str, default="phix174")

    fasta_parser = subparsers.add_parser("jsonl-to-fasta", help="Convert infer_evo2 JSONL outputs to FASTA")
    fasta_parser.add_argument("--input-jsonl", type=Path, nargs="*", default=[])
    fasta_parser.add_argument("--input-dir", type=Path, default=None)
    fasta_parser.add_argument("--glob", type=str, default="*.jsonl")
    fasta_parser.add_argument("--output-fasta", type=Path, required=True)
    fasta_parser.add_argument("--no-source-stem", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare-rl-prompts":
        for path in ensure_paper_useful_rl_prompt_files(args.data_dir, overwrite=args.overwrite).values():
            print(path)
    elif args.command == "write-prompts":
        for path in write_prompt_sweep_jsonl(
            args.output_dir,
            reference_start=args.reference_start,
            prompt_lengths=args.prompt_lengths,
            prompt_prefix=args.prompt_prefix,
            num_prompts=args.num_prompts,
            id_prefix=args.id_prefix,
        ):
            print(path)
    elif args.command == "jsonl-to-fasta":
        paths = _resolve_jsonl_inputs(args.input_jsonl, args.input_dir, args.glob)
        print(infer_jsonl_to_fasta(paths, args.output_fasta, include_source_stem=not args.no_source_stem))


if __name__ == "__main__":
    main()
