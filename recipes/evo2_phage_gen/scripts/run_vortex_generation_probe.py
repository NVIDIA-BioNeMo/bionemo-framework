#!/usr/bin/env python

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

"""Run a small Arc/Vortex Evo2 generation probe for phage prompts."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from evo2 import Evo2

from bionemo.evo2_phage_gen.qc import prompt_nucleotides, trim_at_first_eos


logger = logging.getLogger(__name__)


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _score_float(value: object) -> float | None:
    """Convert a scalar or singleton-nested score to a float when possible."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, (list, tuple)) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_score_values(scores: object, expected_count: int) -> list[float | None]:
    """Return one position-preserving optional scalar score per generated sequence."""
    if expected_count < 0:
        raise ValueError("expected score count must be non-negative")
    if scores is None:
        return [None] * expected_count
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    if not isinstance(scores, (list, tuple)):
        return [_score_float(scores)] if expected_count == 1 else [None] * expected_count
    if len(scores) != expected_count:
        return [None] * expected_count
    return [_score_float(value) for value in scores]


def _completion_token(sequence: object) -> str:
    """Extract generated nucleotides before a textual EOS marker or whitespace."""
    trimmed = trim_at_first_eos(str(sequence).replace("\n", "").strip())
    fields = trimmed.split(maxsplit=1)
    return fields[0] if fields else ""


def main() -> None:
    """Run the command-line generation probe."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Run Arc/Vortex Evo2 generation for a small phage probe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default="+~GAGT")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-generations", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--total-nt", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-name", type=str, default="evo2_7b_microviridae")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_nt = prompt_nucleotides(args.prompt).upper()
    n_tokens = args.total_nt - len(prompt_nt)
    if n_tokens <= 0:
        raise ValueError(f"--total-nt must exceed nucleotide prompt length {len(prompt_nt)}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    batch_suffix = f"_batch{args.batch_size}" if args.batch_size != 1 else ""
    jsonl_path = (
        args.output_dir
        / f"vortex_prompt{len(prompt_nt)}_temp{args.temperature}_n{args.num_generations}{batch_suffix}.jsonl"
    )
    fasta_path = (
        args.output_dir
        / f"vortex_prompt{len(prompt_nt)}_temp{args.temperature}_n{args.num_generations}{batch_suffix}.fasta"
    )

    logger.info("loading model %s from %s", args.model_name, args.checkpoint)
    load_start = time.perf_counter()
    model = Evo2(args.model_name, local_path=str(args.checkpoint), use_kernels=False)
    _synchronize_cuda()
    load_elapsed_s = time.perf_counter() - load_start
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger.info(
        "generating %s sequences: prompt=%r, n_tokens=%s, batch_size=%s",
        args.num_generations,
        args.prompt,
        n_tokens,
        args.batch_size,
    )
    _synchronize_cuda()
    generation_start = time.perf_counter()
    total_completion_tokens = 0
    total_batches = 0
    with jsonl_path.open("w") as jsonl, fasta_path.open("w") as fasta:
        generated_count = 0
        while generated_count < args.num_generations:
            batch_count = min(args.batch_size, args.num_generations - generated_count)
            _synchronize_cuda()
            batch_start = time.perf_counter()
            result = model.generate(
                prompt_seqs=[args.prompt] * batch_count,
                n_tokens=n_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                batched=True,
                cached_generation=True,
                verbose=0,
                force_prompt_threshold=min(200, len(args.prompt)),
            )
            _synchronize_cuda()
            seqs = getattr(result, "sequences", result[0] if isinstance(result, tuple) else result)
            scores = getattr(result, "scores", result[1] if isinstance(result, tuple) and len(result) > 1 else None)
            if isinstance(seqs, str):
                seqs = [seqs]
            if len(seqs) != batch_count:
                raise ValueError(f"expected {batch_count} generated sequences, got {len(seqs)}")

            score_values = _normalize_score_values(scores, len(seqs))

            batch_elapsed_s = time.perf_counter() - batch_start
            completions = [_completion_token(seq) for seq in seqs]
            batch_completion_tokens = sum(len(completion) for completion in completions)
            batch_completion_tokens_per_s = batch_completion_tokens / batch_elapsed_s if batch_elapsed_s > 0 else 0.0

            for batch_idx, completion in enumerate(completions):
                idx = generated_count + batch_idx
                completion_tokens = len(completion)
                total_completion_tokens += completion_tokens
                full_seq = f"{prompt_nt}{completion}".upper()
                score = score_values[batch_idx]
                record = {
                    "id": f"vortex_prompt{len(prompt_nt)}_temp{args.temperature}_{idx:04d}",
                    "prompt": args.prompt,
                    "completion": completion,
                    "score": score,
                    "finish_reason": "length",
                    "usage": {
                        "prompt_tokens": len(args.prompt),
                        "completion_tokens": completion_tokens,
                        "total_tokens": len(args.prompt) + completion_tokens,
                    },
                    "batch_index": total_batches,
                    "batch_size": batch_count,
                    "batch_elapsed_s": batch_elapsed_s,
                    "batch_completion_tokens_per_s": batch_completion_tokens_per_s,
                    "elapsed_s": batch_elapsed_s,
                    "completion_tokens_per_s": completion_tokens / batch_elapsed_s if batch_elapsed_s > 0 else 0.0,
                }
                jsonl.write(json.dumps(record) + "\n")
                fasta.write(f">{record['id']}\n{full_seq}\n")

            jsonl.flush()
            fasta.flush()
            logger.info(
                "generated %s/%s batch_size=%s batch_elapsed_s=%.3f batch_tok_s=%.2f",
                generated_count + batch_count,
                args.num_generations,
                batch_count,
                batch_elapsed_s,
                batch_completion_tokens_per_s,
            )
            generated_count += batch_count
            total_batches += 1

    _synchronize_cuda()
    generation_elapsed_s = time.perf_counter() - generation_start
    summary = {
        "model_name": args.model_name,
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "prompt_nt_length": len(prompt_nt),
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "num_generations": args.num_generations,
        "batch_size": args.batch_size,
        "total_batches": total_batches,
        "n_tokens_per_generation": n_tokens,
        "load_elapsed_s": load_elapsed_s,
        "generation_elapsed_s": generation_elapsed_s,
        "total_elapsed_s": load_elapsed_s + generation_elapsed_s,
        "total_completion_tokens": total_completion_tokens,
        "generation_completion_tokens_per_s": (
            total_completion_tokens / generation_elapsed_s if generation_elapsed_s > 0 else 0.0
        ),
        "end_to_end_completion_tokens_per_s": (
            total_completion_tokens / (load_elapsed_s + generation_elapsed_s)
            if (load_elapsed_s + generation_elapsed_s) > 0
            else 0.0
        ),
    }
    summary_path = (
        args.output_dir
        / f"vortex_prompt{len(prompt_nt)}_temp{args.temperature}_n{args.num_generations}{batch_suffix}_summary.json"
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"jsonl: {jsonl_path}", flush=True)
    print(f"fasta: {fasta_path}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    logger.info(
        "throughput: generation_tok_s=%.2f end_to_end_tok_s=%.2f",
        summary["generation_completion_tokens_per_s"],
        summary["end_to_end_completion_tokens_per_s"],
    )


if __name__ == "__main__":
    main()
