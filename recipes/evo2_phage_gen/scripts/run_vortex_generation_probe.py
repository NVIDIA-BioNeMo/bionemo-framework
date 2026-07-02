#!/usr/bin/env python
"""Run a small Arc/Vortex Evo2 generation probe for phage prompts."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from evo2 import Evo2


DNA_ALPHABET = frozenset("ACGTacgt")


def _prompt_nucleotides(prompt: str) -> str:
    return "".join(char for char in prompt if char in DNA_ALPHABET).upper()


def _synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
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
    prompt_nt = _prompt_nucleotides(args.prompt)
    n_tokens = args.total_nt - len(prompt_nt)
    if n_tokens <= 0:
        raise ValueError(f"--total-nt must exceed nucleotide prompt length {len(prompt_nt)}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    batch_suffix = f"_batch{args.batch_size}" if args.batch_size != 1 else ""
    jsonl_path = (
        args.output_dir / f"vortex_prompt{len(prompt_nt)}_temp{args.temperature}_n{args.num_generations}{batch_suffix}.jsonl"
    )
    fasta_path = (
        args.output_dir / f"vortex_prompt{len(prompt_nt)}_temp{args.temperature}_n{args.num_generations}{batch_suffix}.fasta"
    )

    print(f"loading model {args.model_name} from {args.checkpoint}", flush=True)
    load_start = time.perf_counter()
    model = Evo2(args.model_name, local_path=str(args.checkpoint), use_kernels=False)
    _synchronize_cuda()
    load_elapsed_s = time.perf_counter() - load_start
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(
        f"generating {args.num_generations} sequences: prompt={args.prompt!r}, "
        f"n_tokens={n_tokens}, batch_size={args.batch_size}",
        flush=True,
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

            if scores is None:
                score_values = [None] * len(seqs)
            else:
                if isinstance(scores, torch.Tensor):
                    scores = scores.detach().cpu().tolist()
                if not isinstance(scores, (list, tuple)):
                    scores = [scores]
                score_values = scores

            batch_elapsed_s = time.perf_counter() - batch_start
            completions = [str(seq).replace("\n", "").strip().split(maxsplit=1)[0] for seq in seqs]
            batch_completion_tokens = sum(len(completion) for completion in completions)
            batch_completion_tokens_per_s = (
                batch_completion_tokens / batch_elapsed_s if batch_elapsed_s > 0 else 0.0
            )

            for batch_idx, completion in enumerate(completions):
                idx = generated_count + batch_idx
                completion_tokens = len(completion)
                total_completion_tokens += completion_tokens
                full_seq = f"{prompt_nt}{completion}".upper()
                score = None if score_values[batch_idx] is None else float(score_values[batch_idx])
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
            print(
                f"generated {generated_count + batch_count}/{args.num_generations} "
                f"batch_size={batch_count} batch_elapsed_s={batch_elapsed_s:.3f} "
                f"batch_tok_s={batch_completion_tokens_per_s:.2f}",
                flush=True,
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
    print(
        "throughput: "
        f"generation_tok_s={summary['generation_completion_tokens_per_s']:.2f} "
        f"end_to_end_tok_s={summary['end_to_end_completion_tokens_per_s']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
