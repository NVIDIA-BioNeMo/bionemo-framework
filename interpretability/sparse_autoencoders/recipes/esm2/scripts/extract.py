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

r"""Step 1 (15B): Extract activations from ESM2-15B and save to disk.

Uses nvidia/esm2_t48_15B_UR50D with TransformerEngine for memory-efficient
BF16 inference. No need to clone bionemo-framework -- the model code is
fetched automatically from HuggingFace Hub via trust_remote_code.

This is step 1 of the 3-step ESM2-15B SAE workflow:
    1. step1_15b_extract.py  -- extract activations from ESM2-15B
    2. step2_15b_train.py    -- train SAE on cached activations
    3. step3_15b_eval.py     -- evaluate SAE + build dashboard

Requirements:
    pip install transformer-engine   # or use an NVIDIA NGC container (>=26.01)

Single-GPU:
    python scripts/step1_15b_extract.py \\
        --source uniref50 \\
        --num-proteins 1000000 \\
        --layer 24 \\
        --output ./activations/esm2_15b_layer24

Multi-GPU (each GPU loads full model, needs ~30GB/GPU):
    torchrun --nproc_per_node=4 scripts/step1_15b_extract.py \\
        --source uniref50 \\
        --num-proteins 1000000 \\
        --layer 24 \\
        --output ./activations/esm2_15b_layer24
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import torch
from esm2_sae.data import download_swissprot, download_uniref50, read_fasta
from sae.activation_store import ActivationStore, ActivationStoreConfig
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def parse_args():
    """Parse command-line arguments for activation extraction."""
    p = argparse.ArgumentParser(description="Extract ESM2-15B layer activations")
    # Data source: either --fasta or --source
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--fasta", type=str, help="Path to input FASTA file")
    data.add_argument("--source", type=str, choices=["uniref50", "swissprot"], help="Download from this source")
    p.add_argument("--data-dir", type=str, default="./data", help="Directory for downloaded data (with --source)")
    p.add_argument("--num-proteins", type=int, default=None, help="Number of proteins to extract")

    p.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to extract (0 = embedding output, 1..48 = transformer layers)",
    )
    p.add_argument("--output", type=str, required=True, help="Output directory for activation shards")
    p.add_argument("--model-name", type=str, default="nvidia/esm2_t48_15B_UR50D")
    p.add_argument("--batch-size", type=int, default=1, help="Sequences per batch (keep small for 15B)")
    p.add_argument("--max-length", type=int, default=1024, help="Truncate sequences longer than this")
    p.add_argument("--remove-special-tokens", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", type=str, default=None, help="HF device_map (e.g. 'auto') for model parallelism")
    p.add_argument("--shard-size", type=int, default=100_000, help="Activations per parquet shard")
    p.add_argument(
        "--filter-length",
        action="store_true",
        default=False,
        help="Filter by --max-length during download so exactly --num-proteins short sequences are collected",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _merge_rank_stores(cache_path: Path, world_size: int, metadata: dict) -> None:
    """Merge per-rank temp stores into a single activation store.

    Tolerant of partial failures: skips ranks that didn't finalize,
    merges whatever completed successfully.
    """
    cache_path.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    total_samples = 0
    total_sequences = 0
    hidden_dim = None
    shard_size = None
    merged_ranks = []
    failed_ranks = []

    for r in range(world_size):
        tmp_dir = cache_path / f".tmp_rank_{r}"
        meta_path = tmp_dir / "metadata.json"

        if not meta_path.exists():
            failed_ranks.append(r)
            print(f"  WARNING: Rank {r} did not finalize (no metadata.json). Skipping.")
            # Clean up partial data
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            continue

        with open(meta_path) as f:
            tmp_meta = json.load(f)

        hidden_dim = tmp_meta["hidden_dim"]
        shard_size = tmp_meta["shard_size"]

        for i in range(tmp_meta["n_shards"]):
            src = tmp_dir / f"shard_{i:05d}.parquet"
            dst = cache_path / f"shard_{shard_idx:05d}.parquet"
            shutil.move(str(src), str(dst))
            shard_idx += 1

        total_samples += tmp_meta["n_samples"]
        total_sequences += tmp_meta.get("n_sequences", 0)
        merged_ranks.append(r)
        shutil.rmtree(tmp_dir)

    if not merged_ranks:
        raise RuntimeError("All ranks failed — no data to merge.")

    metadata.update(
        n_samples=total_samples,
        n_shards=shard_idx,
        n_sequences=total_sequences,
        hidden_dim=hidden_dim,
        shard_size=shard_size,
    )
    with open(cache_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    if failed_ranks:
        print(f"Merged {len(merged_ranks)}/{world_size} ranks: {total_samples:,} tokens, {shard_idx} shards")
        print(f"  WARNING: Ranks {failed_ranks} failed. Data is partial ({total_sequences:,} of intended sequences).")
    else:
        print(f"Merged {world_size} rank stores: {total_samples:,} tokens, {shard_idx} shards")


def main():
    """Extract ESM2 layer activations and save to disk."""
    args = parse_args()
    torch.manual_seed(args.seed)

    # --- Distributed setup ---
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        from datetime import timedelta

        import torch.distributed as dist

        if not dist.is_initialized():
            dist.init_process_group("nccl", timeout=timedelta(hours=48))
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Rank {rank}/{world_size}] Device: {device}")

    # --- Check existing cache ---
    cache_path = Path(args.output)
    if (cache_path / "metadata.json").exists():
        if rank == 0:
            with open(cache_path / "metadata.json") as f:
                meta = json.load(f)
            print(f"Cache exists at {cache_path}: {meta['n_samples']:,} tokens. Skipping.")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return

    # Clean stale temp dirs from a previous failed run
    if rank == 0 and cache_path.exists():
        for tmp in cache_path.glob(".tmp_rank_*"):
            shutil.rmtree(tmp)

    # --- Load sequences ---
    if args.fasta:
        fasta_path = Path(args.fasta)
    else:
        # Download from source (only rank 0 downloads)
        data_dir = Path(args.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        if args.source == "uniref50":
            dl_max_length = args.max_length if args.filter_length else None
            if args.num_proteins:
                if dl_max_length:
                    fasta_path = data_dir / f"uniref50_{args.num_proteins}_maxlen{dl_max_length}.fasta"
                else:
                    fasta_path = data_dir / f"uniref50_first_{args.num_proteins}.fasta"
            else:
                fasta_path = data_dir / "uniref50.fasta.gz"
            if fasta_path.exists():
                if rank == 0:
                    print(f"Reusing existing FASTA: {fasta_path}")
            else:
                if rank == 0:
                    print(f"Downloading UniRef50 to {fasta_path}")
                    download_uniref50(data_dir, max_proteins=args.num_proteins, max_length=dl_max_length)
        elif args.source == "swissprot":
            fasta_path = data_dir / "uniprot_sprot.fasta.gz"
            if not fasta_path.exists():
                if rank == 0:
                    print(f"Downloading SwissProt to {fasta_path}")
                    download_swissprot(data_dir)

    # All ranks wait here — ensures download is complete before anyone reads
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()

    records = read_fasta(fasta_path, max_sequences=args.num_proteins, max_length=args.max_length)
    sequences = [r.sequence for r in records]
    total_sequences = len(sequences)

    if rank == 0:
        print(f"Loaded {total_sequences} sequences from {fasta_path}")

    if world_size > 1:
        dist.barrier()
        chunk = total_sequences // world_size
        start = rank * chunk
        end = total_sequences if rank == world_size - 1 else (rank + 1) * chunk
        my_sequences = sequences[start:end]
        print(f"[Rank {rank}] sequences {start}-{end} ({len(my_sequences)})")
    else:
        my_sequences = sequences

    # --- Load model ---
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model_dtype = dtype_map[args.dtype]

    model_kwargs = {
        "trust_remote_code": True,
        "dtype": model_dtype,
        "add_pooling_layer": False,
    }
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    if rank == 0:
        print(f"Loading {args.model_name} ({args.dtype})...")

    model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    if not args.device_map:
        model = model.to(device)
    model.eval()

    num_layers = model.config.num_hidden_layers
    if args.layer < 0 or args.layer > num_layers:
        raise ValueError(
            f"--layer {args.layer} out of range [0, {num_layers}]. "
            f"0 = embedding output, {num_layers} = last transformer layer."
        )

    # Tokenizer: try model repo first, fall back to facebook's (same tokenizer)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    if rank == 0:
        print(f"Extracting layer {args.layer}/{num_layers} (hidden_dim={model.config.hidden_size})")

    # --- Extract activations ---
    store_path = cache_path / f".tmp_rank_{rank}" if world_size > 1 else cache_path
    store = ActivationStore(store_path, ActivationStoreConfig(shard_size=args.shard_size))

    input_device = next(model.parameters()).device

    n_batches = (len(my_sequences) + args.batch_size - 1) // args.batch_size
    iterator = range(0, len(my_sequences), args.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting")

    log_interval = max(1, n_batches // 20)  # ~5% progress logs for non-rank-0

    t0 = time.time()
    extraction_error = None
    batches_done = 0
    try:
        with torch.no_grad():
            for i in iterator:
                batch_seqs = my_sequences[i : i + args.batch_size]

                inputs = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=args.max_length,
                )
                inputs = {k: v.to(input_device) for k, v in inputs.items()}

                outputs = model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[args.layer].float().cpu()  # [B, L, D]
                mask = inputs["attention_mask"].cpu()  # [B, L]

                if args.remove_special_tokens:
                    # Remove CLS (position 0) and EOS (last real token) per sequence
                    keep = mask.clone()
                    keep[:, 0] = 0
                    lengths = mask.sum(dim=1)
                    for b in range(keep.shape[0]):
                        eos = int(lengths[b].item()) - 1
                        if eos > 0:
                            keep[b, eos] = 0
                    mask = keep

                flat = hidden[mask.bool()]  # [N_tokens, hidden_dim]
                store.append(flat)
                batches_done += 1

                # Progress logging for non-rank-0 processes
                if rank != 0 and batches_done % log_interval == 0:
                    print(
                        f"[Rank {rank}] {batches_done}/{n_batches} batches ({100 * batches_done / n_batches:.0f}%)",
                        flush=True,
                    )

    except Exception as e:
        extraction_error = e
        print(
            f"[Rank {rank}] EXTRACTION FAILED at batch {batches_done}/{n_batches}: {type(e).__name__}: {e}", flush=True
        )

    if extraction_error is None:
        store.finalize(
            metadata={
                "model_name": args.model_name,
                "layer": args.layer,
                "n_sequences": len(my_sequences),
                "dtype": args.dtype,
            }
        )

        elapsed = time.time() - t0
        print(
            f"[Rank {rank}] {store.metadata['n_samples']:,} tokens from "
            f"{len(my_sequences)} sequences in {elapsed:.1f}s",
            flush=True,
        )
    else:
        print(
            f"[Rank {rank}] Extraction incomplete: {batches_done}/{n_batches} batches. Store NOT finalized.",
            flush=True,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Multi-GPU merge ---
    if world_size > 1:
        import torch.distributed as dist

        dist.barrier()
        if rank == 0:
            _merge_rank_stores(
                cache_path,
                world_size,
                {
                    "model_name": args.model_name,
                    "layer": args.layer,
                    "n_sequences": total_sequences,
                    "dtype": args.dtype,
                },
            )
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        with open(cache_path / "metadata.json") as f:
            meta = json.load(f)
        print("\nExtraction complete:")
        print(f"  Output:     {cache_path}")
        print(f"  Sequences:  {meta.get('n_sequences', '?')}")
        print(f"  Tokens:     {meta['n_samples']:,}")
        print(f"  Hidden dim: {meta['hidden_dim']}")
        print(f"  Shards:     {meta['n_shards']}")


if __name__ == "__main__":
    main()
