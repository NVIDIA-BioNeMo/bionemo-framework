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

"""Step 1: Extract activations from CodonFM (Encodon) and save to disk.

Single-GPU:
    python scripts/extract.py \
        --csv-path path/to/Primates.csv \
        --model-path path/to/encodon_1b \
        --layer -2 \
        --output .cache/activations/encodon_1b_layer-2

Multi-GPU:
    torchrun --nproc_per_node=4 scripts/extract.py \
        --csv-path path/to/Primates.csv \
        --model-path path/to/encodon_1b \
        --layer -2 \
        --output .cache/activations/encodon_1b_layer-2
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


# Use codonfm_ptl_te recipe (has TransformerEngine support)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
_CODONFM_TE_DIR = _REPO_ROOT / "recipes" / "codonfm_ptl_te"
sys.path.insert(0, str(_CODONFM_TE_DIR))

from codonfm_sae.data import read_codon_csv  # noqa: E402
from sae.activation_store import ActivationStore, ActivationStoreConfig  # noqa: E402
from src.data.preprocess.codon_sequence import process_item  # noqa: E402
from src.inference.encodon import EncodonInference  # noqa: E402


def parse_args():  # noqa: D103
    p = argparse.ArgumentParser(description="Extract CodonFM layer activations")
    p.add_argument(
        "--csv-path", type=str, required=True, help="Path to CSV with DNA sequences (auto-detects 'seq'/'cds' column)"
    )
    p.add_argument("--seq-column", type=str, default=None, help="Column name for sequences (auto-detect if omitted)")
    p.add_argument("--num-sequences", type=int, default=None, help="Max sequences to extract")
    p.add_argument(
        "--model-path", type=str, required=True, help="Path to Encodon checkpoint (.ckpt, .safetensors, or directory)"
    )
    p.add_argument("--layer", type=int, required=True, help="Layer index (negative = from end, e.g. -2 = penultimate)")
    p.add_argument("--context-length", type=int, default=2048, help="Max context length in codons (default: 2048)")
    p.add_argument("--output", type=str, required=True, help="Output directory for activation shards")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=100_000)
    p.add_argument(
        "--use-transformer-engine",
        action="store_true",
        default=True,
        help="Use TransformerEngine model (default: True, for TE checkpoints)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _merge_rank_stores(cache_path: Path, world_size: int, metadata: dict) -> None:
    """Merge per-rank temp stores into a single activation store."""
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
            print(f"  WARNING: Rank {r} did not finalize. Skipping.")
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            continue

        with open(meta_path) as f:
            tmp_meta = json.load(f)

        hidden_dim = tmp_meta["hidden_dim"]
        shard_size = tmp_meta["shard_size"]

        for i in range(tmp_meta["n_shards"]):
            src_file = tmp_dir / f"shard_{i:05d}.parquet"
            dst_file = cache_path / f"shard_{shard_idx:05d}.parquet"
            shutil.move(str(src_file), str(dst_file))
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
        print(f"  WARNING: Ranks {failed_ranks} failed.")
    else:
        print(f"Merged {world_size} rank stores: {total_samples:,} tokens, {shard_idx} shards")


def main():  # noqa: D103
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

    # Clean stale temp dirs
    if rank == 0 and cache_path.exists():
        for tmp in cache_path.glob(".tmp_rank_*"):
            shutil.rmtree(tmp)

    # --- Load sequences ---
    max_codons = args.context_length - 2  # room for CLS + SEP
    records = read_codon_csv(
        args.csv_path,
        seq_column=args.seq_column,
        max_sequences=args.num_sequences,
        max_codons=max_codons,
    )
    sequences = [r.sequence for r in records]
    total_sequences = len(sequences)

    if rank == 0:
        print(f"Loaded {total_sequences} sequences from {args.csv_path}")

    # Shard across ranks
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
    if rank == 0:
        print(f"Loading model from {args.model_path}...")

    inf = EncodonInference(
        model_path=args.model_path,
        task_type="embedding_prediction",
        use_transformer_engine=args.use_transformer_engine,
    )
    inf.configure_model()
    inf.model.to(device).eval()

    num_layers = len(inf.model.model.layers)
    target_layer = args.layer if args.layer >= 0 else num_layers + args.layer
    hidden_dim = inf.model.model.config.hidden_size

    if rank == 0:
        print(f"Extracting layer {target_layer}/{num_layers} (hidden_dim={hidden_dim})")

    # --- Extract activations ---
    store_path = cache_path / f".tmp_rank_{rank}" if world_size > 1 else cache_path
    store = ActivationStore(store_path, ActivationStoreConfig(shard_size=args.shard_size))

    n_batches = (len(my_sequences) + args.batch_size - 1) // args.batch_size
    iterator = range(0, len(my_sequences), args.batch_size)
    if rank == 0:
        iterator = tqdm(iterator, total=n_batches, desc="Extracting")

    log_interval = max(1, n_batches // 20)

    t0 = time.time()
    extraction_error = None
    batches_done = 0
    try:
        with torch.no_grad():
            for i in iterator:
                batch_seqs = my_sequences[i : i + args.batch_size]
                items = [
                    process_item(s, context_length=args.context_length, tokenizer=inf.tokenizer) for s in batch_seqs
                ]

                batch = {
                    "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
                    "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
                }

                out = inf.model(batch, return_hidden_states=True)
                layer_acts = out.all_hidden_states[args.layer]  # [B, L, D]

                # Strip CLS (pos 0) and SEP (last real pos), keep only codon positions
                for j, it in enumerate(items):
                    seq_len = it["attention_mask"].sum()
                    acts = layer_acts[j, 1 : seq_len - 1, :].float().cpu()  # [num_codons, hidden_dim]
                    store.append(acts)

                batches_done += 1
                del out, layer_acts, batch
                torch.cuda.empty_cache()

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
                "model_path": args.model_path,
                "layer": args.layer,
                "target_layer": target_layer,
                "num_layers": num_layers,
                "n_sequences": len(my_sequences),
                "context_length": args.context_length,
            }
        )

        elapsed = time.time() - t0
        print(
            f"[Rank {rank}] {store.metadata['n_samples']:,} tokens from "
            f"{len(my_sequences)} sequences in {elapsed:.1f}s",
            flush=True,
        )
    else:
        print(f"[Rank {rank}] Extraction incomplete. Store NOT finalized.", flush=True)

    del inf
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
                    "model_path": args.model_path,
                    "layer": args.layer,
                    "target_layer": target_layer,
                    "num_layers": num_layers,
                    "n_sequences": total_sequences,
                    "context_length": args.context_length,
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
