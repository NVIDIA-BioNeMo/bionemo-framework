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

"""Train SAE from cached activations.

Loads pre-extracted activations from an ActivationStore cache directory
and trains a Sparse Autoencoder. Requires extract.py to have been run
first to create the cache.

Single-GPU:
    python scripts/train.py \
        --cache-dir .cache/activations/3b_50k_layer24 \
        --model-name nvidia/esm2_t36_3B_UR50D --layer 24 \
        --expansion-factor 8 --top-k 32 --batch-size 4096 --n-epochs 3

Multi-GPU DDP:
    torchrun --nproc_per_node=4 scripts/train.py \
        --cache-dir .cache/activations/3b_50k_layer24 \
        --model-name nvidia/esm2_t36_3B_UR50D --layer 24 \
        --expansion-factor 8 --top-k 32 --batch-size 4096 --n-epochs 3 \
        --dp-size 4

For evaluation, use scripts/eval.py separately.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sae.activation_store import load_activations
from sae.architectures import ReLUSAE, TopKSAE
from sae.perf_logger import PerfLogger
from sae.training import ParallelConfig, Trainer, TrainingConfig, WandbConfig
from sae.utils import get_device, set_seed


def parse_args():
    """Parse command-line arguments for SAE training."""
    p = argparse.ArgumentParser(
        description="Train SAE from cached activations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--cache-dir", type=str, required=True, help="Path to activation cache (from extract.py)")
    p.add_argument("--model-name", type=str, required=True, help="ESM2 model name (for cache validation)")
    p.add_argument("--layer", type=int, required=True, help="Layer index (for cache validation)")

    # SAE architecture
    sae_group = p.add_argument_group("SAE model")
    sae_group.add_argument("--model-type", type=str, default="topk", choices=["topk", "relu"])
    sae_group.add_argument("--expansion-factor", type=int, default=8)
    sae_group.add_argument("--top-k", type=int, default=32)
    sae_group.add_argument("--normalize-input", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument(
        "--auxk", type=int, default=None, help="Auxiliary latents for dead latent loss (None=disabled)"
    )
    sae_group.add_argument("--auxk-coef", type=float, default=1 / 32)
    sae_group.add_argument("--dead-tokens-threshold", type=int, default=10_000_000)
    sae_group.add_argument("--init-pre-bias", action=argparse.BooleanOptionalAction, default=False)
    sae_group.add_argument("--l1-coeff", type=float, default=1e-2, help="L1 coefficient (relu only)")

    # Training
    train_group = p.add_argument_group("Training")
    train_group.add_argument("--lr", type=float, default=3e-4)
    train_group.add_argument("--n-epochs", type=int, default=3)
    train_group.add_argument("--batch-size", type=int, default=4096)
    train_group.add_argument("--log-interval", type=int, default=50)
    train_group.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    train_group.add_argument("--num-workers", type=int, default=0)
    train_group.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument("--max-grad-norm", type=float, default=None)
    train_group.add_argument("--lr-scale-with-latents", action=argparse.BooleanOptionalAction, default=False)
    train_group.add_argument("--lr-reference-hidden-dim", type=int, default=2048)
    train_group.add_argument("--grad-accumulation-steps", type=int, default=1, help="Gradient accumulation steps")
    train_group.add_argument("--warmup-steps", type=int, default=0, help="Linear LR warmup steps")
    train_group.add_argument(
        "--lr-schedule",
        type=str,
        default="constant",
        choices=["constant", "cosine", "linear"],
        help="LR schedule after warmup",
    )
    train_group.add_argument("--lr-min", type=float, default=0.0, help="Minimum LR for decay schedules")
    train_group.add_argument(
        "--lr-decay-steps",
        type=int,
        default=None,
        help="Total steps for LR decay (None = full training)",
    )

    # W&B
    wb_group = p.add_argument_group("Weights & Biases")
    wb_group.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False, dest="wandb_enabled")
    wb_group.add_argument("--wandb-project", type=str, default="sae_esm2_recipe")
    wb_group.add_argument("--wandb-run-name", type=str, default=None)
    wb_group.add_argument("--wandb-group", type=str, default=None)
    wb_group.add_argument("--wandb-job-type", type=str, default=None)

    # Checkpointing
    ckpt_group = p.add_argument_group("Checkpointing")
    ckpt_group.add_argument("--checkpoint-dir", type=str, default=None)
    ckpt_group.add_argument("--checkpoint-steps", type=int, default=None)
    ckpt_group.add_argument("--resume-from", type=str, default=None)

    # Infrastructure
    p.add_argument("--dp-size", type=int, default=1, help="Data parallel size (use with torchrun)")
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--num-proteins",
        type=int,
        default=None,
        help="Subset cached activations to this many proteins' worth of shards",
    )

    return p.parse_args()


def build_sae(args, input_dim: int) -> torch.nn.Module:
    """Build an SAE model from command-line arguments."""
    hidden_dim = input_dim * args.expansion_factor

    if args.model_type == "topk":
        return TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            top_k=args.top_k,
            normalize_input=args.normalize_input,
            auxk=args.auxk,
            auxk_coef=args.auxk_coef,
            dead_tokens_threshold=args.dead_tokens_threshold,
        )
    elif args.model_type == "relu":
        return ReLUSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            l1_coeff=args.l1_coeff,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")


def build_training_config(args, device: str) -> TrainingConfig:
    """Build a TrainingConfig from command-line arguments."""
    return TrainingConfig(
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        device=device,
        log_interval=args.log_interval,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_steps=args.checkpoint_steps,
        lr_scale_with_latents=args.lr_scale_with_latents,
        lr_reference_hidden_dim=args.lr_reference_hidden_dim,
        grad_accumulation_steps=args.grad_accumulation_steps,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        lr_schedule=args.lr_schedule,
        lr_min=args.lr_min,
        lr_decay_steps=args.lr_decay_steps,
    )


def build_wandb_config(args) -> WandbConfig:
    """Build a WandbConfig from command-line arguments."""
    return WandbConfig(
        enabled=args.wandb_enabled,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        config=vars(args),
    )


def build_parallel_config(args) -> ParallelConfig:
    """Build a ParallelConfig from command-line arguments."""
    return ParallelConfig(dp_size=args.dp_size)


def main():
    """Train an SAE from cached activations."""
    args = parse_args()

    set_seed(args.seed)
    device = args.device or get_device()
    print(f"Using device: {device}")
    print(f"Config: {vars(args)}")

    # Load cached activations
    cache_path = Path(args.cache_dir)
    if not (cache_path / "metadata.json").exists():
        raise FileNotFoundError(f"No cache found at {cache_path}. Run extract.py first.")

    store = load_activations(cache_path)
    meta = store.metadata

    # Validate cache matches config
    if meta.get("model_name") != args.model_name:
        raise ValueError(f"Cache model mismatch: {meta['model_name']} vs {args.model_name}")
    if meta.get("layer") != args.layer:
        raise ValueError(f"Cache layer mismatch: {meta['layer']} vs {args.layer}")

    # Compute subsetting
    cached_sequences = meta.get("n_sequences", None)
    max_shards = None
    if args.num_proteins and cached_sequences and args.num_proteins < cached_sequences:
        keep_ratio = args.num_proteins / cached_sequences
        max_shards = max(1, int(np.ceil(keep_ratio * meta["n_shards"])))
        print(
            f"Subsetting: {args.num_proteins}/{cached_sequences} proteins "
            f"-> using {max_shards}/{meta['n_shards']} shards (~{keep_ratio:.1%})"
        )

    # Estimate memory needed to decide: stream vs load-all
    n_shards_to_use = max_shards or meta["n_shards"]
    shard_size = meta.get("shard_size", 100_000)
    est_tokens = n_shards_to_use * shard_size
    est_gb = est_tokens * meta["hidden_dim"] * 4 / (1024**3)
    use_streaming = est_gb > 50  # stream if >50GB per process

    input_dim = meta["hidden_dim"]
    sae = build_sae(args, input_dim)
    print(f"SAE: {args.model_type}, input_dim={input_dim}, hidden_dim={sae.hidden_dim}")

    # Initialize pre_bias from a single shard
    if args.init_pre_bias and hasattr(sae, "init_pre_bias_from_data"):
        print("Initializing pre_bias from geometric median of data...")
        first_shard = torch.from_numpy(store._load_shard(0)).float()
        sample_size = min(32768, len(first_shard))
        sae.init_pre_bias_from_data(first_shard[:sample_size])
        print(f"  pre_bias initialized (mean={sae.pre_bias.mean().item():.4f})")
        del first_shard

    # Build configs
    training_config = build_training_config(args, device)
    wandb_config = build_wandb_config(args)
    parallel_config = build_parallel_config(args)

    perf_logger = PerfLogger(
        log_interval=args.log_interval,
        use_wandb=args.wandb_enabled,
        print_logs=True,
        device=device,
    )

    # Train
    trainer = Trainer(
        sae,
        training_config,
        wandb_config=wandb_config,
        perf_logger=perf_logger,
        parallel_config=parallel_config,
    )

    if use_streaming:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        print(
            f"Streaming from disk (~{est_gb:.0f}GB too large for RAM). "
            f"Peak RAM: ~{shard_size * meta['hidden_dim'] * 4 / (1024**3):.1f}GB/process"
        )

        dataloader = store.get_streaming_dataloader(
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            max_shards=max_shards,
        )
        # Compute min batch count across all ranks to keep DDP in sync
        # Read parquet footers for all ranks' shards (a few KB each, no data loading)
        if world_size > 1:
            import pyarrow.parquet as pq_meta

            dataset = dataloader.dataset
            per_rank = len(dataset.shard_indices)
            min_batches = None
            for r in range(world_size):
                total_rows = sum(
                    pq_meta.read_metadata(store.path / f"shard_{idx:05d}.parquet").num_rows
                    for idx in range(r * per_rank, (r + 1) * per_rank)
                )
                batches = total_rows // args.batch_size
                if min_batches is None or batches < min_batches:
                    min_batches = batches
            dataset.max_batches = min_batches
            print(f"[rank {rank}] capped to {min_batches} batches/epoch for DDP sync")
        trainer.fit(
            dataloader,
            resume_from=args.resume_from,
            data_sharded=True,
        )
    else:
        # Load all shards into memory
        shards = []
        for i, shard in enumerate(store.iter_shards(shuffle_shards=False)):
            if max_shards is not None and i >= max_shards:
                break
            shards.append(torch.from_numpy(shard).float())
        activations_flat = torch.cat(shards)
        print(f"Loaded {activations_flat.shape[0]:,} cached activations into memory")

        trainer.fit(
            activations_flat,
            resume_from=args.resume_from,
        )

    print("Training complete.")


if __name__ == "__main__":
    main()
