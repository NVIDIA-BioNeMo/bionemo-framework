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

"""FSDP2 training script for ESM2-MiniFold TE structure prediction.

Hydra-based training loop that:
1. Loads a frozen ESM-2 backbone (HuggingFace)
2. Trains the TE folding head on distogram prediction (Stage 1)
3. Optionally trains the structure module for full 3D prediction (Stage 2)

Three parameter groups with separate learning rates:
- Backbone: frozen (lr=0) or fine-tuned (lr=3e-5)
- Folding head: lr=1e-4
- Structure module: lr=1e-4

Usage:
    # Single GPU
    python train_fsdp2.py --config-name L0_sanity

    # Multi GPU
    torchrun --nproc_per_node=8 train_fsdp2.py --config-name defaults
"""

import logging
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW

from checkpoint import load_checkpoint_fsdp2, save_checkpoint_fsdp2, save_final_model_fsdp2, should_save_checkpoint
from dataset import create_dataloader
from distributed_config import DistributedConfig
from modeling_esm2_minifold_te import ESM2MiniFoldTE
from perf_logger import PerfLogger
from quantization import (
    BufferedQuantLogger,
    ComponentPrecisionConfig,
    initialize_quant_stats_logging,
    resolve_layer_precision,
)
from scheduler import get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def set_global_seed(seed: int, local_rank: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible short-run comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make per-rank streams deterministic but distinct when needed.
    torch.cuda.set_device(local_rank)


def compute_distogram_loss(preds, coords, mask, no_bins=64, max_dist=25.0):
    """Compute distogram cross-entropy loss.

    Args:
        preds: Predicted distogram logits (B, L, L, no_bins).
        coords: Ca coordinates (B, L, 3).
        mask: Residue mask (B, L).
        no_bins: Number of distance bins.
        max_dist: Maximum distance in Angstroms.

    Returns:
        Scalar loss tensor.
    """
    # Compute pairwise Ca distances
    dists = torch.cdist(coords, coords)

    # Bin distances into one-hot labels
    boundaries = torch.linspace(2, max_dist, no_bins - 1, device=dists.device)
    labels = F.one_hot(
        (dists.unsqueeze(-1) > boundaries).sum(dim=-1),
        no_bins,
    ).to(preds.dtype)

    # Cross-entropy loss
    errors = -torch.sum(labels * F.log_softmax(preds, dim=-1), dim=-1)

    # Square mask (exclude self-distances and padding)
    square_mask = mask[:, None, :] * mask[:, :, None]
    eye = torch.eye(mask.shape[1], device=mask.device).unsqueeze(0)
    square_mask = square_mask * (1 - eye)

    # FP16-friendly mean
    denom = 1e-5 + square_mask.sum(dim=(-1, -2))
    mean = (errors * square_mask).sum(dim=-1) / denom[..., None]
    mean = mean.sum(dim=-1)
    return mean.mean()


def compute_distogram_metrics(preds, coords, mask, no_bins=64, max_dist=25.0, contact_threshold=8.0):
    """Compute structure prediction quality metrics from distogram predictions.

    Args:
        preds: Predicted distogram logits (B, L, L, no_bins).
        coords: Ca coordinates (B, L, 3).
        mask: Residue mask (B, L).
        no_bins: Number of distance bins.
        max_dist: Maximum distance in Angstroms.
        contact_threshold: Distance threshold for contact prediction (Angstroms).

    Returns:
        Dict with: distogram_acc, contact_precision, contact_recall,
                   lddt_from_distogram, mean_distance_error.
    """
    with torch.no_grad():
        # True pairwise distances
        true_dists = torch.cdist(coords, coords)

        # Bin boundaries and centers
        boundaries = torch.linspace(2, max_dist, no_bins - 1, device=preds.device)
        bin_centers = torch.cat(
            [
                torch.tensor([1.0], device=preds.device),
                (boundaries[:-1] + boundaries[1:]) / 2,
                torch.tensor([max_dist + 2.0], device=preds.device),
            ]
        )

        # True bin indices
        true_bins = (true_dists.unsqueeze(-1) > boundaries).sum(dim=-1)

        # Predicted bin indices and probabilities
        pred_bins = preds.argmax(dim=-1)
        pred_probs = F.softmax(preds, dim=-1)

        # Expected predicted distance from distogram
        pred_dists = (pred_probs * bin_centers).sum(dim=-1)

        # Valid pair mask (exclude self and padding)
        square_mask = mask[:, None, :] * mask[:, :, None]
        eye = torch.eye(mask.shape[1], device=mask.device).unsqueeze(0)
        pair_mask = square_mask * (1 - eye)
        n_pairs = pair_mask.sum().clamp(min=1)

        # 1. Distogram accuracy
        correct = (pred_bins == true_bins).float() * pair_mask
        distogram_acc = correct.sum() / n_pairs

        # 2. Contact precision and recall at threshold
        true_contacts = (true_dists < contact_threshold).float() * pair_mask
        pred_contacts = (pred_dists < contact_threshold).float() * pair_mask

        tp = (true_contacts * pred_contacts).sum()
        contact_precision = tp / pred_contacts.sum().clamp(min=1)
        contact_recall = tp / true_contacts.sum().clamp(min=1)

        # 3. lDDT from distogram expected distances
        # Standard lDDT: fraction of pairwise distances within thresholds
        dist_error = torch.abs(pred_dists - true_dists)
        lddt_score = (
            (dist_error < 0.5).float()
            + (dist_error < 1.0).float()
            + (dist_error < 2.0).float()
            + (dist_error < 4.0).float()
        ) * 0.25

        # Only score pairs within 15Å cutoff (standard lDDT)
        lddt_mask = pair_mask * (true_dists < 15.0).float()
        lddt_from_distogram = (lddt_score * lddt_mask).sum() / lddt_mask.sum().clamp(min=1)

        # 4. Mean distance error (on valid pairs within 15Å)
        mean_dist_error = (dist_error * lddt_mask).sum() / lddt_mask.sum().clamp(min=1)

        return {
            "distogram_acc": distogram_acc,
            "contact_precision_8A": contact_precision,
            "contact_recall_8A": contact_recall,
            "lddt_from_distogram": lddt_from_distogram,
            "mean_distance_error": mean_dist_error,
        }


@hydra.main(config_path="hydra_config", config_name="L0_sanity", version_base="1.2")
def main(args: DictConfig) -> float | None:
    """Train ESM2-MiniFold TE with FSDP2.

    Returns:
        float: The final loss value.
    """
    os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Initialize distributed
    dist_config = DistributedConfig()
    logger.info("Initializing distributed training: %s", dist_config)
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)
    set_global_seed(int(args.seed), dist_config.local_rank)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size,),
        mesh_dim_names=("dp",),
    )

    # Resolve per-block quantization precision
    block_precision = resolve_layer_precision(
        num_layers=args.model.num_blocks,
        fp8_enabled=args.fp8_config.enabled,
        fp4_enabled=args.fp4_config.enabled,
        fp8_layers=OmegaConf.to_container(args.fp8_layers, resolve=True) if args.fp8_layers is not None else None,
        fp4_layers=OmegaConf.to_container(args.fp4_layers, resolve=True) if args.fp4_layers is not None else None,
    )

    fp8_recipe = None
    fp4_recipe = None
    if args.fp8_config.enabled:
        from transformer_engine.common.recipe import Format

        fp8_recipe = hydra.utils.get_class(args.fp8_config.fp8_recipe)(
            fp8_format=Format[args.fp8_config.fp8_format], **args.fp8_config.fp8_recipe_kwargs
        )
    if args.fp4_config.enabled:
        from transformer_engine.common.recipe import Format

        fp4_recipe = hydra.utils.get_class(args.fp4_config.fp4_recipe)(
            fp4_format=Format[args.fp4_config.fp4_format], **args.fp4_config.fp4_recipe_kwargs
        )

    # Component-level precision overrides
    component_precision = ComponentPrecisionConfig(**OmegaConf.to_container(args.component_precision, resolve=True))

    # Quant stats logging
    quant_logger = None
    if args.quant_stats_config.enabled:
        if dist_config.is_main_process():
            quant_logger = BufferedQuantLogger()
        initialize_quant_stats_logging(
            quant_stats_file=args.quant_stats_config.quant_stats_file,
            quant_log_dir=args.quant_stats_config.quant_log_dir,
            rank=dist_config.rank,
            layer_precision=block_precision,
            statistics_logger=quant_logger,
            component_precision=component_precision,
        )

    # Create model
    params_dtype = torch.float32
    model = ESM2MiniFoldTE(
        esm_model_name=args.esm_model_name,
        c_s=args.model.c_s,
        c_z=args.model.c_z,
        num_blocks=args.model.num_blocks,
        no_bins=args.model.no_bins,
        use_structure_module=args.model.use_structure_module,
        params_dtype=params_dtype,
        block_precision=block_precision,
        fp8_recipe=fp8_recipe,
        fp4_recipe=fp4_recipe,
        component_precision=component_precision,
    ).to(device)

    logger.info("Model created: %d parameters", sum(p.numel() for p in model.parameters()))
    logger.info("Trainable: %d parameters", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # FSDP2: shard MiniFormer blocks individually for memory efficiency
    if args.use_fp32_master_weights:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,  # Cast params to BF16 for forward/backward
            reduce_dtype=torch.float32,  # Gradient reductions in FP32
            output_dtype=torch.bfloat16,  # Forward output dtype
            cast_forward_inputs=False,
        )
    else:
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    for block in model.fold.miniformer.blocks:
        fully_shard(block, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    # Assign layer names for quant stats debug API
    if args.quant_stats_config.enabled:
        import nvdlfw_inspect.api as debug_api

        debug_api.infer_and_assign_layer_names(model)

    # Optimizer with parameter groups
    param_groups = [
        {
            "params": list(model.get_folding_head_params()),
            "lr": args.optimizer.folding_lr,
            "name": "folding_head",
        },
    ]
    if args.model.use_structure_module:
        param_groups.append(
            {
                "params": list(model.get_structure_module_params()),
                "lr": args.optimizer.struct_lr,
                "name": "structure_module",
            }
        )

    optimizer = AdamW(
        param_groups,
        betas=tuple(args.optimizer.betas),
        eps=args.optimizer.eps,
        weight_decay=args.optimizer.weight_decay,
        fused=True,
    )
    scheduler = get_linear_schedule_with_warmup(optimizer, **args.lr_scheduler_kwargs)

    # Create dataloader
    train_dataloader, sampler = create_dataloader(dist_config, **args.dataset)

    if dist_config.is_main_process():
        logger.info("Block precision: %s", block_precision)

    # Resume from checkpoint
    ckpt_path = Path(args.checkpoint.ckpt_dir) / "train_fsdp2" if args.checkpoint.ckpt_dir else None
    if args.checkpoint.resume_from_checkpoint and ckpt_path:
        model, optimizer, scheduler, train_dataloader, start_step, epoch = load_checkpoint_fsdp2(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ckpt_path=ckpt_path,
            dist_config=dist_config,
            dataloader=train_dataloader,
        )
    else:
        start_step = 0
        epoch = 0

    perf_logger = PerfLogger(dist_config, args, quant_logger=quant_logger)

    # Training loop
    step = start_step
    while step < args.num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass (BF16 handled by FSDP2 MixedPrecisionPolicy)
            r_dict = model(batch, num_recycling=args.model.get("num_recycling", 0))

            # Compute distogram loss
            disto_loss = compute_distogram_loss(
                preds=r_dict["preds"],
                coords=batch["coords"],
                mask=batch["mask"],
                no_bins=args.model.no_bins,
            )

            total_loss = disto_loss

            # Optional structure module loss (Stage 2)
            if args.model.use_structure_module and "sm" in r_dict:
                from loss import AlphaFoldLoss

                loss_of, _ = AlphaFoldLoss(r_dict, batch.get("batch_of", {}))
                total_loss = 0.8 * disto_loss + 0.2 * loss_of

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # Compute structure quality metrics (no grad, cheap)
            structure_metrics = compute_distogram_metrics(
                preds=r_dict["preds"].float(),
                coords=batch["coords"],
                mask=batch["mask"],
                no_bins=args.model.no_bins,
            )

            # Count unpadded tokens across all GPUs
            unpadded_tokens = batch["mask"].sum().item() * dist_config.world_size

            # Logging
            perf_logger.log_step(
                step=step,
                loss=total_loss,
                disto_loss=disto_loss,
                grad_norm=total_norm,
                lr=optimizer.param_groups[0]["lr"],
                structure_metrics=structure_metrics,
                unpadded_tokens=unpadded_tokens,
            )

            # Checkpointing
            if ckpt_path and should_save_checkpoint(step, args.checkpoint.save_every_n_steps):
                save_checkpoint_fsdp2(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    ckpt_path=ckpt_path,
                    step=step,
                    epoch=epoch,
                    dist_config=dist_config,
                    max_checkpoints=args.checkpoint.max_checkpoints,
                )

            step += 1
            if step >= args.num_train_steps:
                break

        epoch += 1
        sampler.set_epoch(epoch)

    # Save final model
    if args.checkpoint.save_final_model and ckpt_path:
        save_final_model_fsdp2(
            model=model,
            save_directory=ckpt_path / "final_model",
            dist_config=dist_config,
        )

    perf_logger.finish()
    torch.distributed.destroy_process_group()

    return perf_logger.min_loss


if __name__ == "__main__":
    main()
