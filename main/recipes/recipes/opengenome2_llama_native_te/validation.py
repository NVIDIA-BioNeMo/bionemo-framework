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

"""Validation utilities for OpenGenome2 training.

Provides validation loop with per-token and per-batch loss metrics,
following both HuggingFace and Megatron-style loss computation conventions.
"""

import logging

import torch
import transformer_engine.pytorch

from distributed_config import DistributedConfig


logger = logging.getLogger(__name__)


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_dataloader,
    num_batches: int,
    device: torch.device,
    dist_config: DistributedConfig,
) -> dict:
    """Run validation and compute loss metrics.

    FP8 is disabled during validation to match NeMo's behavior -- TE's FP8 state tracking
    (amax history, scaling factors) does not handle eval/no_grad mode properly with FSDP2.

    Args:
        model: The model to evaluate.
        val_dataloader: DataLoader for validation data.
        num_batches: Number of batches to evaluate.
        device: Device to run on.
        dist_config: Distributed config for logging.

    Returns:
        Dictionary with val_loss, val_ppl, and Megatron-style metrics.
    """
    model.eval()

    total_loss = 0.0  # Sum of per-batch mean losses (HF-style)
    total_weighted_loss = 0.0  # Sum of (batch_loss * batch_tokens) for Megatron-style
    total_tokens = 0
    num_evaluated = 0

    val_iter = iter(val_dataloader)

    for _ in range(num_batches):
        try:
            batch = next(val_iter)
        except StopIteration:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        try:
            with transformer_engine.pytorch.autocast(enabled=False):
                outputs = model(**batch)

            loss = outputs.loss
            if loss is not None:
                loss_val = loss.item()
                total_loss += loss_val
                labels = batch.get("labels", None)
                num_tokens = (labels != -100).sum().item() if labels is not None else batch["input_ids"].numel()
                total_tokens += num_tokens
                total_weighted_loss += loss_val * num_tokens
            num_evaluated += 1
        except Exception as e:
            logger.warning(f"Validation forward pass failed on rank {dist_config.rank}: {e}")
            continue

    if num_evaluated == 0:
        raise RuntimeError(f"All {num_batches} validation batches failed on rank {dist_config.rank}")

    torch.distributed.barrier()

    # Aggregate across ranks
    loss_tensor = torch.tensor(
        [total_loss, float(total_tokens), float(num_evaluated), total_weighted_loss], device=device
    )
    torch.distributed.all_reduce(loss_tensor)
    global_loss = loss_tensor[0].item()
    global_tokens = int(loss_tensor[1].item())
    global_batches = int(loss_tensor[2].item())
    global_weighted_loss = loss_tensor[3].item()

    avg_loss = global_loss / max(global_batches, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    megatron_style_loss = global_weighted_loss / max(global_tokens, 1)
    megatron_ppl = torch.exp(torch.tensor(megatron_style_loss)).item()

    if dist_config.rank == 0:
        logger.info(
            f"[VAL] HF loss={avg_loss:.4f} (ppl={perplexity:.2f}) | "
            f"Megatron loss={megatron_style_loss:.4f} (ppl={megatron_ppl:.2f}) | "
            f"batches={global_batches} tokens={global_tokens}"
        )

    model.train()

    return {
        "val_loss": avg_loss,
        "val_ppl": perplexity,
        "val_loss_megatron": megatron_style_loss,
        "val_ppl_megatron": megatron_ppl,
        "val_tokens": global_tokens,
        "val_batches": global_batches,
    }
