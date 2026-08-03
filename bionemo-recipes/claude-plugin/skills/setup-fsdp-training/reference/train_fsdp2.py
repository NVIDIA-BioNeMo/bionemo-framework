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

"""Reference: FSDP2 training script for TransformerEngine models.

Trimmed and annotated version showing the essential distributed training patterns.
Dataset-specific code and logging removed for clarity.

Source: bionemo-recipes/recipes/esm2_native_te/train_fsdp2.py

Sequence:
1. Initialize distributed process group
2. Create device mesh
3. Create model (optionally on meta device)
4. Apply FSDP sharding to individual layers, then full model
5. Materialize meta-device parameters
6. Create optimizer (MUST be after FSDP wrapping)
7. Training loop with grad clipping and checkpointing
8. Clean up
"""

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.optim import AdamW


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Distributed configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DistributedConfig:
    """Reads RANK, LOCAL_RANK, WORLD_SIZE from env vars set by torchrun."""

    rank: int = field(default_factory=lambda: int(os.environ.setdefault("RANK", "0")))
    local_rank: int = field(default_factory=lambda: int(os.environ.setdefault("LOCAL_RANK", "0")))
    world_size: int = field(default_factory=lambda: int(os.environ.setdefault("WORLD_SIZE", "1")))

    def is_main_process(self) -> bool:
        """Return True if this is the main (rank 0) process."""
        return self.rank == 0


def train(config, model_cls, get_layer_fn, train_dataloader, num_train_steps, use_meta_device=True):
    """Main training function demonstrating the FSDP2 pattern.

    Args:
        config: Model config (e.g., NVEsmConfig).
        model_cls: Model class (e.g., NVEsmForMaskedLM).
        get_layer_fn: Function that extracts transformer layers from the model.
                      E.g., lambda m: m.esm.encoder.layers (encoder)
                      or    lambda m: m.model.layers (decoder)
        train_dataloader: DataLoader yielding batches.
        num_train_steps: Total training steps.
        use_meta_device: Whether to use meta device init (recommended for large models).
    """
    # --- Step 1: Initialize distributed ---
    dist_config = DistributedConfig()
    device = torch.device(f"cuda:{dist_config.local_rank}")
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.cuda.set_device(dist_config.local_rank)

    # --- Step 2: Create device mesh ---
    # 1D mesh = pure data parallelism. For FSDP+CP, use 2D: (dp_size, cp_size).
    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(dist_config.world_size,),
        mesh_dim_names=("dp",),
    )

    # --- Step 3: Create model (optionally on meta device) ---
    # Meta device creates parameter metadata without allocating GPU memory.
    # Parameters are materialized after FSDP sharding.
    with torch.device("meta") if use_meta_device else nullcontext():
        model = model_cls(config)

    # --- Step 4: Apply FSDP sharding ---
    # Mixed precision: FP32 master weights, BF16 compute.
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=False,
    )

    # CRITICAL: Shard individual layers FIRST, then the full model.
    # This makes each layer an independent FSDP unit for better
    # communication/computation overlap.
    transformer_layers = get_layer_fn(model)
    for layer in transformer_layers:
        fully_shard(layer, mesh=device_mesh["dp"], mp_policy=mp_policy)
    fully_shard(model, mesh=device_mesh["dp"], mp_policy=mp_policy)

    # --- Step 5: Materialize meta-device parameters ---
    # MUST happen after FSDP sharding but BEFORE optimizer creation.
    if use_meta_device:
        model.init_empty_weights()  # TE layers use reset_parameters() internally

    # --- Step 6: Create optimizer AFTER FSDP wrapping ---
    # FSDP replaces original parameters with DTensor shards.
    optimizer = AdamW(model.parameters(), lr=4e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.01)

    # --- Step 7: Training loop ---
    step = 0
    while step < num_train_steps:
        for batch in train_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward
            outputs = model(**batch)

            # Backward
            loss = outputs.loss
            loss.backward()

            # Gradient clipping (works across FSDP shards)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()

            if dist_config.is_main_process() and step % 100 == 0:
                logger.info(f"Step {step}, Loss: {loss.item():.4f}")

            step += 1
            if step >= num_train_steps:
                break

    # --- Step 8: Clean up ---
    torch.distributed.destroy_process_group()
