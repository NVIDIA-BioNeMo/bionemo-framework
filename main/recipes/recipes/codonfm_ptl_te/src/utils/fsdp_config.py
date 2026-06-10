# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import logging
from functools import partial
from typing import Set

import torch
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformer_engine.pytorch import TransformerLayer as TETransformerLayer

from src.models.components.encodon_layer import EncoderLayer
from src.models.components.encodon_te_layer import EncodonTELayer


logging = logging.getLogger(__name__)


def get_fsdp_strategy(
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    use_fsdp2: bool = True,
) -> FSDPStrategy:
    """Create a properly configured FSDP/FSDP2 strategy for EnCodonTE models.

    This configuration ensures FSDP uses LESS memory than DDP by:
    1. Sharding model parameters across GPUs (each GPU holds only a fraction)
    2. Using per-layer auto-wrapping for efficient memory management
    3. Enabling FSDP2 optimizations (PyTorch 2.1+) for reduced memory overhead

    Args:
        cpu_offload: Whether to offload parameters to CPU when not in use
        activation_checkpointing: Whether to use activation checkpointing to reduce memory
        use_fsdp2: Whether to use FSDP2 (PyTorch 2.1+) for better performance

    Returns:
        Configured FSDPStrategy instance that uses less memory than DDP
    """
    # Define which layer types should be wrapped as FSDP units
    # This allows each transformer layer to be sharded independently
    transformer_layer_classes: Set[type] = {EncodonTELayer, TETransformerLayer, EncoderLayer}

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_layer_classes,
    )

    # Check PyTorch version for FSDP2 support
    torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
    fsdp2_available = torch_version >= (2, 1)

    if use_fsdp2 and not fsdp2_available:
        logging.warning(
            f"Warning: FSDP2 requested but PyTorch version {torch.__version__} < 2.1. Falling back to FSDP1."
        )
        use_fsdp2 = False

    # FSDP2 uses different configuration for better memory efficiency
    if use_fsdp2:
        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=transformer_layer_classes if activation_checkpointing else None,
            cpu_offload=cpu_offload,
            # FSDP2-specific optimizations
            sharding_strategy="FULL_SHARD",  # Each GPU holds 1/N of parameters
            state_dict_type="sharded",
            # FSDP2 improvements that reduce memory usage
            use_orig_params=True,  # Better memory management and optimizer compatibility
            limit_all_gathers=True,  # Reduce memory spikes during forward pass
        )
    else:
        logging.info("Using FSDP1 (consider upgrading to PyTorch 2.1+ for FSDP2 benefits)")
        # FSDP1 configuration
        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=transformer_layer_classes if activation_checkpointing else None,
            cpu_offload=cpu_offload,
            sharding_strategy="FULL_SHARD",
            state_dict_type="sharded",
        )

    return strategy
