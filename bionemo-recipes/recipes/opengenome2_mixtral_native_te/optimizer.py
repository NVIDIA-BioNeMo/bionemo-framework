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

"""Optimizer utilities for OpenGenome2 training."""

import logging

import torch


logger = logging.getLogger(__name__)


def get_parameter_groups_with_weight_decay(
    model: torch.nn.Module,
    weight_decay: float,
    skip_embeddings: bool = False,
) -> list[dict]:
    """Create optimizer parameter groups with Megatron-style weight decay filtering."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        should_skip_decay = name.endswith(".bias") or param.dim() == 1 or (skip_embeddings and "embed" in name.lower())

        if should_skip_decay:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    logger.info(
        "Weight decay groups: %d params with decay, %d params without decay",
        len(decay_params),
        len(no_decay_params),
    )

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
