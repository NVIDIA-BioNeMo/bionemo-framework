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

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_annealing_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2_000,
    num_decay_steps=500_000,
    min_lr_ratio=0.0,
    last_epoch=-1,
):
    """Cosine annealing scheduler with warmup.

    The learning rate is linearly warmed up from 0 to max_lr over num_warmup_steps,
    then follows a cosine annealing schedule from max_lr to min_lr over num_decay_steps.
    After warmup_steps + decay_steps, the learning rate remains at min_lr.

    Args:
        optimizer: The optimizer to schedule.
        num_warmup_steps: Number of warmup steps.
        num_decay_steps: Number of decay steps after warmup.
        min_lr_ratio: Minimum learning rate as a ratio of the initial learning rate.
            If 0.0, decays to 0. Otherwise, decays to max_lr * min_lr_ratio.
        last_epoch: The index of the last epoch. Default: -1.
    """
    # Get the initial learning rate (max_lr) from the optimizer
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr * min_lr_ratio

    def lr_lambda(current_step: int):
        if num_warmup_steps > 0 and current_step <= num_warmup_steps:
            # Warmup phase: linearly increase learning rate from 0 to max_lr
            # LambdaLR multiplies by this value, so return step/warmup_steps
            return float(current_step) / float(max(1, num_warmup_steps))
        # For any steps larger than warmup_steps + decay_steps, use min_lr
        if current_step > num_warmup_steps + num_decay_steps:
            # Return multiplier that gives min_lr when multiplied by max_lr
            return min_lr_ratio
        # Cosine annealing phase: decay from max_lr to min_lr
        num_steps_ = current_step - num_warmup_steps
        decay_steps_ = num_decay_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        delta_lr = max_lr - min_lr
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        # Calculate the actual LR, then convert to multiplier
        actual_lr = min_lr + coeff * delta_lr
        return actual_lr / max_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
