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

import math

from torch.optim.lr_scheduler import LambdaLR


def get_cosine_annealing_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2_000,
    num_decay_steps=500_000,
    min_lr_ratio=0.0,
    last_epoch=-1,
):
    """Cosine annealing scheduler with warmup."""
    max_lr = optimizer.param_groups[0]["lr"]
    min_lr = max_lr * min_lr_ratio

    def lr_lambda(current_step: int):
        if num_warmup_steps > 0 and current_step <= num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_warmup_steps + num_decay_steps:
            return min_lr_ratio
        num_steps_ = current_step - num_warmup_steps
        decay_ratio = float(num_steps_) / float(num_decay_steps)
        delta_lr = max_lr - min_lr
        coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        actual_lr = min_lr + coeff * delta_lr
        return actual_lr / max_lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)
