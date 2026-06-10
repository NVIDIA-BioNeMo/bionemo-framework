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


def linear_scheduler_with_warmup_exp_decay(
    iteration, total_iterations, initial_lr, warmup_iterations=1000, min_lr=1e-6
):
    """Linearly warm up the learning rate, then apply exponential decay.

    The scheduler increases the LR linearly for `warmup_iterations`, then decays
    exponentially to `min_lr` over the remaining iterations.

    Args:
        iteration: Current iteration (0-indexed).
        total_iterations: Total number of iterations for the schedule.
        initial_lr: Initial learning rate at peak after warmup.
        warmup_iterations: Number of warmup iterations.
        min_lr: Minimum learning rate floor.

    Returns:
        The learning rate for the given iteration.
    """
    decay_factor = (min_lr / initial_lr) ** (1 / (total_iterations - warmup_iterations))

    if iteration < warmup_iterations:
        lr = initial_lr * (iteration + 1) / warmup_iterations
    else:
        lr = initial_lr * (decay_factor ** (iteration - warmup_iterations))

    if lr < min_lr:
        lr = min_lr

    return lr


def linear_scheduler_with_warmup_lr_lambda(iteration: int, warmup_iterations: int, total_iterations: int):
    """Linear warmup followed by linear decay lambda.

    Mirrors the schedule used in Transformers' optimization utilities.

    Args:
        iteration: Current iteration (0-indexed).
        warmup_iterations: Number of warmup iterations.
        total_iterations: Total number of iterations.

    Returns:
        A multiplicative factor in [0, 1] to scale the base learning rate.
    """
    if iteration < warmup_iterations:
        return float(iteration) / float(max(1, warmup_iterations))
    return max(0.0, float(total_iterations - iteration) / float(max(1, total_iterations - warmup_iterations)))
