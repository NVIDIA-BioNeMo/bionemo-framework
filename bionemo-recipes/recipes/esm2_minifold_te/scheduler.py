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

from torch.optim.lr_scheduler import LambdaLR


def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=2_000,
    num_training_steps=500_000,
    last_epoch=-1,
):
    """Linear warmup and decay scheduler for ESM-2 pretraining.

    The description from Lin 2022 is: The learning rate is warmed up over the first 2,000 steps
    to a peak value of 4e-4 (1.6e-4 for the 15B parameter model), and then linearly decayed to
    one tenth of its peak value over the 90% of training duration. We've found internally that a
    longer warmup helps convergence for larger models (3B+) with bf16 precision.
    """
    decay_steps = int(num_training_steps * 0.9)

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Warmup phase: linearly increase learning rate
            return float(current_step) / float(max(1, num_warmup_steps))
        # Decay phase: linearly decay to one tenth of peak over 90% of training
        elif current_step > decay_steps:
            return 0.1  # one tenth of peak learning rate after decay period
        else:
            # Linear decay from 1.0 to 0.1 over decay_steps-num_warmup_steps
            return 1.0 - 0.9 * (current_step - num_warmup_steps) / float(max(1, decay_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda, last_epoch)
