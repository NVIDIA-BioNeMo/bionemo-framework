# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch

from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHold, WarmupAnnealDecayHoldScheduler


def test_warmup_anneal_decay_hold_scheduler_exists():
    scheduler = WarmupAnnealDecayHoldScheduler(warmup_steps=2000, min_lr=4e-5, max_steps=500000, max_lr=4e-4)
    assert scheduler is not None
    assert scheduler.max_steps == 500000
    assert scheduler.warmup_steps == 2000
    assert scheduler.max_lr == 4e-4
    assert scheduler.min_lr == 4e-5


def test_warmup_anneal_decay_hold_works():
    optim = torch.optim.Adam(torch.nn.Linear(10, 1).parameters(), lr=4e-4, weight_decay=0.01, betas=[0.9, 0.98])
    max_lr = 0.1
    min_lr = 0.01
    anneal_percentage = 0.50
    constant_value = anneal_percentage * max_lr
    scheduler = WarmupAnnealDecayHold(
        optimizer=optim,
        warmup_steps=20,
        min_lr=min_lr,
        max_steps=100,
        max_lr=max_lr,
        anneal_percentage=anneal_percentage,
    )

    assert scheduler.get_lr()[0] == min_lr
    # Check initial LR
    for _ in range(20):
        scheduler.step()
    # Check warmup phase
    assert scheduler.get_lr()[0] == max_lr

    # Check decay is lower than max
    for _ in range(20):
        scheduler.step()

    decay_lr = scheduler.get_lr()[0]
    # Check decay is lower than last decay
    assert decay_lr < max_lr

    # Keep decay stepping
    for _ in range(20):
        scheduler.step()

    decay_low = scheduler.get_lr()[0]
    assert decay_low < decay_lr
    assert decay_low == constant_value

    for _ in range(30):
        scheduler.step()

    assert scheduler.get_lr()[0] == constant_value

    # Check hold phase. Run it much longer and confirm
    for _ in range(300):
        scheduler.step()

    assert scheduler.get_lr()[0] == constant_value
