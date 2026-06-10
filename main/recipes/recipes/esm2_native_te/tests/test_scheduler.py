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

"""Tests for get_linear_schedule_with_warmup."""

import pytest
import torch

from scheduler import get_linear_schedule_with_warmup


@pytest.fixture
def optimizer():
    """Create a dummy optimizer for scheduler testing."""
    model = torch.nn.Linear(2, 2)
    return torch.optim.SGD(model.parameters(), lr=1.0)


def test_step_zero(optimizer):
    """Step 0 should have lr=0."""
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    assert scheduler.get_last_lr()[0] == pytest.approx(0.0)


def test_mid_warmup(optimizer):
    """Mid-warmup should have lr ~0.5."""
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    for _ in range(50):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.5, abs=1e-6)


def test_end_of_warmup(optimizer):
    """End of warmup should have lr=1.0."""
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=1000)
    for _ in range(100):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(1.0, abs=1e-6)


def test_mid_decay(optimizer):
    """Mid-decay should be between 1.0 and 0.1."""
    num_warmup = 100
    num_training = 1000
    decay_steps = int(num_training * 0.9)  # 900
    mid_decay_step = (num_warmup + decay_steps) // 2  # 500
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training
    )
    for _ in range(mid_decay_step):
        optimizer.step()
        scheduler.step()
    lr = scheduler.get_last_lr()[0]
    assert 0.1 < lr < 1.0


def test_at_decay_boundary(optimizer):
    """At the decay boundary (90% of training), lr should be ~0.1."""
    num_warmup = 100
    num_training = 1000
    decay_steps = int(num_training * 0.9)  # 900
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training
    )
    for _ in range(decay_steps):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1, abs=1e-6)


def test_past_decay(optimizer):
    """Past the decay boundary, lr should stay at 0.1."""
    num_warmup = 100
    num_training = 1000
    decay_steps = int(num_training * 0.9)  # 900
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training
    )
    for _ in range(decay_steps + 50):
        optimizer.step()
        scheduler.step()
    assert scheduler.get_last_lr()[0] == pytest.approx(0.1, abs=1e-6)


def test_monotonically_decreasing_after_warmup(optimizer):
    """LR should be monotonically decreasing after warmup."""
    num_warmup = 100
    num_training = 1000
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup, num_training_steps=num_training
    )
    # Step through warmup
    for _ in range(num_warmup):
        optimizer.step()
        scheduler.step()

    prev_lr = scheduler.get_last_lr()[0]
    for _ in range(num_training - num_warmup):
        optimizer.step()
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        assert current_lr <= prev_lr + 1e-9
        prev_lr = current_lr
