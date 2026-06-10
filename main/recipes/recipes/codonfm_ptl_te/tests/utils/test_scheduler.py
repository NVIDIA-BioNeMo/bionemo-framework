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


import pytest

from src.utils.scheduler import (
    linear_scheduler_with_warmup_exp_decay,
    linear_scheduler_with_warmup_lr_lambda,
)


class TestLinearSchedulerWithWarmupExpDecay:
    def test_warmup_phase(self):
        lr = linear_scheduler_with_warmup_exp_decay(
            iteration=500,
            total_iterations=10000,
            initial_lr=1e-4,
            warmup_iterations=1000,
            min_lr=1e-6,
        )
        assert lr == pytest.approx(1e-4 * 501 / 1000)

    def test_decay_phase(self):
        lr_after_warmup = linear_scheduler_with_warmup_exp_decay(
            iteration=1001,  # Changed from 1000 to 1001 to be in decay phase
            total_iterations=10000,
            initial_lr=1e-4,
            warmup_iterations=1000,
            min_lr=1e-6,
        )
        assert lr_after_warmup < 1e-4

        lr_middle = linear_scheduler_with_warmup_exp_decay(
            iteration=5000,
            total_iterations=10000,
            initial_lr=1e-4,
            warmup_iterations=1000,
            min_lr=1e-6,
        )
        assert lr_middle < lr_after_warmup
        assert lr_middle > 1e-6

    def test_min_lr(self):
        lr = linear_scheduler_with_warmup_exp_decay(
            iteration=9999,
            total_iterations=10000,
            initial_lr=1e-4,
            warmup_iterations=1000,
            min_lr=1e-6,
        )
        assert lr >= 1e-6

        # check if it gets clamped
        # with very high decay it should hit min_lr early
        lr_clamped = linear_scheduler_with_warmup_exp_decay(
            iteration=5000,
            total_iterations=10000,
            initial_lr=1e-2,  # high lr
            warmup_iterations=1000,
            min_lr=1e-3,
        )
        # Manually calculate decay
        decay_factor = (1e-3 / 1e-2) ** (1 / (10000 - 1000))
        expected_lr = 1e-2 * (decay_factor ** (5000 - 1000))
        assert lr_clamped == expected_lr

        lr_at_end = linear_scheduler_with_warmup_exp_decay(
            iteration=10000,
            total_iterations=10000,
            initial_lr=1.0,
            warmup_iterations=0,
            min_lr=0.1,
        )
        assert lr_at_end == pytest.approx(0.1)


class TestLinearSchedulerWithWarmupLRLambda:
    def test_warmup_phase(self):
        # at half of warmup
        val = linear_scheduler_with_warmup_lr_lambda(iteration=500, warmup_iterations=1000, total_iterations=10000)
        assert val == pytest.approx(0.5)

        # at the end of warmup
        val = linear_scheduler_with_warmup_lr_lambda(iteration=1000, warmup_iterations=1000, total_iterations=10000)
        assert val == pytest.approx(1.0)

    def test_decay_phase(self):
        # right after warmup
        val = linear_scheduler_with_warmup_lr_lambda(iteration=1000, warmup_iterations=1000, total_iterations=10000)
        assert val == pytest.approx(1.0)

        # in the middle of decay
        val = linear_scheduler_with_warmup_lr_lambda(iteration=5500, warmup_iterations=1000, total_iterations=10000)
        # total_iterations - warmup = 9000
        # total_iterations - iteration = 4500
        # 4500 / 9000 = 0.5
        assert val == pytest.approx(0.5)

    def test_end_of_training(self):
        val = linear_scheduler_with_warmup_lr_lambda(iteration=10000, warmup_iterations=1000, total_iterations=10000)
        assert val == pytest.approx(0.0)
