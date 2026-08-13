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

import pytest
import torch
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn


def _loss(*, token_level_loss: bool) -> float:
    token_mask = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 0, 0, 0]], dtype=torch.float32)
    advantages = torch.tensor([[0, 1, 1, 1, 1], [0, -1, -1, -1, -1]], dtype=torch.float32)
    zeros = torch.zeros_like(advantages)
    data = {
        "advantages": advantages,
        "prev_logprobs": zeros,
        "generation_logprobs": zeros,
        "reference_policy_logprobs": zeros,
        "token_mask": token_mask,
        "sample_mask": torch.ones(2),
    }
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            token_level_loss=token_level_loss,
            reference_policy_kl_penalty=0,
            force_on_policy_ratio=True,
        )
    )
    loss, _ = loss_fn(
        torch.zeros((2, 4)),
        data,
        global_valid_seqs=torch.tensor(2.0),
        global_valid_toks=torch.tensor(5.0),
    )
    return loss.item()


def test_sequence_loss_does_not_discount_short_negative_response():
    assert _loss(token_level_loss=True) == pytest.approx(-0.6)
    assert _loss(token_level_loss=False) == pytest.approx(0.0)
