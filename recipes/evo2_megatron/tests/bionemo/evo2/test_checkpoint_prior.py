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

"""Tests for ``bionemo.evo2.utils.checkpoint.checkpoint_prior``."""

import torch

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.models.evo2_provider import HyenaTestModelProvider
from bionemo.evo2.utils.checkpoint.checkpoint_prior import (
    _decay_prior_stats,
    _gamma_prior_stats,
    _p_prior_stats,
    load_filter_tensors,
)
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import package_mbridge_checkpoint
from bionemo.evo2.utils.checkpoint.vortex_to_mbridge import _compute_initial_medium_decay


def test_gamma_prior_stats_reports_init_range_membership():
    """Gamma stats should compare against log(U(gamma_min, gamma_max)) support."""
    gamma = torch.tensor([0.01, 0.05, 0.1], dtype=torch.float32).log()

    stats = _gamma_prior_stats([gamma], gamma_min=0.01, gamma_max=0.1)

    assert stats["fraction_inside_init_range"] == 1.0
    assert stats["mean_distance_outside_range"] == 0.0
    assert stats["max_distance_outside_range"] == 0.0


def test_p_prior_stats_reports_distance_from_negative_one():
    """p initializes to -1 in the implicit modal filter."""
    stats = _p_prior_stats([torch.tensor([-1.0, -0.5, -1.5], dtype=torch.float32)])

    assert stats["init_value"] == -1.0
    assert stats["median_abs_distance_to_init"] == 0.5


def test_decay_prior_stats_matches_weak_initialization():
    """Decay stats should report zero distance for the deterministic initialization."""
    expected = _compute_initial_medium_decay(4, 8, decay_preset="weak")

    stats = _decay_prior_stats([expected], decay_preset="weak")

    assert stats["per_tensor"][0]["rmse_to_init"] == 0.0
    assert stats["per_tensor"][0]["relative_rmse_to_init"] == 0.0
    assert stats["per_tensor"][0]["max_abs_distance_to_init"] == 0.0


def test_packaged_checkpoint_filter_tensors_are_loadable(tmp_path):
    """The shared writer should produce DCP that prior analysis can resolve and load."""
    expected = {
        "decoder.layers.0.mixer.mixer.filter.p": torch.tensor([[-1.0], [-0.5]], dtype=torch.float32),
        "decoder.layers.0.mixer.mixer.filter.gamma": torch.tensor([[-2.0], [-1.5]], dtype=torch.float32),
        "decoder.layers.0.mixer.mixer.filter.R": torch.tensor([[0.25], [0.75]], dtype=torch.float32),
    }
    state_dict = {
        **expected,
        "decoder.layers.0.mixer.dense.weight": torch.ones(2, 2, dtype=torch.bfloat16),
    }
    provider = HyenaTestModelProvider(
        hybrid_override_pattern="H",
        hidden_size=2,
        num_groups_hyena=2,
        num_attention_heads=1,
        ffn_hidden_size=4,
    )

    checkpoint_dir = package_mbridge_checkpoint(
        state_dict,
        mbridge_ckpt_dir=tmp_path / "mbridge",
        model_provider=provider,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
    )
    loaded = load_filter_tensors(checkpoint_dir)

    assert set(loaded) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(loaded[key], tensor)
