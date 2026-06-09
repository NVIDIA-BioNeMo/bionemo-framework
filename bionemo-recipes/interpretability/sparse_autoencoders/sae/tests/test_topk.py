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

"""Tests for TopKSAE training-quality options: loss reduction + global dead-latent counting."""

import torch
from sae.architectures import topk as topk_mod
from sae.architectures.topk import TopKSAE


def _make_sae(**kw):
    torch.manual_seed(0)
    return TopKSAE(input_dim=8, hidden_dim=16, top_k=4, normalize_input=False, **kw)


def test_recon_loss_aggregate_matches_batch_fvu():
    """aggregate_loss=True equals the batch-level FVU mse.mean()/var.mean()."""
    x = torch.randn(32, 8)
    sae = _make_sae(aggregate_loss=True)
    recon = sae.forward_with_aux(x)["recon"]
    expected = (recon - x).pow(2).mean() / ((x - sae.pre_bias).pow(2).mean() + 1e-8)
    assert torch.allclose(sae.loss(x)["total"], expected)


def test_dead_latent_count_global_vs_local(monkeypatch):
    """dead_count_global advances the inactivity counter by tokens x world_size; else local."""
    # Pretend we're in a 4-rank distributed run.
    monkeypatch.setattr(topk_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(topk_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(topk_mod.dist, "get_world_size", lambda: 4)

    codes = torch.zeros(10, 16)
    codes[:, 0] = 1.0  # only latent 0 fires

    g = _make_sae(dead_count_global=True)
    g.stats_last_nonzero.zero_()
    g._update_dead_latent_stats(codes)
    assert int(g.stats_last_nonzero[0]) == 0  # fired -> reset
    assert int(g.stats_last_nonzero[1]) == 10 * 4  # inactive -> tokens x world_size

    loc = _make_sae(dead_count_global=False)
    loc.stats_last_nonzero.zero_()
    loc._update_dead_latent_stats(codes)
    assert int(loc.stats_last_nonzero[1]) == 10  # inactive -> local micro-batch only


def test_opted_in_options_round_trip_through_config():
    """Opted-in (non-default) options serialize in the checkpoint config so a reload keeps them."""
    cfg = _make_sae(aggregate_loss=True, dead_count_global=True)._get_config()
    assert cfg["aggregate_loss"] is True
    assert cfg["dead_count_global"] is True
