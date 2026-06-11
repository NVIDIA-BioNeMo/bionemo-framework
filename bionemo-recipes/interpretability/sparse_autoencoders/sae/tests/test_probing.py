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

"""CPU tests for sae.eval.probing: the metrics reproduce on synthetic data (no model/GPU)."""

import numpy as np
import torch
from sae.eval.probing import ActivationBuffer, auroc_all, split_indices, standardize


def test_auroc_separates_predictive_from_noise():
    """A feature equal to the label scores ~1.0 AUROC; a random feature scores near chance."""
    torch.manual_seed(0)
    n = 400
    y = (torch.arange(n) % 2).float()
    predictive = y + torch.randn(n) * 0.01  # near-perfect detector
    noise = torch.randn(n)  # uninformative
    x = torch.stack([predictive, noise], dim=1)  # [N, 2 features]
    au = auroc_all(x, y.unsqueeze(1))  # [2 features, 1 label]
    assert float(au[0, 0]) > 0.99
    assert float(au[1, 0]) < 0.7


def test_split_indices_disjoint_and_complete():
    """Train/test indices partition range(n) with the requested test fraction."""
    tr, te = split_indices(100, test_frac=0.4, seed=0)
    tr_s, te_s = set(tr.tolist()), set(te.tolist())
    assert tr_s.isdisjoint(te_s)
    assert tr_s | te_s == set(range(100))
    assert 0.35 < len(te_s) / 100 < 0.45


def test_standardize_zero_means_train_split():
    """standardize() returns train-split mean/std that center the train rows."""
    torch.manual_seed(0)
    x = torch.randn(100, 5) * 3 + 7
    tr = torch.arange(80)
    mu, sd = standardize(x, tr)
    z = (x[tr] - mu) / sd
    assert torch.allclose(z.mean(0), torch.zeros(5), atol=1e-4)


def test_activation_buffer_roundtrip(tmp_path):
    """ActivationBuffer save/load preserves codes, labels, names (+ name_idx mapping)."""
    rng = np.random.default_rng(0)
    codes = rng.random((10, 4)).astype(np.float16)
    labels = np.tile(np.array([True, False]), (10, 1))
    buf = ActivationBuffer(codes=codes, labels=labels, label_names=["motif_atg", "is_prok"])
    path = str(tmp_path / "buf.npz")
    buf.save(path)

    loaded = ActivationBuffer.load(path)
    assert np.array_equal(loaded.codes, codes)
    assert [str(n) for n in loaded.label_names] == ["motif_atg", "is_prok"]
    assert loaded.name_idx["is_prok"] == 1
