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

"""CPU unit tests for the engine logic that needs no model load.

These exercise the pure / orchestration parts of ``evo2_sae.core`` — DNA cleaning, organism-tag
resolution, top-feature selection, SAE checkpoint loading, and ``generate`` input guards — on an
*unloaded* engine (``__init__`` only records config; it touches no GPU and imports no Evo2). So
they run in CI even when the GPU/1B path skips, and cover the code paths the model smoke can't.
"""

import pytest
import torch
from evo2_sae import Evo2SAE
from sae.architectures import TopKSAE


def _engine(sae_ckpt_path="unused.pt", **kw):
    """An unloaded engine — __init__ records config only (no GPU, no Evo2/SAE load)."""
    return Evo2SAE(evo2_ckpt_dir="unused", sae_ckpt_path=str(sae_ckpt_path), layer=0, device="cpu", **kw)


# ------------------------------------------------------------------------------- top_features
def test_top_features_ranks_positive_only_and_skips_tag():
    eng = _engine()  # self.labels defaults to {}
    codes = torch.zeros(4, 5)
    codes[:, 2] = 3.0  # strongest over the DNA region
    codes[:, 4] = 1.0  # weaker, still positive
    codes[0, 1] = 9.0  # lives only in row 0, which tag_len=1 skips

    feats = eng.top_features(codes, tag_len=1, k=8)
    ids = [f["feature_id"] for f in feats]
    assert ids == [2, 4]  # ranked by per-base max, positive-only
    assert 1 not in ids  # tag region (row 0) excluded by tag_len
    assert 0 not in ids and 3 not in ids  # zero-activation features dropped
    assert feats[0]["max_activation"] == 3.0 and feats[0]["label"] is None


def test_top_features_empty_codes_returns_empty():
    assert _engine().top_features(torch.zeros(0, 5)) == []


# ------------------------------------------------------------------------------- _load_sae
def _save_topk(path, prefix=""):
    """Write a tiny TopK SAE checkpoint in the format _load_sae expects (optionally DDP-prefixed)."""
    torch.manual_seed(0)
    sae = TopKSAE(input_dim=8, hidden_dim=16, top_k=4)
    state = {prefix + k: v for k, v in sae.state_dict().items()}
    torch.save({"model_config": sae._get_config(), "model_state_dict": state}, path)


def test_load_sae_topk_returns_sae_and_n_features(tmp_path):
    ckpt = tmp_path / "tiny_topk.pt"
    _save_topk(ckpt)
    sae, n_features = _engine(ckpt)._load_sae()
    assert isinstance(sae, TopKSAE) and n_features == 16


def test_load_sae_strips_module_prefix(tmp_path):
    ckpt = tmp_path / "ddp_topk.pt"
    _save_topk(ckpt, prefix="module.")  # as saved under DDP
    sae, n_features = _engine(ckpt)._load_sae()
    assert isinstance(sae, TopKSAE) and n_features == 16


# ------------------------------------------------------------------------------- generate input guards
def test_generate_rejects_unknown_organism():
    with pytest.raises(ValueError, match="organism"):
        _engine().generate(prompt="ACGT", organism="Klingon")


def test_generate_rejects_empty_prompt():
    # "None (raw DNA)" has an empty tag, so an empty prompt leaves nothing to seed generation.
    with pytest.raises(ValueError):
        _engine().generate(prompt="", organism="None (raw DNA)")
