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


def test_check_dim_rejects_sae_model_mismatch():
    # an SAE whose input_dim != the model's hidden size would matmul-fail on the first encode;
    # load() catches it with a clear error instead. (hidden=None -> unknown -> skip, never blocks.)
    with pytest.raises(ValueError, match="does not match"):
        Evo2SAE._check_dim(sae_input_dim=4096, hidden=1920, layer=26)
    Evo2SAE._check_dim(sae_input_dim=1920, hidden=1920, layer=26)  # match -> no raise
    Evo2SAE._check_dim(sae_input_dim=1920, hidden=None, layer=26)  # unknown -> skip


def test_encode_batch_length_bucketing_preserves_order():
    # encode_batch sorts work by length (bucketing) but must return results in INPUT order.
    # Stub the model so this runs on CPU: each sequence's first token carries its (distinct) length
    # as a marker, _forward_hidden echoes it, and the SAE is identity — so out[i] should carry the
    # marker of seqs[i] regardless of the internal length-sort.
    import types

    eng = _engine()
    seqs = ["AC", "ACGTACGT", "A" * 20, "ACG", "A" * 15]  # lengths 2,8,20,3,15 (distinct markers)
    eng.tokenize = lambda s: [len(s)] + [0] * (len(s) - 1)
    eng._forward_hidden = lambda id_lists: [torch.tensor([[float(ids[0])]]) for ids in id_lists]
    eng.sae = types.SimpleNamespace(encode=lambda h: h)
    eng.n_features = 1
    out = eng.encode_batch(seqs, batch_size=2)
    assert len(out) == len(seqs)
    assert [int(o[0, 0].item()) for o in out] == [len(s) for s in seqs]  # input order preserved


# ------------------------------------------------------------------------------- generate input guards
def test_generate_rejects_unknown_organism():
    with pytest.raises(ValueError, match="organism"):
        _engine().generate(prompt="ACGT", organism="Klingon")


def test_generate_rejects_empty_prompt():
    # "None (raw DNA)" has an empty tag, so an empty prompt leaves nothing to seed generation.
    with pytest.raises(ValueError):
        _engine().generate(prompt="", organism="None (raw DNA)")


def test_generate_rejects_overlong_prompt():
    # An over-context prompt is rejected (server -> 413), not silently truncated by tokenize().
    with pytest.raises(ValueError, match="too long"):
        _engine(max_seq_len=16).generate(prompt="A" * 32, organism="None (raw DNA)")


# ------------------------------------------------------------------------------- rename persistence
def test_set_label_persists_and_reverts(tmp_path, monkeypatch):
    monkeypatch.delenv("FEATURE_RENAMES", raising=False)  # honor the sidecar-next-to-annotations path
    eng = _engine(feature_annotations=str(tmp_path / "annotations.parquet"))
    eng.labels = {0: "base0"}
    eng._base_labels = {0: "base0"}
    eng.renames = {}

    # relabel a labeled feature + name an unlabeled one -> in-memory AND persisted to the sidecar
    assert eng.set_label(0, "renamed0") == "renamed0"
    assert eng.set_label(3, "new3") == "new3"
    assert eng.labels == {0: "renamed0", 3: "new3"}
    assert eng._load_renames() == {0: "renamed0", 3: "new3"}  # survives a fresh read from disk

    # blank reverts: feature 0 back to its base label; feature 3 (no base) becomes unlabeled — persisted
    assert eng.set_label(0, "  ") == "base0"
    assert eng.set_label(3, "") is None
    assert eng.labels == {0: "base0"}
    assert eng._load_renames() == {}


def test_load_renames_ignores_corrupt_sidecar(tmp_path, monkeypatch):
    monkeypatch.delenv("FEATURE_RENAMES", raising=False)
    eng = _engine(feature_annotations=str(tmp_path / "annotations.parquet"))
    eng._renames_path().write_text("{ not valid json")
    assert eng._load_renames() == {}  # a bad sidecar is ignored, never fatal
