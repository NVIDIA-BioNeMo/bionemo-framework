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

"""End-to-end CPU integration tests for the probing harness (no model).

These cover the *seams* between the eval layers that per-layer unit tests miss:

  * ``evo2_sae.eval.probing`` (#1629) <-> the probe CLI (#1636): a buffer is written by
    ``ActivationBuffer.save``, reloaded, and scored through the real ``probe.main()``
    dispatch (``auroc`` / ``annotate`` / ``linear``). This is the path that silently
    broke when ``save`` dropped the dense twin — the ``linear`` SAE-vs-dense comparison
    only renders if ``dense`` survives the round trip.
  * ``annot_tracks.label_windows`` (#1630) <-> ``domain_f1`` (#1629): interval tracks ->
    per-token mask + instance ids -> instance-level F1, with no model in the loop.

A feature is *planted* to track one concept so the metrics have a known right answer;
everything else is noise, so a green run means the whole pipeline carried the signal.
"""

import contextlib
import sys

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from evo2_sae.eval.probing import ActivationBuffer, auroc_all, domain_f1, probe
from evo2_sae.eval.probing.annot_tracks import label_windows
from evo2_sae.eval.probing.evo2_buffer import KINGDOM_TAGS, build_buffer, forward_codes
from sae.architectures import TopKSAE


# Feature index deliberately planted to track the "planted" concept.
_PLANTED_FEATURE = 3


def _planted_buffer(path, n=400, n_features=6, hidden=8, seed=0, with_dense=True):
    """Write a synthetic buffer where ``_PLANTED_FEATURE`` cleanly separates the "planted" label.

    ``with_dense=True`` is a point buffer (dense twin, scored by AUROC); ``with_dense=False`` drops
    the dense twin so ``annotate`` routes it through domain-F1 (region buffer). Either way it carries
    per-concept instance ids so the save/load round trip is exercised. Returns the in-memory buffer.
    """
    rng = np.random.default_rng(seed)
    label_names = ["planted", "noise"]
    labels = np.zeros((n, len(label_names)), dtype=bool)
    labels[:, 0] = rng.random(n) < 0.4  # planted positives (~40%)
    labels[:, 1] = rng.random(n) < 0.4  # uncorrelated noise concept

    codes = (rng.random((n, n_features)) * 0.1).astype(np.float16)  # background noise
    pos = labels[:, 0]
    codes[pos, _PLANTED_FEATURE] = (rng.random(int(pos.sum())) * 5 + 5).astype(np.float16)  # strong, positives only

    dense = (rng.random((n, hidden))).astype(np.float16) if with_dense else None
    # one instance per run of 5 positive tokens; -1 outside the concept
    inst = np.full(n, -1, np.int32)
    inst[pos] = (np.cumsum(pos)[pos] - 1) // 5
    instances = {"planted": inst}

    buf = ActivationBuffer(codes=codes, labels=labels, label_names=label_names, dense=dense, instances=instances)
    buf.save(str(path))
    return buf


def _run_cli(monkeypatch, *argv):
    """Drive the real probe.py CLI (arg-parse -> dispatch) the way a user would."""
    monkeypatch.setattr(sys, "argv", ["probe.py", *argv])
    probe.main()


# --------------------------------------------------------------- probing <-> CLI buffer seam
def test_buffer_roundtrip_preserves_dense_and_instances(tmp_path):
    """save() -> load() must carry the dense twin + instance ids (regression: they were dropped)."""
    buf = _planted_buffer(tmp_path / "buf.npz")
    lo = ActivationBuffer.load(str(tmp_path / "buf.npz"))
    assert np.array_equal(lo.codes, buf.codes) and np.array_equal(lo.labels, buf.labels)
    assert lo.dense is not None and np.array_equal(lo.dense, buf.dense)
    assert lo.instances is not None and np.array_equal(lo.instances["planted"], buf.instances["planted"])


def test_annotate_cli_labels_planted_feature(tmp_path, monkeypatch, capsys):
    """``annotate`` over a reloaded buffer writes a parquet that labels the planted feature."""
    _planted_buffer(tmp_path / "buf.npz")
    out = tmp_path / "feature_annotations.parquet"
    _run_cli(
        monkeypatch,
        "annotate",
        "--acts",
        str(tmp_path / "buf.npz"),
        "--out",
        str(out),
        "--min-auroc",
        "0.85",
        "--device",
        "cpu",
    )
    tbl = pq.read_table(out).to_pydict()
    assert set(tbl) == {"feature_id", "label", "method", "score", "auroc", "activation_freq", "max_activation"}
    rows = {fid: (lab, m, au) for fid, lab, m, au in zip(tbl["feature_id"], tbl["label"], tbl["method"], tbl["auroc"])}
    assert _PLANTED_FEATURE in rows, "planted feature not annotated"
    label, method, auroc = rows[_PLANTED_FEATURE]
    assert label == "planted" and method == "auroc" and auroc >= 0.85


def test_annotate_cli_domain_f1_labels_region_feature(tmp_path, monkeypatch):
    """A region buffer (instance ids, no dense twin) routes ``annotate`` through domain-F1."""
    _planted_buffer(tmp_path / "region.npz", with_dense=False)
    out = tmp_path / "region_annotations.parquet"
    _run_cli(
        monkeypatch,
        "annotate",
        "--acts",
        str(tmp_path / "region.npz"),
        "--out",
        str(out),
        "--min-f1",
        "0.5",
        "--device",
        "cpu",
    )
    tbl = pq.read_table(out).to_pydict()
    rows = {fid: (lab, m, sc) for fid, lab, m, sc in zip(tbl["feature_id"], tbl["label"], tbl["method"], tbl["score"])}
    assert _PLANTED_FEATURE in rows, "planted feature not annotated by domain-F1"
    label, method, score = rows[_PLANTED_FEATURE]
    assert label == "planted" and method == "domain_f1" and score >= 0.5


def test_annotate_cli_merges_point_and_region(tmp_path, monkeypatch):
    """Multiple --acts merge into one parquet; AUROC wins over F1 for a feature scored by both."""
    _planted_buffer(tmp_path / "point.npz", with_dense=True)  # -> AUROC
    _planted_buffer(tmp_path / "region.npz", with_dense=False)  # -> domain-F1
    out = tmp_path / "merged.parquet"
    _run_cli(
        monkeypatch,
        "annotate",
        "--acts",
        str(tmp_path / "point.npz"),
        str(tmp_path / "region.npz"),
        "--out",
        str(out),
        "--min-auroc",
        "0.85",
        "--min-f1",
        "0.5",
        "--device",
        "cpu",
    )
    tbl = pq.read_table(out).to_pydict()
    method = dict(zip(tbl["feature_id"], tbl["method"]))
    assert method[_PLANTED_FEATURE] == "auroc", "AUROC should win the tie over domain-F1"


def test_region_buffer_helper_stacks_masks_and_rekeys_instances(tmp_path):
    """_region_buffer -> dense-free region buffer; euk-f1 pairs the cds mask with gene instances."""
    p, f = 12, 5
    code_buf = torch.rand(p, f, dtype=torch.float16)
    lab = {k: torch.zeros(p, dtype=torch.bool) for k in ("exon", "intron", "cds")}
    lab["exon"][2:6] = True
    lab["cds"][3:6] = True
    lab["intron"][7:9] = True
    inst = {k: torch.full((p,), -1, dtype=torch.long) for k in ("exon", "intron", "gene")}
    inst["exon"][2:6], inst["gene"][2:9], inst["intron"][7:9] = 0, 0, 0
    buf = probe._region_buffer(code_buf, lab, inst, ("exon", "intron", "cds"), ("exon", "intron", "gene"))
    assert buf.dense is None and list(buf.label_names) == ["exon", "intron", "cds"]
    assert set(buf.instances) == {"exon", "intron", "cds"}
    assert np.array_equal(buf.instances["cds"], inst["gene"].numpy().astype(np.int32))  # cds mask <- gene instances
    buf.save(str(tmp_path / "r.npz"))  # round trips as a real region buffer
    lo = ActivationBuffer.load(str(tmp_path / "r.npz"))
    assert lo.dense is None and lo.labels.shape == (p, 3)
    assert np.array_equal(lo.instances["cds"], inst["gene"].numpy().astype(np.int32))


def test_auroc_cli_recovers_planted_feature(tmp_path, monkeypatch, capsys):
    """``auroc`` ranks the planted feature ~1.0 for the planted concept (parsed from the table)."""
    _planted_buffer(tmp_path / "buf.npz")
    _run_cli(monkeypatch, "auroc", "--acts", str(tmp_path / "buf.npz"), "--labels", "planted,noise", "--device", "cpu")
    line = next(ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("planted"))
    _, _pct, best_auroc, feature = line.split()
    assert float(best_auroc) >= 0.95 and int(feature) == _PLANTED_FEATURE


def test_linear_cli_emits_dense_comparison(tmp_path, monkeypatch, capsys):
    """``linear`` prints the dense columns — proof the dense twin survived save->load into the probe."""
    _planted_buffer(tmp_path / "buf.npz")
    _run_cli(
        monkeypatch,
        "linear",
        "--acts",
        str(tmp_path / "buf.npz"),
        "--labels",
        "planted",
        "--device",
        "cpu",
        "--steps",
        "100",
    )
    out = capsys.readouterr().out
    assert "dense single" in out and "Δ" in out  # the SAE-vs-dense comparison only renders if dense loaded
    planted = next(ln for ln in out.splitlines() if ln.startswith("planted"))
    sae_single = float(planted.split("|")[1].split()[0])
    assert sae_single >= 0.95  # the planted feature separates the concept under the linear probe too


# ----------------------------------------------- engine -> buffer harness (evo2_buffer)
class _FakeEngine:
    """Minimal stand-in for the Evo2SAE engine: a real SAE + random hidden states, no model.

    Byte-level tokenization (1 char = 1 token, like Evo2) so per-token labels line up with codes.
    """

    def __init__(self, sae):
        self.device = "cpu"
        self.sae = sae
        self.n_features = sae.hidden_dim
        self._lock = contextlib.nullcontext()  # no GPU to serialize in the CPU test

    def tokenize(self, text):
        return list(range(len(text)))

    def _forward_hidden(self, id_lists):
        h = self.sae.pre_bias.shape[0]
        return [torch.randn(len(ids), h) for ids in id_lists]


def test_forward_codes_pairs_hidden_with_sae_codes():
    """forward_codes returns (hidden, codes) per input, codes == the SAE's own encode of hidden."""
    sae = TopKSAE(input_dim=8, hidden_dim=16, top_k=4, normalize_input=False)
    eng = _FakeEngine(sae)
    id_lists = [[1, 2, 3], [4, 5]]
    out = forward_codes(eng, id_lists)
    assert len(out) == len(id_lists)
    for (h, codes), ids in zip(out, id_lists):
        assert h.shape == (len(ids), 8) and codes.shape == (len(ids), 16)
        assert torch.allclose(codes, sae.encode(h))


def test_build_buffer_shapes_and_label_alignment_with_fake_engine():
    """build_buffer streams seqs -> ActivationBuffer with codes/dense/labels aligned and tag-offset right.

    Exercises the harness buffer build (forward_codes + labelers + ActivationBuffer) end to end on CPU,
    with no model: the per-base `base_A` label must fire exactly on the DNA 'A' positions and never on
    the phylo-tag prefix.
    """
    sae = TopKSAE(input_dim=8, hidden_dim=16, top_k=4, normalize_input=False)
    eng = _FakeEngine(sae)
    dna = "ACGT" * 20  # 80 bp, above build_buffer's 60 bp floor
    tag = KINGDOM_TAGS["euk"]
    T = len(tag) + len(dna)

    buf = build_buffer(
        eng, [("euk", dna)], ["base_A", "base_C", "base_G", "base_T"], subsample=512, auroc_device="cpu"
    )

    assert buf.codes.shape == (T, eng.n_features)
    assert buf.dense.shape == (T, 8) and buf.labels.shape == (T, 4)
    assert list(buf.label_names) == ["base_A", "base_C", "base_G", "base_T"]
    base_a = buf.labels[:, 0]
    assert base_a[: len(tag)].sum() == 0  # phylo-tag prefix is never base-labeled
    assert base_a[len(tag) :].sum() == dna.count("A")  # every DNA 'A', and only those


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU (loads the Evo2 model)")
def test_build_buffer_and_score_real_engine(evo2_ckpt_dir, sae_ckpt_path, embedding_layer):
    """End-to-end against the REAL Evo2SAE engine — the #1636<->#1622 seam, run in the merged GPU lane.

    Skips unless CUDA + the `evo2_sae` engine (#1622) are present. The evo2_ckpt_dir / sae_ckpt_path /
    embedding_layer fixtures come from the recipe `conftest.py` once the serve + eval stacks share
    `recipes/evo2/` — so this validates the real model -> codes -> labels -> scoring path after merge.
    """
    if not torch.cuda.is_available():
        pytest.skip("real-engine test requires CUDA")
    try:
        from evo2_sae.core import Evo2SAE
    except ImportError:
        pytest.skip("evo2_sae engine (#1622) not importable in this env")

    eng = Evo2SAE(evo2_ckpt_dir=evo2_ckpt_dir, sae_ckpt_path=sae_ckpt_path, layer=embedding_layer).load()
    seqs = [("euk", "ACGTACGT" * 16), ("prok", "ATGGCCGAATTC" * 10)]
    label_names = ["base_A", "base_C", "base_G", "base_T", "motif_ATG"]

    buf = build_buffer(eng, seqs, label_names, subsample=1024, auroc_device="cpu", batch_size=2)
    assert buf.codes.shape[0] > 0 and buf.codes.shape[1] == eng.n_features
    assert buf.dense is not None and buf.dense.shape[0] == buf.codes.shape[0]
    assert np.isfinite(buf.codes).all()

    au = auroc_all(torch.from_numpy(buf.codes).float(), torch.from_numpy(buf.labels).bool())
    assert au.shape == (eng.n_features, len(label_names)) and torch.isfinite(au).all()


# --------------------------------------------------- labels (#1630) <-> domain_f1 (#1629) seam
def test_label_windows_feed_domain_f1(tmp_path):
    """annot_tracks windows (mask + instance ids) drive instance-level domain_f1 end to end.

    A feature planted to fire exactly on the concept mask must beat a shuffled-label null.
    """
    seqs = {"chr1": "ACGT" * 300}  # 1200 bp
    tracks = {"site": {"chr1": [(20, 90), (300, 380), (700, 760)]}}  # three instances
    windows, stats = label_windows(seqs, tracks, seq_len=200)
    assert windows and stats["n_inst"]["site"] == 3

    mask = np.concatenate([w["labels"]["site"] for w in windows])
    inst = np.concatenate([w["instances"]["site"] for w in windows])
    n = mask.shape[0]
    rng = np.random.default_rng(0)
    codes = (rng.random((n, 4)) * 0.1).astype(np.float32)
    codes[mask, 0] = 5.0  # feature 0 fires on the concept, nowhere else

    codes_t = torch.from_numpy(codes)
    fmax = codes_t.max(0).values
    mask_t = torch.from_numpy(mask)
    inst_t = torch.from_numpy(inst.astype(np.int64))
    f1, _thr = domain_f1(codes_t, fmax, mask_t, inst_t)

    order = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    f1_null, _ = domain_f1(codes_t, fmax, mask_t[order], inst_t[order])
    assert int(f1.argmax()) == 0  # the planted feature wins
    assert float(f1.max()) > 2 * float(f1_null.max()) + 1e-6  # and clears the shuffled null
