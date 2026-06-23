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

  * ``sae.eval.probing`` (#1629) <-> the probe CLI (#1636): a buffer is written by
    ``ActivationBuffer.save``, reloaded, and scored through the real ``probe.main()``
    dispatch (``auroc`` / ``annotate`` / ``linear``). This is the path that silently
    broke when ``save`` dropped the dense twin — the ``linear`` SAE-vs-dense comparison
    only renders if ``dense`` survives the round trip.
  * ``annot_tracks.label_windows`` (#1630) <-> ``domain_f1`` (#1629): interval tracks ->
    per-token mask + instance ids -> instance-level F1, with no model in the loop.

A feature is *planted* to track one concept so the metrics have a known right answer;
everything else is noise, so a green run means the whole pipeline carried the signal.

The eval modules live in ``scripts/`` (run as scripts, not installed), so it's added to
``sys.path`` below — matching the sibling test modules.
"""

import contextlib
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sae.architectures import TopKSAE
from sae.eval.probing import ActivationBuffer, domain_f1


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import probe
from annot_tracks import label_windows
from evo2_buffer import forward_codes


# Feature index deliberately planted to track the "planted" concept.
_PLANTED_FEATURE = 3


def _planted_buffer(path, n=400, n_features=6, hidden=8, seed=0):
    """Write a synthetic buffer where ``_PLANTED_FEATURE`` cleanly separates the "planted" label.

    Includes a dense twin + per-concept instance ids so the save/load round trip and the
    SAE-vs-dense ``linear`` path are both exercised. Returns the in-memory buffer too.
    """
    rng = np.random.default_rng(seed)
    label_names = ["planted", "noise"]
    labels = np.zeros((n, len(label_names)), dtype=bool)
    labels[:, 0] = rng.random(n) < 0.4  # planted positives (~40%)
    labels[:, 1] = rng.random(n) < 0.4  # uncorrelated noise concept

    codes = (rng.random((n, n_features)) * 0.1).astype(np.float16)  # background noise
    pos = labels[:, 0]
    codes[pos, _PLANTED_FEATURE] = (rng.random(int(pos.sum())) * 5 + 5).astype(np.float16)  # strong, positives only

    dense = (rng.random((n, hidden))).astype(np.float16)
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
    assert set(tbl) == {"feature_id", "label", "auroc", "activation_freq", "max_activation"}
    rows = {fid: (lab, au) for fid, lab, au in zip(tbl["feature_id"], tbl["label"], tbl["auroc"])}
    assert _PLANTED_FEATURE in rows, "planted feature not annotated"
    label, auroc = rows[_PLANTED_FEATURE]
    assert label == "planted" and auroc >= 0.85


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


# ----------------------------------------------- shared engine->codes helper (evo2_buffer)
class _FakeEngine:
    """Minimal stand-in for the Evo2SAE engine: a real SAE + random hidden states, no model."""

    def __init__(self, sae):
        self.device = "cpu"
        self.sae = sae
        self._lock = contextlib.nullcontext()  # no GPU to serialize in the CPU test

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
