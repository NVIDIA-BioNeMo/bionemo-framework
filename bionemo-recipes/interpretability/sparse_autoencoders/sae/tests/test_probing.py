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

"""CPU correctness tests for sae.eval.probing (no model / no GPU).

One strong test per non-trivial metric: each checks the result against an independent
reference (a definitional oracle or a hand-computed value) rather than a loose sanity bound.
The trivial standardize helper is exercised transitively (decode_eval test); split_indices
folds into the buffer roundtrip.
"""

import numpy as np
import torch
from sae.eval.probing import (
    ActivationBuffer,
    annotate_features,
    auroc_all,
    best_single_train_test,
    decode_eval,
    domain_f1,
    split_indices,
)


def _auroc_ref(scores: torch.Tensor, y: torch.Tensor) -> float:
    """Definitional AUROC oracle: P(score+ > score-) over all positive/negative pairs.

    Computed by brute-force pair comparison — independent of the argsort rank-sum used by
    auroc_all, so agreement validates that implementation (randn inputs => no ties).
    """
    pos, neg = scores[y], scores[~y]
    return float((pos[:, None] > neg[None, :]).float().mean())


def test_auroc_all_matches_definition():
    """auroc_all matches the pairwise-definition AUROC for every (feature, label)."""
    torch.manual_seed(0)
    n, f, ell = 200, 6, 3
    x = torch.randn(n, f)
    y = torch.randn(n, ell) > 0
    au = auroc_all(x, y)  # [F, L]
    for fi in range(f):
        for li in range(ell):
            assert abs(float(au[fi, li]) - _auroc_ref(x[:, fi], y[:, li])) < 1e-6


def test_best_single_reports_flipped_test_auroc():
    """best_single picks the most-separating TRAIN feature and reports ITS test AUROC,
    flipping a feature that separates by firing on the negatives (no winner's curse)."""
    torch.manual_seed(0)
    y = torch.cat([torch.zeros(10), torch.ones(10)]).bool()
    # 'anti' fires on the y=0 class (train AUROC ~0 -> selected via 1-AUROC, flip=True);
    # it stays anti-correlated on test, so the reported (flipped) test AUROC is ~1.
    anti_tr = torch.cat([torch.ones(10), torch.zeros(10)]) + torch.randn(20) * 0.01
    anti_te = torch.cat([torch.ones(10), torch.zeros(10)]) + torch.randn(20) * 0.01
    xtr = torch.stack([anti_tr, torch.randn(20)], 1)  # 2nd feature is noise
    xte = torch.stack([anti_te, torch.randn(20)], 1)
    assert best_single_train_test(xtr, y, xte, y.clone()) > 0.9


def test_domain_f1_matches_hand_computed():
    """domain_f1 = precision-per-position, recall-per-instance, best over the threshold sweep.

    Two binary features over 6 positions, 2 annotation instances ({0,1} and {4}):
      feat0 fires at an extra non-concept position -> prec 3/4, recall 2/2 -> F1 = 6/7
      feat1 fires exactly on concept positions     -> prec 1,   recall 2/2 -> F1 = 1
    """
    codes = torch.tensor([[1, 1], [1, 1], [1, 0], [0, 0], [1, 1], [0, 0]], dtype=torch.float)
    fmax = codes.max(0).values
    concept_mask = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    inst_ids = torch.tensor([0, 0, -1, -1, 1, -1])
    f1, _ = domain_f1(codes, fmax, concept_mask, inst_ids)
    assert abs(float(f1[0]) - 6 / 7) < 1e-4
    assert abs(float(f1[1]) - 1.0) < 1e-4


def test_decode_eval_recovers_separable_classes():
    """The softmax decoder (fit_softmax + macro_auroc) separates separable classes and not noise."""
    torch.manual_seed(0)
    dim, nclass = 8, 3
    centers = torch.eye(nclass, dim) * 6.0

    def make(per):
        ys = torch.arange(nclass).repeat_interleave(per)
        return centers[ys] + torch.randn(len(ys), dim), ys

    xtr, ytr = make(40)
    xte, yte = make(20)
    acc, mauc, ncls = decode_eval(xtr, ytr, xte, yte, nclass, steps=400, lr=0.1)
    assert acc > 0.9 and mauc > 0.9 and ncls == 3

    # random features/labels -> no better than chance (1/3)
    xr, yr = torch.randn(120, dim), torch.randint(0, nclass, (120,))
    acc_rand, _, _ = decode_eval(xr[:90], yr[:90], xr[90:], yr[90:], nclass, steps=400, lr=0.1)
    assert acc_rand < 0.6


def test_annotate_features_assigns_best_concept_above_threshold():
    """Each feature gets the concept it best separates; unconfident features stay unlabeled."""
    torch.manual_seed(0)
    n = 200
    labels = torch.stack([torch.arange(n) % 2 == 0, torch.arange(n) < n // 2], 1)  # [N, 2]: 'even', 'first_half'
    detector = labels[:, 0].float() + torch.randn(n) * 0.01  # cleanly tracks 'even'
    noise = torch.randn(n)  # tracks nothing
    codes = torch.stack([detector, noise], 1)  # [N, 2 features]
    ann = annotate_features(codes, labels, ["even", "first_half"], min_auroc=0.9)
    assert {a["feature_id"]: a["label"] for a in ann} == {0: "even"}  # feature 1 (noise) excluded
    assert ann[0]["auroc"] > 0.99


def test_buffer_roundtrip_and_split(tmp_path):
    """ActivationBuffer save/load preserves codes/labels/names/dense/instances; split is a partition."""
    rng = np.random.default_rng(0)
    codes = rng.random((10, 4)).astype(np.float16)
    labels = np.tile(np.array([True, False, True]), (10, 1))
    dense = rng.random((10, 8)).astype(np.float16)
    instances = {"exon": np.array([0, 0, -1, 1, 1, -1, 2, 2, 2, -1], np.int32)}
    buf = ActivationBuffer(codes, labels, ["a", "b", "c"], dense=dense, instances=instances)
    path = str(tmp_path / "buf.npz")
    buf.save(path)

    lo = ActivationBuffer.load(path)
    assert np.array_equal(lo.codes, codes)
    assert np.array_equal(lo.dense, dense)
    assert np.array_equal(lo.instances["exon"], instances["exon"])
    assert lo.name_idx["c"] == 2

    tr, te = split_indices(100, test_frac=0.4, seed=0)
    s_tr, s_te = set(tr.tolist()), set(te.tolist())
    assert s_tr.isdisjoint(s_te) and (s_tr | s_te) == set(range(100)) and len(s_te) == 40
