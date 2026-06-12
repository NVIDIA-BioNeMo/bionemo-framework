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

"""Model-agnostic SAE feature-probing metrics + the activation-buffer artifact.

Everything here is a pure function of a probing buffer (per-token feature codes,
an optional dense-residual twin, per-token labels, optional instance IDs). Recipe
drivers (e.g. Evo2) only produce the buffer; all scoring lives here so it is shared
and reusable. Companions in this package: loss_recovered (fidelity), reconstruction,
sparsity, dead_latents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch


# ───────────────────────────────────────────────────────────── artifact
@dataclass
class ActivationBuffer:
    """A probing buffer: SAE codes (+ optional dense twin), per-token labels, instance IDs."""

    codes: np.ndarray  # [N, F] float16  SAE feature activations
    labels: np.ndarray  # [N, L] bool
    label_names: list
    dense: Optional[np.ndarray] = None  # [N, H] float16  raw layer residual (dense twin)
    instances: Optional[Dict[str, np.ndarray]] = None  # {concept: [N] int32, -1 outside}

    def save(self, path: str) -> None:
        """Write codes, labels, names (+ optional dense twin / instance ids) to an .npz."""
        d = {"codes": self.codes, "labels": self.labels, "label_names": np.array(self.label_names)}
        if self.dense is not None:
            d["dense"] = self.dense
        for k, v in (self.instances or {}).items():
            d[f"inst_{k}"] = v
        np.savez(path, **d)

    @classmethod
    def load(cls, path: str) -> "ActivationBuffer":
        """Load an ActivationBuffer from an .npz written by save()."""
        z = np.load(path, allow_pickle=True)
        inst = {k[5:]: z[k] for k in z.files if k.startswith("inst_")}
        return cls(
            codes=z["codes"],
            labels=z["labels"],
            label_names=list(z["label_names"]),
            dense=z["dense"] if "dense" in z.files else None,
            instances=inst or None,
        )

    @property
    def name_idx(self):
        """Map each label name to its column index in ``labels``."""
        return {n: i for i, n in enumerate(self.label_names)}


def split_indices(n, test_frac=0.4, seed=0):
    """Deterministic train/test split of ``range(n)``; returns (train_idx, test_idx)."""
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    nte = int(n * test_frac)
    return perm[nte:], perm[:nte]  # train, test


def standardize(X, tr):
    """Return (mean, std) of ``X`` over the train rows ``tr`` (std floored by 1e-6)."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    return mu, sd


# ───────────────────────────────────────────────────────────── AUROC
@torch.no_grad()
def auroc_all(X, Y, chunk=1024):
    """X [N,F], Y [N,L] bool -> AUROC [F,L] via vectorized rank statistic."""
    N, F = X.shape
    L = Y.shape[1]
    y = Y.float()
    npos = y.sum(0)
    nneg = N - npos
    valid = (npos > 0) & (nneg > 0)
    denom = (npos * nneg).clamp_min(1.0)
    half = npos * (npos + 1) / 2.0
    out = torch.full((F, L), 0.5, device=X.device)
    for c0 in range(0, F, chunk):
        c1 = min(c0 + chunk, F)
        ranks = X[:, c0:c1].float().argsort(0).argsort(0).float() + 1.0
        au = (y.t() @ ranks - half[:, None]) / denom[:, None]
        out[c0:c1] = au.t()
    out[:, ~valid] = 0.5
    return out


@torch.no_grad()
def auroc_vec(scores, y):
    """AUROC of a single score vector against boolean labels ``y`` (0.5 if degenerate)."""
    n = scores.numel()
    npos = int(y.sum())
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return 0.5
    ranks = scores.argsort().argsort().float() + 1.0
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


@torch.no_grad()
def best_single_train_test(Xtr, ytr, Xte, yte, chunk=2048):
    """Pick the best single dim on TRAIN, report ITS AUROC on TEST (no winner's curse)."""

    def per_feat(X, y):
        n = X.shape[0]
        npos = int(y.sum())
        nneg = n - npos
        if npos == 0 or nneg == 0:
            return None
        yf = y.float()
        F = X.shape[1]
        out = torch.empty(F, device=X.device)
        for c0 in range(0, F, chunk):
            ranks = X[:, c0 : c0 + chunk].float().argsort(0).argsort(0).float() + 1.0
            out[c0 : c0 + chunk] = (yf @ ranks - npos * (npos + 1) / 2) / (npos * nneg)
        return out

    a_tr = per_feat(Xtr, ytr)
    if a_tr is None:
        return float("nan")
    f = int(torch.maximum(a_tr, 1 - a_tr).argmax())
    flip = bool(a_tr[f] < 0.5)
    a_te = auroc_vec(Xte[:, f].float(), yte)
    return float(1 - a_te if flip else a_te)


@torch.no_grad()
def annotate_features(codes, labels, label_names, min_auroc: float = 0.8, chunk: int = 1024):
    """Assign each feature the concept it best separates (by AUROC) -> the feature->label table.

    The persistence half of probing: turns a buffer (codes + concept labels) into per-feature
    annotations. For each feature, takes the concept with the highest AUROC and keeps it only if
    that AUROC >= ``min_auroc`` (unconfident features stay unlabeled).

    Args:
        codes: [N, F] feature activations.
        labels: [N, L] bool concept masks.
        label_names: length-L concept names.
        min_auroc: keep a feature's annotation only if its best AUROC clears this.
        chunk: feature chunk size for ``auroc_all``.

    Returns:
        ``[{"feature_id": int, "label": str, "auroc": float}]`` sorted by feature_id.
    """
    au = auroc_all(codes, labels, chunk=chunk)  # [F, L]
    best = au.max(dim=1)
    names = list(label_names)
    out = []
    for f in range(au.shape[0]):
        score = float(best.values[f])
        if score >= min_auroc:
            out.append({"feature_id": int(f), "label": str(names[int(best.indices[f])]), "auroc": round(score, 4)})
    return out


# ───────────────────────────────────────────────────────────── linear probes
def fit_logreg(Xtr, ytr, steps=400, lr=0.05, wd=1e-2):
    """Fit a logistic-regression probe (Adam + BCE-with-logits); returns (w, b)."""
    w = torch.zeros(Xtr.shape[1], device=Xtr.device, requires_grad=True)
    b = torch.zeros(1, device=Xtr.device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.BCEWithLogitsLoss()
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            lossf(Xtr @ w + b, ytr).backward()
            opt.step()
    return w.detach(), b.detach()


def fit_softmax(Xtr, ytr, nclass, steps=400, lr=0.05, wd=1e-2):
    """Fit a multinomial-softmax probe (Adam + cross-entropy); returns (W, b)."""
    W = torch.zeros(Xtr.shape[1], nclass, device=Xtr.device, requires_grad=True)
    b = torch.zeros(nclass, device=Xtr.device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    lossf = torch.nn.CrossEntropyLoss()
    with torch.enable_grad():
        for _ in range(steps):
            opt.zero_grad()
            lossf(Xtr @ W + b, ytr).backward()
            opt.step()
    return W.detach(), b.detach()


@torch.no_grad()
def macro_auroc(logits, y, nclass):
    """Macro-averaged one-vs-rest AUROC over ``nclass``; returns (mean_auroc, n_classes_scored)."""
    aucs = []
    for c in range(nclass):
        yc = y == c
        npos = int(yc.sum())
        if npos == 0 or npos == len(y):
            continue
        ranks = logits[:, c].argsort().argsort().float() + 1.0
        aucs.append(float((ranks[yc].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos))))
    return (sum(aucs) / max(1, len(aucs))), len(aucs)


def decode_eval(Xtr, ytr, Xte, yte, nclass, **kw):
    """Fit a softmax probe on train; return (accuracy, macro_auroc, n_classes) on test."""
    W, b = fit_softmax(Xtr, ytr, nclass, **kw)
    logits = Xte @ W + b
    acc = float((logits.argmax(1) == yte).float().mean())
    mauc, ncls = macro_auroc(logits, yte, nclass)
    return acc, mauc, ncls


# ───────────────────────────────────────────────────────────── domain-adjusted F1
@torch.no_grad()
def domain_f1(codes, fmax, concept_mask, inst_ids, thresholds=(0.15, 0.3, 0.5, 0.6, 0.8), chunk=1024):
    """InterPLM domain-adjusted F1 per feature: precision-per-position, recall-per-instance.

    codes [P,F] (>=0), fmax [F], concept_mask [P] bool, inst_ids [P] int (-1 outside).
    Returns (best_f1[F], best_threshold[F]) over the threshold sweep.
    """
    _, F = codes.shape
    dev = codes.device
    valid = inst_ids >= 0
    uniq = torch.unique(inst_ids[valid])
    n_inst = len(uniq)
    if n_inst == 0:
        return torch.zeros(F, device=dev), torch.zeros(F, device=dev)
    remap = torch.full((int(inst_ids.max().item()) + 2,), -1, device=dev, dtype=torch.long)
    remap[uniq.long()] = torch.arange(n_inst, device=dev)
    inst_c = torch.where(valid, remap[inst_ids.long()], torch.full_like(inst_ids, -1, dtype=torch.long))
    best_f1 = torch.zeros(F, device=dev)
    best_t = torch.zeros(F, device=dev)
    for c0 in range(0, F, chunk):
        c1 = min(c0 + chunk, F)
        cn = codes[:, c0:c1] / fmax[c0:c1].clamp_min(1e-6)
        C = c1 - c0
        cb = torch.zeros(C, device=dev)
        ct = torch.zeros(C, device=dev)
        for t in thresholds:
            fire = cn > t
            firing = fire.sum(0).float()
            prec = torch.where(
                firing > 0, (fire & concept_mask[:, None]).sum(0).float() / firing, torch.zeros(C, device=dev)
            )
            bucket = torch.zeros(n_inst, C, device=dev)
            vm = inst_c >= 0
            bucket.index_reduce_(0, inst_c[vm], fire[vm].float(), "amax", include_self=False)
            recall = (bucket > 0).sum(0).float() / n_inst
            f1 = torch.where((prec + recall) > 0, 2 * prec * recall / (prec + recall), torch.zeros(C, device=dev))
            upd = f1 > cb
            cb = torch.where(upd, f1, cb)
            ct = torch.where(upd, torch.full_like(ct, t), ct)
        best_f1[c0:c1] = cb
        best_t[c0:c1] = ct
    return best_f1, best_t
