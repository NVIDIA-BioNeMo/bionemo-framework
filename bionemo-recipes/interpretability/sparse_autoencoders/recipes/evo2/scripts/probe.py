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

r"""Unified Evo2 SAE probing CLI. All scoring is sae.eval.probing (model-agnostic);
this driver only knows how to build/load Evo2 buffers and pick label sets.

  probe.py extract       --out BUF [...]        build an ActivationBuffer (needs the model)
  probe.py auroc         --acts BUF --labels .. per-feature AUROC table (prints)
  probe.py annotate      --acts BUF --out P     assign each feature its best concept -> annotation parquet
  probe.py linear        --acts BUF --labels .. SAE-vs-dense single + multi (disentanglement/distributed)
  probe.py codon-aa      --acts CODON_BUF       codon/AA decoders + family-disjoint, SAE vs dense
  probe.py euk-f1        --fasta .. --gff ..    RefSeq gene-structure domain-F1 (needs the model)
  probe.py domain-eval   --fasta .. --track ..  user annotated dataset -> per-feature domain-F1 + AUROC vs
                                                any BED/GFF tracks (RefSeq/Rfam/JASPAR/ENCODE) (needs the model)
  probe.py loss-recovered [...]                 fidelity via sae.eval.loss_recovered (needs the model)

Example end-to-end flow (7B / layer 26; $CKPT = MBridge dir, $SAE = trained SAE .pt):

  # 1. Build the probing buffer once: SAE codes + dense twin + per-token labels (needs the model)
  python probe.py extract --evo2-ckpt-dir $CKPT --sae-checkpoint $SAE --layer 26 \
      --fasta probe_set.fa --out buf.npz

  # 2. Score the buffer (no model): per-feature AUROC, then SAE-vs-dense linear probes
  python probe.py auroc  --acts buf.npz --labels motif_ATG,motif_stop,cds_coding,is_prok
  python probe.py linear --acts buf.npz --labels cds_coding,is_prok

  # 3. Persist annotations (no model): each feature's best concept (incl. base_A/C/G/T) ->
  #    the feature-annotation parquet the engine/dashboard load via --feature-annotations
  python probe.py annotate --acts buf.npz --out feature_annotations.parquet --min-auroc 0.85

  # 4. User annotated dataset -> per-feature domain-F1 (prec/nt, recall/annotation) + AUROC,
  #    vs any BED/GFF tracks (RefSeq/Rfam/JASPAR/ENCODE) (needs the model)
  python probe.py domain-eval --evo2-ckpt-dir $CKPT --sae-checkpoint $SAE --layer 26 \
      --fasta GRCh38_chr20.fa --track exon=refseq.gff3:exon --track cCRE=encode_ccre.bed

  # 5. SAE fidelity (loss recovered) — separate script, needs the model
  python probe_loss_recovered.py --evo2-ckpt-dir $CKPT --sae-checkpoint $SAE --layer 26 --fasta probe_set.fa
"""  # noqa: D205

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2] / "sae" / "src"))  # sparse_autoencoders/sae/src

import labelers as L  # noqa: E402
from sae.eval.probing import (  # noqa: E402
    ActivationBuffer,
    auroc_all,
    auroc_vec,
    best_single_train_test,
    decode_eval,
    fit_softmax,
    split_indices,
    standardize,
)


def _z(X, tr):
    # Standardize X by the train-split mean/std (reuses sae.eval.probing.standardize).
    mu, sd = standardize(X, tr)
    return (X - mu) / sd


def _load(a):
    """Load the probing buffer + resolve requested label names (default: all labels in the buffer)."""
    buf = ActivationBuffer.load(a.acts)
    labels = a.labels.split(",") if getattr(a, "labels", None) else list(buf.label_names)
    return buf, [t for t in labels if t in buf.name_idx]


# ───────────────────────────────────────── buffer-only subcommands (no model)
def cmd_auroc(a):  # noqa: D103
    buf, names = _load(a)
    dev = a.device
    X = torch.from_numpy(buf.codes).to(dev).float()
    Y = torch.stack([torch.from_numpy(buf.labels[:, buf.name_idx[n]]).to(dev) for n in names], 1)
    au = auroc_all(X, Y).cpu().numpy()
    print(f"{'label':18s} {'%pos':>6s} {'best AUROC':>10s} {'feature':>8s}")
    for i, n in enumerate(names):
        print(
            f"{n:18s} {buf.labels[:, buf.name_idx[n]].mean():6.1%} {au[:, i].max():10.3f} {int(au[:, i].argmax()):8d}"
        )


def _eval_matrix(mat, buf, names, tr, te, dev, steps, wd):
    X = torch.from_numpy(mat).to(dev).float()
    Xz = _z(X, tr)
    out = {}
    from sae.eval.probing import fit_logreg

    for n in names:
        ytr = torch.from_numpy(buf.labels[tr.numpy(), buf.name_idx[n]]).to(dev).float()
        yte = torch.from_numpy(buf.labels[te.numpy(), buf.name_idx[n]]).to(dev)
        if ytr.sum() in (0, len(ytr)) or yte.sum() == 0:
            out[n] = (float("nan"), float("nan"))
            continue
        w, b = fit_logreg(Xz[tr], ytr, steps=steps, wd=wd)
        out[n] = (best_single_train_test(Xz[tr], ytr, Xz[te], yte), auroc_vec((Xz[te] @ w + b).float(), yte))
    del X, Xz
    torch.cuda.empty_cache()
    return out


def cmd_linear(a):  # noqa: D103
    buf, names = _load(a)
    dev = a.device
    tr, te = split_indices(buf.codes.shape[0], a.test_frac, a.seed)
    sae = _eval_matrix(buf.codes, buf, names, tr, te, dev, a.steps, a.weight_decay)
    den = _eval_matrix(buf.dense, buf, names, tr, te, dev, a.steps, a.weight_decay) if buf.dense is not None else None
    h = f"{'label':18s} {'%pos':>6s} | {'SAE single':>10s} {'SAE multi':>9s}"
    if den:
        h += f" | {'dense single':>12s} {'dense multi':>11s} | {'Δ':>7s}"
    print(h)
    for n in names:
        pos = buf.labels[:, buf.name_idx[n]].mean()
        ss, sm = sae[n]
        row = f"{n:18s} {pos:6.1%} | {ss:10.3f} {sm:9.3f}"
        if den:
            ds, dm = den[n]
            row += f" | {ds:12.3f} {dm:11.3f} | {ss - ds:+7.3f}"
        print(row)


def cmd_codon_aa(a):  # noqa: D103
    z = np.load(a.acts)
    dev = a.device
    codon = torch.from_numpy(z["codon"].astype(np.int64)).to(dev)
    aa = torch.from_numpy(z["aa"].astype(np.int64)).to(dev)
    codon_np = z["codon"].astype(np.int64)
    ncod, naa = len(L.CODON_LIST), len(L.AA_LIST)
    held = {"L": ["TTA", "TTG"], "S": ["AGT", "AGC"], "R": ["AGA", "AGG"]}
    hidx = [L.CODON_TO_IDX[c] for v in held.values() for c in v]
    print(f"{'matrix':6s} {'codon mAUROC':>12s} {'AA mAUROC':>10s} | family-disjoint recall L/S/R (chance)")
    for nm in ("sae", "dense"):
        if nm not in z.files:
            continue
        X = torch.from_numpy(z[nm]).to(dev).float()
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-6)
        tr, te = split_indices(X.shape[0], a.test_frac, a.seed)
        _, ca, _ = decode_eval(Xz[tr], codon[tr], Xz[te], codon[te], ncod, steps=a.steps, wd=a.weight_decay)
        _, aaa, _ = decode_eval(Xz[tr], aa[tr], Xz[te], aa[te], naa, steps=a.steps, wd=a.weight_decay)
        trn = torch.from_numpy(np.nonzero(~np.isin(codon_np, hidx))[0]).to(dev)
        W, b = fit_softmax(Xz[trn], aa[trn], naa, steps=a.steps, wd=a.weight_decay)
        rec = []
        for A, cods in held.items():
            m = np.isin(codon_np, [L.CODON_TO_IDX[c] for c in cods])
            pred = (Xz[torch.from_numpy(np.nonzero(m)[0]).to(dev)] @ W + b).argmax(1).cpu().numpy()
            rec.append(
                f"{A}={float((pred == L.AA_TO_IDX[A]).mean()):.2f}({float((aa == L.AA_TO_IDX[A]).float().mean()):.2f})"
            )
        del X, Xz
        torch.cuda.empty_cache()
        print(f"{nm:6s} {ca:12.3f} {aaa:10.3f} | {'  '.join(rec)}")


def cmd_annotate(a):
    """Buffer -> feature-annotation parquet: each feature's best concept by AUROC + activation stats.

    The persist step (uses sae.eval.probing.annotate_features). Writes a feature_metadata-style
    parquet — {feature_id, label, auroc, activation_freq, max_activation} — the engine/dashboard
    load via --feature-annotations. Concepts default to all labels in the buffer (incl. base_*).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from sae.eval.probing import annotate_features

    buf, names = _load(a)
    dev = a.device
    X = torch.from_numpy(buf.codes).to(dev).float()
    Y = torch.stack([torch.from_numpy(buf.labels[:, buf.name_idx[n]]).to(dev) for n in names], 1)
    ann = annotate_features(X, Y, names, min_auroc=a.min_auroc)
    cols = {"feature_id": [], "label": [], "auroc": [], "activation_freq": [], "max_activation": []}
    for r in ann:
        col = X[:, r["feature_id"]]
        cols["feature_id"].append(r["feature_id"])
        cols["label"].append(r["label"])
        cols["auroc"].append(r["auroc"])
        cols["activation_freq"].append(round(float((col > 0).float().mean()), 6))
        cols["max_activation"].append(round(float(col.max()), 4))
    pq.write_table(pa.table(cols), a.out, compression="snappy")
    print(f"[annotate] {len(ann)} features labeled (AUROC >= {a.min_auroc}) over {len(names)} concepts -> {a.out}")


# ───────────────────────────────────────── model subcommands (need Evo2)
def _encode_windows(eng, windows, tag_ids, lab_keys, inst_keys, tot, a):
    """Stream tiled windows through the SAE -> (code_buf[filled,F], lab{k:bool}, inst{k:long}, fmax[F]).

    Shared by euk-f1 and domain-eval: encodes each window (skipping the phylo-tag prefix) and
    fills per-concept label masks (lab_keys) + instance ids (inst_keys). Buffers are trimmed to
    the number of positions actually filled.
    """
    adev, tlen = a.auroc_device, len(tag_ids)
    code_buf = torch.zeros(tot, eng.n_features, dtype=torch.float16, device=adev)
    lab = {k: torch.zeros(tot, dtype=torch.bool, device=adev) for k in lab_keys}
    inst = {k: torch.full((tot,), -1, dtype=torch.long, device=adev) for k in inst_keys}
    filled = 0
    for s0 in range(0, len(windows), a.batch_size):
        batch = windows[s0 : s0 + a.batch_size]
        with eng._lock:
            for h, w in zip(eng._forward_hidden([tag_ids + eng.tokenize(w["dna"]) for w in batch]), batch):
                if h.shape[0] == 0:
                    continue
                codes = eng.sae.encode(h.to(a.device))
                take = min(len(w["dna"]), codes.shape[0] - tlen, tot - filled)
                if take <= 0:
                    continue
                code_buf[filled : filled + take] = codes[tlen : tlen + take].to(torch.float16).to(adev)
                for k in lab:
                    lab[k][filled : filled + take] = torch.from_numpy(w["labels"][k][:take]).to(adev)
                for k in inst:
                    inst[k][filled : filled + take] = torch.from_numpy(w["instances"][k][:take].astype(np.int64)).to(
                        adev
                    )
                filled += take
    code_buf = code_buf[:filled]
    for d in (lab, inst):
        for k in d:
            d[k] = d[k][:filled]
    fmax = code_buf.max(0).values.float() if filled else torch.zeros(eng.n_features, device=adev)
    return code_buf, lab, inst, fmax


def cmd_euk(a):
    """Eukaryotic exon/intron/CDS domain-adjusted F1 vs shuffle null (chr21 FASTA+GFF)."""
    from euk_windows import build_windows
    from evo2_sae.core import DEFAULT_ORGANISM_TAGS, Evo2SAE
    from sae.eval.probing import domain_f1

    eng = Evo2SAE(a.evo2_ckpt_dir, a.sae_checkpoint, a.layer, device=a.device).load()
    windows, stats, tot, _ = build_windows(a.fasta, a.gff, a.seq_len, a.max_tokens, seed=a.seed)
    print(
        f"windows={len(windows)} tokens={tot} genes={stats['genes']} exons={stats['exons']} introns={stats['introns']}"
    )
    tag_ids = eng.tokenize(DEFAULT_ORGANISM_TAGS.get(a.organism, ""))
    code_buf, lab, inst, fmax = _encode_windows(
        eng, windows, tag_ids, ("exon", "intron", "cds"), ("exon", "intron", "gene"), tot, a
    )
    filled, adev = code_buf.shape[0], a.auroc_device
    g = torch.Generator(device=adev).manual_seed(a.seed)
    print(f"encoded {filled} positions\n{'concept':8s} {'domF1':>6s} {'null':>6s} {'ratio':>6s} {'%pos':>6s}")
    for c, ic in {"exon": "exon", "intron": "intron", "cds": "gene"}.items():
        f1, _ = domain_f1(code_buf, fmax, lab[c], inst[ic])
        order = torch.randperm(filled, generator=g, device=adev)
        f1n, _ = domain_f1(code_buf, fmax, lab[c][order], inst[ic][order])
        bf, nl = float(f1.max()), float(f1n.max())
        print(f"{c:8s} {bf:6.3f} {nl:6.3f} {bf / max(nl, 1e-9):6.2f} {float(lab[c].float().mean()):6.1%}")


def _parse_track_spec(spec):
    """Parse a ``--track NAME=PATH[:GFF_FEATURE]`` spec -> (name, path, feature_type|None)."""
    name, rest = spec.split("=", 1)
    ftype = None
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if "/" not in tail and "." not in tail:  # a GFF feature type, not part of a path
            rest, ftype = head, tail
    return name, rest, ftype


def cmd_domain_eval(a):
    """User-supplied annotated dataset -> per-feature domain-F1 (prec/nt, recall/annotation) + AUROC.

    Each ``--track NAME=PATH[:GFF_FEATURE]`` is one concept; its BED/GFF intervals are the
    annotation instances (RefSeq/Rfam/JASPAR/ENCODE, or anything the user supplies). The SAE
    annotates the windows, then per concept we report the best feature by instance-level
    domain-F1 (precision-per-nt, recall-per-annotation) and — threshold-free — by AUROC.
    """
    from annot_tracks import label_windows, load_track, read_fasta_dict
    from evo2_sae.core import DEFAULT_ORGANISM_TAGS, Evo2SAE
    from sae.eval.probing import auroc_all, domain_f1

    tracks = {}
    for spec in a.track:
        name, path, ftype = _parse_track_spec(spec)
        tracks[name] = load_track(path, feature_type=ftype)
    seqs = read_fasta_dict(a.fasta)
    windows, stats = label_windows(seqs, tracks, a.seq_len, max_tokens=a.max_tokens)
    concepts = stats["concepts"]

    eng = Evo2SAE(a.evo2_ckpt_dir, a.sae_checkpoint, a.layer, device=a.device).load()
    tag_ids = eng.tokenize(DEFAULT_ORGANISM_TAGS.get(a.organism, ""))
    code_buf, lab, inst, fmax = _encode_windows(eng, windows, tag_ids, concepts, concepts, stats["tokens"], a)
    au = auroc_all(code_buf.float().to(a.device), torch.stack([lab[c] for c in concepts], 1).to(a.device)).cpu()
    print(f"encoded {code_buf.shape[0]} positions across {len(concepts)} concept(s)")
    print(
        f"{'concept':14s} {'%pos':>6s} {'#inst':>6s} | "
        f"{'domF1':>6s} {'@thr':>5s} {'feat':>7s} | {'AUROC':>6s} {'feat':>7s}"
    )
    for i, c in enumerate(concepts):
        f1, thr = domain_f1(code_buf, fmax, lab[c], inst[c])
        bi, ai = int(f1.argmax()), int(au[:, i].argmax())
        print(
            f"{c:14s} {float(lab[c].float().mean()):6.1%} {stats['n_inst'][c]:6d} | "
            f"{float(f1[bi]):6.3f} {float(thr[bi]):5.2f} {bi:7d} | {float(au[ai, i]):6.3f} {ai:7d}"
        )


def cmd_extract(a):  # noqa: D103
    from evo2_buffer import build_buffer, sample_sequences
    from evo2_sae.core import Evo2SAE

    eng = Evo2SAE(a.evo2_ckpt_dir, a.sae_checkpoint, a.layer, device=a.device).load()
    label_names = list(L.LABELERS.keys())
    kingdoms = [k for k in a.kingdoms.split(",") if k]
    seqs = sample_sequences(a.fasta, a.max_tokens, a.seq_len, kingdoms=kingdoms, seed=a.seed)
    print(f"probe set: {len(seqs)} seqs (kingdoms={kingdoms})")
    buf = build_buffer(
        eng,
        seqs,
        label_names,
        subsample=a.subsample,
        auroc_device=a.auroc_device,
        annotate_cds=a.annotate_cds,
        batch_size=a.batch_size,
        log=print,
    )
    buf.save(a.out)
    print(f"saved buffer -> {a.out} ({buf.codes.shape[0]} x {buf.codes.shape[1]}, dense {buf.dense.shape[1]})")


def _add_model_args(p, *, required=(), max_tokens=160_000):
    """Shared model + encoding args for the model-backed subcommands (extract/euk-f1/domain-eval)."""
    for arg in ("--evo2-ckpt-dir", "--sae-checkpoint", "--fasta", *required):
        p.add_argument(arg, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--max-tokens", type=int, default=max_tokens)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--auroc-device", default="cuda:1")


def main():  # noqa: D103
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="cuda:0")
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--steps", type=int, default=400)
    common.add_argument("--weight-decay", type=float, default=1e-2)
    common.add_argument("--test-frac", type=float, default=0.4)
    for name, fn, needs_labels in [
        ("auroc", cmd_auroc, True),
        ("linear", cmd_linear, True),
        ("codon-aa", cmd_codon_aa, False),
    ]:
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--acts", required=True)
        if needs_labels:
            p.add_argument("--labels", required=True)
        p.set_defaults(func=fn)
    pan = sub.add_parser("annotate", parents=[common])
    pan.add_argument("--acts", required=True)
    pan.add_argument("--out", required=True)
    pan.add_argument(
        "--labels", default=None, help="comma-separated concept subset; default = all labels in the buffer"
    )
    pan.add_argument("--min-auroc", type=float, default=0.8)
    pan.set_defaults(func=cmd_annotate)
    pe = sub.add_parser("extract", parents=[common])
    _add_model_args(pe, required=("--out",), max_tokens=200_000)
    pe.add_argument("--kingdoms", default="prok,euk")
    pe.add_argument("--annotate-cds", action="store_true")
    pe.add_argument("--subsample", type=int, default=50_000)
    pe.set_defaults(func=cmd_extract)
    pk = sub.add_parser("euk-f1", parents=[common])
    _add_model_args(pk, required=("--gff",))
    pk.add_argument("--organism", default="Human")
    pk.set_defaults(func=cmd_euk)
    pd = sub.add_parser("domain-eval", parents=[common])
    _add_model_args(pd)
    pd.add_argument(
        "--track",
        action="append",
        required=True,
        metavar="NAME=PATH[:GFF_FEATURE]",
        help="annotation track; BED or GFF intervals = instances of concept NAME. Repeatable "
        "(e.g. --track exon=refseq.gff3:exon --track tfbs=jaspar.bed --track cCRE=encode.bed).",
    )
    pd.add_argument("--organism", default="Human")
    pd.set_defaults(func=cmd_domain_eval)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    args.func(args)


if __name__ == "__main__":
    main()
