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

"""Evo2-specific bit: turn DNA sequences into a probing ActivationBuffer.

The only model-touching code in the probing pipeline. Streams sequences through
the Evo2SAE engine (Evo2 -> layer-L residual -> SAE.encode), keeps the dense
residual twin, and computes per-token labels (+ instance IDs) from labelers.py.
All scoring is done elsewhere by the model-agnostic evo2_sae.eval.probing metrics.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from evo2_sae.eval.probing import ActivationBuffer

from . import labelers as L


KINGDOM_TAGS = {"prok": "|d__Bacteria|", "euk": "|d__Eukaryota|"}


@torch.no_grad()
def forward_codes(engine, id_lists):
    """Run token-id lists through the engine -> ``[(hidden[S,H], sae_codes[S,F])]`` on engine.device.

    The single place the probing harness touches the engine internals — the GPU forward plus the
    SAE encode, serialized by the engine lock. ``build_buffer`` (here) and ``probe._encode_windows``
    both go through this, so the coupling to the engine API lives in exactly one spot.
    """
    dev = engine.device
    with engine._lock:
        return [(h, engine.sae.encode(h.to(dev))) for h in engine._forward_hidden(id_lists)]


def sample_sequences(fasta, max_tokens, seq_len, kingdoms=("prok", "euk"), seed=0):  # noqa: D103
    # Imported here (not at module top) so this module is usable without the evo2_sae engine
    # installed — e.g. the CPU tests that exercise forward_codes / labels.
    from evo2_sae.core import clean_dna
    from evo2_sae.fasta import read_fasta  # shared reader; also handles .gz

    kingdoms = list(kingdoms)
    pools = {k: [] for k in kingdoms}
    need = max_tokens // seq_len + 50
    for header, seq in read_fasta(fasta):
        kg = "prok" if header.lower().startswith("prok") else "euk"
        if kg not in pools:
            continue
        dna = clean_dna(seq)[:seq_len]
        if len(dna) < 60:
            continue
        pools[kg].append(dna)
        if all(len(pools[k]) >= need for k in kingdoms):
            break
    rng = random.Random(seed)
    for k in kingdoms:
        rng.shuffle(pools[k])
    out, tok, i = [], 0, 0
    maxlen = max((len(pools[k]) for k in kingdoms), default=0)
    while tok < max_tokens and i < maxlen:
        for k in kingdoms:
            if i < len(pools[k]):
                out.append((k, pools[k][i]))
                tok += len(pools[k][i]) + len(KINGDOM_TAGS[k])
        i += 1
    rng.shuffle(out)
    return out


@torch.no_grad()
def build_buffer(engine, seqs, label_names, *, subsample, auroc_device, annotate_cds=False, batch_size=8, log=print):
    """Stream seqs through engine -> ActivationBuffer (codes + dense + labels [+ cds instances])."""
    F = engine.n_features
    Hd = engine.sae.pre_bias.shape[0]
    dev = engine.device
    S = subsample
    code_buf = torch.zeros(S, F, dtype=torch.float16, device=auroc_device)
    dense_buf = torch.zeros(S, Hd, dtype=torch.float16, device=auroc_device)
    lab_buf = torch.zeros(S, len(label_names), dtype=torch.bool, device=auroc_device)
    filled = 0
    for start in range(0, len(seqs), batch_size):
        if filled >= S:
            break
        batch = seqs[start : start + batch_size]
        id_lists, metas = [], []
        for kg, dna in batch:
            tag = KINGDOM_TAGS[kg]
            tids = engine.tokenize(tag)
            id_lists.append(tids + engine.tokenize(dna))
            metas.append((tag, len(tids), kg, dna))
        # GPU forward + SAE encode happen inside forward_codes (engine-locked); the per-token
        # label computation + buffer fills below are CPU/copy work and need no lock.
        for (h, codes), (tag, tlen, kg, dna) in zip(forward_codes(engine, id_lists), metas):
            if h.shape[0] == 0 or filled >= S:
                continue
            hd = h.to(dev)
            norm = h.float().norm(dim=-1).cpu().numpy()
            T = codes.shape[0]
            cds_mask = cds_frame = gene_starts = None
            if annotate_cds and kg == "prok":
                cds_mask, cds_frame, gene_starts = L.predict_cds(dna)
            ctx = L.SeqContext(
                text=(tag + dna)[:T],
                tag_len=tlen,
                dna=dna,
                kingdom=kg,
                hidden_norm=norm[:T],
                cds_mask=cds_mask,
                cds_frame=cds_frame,
                gene_starts=gene_starts,
            )
            lab = np.stack([L.LABELERS[n](ctx)[:T] for n in label_names], axis=1)
            take = min(T, S - filled)
            code_buf[filled : filled + take] = codes[:take].to(torch.float16).to(auroc_device)
            dense_buf[filled : filled + take] = hd[:take].to(torch.float16).to(auroc_device)
            lab_buf[filled : filled + take] = torch.from_numpy(lab[:take]).to(auroc_device)
            filled += take
        if (start // batch_size) % 10 == 0:
            log(f"  {start + len(batch)}/{len(seqs)} seqs | buf {filled}/{S}")
    return ActivationBuffer(
        codes=code_buf[:filled].cpu().numpy(),
        dense=dense_buf[:filled].cpu().numpy(),
        labels=lab_buf[:filled].cpu().numpy(),
        label_names=list(label_names),
    )
