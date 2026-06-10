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

r"""Loss recovered (fidelity) for the Evo2 SAE — reuses sae.eval.loss_recovered (Jared Wilber).

    loss_recovered = 1 - (CE_sae - CE_clean) / (CE_zero - CE_clean)

We just provide Evo2-specific callables to his generic evaluator:
  - get_hiddens(batch): capture the layer-`L` residual via a forward hook
  - compute_ce(batch, override): full-model next-token CE, optionally patching the
    layer-`L` output with `override` (zero-ablation or SAE reconstruction)
The SAE reconstruction is DENORMALIZED per token (normalize_input) so it is in the
raw residual space the layer actually emits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as Fn


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from evo2_buffer import sample_sequences  # noqa: E402
from evo2_sae_infer.core import Evo2SAE  # noqa: E402
from sae.eval.loss_recovered import evaluate_loss_recovered  # noqa: E402  (Jared's code)


KINGDOM_TAGS = {"prok": "|d__Bacteria|", "euk": "|d__Eukaryota|"}


class SAEWrap(nn.Module):
    """sae.forward(x[N,H]) -> (recon, codes) in RAW residual space (denormalized)."""

    def __init__(self, sae):  # noqa: D107
        super().__init__()
        self.sae = sae

    def forward(self, x):  # noqa: D102
        s = self.sae
        codes = s.encode(x)  # encode normalizes internally if normalize_input
        recon = s.decoder(codes) + s.pre_bias
        if getattr(s, "normalize_input", False):
            mu = x.mean(-1, keepdim=True)
            std = x.std(-1, keepdim=True) + 1e-8
            recon = recon * std + mu
        return recon, codes


class L26Hook:  # noqa: D101
    def __init__(self):  # noqa: D107
        self.mode = "off"  # off | capture | replace
        self.override = None
        self.captured = None

    def __call__(self, module, inp, output):  # noqa: D102
        hs = output[0] if isinstance(output, tuple) else output
        if self.mode == "replace" and self.override is not None:
            new = self.override.to(hs.dtype)
            return (new, *output[1:]) if isinstance(output, tuple) else new
        if self.mode == "capture":
            self.captured = hs.detach()
        return output


def main():  # noqa: D103
    ap = argparse.ArgumentParser()
    ap.add_argument("--evo2-ckpt-dir", required=True)
    ap.add_argument("--sae-checkpoint", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--n-seqs", type=int, default=80)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    dev = args.device

    engine = Evo2SAE(args.evo2_ckpt_dir, args.sae_checkpoint, args.layer, device=dev).load()
    from megatron.core.utils import unwrap_model

    gen = engine._ensure_gen_model()
    layer = unwrap_model(gen).decoder.layers[args.layer]
    hook = L26Hook()
    layer.register_forward_hook(hook)

    pairs = sample_sequences(
        args.fasta, args.n_seqs * args.seq_len, args.seq_len, kingdoms=["prok", "euk"], seed=args.seed
    )[: args.n_seqs]
    batches = []
    for kingdom, dna in pairs:
        ids = engine.tokenize(KINGDOM_TAGS[kingdom] + dna)
        if len(ids) > 4:
            batches.append(torch.tensor([ids], dtype=torch.long, device=dev))

    def fwd(ids):
        return gen(input_ids=ids, position_ids=None, attention_mask=None, labels=None, runtime_gather_output=True)

    def get_hiddens(batch):
        hook.mode = "capture"
        fwd(batch)
        hook.mode = "off"
        return hook.captured  # [S, 1, H]

    def compute_ce(batch, override):
        if override is None:
            hook.mode = "off"
        else:
            hook.mode = "replace"
            hook.override = override
        logits = fwd(batch)
        hook.mode = "off"
        hook.override = None
        lg = logits[0, :-1].float()  # [S-1, V]
        tgt = batch[0, 1:]
        ce = Fn.cross_entropy(lg, tgt, reduction="sum")
        return float(ce), int(tgt.numel())

    with engine._lock:
        res = evaluate_loss_recovered(SAEWrap(engine.sae), batches, get_hiddens, compute_ce, device=dev)
    print("\n==== Evo2 7B layer-%d SAE — loss recovered ====" % args.layer)
    print(res)
    print(
        f"loss_recovered = {res.loss_recovered:.3f}  "
        f"(CE clean={res.ce_original:.3f}, SAE={res.ce_sae:.3f}, zero={res.ce_zero:.3f}, n_tok={res.n_tokens})"
    )


if __name__ == "__main__":
    main()
