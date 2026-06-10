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

"""GPU inference tests: SAE encode + feature steering.

`test_clamp_math` (CPU) checks the steering arithmetic the forward hook applies; it needs no
model and runs in CI. The GPU tests (slow, checkpoint-gated) drive the real engine:
`test_encode_smoke` guards the bf16 encode forward, and `test_unsteered_is_dna` /
`test_steering_changes_continuation` drive `generate()` through the recipe inference engine
and assert steering (on a discovered active feature) changes the continuation only. Gated by
EVO2_CKPT_DIR + SAE_CKPT_PATH.
"""

import os

import pytest
import torch


# --------------------------------------------------------------------- CPU: clamp math
def test_clamp_math():
    """The decode-only hook applies h <- h + Σ_f (target_f - relu((h-pre_bias)@enc_f + b_f))·dec_f."""
    from evo2_sae.core import Evo2SAE

    H, F = 4, 2
    torch.manual_seed(0)
    pre_bias = torch.randn(H)
    enc = torch.randn(F, H)
    bias = torch.randn(F)
    dec = torch.randn(H, F)
    specs = [(enc[f], float(bias[f]), dec[:, f], float(f + 1)) for f in range(F)]

    eng = Evo2SAE.__new__(Evo2SAE)  # bare instance — exercise the hook only
    hook = eng._clamp_hook(specs, pre_bias)

    h = torch.randn(1, 1, H)  # one decode token: [S=1, B=1, H]
    out = hook(None, None, h)

    xc = h - pre_bias
    expected = h.clone()
    for enc_f, b_f, dec_f, target in specs:
        a = torch.relu(xc @ enc_f + b_f)
        expected = expected + (target - a).unsqueeze(-1) * dec_f
    torch.testing.assert_close(out, expected)

    # prefill (S>1) must be left untouched (continuation-only steering)
    prefill = torch.randn(5, 1, H)
    assert torch.equal(hook(None, None, prefill), prefill)


# --------------------------------------------------------------------- GPU: real generation
_CKPT = os.environ.get("EVO2_CKPT_DIR")
_SAE = os.environ.get("SAE_CKPT_PATH")
_LAYER = int(os.environ.get("EMBEDDING_LAYER", "19"))
_PROMPT = "ATGGCCGAATTCGGCACGAGGACGTGCTGAAAGCTAGCTAGGCTAACCGGTTACGTGCAT"
_ORG = "Human"


@pytest.fixture(scope="module")
def engine():
    """Load the Evo2 + SAE engine once (skips unless CUDA + checkpoints are available)."""
    if not torch.cuda.is_available():
        pytest.skip("steering tests require CUDA")
    if not (_CKPT and _SAE):
        pytest.skip("set EVO2_CKPT_DIR and SAE_CKPT_PATH to run the steering tests")
    from evo2_sae import Evo2SAE

    return Evo2SAE(evo2_ckpt_dir=_CKPT, sae_ckpt_path=_SAE, layer=_LAYER).load()


def _gen(engine, features):
    torch.manual_seed(0)
    return engine.generate(prompt=_PROMPT, organism=_ORG, features=features, n_tokens=48, temperature=0.0, top_k=1)


def _tag(engine):
    return engine.resolve_tag(_ORG, None) or ""


@pytest.mark.slow
def test_encode_smoke(engine):
    """encode runs the truncated bf16 forward and returns finite per-feature codes (>=1 firing).

    Guards the TransformerEngine bf16/fp32 autocast path: a dtype mismatch would crash here.
    """
    codes = engine.encode(_tag(engine) + _PROMPT)
    assert codes.ndim == 2 and codes.shape[1] == engine.n_features
    assert torch.isfinite(codes).all()
    assert (codes > 0).any()


@pytest.mark.slow
def test_unsteered_is_dna(engine):
    """Unsteered generation yields a non-empty ACGT string (Evo2 stays in-distribution)."""
    seq = _gen(engine, [])["generation"]["sequence"]
    assert seq and set(seq) <= set("ACGTN")


@pytest.mark.slow
def test_steering_changes_continuation(engine):
    """Clamping a KNOWN-ACTIVE feature hard changes the continuation; empty clamp is a no-op.

    Discovers the most-active feature on the prompt (SAE-agnostic) so the clamp has real signal —
    an arbitrary/dead feature would leave greedy decoding unchanged and make this test useless.
    """
    per = engine.encode(_tag(engine) + _PROMPT).max(dim=0).values
    fid, peak = int(per.argmax()), float(per.max())
    base = _gen(engine, [])["generation"]["sequence"]
    steered = _gen(engine, [{"feature_id": fid, "strength": max(peak * 3.0, 50.0)}])["generation"]["sequence"]
    assert steered != base  # the clamp on an active feature changed the continuation
    assert _gen(engine, [])["generation"]["sequence"] == base  # determinism + empty-clamp no-op
