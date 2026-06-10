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

"""Tests for SAE feature steering during generation.

`test_clamp_math` (CPU) checks the steering arithmetic the forward hook applies; it needs
no model and runs in CI. `test_steering_*` (GPU, slow, checkpoint-gated) drive the real
`generate()` through the recipe inference engine and assert the steering hook fires on the
continuation only and changes the output. Gated by EVO2_CKPT_DIR + SAE_CKPT_PATH.
"""

import os

import pytest
import torch


# --------------------------------------------------------------------- CPU: clamp math
def test_clamp_math():
    """The decode-only hook applies h <- h + Σ_f (target_f - relu((h-pre_bias)@enc_f + b_f))·dec_f."""
    from evo2_sae_infer.core import Evo2SAE

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
_PROMPT = "ACGTACGTACGTACGTACGT"


@pytest.fixture(scope="module")
def engine():
    """Load the Evo2 + SAE engine once (skips unless CUDA + checkpoints are available)."""
    if not torch.cuda.is_available():
        pytest.skip("steering tests require CUDA")
    if not (_CKPT and _SAE):
        pytest.skip("set EVO2_CKPT_DIR and SAE_CKPT_PATH to run the steering tests")
    from evo2_sae_infer import Evo2SAE

    return Evo2SAE(evo2_ckpt_dir=_CKPT, sae_ckpt_path=_SAE, layer=_LAYER).load()


def _gen(engine, features):
    torch.manual_seed(0)
    return engine.generate(
        prompt=_PROMPT, organism="None (raw DNA)", features=features, n_tokens=48, temperature=0.0, top_k=1
    )


@pytest.mark.slow
def test_unsteered_is_dna(engine):
    """Unsteered generation yields a non-empty ACGT string (Evo2 stays in-distribution)."""
    seq = _gen(engine, [])["generation"]["sequence"]
    assert seq and set(seq) <= set("ACGTN")


@pytest.mark.slow
def test_steering_changes_output(engine):
    """A strong clamp changes the generated continuation; an empty clamp is a deterministic no-op."""
    base = _gen(engine, [])["generation"]["sequence"]
    steered = _gen(engine, [{"feature_id": 0, "strength": 10.0}])["generation"]["sequence"]
    assert steered != base
    assert _gen(engine, [])["generation"]["sequence"] == base  # determinism / no-op
