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

import math

import pytest
import torch
from evo2_sae import Evo2SAE
from evo2_sae.core import MAX_CLAMP_STRENGTH, _sanitize_steering


# The delta-clamp math + decode-only/prefill behavior is covered against the production
# hook in sae/tests/test_steering.py (sae.steering.clamp_hook). This file covers the
# engine-level input guards (CPU) and the live steering smoke tests (GPU).


# --------------------------------------------------- CPU: steering input guards (no model)
# Each guards against a CUDA device-side assert that would corrupt the context and wedge the
# server: out-of-range id (indexes off the SAE codes), extreme clamp (NaN logits), and
# temperature 0 (sampler divides logits by temperature -> NaN under multinomial).
def test_sanitize_rejects_out_of_range_id():
    """feature_id outside [0, n_features) -> ValueError (server maps to 400, not a wedge)."""
    with pytest.raises(ValueError):
        _sanitize_steering([{"feature_id": 70000, "strength": 1.0}], n_features=65536, temperature=1.0, top_k=0)
    with pytest.raises(ValueError):
        _sanitize_steering([{"feature_id": -1, "strength": 1.0}], n_features=65536, temperature=1.0, top_k=0)


def test_sanitize_caps_clamp_magnitude():
    """An extreme target is capped to ±MAX_CLAMP_STRENGTH (prevents NaN logits)."""
    clamps, _, _, _ = _sanitize_steering([{"feature_id": 5, "strength": 99999.0}], 65536, 1.0, 0)
    assert clamps[5] == MAX_CLAMP_STRENGTH
    clamps, _, _, _ = _sanitize_steering([{"feature_id": 5, "strength": -99999.0}], 65536, 1.0, 0)
    assert clamps[5] == -MAX_CLAMP_STRENGTH


def test_sanitize_temperature_zero_forces_greedy_topk():
    """temperature<=0 (divide-by-zero in the sampler) is coerced to greedy top-1."""
    _, _, _, top_k = _sanitize_steering([{"feature_id": 5, "strength": 1.0}], 65536, 0.0, 0)
    assert top_k == 1  # bumped from 0 so the logits/temperature path is skipped


def test_sanitize_passthrough():
    """Valid inputs pass through unchanged — no spurious capping or top_k change."""
    clamps, fids, temp, top_k = _sanitize_steering([{"feature_id": 5, "strength": 2.0}], 65536, 0.8, 0)
    assert clamps == {5: 2.0} and fids == [5] and temp == 0.8 and top_k == 0


def test_sanitize_neutralizes_nonfinite_strength():
    """NaN / ±inf are capped to a finite magnitude <= MAX_CLAMP_STRENGTH, never propagated.

    A non-finite clamp would blow the logits to NaN (a CUDA device-assert that wedges the
    server) — exactly what this guard exists to prevent.
    """
    for bad in (math.nan, math.inf, -math.inf):
        clamps, _, _, _ = _sanitize_steering([{"feature_id": 3, "strength": bad}], 65536, 1.0, 0)
        assert math.isfinite(clamps[3]) and abs(clamps[3]) <= MAX_CLAMP_STRENGTH


def test_sanitize_feature_contract():
    """Duplicate ids collapse (last wins), missing strength defaults to 1.0, strength 0 is kept."""
    clamps, fids, _, _ = _sanitize_steering(
        [
            {"feature_id": 5, "strength": 1.0},
            {"feature_id": 5, "strength": 2.0},  # duplicate id
            {"feature_id": 7},  # missing strength
            {"feature_id": 9, "strength": 0.0},  # suppress-to-zero
        ],
        65536,
        1.0,
        0,
    )
    assert clamps[5] == 2.0  # duplicate id -> last value wins
    assert clamps[7] == 1.0  # missing strength -> default 1.0
    assert clamps[9] == 0.0  # strength 0 retained, not dropped
    assert fids == [5, 7, 9]  # one entry per distinct feature, in insertion order


# --------------------------------------------------------------------- GPU: real generation
_PROMPT = "ATGGCCGAATTCGGCACGAGGACGTGCTGAAAGCTAGCTAGGCTAACCGGTTACGTGCAT"
_ORG = "Human"


@pytest.fixture(scope="module")
def engine(evo2_ckpt_dir, sae_ckpt_path, embedding_layer):
    """Load the Evo2 + SAE engine once (skips unless CUDA is available).

    ``evo2_ckpt_dir`` / ``sae_ckpt_path`` come from conftest: they honor EVO2_CKPT_DIR /
    SAE_CKPT_PATH for manual or 7B runs, and otherwise build the 1B-8k-bf16 + a synthesized
    tiny SAE in CI (memory-gated). Those fixtures skip if no checkpoint can be produced.
    """
    if not torch.cuda.is_available():
        pytest.skip("steering tests require CUDA")
    return Evo2SAE(evo2_ckpt_dir=evo2_ckpt_dir, sae_ckpt_path=sae_ckpt_path, layer=embedding_layer).load()


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
    Also exercises ``compare_baseline=True`` (returns the steered generation + the unsteered baseline
    in one call).
    """
    per = engine.encode(_tag(engine) + _PROMPT).max(dim=0).values
    fid, peak = int(per.argmax()), float(per.max())
    feature = {"feature_id": fid, "strength": max(peak * 3.0, 50.0)}
    base = _gen(engine, [])["generation"]["sequence"]
    steered = _gen(engine, [feature])["generation"]["sequence"]
    assert steered != base  # the clamp on an active feature changed the continuation
    assert _gen(engine, [])["generation"]["sequence"] == base  # determinism + empty-clamp no-op

    # compare_baseline=True returns both halves in one call.
    both = engine.generate(
        prompt=_PROMPT, organism=_ORG, features=[feature], n_tokens=48, temperature=0.0, top_k=1, compare_baseline=True
    )
    assert both["generation"]["sequence"] == steered  # steered half matches
    assert both["baseline"]["sequence"] == base  # baseline half == unsteered


@pytest.mark.slow
def test_encode_batch_preserves_order_and_lengths(engine):
    """Batched encode keeps input order + per-sequence length, with finite codes (real model, no mock)."""
    seqs = [_tag(engine) + _PROMPT[:20], _tag(engine) + _PROMPT, _tag(engine) + _PROMPT[:40]]
    batch = engine.encode_batch(seqs)
    assert [c.shape[0] for c in batch] == [len(engine.tokenize(s)) for s in seqs]  # order + length preserved
    assert all(c.shape[1] == engine.n_features and torch.isfinite(c).all() for c in batch)


@pytest.mark.slow
def test_max_clamp_strength_stays_in_distribution(engine):
    """Clamping at MAX_CLAMP_STRENGTH still produces valid DNA — proves the cap is a *safe* level.

    The CPU sanitize test only checks the value is capped; this proves the capped magnitude
    doesn't blow the logits to NaN (a CUDA device-assert that would wedge the server).
    """
    fid = int(engine.encode(_tag(engine) + _PROMPT).max(dim=0).values.argmax())
    seq = _gen(engine, [{"feature_id": fid, "strength": MAX_CLAMP_STRENGTH}])["generation"]["sequence"]
    assert seq and set(seq) <= set("ACGTN")
