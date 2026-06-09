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

"""Parity test for the cached (KV + Hyena SSM) generation path.

The cached decode path (`generate(use_cache=True)`) must produce the *same* token
sequence as the cache-free reference (`use_cache=False`) — both unsteered and with an
SAE-feature clamp — since the only difference is reusing the attention KV + Hyena
conv-state cache instead of a full forward per step. Greedy decoding makes the
comparison deterministic. GPU-only; skipped unless EVO2_CKPT_DIR + SAE_CKPT_PATH point
at an MBridge checkpoint and a trained SAE.
"""

import os

import pytest
import torch


pytestmark = pytest.mark.slow

_CKPT = os.environ.get("EVO2_CKPT_DIR")
_SAE = os.environ.get("SAE_CKPT_PATH")
_LAYER = int(os.environ.get("EMBEDDING_LAYER", "19"))
_PROMPT = "ACGTACGTACGTACGTACGT"
_N_TOKENS = 64


@pytest.fixture(scope="module")
def engine():
    """Load the Evo2 + SAE engine once (skips unless CUDA + checkpoints are available)."""
    if not torch.cuda.is_available():
        pytest.skip("cached-decode parity test requires CUDA")
    if not (_CKPT and _SAE):
        pytest.skip("set EVO2_CKPT_DIR and SAE_CKPT_PATH to run the parity test")
    from evo2_sae_infer import Evo2SAE

    return Evo2SAE(evo2_ckpt_dir=_CKPT, sae_ckpt_path=_SAE, layer=_LAYER).load()


def _generate(engine, *, use_cache, features):
    """Greedy (deterministic) generation for the fixed prompt; returns the DNA string."""
    torch.manual_seed(0)
    return engine.generate(
        prompt=_PROMPT,
        organism="None (raw DNA)",
        features=features,
        n_tokens=_N_TOKENS,
        temperature=0.0,
        top_k=1,
        use_cache=use_cache,
    )["generation"]["sequence"]


def test_cached_matches_cachefree_unsteered(engine):
    """Cached decode reproduces cache-free decode exactly, with no steering."""
    assert _generate(engine, use_cache=True, features=[]) == _generate(engine, use_cache=False, features=[])


def test_cached_matches_cachefree_steered(engine):
    """Cached decode reproduces cache-free decode exactly under an SAE-feature clamp."""
    feats = [{"feature_id": 0, "strength": 5.0}]
    assert _generate(engine, use_cache=True, features=feats) == _generate(engine, use_cache=False, features=feats)
