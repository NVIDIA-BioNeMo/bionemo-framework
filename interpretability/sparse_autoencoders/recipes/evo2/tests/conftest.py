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

"""Shared checkpoint fixtures for the Evo2-SAE GPU tests.

The GPU steering tests need an mbridge Evo2 checkpoint + an SAE checkpoint. These fixtures
resolve them two ways:

  * **Manual / 7B:** if ``EVO2_CKPT_DIR`` / ``SAE_CKPT_PATH`` are set, use those as-is.
  * **CI:** otherwise build the **1B-8k-bf16** the way ``evo2_megatron``'s own tests do
    (``bionemo_load`` -> ``run_nemo2_to_mbridge``) and synthesize a tiny random TopK SAE
    matching the 1B layer dim. Gated by a GPU-memory check (not ``skipif(CI)``), so it runs
    on the L4 the same way evo2's ``test_batch_generate`` 1B case does, and skips on smaller
    GPUs / when the checkpoint can't be fetched.

The fixtures are lazy (heavy imports + the build happen only when a GPU test requests the
engine), so the CPU-only sanitize tests collect and run without CUDA or bionemo.evo2.
"""

import os

import pytest
import torch
from sae.architectures import TopKSAE


# The 1B (Hyena1bModelProvider) needs ~18 GB to load+generate; the L4 (24 GB) fits, smaller
# GPUs skip. The 7B path is reached only via EVO2_CKPT_DIR (manual), never auto-built here.
_MIN_GPU_GB = 20
_EVO2_1B_HIDDEN = 1920  # Hyena1bModelProvider.hidden_size -> SAE input_dim


@pytest.fixture(scope="session")
def evo2_ckpt_dir(tmp_path_factory) -> str:
    """MBridge Evo2 checkpoint dir for the engine.

    Honors ``EVO2_CKPT_DIR``; otherwise builds the 1B-8k-bf16 from NGC (memory-gated).
    """
    env = os.environ.get("EVO2_CKPT_DIR")
    if env:
        return env

    if not torch.cuda.is_available():
        pytest.skip("building/loading the 1B requires CUDA (or set EVO2_CKPT_DIR)")
    total_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3
    if total_gb < _MIN_GPU_GB:
        pytest.skip(f"need >= {_MIN_GPU_GB} GB GPU for the 1B; have {total_gb:.1f} GB")

    try:
        # The NGC loader lives in bionemo.common on the migrated recipes layout; older framework
        # builds expose it as bionemo.core. Support both so the fixture works either way.
        try:
            from bionemo.common.data.load import load as bionemo_load
        except ImportError:
            from bionemo.core.data.load import load as bionemo_load
        from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
        from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge
    except ImportError as e:  # pragma: no cover - environment guard
        pytest.skip(f"bionemo.evo2 not importable (build via .ci_build.sh): {e}")

    try:
        nemo2_ckpt_path = bionemo_load("evo2/1b-8k-bf16:1.0")
    except ValueError as e:
        pytest.skip(f"could not fetch evo2/1b-8k-bf16 (try BIONEMO_DATA_SOURCE=pbss): {e}")

    out_dir = tmp_path_factory.mktemp("evo2_1b_mbridge_session")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_ckpt_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=out_dir / "evo2_1b_mbridge",
        model_size="evo2_1b_base",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return str(mbridge_ckpt_dir / "iter_0000001")


@pytest.fixture(scope="session")
def sae_ckpt_path(tmp_path_factory) -> str:
    """SAE checkpoint for the engine.

    Honors ``SAE_CKPT_PATH``; otherwise synthesizes a tiny random TopK SAE whose ``input_dim``
    matches the 1B layer (1920). Random weights — this exercises the load/encode/steer code
    path, not SAE fidelity, which is the point of the CI smoke.
    """
    env = os.environ.get("SAE_CKPT_PATH")
    if env:
        return env

    torch.manual_seed(0)
    sae = TopKSAE(input_dim=_EVO2_1B_HIDDEN, hidden_dim=256, top_k=16)
    with torch.no_grad():
        # Real (random) signal so encode produces nonzero codes and steering has something to clamp.
        sae.decoder.weight.normal_(0, 0.02)
        sae.encoder.weight.copy_(sae.decoder.weight.t())

    out = tmp_path_factory.mktemp("tiny_sae") / "tiny_sae_1b.pt"
    torch.save({"model_config": sae._get_config(), "model_state_dict": sae.state_dict()}, out)
    return str(out)


@pytest.fixture(scope="session")
def embedding_layer() -> int:
    """Layer whose residual stream the SAE reads/steers (1B has 25 layers; default 19)."""
    return int(os.environ.get("EMBEDDING_LAYER", "19"))


# ---------------------------------------------------------------------------------------------
# CPU fixtures for the serve layer (#1637). `FakeEngine` is the one mock the CLI tests
# (test_cli.py) and the server contract tests (test_server.py) both drive — no model, CPU-only —
# so the two suites stay in lockstep on the engine surface the server/CLI touch.
# ---------------------------------------------------------------------------------------------
from evo2_sae import core  # noqa: E402


class FakeEngine:
    """Minimal stand-in for Evo2SAE exposing only what the CLI + server touch."""

    def __init__(self):
        self.ready = True
        self.layer = 19
        self.n_features = 4
        self.labels = {0: "feat0", 1: "feat1"}
        self.peaks = {0: 0.5}
        # Curated per-feature extras (steerable/description/auroc) — feat0 is flagged steerable so tests
        # can assert these flow through /features; feat1 has none (defaults to steerable=False).
        self.feature_extra = {0: {"steerable": True, "description": "fires on feat0 motifs", "auroc": 0.91}}
        self.organism_tags = {"None (raw DNA)": "", "Human": "|tag|"}
        self.device = "cpu"
        self.sae_ckpt_path = "fake.pt"
        self.max_seq_len = 8192
        self.gen_kwargs = None  # records the last generate() call for CLI assertions
        self.last_k = None  # records the k top_features() was last called with

    def load(self):
        self.ready = True
        return self  # the CLI uses `_engine(args).load()`

    def resolve_tag(self, organism, tag):
        return tag if tag is not None else self.organism_tags.get(organism)

    def encode(self, full):
        codes = torch.zeros(len(full), self.n_features)
        codes[:, 0] = 1.0  # feature 0 fires everywhere
        return codes

    def encode_batch(self, seqs, batch_size=8, cancel=None):
        # Mirror the real engine's cooperative-cancel checkpoint (real encode_batch checks between
        # micro-batches). embed_bundle — bound from the real Evo2SAE below — forwards cancel here.
        if cancel is not None and cancel():
            raise core.RequestAborted("cancelled")
        return [self.encode(s) for s in seqs]

    def top_features(self, codes, tag_len=0, k=8):
        self.last_k = k  # so tests can assert the server clamped k into [1, 64]
        feats = [{"feature_id": i, "label": self.labels.get(i), "max_activation": 1.0} for i in range(self.n_features)]
        return feats[:k]

    def generate(self, **kw):
        self.gen_kwargs = kw
        # Mirror the real engine: clamps normalize through the shared parser (a malformed --clamp
        # raises), out-of-range ids are rejected (the wedge guard), and a seedless request is too.
        specs = [core.parse_clamp_spec(f) for f in (kw.get("features") or [])]
        bad = [s["feature_id"] for s in specs if not (0 <= s["feature_id"] < self.n_features)]
        if bad:
            raise ValueError(f"feature_id(s) {bad} out of range [0, {self.n_features})")
        prompt = core.clean_dna(kw.get("prompt", ""))
        if not prompt and kw.get("organism") == "None (raw DNA)" and not kw.get("tag"):
            raise ValueError("need a seed")
        if len(prompt) > self.max_seq_len:  # mirror the real over-context reject (server -> 413)
            raise ValueError(f"prompt too long: {len(prompt)} > max_seq_len ({self.max_seq_len})")
        # Match the real engine's response: features are feat_meta dicts keyed {id, label, strength}
        # (NOT feature_id) — this is the shape the dashboard consumes through /generate.
        feat_meta = [
            {"id": s["feature_id"], "label": self.labels.get(s["feature_id"]), "strength": s["strength"]}
            for s in specs
        ]
        resp = {
            "prompt": prompt,
            "organism": kw.get("organism"),
            "tag": kw.get("tag"),
            "steered": bool(specs),
            "features": feat_meta,
            "generation": {"sequence": "ACGT", "activations": {0: [1.0, 1.0, 1.0, 1.0]}},
            "baseline": None,
        }
        if kw.get("compare_baseline") and specs:
            resp["baseline"] = {"sequence": "TTTT", "activations": {}}
        return resp

    # /gene_embed delegates pooling/packing to the real Evo2SAE.embed_bundle (uses only
    # encode_batch + labels), so bind it here to exercise the shared implementation.
    embed_bundle = core.Evo2SAE.embed_bundle


@pytest.fixture
def fake_engine():
    """A fresh CPU FakeEngine instance."""
    return FakeEngine()
