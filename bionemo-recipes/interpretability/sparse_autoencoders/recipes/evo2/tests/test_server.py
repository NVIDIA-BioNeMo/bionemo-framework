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

"""Server contract tests — the API the feature-explorer viz consumes.

A mocked engine (no model, CPU-only) drives the FastAPI app so these run in CI and lock the
response shapes + error codes the dashboard depends on: /health, /features, /annotate (per-base
activations), /generate. Real model inference is covered by test_steering.py.
"""

import pytest
import torch
from evo2_sae.server import build_app
from fastapi.testclient import TestClient


class FakeEngine:
    """Minimal stand-in for Evo2SAE exposing only what the server endpoints touch."""

    def __init__(self):
        self.ready = True
        self.layer = 19
        self.n_features = 4
        self.labels = {0: "feat0", 1: "feat1"}
        self.peaks = {0: 0.5}
        self.organism_tags = {"None (raw DNA)": "", "Human": "|tag|"}
        self.device = "cpu"
        self.sae_ckpt_path = "fake.pt"

    def load(self):
        self.ready = True

    def resolve_tag(self, organism, tag):
        return tag if tag is not None else self.organism_tags.get(organism)

    def encode(self, full):
        codes = torch.zeros(len(full), self.n_features)
        codes[:, 0] = 1.0  # feature 0 fires everywhere
        return codes

    def top_features(self, codes, tag_len=0, k=8):
        return [{"feature_id": 0, "label": self.labels.get(0), "max_activation": 1.0}]

    def generate(self, **kw):
        if not kw.get("prompt") and kw.get("organism") == "None (raw DNA)" and not kw.get("tag"):
            raise ValueError("need a seed")
        return {
            "generation": {"sequence": "ACGT", "activations": {0: [1.0, 1.0, 1.0, 1.0]}},
            "baseline": None,
            "features": [],
            "steered": False,
        }


@pytest.fixture
def client():
    with TestClient(build_app(FakeEngine())) as c:
        yield c


def test_health(client):
    b = client.get("/health").json()
    assert b["ready"] is True and b["layer"] == 19
    assert "None (raw DNA)" in b["organisms"]


def test_features(client):
    rows = client.get("/features").json()
    assert {"id", "label", "natural_peak"} <= set(rows[0])


def test_annotate_returns_per_base_activations(client):
    b = client.post("/annotate", json={"sequence": "ACGTACGT", "organism": "None (raw DNA)"}).json()
    assert {"sequence", "features", "bases", "tag_len", "layer", "n_tokens"} <= set(b)
    assert b["features"][0]["activations"]  # the per-base track the viz plots


def test_annotate_rejects_non_dna(client):
    assert client.post("/annotate", json={"sequence": "ZZZZ"}).status_code == 400


def test_generate_returns_sequence(client):
    b = client.post("/generate", json={"prompt": "ACGT", "organism": "None (raw DNA)"}).json()
    assert b["generation"]["sequence"]


def test_endpoints_503_until_ready():
    eng = FakeEngine()
    eng.ready = False
    eng.load = lambda: None  # startup leaves it not-ready
    with TestClient(build_app(eng)) as c:
        assert c.get("/features").status_code == 503
        assert c.post("/annotate", json={"sequence": "ACGT"}).status_code == 503
        assert c.post("/generate", json={"prompt": "ACGT", "organism": "None (raw DNA)"}).status_code == 503
