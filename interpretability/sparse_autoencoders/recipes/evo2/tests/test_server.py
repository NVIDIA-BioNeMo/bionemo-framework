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
from evo2_sae.server import build_app
from fastapi.testclient import TestClient


# FakeEngine lives in conftest.py — shared with test_cli.py so both suites mock the same surface.


@pytest.fixture
def client(fake_engine):
    with TestClient(build_app(fake_engine)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200  # 200 only when ready
    b = r.json()
    assert b["ready"] is True and b["layer"] == 19
    assert "None (raw DNA)" in b["organisms"]


def test_annotate_rejects_too_long(client, fake_engine):
    seq = "A" * (fake_engine.max_seq_len + 1)  # exceeds the context budget
    assert client.post("/annotate", json={"sequence": seq}).status_code == 413


def test_features(client):
    rows = client.get("/features").json()
    assert {"id", "label", "natural_peak"} <= set(rows[0])


def test_annotate_returns_per_base_activations(client):
    b = client.post("/annotate", json={"sequence": "ACGTACGT", "organism": "None (raw DNA)"}).json()
    assert {"sequence", "features", "bases", "tag_len", "layer", "n_tokens"} <= set(b)
    assert b["features"][0]["activations"]  # the per-base track the viz plots


def test_annotate_rejects_non_dna(client):
    assert client.post("/annotate", json={"sequence": "ZZZZ"}).status_code == 400


def test_annotate_pick_mode(client):
    b = client.post("/annotate", json={"sequence": "ACGTACGT", "mode": "pick", "feature_ids": [1]}).json()
    assert [f["feature_id"] for f in b["features"]] == [1]
    assert b["features"][0]["activations"]  # per-base track returned for the picked feature


def test_annotate_pick_requires_ids(client):
    assert client.post("/annotate", json={"sequence": "ACGT", "mode": "pick"}).status_code == 400


def test_annotate_pick_rejects_out_of_range_id(client, fake_engine):
    # user-supplied pick ids: an over-range id must 400 (not 500/IndexError), a negative one must
    # 400 (not silently index the wrong feature via torch negative-indexing).
    over = client.post("/annotate", json={"sequence": "ACGT", "mode": "pick", "feature_ids": [fake_engine.n_features]})
    assert over.status_code == 400
    neg = client.post("/annotate", json={"sequence": "ACGT", "mode": "pick", "feature_ids": [-1]})
    assert neg.status_code == 400


def test_annotate_rejects_invalid_mode(client):
    assert client.post("/annotate", json={"sequence": "ACGT", "mode": "bogus"}).status_code == 400


def test_annotate_clamps_k_into_range(client, fake_engine):
    client.post("/annotate", json={"sequence": "ACGT", "k": 999})
    assert fake_engine.last_k == 64  # upper bound
    client.post("/annotate", json={"sequence": "ACGT", "k": 0})
    assert fake_engine.last_k == 1  # lower bound


def test_generate_returns_sequence(client):
    b = client.post("/generate", json={"prompt": "ACGT", "organism": "None (raw DNA)"}).json()
    assert b["generation"]["sequence"]


def test_generate_rejects_out_of_range_feature(client):
    r = client.post("/generate", json={"prompt": "ACGT", "features": [{"feature_id": 999}]})
    assert r.status_code == 400  # the wedge guard, surfaced to the client


def test_generate_compare_baseline(client):
    b = client.post(
        "/generate",
        json={"prompt": "ACGT", "features": [{"feature_id": 0, "strength": 5.0}], "compare_baseline": True},
    ).json()
    assert b["baseline"]["sequence"]  # unsteered comparison returned alongside the steered one


def test_rejects_oversized_body(monkeypatch, fake_engine):
    monkeypatch.setenv("MAX_BODY_BYTES", "100")
    with TestClient(build_app(fake_engine)) as c:
        assert c.post("/annotate", json={"sequence": "ACGT" * 100}).status_code == 413


def test_endpoints_503_until_ready(fake_engine):
    fake_engine.ready = False
    fake_engine.load = lambda: None  # startup leaves it not-ready
    with TestClient(build_app(fake_engine)) as c:
        assert c.get("/health").status_code == 503  # readiness probe sheds the pod
        assert c.get("/features").status_code == 503
        assert c.post("/annotate", json={"sequence": "ACGT"}).status_code == 503
        assert c.post("/generate", json={"prompt": "ACGT", "organism": "None (raw DNA)"}).status_code == 503
