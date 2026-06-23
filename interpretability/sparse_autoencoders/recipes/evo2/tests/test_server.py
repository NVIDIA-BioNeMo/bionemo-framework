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


@pytest.fixture
def dist_dir(tmp_path):
    """A minimal stand-in for a built frontend (feature_explorer/dist) — index + one asset."""
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text('<!doctype html><html><body><div id="root"></div></body></html>')
    (d / "assets" / "app.js").write_text("console.log('dashboard')")
    return d


@pytest.fixture
def static_client(fake_engine, dist_dir):
    """A client for the single-container shape: API under /api + the frontend served at /."""
    with TestClient(build_app(fake_engine, static_dir=str(dist_dir))) as c:
        yield c


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200  # 200 only when ready
    b = r.json()
    assert b["ready"] is True and b["layer"] == 19
    assert "None (raw DNA)" in b["organisms"]


def test_annotate_rejects_too_long(client, fake_engine):
    seq = "A" * (fake_engine.max_seq_len + 1)  # exceeds the context budget
    assert client.post("/api/annotate", json={"sequence": seq}).status_code == 413


def test_features(client):
    rows = client.get("/api/features").json()
    assert {"id", "label", "natural_peak"} <= set(rows[0])


def test_annotate_returns_per_base_activations(client):
    b = client.post("/api/annotate", json={"sequence": "ACGTACGT", "organism": "None (raw DNA)"}).json()
    assert {"sequence", "features", "bases", "tag_len", "layer", "n_tokens"} <= set(b)
    assert b["features"][0]["activations"]  # the per-base track the viz plots


def test_annotate_rejects_non_dna(client):
    assert client.post("/api/annotate", json={"sequence": "ZZZZ"}).status_code == 400


def test_annotate_pick_mode(client):
    b = client.post("/api/annotate", json={"sequence": "ACGTACGT", "mode": "pick", "feature_ids": [1]}).json()
    assert [f["feature_id"] for f in b["features"]] == [1]
    assert b["features"][0]["activations"]  # per-base track returned for the picked feature


def test_annotate_pick_requires_ids(client):
    assert client.post("/api/annotate", json={"sequence": "ACGT", "mode": "pick"}).status_code == 400


def test_annotate_pick_rejects_out_of_range_id(client, fake_engine):
    # user-supplied pick ids: an over-range id must 400 (not 500/IndexError), a negative one must
    # 400 (not silently index the wrong feature via torch negative-indexing).
    over = client.post(
        "/api/annotate", json={"sequence": "ACGT", "mode": "pick", "feature_ids": [fake_engine.n_features]}
    )
    assert over.status_code == 400
    neg = client.post("/api/annotate", json={"sequence": "ACGT", "mode": "pick", "feature_ids": [-1]})
    assert neg.status_code == 400


def test_annotate_rejects_invalid_mode(client):
    assert client.post("/api/annotate", json={"sequence": "ACGT", "mode": "bogus"}).status_code == 400


def test_annotate_clamps_k_into_range(client, fake_engine):
    client.post("/api/annotate", json={"sequence": "ACGT", "k": 999})
    assert fake_engine.last_k == 64  # upper bound
    client.post("/api/annotate", json={"sequence": "ACGT", "k": 0})
    assert fake_engine.last_k == 1  # lower bound


def test_generate_returns_sequence(client):
    b = client.post("/api/generate", json={"prompt": "ACGT", "organism": "None (raw DNA)"}).json()
    assert b["generation"]["sequence"]


def test_gene_embed_returns_decodable_matrix(client):
    import base64

    import numpy as np

    genes = [{"symbol": "g1", "sequence": "ACGTACGT"}, {"symbol": "g2", "sequence": "TTTTGGGG"}]
    b = client.post("/gene_embed", json={"genes": genes, "min_firing": 1}).json()
    assert {"G_b64", "Gmax_b64", "n_features", "n_genes", "genes", "feature_ids"} <= set(b)
    assert b["n_genes"] == 2 and len(b["genes"]) == 2
    # only firing columns are shipped: feature_ids maps column -> real SAE feature id
    assert len(b["feature_ids"]) == b["n_features"]
    assert b["feature_ids"] == [0]  # the fake fires feature 0 in every sequence, nothing else
    g = np.frombuffer(base64.b64decode(b["G_b64"]), dtype=np.float32)
    assert g.size == b["n_genes"] * b["n_features"]  # [n_genes x n_firing_features], what the client UMAPs


def test_generate_rejects_out_of_range_feature(client):
    r = client.post("/api/generate", json={"prompt": "ACGT", "features": [{"feature_id": 999}]})
    assert r.status_code == 400  # the wedge guard, surfaced to the client


def test_generate_rejects_too_long(client, fake_engine):
    seq = "A" * (fake_engine.max_seq_len + 1)  # exceeds the context budget -> 413 (parity w/ annotate)
    assert client.post("/api/generate", json={"prompt": seq}).status_code == 413


def test_generate_compare_baseline(client):
    b = client.post(
        "/api/generate",
        json={"prompt": "ACGT", "features": [{"feature_id": 0, "strength": 5.0}], "compare_baseline": True},
    ).json()
    assert b["baseline"]["sequence"]  # unsteered comparison returned alongside the steered one


def test_rejects_oversized_body(monkeypatch, fake_engine):
    monkeypatch.setenv("MAX_BODY_BYTES", "100")
    with TestClient(build_app(fake_engine)) as c:
        assert c.post("/api/annotate", json={"sequence": "ACGT" * 100}).status_code == 413


def test_endpoints_503_until_ready(fake_engine):
    fake_engine.ready = False
    fake_engine.load = lambda: None  # startup leaves it not-ready
    with TestClient(build_app(fake_engine)) as c:
        assert c.get("/api/health").status_code == 503  # readiness probe sheds the pod
        assert c.get("/api/features").status_code == 503
        assert c.post("/api/annotate", json={"sequence": "ACGT"}).status_code == 503
        assert c.post("/api/generate", json={"prompt": "ACGT", "organism": "None (raw DNA)"}).status_code == 503


# ---------------------------------------------------------- single-container: SPA at / + API at /api
def test_serves_spa_index(static_client):
    """With a built frontend mounted, GET / returns the SPA index (HTML), not an API response."""
    r = static_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert 'id="root"' in r.text


def test_serves_static_asset(static_client):
    """Frontend assets are served from the mount (so the SPA's JS/CSS load same-origin)."""
    r = static_client.get("/assets/app.js")
    assert r.status_code == 200
    assert "dashboard" in r.text


def test_api_reachable_under_prefix_with_frontend(static_client):
    """The API stays fully reachable under /api even with the SPA mounted at / (no shadowing)."""
    b = static_client.get("/api/health").json()
    assert b["ready"] is True and b["layer"] == 19


def test_unknown_api_path_is_404_not_spa(static_client):
    """An unknown /api/* path 404s — the /api namespace never falls through to the SPA index."""
    r = static_client.get("/api/does-not-exist")
    assert r.status_code == 404
    assert 'id="root"' not in r.text


def test_api_only_when_no_frontend(fake_engine, monkeypatch):
    """No static_dir and no DASHBOARD_DIST -> API-only: / 404s (nothing mounted), /api still works.

    Guards dev / a frontend-less image: a missing build degrades to API-only, never a crash.
    """
    monkeypatch.delenv("DASHBOARD_DIST", raising=False)
    with TestClient(build_app(fake_engine)) as c:
        assert c.get("/").status_code == 404
        assert c.get("/api/health").status_code == 200
