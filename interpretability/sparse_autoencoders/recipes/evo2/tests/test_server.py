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
    assert {"id", "label", "natural_peak", "steerable", "description", "auroc"} <= set(rows[0])
    by_id = {r["id"]: r for r in rows}
    # curated extras flow through from engine.feature_extra, keyed per feature
    assert by_id[0]["steerable"] is True and by_id[0]["description"] == "fires on feat0 motifs"
    assert by_id[1]["steerable"] is False and by_id[1]["description"] is None  # no extra → sensible defaults


def test_annotate_returns_per_base_activations(client):
    b = client.post("/api/annotate", json={"sequence": "ACGTACGT", "organism": "None (raw DNA)"}).json()
    assert {"sequence", "features", "bases", "tag_len", "layer", "n_tokens"} <= set(b)
    assert b["features"][0]["activations"]  # the per-base track the viz plots


def test_annotate_rejects_non_dna(client):
    assert client.post("/api/annotate", json={"sequence": "ZZZZ"}).status_code == 400


def test_annotate_rejects_unknown_organism(client):
    # an organism with no preset tag (and no custom tag) -> 400, not a 500 from a None tag downstream
    r = client.post("/api/annotate", json={"sequence": "ACGT", "organism": "Klingon"})
    assert r.status_code == 400


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
    b = client.post("/api/gene_embed", json={"genes": genes, "min_firing": 1}).json()
    assert {"G_b64", "Gmax_b64", "n_features", "n_genes", "genes", "feature_ids"} <= set(b)
    assert b["n_genes"] == 2 and len(b["genes"]) == 2
    # only firing columns are shipped: feature_ids maps column -> real SAE feature id
    assert len(b["feature_ids"]) == b["n_features"]
    assert b["feature_ids"] == [0]  # the fake fires feature 0 in every sequence, nothing else
    g = np.frombuffer(base64.b64decode(b["G_b64"]), dtype=np.float32)
    assert g.size == b["n_genes"] * b["n_features"]  # [n_genes x n_firing_features], what the client UMAPs
    # accounting fields so the UI can warn instead of silently embedding fewer than submitted —
    # lock the full set the banner interpolates (counts + the limits it names in the message).
    assert b["n_received"] == 2 and b["n_skipped_short"] == 0 and b["n_clamped"] == 0
    assert b["n_dropped_over_cap"] == 0
    assert b["max_seq_len"] > 0 and b["max_genes"] == 1000  # referenced verbatim in the UI warning


def test_gene_embed_clamps_overlength(client, fake_engine):
    """Over-length sequences are CLAMPED to the context window and still embedded (reported via
    n_clamped) — the UMAP tab keeps every point rather than dropping. Too-short ones are dropped+reported."""
    genes = [
        {"symbol": "ok", "sequence": "ACGTACGT"},
        {"symbol": "short", "sequence": "AC"},  # < 3 bases -> dropped
        {"symbol": "toolong", "sequence": "A" * (fake_engine.max_seq_len + 1)},  # > context -> clamped, embedded
    ]
    b = client.post("/api/gene_embed", json={"genes": genes, "min_firing": 1}).json()
    assert b["n_received"] == 3
    assert b["n_genes"] == 2  # the valid one + the clamped one
    assert b["n_skipped_short"] == 1 and b["n_clamped"] == 1


def test_gene_embed_all_invalid_400(client, fake_engine):
    """If nothing is embeddable (all too short), 400 with a message accounting for why."""
    r = client.post("/api/gene_embed", json={"genes": [{"sequence": "AC"}, {"sequence": "A"}]})
    assert r.status_code == 400
    assert "too short" in r.json()["detail"]


def test_gene_embed_rejects_unknown_organism(client):
    # unknown organism (no preset tag, no custom tag) -> 400 before embedding, not a 500
    r = client.post("/api/gene_embed", json={"genes": [{"sequence": "ACGT"}], "organism": "Klingon"})
    assert r.status_code == 400


def test_embed_bundle_forwards_cancel(fake_engine):
    """The real embed_bundle must forward its cancel predicate to encode_batch so a disconnected
    /gene_embed stops between micro-batches instead of encoding every sequence. Exercises the real
    core.Evo2SAE.embed_bundle (bound onto FakeEngine) + the fake's cooperative-cancel checkpoint;
    the TestClient can't simulate a mid-request disconnect, so this locks the plumbing directly."""
    from evo2_sae.core import RequestAborted

    with pytest.raises(RequestAborted):
        fake_engine.embed_bundle(["ACGTACGT", "TTTTGGGG"], 0, [{}, {}], cancel=lambda: True)
    # cancel=None (the default / non-server path) must never trip the checkpoint.
    assert fake_engine.embed_bundle(["ACGTACGT"], 0, [{}], cancel=None) is not None


def test_restart_disabled_by_default_403(client):
    """POST /restart is gated: without ALLOW_ENGINE_RESTART it returns 403 (never exits the process).
    The enabled path calls os._exit, so it's validated live, not in-process here."""
    r = client.post("/api/restart")
    assert r.status_code == 403
    assert "not enabled" in r.json()["detail"].lower()


def test_health_reports_restart_disabled(client):
    assert client.get("/api/health").json()["restart_enabled"] is False


def test_generate_409_when_engine_busy(client):
    """Single-flight: a request arriving while the engine is occupied gets a fast 409, not a silent
    queue behind the running one. (Hold the module busy-lock to simulate an in-flight request.)"""
    import evo2_sae.server as srv

    assert srv._engine_busy.acquire(blocking=False)
    try:
        r = client.post("/api/generate", json={"prompt": "ACGT"})
        assert r.status_code == 409
        assert "busy" in r.json()["detail"].lower()
    finally:
        srv._engine_busy.release()


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


def test_bad_static_dir_degrades_to_api_only(fake_engine, tmp_path, monkeypatch):
    """A bogus static_dir (e.g. a wrong DASHBOARD_DIST) must NOT crash the app — it degrades to
    API-only (/ 404s, /api still works) instead of erroring at mount time."""
    monkeypatch.delenv("DASHBOARD_DIST", raising=False)
    missing = tmp_path / "does-not-exist"  # not a directory
    with TestClient(build_app(fake_engine, static_dir=str(missing))) as c:
        assert c.get("/").status_code == 404
        assert c.get("/api/health").status_code == 200
