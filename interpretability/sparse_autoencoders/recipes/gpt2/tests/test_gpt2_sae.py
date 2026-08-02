"""Fast, model-free tests for the GPT-2 SAE recipe.

These do NOT download GPT-2 or the SAE (no `engine.load()`), so they run on CPU in seconds: they check
the label-priority logic, the bundled dashboard data, and that the API surface is wired up.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest


# GPT-2 ships only its data now (the dashboard front-end is reused from ../evo2/feature_explorer/dist).
DATA = Path(__file__).resolve().parents[1] / "dashboard_data"


def test_label_priority():
    """label_for: user ⭐ label > Neuronpedia label > 'Feature N' fallback."""
    from gpt2_sae.server import Engine

    eng = Engine()
    eng.np_labels = {5: "references to the city of Paris"}
    eng.user_labels = {5: "⭐ France / Paris"}
    assert eng.label_for(5) == "⭐ France / Paris"  # user wins
    eng.user_labels = {}
    assert eng.label_for(5) == "references to the city of Paris"  # np fallback
    assert eng.label_for(999) == "Feature 999"  # final fallback


def test_api_surface():
    """The expected JSON endpoints are mounted."""
    from gpt2_sae.server import app

    paths = {r.path for r in app.routes}
    for p in ("/api/health", "/api/features", "/api/annotate", "/api/generate", "/api/gene_embed"):
        assert p in paths, f"missing route {p}"


def test_health_flips_dashboard_to_text_mode():
    """/api/health returns the `ui` block that flips the reused (evo2) dashboard into text mode."""
    from gpt2_sae.server import health

    class _Resp:
        status_code = 200

    info = health(_Resp())
    assert info["ui"]["textMode"] is True
    assert "GPT-2" in info["ui"]["brand"]


def test_data_routes_shadow_static_mount():
    """GPT-2's own data files are served by explicit routes (so they shadow evo2's dist)."""
    from gpt2_sae.server import app

    paths = {r.path for r in app.routes}
    for name in (
        "features_atlas.parquet",
        "feature_metadata.parquet",
        "feature_examples.parquet",
        "sequence_library.json",
        "neuronpedia_labels.json",
        "user_labels.json",
    ):
        assert f"/{name}" in paths, f"missing data route /{name}"


@pytest.mark.skipif(not (DATA / "features_atlas.parquet").exists(), reason="atlas not bundled")
def test_bundled_atlas_has_real_labels():
    """The bundled atlas carries semantic labels, not just 'Feature N' placeholders."""
    t = pq.read_table(DATA / "features_atlas.parquet")
    assert t.num_rows > 20000
    labels = t.column("label").to_pylist()
    real = sum(1 for x in labels if x and not str(x).startswith("Feature "))
    assert real / len(labels) > 0.9  # ~all features labeled (Neuronpedia + ⭐)


@pytest.mark.skipif(not (DATA / "user_labels.json").exists(), reason="no curated labels")
def test_starred_features_present():
    """The curated ⭐ steerable set is bundled and marked."""
    user = json.loads((DATA / "user_labels.json").read_text())
    starred = [v for v in user.values() if str(v).startswith("⭐")]
    assert len(starred) >= 5
