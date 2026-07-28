"""Pull Neuronpedia's auto-interp explanations for this SAE and bake them into the atlas.

The dashboard SAE (jbloom gpt2-small `7-res-jb`) is index-aligned 1:1 with Neuronpedia's public
explanation set, so feature N here == feature N there. This script:

  1. downloads the 48 explanation batches from the public Neuronpedia S3 export,
  2. writes {feature_id -> description} to neuronpedia_labels.json,
  3. rewrites the `label`/`description` columns of features_atlas.parquet + feature_metadata.parquet
     so the browse UI shows semantic names (user ⭐ labels in user_labels.json win over these).

Run after build_atlas.py / build_meta.py. Output dir defaults to feature_explorer/dist/.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import urllib.request
from pathlib import Path


BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
PREFIX = "v1/gpt2-small/7-res-jb/explanations"
N_BATCHES = 48
OUT = Path(os.environ.get("GPT2_SAE_DATA", Path(__file__).resolve().parents[1] / "feature_explorer" / "dist"))


def fetch_labels() -> dict[int, str]:
    """Download Neuronpedia's explanation batches into {feature_id -> description}."""
    labels: dict[int, str] = {}
    for i in range(N_BATCHES):
        url = f"{BUCKET}/{PREFIX}/batch-{i}.jsonl.gz"
        with urllib.request.urlopen(url, timeout=120) as r:
            raw = r.read()
        for line in gzip.GzipFile(fileobj=io.BytesIO(raw)):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            idx, desc = d.get("index"), (d.get("description") or "").strip()
            if idx is None or not desc:
                continue
            idx = int(idx)
            # keep the longer of duplicate explanations (~2 per feature)
            if idx not in labels or len(desc) > len(labels[idx]):
                labels[idx] = desc
        print(f"  batch {i + 1}/{N_BATCHES}: {len(labels)} features so far", flush=True)
    return labels


def bake(labels: dict[int, str]) -> None:
    """Rewrite the label/description columns of the atlas + metadata parquets in place."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    user = {}
    ul = OUT / "user_labels.json"
    if ul.exists():
        user = {int(k): v for k, v in json.loads(ul.read_text()).items()}

    def label_for(fid: int) -> str:
        return user.get(fid) or labels.get(fid) or f"Feature {fid}"

    for fn, col in [("features_atlas.parquet", "label"), ("feature_metadata.parquet", "description")]:
        path = OUT / fn
        if not path.exists():
            print(f"  skip {fn} (not built yet)")
            continue
        t = pq.read_table(path)
        fids = t.column("feature_id").to_pylist()
        new = pa.array([label_for(int(f)) for f in fids], type=pa.string())
        t = t.set_column(t.column_names.index(col), col, new)
        pq.write_table(t, path)
        real = sum(1 for f in fids if not label_for(int(f)).startswith("Feature "))
        print(f"  baked {fn}[{col}]: {real}/{len(fids)} real labels")


def main():
    """Fetch the Neuronpedia labels, write neuronpedia_labels.json, and bake them into the parquets."""
    OUT.mkdir(parents=True, exist_ok=True)
    labels = fetch_labels()
    (OUT / "neuronpedia_labels.json").write_text(json.dumps(labels))
    print(f"  wrote neuronpedia_labels.json ({len(labels)} labels)")
    bake(labels)


if __name__ == "__main__":
    main()
