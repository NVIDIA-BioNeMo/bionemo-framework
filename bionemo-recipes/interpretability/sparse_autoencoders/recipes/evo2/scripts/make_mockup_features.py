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

"""Generate synthetic features.json + features_atlas.parquet for the evo2 SAE mockup dashboard.

Run once, commit outputs as fixtures. No real SAE involved — this is a v1 demo of the
visualization shell. The data shape is the contract the real eval pipeline will target later.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Plausible biological labels (20 features) — visible motifs in the synthetic sequences.
LABELS = [
    "exon-start motif",
    "tRNA acceptor stem",
    "intergenic GC-rich",
    "stop codon context",
    "ribosome binding site",
    "promoter -10 box",
    "CpG island",
    "splice donor",
    "polyA signal",
    "start codon ATG context",
    "transposon repeat",
    "rRNA conserved region",
    "operon intergenic",
    "frameshift-sensitive region",
    "high-conservation coding",
    "intron branch point",
    "TF binding motif",
    "phage integrase region",
    "origin of replication",
    "Shine-Dalgarno sequence",
]

# Plausible accessions to rotate across examples.
SEQ_IDS = ["NC_000913.3", "NC_002695.2", "chr1", "chr17"]

# Central motifs to splice into each feature's top-activating windows.
# Doesn't need biological rigor — just makes features visually distinguishable in the demo.
CENTRAL_MOTIFS = {
    "exon-start motif": "AGGTAAGT",
    "tRNA acceptor stem": "CCCGGGT",
    "intergenic GC-rich": "GCGCGCGC",
    "stop codon context": "TAATAATAA",
    "ribosome binding site": "AGGAGG",
    "promoter -10 box": "TATAAT",
    "CpG island": "CGCGCGCG",
    "splice donor": "GTAAGT",
    "polyA signal": "AATAAA",
    "start codon ATG context": "ATGGCC",
    "transposon repeat": "TTAATTAA",
    "rRNA conserved region": "GUCAGCUGGUC".replace("U", "T"),
    "operon intergenic": "AAATTT",
    "frameshift-sensitive region": "AAAAAAA",
    "high-conservation coding": "GCAGCAGCA",
    "intron branch point": "TACTAAC",
    "TF binding motif": "TGACTCA",
    "phage integrase region": "GCTAGGTGT",
    "origin of replication": "ATCGATCG",
    "Shine-Dalgarno sequence": "AGGAGGT",
}


def _random_dna(rng: np.random.Generator, length: int) -> str:
    """Generate a length-N DNA string by uniform-sampling A/C/G/T."""
    return "".join(rng.choice(list("ACGT"), size=length))


def _make_example(rng: np.random.Generator, label: str, feature_max: float, window: int = 200) -> dict:
    """Build one top-activating example: 200bp window with a central motif + a gaussian activation bump."""
    seq = list(_random_dna(rng, window))

    # Splice the feature's central motif into the middle ± a few bp jitter.
    motif = CENTRAL_MOTIFS[label]
    center = window // 2 + int(rng.integers(-5, 6))
    motif_start = center - len(motif) // 2
    for i, base in enumerate(motif):
        pos = motif_start + i
        if 0 <= pos < window:
            seq[pos] = base

    # Activation bump: gaussian centered in [80, 120], sigma ~= 8 bp, peak = feature_max * U(0.5, 1.0).
    bump_center = int(rng.integers(80, 121))
    sigma = 8.0
    peak = float(feature_max * rng.uniform(0.5, 1.0))
    positions = np.arange(window)
    activations = peak * np.exp(-((positions - bump_center) ** 2) / (2 * sigma**2))
    activations[activations < 0.01] = 0.0  # zero out the tails so the JSON is sparse-ish

    seq_id = SEQ_IDS[int(rng.integers(0, len(SEQ_IDS)))]
    start = int(rng.integers(1, 5_000_001))

    return {
        "sequence_id": seq_id,
        "start": start,
        "end": start + window,
        "sequence": "".join(seq),
        "activations": [round(float(a), 3) for a in activations],
        "max_activation": round(float(activations.max()), 4),
        "max_activation_position": int(activations.argmax()),
    }


def _make_features(rng: np.random.Generator) -> list[dict]:
    """Build the 20 synthetic feature entries for features.json."""
    features = []
    for fid, label in enumerate(LABELS):
        activation_freq = float(np.exp(rng.uniform(np.log(0.001), np.log(0.1))))
        max_activation = float(rng.uniform(5.0, 30.0))
        examples = [_make_example(rng, label, max_activation) for _ in range(30)]

        features.append(
            {
                "feature_id": fid,
                "description": label,
                "label": label,
                "activation_freq": round(activation_freq, 6),
                "max_activation": round(max_activation, 4),
                "top_positive_logits": [],
                "top_negative_logits": [],
                "examples": examples,
            }
        )
    return features


def _make_atlas(rng: np.random.Generator, features: list[dict]) -> pd.DataFrame:
    """Build features_atlas.parquet — synthetic UMAP coords with 4 visible clusters of 5 features each."""
    n_clusters = 4
    cluster_centers = rng.uniform(-5.0, 5.0, size=(n_clusters, 2))
    coords = []
    for fid in range(len(features)):
        cluster_idx = fid // (len(features) // n_clusters)
        center = cluster_centers[cluster_idx]
        xy = center + rng.normal(0, 0.5, size=2)
        coords.append(xy)
    coords = np.array(coords)

    return pd.DataFrame(
        {
            "feature_id": [f["feature_id"] for f in features],
            "x": coords[:, 0].round(4),
            "y": coords[:, 1].round(4),
            "label": [f["label"] for f in features],
            "activation_freq": [f["activation_freq"] for f in features],
            "log_frequency": [round(float(np.log10(f["activation_freq"])), 4) for f in features],
            "max_activation": [f["max_activation"] for f in features],
            "cluster": [fid // (len(features) // n_clusters) for fid in range(len(features))],
        }
    )


def _make_examples_table(features: list[dict]) -> pd.DataFrame:
    """Flatten per-feature examples into a long table for feature_examples.parquet.

    One row per (feature_id, example_rank). The dashboard lazy-loads these via DuckDB.
    """
    rows = []
    for feature in features:
        for rank, ex in enumerate(feature["examples"]):
            rows.append(
                {
                    "feature_id": feature["feature_id"],
                    "example_rank": rank,
                    "sequence_id": ex["sequence_id"],
                    "start": ex["start"],
                    "end": ex["end"],
                    "sequence": ex["sequence"],
                    "activations": ex["activations"],
                    "max_activation": ex["max_activation"],
                    "max_activation_position": ex["max_activation_position"],
                    "best_annotation": None,
                }
            )
    return pd.DataFrame(rows)


def main():
    """Generate synthetic parquet fixtures (atlas + metadata + examples) for the mockup dashboard."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "evo2_dashboard_mockup" / "public",
        help="Where to write the three parquet fixtures",
    )
    p.add_argument(
        "--write-json",
        action="store_true",
        help="Also write features.json (only useful if you point the dashboard at the legacy JSON path)",
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = _make_features(rng)

    atlas = _make_atlas(rng, features)
    atlas.to_parquet(args.output_dir / "features_atlas.parquet", index=False)
    # feature_metadata is the same shape as the atlas for the mockup — the dashboard
    # loads them as two tables but the queried columns are identical.
    atlas.to_parquet(args.output_dir / "feature_metadata.parquet", index=False)

    examples = _make_examples_table(features)
    examples.to_parquet(args.output_dir / "feature_examples.parquet", index=False)

    if args.write_json:
        with open(args.output_dir / "features.json", "w") as f:
            json.dump({"features": features}, f)
        print(f"Wrote {len(features)} features -> {args.output_dir / 'features.json'}")

    print(f"Wrote {len(atlas)} atlas rows -> {args.output_dir / 'features_atlas.parquet'}")
    print(f"Wrote {len(atlas)} metadata rows -> {args.output_dir / 'feature_metadata.parquet'}")
    print(f"Wrote {len(examples)} example rows -> {args.output_dir / 'feature_examples.parquet'}")


if __name__ == "__main__":
    main()
