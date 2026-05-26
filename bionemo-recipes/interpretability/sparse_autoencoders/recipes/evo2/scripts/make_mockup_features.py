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


# DNA-native labels for evo2 features, each with a real central signature spliced into
# the middle of the 200bp window so the mockup features are visually distinguishable.
LABELS = [
    "Start codon (ATG) context",
    "TATA box",
    "Polyadenylation signal",
    "Bacterial promoter -10 box",
    "CpG island",
    "Shine-Dalgarno sequence",
    "Bacterial promoter -35 box",
    "Splice donor site",
    "Splice acceptor site",
    "Stop codon (TAA) context",
    "Stop codon (TAG) context",
]

# Plausible accessions to rotate across examples.
SEQ_IDS = ["NC_000913.3", "NC_002695.2", "chr1", "chr17"]

# Central motif spliced into the middle ~20bp of each top-activating window.
CENTRAL_MOTIFS = {
    "Start codon (ATG) context": "GCCACCATGGCC",
    "TATA box": "TATAAA",
    "Polyadenylation signal": "AATAAA",
    "Bacterial promoter -10 box": "TATAAT",
    "CpG island": "CGCGCGCGCGCGCGCG",
    "Shine-Dalgarno sequence": "AGGAGGT",
    "Bacterial promoter -35 box": "TTGACA",
    "Splice donor site": "GTAAGT",  # GT at exon-intron boundary, with consensus context
    "Splice acceptor site": "TTTTCAGG",  # AG at intron-exon boundary, with pyrimidine tract
    "Stop codon (TAA) context": "GCCTAAGCC",  # TAA in coding context
    "Stop codon (TAG) context": "GCCTAGGCC",  # TAG in coding context
}

# Annotation-database source for each feature label.
DB_SOURCES = {
    "Start codon (ATG) context": "RefSeq",
    "TATA box": "JASPAR / ENCODE",
    "Polyadenylation signal": "RefSeq UTR",
    "Bacterial promoter -10 box": "bacterial annotation",
    "CpG island": "ENCODE / RefSeq",
    "Shine-Dalgarno sequence": "bacterial annotation",
    "Bacterial promoter -35 box": "bacterial annotation",
    "Splice donor site": "RefSeq",
    "Splice acceptor site": "RefSeq",
    "Stop codon (TAA) context": "RefSeq",
    "Stop codon (TAG) context": "RefSeq",
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
                "db_source": DB_SOURCES[label],
                "activation_freq": round(activation_freq, 6),
                "max_activation": round(max_activation, 4),
                "top_positive_logits": [],
                "top_negative_logits": [],
                "examples": examples,
            }
        )
    return features


def _make_atlas(rng: np.random.Generator, features: list[dict]) -> pd.DataFrame:
    """Build features_atlas.parquet — UMAP coords grouped into thematic clusters.

    Labeled features sit in 3 clusters: eukaryotic regulatory (0), bacterial regulatory (1),
    codon context (2). Unlabeled features (label==None) land in a 4th "uninterpreted" cluster (3)
    spread more diffusely between the others — mimicking what a real SAE would look like.
    """
    cluster_assignments = {
        "Start codon (ATG) context": 2,
        "TATA box": 0,
        "Polyadenylation signal": 0,
        "Bacterial promoter -10 box": 1,
        "CpG island": 0,
        "Shine-Dalgarno sequence": 1,
        "Bacterial promoter -35 box": 1,
        "Splice donor site": 0,
        "Splice acceptor site": 0,
        "Stop codon (TAA) context": 2,
        "Stop codon (TAG) context": 2,
    }
    cluster_centers = {
        0: (-3.0, 1.5),
        1: (3.0, 1.5),
        2: (0.0, -3.0),
        3: (0.0, 0.5),  # uninterpreted: between the other clusters
    }

    coords = []
    cluster_ids = []
    for f in features:
        if f["label"] is None:
            cid = 3
            sigma = 1.4  # diffuse for the unlabeled cloud
        else:
            cid = cluster_assignments[f["label"]]
            sigma = 0.4
        cx, cy = cluster_centers[cid]
        x = cx + rng.normal(0, sigma)
        y = cy + rng.normal(0, sigma)
        coords.append((x, y))
        cluster_ids.append(cid)
    coords = np.array(coords)

    return pd.DataFrame(
        {
            "feature_id": [f["feature_id"] for f in features],
            "x": coords[:, 0].round(4),
            "y": coords[:, 1].round(4),
            "label": [f["label"] for f in features],
            "db_source": [f["db_source"] for f in features],
            "activation_freq": [f["activation_freq"] for f in features],
            "log_frequency": [round(float(np.log10(f["activation_freq"])), 4) for f in features],
            "max_activation": [f["max_activation"] for f in features],
            "cluster_id": cluster_ids,
        }
    )


def _make_unlabeled_example(rng: np.random.Generator, feature_max: float, window: int = 200) -> dict:
    """A top-activating example for an unlabeled feature: random sequence + gaussian activation bump."""
    seq = _random_dna(rng, window)
    bump_center = int(rng.integers(60, 141))
    sigma = 8.0
    peak = float(feature_max * rng.uniform(0.5, 1.0))
    positions = np.arange(window)
    activations = peak * np.exp(-((positions - bump_center) ** 2) / (2 * sigma**2))
    activations[activations < 0.01] = 0.0

    seq_id = SEQ_IDS[int(rng.integers(0, len(SEQ_IDS)))]
    start = int(rng.integers(1, 5_000_001))

    return {
        "sequence_id": seq_id,
        "start": start,
        "end": start + window,
        "sequence": seq,
        "activations": [round(float(a), 3) for a in activations],
        "max_activation": round(float(activations.max()), 4),
        "max_activation_position": int(activations.argmax()),
    }


def _make_unlabeled_features(rng: np.random.Generator, n: int, start_id: int) -> list[dict]:
    """Build n unlabeled features — no semantic label, random top-activator sequences."""
    out = []
    for i in range(n):
        fid = start_id + i
        activation_freq = float(np.exp(rng.uniform(np.log(0.001), np.log(0.1))))
        max_activation = float(rng.uniform(5.0, 30.0))
        examples = [_make_unlabeled_example(rng, max_activation) for _ in range(30)]
        out.append(
            {
                "feature_id": fid,
                "description": None,
                "label": None,
                "db_source": None,
                "activation_freq": round(activation_freq, 6),
                "max_activation": round(max_activation, 4),
                "top_positive_logits": [],
                "top_negative_logits": [],
                "examples": examples,
            }
        )
    return out


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
                    "best_annotation": feature["db_source"],
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
    p.add_argument("--n-unlabeled", type=int, default=9, help="How many unlabeled features to add alongside the labeled ones")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = _make_features(rng)
    features += _make_unlabeled_features(rng, n=args.n_unlabeled, start_id=len(features))

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
