# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Generate synthetic steering_data.json for the SteeringExplorer page.

Hand-rolling 14 pairs × 200 positions × 4 clamps × 4 bases = ~45k probability
floats is impractical and would look fake. This script defines per-pair
"steering effect" functions over positions, then produces per-position
P(A/C/G/T) distributions at clamp ∈ {-2, 0, +2, +5}. The frontend
linearly interpolates between those four discrete points as the slider moves.

Each (seed, feature) pair has:
  - a baseline distribution function over positions (varies by seed)
  - an effect map: which positions are affected, how strongly, in what direction

Pairs are designed to tell a clear story:
  - ecoli_16s × kanamycin_resistance: position-8 A→G shift (A1408G analog)
  - promoter × TATA_box: positions 50-56 collapse to TATAAA consensus
  - brca1_exon × alpha_helix: coding region biases toward helix-favoring codons
  - random × anything: minimal effect (negative control)
  - 3 "null result" pairs with effect amplitude ≈ 0
"""

import json
import math
import random
from pathlib import Path


BASES = ["A", "C", "G", "T"]
CLAMPS = [-2, 0, 2, 5]
SEQ_LEN = 200


def _softmax(scores: dict) -> dict:
    """Stable softmax over the 4 bases."""
    mx = max(scores.values())
    expd = {b: math.exp(s - mx) for b, s in scores.items()}
    z = sum(expd.values())
    return {b: round(v / z, 4) for b, v in expd.items()}


def _baseline_for_seed(seed_id: str, rng: random.Random) -> list[dict]:
    """Per-position baseline distribution for a seed. Mildly peaked toward the
    actual base in the sequence (i.e., the model has reasonable baseline
    confidence in the wild-type position)."""
    seqs = {
        "ecoli_16s": ECOLI_16S,
        "promoter": PROMOTER,
        "brca1_exon": BRCA1_EXON,
        "random": RANDOM_CONTROL,
    }
    seq = seqs[seed_id]
    out = []
    for pos in range(SEQ_LEN):
        base = seq[pos]
        scores = {b: rng.gauss(0, 0.3) for b in BASES}
        # Bump the wild-type base — represents the model "knowing" what's there.
        scores[base] += rng.uniform(1.5, 2.5)
        out.append(_softmax(scores))
    return out


def _steered_for_pair(pair_id: str, baseline: list[dict], clamp: float, rng: random.Random) -> list[dict]:
    """Apply a pair-specific steering effect to baseline at a given clamp."""
    cfg = PAIR_EFFECTS[pair_id]
    out = []
    for pos in range(SEQ_LEN):
        base_p = baseline[pos]
        scores = {b: math.log(max(base_p[b], 1e-6)) for b in BASES}
        # Position-local effect strength: 0 outside the region, 1 at center,
        # taper at edges (gaussian).
        local_strength = 0.0
        for region in cfg["regions"]:
            center, width, weight = region
            d = abs(pos - center)
            local_strength += weight * math.exp(-(d * d) / (2 * width * width))
        # Direction: positive clamp pushes toward `target_base`, negative away.
        # Magnitude scales with |clamp| and local_strength.
        push = clamp * local_strength * cfg["amplitude"]
        target = cfg["target_base"]
        away_from = cfg.get("away_from", None)
        for b in BASES:
            if b == target:
                scores[b] += push
            elif away_from and b == away_from:
                scores[b] -= push * 0.5
            else:
                scores[b] -= push * 0.15
        # Small noise so the visual doesn't look mechanical.
        for b in BASES:
            scores[b] += rng.gauss(0, 0.05 + abs(push) * 0.05)
        out.append(_softmax(scores))
    return out


def _feature_activation_per_pos(pair_id: str, clamp: float, rng: random.Random) -> list[float]:
    """Synthetic feature activation per position, varying with clamp."""
    cfg = PAIR_EFFECTS[pair_id]
    base = 0.3 + 0.1 * rng.random()
    out = []
    for pos in range(SEQ_LEN):
        local_strength = 0.0
        for region in cfg["regions"]:
            center, width, weight = region
            d = abs(pos - center)
            local_strength += weight * math.exp(-(d * d) / (2 * width * width))
        # Activation rises with clamp on the affected positions.
        act = base + max(0, clamp) * local_strength * 0.8 + abs(clamp) * 0.1
        if clamp < 0:
            act = max(0.0, base + clamp * local_strength * 0.3)
        out.append(round(act + rng.gauss(0, 0.05), 3))
    return out


# Pre-baked sequences (200bp each). Realistic-looking but not real.
def _make_seq(rng: random.Random, biased_regions=None) -> str:
    seq = [rng.choice(BASES) for _ in range(SEQ_LEN)]
    if biased_regions:
        for start, motif in biased_regions:
            for i, b in enumerate(motif):
                if start + i < SEQ_LEN:
                    seq[start + i] = b
    return "".join(seq)


_seed_rng = random.Random(0)
ECOLI_16S = _make_seq(_seed_rng, biased_regions=[(8, "A")])  # position 8 = our A1408 analog
PROMOTER = _make_seq(_seed_rng, biased_regions=[(50, "TATAAT"), (75, "TTGACA")])  # -10, -35
BRCA1_EXON = _make_seq(_seed_rng, biased_regions=[(30, "ATG"), (170, "TAA")])  # start, stop
RANDOM_CONTROL = _make_seq(_seed_rng)


# Per-pair steering configurations:
#   regions: list of (center_position, width, weight) — gaussian effect zones
#   target_base: which base the feature pushes toward when amplified
#   away_from: optional, which base loses probability fastest
#   amplitude: how strong the push is per unit of clamp
PAIR_EFFECTS = {
    # ecoli_16s × *
    "ecoli_16s__kanamycin_resistance": {
        "regions": [(8, 2, 1.0)],
        "target_base": "G",
        "away_from": "A",
        "amplitude": 0.9,
        "narrative": "Matches the known A1408G aminoglycoside-resistance mutation. Amplifying the feature pushes position 8 from A toward G.",
    },
    "ecoli_16s__rRNA_structural": {
        "regions": [(c, 8, 0.4) for c in range(20, 200, 30)],
        "target_base": "A",
        "amplitude": 0.3,
        "narrative": "rRNA structural feature reinforces baseline preferences across the helical regions.",
    },
    "ecoli_16s__alpha_helix": {
        "regions": [(100, 10, 0.05)],
        "target_base": "G",
        "amplitude": 0.05,
        "narrative": "Null result: α-helix is a protein-coding feature with no traction in a non-coding rRNA sequence.",
    },
    # promoter × *
    "promoter__TATA_box": {
        "regions": [(53, 4, 1.0)],
        "target_base": "T",
        "amplitude": 0.7,
        "narrative": "Amplifying TATA_box collapses positions 50-56 toward the TATAAT consensus.",
    },
    "promoter__exon_start": {
        "regions": [(100, 15, 0.04)],
        "target_base": "G",
        "amplitude": 0.05,
        "narrative": "Null result: exon_start has minimal activity in a bacterial promoter context.",
    },
    "promoter__kanamycin_resistance": {
        "regions": [(50, 5, 0.02)],
        "target_base": "G",
        "amplitude": 0.02,
        "narrative": "Null result: kanamycin_resistance is rRNA-specific, doesn't apply to promoter DNA.",
    },
    # brca1_exon × *
    "brca1_exon__alpha_helix": {
        "regions": [(c, 15, 0.5) for c in range(40, 170, 30)],
        "target_base": "G",
        "amplitude": 0.4,
        "narrative": "Amplifying α-helix biases codon-3 positions in the coding region (30-180) toward G; helix-favoring codons end in G/C.",
    },
    "brca1_exon__beta_sheet": {
        "regions": [(c, 12, 0.25) for c in range(50, 170, 25)],
        "target_base": "C",
        "amplitude": 0.25,
        "narrative": "β-sheet propensity nudges codon-3 positions toward C (β-sheet residues like Val/Ile use C-ending codons).",
    },
    "brca1_exon__exon_start": {
        "regions": [(30, 4, 1.0)],
        "target_base": "G",
        "amplitude": 0.6,
        "narrative": "Amplifying exon_start sharpens the start codon region (ATG at position 30).",
    },
    "brca1_exon__kanamycin_resistance": {
        "regions": [(100, 20, 0.02)],
        "target_base": "G",
        "amplitude": 0.02,
        "narrative": "Null result: bacterial-rRNA feature has no purchase in a human exon.",
    },
    # random × *
    "random__alpha_helix": {
        "regions": [(c, 20, 0.05) for c in [50, 100, 150]],
        "target_base": "G",
        "amplitude": 0.08,
        "narrative": "Modest broadcast effect on random sequence — demonstrates the model doesn't blindly comply with steering when input doesn't support the feature.",
    },
    "random__TATA_box": {
        "regions": [(c, 15, 0.08) for c in [70, 130]],
        "target_base": "T",
        "amplitude": 0.1,
        "narrative": "Amplifying TATA_box on random sequence creates a faint bias toward T but no clear motif emergence.",
    },
    "random__kanamycin_resistance": {
        "regions": [(100, 30, 0.03)],
        "target_base": "G",
        "amplitude": 0.03,
        "narrative": "Null result: random sequence has no rRNA context for the feature to engage.",
    },
    # extra pair: tRNA_structural
    "ecoli_16s__tRNA_structural": {
        "regions": [(c, 10, 0.5) for c in [25, 75, 125, 175]],
        "target_base": "C",
        "amplitude": 0.3,
        "narrative": "tRNA-structural feature reinforces base-pairing positions along the rRNA scaffold.",
    },
}


def main():
    """Build the steering_data.json bundle from PAIR_EFFECTS + seed sequences."""
    rng = random.Random(42)
    seeds = {
        "ecoli_16s": {"name": "E. coli 16S rRNA region", "sequence": ECOLI_16S, "length": SEQ_LEN},
        "promoter": {"name": "σ70 promoter region", "sequence": PROMOTER, "length": SEQ_LEN},
        "brca1_exon": {"name": "BRCA1 exon fragment", "sequence": BRCA1_EXON, "length": SEQ_LEN},
        "random": {"name": "Random control", "sequence": RANDOM_CONTROL, "length": SEQ_LEN},
    }
    features_available = [
        {"id": 0, "label": "alpha_helix"},
        {"id": 1, "label": "beta_sheet"},
        {"id": 4, "label": "TATA_box"},
        {"id": 7, "label": "exon_start"},
        {"id": 12, "label": "kanamycin_resistance"},
        {"id": 18, "label": "rRNA_structural"},
        {"id": 21, "label": "tRNA_structural"},
        {"id": 19, "label": "beta_sheet"},
    ]
    # dedupe by id
    seen_ids = set()
    features_available = [f for f in features_available if not (f["id"] in seen_ids or seen_ids.add(f["id"]))]

    comparisons = {}
    for pair_id, cfg in PAIR_EFFECTS.items():
        seed_id = pair_id.split("__")[0]
        baseline = _baseline_for_seed(seed_id, rng)
        steered = {}
        feature_act = {}
        for clamp in CLAMPS:
            steered[str(clamp)] = _steered_for_pair(pair_id, baseline, clamp, rng)
            feature_act[str(clamp)] = _feature_activation_per_pos(pair_id, clamp, rng)
        comparisons[pair_id] = {
            "feature_label": pair_id.split("__")[1],
            "baseline_distributions": baseline,
            "steered_distributions": steered,
            "feature_activation": feature_act,
            "narrative": cfg["narrative"],
        }

    bundle = {
        "seeds": seeds,
        "features_available": features_available,
        "clamp_points": CLAMPS,
        "comparisons": comparisons,
    }

    out_path = Path(
        "/workspace/bionemo-dashboard/bionemo-recipes/interpretability/sparse_autoencoders/recipes/evo2/evo2_dashboard_mockup/public/steering_data.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(bundle, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"wrote {out_path}: {len(comparisons)} pairs × {SEQ_LEN} positions × {len(CLAMPS)} clamps ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
