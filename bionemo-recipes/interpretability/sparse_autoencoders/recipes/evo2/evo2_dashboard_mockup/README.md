# Evo 2 SAE Feature Explorer — Mockup

**Mockup, not a real artifact.** This is a fork of `recipes/codonfm/codon_dashboard` adapted for DNA / Evo 2, populated with **synthetic data**. No real SAE outputs flow through it yet. The point of this v1 is to lock in the data contract that the future real eval pipeline will target.

A `MOCKUP — synthetic data, not from a real SAE run` banner is rendered at the top of the app so nobody mistakes it for actual results.

## Quick start (local)

```bash
# In this directory:
npm install
npm run dev
# open http://localhost:5173
```

The dashboard reads three parquet fixtures from `public/`:

- `features_atlas.parquet` — UMAP coordinates + per-feature aggregates
- `feature_metadata.parquet` — feature label/stats table
- `feature_examples.parquet` — long table of (feature_id, example_rank, sequence_id, start, end, sequence, activations, ...) rows

The fixtures are committed to the repo. To regenerate them:

```bash
python ../scripts/make_mockup_features.py
```

That writes all three files into `public/`. Seed is fixed (`--seed 42`).

## What's mocked vs. real

| Thing                                | Source                                                          |
| ------------------------------------ | --------------------------------------------------------------- |
| Number of features                   | 20, hardcoded                                                   |
| Feature labels                       | Hardcoded biological-sounding strings                           |
| UMAP coordinates                     | 4 cluster centers + gaussian noise — fake but visibly clustered |
| Top activator windows                | Random `ACGT` with a label-matching central motif spliced in    |
| Per-token activations                | Gaussian bump centered randomly in [80, 120], sigma ~= 8 bp     |
| Vocab logits (promoted / suppressed) | Empty arrays — not in scope for v1                              |

## v2 roadmap placeholders

A few greyed-out stats on each feature card (`Annotation`, `Sensitivity`, `Recon Δ`) and two empty sections on the feature detail page (`Annotations`, `Conservation`) hint at what's coming in v2. They render as em-dashes / dashed empty boxes with hover tooltips explaining what they'll show.

## Out of scope (v1)

- Real SAE inference or activation pass
- Annotation overlays (RefSeq / Rfam / JASPAR)
- Conservation tracks (phyloP)
- Strand handling, codon framing, chromosome ideograms
- External link-outs (UCSC, Ensembl)
- `sae.launch_dashboard()` Python wiring — run `npm run dev` directly
- Lepton-based serving

These are deferred to v2, once the real eval pipeline produces matching parquet shapes.
