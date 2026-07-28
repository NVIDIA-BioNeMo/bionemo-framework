# GPT-2 SAE Recipe

An interactive **sparse-autoencoder feature explorer for GPT-2 small** — the natural-language sibling
of the `evo2` / `esm2` / `codonfm` recipes, built on the same model-agnostic [`sae`](../../sae) library
and the same feature-explorer dashboard.

Unlike the other recipes, this one needs **no training and no gated model**: it loads OpenAI GPT-2 small
(124M) through [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) and a
**pretrained residual-stream SAE** (Joseph Bloom's `blocks.7.hook_resid_pre`, 24,576 features). It runs
on a single GPU or on **CPU** (GPT-2 small is small), so it doubles as a zero-setup demo of the toolchain.

## Quick start

```bash
# from the repo root (uv workspace)
uv sync

# serve the dashboard + API (loads model + SAE once; GPU if available, else CPU)
gpt2-sae serve            # -> http://localhost:8749
```

The dashboard ships **prebuilt** (`feature_explorer/dist/` with the atlas parquets and 24,570 semantic
feature labels bundled in), so `serve` is the only step needed.

## Dashboard

Four tabs (the same explorer used across recipes, re-skinned for text):

- **Feature atlas** — browse every SAE feature: firing rate, decoder-space UMAP, top-activating text
  snippets, and its label. Static — no backend needed.
- **Text inspector** — paste text, see per-token feature activations (top-k or features you pick).
- **Generative steering** — generate from a prompt while clamping chosen features on the continuation,
  vs. an unsteered baseline.
- **Text UMAP** — embed a set of texts into per-feature SAE vectors and lay them out in 2-D.

Feature labels come from [Neuronpedia](https://www.neuronpedia.org/)'s auto-interp explanations for this
exact SAE (index-aligned 1:1), so any concept is searchable in the pickers. A curated set of ⭐
verified-steerable features (France/Paris, music, ocean, football, …) leads the catalog.

## CLI

```bash
gpt2-sae annotate "the cat sat on the mat" -k 8            # top-k features per token
gpt2-sae generate "I think the best thing to do is" \
    -f 20174:45 -f 21634:40 -n 24 --baseline               # steer ocean + Paris features
gpt2-sae serve 8749                                        # dashboard + API
```

## Rebuilding the bundled data (optional)

The parquets + labels are committed under `feature_explorer/dist/`. To regenerate them from scratch:

```bash
uv sync --extra build                       # scikit-learn + umap-learn
python scripts/build_atlas.py               # features_atlas.parquet (UMAP + stats)
python scripts/build_meta.py                # feature_metadata.parquet + feature_examples.parquet
python scripts/fetch_neuronpedia.py         # neuronpedia_labels.json + bakes labels into the parquets
python scripts/curate_steerable.py          # (server running) probe -> steer-test -> user_labels.json
```

Then rebuild the dashboard (`cd feature_explorer && npm ci && npm run build`) and re-copy the parquets
into `dist/`.

## Why TransformerLens

The Bloom SAE was trained on **TransformerLens** activations (with `fold_ln` / centering), not raw
HuggingFace hidden states. Encoding with raw HF activations gives garbage reconstructions (FVU ≫ 1);
TransformerLens gives FVU ≈ 0.001. The `encode_fn` runtime must match the SAE's training runtime — hence
`HookedTransformer` here rather than a bare `transformers` forward pass.

## Layout

```
gpt2/
├── src/gpt2_sae/        # server.py (Engine + FastAPI), cli.py
├── feature_explorer/    # React dashboard (source) + prebuilt dist/ (with bundled parquets + labels)
├── scripts/             # build_atlas, build_meta, fetch_neuronpedia, curate_steerable
└── tests/
```
