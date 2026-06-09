# Nemotron SAE Feature Atlas

Interactive dashboard for exploring Sparse Autoencoder (SAE) features trained on **Nemotron-3-Nano** activations, with a UMAP embedding view and crossfiltering.

This is a port of the `codonfm/codon_dashboard` to natural-language text data: instead of highlighting DNA codons in the feature cards, it highlights **tokenized text sequences** by activation.

## Features

- **UMAP Embedding View**: Interactive scatter plot of feature embeddings with pan/zoom
- **Crossfiltering**: Brush selection on the UMAP and histograms filters the feature list
- **Feature Cards**: Expandable cards showing:
  - Feature description/label (editable, persisted in `localStorage`)
  - Activation frequency and max activation stats
  - Decoder logits — top tokens the feature promotes (green) / suppresses (red)
  - Top activating text examples with per-token activation highlighting
- **Search**: Filter features by description text or feature id
- **Color by**: Color points by any detected categorical or sequential column

## Usage

1. Copy your generated data files into the `public/` directory:

   - `features_atlas.parquet` — UMAP coordinates, label, and stats (one row per feature)
   - `feature_metadata.parquet` — per-feature stats (`feature_id`, `activation_freq`, `max_activation`, …)
   - `feature_examples.parquet` — top activating text examples (one row per example)
   - `vocab_logits.json` *(optional)* — decoder logits per feature
   - `cluster_labels.json` *(optional)* — cluster annotations for the UMAP

2. Install dependencies and run:

   ```bash
   npm install
   npm run dev
   ```

3. Open http://localhost:5177

You can also point the app at a parquet via the URL: `?data=https://.../features_atlas.parquet`.

## Data Format

### features_atlas.parquet

Required columns:

- `feature_id`: Integer feature ID
- `x`, `y`: UMAP coordinates
- `label`: Feature label for display
- `log_frequency`, `activation_freq`, `max_activation`: stats for histograms / coloring

Optional columns for coloring:

- Any `VARCHAR` column with ≤ 50 unique values (treated as categorical)
- `cluster_id` / `*category*` / `*group*` integer columns (categorical)

### feature_metadata.parquet

- `feature_id`, `activation_freq`, `max_activation`

### feature_examples.parquet

One row per (feature, example):

- `feature_id`: Integer feature ID
- `example_rank`: Integer rank (0 = strongest activating example)
- `text_id`: Source document/text identifier (optional)
- `tokens`: `LIST<VARCHAR>` — the token strings of the example
- `activations`: `LIST<FLOAT>` — per-token activation values (parallel to `tokens`)
- `max_activation`: Max activation in this example
- `best_annotation`: Optional free-text annotation shown next to the example

### vocab_logits.json

```json
{
  "0": {
    "top_positive": [[" the", 2.5], [" a", 2.1]],
    "top_negative": [["404", -1.8], [" zzz", -1.5]]
  }
}
```

Each entry maps a `feature_id` (as a string) to its top promoted / suppressed tokens
(`[token, logit_value]` pairs).
