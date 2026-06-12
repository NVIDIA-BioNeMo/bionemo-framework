# SAE Feature Dashboard

Interactive dashboard for exploring Sparse Autoencoder (SAE) features with UMAP embedding visualization and crossfiltering.

## Features

- **UMAP Embedding View**: Interactive scatter plot of feature embeddings with pan/zoom
- **Crossfiltering**: Brush selection on UMAP and histograms filters the feature list
- **Feature Cards**: Expandable cards showing:
  - Feature description/label
  - Activation frequency and max activation stats
  - Top positive/negative logits (tokens the feature promotes/suppresses)
  - Top activating examples with token highlighting
- **Search**: Filter features by description text
- **Color by Category**: Color points by categorical or sequential columns

## Usage

### From Python (via `sae.launch_dashboard`)

```python
from sae import launch_dashboard

# Launch dashboard with your data
launch_dashboard(
    features_json="path/to/features.json",
    atlas_parquet="path/to/features_atlas.parquet",
    port=5173,
)
```

### Manual Setup

1. Copy your data files to the `public/` directory:

   - `features.json` - Feature metadata with examples
   - `features_atlas.parquet` - UMAP coordinates and stats

2. Install dependencies and run:

   ```bash
   npm install
   npm run dev
   ```

3. Open http://localhost:5173

## Data Format

### features.json

```json
{
  "features": [
    {
      "feature_id": 0,
      "description": "Feature description",
      "activation_freq": 0.05,
      "max_activation": 12.5,
      "top_positive_logits": [["token1", 2.5], ["token2", 2.1]],
      "top_negative_logits": [["token3", -1.8], ["token4", -1.5]],
      "examples": [
        {
          "max_activation": 10.2,
          "tokens": [
            {"token": "hello", "activation": 0.0},
            {"token": " world", "activation": 10.2}
          ]
        }
      ]
    }
  ]
}
```

### features_atlas.parquet

Required columns:

- `feature_id`: Integer feature ID
- `x`, `y`: UMAP coordinates
- `label` or `best_annotation`: Feature label for display
- `log_frequency`: Log of activation frequency
- `max_activation`: Maximum activation value

Optional columns for coloring:

- Any VARCHAR column with \<= 50 unique values (categorical)
- `cluster`, `category`, `group` integer columns (categorical)
