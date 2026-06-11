# SAE Feature Dashboard

Interactive dashboard for exploring Sparse Autoencoder (SAE) features with UMAP embedding visualization and crossfiltering.

## Features

- **UMAP Embedding View**: Interactive scatter plot of feature embeddings with pan/zoom
- **Crossfiltering**: Brush selection on UMAP and histograms filters the feature list
- **Feature Cards**: Expandable cards showing:
  - Feature description/label
  - Activation frequency and max activation stats
  - Top activating examples with per-residue activation highlighting
- **Search**: Filter features by description text
- **Color by Category**: Color points by categorical or sequential columns

## Usage

### From Python (via `esm2_sae.launch_protein_dashboard`)

```python
from esm2_sae import launch_protein_dashboard

# Launch dashboard with your data
proc = launch_protein_dashboard(
    "path/to/features_atlas.parquet",
    features_dir="path/to/dashboard_dir",
)
input("Dashboard running. Press Enter to stop.\n")
proc.terminate()
```

### Manual Setup

1. Copy your data files to the `public/` directory:

   - `features_atlas.parquet` - UMAP coordinates and stats
   - `feature_metadata.parquet` - Feature metadata
   - `feature_examples.parquet` - Top activating examples per feature

2. Install dependencies and run:

   ```bash
   npm install
   npm run dev
   ```

3. Open http://localhost:5173

## Data Format

The dashboard loads three Parquet files from the `public/` directory via DuckDB-WASM.

### `features_atlas.parquet`

One row per SAE feature. Drives the UMAP scatter plot and histograms.

| Column              | Type    | Description                                            |
| ------------------- | ------- | ------------------------------------------------------ |
| `feature_id`        | INT32   | Feature index (0 to n_features-1)                      |
| `label`             | VARCHAR | Display label (e.g. "Kinase (F1:0.82)" or "Feature 5") |
| `x`                 | FLOAT   | UMAP x coordinate (from decoder weights)               |
| `y`                 | FLOAT   | UMAP y coordinate                                      |
| `activation_freq`   | FLOAT   | Fraction of residues where feature fires (> 0)         |
| `log_frequency`     | FLOAT   | log10(activation_freq), clamped to -10 when zero       |
| `mean_activation`   | FLOAT   | Mean activation when active                            |
| `max_activation`    | FLOAT   | Maximum activation observed                            |
| `std_activation`    | FLOAT   | Std dev of activation when active                      |
| `total_activations` | INT64   | Total count of firings                                 |
| `cluster_id`        | INT32   | HDBSCAN cluster (NULL for noise points)                |

Any additional VARCHAR column with \<= 50 unique values is available as a coloring option.

### `feature_metadata.parquet`

One row per SAE feature. Loaded into a DuckDB table for feature card display.

| Column            | Type    | Description                                       |
| ----------------- | ------- | ------------------------------------------------- |
| `feature_id`      | INT32   | Feature index                                     |
| `description`     | VARCHAR | Best annotation or "Feature {id}" if unlabeled    |
| `activation_freq` | FLOAT32 | Fraction of residues where feature fires          |
| `max_activation`  | FLOAT32 | Global maximum activation                         |
| `best_f1`         | FLOAT32 | F1 score for best SwissProt annotation (nullable) |
| `best_annotation` | VARCHAR | Best SwissProt annotation string (nullable)       |

### `feature_examples.parquet`

Top activating protein examples per feature. Loaded as a DuckDB view and queried lazily when a feature card is expanded.

| Column           | Type           | Description                                          |
| ---------------- | -------------- | ---------------------------------------------------- |
| `feature_id`     | INT32          | Feature index                                        |
| `example_rank`   | INT8           | Rank within feature (0 = highest activation)         |
| `protein_id`     | VARCHAR        | UniProt accession (e.g. "sp\|P12345\|...")           |
| `alphafold_id`   | VARCHAR        | AlphaFold structure ID (e.g. "AF-P12345-F1")         |
| `sequence`       | VARCHAR        | Amino acid sequence                                  |
| `activations`    | LIST\<FLOAT32> | Per-residue activation values (same len as sequence) |
| `max_activation` | FLOAT32        | Max activation for this protein on this feature      |

Sorted by `feature_id` for efficient row-group pushdown queries.

## Inspecting the Data

```bash
# Requires: pip install duckdb
python -c "
import duckdb
con = duckdb.connect()

# Atlas overview
con.sql(\"SELECT * FROM 'features_atlas.parquet' LIMIT 5\").show()

# Top annotated features
con.sql(\"\"\"
    SELECT feature_id, description, best_f1, activation_freq
    FROM 'feature_metadata.parquet'
    WHERE best_f1 IS NOT NULL
    ORDER BY best_f1 DESC
    LIMIT 10
\"\"\").show()

# Top examples for a specific feature
con.sql(\"\"\"
    SELECT feature_id, example_rank, protein_id, max_activation
    FROM 'feature_examples.parquet'
    WHERE feature_id = 42
\"\"\").show()
"
```
