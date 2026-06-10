# RiboNN Translation Efficiency Evaluation

This directory contains scripts that reproduce the RiboNN translation-efficiency (TE)
downstream evaluation from the EnCodon paper, using a Hugging Face CodonFM checkpoint
from this recipe.

The task: given a coding sequence (CDS) from a human mRNA, predict its translation
efficiency. We do this by extracting CLS-token embeddings from a frozen pretrained
CodonFM model and training a `RandomForestRegressor` on top, evaluated with
leave-one-fold-out cross-validation against the public RiboNN labels.

Data source: [`CenikLab/TE_classic_ML`](https://github.com/CenikLab/TE_classic_ML)
(`data_with_human_TE_cellline_all_NA_plain.csv`), ~11k transcripts with `mean_te`
labels and a precomputed 10-fold split.

Reference:

> Zheng, Dinghai, et al. *Predicting the translation efficiency of messenger RNA in
> mammalian cells.* Nature Biotechnology (2025): 1-14.

## Scripts

- `preprocess.py` — Downloads the RiboNN TSV, slices the transcript into
  CDS / 5'UTR / 3'UTR using `utr5_size` and `cds_size`, adds a row-index `id`, and
  writes `ribonn_cds.parquet` containing only the columns the downstream evaluation
  needs (`id`, `cds_sequence`, `mean_te`, `fold`).
- `evaluate_rf.py` — Loads `ribonn_cds.parquet`, extracts CLS embeddings from a HF
  CodonFM checkpoint (reusing `extract_embeddings.py` from the recipe root), runs
  leave-one-fold-out CV with a `RandomForestRegressor`, and writes per-fold metrics
  to `metrics.csv` and aggregate stats to `metrics_summary.csv`. Embeddings are
  cached to a `.npz` and validated on load (ids, targets, folds, sequence hash, and
  `max_seq_length` must all match), so re-running only re-trains the RF unless an
  input actually changed.

## Usage

Run from inside this directory.

### 1. Preprocess the dataset

```bash
python preprocess.py
```

Downloads `data_with_human_TE_cellline_all_NA_plain.csv` next to the script if not
already present and writes `ribonn_cds.parquet`.

### 2. Run the evaluation

```bash
python evaluate_rf.py --model-name-or-path nvidia/NV-CodonFM-Encodon-TE-Cdwt-1B-v1
```

Useful flags:

- `--demo-size N` — stratified-sample `N` rows by `fold` for a quick smoke run
  (the original notebook uses `--demo-size 500`). Omit for the full ~11k dataset.
- `--batch-size`, `--device` — passed through to the embedding extractor
  (defaults: `16`, `cuda`).
- `--output-dir` — where to write the embeddings cache and metrics CSVs
  (default: this directory).
- `--force-extract` — re-extract embeddings even if a valid cache exists.
- `--seed` — RNG seed for subsampling and the Random Forest (default: `42`).

### Outputs

- `embeddings_<model_slug>_n<rows>.npz` — cached CLS embeddings + the inputs they
  were extracted from (used to validate the cache on subsequent runs).
- `metrics.csv` — one row per fold with `fold, r2, pearson_r, mse, rmse`.
- `metrics_summary.csv` — single-row aggregate: `mean_r2, std_r2, mean_pearson_r, std_pearson_r, mean_rmse` (mean RMSE follows the notebook convention,
  `sqrt(mean(MSE))`, not `mean(sqrt(MSE))`).
