# CodonFM SAE Recipe

Train and analyze sparse autoencoders on [CodonFM](https://huggingface.co/nvidia/NV-CodonFM-Encodon-1B-v1) Encodon codon language models. The pipeline extracts residual stream activations, trains a TopK SAE, evaluates reconstruction quality, and optionally generates an interactive feature dashboard.

## Pipeline

```
Extract activations -> Train SAE -> Evaluate -> Analyze (optional) -> Dashboard (optional)
```

**Extract** runs the Encodon model over DNA coding sequences, saving per-codon hidden states from a target layer to sharded Parquet files. **Train** fits a TopK SAE (8x expansion, top-32 sparsity by default) on those activations. **Evaluate** measures loss recovered by comparing model logits with and without the SAE bottleneck. **Analyze** computes per-feature interpretability annotations (codon usage bias, amino acid identity, wobble position, CpG content) and optionally generates LLM-based feature labels. **Dashboard** builds UMAP embeddings and exports data for a React-based interactive feature explorer.

## Prerequisites

1. Encodon checkpoint (`.safetensors` or `.ckpt` with accompanying `config.json`):

   ```bash
   huggingface-cli download nvidia/NV-CodonFM-Encodon-1B-v1 --local-dir ./checkpoints/encodon_1b
   ```

2. DNA sequence data as a CSV with a coding sequence column (`cds`, `seq`, or `sequence` -- auto-detected).

3. Install dependencies:

   ```bash
   # From repo root (UV workspace)
   uv sync
   ```

## Quick Start

```bash
# Full pipeline: extract -> train -> eval
python run.py model=1b csv_path=path/to/Primates.csv

# Skip extraction if activations are already cached
python run.py model=1b csv_path=path/to/data.csv steps.extract=false

# Smoke test
python run.py model=1b csv_path=path/to/data.csv num_sequences=100 train.n_epochs=1 nproc=1 dp_size=1
```

## Step-by-Step

### 1. Extract Activations

```bash
# Single GPU
python scripts/extract.py \
    --csv-path path/to/Primates.csv \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 \
    --num-sequences 50000 \
    --output .cache/activations/encodon_1b_layer-2

# Multi-GPU
torchrun --nproc_per_node=4 scripts/extract.py \
    --csv-path path/to/Primates.csv \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 \
    --output .cache/activations/encodon_1b_layer-2
```

Outputs sharded Parquet files + `metadata.json` to the cache directory. CLS and SEP tokens are stripped; only codon-position activations are saved.

### 2. Train SAE

```bash
python scripts/train.py \
    --cache-dir .cache/activations/encodon_1b_layer-2 \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 \
    --expansion-factor 8 --top-k 32 \
    --batch-size 4096 --n-epochs 3 \
    --output-dir ./outputs/encodon_1b

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py \
    --cache-dir .cache/activations/encodon_1b_layer-2 \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --dp-size 4 \
    --expansion-factor 8 --top-k 32 \
    --batch-size 4096 --n-epochs 3 \
    --output-dir ./outputs/encodon_1b
```

Saves checkpoint to `./outputs/encodon_1b/checkpoints/checkpoint_final.pt`.

### 3. Evaluate

```bash
python scripts/eval.py \
    --checkpoint ./outputs/encodon_1b/checkpoints/checkpoint_final.pt \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --top-k 32 \
    --csv-path path/to/data.csv \
    --output-dir ./outputs/encodon_1b/eval
```

### 4. Analyze Features (optional)

```bash
python scripts/analyze.py \
    --checkpoint ./outputs/encodon_1b/checkpoints/checkpoint_final.pt \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --top-k 32 \
    --csv-path path/to/Primates.csv \
    --output-dir ./outputs/encodon_1b/analysis \
    --auto-interp --max-auto-interp-features 500
```

Produces `vocab_logits.json`, `feature_analysis.json`, and `feature_labels.json`.

### 5. Dashboard (optional)

```bash
# Generate dashboard data
python scripts/dashboard.py \
    --checkpoint ./outputs/encodon_1b/checkpoints/checkpoint_final.pt \
    --model-path path/to/encodon_1b/NV-CodonFM-Encodon-1B-v1.safetensors \
    --layer -2 --top-k 32 \
    --csv-path path/to/Primates.csv \
    --output-dir ./outputs/encodon_1b/dashboard

# Launch web UI
python scripts/launch_dashboard.py --data-dir ./outputs/encodon_1b/dashboard
```

## Model Sizes

| Model        | Params | Layers | Hidden Dim | Batch Size | Config       |
| ------------ | ------ | ------ | ---------- | ---------- | ------------ |
| Encodon 80M  | 80M    | 6      | 1024       | 32         | `model=80m`  |
| Encodon 600M | 600M   | 12     | 2048       | 16         | `model=600m` |
| Encodon 1B   | 1B     | 18     | 2048       | 8          | `model=1b`   |
| Encodon 5B   | 5B     | 24     | 4096       | 2          | `model=5b`   |

## Configuration

Hydra configs live in `run_configs/`. The base config (`config.yaml`) sets defaults for all steps. Model-specific configs in `run_configs/model/` override `model_path`, `run_name`, `num_sequences`, and `batch_size`.

Override any parameter on the command line:

```bash
python run.py model=1b csv_path=data.csv train.n_epochs=5 train.lr=1e-4 nproc=8
```

Key training defaults: `expansion_factor=8`, `top_k=32`, `lr=3e-4`, `n_epochs=3`, `batch_size=4096`, `layer=-2`. # gitleaks:allow

## Project Structure

```
recipes/codonfm/
  run.py                    Hydra pipeline orchestrator
  run_configs/              Hydra configs (config.yaml, model/*.yaml)
  scripts/
    extract.py              Extract layer activations (multi-GPU)
    train.py                Train TopK SAE (multi-GPU)
    eval.py                 Loss recovered evaluation
    analyze.py              Feature interpretability annotations
    dashboard.py            UMAP + dashboard data export
    launch_dashboard.py     Serve interactive web UI
    mutation_features.py    Mutation-site feature analysis
  src/codonfm_sae/          Recipe-specific code (CSV loader, eval)
  codon-fm/                 CodonFM model code (tokenizer, inference, models)
  codon_dashboard/          React/Vite interactive dashboard
  notebooks/                Jupyter notebooks (UMAP exploration)
```

## Data Format

CSV with a DNA coding sequence column. The loader auto-detects columns named `cds`, `seq`, or `sequence`. Each sequence should be a string of nucleotides whose length is divisible by 3 (codons). The tokenizer splits into 3-mer codons from a 69-token vocabulary (5 special + 64 DNA codons).
