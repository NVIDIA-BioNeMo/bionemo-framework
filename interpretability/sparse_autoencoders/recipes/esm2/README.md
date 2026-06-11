# ESM2 SAE Recipe

Train and analyze sparse autoencoders on ESM2 protein language models using [NVIDIA BioNeMo TransformerEngine-optimized checkpoints](https://huggingface.co/nvidia/esm2_t36_3B_UR50D). The pipeline extracts residual stream activations from protein sequences, trains a TopK SAE, evaluates reconstruction quality and biological alignment (F1 against Swiss-Prot annotations), and generates an interactive feature dashboard.

## Pipeline

```
Extract activations -> Train SAE -> Evaluate + Dashboard
```

**Extract** runs ESM2 over protein sequences (from UniRef50, SwissProt, or FASTA files), saving per-residue hidden states from a target layer to sharded Parquet files. **Train** fits a TopK SAE (8x expansion, top-32 sparsity by default) on those activations. **Evaluate** measures loss recovered, computes F1 scores against Swiss-Prot functional annotations, generates UMAP embeddings, and exports data for the interactive dashboard.

## Prerequisites

Install dependencies:

```bash
# From repo root (UV workspace)
uv sync
```

Protein data is downloaded automatically during extraction (UniRef50 or SwissProt). To use custom sequences, provide a FASTA file.

## Quick Start

```bash
# Full pipeline: extract -> train -> eval
python run.py model=3b

# Skip extraction if activations are already cached
python run.py model=3b steps.extract=false

# Smoke test
python run.py model=650m num_proteins=100 train.n_epochs=1 nproc=1 dp_size=1
```

## Step-by-Step

### 1. Extract Activations

```bash
# Single GPU
python scripts/extract.py \
    --source uniref50 \
    --num-proteins 50000 \
    --model-name nvidia/esm2_t36_3B_UR50D \
    --layer 24 \
    --output .cache/activations/3b_50k_layer24

# Multi-GPU
torchrun --nproc_per_node=4 scripts/extract.py \
    --source uniref50 \
    --num-proteins 50000 \
    --model-name nvidia/esm2_t36_3B_UR50D \
    --layer 24 \
    --output .cache/activations/3b_50k_layer24
```

Supported data sources: `uniref50`, `swissprot`, or a path to a FASTA file. Downloads are cached to `./data/`.

### 2. Train SAE

```bash
python scripts/train.py \
    --cache-dir .cache/activations/3b_50k_layer24 \
    --model-name nvidia/esm2_t36_3B_UR50D \
    --layer 24 \
    --expansion-factor 8 --top-k 32 \
    --batch-size 4096 --n-epochs 3 \
    --output-dir ./outputs/esm2_3b

# Multi-GPU
torchrun --nproc_per_node=4 scripts/train.py \
    --cache-dir .cache/activations/3b_50k_layer24 \
    --model-name nvidia/esm2_t36_3B_UR50D \
    --layer 24 --dp-size 4 \
    --expansion-factor 8 --top-k 32 \
    --batch-size 4096 --n-epochs 3 \
    --output-dir ./outputs/esm2_3b
```

Saves checkpoint to `./outputs/esm2_3b/checkpoints/checkpoint_final.pt`.

### 3. Evaluate + Dashboard

```bash
python scripts/eval.py \
    --checkpoint ./outputs/esm2_3b/checkpoints/checkpoint_final.pt \
    --model-name nvidia/esm2_t36_3B_UR50D \
    --layer 24 --top-k 32 \
    --num-proteins 1000 \
    --output-dir ./outputs/esm2_3b/eval
```

This computes F1 scores against Swiss-Prot annotations, loss recovered metrics, feature statistics, UMAP coordinates, and top activating protein examples. Output goes to `eval/` as Parquet files ready for the dashboard.

## Model Sizes

| Model     | Params | Embedding Dim | Layers | Batch Size | Config       |
| --------- | ------ | ------------- | ------ | ---------- | ------------ |
| ESM2-650M | 650M   | 1280          | 33     | 16         | `model=650m` |
| ESM2-3B   | 3B     | 2560          | 36     | 4          | `model=3b`   |
| ESM2-15B  | 15B    | 5120          | 48     | 1          | `model=15b`  |

## Configuration

Hydra configs live in `run_configs/`. The base config (`config.yaml`) sets defaults for all steps. Model-specific configs in `run_configs/model/` override `model_name`, `run_name`, `num_proteins`, and `batch_size`.

Override any parameter on the command line:

```bash
python run.py model=3b train.n_epochs=5 train.lr=1e-4 nproc=8 source=swissprot
```

Key training defaults: `expansion_factor=8`, `top_k=32`, `lr=3e-4`, `n_epochs=3`, `batch_size=4096`, `layer=24`, `source=uniref50`. <!-- gitleaks:allow -->

## Project Structure

```
recipes/esm2/
  run.py                    Hydra pipeline orchestrator
  run_configs/              Hydra configs (config.yaml, model/*.yaml)
  scripts/
    extract.py              Extract layer activations (multi-GPU)
    train.py                Train TopK SAE (multi-GPU)
    eval.py                 Evaluate + build dashboard data
    launch_dashboard.py     Launch the interactive dashboard
    650m.sh, 3b.sh, 15b.sh  Ready-to-run shell scripts
  src/esm2_sae/             Recipe-specific code
    data/                   Protein data loaders (FASTA, SwissProt, UniRef50)
    eval/                   F1 scores, loss recovered
    analysis/               Protein ranking and interpretability
    viz/                    UMAP, feature stats, top examples
    data_export.py          Parquet/DuckDB export
  protein_dashboard/        React/Vite interactive dashboard
```

## Python API

```python
from sae.architectures import TopKSAE
from esm2_sae import read_fasta, download_swissprot

# Load sequences (returns list of FastaRecord with .id and .sequence)
records = read_fasta("proteins.fasta")
sequences = [r.sequence for r in records]

# Load trained SAE
import torch

ckpt = torch.load("checkpoint_final.pt", map_location="cpu")
sae = TopKSAE(input_dim=2560, hidden_dim=20480, top_k=32)
sae.load_state_dict(ckpt["model_state_dict"])

# Encode to sparse features
codes = sae.encode(embeddings)
```

## Data Export

Save activations to Parquet or DuckDB for downstream analysis:

```python
from esm2_sae import save_activations_parquet, save_activations_duckdb

save_activations_parquet(
    codes=codes, protein_ids=ids, output_path="activations.parquet"
)
save_activations_duckdb(codes=codes, protein_ids=ids, db_path="data.duckdb")
```

Requires: `pip install pyarrow duckdb` or install with the export extra.
