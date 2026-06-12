# Sparse Autoencoders for Model Interpretability

The Sparse Autoencoder(SAE) package provides a domain-agnostic implementation of SAE designed for mechanistic model interpretability. This tool provides a generic SAE library with various SAE architectures, as well as comprehensive training and analysis tools. Model-specific recipes are included to guide researchers on applying SAEs to any chosen model.

[Check out CodonFM's extracted features on this interactive dashboard built with this SAE library.](https://research.nvidia.com/labs/dbr/blog/sae) *CodonFM recipe coming soon!*

> **Early Release** -- This project is under active development. APIs and interfaces may change between versions. Feedback and contributions are welcome.

## Structure

```
sparse_autoencoders/
├── sae/                  # Core SAE library (domain-agnostic)
│   └── src/sae/
│       ├── architectures/ # ReLU-L1, Top-K, MoE SAE implementations
│       ├── training.py    # Trainer with checkpointing and logging
│       ├── eval/          # Reconstruction, sparsity, dead latent metrics
│       ├── autointerp/    # LLM-based automated feature interpretation
│       ├── collector.py   # Activation collection utilities
│       └── dashboard/     # Interactive feature visualization (React)
├── recipes/
│   ├── esm2/             # ESM2 protein language model recipe
│   │   ├── scripts/      # extract, train, eval, dashboard launchers
│   │   ├── configs/      # Hydra configs for 650M, 3B, 15B models
│   │   └── src/esm2_sae/ # ESM2-specific data, eval, and viz
└── pyproject.toml        # UV workspace configuration
```

## Quick Start

```bash
# Install everything (recommended)
uv sync

# Or install packages individually
pip install -e sae/
pip install -e recipes/esm2/
```

### Train an SAE on ESM2

```bash
cd recipes/esm2

# Full pipeline (extract activations -> train -> evaluate)
python run.py model=3b

# Quick smoke test
python run.py model=650m num_proteins=100 train.n_epochs=1 nproc=1 dp_size=1
```

### Use the core library standalone

```python
from sae.architectures import TopKSAE
from sae.training import Trainer, TrainingConfig

sae = TopKSAE(input_dim=512, hidden_dim=4096, top_k=64)
trainer = Trainer(sae, TrainingConfig(lr=3e-4, n_epochs=10, batch_size=4096))
trainer.train(embeddings)
```

## SAE Architectures

| Architecture       | Class     | Sparsity Mechanism                     |
| ------------------ | --------- | -------------------------------------- |
| ReLU + L1          | `ReLUSAE` | L1 penalty on activations              |
| Top-K              | `TopKSAE` | Only top-K features activate per input |
| Mixture of Experts | `MoESAE`  | Expert routing with sparse gating      |

## Feature Exploration

Each recipe includes an interactive exploration UI built on [Mosaic](https://github.com/uwdata/mosaic) and [DuckDB](https://duckdb.org/), with an [Embedding Atlas](https://github.com/apple/embedding-atlas) view for navigating learned features. The dashboard supports browsing per-feature activation patterns, filtering by annotation or sequence, and exploring UMAP projections of the feature space -- all backed by fast, in-browser SQL queries over Parquet files.

## Applicability Beyond Biology

While the included recipes target biological foundation models such as ESM2, the core `sae` library is entirely domain-agnostic. Sparse autoencoders are a general-purpose interpretability technique and work equally well on natural language and vision transformer activations. To apply SAEs to a new domain, create a recipe that handles embedding extraction and domain-specific evaluation while reusing the shared training and architecture code.

## Roadmap

- **Parallelism beyond DDP** -- tensor parallelism and fused sparse Top-K kernels for training SAEs on large-scale activations
- **Additional architectures** -- gated SAEs, JumpReLU, and other emerging variants
- **More recipes** -- Geneformer, AmplifyProt, ESM-C, and non-biological reference recipes

## Adding a New Recipe

1. Create `recipes/mymodel/` with `src/`, `scripts/`, `configs/` dirs
2. Add a `pyproject.toml` depending on `sae>=0.1.0`
3. Register in the root `pyproject.toml` workspace members
4. See `recipes/esm2/` as a reference implementation
