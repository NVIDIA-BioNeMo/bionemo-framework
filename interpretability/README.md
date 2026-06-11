# BioNeMo Interpretability

This directory contains packages and recipes for applying interpretability techniques to BioNeMo models. These packages provide practical, reproducible workflows that help researchers understand what biological foundation models learn — from individual neurons to emergent representations.

## Overview

As biological foundation models grow in scale and capability, understanding their internal representations becomes increasingly important for both scientific insight and responsible deployment. BioNeMo interpretability recipes are designed to be self-contained, hackable references that pair naturally with the models and training recipes in the rest of this repository.

## Techniques

### Sparse Autoencoders (`sparse-autoencoders/`)

Sparse autoencoders (SAEs) are a mechanistic interpretability technique for decomposing dense model activations into sparse, human-interpretable features. This directory provides:

- **`sae/`** — A reusable SAE implementation designed to work with BioNeMo model activations
- **`recipes/`** — End-to-end training and analysis recipes for applying SAEs to specific models:
  - **`esm2/`** — SAE recipes for ESM-2 protein language model representations

## Structure

```
interpretability/
    sparse-autoencoders/
        sae/               # Core SAE implementation
        recipes/
            esm2/          # Recipes for ESM-2
```

## Getting Started

Each recipe directory contains its own README with setup instructions, Docker build commands, and guidance on interpreting results. The general pattern follows the rest of BioNeMo Recipes:

```bash
cd sparse-autoencoders/recipes/esm2
docker build -t esm2_sae .
docker run --rm -it --gpus all esm2_sae python train_sae.py
```

## Contributing

New interpretability techniques and model coverage are welcome. Follow the coding guidelines in the top-level [BioNeMo Recipes README](../README.md) — recipes should be self-contained, clearly documented, and demonstrate one technique well rather than trying to cover everything.
