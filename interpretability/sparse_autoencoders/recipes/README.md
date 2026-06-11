# Recipes

This directory contains model-specific implementations of Sparse Autoencoders following the NVIDIA BioNeMo pattern. Each recipe is a self-contained package that builds on the generic `sae` core package.

## Structure

```
recipes/
├── README.md              # This file
└── esm2/                  # ESM2 protein language model recipe
    ├── pyproject.toml     # Package configuration
    ├── README.md          # ESM2-specific documentation
    ├── src/
    │   └── esm2_sae/     # ESM2-specific implementation
    ├── scripts/           # Training scripts
    ├── configs/           # Hydra configuration files
    └── data/             # Data directory
```

## Available Recipes

### ESM2 (`recipes/esm2/`)

Sparse Autoencoders for ESM2 protein language models.

**Features:**

- ESM2 model wrappers (8M to 3B parameters)
- Protein dataset loaders (FASTA, SwissProt)
- F1 evaluation against Swiss-Prot annotations
- Visualization pipeline for feature analysis
- Hydra-based training configs

**Quick Start:**

```bash
cd recipes/esm2
python scripts/train.py --config-name config_production
```

See `recipes/esm2/README.md` for detailed documentation.

## Recipe Philosophy

Each recipe follows these principles:

1. **Self-Contained**: Can be installed and used independently
2. **Depends on Core**: Imports from generic `sae` package for SAE implementations
3. **Domain-Specific**: Contains only model/domain-specific code
4. **Organized**: Following NVIDIA BioNeMo structure (src/, scripts/, configs/, data/)
5. **Documented**: Comprehensive README with examples

## Adding a New Recipe

To add a new recipe (e.g., for a different model):

1. **Create directory structure:**

   ```bash
   mkdir -p recipes/mymodel/src/mymodel_sae
   mkdir -p recipes/mymodel/{scripts,configs,data}
   ```

2. **Create `pyproject.toml`:**

   ```toml
   [project]
   name = "mymodel-sae"
   dependencies = [
       "sae>=0.1.0",  # Depend on core SAE package
       # Add model-specific dependencies
   ]
   ```

3. **Implement model-specific code:**

   - `src/mymodel_sae/models/`: Model wrappers
   - `src/mymodel_sae/data/`: Dataset loaders
   - `src/mymodel_sae/eval/`: Domain-specific evaluation
   - `scripts/`: Training scripts
   - `configs/`: Hydra configs

4. **Update workspace:**
   Add to root `pyproject.toml`:

   ```toml
   [tool.uv.workspace]
   members = ["sae", "recipes/esm2", "recipes/mymodel"]
   ```

5. **Document:**
   Create `recipes/mymodel/README.md` with usage examples

## Recipe vs. Core

**Core SAE Package (`sae/`):**

- Generic SAE architectures (ReLU-L1, Top-K)
- Training loop and configuration
- Generic evaluation metrics
- No model/domain-specific code

**Recipe Packages (`recipes/*/`):**

- Model wrappers for embedding extraction
- Domain-specific data loaders
- Domain-specific evaluation metrics
- Training scripts with configs
- Visualization pipelines

## Installation

### Development (all recipes)

```bash
# From repository root
uv sync
```

### Individual recipe

```bash
# Install core first
pip install -e sae/

# Then install recipe
pip install -e recipes/esm2/
```

## Example: Training Pipeline

1. **Recipe provides model and data:**

   ```python
   from esm2_sae.models import ESM2Model
   from esm2_sae.data import download_swissprot, read_fasta
   ```

2. **Core provides SAE and training:**

   ```python
   from sae.architectures import TopKSAE
   from sae.training import Trainer, TrainingConfig
   ```

3. **Recipe provides domain-specific evaluation:**

   ```python
   from esm2_sae.eval import compute_f1_scores
   ```

This separation keeps the core package minimal and domain-agnostic while allowing rich, domain-specific functionality in recipes.

## Future Recipes

Potential recipes to add:

- `recipes/geneformer/`: Sparse SAEs for Geneformer gene expression models
- `recipes/amplify/`: Sparse SAEs for NVIDIA BioNeMo AmplifyProt
- `recipes/esmc/`: Sparse SAEs for ESM-C protein folding models
- `recipes/vision/`: Sparse SAEs for vision transformers (non-bio example)

## Contributing

When adding a new recipe:

1. Follow the structure of `recipes/esm2/`
2. Keep domain-specific code in the recipe
3. Contribute generic improvements to `sae/`
4. Include comprehensive README
5. Add example scripts
6. Include configs for reproducibility

## License

MIT License
