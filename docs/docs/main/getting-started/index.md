# Getting Started

## Repository Structure

BioNeMo Recipes is organized around three root directories:

- `models/`: TransformerEngine-backed model implementations and export utilities.
- `recipes/`: training, fine-tuning, inference, and convergence recipes.
- `interpretability/`: interpretability workflows such as sparse autoencoders.

Documentation source is stored in `docs/`.

## Local Development

Start from the README in the model or recipe directory you are modifying:

```bash
cd recipes/evo2_megatron
bash .ci_build.sh
pytest -v .
```

Many recipes are self-contained packages. Shared helper code used by the megatron recipes lives inside each package as `bionemo.common`.

## Next Steps

- For model implementations, start in `models/`.
- For runnable workflows, start in `recipes/`.
- For interpretability workflows, start in `interpretability/`.
