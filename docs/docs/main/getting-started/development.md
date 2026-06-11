# Development

BioNeMo Recipes favors self-contained, readable recipe code. Each model or recipe directory owns its dependency declarations, tests, and setup instructions.

## Common Workflow

```bash
cd models/esm2      # or recipes/<recipe>
pip install -r requirements.txt
pytest -v .
```

For megatron recipes:

```bash
cd recipes/evo2_megatron
bash .ci_build.sh
source .ci_test_env.sh
pytest -v .
```

## Shared Helpers

The megatron recipes include a local `bionemo.common` package for data loading, FASTA utilities, sharded dataset helpers, inference collation, and checkpoint cleanup. Update the canonical copy in `recipes/evo2_megatron/src/bionemo/common` and sync it to `recipes/eden_megatron/src/bionemo/common`.

## Data And Checkpoints

Use `download_bionemo_data` to list or fetch supported resources:

```bash
download_bionemo_data --list-resources
```
