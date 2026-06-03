# bionemo-core

Common code that all BioNeMo framework packages depend on. Contains highly reusable, battle-tested
abstractions and implementations that are valuable across a wide variety of domains and applications.

Crucially, the `bionemo-core` Python package (namespace `bionemo.core`) depends on PyTorch and PyTorch
Lightning. Other BioNeMo component libraries obtain their PyTorch dependencies via `bionemo-core`.

## Developer Setup

After following the setup specified in the [README](../../README.md),
you may install this project's code in your environment via executing:

```bash
pip install -e .
```

To run unit tests with code coverage, execute:

```bash
pytest -v --cov=bionemo --cov-report=term .
```

## Package Highlights

In `bionemo.core.utils`:

- the `batching_utils` module's `pad_token_ids`, which pads token ids with padding value & returns a mask.
- the `dtype` module's `get_autocast_dtype`, which converts from various precision-type representations to their PyTorch equivalents.
- the `random_utils` module, which includes functions for managing random seeds and performing sampling.

In the `bionemo.data` package, there is:

- `multi_epoch_dataset`: contains many dataset implements that are useful for mutli-epoch training.
- `resamplers`: contains a P-RNG based Dataset implementation.

There's a constant global value, `bionemo.core.BIONEMO_CACHE_DIR`, which is used as a local on-disk cache for resources.
