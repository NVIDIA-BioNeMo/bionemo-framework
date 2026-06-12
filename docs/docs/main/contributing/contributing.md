# Contributing

## CI Labels

- `ciflow:all`: run all tests, including all recipe and model directories.
- `ciflow:notebooks`: run notebook validation where configured.
- `ciflow:skip`: skip recipe tests.

## Local Checks

Run the checks relevant to the directory you modified:

```bash
python ci/scripts/check_copied_files.py
python ci/scripts/recipes_local_test.py recipes/<recipe>
pre-commit run --all-files
```

For copied files, edit the source listed in `ci/scripts/check_copied_files.py`, then run:

```bash
python ci/scripts/check_copied_files.py --fix
```

## Recipe Expectations

Recipes should be self-contained, documented in their local README, and runnable from their own directory. Shared helper code used by megatron recipes lives in `bionemo.common` inside the recipe directories under `recipes/`.
