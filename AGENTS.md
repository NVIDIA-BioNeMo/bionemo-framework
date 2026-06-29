# Repository Instructions

## General development

- Keep changes scoped to the relevant model, recipe, or CI path. Avoid drive-by refactors while
  fixing a narrow issue.
- Prefer the patterns already used by the nearest model or recipe over introducing new helper
  layers.
- Add or update focused tests when changing model behavior, conversion logic, training behavior, or
  CI checks.
- For Python formatting and linting, follow the local `.ruff.toml` or per-folder Ruff config when
  present.

## Copied files

Some model and recipe files are intentionally duplicated. Keep the mapping in
`ci/scripts/check_copied_files.py` as the source of truth for these copies.

- When editing a file listed in `SOURCE_TO_DESTINATION_MAP`, update the source copy and run:

  ```bash
  python ci/scripts/check_copied_files.py --fix
  ```

- If adding, moving, or deleting an intentionally copied file or directory, update
  `SOURCE_TO_DESTINATION_MAP` in `ci/scripts/check_copied_files.py` in the same change.

- Do not hand-edit destination copies that contain a copied-file notice; change the source and
  regenerate the destinations with the script.

## Model and recipe boundaries

Treat each top-level `models/{model_name}` and `recipes/{recipe_name}` folder as an independent
entity.

- Do not add imports from one model folder into another model folder.
- Do not add imports from one recipe folder into another recipe folder.
- Do not make recipes import implementation details from `models/` folders unless the existing
  recipe already owns that dependency pattern.
- Prefer intentional duplication over cross-folder shared imports. If code must remain identical
  across folders, add or update the copy mapping in `ci/scripts/check_copied_files.py` instead.

## Transformer Engine model tests

When adding a new Transformer Engine model under `models/`, use the shared common test harness as
the starting point instead of building a one-off suite.

- Follow the pattern in `models/llama3/tests/common/README.md` and the existing model tests that
  inherit from `tests.common.BaseModelTest`.
- Add `pytest_plugins = ["tests.common.fixtures"]` to the model's `tests/conftest.py` so the shared
  fixtures load correctly.
- Implement the required `BaseModelTest` hooks for the new model, including model/config classes,
  tokenizer, upstream Hugging Face model details, layer access, sample inputs, and HF-to-TE /
  TE-to-HF conversion functions.
- Cover golden-value parity, conversion round trips, initialization behavior, FP8 paths, and smoke
  inputs through the harness where applicable.
- The common harness is also managed by `ci/scripts/check_copied_files.py`; update the source copy
  and regenerate destinations if the harness itself changes.
