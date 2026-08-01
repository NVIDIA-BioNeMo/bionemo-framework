# Validation

Phase 5. A port that runs is not a port that works. The question is always *"does the accelerated
model compute the same thing as the original?"* — and this repo already has the harness that
answers it.

Validation is tiered. Each tier is **mandatory when its precondition holds**. Do not report success
with a failing tier; fix the port.

______________________________________________________________________

## Tier 1 — parity check (always)

Generate `parity_check.py` in the target repo from `assets/parity_check.py.tmpl`. It:

1. Builds the original and ported models from the same seed.
2. Runs the converter (Depth B) or loads the same weights (Depths A and C).
3. Asserts forward logits and loss match at BF16 tolerances.
4. Asserts the backward pass runs and all gradients are finite.
5. Sweeps every hardware-supported recipe at FP8 tolerances.
6. If packing was applied, asserts BSHD and THD produce the same outputs.

### Tolerances

Take them from `$BIONEMO_RECIPES/models/esm2/tests/common/test_modeling_common.py::TestTolerances` rather than
inventing your own:

| Tolerance                                  | Default         |
| ------------------------------------------ | --------------- |
| `golden_value_loss_atol` / `rtol`          | `1e-2` / `1e-3` |
| `golden_value_logits_atol` / `rtol`        | `2.0` / `1e-4`  |
| `golden_value_hidden_states_atol` / `rtol` | `0.1` / `0.05`  |
| `fp8_loss_atol` / `rtol`                   | `0.1` / `0.05`  |
| `fp8_logits_atol` / `rtol`                 | `5.0` / `0.1`   |
| `init_mean_atol` / `rtol`                  | `1e-3` / `1e-4` |

The logits `atol` looks enormous because logits are compared with a tiny `rtol` — the pair is
calibrated together. Do not tighten one without the other.

Per-model overrides are legitimate and the references do it — see
`$BIONEMO_RECIPES/models/llama3/tests/test_modeling_llama_te.py::TestLlama3Model.get_tolerances`, which loosens loss
tolerances and raises CP tolerances "due to causal LM boundary effects". If you loosen a tolerance,
**say so in the report with the reason**.

______________________________________________________________________

## Tier 2 — `BaseModelTest` harness (when Depth B produced a converter)

This is the strongest evidence available, and it costs almost nothing because the tests are
inherited.

### Copy the harness

Copy `$BIONEMO_RECIPES/models/esm2/tests/common/` from this repo into `<target>/tests/common/`. Copy from
`$BIONEMO_RECIPES/models/esm2/` specifically — it is the source in
`$BIONEMO_RECIPES/ci/scripts/check_copied_files.py::SOURCE_TO_DESTINATION_MAP`; the copies under
`$BIONEMO_RECIPES/models/llama3|mixtral|qwen|codonfm/tests/common/` are generated destinations. Because this skill
lives inside the repo, you are always copying the current version — there is no vendored snapshot to
drift.

Contents: `__init__.py` (exports `BaseModelTest`, `TestTolerances`, `HAS_DATA_CENTER_GPU`),
`test_modeling_common.py`, `fixtures.py`, `README.md`.

### Wire the conftest

`<target>/tests/conftest.py` must contain:

```python
pytest_plugins = ["tests.common.fixtures"]
```

Without it the `fp8_recipe`, `input_format`, and `te_attn_backend` fixtures do not load and
collection fails. Use `assets/conftest.py.tmpl`; it also appends the model root to `sys.path`, since
the reference recipes import modules by flat name.

### Implement the hooks

Generate from `assets/test_modeling_ported.py.tmpl`. Ten abstract methods, all required:

| Hook                                              | Returns                                             |
| ------------------------------------------------- | --------------------------------------------------- |
| `get_model_class()`                               | the TE `PreTrainedModel` subclass                   |
| `get_config_class()`                              | its `PretrainedConfig` subclass                     |
| `get_tokenizer()`                                 | a `PreTrainedTokenizer` (set `pad_token` if `None`) |
| `get_upstream_model_id()`                         | HF hub id of the *original* model                   |
| `get_upstream_model_revision()`                   | a pinned commit — never a moving branch             |
| `get_upstream_model_class()`                      | the original HF model class                         |
| `get_layer_path(model)`                           | `list[nn.Module]` of transformer layers             |
| `get_test_input_data(format, pad_to_multiple_of)` | CUDA tensor dict; must handle `"bshd"` and `"thd"`  |
| `get_hf_to_te_converter()`                        | the conversion callable                             |
| `get_te_to_hf_converter()`                        | the inverse                                         |

Optional: `get_tolerances()`, `get_attn_input_formats()`, `create_test_config(**kwargs)` (override to
cut `num_hidden_layers=2` for speed — llama3 does), `get_reference_model_no_weights()`.

For causal LMs set `is_autoregressive = True` and implement
`create_inference_params(config, batch_size, max_seq_len, num_beams)`. This unlocks the
generation and KV-cache tests, which stay dormant otherwise.

Worked example to copy the shape from: `$BIONEMO_RECIPES/models/llama3/tests/test_modeling_llama_te.py`.

### What you get for free

- **Golden values** — `test_golden_values` (TE vs HF reference in bf16),
  `test_golden_values_thd` (BSHD vs THD equivalence — the packing correctness proof),
  `test_thd_padding_input_data_equivalence`
- **Conversion** — `test_convert_hf_to_te`, `test_convert_te_to_hf`,
  `test_convert_te_to_hf_roundtrip`, `test_convert_config`
- **FP8**, parametrized over every recipe × input format —
  `test_fp8_forward_and_backward_pass`, `test_quantized_model_init_forward_and_backward`, and the
  `test_legacy_*` variants
- **Init** — `test_cuda_init`, `test_meta_init`, `test_cuda_fp8_init`, `test_meta_fp8_init`
- **Smoke** — `test_smoke_forward_pass`, `test_smoke_backward_pass`,
  `test_smoke_model_with_loss`, `test_forward_and_backward`
- **Generation**, when `is_autoregressive` — `test_generate_without_cache`,
  `test_generate_with_cache`, `test_generate_with_cache_batched`,
  `test_generate_with_cache_beam_search`

### Preserve the xfail-not-skip design

`$BIONEMO_RECIPES/models/esm2/tests/common/fixtures.py::parametrize_recipes_with_support` marks unsupported recipes
**xfail, not skip**, so they still execute and an unexpected pass is visible. Do not "fix" this into
a skip when adapting the harness.

The recipe list it sweeps (`ALL_RECIPES`):

```python
DelayedScaling()
Float8CurrentScaling()
Float8BlockScaling()
MXFP8BlockScaling()
NVFP4BlockScaling(disable_rht=True, disable_stochastic_rounding=True)
```

Note the NVFP4 kwargs — random Hadamard transforms and stochastic rounding are disabled so the test
is deterministic. Keep them.

______________________________________________________________________

## Tier 3 — CI parity (when the target is containerized)

Reproduce this repo's CI exactly:

```bash
python $BIONEMO_RECIPES/ci/scripts/recipes_local_test.py <target_dir>
```

It docker-runs `pip install -e .` (or `-r requirements.txt`) then `pytest -v .` inside the target
directory, in `svcbionemo023/bionemo-framework:pytorch26.04-py3-squashed` with
`--gpus all --shm-size=16G`. Per-directory escape hatches `.ci_build.sh` (custom install) and
`.ci_test_env.sh` (env sourced before pytest) exist if the target needs them.

Also run `pre-commit run --all-files` if the target has a pre-commit config.

### State what CI could not prove

PR CI in this repo runs on a **single L4**. That means the following are *skipped, not passed*:

- everything gated on `requires_multi_gpu` (`torch.cuda.device_count() < 2`)
- everything gated on `requires_datacenter_hardware` / `HAS_DATA_CENTER_GPU`
  (H100/H200/B100/B200/B300)
- MXFP8 and NVFP4 paths, which need Blackwell

A green CI run is **not** evidence those paths work. Say so explicitly in the report and recommend
a DGX/GB200 run for the recommended recipe before any production training.

For real convergence, point at `.github/workflows/convergence-tests.yml`, which submits multi-node
Lepton jobs via `$BIONEMO_RECIPES/ci/lepton/core/launch_job.py`. That is out of PR CI and runs bi-weekly.

______________________________________________________________________

## Recipe-level sanity tests, if the target has a training loop

Beyond numerics, the reference recipes assert the model still *trains*. Pattern from
`$BIONEMO_RECIPES/recipes/esm2_native_te/tests/test_train.py`: build a small Hydra config, run a few hundred steps,
assert the loss threshold.

```python
final_loss = main_fsdp2(sanity_config)
assert final_loss < 3.0, f"Final loss {final_loss} is too high"
```

Generate the equivalent for the target if it has a runnable short config. Gate FP8 tests the way the
reference does:

```python
requires_fp8 = pytest.mark.skipif(
    not torch.cuda.is_available() or not check_fp8_support()[0],
    reason="FP8 not supported on this device",
)
```

Note that at the recipe level FP8 tests mostly assert *it runs*; genuine FP8 numerical parity lives
in Tier 2. Do not present a passing recipe-level FP8 test as parity evidence.
