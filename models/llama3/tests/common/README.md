# BioNeMo Common Test Library

Shared test infrastructure for BioNeMo models. One base class, **BaseModelTest**: inherit and implement the abstract methods to get the full test suite (golden values, conversion, FP8, meta init, smoke tests).

## Structure

```text
tests/common/
├── __init__.py             # Public API exports
├── test_modeling_common.py # BaseModelTest, TestTolerances
├── fixtures.py             # input_format, fp8_recipe, te_attn_backend, etc.
└── README.md
```

**Required:** In your top-level `tests/conftest.py` (e.g. `models/esm2/tests/conftest.py`), add:

```python
pytest_plugins = ["tests.common.fixtures"]
```

Without this, parametrized fixtures will not load.

## BaseModelTest

Inherit from `BaseModelTest` and implement:

| Method                                            | Returns                   | Description                                     |
| ------------------------------------------------- | ------------------------- | ----------------------------------------------- |
| `get_model_class()`                               | `Type[PreTrainedModel]`   | TE model class                                  |
| `get_tokenizer()`                                 | `PreTrainedTokenizer`     | Tokenizer                                       |
| `get_config_class()`                              | `Type[PretrainedConfig]`  | Config class                                    |
| `get_upstream_model_id()`                         | `str`                     | HF model ID                                     |
| `get_upstream_model_revision()`                   | `Optional[str]`           | Revision or None                                |
| `get_upstream_model_class()`                      | `Type[PreTrainedModel]`   | HF model class                                  |
| `get_layer_path(model)`                           | `List[nn.Module]`         | Transformer layers                              |
| `get_test_input_data(format, pad_to_multiple_of)` | `Dict[str, torch.Tensor]` | Inputs on CUDA; `format` is `"bshd"` or `"thd"` |
| `get_hf_to_te_converter()`                        | `Callable`                | HF → TE                                         |
| `get_te_to_hf_converter()`                        | `Callable`                | TE → HF                                         |

**Optional overrides:** `get_tolerances()` → `TestTolerances`, `get_attn_input_formats()`, `get_reference_model_no_weights()`.

**Helpers:** `create_test_config()`, `get_reference_model()`, `get_reference_model_no_weights()`, `compare_outputs()`, `verify_model_parameters_initialized_correctly()`, `get_converted_te_model_checkpoint()`, `get_converted_te_model()`.

**Tests included:** Meta/CUDA init (`test_cuda_init`, `test_meta_init`, …), smoke (parametrized by `input_format`), conversion, golden values (BSHD + THD), FP8 (parametrized by `fp8_recipe`, `input_format`).

## TestTolerances

Dataclass in `test_modeling_common.py`. Override `get_tolerances()` to return a custom instance. Fields: `golden_value_*`, `cp_*`, `fp8_*`, `init_*` (see class definition).

## Fixtures (fixtures.py)

| Fixture           | Description                         |
| ----------------- | ----------------------------------- |
| `input_format`    | `"bshd"` / `"thd"`                  |
| `fp8_recipe`      | FP8 recipe (skipped if unsupported) |
| `te_attn_backend` | `"flash_attn"` / `"fused_attn"`     |
| `unused_tcp_port` | For distributed tests               |
| `use_te_debug`    | Autouse: `NVTE_DEBUG=1`             |

## Usage

1. Create a class inheriting from `BaseModelTest` and implement the abstract methods (see `esm2/tests/test_modeling_esm_te.py` for a full example).
2. Add `pytest_plugins = ["tests.common.fixtures"]` to `tests/conftest.py`.
3. Run `pytest tests/test_modeling_<name>_te.py -v`.
