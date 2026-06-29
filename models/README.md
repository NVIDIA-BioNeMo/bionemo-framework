# Models Directory

This directory contains HuggingFace-compatible model implementations that use TransformerEngine layers internally. These models are designed to be distributed through the Hugging Face Hub and serve as drop-in replacements for standard transformer models with enhanced performance.

## Overview

Models in this directory are **not intended to be pip-installed directly**. Instead, they serve as:

- **Reference implementations** of biological foundation models using TransformerEngine
- **Conversion utilities** for transforming existing model checkpoints to TE-compatible format
- **Export tools** for preparing model releases on the Hugging Face Hub

Users will typically interact with these models by loading pre-converted checkpoints directly from the Hugging Face Hub using standard transformers APIs.

## Adding a New Model

### Minimum Requirements

To add a new model to this directory, you must provide:

#### 1. Golden Value Tests

- **Accuracy validation**: Tests that demonstrate the converted TE model produces identical outputs to the source/reference model
- **Numerical precision**: Verify outputs match within acceptable tolerance (typically `rtol=1e-5, atol=1e-8`)
- **Multiple test cases**: Cover different input shapes, batch sizes, and edge cases

#### 2. State Dict Conversion Functions

- **`convert_hf_to_te()`**: Function to convert HuggingFace model state_dict to TransformerEngine format
- **`convert_te_to_hf()`**: Function to convert TransformerEngine state_dict back to HuggingFace format
- **Bidirectional validation**: Tests ensuring round-trip conversion preserves model weights

#### 3. Checkpoint Export Script

- **`export.py`**: Script that packages all necessary files for Hugging Face Hub upload
- **Complete asset bundling**: Must include all required files, refer to [Export Requirements](#export-requirements)
- **Automated process**: Should be runnable with minimal manual intervention

#### 4. Open Source License

- **Approved license**: Must include an approved open-source license (MIT, Apache 2.0, etc.)
- **License compatibility**: Ensure license is compatible with source model and dependencies
- **Clear attribution**: Proper attribution to original model authors if applicable

### Directory Structure

Each model should follow this standardized layout:

```
models/{model_name}/
├── Dockerfile                   # Container definition for testing
├── .dockerignore                # Docker ignore patterns
├── pyproject.toml               # Python package configuration
├── README.md                    # Model-specific documentation
├── model_readme.template        # Detailed model card for Hub upload
├── export.py                    # Checkpoint export utilities
├── LICENSE                      # Open source license file
├── src/                         # Source code directory
│   └── {model_name}/           # Package directory
│       ├── __init__.py         # Package initialization
│       ├── {model_name}_te.py  # TransformerEngine model implementation
│       ├── convert.py          # HF ↔ TE conversion utilities
│       └── modeling_{...}.py   # Additional model-specific modules
└── tests/                       # Test suite
    ├── conftest.py              # Pytest configuration and fixtures
    ├── test_golden_values.py    # Golden value validation tests
    ├── test_conversion.py       # State dict conversion tests
    └── test_checkpoint.py       # Save/load functionality tests
```

## Implementation Guidelines

### Model Implementation (`{model_name}_te.py`)

Your TransformerEngine model should:

- **Inherit from `PreTrainedModel`**: Follow HuggingFace conventions for model structure
- **Use TE layers**: Replace standard PyTorch layers with TransformerEngine equivalents
- **Maintain API compatibility**: Support the same forward pass signature as the original model
- **Include configuration**: Provide a configuration class that extends `PretrainedConfig`

```python
from transformers import PreTrainedModel, PretrainedConfig
from transformer_engine.pytorch import TransformerLayer


class MyModelTEConfig(PretrainedConfig):
    model_type = "my_model_te"
    # ... configuration parameters


class MyModelTE(PreTrainedModel):
    config_class = MyModelTEConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize with TE layers

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # Forward pass implementation
        pass
```

### Conversion Functions (`convert.py`)

Implement bidirectional conversion between HuggingFace and TransformerEngine state dictionaries. We use a module adapted
from the nemo.lightning.io.apply_transforms function to handle the conversion.

```python
def convert_hf_to_te(model_hf: nn.Module, **config_kwargs) -> nn.Module:
    """Convert HuggingFace model to TransformerEngine format."""
    te_config = MyModelTEConfig(**model_hf.config.to_dict(), **config_kwargs)
    with init_empty_weights():
        model_te = MyModelTE(te_config, dtype=te_config.dtype)

    output_model = io.apply_transforms(model_hf, model_te, ...)
    return output_model


def convert_te_to_hf(
    hf_model_tag: str, model_te: nn.Module, **config_kwargs
) -> nn.Module:
    """Convert TransformerEngine model to HuggingFace format."""
    with init_empty_weights():
        model_hf = AutoModel.from_pretrained(hf_model_tag)

    output_model = io.apply_transforms(model_te, model_hf, ...)

    return output_model
```

### Testing Requirements

#### Golden Value Tests (`test_golden_values.py`)

```python
import pytest
import torch
from transformers import AutoModel
from src.my_model.my_model_te import MyModelTE
from src.my_model.convert import convert_hf_to_te


def test_model_outputs_match_reference():
    """Test that TE model outputs match reference HF model."""
    # Load reference model
    reference_model = AutoModel.from_pretrained("original/model")

    # Create TE model with converted weights
    te_model = MyModelTE.from_pretrained("original/model")

    # Test with various inputs
    test_inputs = [...]  # Different input shapes and types

    for inputs in test_inputs:
        with torch.no_grad():
            ref_output = reference_model(**inputs)
            te_output = te_model(**inputs)

        torch.testing.assert_close(
            te_output.last_hidden_state,
            ref_output.last_hidden_state,
            rtol=1e-5,
            atol=1e-8,
        )
```

#### Conversion Tests (`test_conversion.py`)

```python
def test_bidirectional_conversion():
    """Test that state dict conversion is bidirectional."""
    # Load original state dict
    original_state_dict = ...

    # Convert HF -> TE -> HF
    te_state_dict = convert_hf_to_te(original_state_dict)
    recovered_state_dict = convert_te_to_hf(te_state_dict)

    # Verify weights are preserved
    for key in original_state_dict:
        torch.testing.assert_close(original_state_dict[key], recovered_state_dict[key])
```

#### Checkpoint Tests (`test_checkpoint.py`)

```python
def test_save_and_load_pretrained():
    """Test that model can be saved and loaded with HF APIs."""
    model = MyModelTE.from_pretrained("original/model")

    # Save model
    model.save_pretrained("./test_checkpoint")

    # Load model
    loaded_model = MyModelTE.from_pretrained("./test_checkpoint")

    # Verify models are equivalent
    # ... test logic
```

## Export Requirements

The `export.py` script must bundle all necessary assets for Hugging Face Hub upload:

### Required Files

1. **Model weights**: `pytorch_model.bin` or `model.safetensors`
2. **Configuration**: `config.json` with proper `auto_map` section
3. **Model code**: All source files needed to instantiate the model
4. **Tokenizer**: `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, etc.
5. **Model Card**: `model_readme.template` model card
6. **License**: `LICENSE` file with approved open-source license
7. **Requirements**: `requirements.txt` for any additional dependencies

### Config.json Auto Map

Ensure the exported `config.json` includes the auto_map section:

```json
{
    "auto_map": {
        "AutoModel": "modeling_my_model.MyModelTE",
        "AutoConfig": "configuration_my_model.MyModelTEConfig"
    }
}
```

### Export Script Template

```python
#!/usr/bin/env python3
"""Export script for MyModel checkpoint."""

import os
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from src.my_model.my_model_te import MyModelTE


def export_checkpoint(output_dir: str):
    """Export complete checkpoint for Hugging Face Hub."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load and save model
    model = MyModelTE.from_pretrained("source/checkpoint")
    model.save_pretrained(output_path)

    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained("source/checkpoint")
    tokenizer.save_pretrained(output_path)

    # Copy source code files
    src_files = ["modeling_my_model.py", "configuration_my_model.py"]
    for file in src_files:
        shutil.copy(f"src/my_model/{file}", output_path / file)

    # Copy documentation and license
    shutil.copy(
        "model_readme.md", output_path / "README.md"
    )  # Or alternative template-based creation of README.md
    shutil.copy("LICENSE", output_path / "LICENSE")

    print(f"Checkpoint exported to {output_path}")


if __name__ == "__main__":
    export_checkpoint("./exported_checkpoint")
```

## CI/CD Integration

Each model must pass the standard CI/CD contract:

```bash
cd models/my_model
docker build -t my_model_test .
docker run --rm -it --gpus all my_model_test pytest -v .
```

The Docker container should:

- Install all required dependencies
- Run the complete test suite
- Validate that all conversion and export functionality works
- Complete in reasonable time for CI/CD pipelines

## Examples

For reference implementations, see existing models in this directory:

- `esm2/`: Protein language model with bidirectional conversion
- `amplify/`: DNA foundation model with comprehensive export utilities
- `geneformer/`: Single-cell gene expression model

## License Requirements

Ensure your exported model is packaged with a LICENSE file containing an approved open-source
license for external distribution.

## Support

For questions about adding new models:

1. Review existing model implementations for examples
2. Check the main project README for general guidelines
3. Ensure all tests pass before submitting contributions

Remember: The goal is to provide high-performance, easy-to-use biological foundation models that researchers can readily adopt and adapt for their work.
