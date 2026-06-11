# Example 8M ESM-2 Checkpoint

This directory contains the model and tokenizer configuration and forward pass code for an ESM-2 8M model, identical to
huggingface.co/nvidia/esm2_t6_8m_ur50d. This directory can be useful for testing modifications to the model's forward
pass, and allows the unit tests to run without requiring external HuggingFace API calls. When loading a model directly
from huggingface, the forward pass code is downloaded and run from the user's HF_HOME cache directory, which can
sometimes make interactive debugging difficult.

We use a pre-commit hook (./ci/scripts/check_copied_files.py) to ensure that this model's forward pass stays up to date
with the model definition in the `models/esm2` directory.
