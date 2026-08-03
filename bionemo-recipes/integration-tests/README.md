# BioNeMo Recipes Integration Tests

Automated tests that validate Claude Code + the bionemo-recipes plugin can successfully
convert vanilla HuggingFace models to use TransformerEngine.

## Prerequisites

- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)
- `ANTHROPIC_API_KEY` environment variable set
- Python with pytest

## Running Tests

```bash
cd bionemo-recipes/integration-tests
pip install -r requirements.txt
pytest -v --timeout=600
```

## Test Structure

- `fixtures/barebones-bert/` — Minimal vanilla BERT model (encoder, MHA)
- `fixtures/barebones-llama/` — Minimal vanilla Llama model (decoder, GQA, SwiGLU)
- `test_te_conversion.py` — Tests that Claude can TE-ify both model types
- `test_fp8_addition.py` — Tests that Claude can add FP8 to an existing TE model
- `test_export.py` — Tests that Claude can create an export script
- `validators/` — Code validation utilities (AST, pattern matching, file checks)

## How It Works

1. Each test copies a fixture model to a temp directory
2. Sends a prompt to Claude Code with the bionemo-recipes plugin loaded
3. Validates the generated code using AST parsing and pattern matching
4. Optionally runs the generated tests on GPU (if available)

## Cost

These tests call the Claude API. Each test costs approximately $1-5 USD depending on complexity.
