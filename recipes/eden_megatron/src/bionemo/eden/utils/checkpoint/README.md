# Eden Checkpoint Conversion Library

This library provides CLI tools and utilities for converting Eden (Llama)
checkpoints between MBridge and HuggingFace formats. All conversions go
through the **MBridge** (Megatron Bridge) checkpoint format, which is the
native format used for training and inference in this recipe.

## MBridge checkpoint structure

An MBridge checkpoint is a directory containing one or more iteration
subdirectories, plus metadata files at the top level:

```
eden_7b_mbridge/
├── latest_checkpointed_iteration.txt
├── latest_train_state.pt
└── iter_0000001/
    ├── run_config.yaml
    ├── common.pt
    ├── train_state.pt
    ├── .metadata
    └── __0_*.distcp          # DCP (Distributed Checkpoint) shard files
```

| File / directory                    | Description                                                                                   |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| `latest_checkpointed_iteration.txt` | Plain text file containing the latest iteration number (e.g. `1`)                             |
| `latest_train_state.pt`             | Top-level training state snapshot                                                             |
| `iter_NNNNNNN/`                     | Checkpoint data for iteration N                                                               |
| `iter_NNNNNNN/run_config.yaml`      | Full Megatron Bridge `ConfigContainer` used to create this checkpoint (model, optimizer, ...) |
| `iter_NNNNNNN/common.pt`            | Shared metadata used by PyTorch Distributed Checkpoint (DCP)                                  |
| `iter_NNNNNNN/train_state.pt`       | Training state (optimizer moments, scheduler, iteration counter)                              |
| `iter_NNNNNNN/.metadata`            | DCP planner metadata describing how weights are sharded                                       |
| `iter_NNNNNNN/__0_*.distcp`         | DCP shard files containing the model weights                                                  |

When a tool expects `--mbridge-ckpt-dir`, point it at the **top-level**
directory (e.g. `eden_7b_mbridge/`). When a tool expects an iteration
directory (e.g. for export), point it at `eden_7b_mbridge/iter_0000001/`.

## CLI tools

| Command                         | Description                                      |
| ------------------------------- | ------------------------------------------------ |
| `eden_convert_nemo2_to_mbridge` | Convert a NeMo2 checkpoint to MBridge format     |
| `eden_export_mbridge_to_hf`     | Export an Eden MBridge checkpoint to HuggingFace |
| `eden_convert_hf_to_mbridge`    | Import a HuggingFace Llama checkpoint to MBridge |

Run any tool with `--help` for full usage details.

### Converting NeMo2 to MBridge

```bash
eden_convert_nemo2_to_mbridge \
  --nemo2-ckpt-dir /path/to/nemo2/checkpoint \
  --mbridge-ckpt-dir eden_7b_mbridge \
  --model-size eden_7b \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_256 \
  --seq-length 8192 \
  --mixed-precision-recipe bf16_mixed
```

### Exporting MBridge to HuggingFace

```bash
eden_export_mbridge_to_hf \
  --mbridge-ckpt-dir eden_7b_mbridge/iter_0000001 \
  --hf-output-dir eden_7b_hf \
  --model-size eden_7b
```

This produces a standard HuggingFace directory with `config.json` and
safetensors weight files, loadable via:

```python
from transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("eden_7b_hf")
```

### Importing HuggingFace to MBridge

```bash
eden_convert_hf_to_mbridge \
  --hf-model-dir eden_7b_hf \
  --mbridge-ckpt-dir eden_7b_mbridge \
  --model-size eden_7b
```

### Common options

- `--model-size` -- model key such as `eden_7b`, `eden_11b`, `eden_35b`, etc.
- `--no-te` -- disable Transformer Engine fused layernorm key mapping.
- `--verbose` / `-v` -- enable debug logging.
