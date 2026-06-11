# Eden Recipe

Eden is a family of genomic models that use the Llama 3.1 architecture, developed by Basecamp Research. Models range from 100M to 35B parameters.

Reference: [**Eden**](https://www.biorxiv.org/content/10.64898/2026.01.12.699009v2) by Basecamp Research.

## Installation

```bash
./.ci_build.sh  # build the virtualenv
source ./.ci_test_env.sh  # source the virtualenv
```

## CLI tools

| Command                         | Description                                          |
| ------------------------------- | ---------------------------------------------------- |
| `train_eden`                    | Train or fine-tune Eden models                       |
| `infer_eden`                    | Autoregressive text generation (greedy/sampling)     |
| `predict_eden`                  | Batch log-likelihood scoring on FASTA sequences      |
| `eden_convert_nemo2_to_mbridge` | Convert NeMo2 checkpoints to MBridge DCP format      |
| `eden_export_mbridge_to_hf`     | Export Eden MBridge checkpoint to HuggingFace Llama  |
| `eden_convert_hf_to_mbridge`    | Convert HuggingFace Llama checkpoint to Eden MBridge |
| `eden_remove_optimizer`         | Strip optimizer state from an MBridge checkpoint     |

## Quick start

### Training with mock data

```bash
torchrun --nproc-per-node 1 --no-python \
  train_eden \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_256 \
  --model-size eden_7b --num-layers 2 --max-steps 5 --eval-interval 5 \
  --eval-iters 1 --mock-data \
  --micro-batch-size 4 --global-batch-size 4 --seq-length 64 \
  --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --mixed-precision-recipe bf16_mixed \
  --no-activation-checkpointing \
  --decay-steps 1000 --warmup-steps 10 \
  --log-interval 1 --seed 41 --dataset-seed 33 \
  --result-dir eden_test
```

Note: fp32_residual_connection is automatically set to False for Eden/TE layers.

### Training with sharded Eden data

For production training, use `--sharded-eden-data` with pre-sharded SQLite sequence
databases and precomputed window databases. See
[`src/bionemo/eden/data/sharded_eden_dataloader.md`](src/bionemo/eden/data/sharded_eden_dataloader.md)
for the full data schema, directory structure, and pre-processing workflow.

```bash
torchrun --nproc-per-node 8 --no-python \
  train_eden \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_256 \
  --model-size eden_7b --max-steps 100000 --eval-interval 500 \
  --eval-iters 32 \
  --sharded-eden-data \
  --sequence-db-dir /path/to/sequence_dbs \
  --train-window-db /path/to/train_windows.db \
  --val-window-db /path/to/val_windows.db \
  --test-window-db /path/to/test_windows.db \
  --micro-batch-size 1 --global-batch-size 64 --seq-length 8192 \
  --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --mixed-precision-recipe bf16_mixed \
  --warmup-steps 2500 --decay-steps 97500 \
  --log-interval 10 --seed 41 --dataset-seed 33 \
  --result-dir /path/to/results
```

The `--stride` (default 7992) and `--window-min-length-threshold` (default 0) flags
control how windows are sampled. Use `--rc-aug` to enable reverse-complement augmentation.

### Fine-tuning from a checkpoint

Resume training from an existing MBridge checkpoint using `--finetune-ckpt-dir`:

```bash
torchrun --nproc-per-node 8 --no-python \
  train_eden \
  --hf-tokenizer-model-path tokenizers/nucleotide_fast_tokenizer_256 \
  --model-size eden_7b --max-steps 10000 --eval-interval 500 \
  --eval-iters 32 --mock-data \
  --finetune-ckpt-dir /path/to/mbridge/checkpoint \
  --micro-batch-size 1 --global-batch-size 64 --seq-length 8192 \
  --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
  --mixed-precision-recipe bf16_mixed \
  --warmup-steps 500 --decay-steps 9500 \
  --lr 1e-4 --min-lr 1e-5 \
  --log-interval 10 --seed 41 \
  --result-dir /path/to/finetune_results
```

The checkpoint directory can contain `iter_*` subdirectories or be a direct checkpoint
directory with `run_config.yaml`. Use `eden_remove_optimizer` first if you only need
the model weights.

### Convert: NeMo2 to MBridge

Convert a NeMo2 DCP checkpoint to MBridge format for use with `train_eden`,
`infer_eden`, and the other MBridge-based tools:

```bash
eden_convert_nemo2_to_mbridge \
  --nemo2-ckpt-dir /path/to/nemo2/checkpoint \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_256 \
  --mbridge-ckpt-dir /path/to/eden_mbridge \
  --model-size eden_7b \
  --seq-length 8192 \
  --mixed-precision-recipe bf16_mixed
```

### Autoregressive generation (infer_eden)

```bash
torchrun --nproc_per_node 1 --no-python \
  infer_eden \
  --ckpt-dir /path/to/mbridge/checkpoint \
  --prompt "ATCGATCGATCGATCG" \
  --max-new-tokens 200 \
  --temperature 1.0 \
  --output-file generated.txt
```

Options: --ckpt-dir, --prompt/--prompt-file, --max-new-tokens, --temperature, --top-k/--top-p, --tensor-parallel-size, --max-seq-length (auto-detected by default, override with EDEN_MAX_SEQ_LEN env var).

### Batch sequence scoring (predict_eden)

```bash
torchrun --nproc_per_node 1 --no-python \
  predict_eden \
  --fasta /path/to/sequences.fasta \
  --ckpt-dir /path/to/mbridge/checkpoint \
  --output-dir predictions/ \
  --micro-batch-size 4 \
  --write-interval epoch
```

## Exporting / importing Eden (Llama) checkpoints

### Export: MBridge to HuggingFace

```bash
eden_export_mbridge_to_hf \
  --mbridge-ckpt-dir /path/to/eden_mbridge/iter_0000001 \
  --hf-output-dir /path/to/eden_hf \
  --model-size eden_7b
```

Produces standard HuggingFace directory loadable with `LlamaForCausalLM.from_pretrained()`.

### Import: HuggingFace to MBridge

```bash
eden_convert_hf_to_mbridge \
  --hf-model-dir /path/to/eden_hf \
  --mbridge-ckpt-dir /path/to/eden_mbridge_reimported \
  --model-size eden_7b
```

## Removing optimizer state from a checkpoint

Training checkpoints include optimizer state (Adam moments, LR scheduler, RNG state)
which roughly triples checkpoint size. Use `eden_remove_optimizer` to produce a
smaller weights-only checkpoint suitable for release or fine-tuning:

```bash
eden_remove_optimizer \
  --src-ckpt-dir /path/to/training/checkpoints \
  --dst-ckpt-dir /path/to/weights_only_checkpoint
```

The tool automatically finds the latest `iter_*` directory, strips optimizer and
scheduler state from the DCP files, and copies model weights, tokenizer, and
config files to the destination. The resulting checkpoint is directly usable
with `--finetune-ckpt-dir` or the export tools.

## Model sizes

| Key         | Description             |
| ----------- | ----------------------- |
| `eden_100m` | Eden ~100M              |
| `eden_300m` | Eden ~300M              |
| `eden_1b`   | Eden ~1B                |
| `eden_7b`   | Eden base (~8B params)  |
| `eden_11b`  | Eden ~11B               |
| `eden_18b`  | Eden ~18B               |
| `eden_21b`  | Eden ~21B               |
| `eden_24b`  | Eden ~24B (32K context) |
| `eden_27b`  | Eden ~27B (32K context) |
| `eden_28b`  | Eden ~28B               |
| `eden_35b`  | Eden ~35B               |

## Data

Eden uses the ShardedEdenDataset from Basecamp Research, backed by SQLite for fast windowed access to genomic sequences. The data utilities are provided by the `bionemo.common` sub-package.

## Docker build

```bash
docker build -t eden_megatron_recipe-$(git rev-parse --short HEAD) .
```
