# Evo2 Checkpoint Conversion Library

This library provides CLI tools and utilities for converting Evo2 (Hyena)
checkpoints between different formats. All
conversions go through the **MBridge** (Megatron Bridge) checkpoint format,
which is the native format used for training and inference in this recipe.

## MBridge checkpoint structure

An MBridge checkpoint is a directory containing one or more iteration
subdirectories, plus metadata files at the top level:

```
evo2_7b_mbridge/
├── latest_checkpointed_iteration.txt
├── latest_train_state.pt
└── iter_0000001/
    ├── run_config.yaml
    ├── common.pt
    ├── train_state.pt
    ├── .metadata
    └── __0_*.distcp          # DCP (Distributed Checkpoint) shard files
```

| File / directory                    | Description                                                                                 |
| ----------------------------------- | ------------------------------------------------------------------------------------------- |
| `latest_checkpointed_iteration.txt` | Plain text file containing the latest iteration number (e.g. `1`)                           |
| `latest_train_state.pt`             | Top-level training state snapshot                                                           |
| `iter_NNNNNNN/`                     | Checkpoint data for iteration N                                                             |
| `iter_NNNNNNN/run_config.yaml`      | Full Megatron Bridge `ConfigContainer` used to create this checkpoint (model, optimizer, …) |
| `iter_NNNNNNN/common.pt`            | Shared metadata used by PyTorch Distributed Checkpoint (DCP)                                |
| `iter_NNNNNNN/train_state.pt`       | Training state (optimizer moments, scheduler, iteration counter)                            |
| `iter_NNNNNNN/.metadata`            | DCP planner metadata describing how weights are sharded                                     |
| `iter_NNNNNNN/__0_*.distcp`         | DCP shard files containing the model weights                                                |

When a tool expects `--mbridge-ckpt-dir`, point it at the **top-level**
directory (e.g. `evo2_7b_mbridge/`). When a tool expects an iteration
directory (e.g. for export), point it at `evo2_7b_mbridge/iter_0000001/`.

## CLI tools

| Command                           | Description                                           |
| --------------------------------- | ----------------------------------------------------- |
| `evo2_convert_nemo2_to_mbridge`   | Convert a NeMo2 checkpoint to MBridge format          |
| `evo2_convert_savanna_to_mbridge` | Convert a Savanna checkpoint to MBridge format        |
| `evo2_export_mbridge_to_vortex`   | Export an MBridge checkpoint to ARC Vortex `.pt` file |

Run any tool with `--help` for full usage details.

### Converting NeMo2 to MBridge

```bash
evo2_convert_nemo2_to_mbridge \
  --nemo2-ckpt-dir /path/to/nemo2/checkpoint \
  --mbridge-ckpt-dir evo2_1b_mbridge \
  --model-size evo2_1b_base \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length 8192 \
  --mixed-precision-recipe bf16_mixed
```

### Converting Savanna to MBridge

Note that `--seq-length` and `--mixed-precision-recipe` are written into the
resulting MBridge config saved in the checkpoint and act as defaults for
future inference and training runs. The `--seq-length` should match the
training sequence length, and `--mixed-precision-recipe` should ideally
reflect how you generally want the model to run in the future.

When converting Evo2 models from ARC to MBridge, note that you need to
convert from the Savanna format, not the vortex/inference format. For
example, rather than `arcinstitute/evo2_7b` use
`arcinstitute/savanna_evo2_7b`. This is because the vortex checkpoints
(the ones used in the evo2 GitHub repo) are missing information that is
required for training. The Savanna checkpoints have all of the weights
necessary for training or inference.

```bash
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path arcinstitute/savanna_evo2_7b \
  --mbridge-ckpt-dir evo2_7b_mbridge \
  --model-size evo2_7b \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512 \
  --seq-length 1048576 \
  --mixed-precision-recipe bf16_mixed
```

The `--savanna-ckpt-path` flag accepts a HuggingFace repo ID
(e.g. `arcinstitute/savanna_evo2_1b_base`) or a local `.pt` file path.

### Exporting MBridge to Vortex

This is how you can convert your checkpoints for use in the [evo2 repo](https://github.com/ArcInstitute/evo2).

```bash
evo2_export_mbridge_to_vortex \
  --mbridge-ckpt-dir evo2_7b_mbridge/iter_0000001 \
  --output-path evo2_7b_vortex.pt \
  --model-size evo2_7b
```

### Common options

- `--model-size` — model key such as `evo2_1b_base`, `evo2_7b`, `evo2_40b`, etc.
- `--no-te` — disable Transformer Engine fused layernorm key mapping.
- `--verbose` / `-v` — enable debug logging.

## Removing optimizer state from a checkpoint

After training, MBridge checkpoints include optimizer state (moments,
scheduler, etc.) which can significantly increase checkpoint size. The
`evo2_remove_optimizer.py` utility strips this state, producing a smaller
checkpoint suitable for distribution or inference. Note that this utility
currently needs updating to work with Megatron Bridge checkpoints
(see the `FIXME` in the source).

## Savanna training checkpoint utilities

The following scripts are included for historical and documentation
purposes. They were used during the original Evo2 training at ARC to
prepare Savanna training checkpoints into a release-ready format that
can then be converted to MBridge using `evo2_convert_savanna_to_mbridge`.

### Converting ZeRO-3 to ZeRO-1

`convert_zero3_to_zero1.py` converts DeepSpeed ZeRO-3 checkpoints into
ZeRO-1 checkpoints:

```bash
python convert_zero3_to_zero1.py <INPUT_DIR> <OUTPUT_DIR> \
  --overwrite --mp_size <MODEL_PARALLEL_SIZE>
```

ZeRO-3 checkpoints have the following structure:

```
global_step1/
├── bf16_zero_pp_rank_*_mp_rank_*_optim_states.pt
├── configs/
│   └── *.yml
└── zero_pp_rank_*_mp_rank_*_model_states.pt
```

### Converting ZeRO-1 MP{N} to ZeRO-1 MP1

`convert_checkpoint_model_parallel_evo2.py` re-shards ZeRO-1 checkpoints
to a different level of model tensor parallelism (typically MP1 for
release):

```bash
python convert_checkpoint_model_parallel_evo2.py \
  --input-checkpoint-dir /path/to/checkpoint/global_step1000 \
  --output-checkpoint-dir /path/to/output/global_step1000 \
  --output-model-parallelism 1
```

ZeRO-1 checkpoints have the following structure:

```
global_step199400/
└── mp_rank_*_model_states.pt
```

The resulting un-sharded (MP1) ZeRO-1 checkpoint is the Savanna format
accepted by `evo2_convert_savanna_to_mbridge --savanna-ckpt-path`.
