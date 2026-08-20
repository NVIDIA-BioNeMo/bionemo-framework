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
| `evo2_convert_vortex_to_mbridge`  | Convert an ARC Vortex `.pt` checkpoint to MBridge     |
| `evo2_export_mbridge_to_vortex`   | Export an MBridge checkpoint to ARC Vortex `.pt` file |
| `evo2_analyze_inverse_prior`      | Analyze Hyena filter priors used by Vortex inversion  |

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

When converting general Evo2 models from ARC to MBridge for continued
training, prefer the Savanna format over the Vortex/inference format. For
example, rather than `arcinstitute/evo2_7b` use
`arcinstitute/savanna_evo2_7b`. The Savanna checkpoints preserve all training
weights directly. The Vortex converter below is for released Vortex-only
checkpoints, such as Microviridae, where the missing MBridge parameters must be
reconstructed from an inference checkpoint.

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

### Converting Vortex to MBridge

Vortex is ARC Institute's inference format for Evo2 Hyena models, used by the
public Evo2 repository. The Vortex checkpoints omit some training-time state,
so `vortex_to_mbridge.py` reconstructs the MBridge parameterization that can be
loaded by Megatron Bridge. Vortex runtime buffers (`filter.t` and
`rotary_emb.inv_freq`) and Transformer Engine extra state are intentionally
excluded because they are not model parameters and each runtime regenerates its
own copies when loading a checkpoint. Stored rotary frequencies are validated
against the target model provider before they are omitted.

```bash
evo2_convert_vortex_to_mbridge \
  --vortex-ckpt-path /path/to/evo2_7b_microviridae.pt \
  --mbridge-ckpt-dir evo2_7b_microviridae_mbridge \
  --model-size evo2_7b_base \
  --seq-length 10240 \
  --tokenizer-path tokenizers/nucleotide_fast_tokenizer_512
```

The converter reuses the MBridge checkpoint packaging path from
`savanna_to_mbridge.py`. It should instantiate the target Evo2 provider with
the same model size, sequence length, dtype, initialization settings, and RNG
seed used for training so any initialization anchors are reproducible.

### Vortex-to-MBridge validation

The optional long-running round-trip test downloads the smaller public 1B Vortex
checkpoint from `arcinstitute/evo2_1b_base`, converts it to MBridge state-dict
form, converts back to Vortex, and asserts exact key and value equality for all
learned model tensors. Descriptor-derived rotary frequencies are compared within
their source precision. Runtime-generated `filter.t` caches and Transformer
Engine extra state are excluded from the comparison.

Conversion fails on missing required learned tensors, invalid core tensor
geometry, or unexpected non-runtime entries. Ordinary parameters are normalized
to the target provider dtype, while Hyena filter parameters retain their required
FP32 representation.

```bash
LONG_TESTS=1 EVO2_CHECKPOINT_CACHE_DIR=/tmp/evo2-checkpoints \
python -m pytest \
  recipes/evo2_megatron/tests/bionemo/evo2/test_vortex_to_mbridge.py \
  -k 1b_base_checkpoint_weight_roundtrip -q
```

The Microviridae bootstrap path has also been validated with
`evo-design/evo-2-7b-8k-microviridae/evo2_7b_microviridae.pt`, a 13 GB Vortex
checkpoint from a 10,240-token, 12,000-iteration fine-tune of
`arcinstitute/evo2_7b_base`. Cached validation confirmed exact equality for all
model tensors after an MBridge-to-Vortex export; Vortex runtime state was
excluded because it is regenerated on load.

### Ambiguous inverse mappings

Some Vortex fields are products or projections of multiple MBridge parameters,
so the reverse conversion needs a principled initialization projection.

- Long Hyena filters: MBridge parameters `p` and `gamma` map to Vortex
  `log_poles = -exp(p) * exp(gamma)`. Reverse conversion searches nearby fp32
  values for an exact round-trip pair and prefers a balanced split,
  `p ~= gamma ~= 0.5 * log(-log_poles)`. This is data-driven: prior analysis
  on the original BioNeMo 1B and 7B checkpoints shows trained `p` and `gamma`
  both move close to zero and track each other, rather than staying close to
  the initial `gamma = log(U(0.01, 0.1))` support.
- Medium explicit filters: MBridge `h` and `decay` map to Vortex `filter.h`
  through a product after truncation. The current exact-reproduction converter
  chooses `decay = 1` and `h = filter.h` so exporting back to Vortex is bitwise
  identical. Use `evo2_analyze_inverse_prior` on original 1B/7B checkpoints to
  decide whether a future training-oriented inverse should instead project
  toward the trained decay distribution.
- Non-ambiguous mappings, such as MLP `w1/w2` split from concatenated
  `linear_fc1.weight`, attention projections, RMSNorm scales, and short-conv
  reshaping, directly invert `mbridge_to_vortex.py`.

### Analyzing inverse priors

`evo2_analyze_inverse_prior` loads only Evo2 Hyena filter tensors from a DCP
checkpoint and emits compact JSON stats for the priors used by the ambiguous
Vortex-to-MBridge inverse. It accepts a top-level MBridge checkpoint directory,
an iteration directory, a `weights/` DCP directory, or a NeMo2 DCP extraction.

```bash
evo2_analyze_inverse_prior \
  --checkpoint-dir $(download_bionemo_data evo2/1b-8k:1.0) \
  --output-json /tmp/evo2-prior-analysis/evo2_1b_8k_prior.json

evo2_analyze_inverse_prior \
  --checkpoint-dir $(download_bionemo_data evo2/7b-8k:1.0) \
  --output-json /tmp/evo2-prior-analysis/evo2_7b_8k_prior.json
```

Both reference reports showed less than 0.01 percent of `gamma` values inside
the original log-init support. The 7B medians were approximately `p=-0.048`
and `gamma=-0.049`; the 1B medians were approximately `p=-0.144` and
`gamma=-0.186`. This supports the balanced inverse prior for continuing
training from a converted Vortex checkpoint.

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
checkpoint suitable for distribution or inference. Its historical default also
omits serialized model-object shards; pass `--preserve-model-object-state` when a
native MBridge consumer requires objects such as Transformer Engine `_extra_state`.

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
