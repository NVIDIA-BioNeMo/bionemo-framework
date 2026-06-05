# Nemotron SAE Recipe

Train and analyze sparse autoencoders on the [NVIDIA Nemotron-3-Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) language model, trained on a large corpus of web text. The pipeline extracts residual-stream activations from a target layer over [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) text, trains a TopK SAE, and evaluates reconstruction quality and loss recovered against the language model.

## Pipeline

The workflow is three decoupled steps (one script each), chained by `scripts/nemotron_30b.sh`, plus an optional dashboard step for interactive feature exploration:

```
extract.py  ->  train.py  ->  eval.py        dashboard.py -> atlas/
 (cache)        (checkpoint)   (metrics)       (parquet+json)  (web UI)
```

**`extract.py`** runs Nemotron-3-Nano over FineWeb documents and writes per-token hidden states from a target layer to a sharded Parquet activation store. This is the expensive step (it loads the 30B model), so it is run **once** — you can then sweep many SAE sizes against the same cache. **`train.py`** streams that cache and fits a TopK SAE (4x expansion, top-32 sparsity by default — see [Scaling](#scaling)), saving a checkpoint. **`eval.py`** reloads the checkpoint and reports reconstruction error, sparsity, dead latents, and cross-entropy *loss recovered* (model logits with vs. without the SAE spliced into the residual stream). **`dashboard.py`** (optional) reloads the checkpoint, runs text through the SAE, and exports the data for the [interactive feature atlas](#dashboard).

> The 30B model does not fit on one GPU, so every step runs as a **single process** with `device_map="auto"` (model sharded across all visible GPUs). Do **not** use `torchrun`. SAE training itself is light (2688-dim).

Prefer to skip the disk round-trip? [Streaming mode](#streaming-mode-no-disk) fuses extract+train into one pass with no activations on disk.

## Model notes

Nemotron-3-Nano-30B-A3B is a hybrid **Mamba-2 / MoE / GQA** model (52 layers, ~2688 hidden dim, loaded in bf16). A few consequences for this recipe:

- The model is loaded via `AutoModelForCausalLM` with `trust_remote_code=True` and `device_map="auto"` (multi-GPU sharding) — by `extract.py` for activations and again by `eval.py` for loss recovered. `train.py` never loads it (it streams the cache).
- Mamba-2 layers require CUDA-only kernels. Install them on your GPU server: `pip install mamba-ssm causal-conv1d`.
- Layer indexing follows the `output_hidden_states` convention: `layer=L` means the output of transformer block `L` (i.e. `hidden_states[L+1]`). Extraction and loss-recovered eval use the same convention, so the SAE is evaluated on exactly the activations it was trained on. The default `layer=39` is ~3/4 depth.

## Prerequisites

Install dependencies from the workspace root:

```bash
# From the sparse_autoencoders workspace root (UV workspace)
uv sync

# Mamba-2 kernels (GPU server only)
pip install mamba-ssm causal-conv1d
```

FineWeb text is streamed automatically during extraction — no manual download required.

## Quick Start

```bash
cd recipes/nemotron

# Full workflow (extract -> train -> eval), 8x baseline. Edit vars at the top to scale.
bash scripts/nemotron_30b.sh
```

Or run the three steps by hand:

```bash
CACHE=.cache/activations/nemotron_l39

# 1. Extract activations once (idempotent — skips if the cache exists)
python scripts/extract.py activations.cache_dir=$CACHE activations.layer=39

# 2. Train an SAE from the cache (sweep sizes cheaply against the same cache)
python scripts/train.py +experiments=topk_k32_8x \
    activations.cache_dir=$CACHE checkpoint.dir=outputs/k32_8x/checkpoints

# 3. Evaluate the trained checkpoint
python scripts/eval.py checkpoint.dir=outputs/k32_8x/checkpoints
```

**Smoke test** (tiny: 2x expansion, 100 docs, 1 epoch; `config_debug` sets its own `cache_dir`):

```bash
python scripts/extract.py --config-name config_debug
python scripts/train.py  --config-name config_debug checkpoint.dir=outputs/debug/checkpoints
python scripts/eval.py   --config-name config_debug checkpoint.dir=outputs/debug/checkpoints
```

## Configuration

Configs are composed with [Hydra](https://hydra.cc/). The main `config.yaml` pulls in three groups — `model/`, `data/`, `training/` — plus top-level `activations`, `eval`, `wandb`, and `checkpoint` blocks. Override anything from the command line:

```bash
# Extract from a different layer
python scripts/extract.py activations.layer=26 activations.cache_dir=.cache/activations/nemotron_l26

# Train with a different LR / more epochs, W&B off
python scripts/train.py activations.cache_dir=.cache/activations/nemotron_l26 \
    training.lr=1e-4 training.n_epochs=20 wandb.enabled=false
```

> `activations.cache_dir` is the handoff between steps: `extract.py` writes it, `train.py` streams it, and `train.py`/`eval.py` validate that the cache's `model_name` and `layer` match the requested config.

## Streaming mode (no disk)

By default the workflow extracts activations to disk, then trains from the cache. As an alternative, **streaming mode** fuses extraction and training into one pass: a background producer thread runs Nemotron over FineWeb and feeds activations through a bounded in-memory queue straight into the SAE `Trainer` — **nothing but checkpoints touches disk**, and host memory is capped by the queue size (the producer blocks when the queue is full). The machinery lives in the `sae` package (`sae.streaming`) and is reusable by any recipe.

It is **off by default**; enable it with `streaming.enabled=true` (no `extract.py`, no `cache_dir` needed):

```bash
python scripts/train.py +experiments=topk_k32_8x \
    streaming.enabled=true \
    checkpoint.dir=outputs/k32_8x/checkpoints
```

| Setting                         | Default | Meaning                                                                          |
| ------------------------------- | ------- | -------------------------------------------------------------------------------- |
| `streaming.enabled`             | `false` | Master flag. When true, train.py extracts on the fly instead of reading a cache. |
| `streaming.queue_size`          | `8`     | Max activation chunks buffered (memory cap / backpressure).                      |
| `streaming.shuffle_buffer_size` | `0`     | `0` = producer order; `>0` = buffer-shuffle this many tokens.                    |
| `streaming.drop_last`           | `false` | Drop the final partial batch.                                                    |

Trade-offs: streaming avoids the disk round-trip and is ideal for a one-off run, but it **re-extracts every epoch** (the producer reruns each epoch) and only supports approximate (buffer) shuffling — so for multi-epoch sweeps over the same activations, the cached `extract.py` → `train.py` path is cheaper. Also, a stream has no fixed length, so non-constant LR schedules need `training.lr_decay_steps` set explicitly. On a single shared GPU the model forward and SAE step still serialize on the CUDA stream; the queue overlaps host-side work (tokenization, copies) and bounds memory — the main win is **no disk**, not extra GPU parallelism.

Key knobs:

| Setting                               | Default         | Meaning                                                                                    |
| ------------------------------------- | --------------- | ------------------------------------------------------------------------------------------ |
| `activations.layer`                   | `39`            | Transformer block whose output is extracted (~3/4 depth).                                  |
| `activations.batch_size`              | `4`             | Forward-pass batch size (keep small for the 30B model).                                    |
| `activations.max_length`              | `2048`          | Token truncation length.                                                                   |
| `activations.cache_dir`               | `null`          | Activation store dir — written by `extract.py`, streamed by `train.py` (required by both). |
| `model.expansion_factor`              | `4`             | SAE hidden dim = `expansion_factor * input_dim`.                                           |
| `model.top_k`                         | `32`            | Active latents per token (TopK sparsity).                                                  |
| `data.max_samples`                    | `10000`         | Number of FineWeb docs to use.                                                             |
| `training.max_steps`                  | `null`          | Stop after N optimizer steps (overrides `n_epochs` duration; loops the data as needed).    |
| `checkpoint.dir` / `checkpoint.steps` | `null` / `null` | Set `dir` to save; `steps=null` saves only `checkpoint_final.pt` at the end.               |

## Scaling

The defaults are intentionally **conservative** so the pipeline runs quickly and cheaply while you validate it. To scale up, increase `expansion_factor` (dictionary size) and/or `top_k` (active latents), and feed more data (`data.max_samples`). Caching activations (`activations.cache_dir`) is strongly recommended before scaling — extraction from a 30B model is the expensive step, and streaming a cache lets you sweep SAE sizes without re-running the model.

A ready-made scaling ladder lives in `configs/experiments/`:

| Experiment     | Expansion | top_k | Notes                              |
| -------------- | --------- | ----- | ---------------------------------- |
| `topk_k32_8x`  | 8x        | 32    | First step up from the 4x default. |
| `topk_k32_16x` | 16x       | 32    | Larger dictionary, same sparsity.  |
| `topk_k32_32x` | 32x       | 32    | Production-scale dictionary.       |

```bash
python scripts/train.py +experiments=topk_k32_16x
```

When scaling the dictionary, consider enabling `model.auxk` (auxiliary loss to revive dead latents) and `training.lr_scale_with_latents=true` (scales LR by `1/sqrt(hidden_dim/2048)` per Gao et al.).

## Dashboard

An interactive web dashboard (`atlas/`) for exploring the trained features: a UMAP of the
decoder directions with crossfiltering, plus per-feature cards showing decoder logits (top
promoted/suppressed tokens) and **top activating text examples with per-token highlighting**.
It is a port of the `codonfm/codon_dashboard` adapted from codons to tokenized text.

Generate the data (loads the 30B model; needs the `viz` extra — `pip install -e '.[viz]'`):

```bash
python scripts/dashboard.py \
    --checkpoint outputs/k32_8x/checkpoints/checkpoint_final.pt \
    --layer 39 --num-texts 1000 --max-length 256 \
    --output-dir outputs/k32_8x/dashboard
```

This writes `features_atlas.parquet`, `feature_metadata.parquet`, `feature_examples.parquet`
(per-token activations), and `vocab_logits.json` (decoder logits for the live features).

Then serve it (copies the data into `atlas/public/`, filters dead latents, runs Vite):

```bash
python scripts/launch_dashboard.py --data-dir outputs/k32_8x/dashboard
# opens http://localhost:5177
```

See [`atlas/README.md`](atlas/README.md) for the data format and manual setup.

## Layout

```
recipes/nemotron/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── extract.py               # Step 1: Nemotron activations -> cache
│   ├── train.py                 # Step 2: cache -> trained SAE checkpoint
│   ├── eval.py                  # Step 3: checkpoint -> metrics
│   ├── dashboard.py             # Optional: checkpoint -> dashboard data (parquet+json)
│   ├── launch_dashboard.py      # Optional: serve the atlas/ dashboard locally
│   └── nemotron_30b.sh          # Chains all three steps
├── atlas/                       # Interactive feature dashboard (React + Vite)
├── configs/
│   ├── config.yaml              # Defaults (conservative)
│   ├── config_debug.yaml        # Tiny smoke-test run
│   ├── model/topk.yaml
│   ├── data/fineweb.yaml
│   ├── training/default.yaml
│   ├── streaming/default.yaml   # Producer-consumer streaming (off by default)
│   └── experiments/             # Scaling ladder
└── src/nemotron_sae/
    ├── models/nemotron.py       # Activation extraction wrapper
    ├── data/fineweb.py          # FineWeb streaming loader
    ├── eval/loss_recovered.py   # Causal-LM loss recovered
    └── interp/format.py         # Auto-interp text formatting
```
