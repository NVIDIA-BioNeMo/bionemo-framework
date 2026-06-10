---
name: bionemo-sae-recipe
description: Build a new sparse-autoencoder recipe under bionemo-recipes/interpretability/sparse_autoencoders/recipes/ for a biological foundation model (e.g. evo2, nemotron, geneformer) — extract activations, train an SAE, and evaluate it. Trigger when the user asks to "add SAE for <model>", "build a new SAE recipe", or "run an SAE on <model>".
---

# Build a new SAE recipe in bionemo-framework

## The pattern

Every SAE recipe in `bionemo-recipes/interpretability/sparse_autoencoders/recipes/` decomposes into the same stages, separated by a universal contract:

```
extractor (model-specific) → ActivationStore parquet shards → train.py (universal) → eval (sae.eval, universal)
                                       ↑ contract
```

- **Extractor** runs the model forward and **streams** layer-L activations *directly* into an `ActivationStore` — no intermediate `.pt` files. The clean pattern (see `evo2/scripts/extract.py`): reuse the model's existing `predict_<model>` CLI but **monkeypatch its per-batch writer** with one that appends `hidden[mask]` to `sae.activation_store.ActivationStore`. Model-specific (~150 lines).
- **ActivationStore** (`sae/src/sae/activation_store.py`) is the universal on-disk format: a directory of `shard_{NNNNN}.parquet` + `metadata.json` (`{model_name, layer, hidden_dim, n_samples, n_shards, shard_size, n_sequences}`).
- **train.py** loads via `sae.activation_store.load_activations(cache_dir)` and trains a TopK/ReLU SAE — **near-identical across recipes, but not a blind verbatim copy.** It must wire the opt-in training flags (`--aggregate-loss` / `--dead-count-global` / `--mix-shards` / `--presample-shards`). Start from a **current** recipe's `train.py` (codonfm/evo2 — they already wire them), then change only the docstring + `--wandb-project` default. **Copying an older train.py silently drops those flags → the losing config** (this is exactly how a "reproduce the winner" run quietly turns into a baseline run). Uses `--model-path` only for a cache-validation warning. (The copy-paste is a known smell; the intended end-state is a single shared train-CLI in `sae`.)
- **eval** (`sae.eval`, universal): `reconstruction` (variance explained), `dead_latents` (%), `loss_recovered` (CE fidelity), and `probing` (per-feature AUROC / linear probes / domain-F1 over a labeled `ActivationBuffer`). Probing scoring is **CPU-only** — it reads saved buffers, no model.

## When this applies

Bringing up an SAE on a new biological foundation model — Evo2, ESM2, CodonFM, Nemotron, Geneformer, etc. A checkpoint (HF or local) is in hand. Scope is the full **extract → train → eval** pipeline. Per-model you write a thin **extractor** (and, for interpretability, **labelers**); everything downstream is shared.

## Steps

### 1. Reconnaissance (read, don't write)

- Templates: `recipes/esm2/` (HF `AutoModel` path), `recipes/codonfm/` (custom checkpoint), `recipes/evo2/` (streaming reuse of a `predict_<model>` CLI). Pick the closest.
- Find the model's inference path in `bionemo-recipes/recipes/<model>_*/`. If it has a `predict_<model>` CLI, reuse it (streaming); else write `extract.py` modeled on `esm2/`.
- Identify hidden_dim, layer count, **trained context length** (critical — see gotchas).

### 2. Build the upstream env (if needed)

Recipes under `bionemo-recipes/recipes/<model>_*/` have `.ci_build.sh` that makes a `--system-site-packages` `.venv` — **assumes the NVIDIA pytorch container** with TransformerEngine preinstalled. Verify first:

```bash
ls /usr/local/lib/python*/dist-packages/transformer_engine 2>/dev/null && echo "OK to build"
```

### 3. Scaffold the recipe dir

```
recipes/<model>/
├── README.md
├── pyproject.toml          # deps: sae, torch, numpy, pyarrow ; [tool.uv.sources] sae = { workspace = true }
└── scripts/
    ├── <model>.sh          # orchestrator: chunk → stream-extract → train
    ├── extract.py          # STREAMING: wraps predict_<model>, writes ActivationStore directly (NO .pt)
    └── train.py            # near-verbatim from a CURRENT recipe (codonfm/evo2): MUST wire the opt-in flags; edit only docstring + wandb default
```

### 4. The streaming extractor

Reuse the upstream forward; swap only the writer:

```python
from bionemo.<model>.run import predict as predict_mod
predict_mod._write_predictions_batch = _store_writer   # appends hidden[pad_mask] to ActivationStore
sys.argv = [sys.argv[0], *forwarded_predict_flags]
predict_mod.main()
```

No `.pt`, ~half the disk, no separate conversion pass. Under DDP each rank writes its own tmp store; rank 0 merges at the end via a **file-based wait** (poll for sibling `metadata.json`) — **not** `dist.barrier()`, because `predict.main()` tears down the process group before the finalize hook runs.

### 5. Launch the training

The orchestrator (`<model>.sh`) chains chunk → extract → train. Launch with `torchrun`, `--dp-size` = #GPUs. **Always smoke first** (20–100 sequences → confirm loss drops), then the full run.

```bash
unset WANDB_API_KEY                  # a leaked key in the shared env overrides ~/.netrc — you'd log as someone else
export WANDB_ENTITY=<your-entity>    # accounts with no default entity fail wandb.init otherwise

torchrun --nproc_per_node 8 scripts/train.py \
  --cache-dir <parquet-dir> --model-path <ckpt> --layer L \
  --model-type topk --expansion-factor 16 --top-k 128 --normalize-input \
  --auxk 2048 --auxk-coef 0.03125 --dead-tokens-threshold 10000000 \
  --init-pre-bias --presample-shards 8 --mix-shards 10 \
  --aggregate-loss --dead-count-global \
  --n-epochs 1 --batch-size 1024 \
  --lr 1e-4 --lr-schedule cosine --lr-min 1e-5 --warmup-steps 1000 --max-grad-norm 1.0 \
  --dp-size 8 --wandb --wandb-project <proj>
```

For a **sweep**, run one config at a time on a fixed GPU group (sequential), not many in parallel — parallel runs contend on the same parquet cache I/O. Give each `torchrun` a distinct `--master-port`.

### 6. Cache guards in the orchestrator

Each long step needs an idempotency check on a sentinel the step itself produces:

```bash
[[ -f "${PARQUET_DIR}/metadata.json" ]] || torchrun ... scripts/extract.py ...   # finalize() writes metadata.json last
```

**Caveat:** guards check existence, not provenance — `rm -rf` the output dir when the input FASTA / model / layer changes.

## Known-good training config (and why)

These defaults reproduced the best Evo2-7B / layer-26 SAE (~21% dead, ~0.10 FVU). All are **opt-in** in the `sae` package (defaults reproduce older behavior):

| flag                   | why it matters                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--normalize-input`    | the single biggest dead-latent lever (∼80% → ∼20% dead)                                                                                           |
| `--aggregate-loss`     | batch-level FVU/AuxK ratio instead of per-token (per-token starves rare high-variance tokens → their latents die)                                 |
| `--dead-count-global`  | counts dead-latent inactivity in **total** tokens (×world_size); the per-rank default fires AuxK revival `world_size`× too late under DDP         |
| `--mix-shards 10`      | shuffles + blends shards; corpus/kingdom-ordered caches otherwise give a visible FVU cliff                                                        |
| `--presample-shards 8` | geometric-median pre-bias over 8 shards, not shard-0 alone — a single-shard sample is corpus-order-biased and **measurably worsens dead latents** |

## Known gotchas (these cost real debug time)

### Training dynamics (learned the hard way)

1. **Never truncate steps without fixing the LR horizon.** `--n-epochs 1` lets cosine decay over the *whole* epoch. Capping steps (e.g. a `--max-steps`/short `--lr-decay-steps`) shrinks the cosine horizon, so LR collapses to `lr_min` early and training is much worse — looks like a model/code regression but it's the schedule.
2. **`--dead-count-global` is a no-op at dp-size 1** (world_size=1). It only does anything under DDP. And it must actually be **passed** — encoding "dcg=true" only in a run *name* while omitting the flag silently runs the per-rank default (a real sweep bug).
3. **Pre-bias from shard-0 only is biased.** On a corpus-ordered cache (e.g. all-prok-then-all-euk), a single-shard geometric-median init mis-centers `pre_bias` toward one kingdom and worsens dead latents. Use `--presample-shards N>1`.

### wandb

4. **`unset WANDB_API_KEY` before launching** — a leaked key in the shared env overrides `~/.netrc`, so your runs log under someone else's account. Then set `WANDB_ENTITY` if your account has no default entity (else `wandb.init` fails / lands in the wrong entity).

### Container / env

5. `.ci_build.sh` assumes system-site-packages TransformerEngine — verify before building (step 2).
6. `huggingface-cli` is deprecated → use `hf` (same args). HF README dir names are unreliable (OpenGenome2's `jsonl/` is really `json/`) — verify the tree and that the downloaded file count is nonzero.

### Checkpoint loading

7. **`weights_only=True` (torch 2.6 default) breaks legacy checkpoints with numpy arrays** — buried in stderr, exit 0, empty output dir. `UnpicklingError: Unsupported global: numpy.core.multiarray._reconstruct`. Patch the upstream `torch.load(...)` to `weights_only=False` if the source is trusted. (For Evo2, the recipe assumes an MBridge checkpoint — conversion from savanna/nemo2 is a prerequisite, not recipe code.)

### Model architecture / extraction (general principle → Evo2 example)

These are **general principles**; the numbers are Evo2 examples — **measure them for your model** (see "Verify the perf claims" below), don't copy the constants.

08. **Long sequences can blow up memory super-linearly on conv/FFT architectures → chunk inputs to the model's trained context before extraction.** *Evo2 example:* Hyena's fftconv OOMs even at micro-batch=1 (intermediates scale super-linearly); chunk to 1B → 8192 bp, 7B → context-extended (check release), 40B → 1M. Don't rely on the inference tool to truncate.
09. **Check your predict CLI's input constraints (compression/format).** *Evo2 example:* `predict_evo2` takes uncompressed FASTA only (`<(zcat ...)` fails); but if your chunker already reads `.gz` → writes plain `.fasta`, no separate gunzip is needed.
10. **micro-batch=1 is rarely optimal — once inputs are short/uniform, raise it.** *Evo2 example:* chunking dropped memory ~10× and gave ~17× per-batch speedup on Evo2 1B, so `--micro-batch-size` could be raised well past 1.

**Verify the perf claims (don't trust the constants):** a few-minute single-GPU micro-benchmark —

- **micro-batch sweep:** fix a chunked FASTA, run the extractor at `--micro-batch-size ∈ {1,4,8,16,32}`, log peak GPU mem (`torch.cuda.max_memory_allocated`) + throughput (tokens/s over fixed N). Find the largest mbs that fits + the throughput curve.
- **seq-length sweep** (for #8): mbs=1, L ∈ {1k,8k,16k,32k}, log peak mem → see the blowup / OOM point for *your* architecture.

### Output format

11. `predict_evo2 --embedding-layer N` yields `{hidden_embeddings:[B,S,H], pad_mask:[B,S], seq_idx:[B], tokens:[B,S], batch_idx:int}`. `pad_mask` is a **loss mask** (1=valid), not an HF attention mask. The streaming `_store_writer` appends `hidden_embeddings[pad_mask.bool()]`.

## Evaluating the SAE

After training, run `sae.eval` on a **held-out** cache (same distribution, disjoint instances):

- `reconstruction` → variance explained; `dead_latents` → dead %; `loss_recovered` → CE fidelity (substitute the SAE recon at the layer-L hook).
- For interpretability, build a labeled `ActivationBuffer` (per-token feature codes + concept labels + optional dense-residual twin) and run `sae.eval.probing` — per-feature AUROC, winner's-curse-corrected best-single, SAE-vs-dense probes, domain-F1. Labelers are **per-domain** (DNA / protein / codon); the scoring is shared.

## Verifying the recipe works (fastest → most confident)

1. **Mechanical** — pipeline runs end-to-end, `checkpoint_final.pt` exists. Smoke on 20–100 sequences (minutes).
2. **Numerical** — `train.py` log shows loss ↓, FVU < 1, dead-% trending toward ~20% (not stuck at ~80%). If dead-% is stuck high, check normalize-input / presample / the LR horizon (gotcha 1).
3. **Shape sanity** — `torch.load(checkpoint_final.pt)`: encoder `[hidden_dim → expansion·hidden_dim]`, decoder the transpose.

## Reference recipes

| Recipe     | Extract path                                                                                                   | Mirror it when                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `esm2/`    | `extract.py` → HF `AutoModel.from_pretrained` + `output_hidden_states`                                         | new model is HF-native with a clean `AutoModel`                   |
| `codonfm/` | `extract.py` → custom inference class                                                                          | new model has its own checkpoint + forward code                   |
| `evo2/`    | **streaming** `extract.py` — wraps `predict_evo2`, monkeypatches its writer to an `ActivationStore` (no `.pt`) | upstream already has a `predict_<model>` CLI; reuse it and stream |

All share a near-identical `train.py` (current copies wire the opt-in flags) and the `ActivationStore` parquet contract — folding the duplicated train-CLI into a shared `sae` entrypoint is a planned follow-up.
