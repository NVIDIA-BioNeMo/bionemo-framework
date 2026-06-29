---
name: bionemo-sae-recipe
description: Build a sparse-autoencoder (SAE) recipe for a biological foundation model (evo2, ESM2, geneformer) — extract layer activations, train an SAE, and evaluate it. Use when asked to add, build, or run an SAE for a model.
---

# Build a new SAE recipe in bionemo-framework

## The pattern

Every SAE recipe in `bionemo-recipes/interpretability/sparse_autoencoders/recipes/` decomposes into the same stages, separated by a universal contract:

```text
extractor (model-specific) → ActivationStore parquet shards → train.py (universal) → eval (sae.eval, universal)
                                       ↑ contract
```

- **Extractor** runs the model forward and **streams** layer-L activations *directly* into an `ActivationStore` — no intermediate `.pt` files. The clean pattern (see `evo2/scripts/extract.py`): reuse the model's existing `predict_<model>` CLI but **monkeypatch its per-batch writer** with one that appends `hidden[mask]` to `sae.activation_store.ActivationStore`. Model-specific (~150 lines).
- **ActivationStore** (`sae/src/sae/activation_store.py`) is the universal on-disk format: a directory of `shard_{NNNNN}.parquet` + `metadata.json` (`{model_name, layer, hidden_dim, n_samples, n_shards, shard_size, n_sequences}`).
- **train.py** loads via `sae.activation_store.load_activations(cache_dir)` and trains a TopK/ReLU SAE — **near-identical across recipes, but not a blind verbatim copy.** It must wire the opt-in training flags (`--aggregate-loss` / `--dead-count-global` / `--mix-shards` / `--presample-shards`). **On `main`, only `evo2/scripts/train.py` wires all four** — `codonfm` and `esm2` are `0/4` (precisely why this belongs in shared `sae`). So copy **evo2's**, then change only the docstring + `--wandb-project` default. **Copying an older train.py silently drops those flags → the losing config** (this is exactly how a "reproduce the winner" run quietly turns into a baseline run). Uses `--model-path` only for a cache-validation warning. (The copy-paste is a known smell; the intended end-state is a single shared train-CLI in `sae`.)
- **eval** (`sae.eval`, universal): `reconstruction` (variance explained), `dead_latents` (%), `loss_recovered` (CE fidelity), and `probing` (per-feature AUROC / linear probes / domain-F1 over a labeled `ActivationBuffer`). Probing scoring is **CPU-only** — it reads saved buffers, no model.

## Purpose

Bringing up an SAE on a new biological foundation model — Evo2, ESM2, CodonFM, Nemotron, Geneformer, etc. Scope is the full **extract → train → eval** pipeline. Per-model you write a thin **extractor** (and, for interpretability, **labelers**); everything downstream is shared.

## Prerequisites (check these first)

This skill assumes the model is **already integrated into bionemo** — i.e. there's a setup in `bionemo-recipes/recipes/` to build on. If one of these is false, that's upstream work, not part of the SAE recipe:

1. **The model has an inference path in the repo** — a `predict_<model>` CLI under `bionemo-recipes/recipes/<model>_*/` (Megatron-style, like Evo2), or it's HF-native (`AutoModel.from_pretrained`, like ESM2). If neither exists, you must add the forward pass first.
2. **You can pull hidden states at a chosen layer** — `[B, S, H]` via `--embedding-layer` (predict CLI) or `output_hidden_states` (HF).
3. **The checkpoint loads in the available env** — an **MBridge directory** for Megatron models (convert in Step 0); `.safetensors`/HF otherwise. The Megatron path needs the NVIDIA pytorch container + TransformerEngine.
4. **Known token↔position mapping** — to label/probe, each activation row must map back to a sequence position. Evo2 byte tokenizer = 1 char/token; CodonFM = 1 codon (3-mer)/token; ESM2 = 1 aa/token. Get this wrong and your labels are misaligned.
5. **Activations are float (fp16/fp32), not bf16** — Arrow/NumPy can't store bf16; cast before `ActivationStore.append`.
6. **Inputs are sequences you can chunk/feed** (FASTA/CSV), and you know the model's **trained context length**.

## Step 0 — get the model + data (and, for Megatron models, convert to MBridge)

Before any extraction you need a **checkpoint in the format the model's `predict`/forward expects** and a **sequence corpus**. This is upstream of the recipe — don't bake it into `extract.py`.

**Model checkpoint:**

```bash
# BioNeMo / Evo2 etc. live on NGC:
ngc registry model download-version "nvidia/clara/<model>:<ver>" --dest ./checkpoints
# HF-native models (ESM2, CodonFM/Encodon) on HuggingFace (use `hf`, not the deprecated huggingface-cli):
hf download <org/repo> --local-dir ./checkpoints/<model>
```

**Convert to MBridge (Megatron models — e.g. Evo2):** `predict_evo2`/Megatron loads an **MBridge checkpoint *directory*** (has `latest_checkpointed_iteration.txt` + sharded weights), **not** a raw HF/savanna file. Convert first; the result is the `--ckpt-dir` you hand the extractor:

```bash
evo2_convert_savanna_to_mbridge \
  --savanna-ckpt-path <hf-id-or-path> --mbridge-ckpt-dir <CKPT_DIR> \
  --model-size <evo2_Nb> --tokenizer-path <tokenizer>
# (or the nemo2 -> mbridge path if you have a nemo2 checkpoint)
```

- **Gotcha:** savanna conversion hits the torch-2.6 `weights_only=True` default → patch the converter's `torch.load(...)` to `weights_only=False` (trusted source); the failure is silent (exit 0, empty dir). See gotcha 7.
- **HF models (esm2/codonfm) skip MBridge entirely** — they load directly from the `.safetensors`/checkpoint.

**Data corpus:**

- Pull the sequence set (Evo2 → OpenGenome2; protein → UniRef/etc.). **Verify the download** — HF README dir names are unreliable (OpenGenome2's `jsonl/` is really `json/`); check the tree + a nonzero file count (`curl -s "https://huggingface.co/api/datasets/<repo>/tree/main" | python3 -m json.tool`).
- Decompress `.gz` if the predict CLI needs plain FASTA, and **chunk to the trained context** (gotcha 8) before extraction.
- Grab a small subset (a few thousand sequences) first to smoke-test the whole pipeline.

## Workflow

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

```text
recipes/<model>/
├── README.md
├── pyproject.toml          # deps: sae, torch, numpy, pyarrow ; [tool.uv.sources] sae = { workspace = true }
└── scripts/
    ├── <model>.sh          # orchestrator: chunk → stream-extract → train
    ├── extract.py          # STREAMING: wraps predict_<model>, writes ActivationStore directly (NO .pt)
    └── train.py            # near-verbatim from evo2/scripts/train.py (the ONLY recipe on main wiring all 4 opt-in flags); edit only docstring + wandb default
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

The orchestrator (`<model>.sh`) chains chunk → extract → train. Launch with `torchrun`, `--dp-size` = #GPUs. **Always smoke first** (20–100 sequences → confirm loss drops), then the full run. The flags below are general; the numeric values are the **Evo2-7B/L26** example — re-tune per model (see "Training config" below).

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

## Training config — which knobs to turn on (general) vs. their values (per-model)

Separate two things:

- **The flags are *available*, not mandatory.** All opt-in in `sae` (defaults = older behavior). Each fixes a specific failure mode we hit on **Evo2** — severe dead latents (`--normalize-input`, `--aggregate-loss`), a corpus/kingdom-ordered cache (`--mix-shards`, `--presample-shards`), and DDP dead-counting (`--dead-count-global`). They mattered a lot *there*. **They are not universally required: CodonFM trained a good SAE with none of them** (its `train.py` wires 0/4). So turn each on only if you actually hit the problem it fixes — don't cargo-cult them. *(The one place they're non-negotiable: **reproducing the Evo2 winner** — which is why copying an older, flag-less `train.py` into an Evo2 recipe silently gives the losing config.)*
- **The values are model-specific.** The numbers in the launch command (`--expansion-factor 16`, `--top-k 128`, `--auxk 2048`, `--mix-shards 10`, `--presample-shards 8`) are what reproduced the best **Evo2-7B / layer-26** SAE (~21% dead, ~0.10 FVU). **Re-tune per model:** expansion/top-k/auxk scale with `hidden_dim` and the sparsity you want; `--mix-shards`/`--presample-shards` only matter for a **corpus-ordered** cache — set both to `1` if your shards are already shuffled.

| flag                   | why it matters                                                                                                                                    |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--normalize-input`    | the single biggest dead-latent lever (∼80% → ∼20% dead)                                                                                           |
| `--aggregate-loss`     | batch-level FVU/AuxK ratio instead of per-token (per-token starves rare high-variance tokens → their latents die)                                 |
| `--dead-count-global`  | counts dead-latent inactivity in **total** tokens (×world_size); the per-rank default fires AuxK revival `world_size`× too late under DDP         |
| `--mix-shards 10`      | shuffles + blends shards; corpus/kingdom-ordered caches otherwise give a visible FVU cliff                                                        |
| `--presample-shards 8` | geometric-median pre-bias over 8 shards, not shard-0 alone — a single-shard sample is corpus-order-biased and **measurably worsens dead latents** |

## Troubleshooting (known gotchas)

Each entry is **Symptom → Cause → Fix**, with the literal error/output where there is one. Scan for your symptom first; these cost real debug time.

### Training dynamics (learned the hard way)

**Training looks like a model/code regression after you changed step counts**

- *Symptom:* FVU plateaus high and loss is much worse than a known-good run, with no code change that explains it.
- *Cause:* capping steps (`--max-steps`, or a short `--lr-decay-steps`) shrinks the cosine horizon, so LR collapses to `lr_min` early.
- *Fix:* never truncate steps without fixing the LR horizon — use `--n-epochs 1` so cosine decays over the *whole* epoch.

**Dead-latent % higher than expected under DDP; AuxK seems to revive latents too late**

- *Symptom:* dead-% stays elevated multi-GPU even though the config matches a good single-GPU run.
- *Cause:* `--dead-count-global` was not actually passed (e.g. "dcg=true" lives only in the run *name*), so AuxK counts per-rank and fires `world_size`× too late. It is also a no-op at dp-size 1 (world_size=1).
- *Fix:* pass `--dead-count-global` explicitly when running DDP, and confirm it's in the launch command, not just the run name.

**Dead latents worse than expected on a corpus-ordered cache**

- *Symptom:* high dead-% that `--normalize-input` alone doesn't fix, on shards written in corpus order (e.g. all-prok-then-all-euk).
- *Cause:* a single-shard (shard-0) geometric-median init mis-centers `pre_bias` toward whatever is first in the corpus.
- *Fix:* spread the pre-bias sample with `--presample-shards N>1` (and blend shards per batch with `--mix-shards N>1`).

### wandb

**Runs log under someone else's W&B account, or `wandb.init` fails**

- *Symptom:* your run appears in the wrong entity/account, or `wandb.init` raises about a missing default entity.
- *Cause:* a leaked `WANDB_API_KEY` in the shared env overrides `~/.netrc`; an account with no default entity can't `init` without one.
- *Fix:* `unset WANDB_API_KEY` before launching, then `export WANDB_ENTITY=<your-entity>`.

### Container / env

**`.ci_build.sh` build fails importing TransformerEngine**

- *Symptom:* `ModuleNotFoundError: No module named 'transformer_engine'` (or similar) when building/running the recipe venv.
- *Cause:* `.ci_build.sh` makes a `--system-site-packages` venv that *assumes* the NVIDIA PyTorch container has TE preinstalled.
- *Fix:* verify TE first — `ls /usr/local/lib/python*/dist-packages/transformer_engine` — and build inside the NVIDIA PyTorch container (step 2).

**Downloaded data is empty or in the "wrong" directory**

- *Symptom:* a `huggingface-cli` deprecation warning, or an expected dir (e.g. OpenGenome2 `jsonl/`) is missing / has zero files.
- *Cause:* `huggingface-cli` is deprecated, and HF README dir names are unreliable (OpenGenome2's `jsonl/` is really `json/`).
- *Fix:* use `hf` (same args); verify the tree and a nonzero file count (`curl -s "https://huggingface.co/api/datasets/<repo>/tree/main" | python3 -m json.tool`).

### Checkpoint loading

**Checkpoint conversion exits 0 but the output dir is empty**

- *Symptom:* buried in stderr: `UnpicklingError: Unsupported global: numpy.core.multiarray._reconstruct`; the process exits 0 with an empty output dir (silent failure).
- *Cause:* torch 2.6 defaults `torch.load(..., weights_only=True)`, which rejects legacy checkpoints that pickle numpy arrays.
- *Fix:* patch the upstream `torch.load(...)` to `weights_only=False` **if the source is trusted**. (For Evo2 the recipe assumes an MBridge checkpoint — savanna/nemo2 conversion is a prerequisite, not recipe code.)

### Model architecture / extraction (general principle → Evo2 example)

These are **general principles**; the numbers are Evo2 examples — **measure them for your model** (see "Verify the perf claims" below), don't copy the constants.

**CUDA OOM during extraction even at micro-batch 1**

- *Symptom:* `torch.cuda.OutOfMemoryError` on long sequences, even with `--micro-batch-size 1`, on conv/FFT architectures.
- *Cause:* intermediates scale super-linearly with sequence length (e.g. Hyena's fftconv).
- *Fix:* chunk inputs to the model's trained context before extraction (*Evo2 example:* 1B → 8192 bp, 7B → context-extended (check release), 40B → 1M). Don't rely on the inference tool to truncate.

**The predict CLI errors on your input file**

- *Symptom:* `predict_evo2` fails on a gzipped or process-substituted FASTA (`<(zcat ...)`).
- *Cause:* the CLI accepts uncompressed FASTA only.
- *Fix:* feed plain `.fasta` (if your chunker already reads `.gz` → writes plain `.fasta`, no separate gunzip is needed).

**Extraction throughput is low / the GPU is underused**

- *Symptom:* tokens/s far below what the GPU should sustain once inputs are short and uniform.
- *Cause:* `--micro-batch-size 1` is rarely optimal once inputs are chunked.
- *Fix:* raise the micro-batch and **measure** (the often-quoted ~10× memory / ~17× speedup on Evo2 1B is an unverified inherited number — measure your own, don't quote it).

**Verify the perf claims (don't trust the constants):** a few-minute single-GPU micro-benchmark —

- **micro-batch sweep:** fix a chunked FASTA, run the extractor at `--micro-batch-size ∈ {1,4,8,16,32}`, log peak GPU mem (`torch.cuda.max_memory_allocated`) + throughput (tokens/s over fixed N). Find the largest mbs that fits + the throughput curve.
- **seq-length sweep:** mbs=1, L ∈ {1k,8k,16k,32k}, log peak mem → see the blowup / OOM point for *your* architecture.

### Output format

**SAE trains on padding, or activations are misaligned to positions**

- *Symptom:* latents look degenerate / dead-% odd because padded positions leaked into the store.
- *Cause:* `predict_evo2 --embedding-layer N` yields `{hidden_embeddings:[B,S,H], pad_mask:[B,S], seq_idx:[B], tokens:[B,S], batch_idx:int}`, and `pad_mask` is a **loss mask** (1=valid), *not* an HF attention mask.
- *Fix:* the streaming `_store_writer` must append `hidden_embeddings[pad_mask.bool()]` (keep valid positions only).

## Evaluating the SAE

After training, run `sae.eval` on a **held-out** cache (same distribution, disjoint instances):

- `reconstruction` → variance explained; `dead_latents` → dead %; `loss_recovered` → CE fidelity (substitute the SAE recon at the layer-L hook).
- For interpretability, build a labeled `ActivationBuffer` (per-token feature codes + concept labels + optional dense-residual twin) and run `sae.eval.probing` — per-feature AUROC, winner's-curse-corrected best-single, SAE-vs-dense probes, domain-F1. Labelers are **per-domain** (DNA / protein / codon); the scoring is shared. **Note:** `reconstruction` / `dead_latents` / `loss_recovered` are already in `sae.eval`; **`probing` is the newest module and lands with the eval recipe PR** — if it's not in your tree yet, that PR is the dependency.

## Verifying the recipe works (fastest → most confident)

1. **Mechanical** — pipeline runs end-to-end, `checkpoint_final.pt` exists. Smoke on 20–100 sequences (minutes).
2. **Numerical** — `train.py` log shows loss ↓, FVU < 1, dead-% trending toward ~20% (not stuck at ~80%). If dead-% is stuck high, check normalize-input / presample / the LR horizon (gotcha 1).
3. **Shape sanity** — `torch.load(checkpoint_final.pt)`: encoder `[hidden_dim → expansion·hidden_dim]`, decoder the transpose.

## Limitations

- **`train.py` is duplicated per recipe** (a known smell). Until it folds into one shared `sae` entrypoint, copying an *older* recipe's `train.py` silently drops the opt-in flags — always copy evo2's.
- **The flags and numeric values are Evo2-tuned examples, not universal defaults.** Re-tune `expansion-factor`/`top-k`/`auxk` to your `hidden_dim`; set `--mix-shards`/`--presample-shards` to `1` if your shards are already shuffled.
- **`probing` lands with the eval-recipe PR.** `reconstruction`/`dead_latents`/`loss_recovered` are already in `sae.eval`; if `sae.eval.probing` isn't in your tree yet, that PR is the dependency.
- **The Megatron/MBridge path requires the NVIDIA PyTorch container + TransformerEngine.** HF-native models (ESM2/CodonFM) skip MBridge entirely.
- **Some inherited perf numbers are unverified** (e.g. the ~10× memory / ~17× speedup note) — measure on your own model before quoting them.

## Reference recipes

| Recipe     | Extract path                                                                                                   | Mirror it when                                                    |
| ---------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `esm2/`    | `extract.py` → HF `AutoModel.from_pretrained` + `output_hidden_states`                                         | new model is HF-native with a clean `AutoModel`                   |
| `codonfm/` | `extract.py` → custom inference class                                                                          | new model has its own checkpoint + forward code                   |
| `evo2/`    | **streaming** `extract.py` — wraps `predict_evo2`, monkeypatches its writer to an `ActivationStore` (no `.pt`) | upstream already has a `predict_<model>` CLI; reuse it and stream |

All share a near-identical `train.py` and the `ActivationStore` parquet contract — **but only evo2's currently wires the opt-in flags** (codonfm/esm2 lag), so copy evo2's. Folding the duplicated train-CLI into one shared `sae` entrypoint (so no recipe can drift) is the planned fix.
