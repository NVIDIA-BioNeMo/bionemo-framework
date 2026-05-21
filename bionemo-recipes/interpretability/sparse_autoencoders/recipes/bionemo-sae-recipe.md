---
name: bionemo-sae-recipe
description: Build a new sparse-autoencoder recipe under bionemo-recipes/interpretability/sparse_autoencoders/recipes/ for a biological foundation model (e.g. evo2, nemotron, geneformer). Trigger when the user asks to "add SAE for <model>", "build a new SAE recipe", or wants to extract activations from a bionemo model and train an SAE on them.
---

# Build a new SAE recipe in bionemo-framework

## The pattern

Every SAE recipe in `bionemo-recipes/interpretability/sparse_autoencoders/recipes/` decomposes into the same three stages, separated by a universal contract:

```
extractor (model-specific)  →  ActivationStore parquet shards  →  train.py (universal)
                                       ↑ contract
```

- **Extractor** does the model forward pass and dumps `[N_tokens, hidden_dim]` activations. Model-specific.
- **ActivationStore** (in `sae/src/sae/activation_store.py`) is the universal on-disk format: a directory of `shard_{NNNNN}.parquet` files plus `metadata.json` with `{model_name, layer, hidden_dim, n_samples, n_shards, shard_size, n_sequences}`.
- **train.py** loads via `sae.activation_store.load_activations(cache_dir)` and trains a TopK or ReLU SAE. It's effectively the same script across recipes — copy verbatim from codonfm or esm2 and only edit the docstring.

If the model has its own inference tool that dumps activations in a different format, **write a shim** (~50 lines) that walks the inference output and appends to `ActivationStore`. Don't fork train.py.

## When this applies

Bringing up an SAE on a new biological foundation model — Evo2, ESM2, CodonFM, Nemotron, Geneformer, etc. A checkpoint (HF or local) is in hand or known. Scope is the extract→shard→train pipeline; eval/dashboard, hyperparameter sweeps, and training the underlying biological model itself are separate work.

## Steps for a new recipe

### 1. Reconnaissance (read, don't write)

- Look at `recipes/esm2/` (HF AutoModel path) and `recipes/codonfm/` (custom checkpoint path) as templates. Pick the closer match.
- Find the model's inference path in `bionemo-recipes/recipes/<model>_*/`. If it has a dedicated `predict_<model>` CLI, use it; otherwise write a custom `extract.py` modeled on `recipes/esm2/scripts/extract.py`.
- Identify: hidden_dim, layer count, trained context length (CRITICAL — see gotchas).

### 2. Build the upstream env if you need one

Recipes under `bionemo-recipes/recipes/<model>_*/` have a `.ci_build.sh` that creates a `.venv` with `--system-site-packages`. **This assumes you are inside the NVIDIA pytorch container** with TransformerEngine pre-installed at `/usr/local/lib/python*/dist-packages/transformer_engine/`. Verify before running:

```bash
ls /usr/local/lib/python*/dist-packages/transformer_engine 2>/dev/null && echo "OK to build"
```

### 3. Scaffold the recipe dir

```
bionemo-recipes/interpretability/sparse_autoencoders/recipes/<model>/
├── README.md
├── pyproject.toml        # deps: sae, torch, numpy, tqdm, pyarrow. workspace source: sae = { workspace = true }
├── src/<model>_sae/__init__.py
└── scripts/
    ├── 1b.sh             # orchestrator
    ├── train.py          # COPY VERBATIM from codonfm/scripts/train.py
    └── <shim>.py         # only if upstream extractor doesn't write parquet directly
```

### 4. Validate end-to-end on a tiny subset before scaling

Always: small FASTA/CSV (20–100 sequences) → full pipeline → confirm SAE loss drops. The smoke test catches every integration bug an hour earlier than the full run does.

### 5. Add cache guards to `1b.sh`

Each long step needs an idempotency check so reruns don't redo the expensive work. Use sentinel files the step itself produces:

```bash
# checkpoint convert: skip if ckpt dir has the expected sentinel
[[ -f "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]] || convert...

# extract: skip if any output exists (use compgen -G, not ls parsing)
compgen -G "${PREDICT_DIR}/predictions__*.pt" > /dev/null || extract...

# shim: skip if ActivationStore metadata.json exists (only finalize() writes it)
[[ -f "${PARQUET_DIR}/metadata.json" ]] || python shim.py...
```

**Caveat**: the cache guard only knows "files exist," not "files are from the right input." Document that input changes require manual invalidation, or hash the input into the .pt dir.

## Known gotchas

These have cost real debug time. Check each one when bringing up a new recipe.

### Container / env

1. **`.ci_build.sh` assumes system-site-packages TE.** Outside the nvidia/pytorch container, `uv venv --system-site-packages` won't pick up TransformerEngine and the build silently breaks later. Verify TE before running.

2. **`huggingface-cli` is deprecated → use `hf`.** Same args.

3. **HF README directory names are unreliable.** OpenGenome2's README says `jsonl/` but the actual dir is `json/`. A bad `--include` pattern silently fetches 0 files. **Always** verify before downloading:

   ```bash
   curl -s "https://huggingface.co/api/datasets/<repo>/tree/main" | python3 -m json.tool
   ```

   And **always** check the downloaded file count is nonzero.

### Checkpoint loading

4. **`weights_only=True` (torch 2.6 default) breaks legacy checkpoints with numpy arrays.** The error is buried in stderr while exit code is 0 and the output dir is empty. Symptom: `UnpicklingError: Unsupported global: numpy.core.multiarray._reconstruct`. Fix: patch the offending `torch.load(...)` to `weights_only=False` if the source is trusted. Audit any `torch.load` call in upstream conversion utilities before running them on legacy checkpoints.

### Model architecture

5. **Hyena (evo2) fftconv OOMs on long sequences even at micro-batch=1.** Hyena's FFT intermediates scale super-linearly with sequence length. A single 100 kb genome can need 30 GB of GPU memory in fftconv alone — beyond H100 80 GB. **Always chunk inputs to the model's trained context length** before extraction:

   - evo2 1B → 8192 bp
   - evo2 7B → check release notes (context-extended)
   - evo2 40B → 1M tokens (per Arc release)
   - Don't rely on the inference tool to truncate; chunk in a preprocessing step.

6. **predict_evo2 takes uncompressed FASTA only.** Decompress `.gz` to a scratch path before invoking. Don't try `<(zcat ...)` — it fails. **But** if your preprocessor (chunker, length filter, etc.) already reads `.gz` and writes plain `.fasta`, you don't need a separate gunzip step in the orchestrator — the preprocessor is enough.

### Output format

7. **predict_evo2 `--embedding-layer N` output schema:** dict of `{hidden_embeddings: [B,S,H], pad_mask: [B,S], seq_idx: [B], tokens: [B,S], batch_idx: int}`. `pad_mask` is a **loss mask** (1 = valid), not a HF attention mask — the name misleads. Filename pattern: `predictions__rank_R__dp_rank_D__batch_B.pt`.

### Pipeline ops

08. **Disk pressure is real.** Per-batch `.pt` files at micro-batch=1 with 8192 bp sequences run ~50 MB each. With un-chunked sequences (100kb+ organelles) they balloon to 200+ MB. Estimate: `n_chunks * 60 MB` to plan `/data` capacity. Add a stream-and-delete mode for production runs.

09. **codonfm's `train.py` works unmodified for new models.** Only uses `--model-path` for a one-line cache-validation warning. Don't fork or parameterize it — accept the warning.

10. **`micro-batch-size 1` may be far from optimal.** Once chunks are uniform and short, GPU memory drops by ~10× and `--micro-batch-size` can be raised significantly. Chunking alone produced a ~17× per-batch speedup on Evo2 1B + organelle FASTA.

11. **Don't extrapolate shim throughput from smoke tests.** The pt→parquet shim is single-threaded and I/O-bound: roughly ~1.3 `.pt` files/sec on 50-MB inputs. A smoke test on 20 small files finishes in seconds; scaling to thousands of full-size files takes hours, not minutes. Parallelize with `ThreadPoolExecutor` if needed — but not worth retrofitting mid-run.

12. **Cache guards check existence, not provenance.** The `[[ -f sentinel ]]` skips in `1b.sh` reuse outputs blindly if the input FASTA, model, or layer changes. Manually `rm -rf` the affected output dirs when invalidating, or hash the input into the output metadata. Easy to forget when iterating on inputs.

## Verifying the recipe works

Three layers, in order of speed-to-confidence:

1. **Mechanical** — pipeline runs end-to-end, no crash, `checkpoint_final.pt` exists. Smoke test on 20–100 sequences proves this in minutes.
2. **Numerical** — training log shows loss decreasing, FVU dropping below 1, dead latents staying near 0. Read off the `Step N | fvu: ... loss: ...` lines from `train.py`.
3. **Shape sanity** — `torch.load(checkpoint_final.pt)` and confirm encoder is `[hidden_dim → expansion_factor × hidden_dim]`, decoder is the transpose. For evo2 1B with default flags: `[1920 → 15360]` / `[15360 → 1920]`.

If 1+2 pass on a smoke test, the same pipeline at full scale is almost certainly fine — most failure modes (OOM, format mismatch, missing files) surface within the first few batches.

## Reference recipes

Three recipes already exist in `bionemo-recipes/interpretability/sparse_autoencoders/recipes/`. Pick the structural template closest to the new model:

| Recipe     | Extract path                                                                 | When to mirror it                                                                                  |
| ---------- | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `esm2/`    | `extract.py` calling HF `AutoModel.from_pretrained` + `output_hidden_states` | New model is HF-native with a clean `AutoModel` path                                               |
| `codonfm/` | `extract.py` calling a custom inference class with its own forward pass      | New model has its own checkpoint format and inference code in `bionemo-recipes/recipes/<model>_*/` |
| `evo2/`    | Upstream `predict_evo2` CLI + `pt_to_parquet.py` shim (no custom extract.py) | Upstream already has a predict-style CLI; reuse it and just convert its output format              |

All three share the same `train.py` (codonfm's is copied verbatim into evo2) and the same `ActivationStore` parquet contract.
