# OpenGenome2 7B FP8 Precision Agent — Milestones 1 & 2

## Run Variables

These can be edited by the user before each run.

```
TOLERANCE_PCT          = 5.0                           # max allowed perplexity difference as % of baseline (e.g. 5.0 = 5%)
BASELINE_LOGFILE       = ./baseline_bf16.json          # BF16 baseline (co-located with this guide)
NUM_TRAIN_STEPS        = ???                           # total training steps; agent stops here
CHECKIN_INTERVAL       = 100                           # steps between check-ins (must align with baseline logfile steps)
LAYERS_PER_PROMOTION   = 2                             # layers demoted to BF16 per failed check-in
NUM_LAYERS             = 32                            # transformer block layers (OG2-7B has layers 1-32)
INITIAL_PRECISION      = fp8                           # starting precision for all transformer block layers
PROMOTION_STRATEGY     = ends_in                       # "ends_in", "tail_in", or "research_guided" (see Promotion Strategies below)
WORKSPACE_ROOT         = /data/savithas/agent_runs      # root for all agent output (NFS)
CHECKPOINT_ROOT        = /data/savithas/checkpoints     # root for model checkpoints (NFS)
RESULTS_FOLDER         = /data/savithas/agent_runs/results/$PROMOTION_STRATEGY  # final reports, scoped by strategy

# Training script & model
TRAINING_SCRIPT        = train_fsdp2.py
CONFIG_NAME            = og2_7b_thd_gqa_fp8            # Hydra config name (inherits og2_7b_thd_gqa)
NPROC_PER_NODE         = 8                             # GPUs per node
NNODES                 = 6                             # Number of Lepton nodes
TOKENIZER              = ./tokenizers/nucleotide_fast_tokenizer
MICRO_BATCH_SIZE       = 1                             # per-GPU micro batch size
GRAD_ACC_STEPS         = 8                             # gradient accumulation steps
DATASET_PATH           = /data/opengenome2/json/pretraining_or_both_phases/metagenomes/data_metagenomics_train_*.jsonl.gz
WANDB_PROJECT          = opengenome2-7b                # WandB project name for all runs
LAUNCH_DIR             = ???                           # NFS directory for multi-node launch coordination (set by prompt)
```

______________________________________________________________________

## Project Overview

We are building an agent that manages a pretraining loop for an OpenGenome2 7B Llama3 model using TransformerEngine with FP8 Block Scaling. The agent's objective is to **maximize the number of layers running in FP8** while keeping pretraining accuracy within tolerance of a BF16 baseline. The agent does this by controlling a per-layer precision schedule and monitoring training metrics at regular intervals. Training runs for a fixed number of steps (`--num_train_steps=$NUM_TRAIN_STEPS`).

### Key Differences from ESM2/NVFP4 Guide

| Aspect            | ESM2 NVFP4 Guide                | OG2 FP8 Guide                                                                        |
| ----------------- | ------------------------------- | ------------------------------------------------------------------------------------ |
| Model             | ESM2-3B (36 layers)             | OG2 Llama3-7B (32 layers)                                                            |
| Precision levels  | NVFP4 → MXFP8 → BF16 (3 levels) | **FP8 → BF16 (2 levels)**                                                            |
| FP8 recipe        | MXFP8BlockScaling               | **Float8BlockScaling**                                                               |
| FP4 support       | Yes (NVFP4BlockScaling)         | **No** — FP4 is not used                                                             |
| Infrastructure    | Single node                     | **Lepton multi-node** (6+ nodes)                                                     |
| Model features    | Standard                        | Spike-No-More init, Megatron scaled init, weight decay grouping, FP32 master weights |
| Sequence handling | Packing                         | THD sequence packing with genomic masking                                            |

### Reference Materials

All reference materials are co-located with this guide:

- **Strategy documents**: [OG2_STRATEGY_ENDS_IN.md](OG2_STRATEGY_ENDS_IN.md), [OG2_STRATEGY_TAIL_IN.md](OG2_STRATEGY_TAIL_IN.md) — full pseudocode, demotion tables, and worked examples for each strategy
- **BF16 baseline**: [baseline_bf16.json](baseline_bf16.json) — extracted from WandB run `8mfsb27t` via [extract_baseline_metrics.py](extract_baseline_metrics.py)
- **Quant stats config**: [fp8_debugging_stats.yaml](fp8_debugging_stats.yaml) — TE debug feature config for `research_guided` strategy
- **Research paper**: [references/NVIDIA-Nemotron-3-Super-Technical-Report.pdf](references/NVIDIA-Nemotron-3-Super-Technical-Report.pdf) — Nemotron-3 Super Technical Report describing the low-precision training agent approach this guide is adapted from

______________________________________________________________________

## Architecture Summary

```
┌─────────────────────────────────────────────────────────┐
│ Milestone 2: Model Persistence & Recovery               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ Milestone 1: Pretraining Agent Loop               │  │
│  │                                                   │  │
│  │  Agent → Change Layer Precision → Pretrain        │  │
│  │    ↑         ↓                                    │  │
│  │    ←── Checkin (Training) ←── Baseline BF16       │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  Checkin (Training) ── Save ──→ Models (disk)           │
│  Models ── Reload LKG (on failure) ──→ Agent            │
└─────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Metrics

### Accuracy Metric: Perplexity (lower is better)

Perplexity is the primary accuracy metric: `perplexity = exp(loss)`. At every check-in, the agent compares current perplexity against the BF16 baseline perplexity at the same step using a relative tolerance. A check-in passes if: `current_perplexity - baseline_perplexity <= baseline_perplexity * (TOLERANCE_PCT / 100)`. This means the tolerance adapts to the scale of the perplexity — it is more lenient early in training when perplexity is high and volatile, and tighter later when perplexity has settled. Only perplexity failures trigger rollbacks.

**Important**: The OG2 training script logs `loss` to stdout and WandB but does NOT log perplexity directly. The agent must compute perplexity from the loss value: `perplexity = exp(loss)`.

### Performance Metric: Unpadded Tokens Per Second (higher is better)

Measures actual training throughput excluding padding tokens. The agent should log this at every check-in to confirm FP8 is yielding a throughput benefit. Throughput drops do NOT trigger rollbacks — they are logged for informational purposes only.

### Metric Retrieval (WandB Local Files)

The agent launches training as a background process with `num_train_steps=$NUM_TRAIN_STEPS` (the full target). Training runs continuously — the agent does NOT stop/restart at each check-in. Instead, it monitors WandB's local log files to read per-step metrics in near-real-time.

WandB writes run data to a local directory during training. The path is printed to stdout at launch:

```
wandb: Run data is saved locally in <training_script_dir>/wandb/run-<timestamp>-<run_id>
```

A symlink `<training_script_dir>/wandb/latest-run` always points to the active run.

The per-step metrics file is:

```
<training_script_dir>/wandb/latest-run/files/wandb-history.jsonl
```

Each line is a JSON object with all metrics logged at that step:

```json
{"train/global_step": 100, "train/loss": 2.72, "train/unpadded_tokens_per_second_per_gpu": 10035.6, ...}
```

**Agent monitoring loop:**

1. Launch training as a background process.
2. Poll `wandb-history.jsonl` periodically (e.g. every 30 seconds).
3. When a new line appears where `train/global_step` matches a check-in step (`$CHECKIN_INTERVAL` multiple), parse the metrics. Key WandB fields:
   - `train/global_step` — step number
   - `train/loss` — compute `perplexity = exp(loss)` (compare against baseline `"perplexity"`)
   - `train/unpadded_tokens_per_second_per_gpu` — throughput per GPU
4. Compare perplexity against baseline (see Check-ins below).
5. Pass → do nothing, let training continue.
6. Fail → kill the training process **IMMEDIATELY**, then perform LKG recovery (see Milestone 2), and relaunch with updated precision schedule.

If multiple new check-in steps appear between polls, process them in order (lowest step first). Kill on the **FIRST** failure — do not continue evaluating later steps. This ensures the LKG checkpoint hasn't been auto-deleted by `max_checkpoints` before the agent acts.

Since the agent `cd`'s into `$(dirname $TRAINING_SCRIPT)` before launching, the wandb directory is relative to that working directory.

### Stdout Metric Format

The OG2 `perf_logger.py` also outputs metrics at every `logger.frequency` steps to stdout in this format:

```
loss: 2.94, learning_rate: 3e-05, grad_norm: 1.23, step_time: 0.456, tokens_per_second_per_gpu: 4.85e+04, unpadded_tokens_per_second_per_gpu: 4.5e+04, total_unpadded_tokens_per_batch: 12345, gpu_memory_allocated_max_gb: 65.3, gpu_memory_allocated_mean_gb: 65.3, global_step: 100
```

Key fields to parse:

- `loss` — compute `perplexity = exp(loss)`
- `unpadded_tokens_per_second_per_gpu` — throughput metric
- `global_step` — current training step

______________________________________________________________________

## CLI Reference

### Base Training Command (OG2-7B on Lepton)

The agent constructs each training launch from this template. First `cd` into the training script directory, then run torchrun. Fields marked `← AGENT CONTROLS` are modified between launches; everything else stays fixed.

```bash
cd $(dirname $TRAINING_SCRIPT)
torchrun \
  --nproc_per_node=$NPROC_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $(basename $TRAINING_SCRIPT) \
  --config-name $CONFIG_NAME \
  num_train_steps=$NUM_TRAIN_STEPS \
  grad_acc_steps=$GRAD_ACC_STEPS \                                # ← FIXED (do NOT change)
  fp8_config.enabled=True \
  fp8_config.fp8_recipe=transformer_engine.common.recipe.Float8BlockScaling \
  fp8_config.fp8_format=E4M3 \
  fp8_layers='[2,3,4,...,31]' \                                   # ← AGENT CONTROLS
  quant_stats_config.enabled=<true|false> \                       # ← AGENT CONTROLS (see below)
  quant_stats_config.quant_stats_file=./fp8_debugging_stats.yaml \
  quant_stats_config.quant_log_dir=$WORKSPACE_ROOT/<run_name>/quant_stats \
  checkpoint.ckpt_dir=$CHECKPOINT_ROOT/<run_name> \               # ← FIXED (same dir for entire session)
  checkpoint.save_every_n_steps=$CHECKIN_INTERVAL \
  checkpoint.resume_from_checkpoint=true \                        # ← FIXED (always true; auto-finds latest checkpoint)
  checkpoint.max_checkpoints=4 \
  checkpoint.save_final_model=true \
  checkpoint.async_save=false \                                      # ← FIXED (sync saves for reliability)
  dataset.use_stateful_dataloader=true \                             # ← FIXED (CRITICAL: preserves dataloader position across relaunches)
  validation.enabled=true \                                          # ← FIXED
  validation.eval_interval=1000 \                                    # ← FIXED (every 1000 steps)
  validation.num_batches=40 \                                        # ← FIXED
  hydra.run.dir=$WORKSPACE_ROOT/<run_name>/hydra_outputs \
  wandb.project=$WANDB_PROJECT \                                  # ← FIXED
  wandb.name=<run_name> \                                         # ← FIXED (same name for entire session — produces one continuous WandB curve)
  +wandb.id=<run_name> \                                          # ← FIXED (WandB resumes by ID, not name — must match across relaunches)
  +wandb.resume=allow                                             # ← FIXED (resumes the same WandB run on relaunch)
```

**CRITICAL**: The agent must use this command template EXACTLY. Do NOT add, remove, or modify any parameters not marked `← AGENT CONTROLS`. In particular:

- Do NOT change `grad_acc_steps` — it is set in the Hydra config and must stay at $GRAD_ACC_STEPS
- Do NOT add `dataset.*`, `adamw_kwargs.*`, or `lr_scheduler_kwargs.*` overrides — these are already set in the Hydra config
- Do NOT scale any parameter by the number of nodes or GPUs
- `logger.frequency` is set in the Hydra config (every step) — do NOT override it

**Notes on OG2 vs ESM2 differences:**

- **No `fp4_config` or `fp4_layers`** — FP4 is not used for OG2 FP8 Block Scaling
- **Only `fp8_layers`** controls which layers run in FP8; layers absent from the list default to BF16
- The `og2_7b_thd_gqa_fp8` config already sets: `use_sequence_packing=true`, `use_fp32_master_weights=true`, `spike_no_more_embedding_init=true`, `use_megatron_scaled_init=true`, `dataset.*`, `adamw_kwargs.*`, `lr_scheduler_kwargs.*`, `logger.frequency=1`, `grad_acc_steps=8`
- **Do NOT override** any Hydra config values that are not in the template above — the config is carefully tuned
- Multi-node Lepton: `MASTER_ADDR`/`MASTER_PORT` are provided by the Lepton environment

The agent modifies these fields between launches:

- `fp8_layers` — updated based on the current precision schedule
- `quant_stats_config.enabled` — `true` for `research_guided` only; `false` for `ends_in` and `tail_in`

These fields are FIXED for the entire session (never change between launches):

- `grad_acc_steps` — always `$GRAD_ACC_STEPS` (do NOT scale by nodes/GPUs — FSDP handles distributed scaling)
- `checkpoint.ckpt_dir` — always `$CHECKPOINT_ROOT/<run_name>` (**NOT** `$WORKSPACE_ROOT/...`). CHECKPOINT_ROOT and WORKSPACE_ROOT are different directories. Same directory for the entire session; matches Lepton job name and wandb run name.
- `num_train_steps` — always `$NUM_TRAIN_STEPS` (absolute target)
- `checkpoint.resume_from_checkpoint` — always `true` (the script auto-finds the latest checkpoint; on first launch with no checkpoints it starts fresh automatically)
- `dataset.use_stateful_dataloader` — always `true` (see Data Integrity section below)
- `wandb.name` — always `<run_name>` (computed once at session start, never changes)
- `+wandb.id` — always `<run_name>` (WandB resumes by ID, not name — must match across relaunches)
- `+wandb.resume` — always `allow` (resumes the same WandB run on relaunch)

### Multi-Node Launch Protocol

Training runs on `$NNODES` nodes. This agent runs on rank 0 only. Worker nodes (ranks 1 through NNODES-1) poll a shared NFS directory for barrier-based round files and execute the same torchrun command.

**IMPORTANT: `$TORCHRUN_PREFIX` is pre-configured in your environment.** It contains the correct `torchrun --nproc_per_node=... --nnodes=... --node_rank=... --master_addr=... --master_port=...` flags for THIS node. You MUST use `$TORCHRUN_PREFIX` for ALL training launches. Do NOT construct your own torchrun command — the prefix has the correct multi-node flags that you cannot easily determine yourself.

**Before EVERY `torchrun` launch, you MUST:**

1. Increment your round counter (start from 1 for the first launch in this session).
2. Write a `round_N_args.env` file to `$LAUNCH_DIR/` containing a `TRAIN_CMD` variable with all Python script arguments (NOT the torchrun prefix — workers prepend that themselves):
   ```bash
   TRAIN_CMD="train_fsdp2.py \
     --config-name og2_7b_thd_gqa_fp8 \
     fp8_layers='[3,4,...,30]' \
     ... all other args ..."
   ```
3. Touch `$LAUNCH_DIR/round_N_ready` to signal workers to start.
4. Then run `$TORCHRUN_PREFIX $TRAIN_CMD` on rank 0 (using the SAME TRAIN_CMD you wrote to the file).

Example (first launch):

```bash
# Step 1: Write training args for workers (use single-quoted heredoc to preserve $variables)
cat > $LAUNCH_DIR/round_1_args.env << 'ARGS_EOF'
TRAIN_CMD="train_fsdp2.py \
  --config-name og2_7b_thd_gqa_fp8 \
  fp8_layers='[3,4,...,30]' \
  num_train_steps=182300 \
  ... all other args ..."
ARGS_EOF

# Step 2: Signal workers to start (MUST come AFTER writing args)
touch $LAUNCH_DIR/round_1_ready

# Step 3: Run torchrun on rank 0 using $TORCHRUN_PREFIX (NEVER construct your own torchrun flags)
cd /data/savithas/bionemo-framework/bionemo-recipes/recipes/opengenome2_llama_native_te
$TORCHRUN_PREFIX train_fsdp2.py --config-name og2_7b_thd_gqa_fp8 \
  fp8_layers='[3,4,...,30]' \
  num_train_steps=182300 \
  ... all other args ...
```

**When killing training:** Kill the torchrun process on rank 0. Workers detect the disconnection (NCCL timeout, up to 30 minutes) and their processes exit automatically. Workers then poll for the next round's ready file. After killing:

1. Write the next round's `round_N_args.env` immediately.
2. Wait at least 2 minutes for workers to finish dying and reach the polling loop.
3. Touch `round_N_ready` to signal workers.
4. Run `$TORCHRUN_PREFIX $TRAIN_CMD` on rank 0. C10D rendezvous waits for all nodes (retry logic handles any stragglers).

**To signal completion:** `touch $LAUNCH_DIR/done` — all workers exit cleanly.

**CRITICAL rules:**

- **ALWAYS use `$TORCHRUN_PREFIX`** to launch training. NEVER write your own `torchrun --nproc_per_node=... --nnodes=...` command. The prefix already contains the correct multi-node flags (nnodes, node_rank, master_addr, master_port). If you construct your own torchrun command, you WILL get single-node training (8 GPUs instead of 48).
- **Use single-quoted heredoc** (`<< 'ARGS_EOF'`) when writing `round_N_args.env`. This preserves `$NODE_RANK`, `$MASTER_ADDR`, and `$MASTER_PORT` as literal variables. Each worker node has these set to its own values by the Lepton environment. If you expand them, all workers will think they are rank 0.
- Write `round_N_args.env` BEFORE touching `round_N_ready`. Workers source the args file immediately after detecting the ready file.
- The `TRAIN_CMD` must NOT include `torchrun` or its flags — only the Python script and its Hydra arguments. Workers prepend the torchrun prefix themselves (with their own `$NODE_RANK`, `$MASTER_ADDR`, etc.).
- When killing training: just kill the torchrun process on rank 0. Workers detect the disconnection (NCCL timeout) and exit automatically. Workers then block on the next round's ready file.
- Track the round counter in `state.json` so you can resume correctly after a crash.

### Post-Launch Verification (MANDATORY)

After the **first** torchrun launch in this session, verify the training setup is correct before proceeding. Check the stdout/WandB output for:

1. **GPU count**: Must show `GPU count: {NPROC_PER_NODE * NNODES}` (e.g., 48 for 6 nodes × 8 GPUs). If it shows only `NPROC_PER_NODE` (e.g., 8), multi-node is broken — kill immediately and debug.
2. **grad_acc_steps**: Must be `$GRAD_ACC_STEPS` (e.g., 8). If it shows any other value, kill immediately and fix.
3. **Effective batch size**: Should be `$MICRO_BATCH_SIZE × $NPROC_PER_NODE × $NNODES × $GRAD_ACC_STEPS` (e.g., 1 × 8 × 6 × 8 = 384).
4. **Resume step**: For warm-start, must show `Starting training loop from step <LKG_STEP + 1>`.

If ANY of these are wrong, kill training immediately, diagnose the issue, fix it, and relaunch. Do NOT let incorrect training continue — it wastes GPU time and produces unusable results.

### Validation as Downstream Signal

Validation is enabled in the training command (`validation.enabled=true`). The training script automatically runs validation every 1000 steps and logs `val/loss` and `val/ppl` to WandB. This provides a downstream-like signal: the FP8 paper (Nemotron-3 Super) notes that training loss can diverge slightly under low-precision without hurting downstream task quality.

**How the agent uses validation metrics:**

- At each check-in, also read `val/loss` and `val/ppl` from `wandb-history.jsonl` (if a validation step has occurred since the last check-in).
- Log validation metrics to `history.json` alongside training metrics.
- Validation metrics are **informational only** — they do NOT trigger rollbacks. Only training perplexity triggers rollbacks.
- In `report.md`, include a comparison of validation perplexity between this FP8 run and the BF16 baseline. This helps determine if FP8 precision loss affects downstream quality.
- If validation fails with an error (e.g., data loading issue), the training script already handles this with try/except — training continues uninterrupted. The agent should log the failure but NOT take any action.

### Layer Precision Control

Precision is controlled via a single list:

- `fp8_layers=[2,3,...,31]` — layers running in FP8 Block Scaling
- Layers NOT in `fp8_layers` default to BF16

Embedding and lm_head are always in BF16 (not addressable via `fp8_layers`). The agent only manages transformer block layer indices 1-32 (1-indexed). A layer cannot appear in `fp8_layers` and also be in BF16 — its presence or absence in the list is the control.

When using FP8, the recipe config must also be passed:

```
fp8_config.enabled=True
fp8_config.fp8_recipe=transformer_engine.common.recipe.Float8BlockScaling
fp8_config.fp8_format=E4M3
fp8_layers=[...]
```

### WandB Run Naming & Resume

`wandb.name` and `+wandb.id` are both set to `<run_name>` (e.g. `ends_in_20260317_143000`). These are computed ONCE at session start and NEVER change. WandB resumes by **run ID** (not name), so `+wandb.id` is required. Combined with `+wandb.resume=allow`, every relaunch appends to the same WandB run, producing a single continuous curve in the dashboard.

This means all training segments (warmup, expansions, rollbacks) appear as one continuous line in WandB, making it easy to see the full training trajectory including any loss spikes from precision changes.

### Checkpointing & Resume

The training script has built-in checkpoint support. To enable:

```
checkpoint.ckpt_dir=$CHECKPOINT_ROOT/<run_name>
checkpoint.save_every_n_steps=$CHECKIN_INTERVAL
checkpoint.resume_from_checkpoint=true
```

To resume after a stop or crash, re-run the exact same command. The script automatically finds the latest `step_*` checkpoint in `checkpoint.ckpt_dir`, restores state, and resumes from step + 1.

**What gets restored on resume:**

- Model weights, optimizer state (AdamW moments), LR scheduler, step counter, epoch counter, dataloader position (`use_stateful_dataloader=true`)

All state — including where the dataloader left off in the dataset — is restored from checkpoint.

**Key behavior:**

- `num_train_steps` is an absolute target, not relative. Resuming at step 400 with `num_train_steps=$NUM_TRAIN_STEPS` trains steps 401–`$NUM_TRAIN_STEPS`.
- `checkpoint.resume_from_checkpoint=false` forces a fresh start even if checkpoints exist.
- Checkpoints are saved as `<ckpt_dir>/train_fsdp2/step_<N>/`.

Additional checkpoint flags:

```
checkpoint.max_checkpoints=4         # keep 4 most recent (buffer so LKG isn't auto-deleted before agent can act)
checkpoint.save_final_model=true     # save .safetensors at end of training
```

LKG recovery: the agent deletes any checkpoint newer than the LKG from `$CHECKPOINT_ROOT/<run_name>/train_fsdp2/`. Since `checkpoint.resume_from_checkpoint=true` always, the training script auto-finds the latest remaining checkpoint (the LKG) and resumes from there. The `checkpoint.ckpt_dir` never changes.

**CRITICAL — Data Integrity on Relaunch:**

The agent kills and relaunches training at every expansion (pass) and every rollback (fail). With `dataset.use_stateful_dataloader=true`, the dataloader position is saved in each checkpoint and restored on resume, so the model sees each training batch exactly once — the same as a continuous baseline run. If `use_stateful_dataloader` is NOT enabled, every relaunch resets the dataloader to the start of the dataset. This causes the model to re-train on early batches multiple times, artificially lowering training loss (overfitting to repeated data) and invalidating comparisons against the BF16 baseline. The agent MUST verify that `dataset.use_stateful_dataloader=true` is set in the training command.

**CRITICAL — Checkpoint Safety Rules:**

- NEVER delete the checkpoint directory itself (`$CHECKPOINT_ROOT/<run_name>/`). Only delete individual `step_<N>/` subdirectories inside it.
- NEVER delete the run directory (`$WORKSPACE_ROOT/<run_name>/`). It contains checkpoints, logs, history, and configs that are irreplaceable.
- NEVER use `rm -rf` on any parent directory. Only `rm -rf` specific `step_<N>/` subdirectories that are newer than the LKG.
- If a checkpoint appears corrupt or incomplete, delete ONLY that specific `step_<N>/` subdirectory — not the entire checkpoints folder.
- Before deleting, always list the contents of the checkpoint directory first to confirm which checkpoints exist and which is the LKG.
- When in doubt, do NOT delete — relaunch and let the training script skip the corrupt checkpoint automatically.

### Quantization Stats Logging

TransformerEngine provides per-layer quantization statistics (underflow %, scale_inv_min, scale_inv_max, MSE). These are **only used by the `research_guided` strategy**.

When enabled (`research_guided` only), these flags apply:

```
quant_stats_config.enabled=true
quant_stats_config.quant_stats_file=./fp8_debugging_stats.yaml
quant_stats_config.quant_log_dir=$WORKSPACE_ROOT/<run_name>/quant_stats
```

Stats are logged to `<quant_log_dir>/rank_*/nvdlfw_inspect_statistics_logs/`. Logging frequency is controlled via the `freq` parameter in the stats YAML config. Available stats for FP8 Block Scaling:

- `underflows%` — percentage of non-zero elements rounded to zero after quantization
- `scale_inv_min` / `scale_inv_max` — range of inverse scaling factors across blocks
- `mse` — mean squared error between quantized and original tensor

These are collected per layer, per tensor type (activation, gradient, weight).

For `ends_in` and `tail_in`: set `quant_stats_config.enabled=false`. These strategies use deterministic demotion orders and do not need runtime quant signals.

Reference implementation: `quantization.py` in this recipe directory.

______________________________________________________________________

## Precision Levels

For OG2 FP8 Block Scaling, there are only **two precision levels**:

```
FP8 (Float8BlockScaling E4M3)  →  BF16
```

Demotion is always FP8 → BF16. There is no intermediate level.

______________________________________________________________________

## Promotion Strategies

The agent uses `PROMOTION_STRATEGY` to decide which layers to **demote from FP8 to BF16** when a check-in fails. Three strategies are available:

### Strategy 1: `ends_in` (default)

Demote `LAYERS_PER_PROMOTION` layers per failed check-in, working inward from both ends simultaneously:

```
Failure 1: demote layers 1, 32    (outermost pair)
Failure 2: demote layers 2, 31
Failure 3: demote layers 3, 30
Failure 4: demote layers 4, 29
...continuing inward...
Last: layers 16, 17              (very center, last to be demoted)
```

**Rationale**: Edge layers (closest to embedding input and projection output) are more sensitive to quantization error. Middle layers are most tolerant. Never demote from the middle outward.

See [OG2_STRATEGY_ENDS_IN.md](OG2_STRATEGY_ENDS_IN.md) for full pseudocode, demotion table, and worked example.

### Strategy 2: `tail_in`

Demote `LAYERS_PER_PROMOTION` layers per failed check-in, starting from the last layer and working toward the first. The agent decides how many total layers to demote — there is no fixed cap.

```
Failure 1: demote layers 32, 31   (last 2 layers)
Failure 2: demote layers 30, 29   (next 2 toward head)
Failure 3: demote layers 28, 27
...continuing toward layer 1...
```

**Rationale**: The final layers of a transformer (closest to the output projection) are typically most sensitive to quantization error. Demoting from the tail inward addresses the most sensitive layers first.

See [OG2_STRATEGY_TAIL_IN.md](OG2_STRATEGY_TAIL_IN.md) for full pseudocode, demotion table, and worked example.

### Strategy 3: `research_guided`

This is the **only strategy that enables quant stats logging**. Set `quant_stats_config.enabled=true` in the training command.

Use the quant stats (underflow %, scale_inv_min, scale_inv_max, MSE) to collect per-layer sensitivity signals at runtime. The agent should:

1. Run an initial segment (e.g. first `CHECKIN_INTERVAL` steps) with all layers in FP8 and quant stats enabled.
2. Collect quant stats and identify which layers show the highest underflow/MSE.
3. When a check-in fails, demote the layers with the worst quant stats first (rather than following a fixed geometric pattern).
4. Document in `report.md` how the demotion order was derived from the runtime quant stats, and how it differs from `ends_in` / `tail_in`.

This strategy is exploratory. The agent has freedom to define the demotion order based on runtime quant stats, but must still respect the check-in / rollback loop and log all decisions to `history.json` and `report.md`.

______________________________________________________________________

## Check-ins

Every `CHECKIN_INTERVAL` training steps, the agent performs a check-in:

1. **Record** the current perplexity: `current_perplexity = exp(loss)`.
2. **Look up** the BF16 baseline perplexity for this step from `BASELINE_LOGFILE`.
3. **Compute tolerance** for this step: `allowed_delta = baseline_perplexity * (TOLERANCE_PCT / 100)`.
4. **Pass** if: `current_perplexity - baseline_perplexity <= allowed_delta`.
5. **Log** current `unpadded_tokens_per_second_per_gpu` for performance tracking.
6. **Pass** → do nothing, let training continue.
7. **Fail** → kill the training process, update precision schedule (demote `LAYERS_PER_PROMOTION` layers using `PROMOTION_STRATEGY`), and trigger recovery (Milestone 2).

**IMPORTANT**: `CHECKIN_INTERVAL` must align with the baseline logfile step intervals (both are 100 steps). If the agent attempts a check-in at a step that does not exist in the baseline logfile, it must immediately stop and report the error to the user (e.g. "Baseline logfile has no entry for step\_\<N>. Check that CHECKIN_INTERVAL aligns with the baseline step intervals."). Do not interpolate or skip — fail hard.

### Configuration

All tunable values are defined in the "Run Variables" block at the top of this document. The agent must read those values at startup. The two metrics are fixed:

- **Accuracy metric**: perplexity (lower is better)
- **Performance metric**: unpadded_tokens_per_sec (higher is better)

______________________________________________________________________

## Baseline Reference Logfile

Before any agent-managed training begins, a full BF16 baseline run must be completed. The baseline logfile contains per-step metrics at 100-step intervals:

```json
{
  "step_100":  {"perplexity": 6.04, "loss": 1.80, "unpadded_tokens_per_sec": 9013},
  "step_200":  {"perplexity": 4.16, "loss": 1.42, "unpadded_tokens_per_sec": 10047},
  "step_300":  {"perplexity": 4.09, "loss": 1.41, "unpadded_tokens_per_sec": 10075},
  ...
}
```

The agent loads this file at startup. At every check-in it looks up the baseline perplexity for the current step and logs both baseline and current values. The logfile path is provided via config — do not generate baseline values within the agent loop.

______________________________________________________________________

## Milestone 1: Pretraining Agent Loop

### Agent Objective

Run as many layers as possible in FP8 while keeping perplexity within tolerance of the BF16 baseline. Start with all 32 transformer block layers in `$INITIAL_PRECISION`. Demote layers to BF16 only when check-ins fail. Stop after `$NUM_TRAIN_STEPS` steps.

### Layer Precision Schedule

The agent maintains a per-layer precision map for transformer block layers only:

```python
# OG2-7B: 32 transformer block layers (1-32, 1-indexed)
layer_precision = {i: "fp8" for i in range(1, 33)}
```

This map translates directly to the `fp8_layers` CLI argument:

```python
fp8_layers = sorted(k for k, v in layer_precision.items() if v == "fp8")
# Layers not in fp8_layers default to BF16
```

### Precision Levels & Demotion Mechanics

Demotion is always FP8 → BF16. When a check-in fails, the agent removes the selected layers from `fp8_layers` (layers absent from the list default to BF16). Example: `fp8_layers=[1,2,3,...,32]` becomes `fp8_layers=[2,3,...,31]` after demoting layers 1 and 32.

______________________________________________________________________

## Milestone 2: Model Persistence & Recovery

### Saving Checkpoints

The agent uses the built-in checkpoint system (see "Checkpointing & Resume" above). Set `checkpoint.save_every_n_steps` to match `$CHECKIN_INTERVAL` so that a checkpoint exists at every check-in step. The most recent checkpoint that passed a check-in is the "last known good" (LKG).

The agent must also persist alongside each checkpoint:

- The current `fp8_layers` list (the layer precision schedule)

(Model weights, optimizer state, LR scheduler, step counter, and RNG states are handled by the built-in checkpoint system. Each relaunch resumes the same WandB run via `+wandb.id=<run_name>` and `+wandb.resume=allow`.)

### Recovery on Failed Check-in

1. Kill the current training process.
2. Delete any checkpoint newer than the LKG from the checkpoint directory (e.g. if LKG is `step_400` and `step_500` exists, delete `step_500`). This ensures the script resumes from the LKG on relaunch.
3. Demote `LAYERS_PER_PROMOTION` layers using `PROMOTION_STRATEGY`. Update `fp8_layers` accordingly.
4. Relaunch training with the updated precision schedule. Do NOT change `num_train_steps` — keep it at `$NUM_TRAIN_STEPS`. The checkpoint stores the step counter; the script automatically loads the LKG, reads the step from it, and resumes toward the same target. `checkpoint.ckpt_dir` stays the same — the script auto-finds the latest remaining checkpoint (the LKG).

The agent discards all training progress since the last successful check-in. The assumption is that divergence started after the LKG point and the updated schedule will prevent it from recurring.

### Recovery Flow Example

```
Check-in fails at step 200 (current_ppl - baseline_ppl > allowed_delta)
→ Kill training
→ Delete step_200 checkpoint
→ Demote layers 1, 32: FP8 → BF16 (ends_in, first demotion)
→ Update fp8_layers to exclude [1, 32]
→ Relaunch (script auto-resumes from step_100)
→ Next check-in at step 200 (re-do this interval)

Check-in fails again at step 200
→ Kill training
→ Delete step_200 checkpoint
→ Demote layers 2, 31: FP8 → BF16 (second demotion)
→ Update fp8_layers to exclude [1, 2, 31, 32]
→ Relaunch from step_100
→ ...
```

______________________________________________________________________

## Milestone 3: Code Intervention Mode (Optional)

Beyond adjusting the layer precision schedule, the agent can optionally be granted permission to **edit the training and model code itself** to fix precision/casting issues it discovers during training. This mode is opt-in — by default the agent must ONLY control behavior via CLI arguments.

### When to Intervene

The agent should consider code intervention when:

- **NaN/Inf loss detected** — likely a casting or autocast bug
- **Loss spikes that don't correlate with precision schedule changes** — may indicate a code-level issue
- **Quant stats show anomalous patterns** across all layers (not just specific ones) — suggests a systematic bug
- **Training crashes** with CUDA errors, dtype mismatches, or TE assertion failures

### What the Agent Can Edit

Files the agent is allowed to modify:

- `opengenome_modeling_llama_te.py` — autocast contexts, `get_layer_autocast()`, `set_recipes()`
- `train_fsdp2.py` — recipe creation, FSDP wrapping, precision schedule application
- `quantization.py` — layer precision resolution logic

### Example Interventions

1. **Fix autocast nesting**: If the agent detects that FP8 layers have a double-nested autocast (outer FP8 + inner FP8), it can fix `get_layer_autocast()` to return `nullcontext()` for FP8 layers.

2. **Fix recipe serialization**: If recipes are lost after FSDP wrapping, the agent can add/fix the `set_recipes()` call.

3. **Fix dtype casting**: If embeddings or the lm_head are accidentally running in FP8, the agent can verify and fix the `te.autocast(enabled=False)` wrappers.

4. **Fix loss computation precision**: If loss is computed in FP8 (causing NaN), the agent can ensure the loss function runs in BF16/FP32.

### Guardrails

- The agent must **log every code change** to `history.json` with a diff and rationale.
- Code changes must be **committed to a branch** (not just edited in place) so they can be reviewed.
- The agent should **never modify the dataset, tokenizer, or optimizer code** — only precision-related code paths.
- After any code edit, the agent must **restart training from the LKG checkpoint** (not continue from current state).

______________________________________________________________________

## Agent Workspace & Artifacts

All agent output must be saved under: `$WORKSPACE_ROOT/<run_name>/`

The agent creates `<run_name>` ONCE at startup using the format: `<strategy>_<YYYYMMDD_HHMMSS>`
This value is computed once and stored — it does NOT change across relaunches within the same session. It is used for the checkpoint directory name, workspace directory, WandB group name, Lepton job name, and results folder — all the same value for easy cross-referencing.
Examples:

- `ends_in_20260317_143000`
- `tail_in_20260317_160000`
- `research_guided_20260318_091500`

The directory layout:

```
$CHECKPOINT_ROOT/<run_name>/    # model checkpoints (set checkpoint.ckpt_dir here)
  train_fsdp2/step_<N>/         # auto-created by the training script

$WORKSPACE_ROOT/<run_name>/
  logs/                         # training stdout/stderr logs from each launch
  quant_stats/                  # quantization stats output (research_guided only)
  configs/                      # copy of every config/CLI invocation used per segment
  graphs/                       # any plots the agent generates (perplexity vs baseline, throughput over time, etc.)
  history.json                  # structured log of all agent decisions:
                                #   - every check-in result (step, baseline ppl, current ppl, pass/fail)
                                #   - every demotion event (which layers, reason)
                                #   - every recovery event (rolled back to which step, new schedule)
                                #   - throughput at each check-in
  state.json                    # agent state for crash recovery
  report.md                     # human-readable summary maintained by the agent (see below)
```

### history.json

```json
[
  {
    "step": 100,
    "current_loss": 1.82,
    "baseline_loss": 1.80,
    "current_ppl": 6.15,
    "baseline_ppl": 6.04,
    "diff": 0.11,
    "allowed_delta": 0.302,
    "passed": true,
    "action": "continue",
    "fp8_layers": [1,2,3,...,32],
    "throughput": 62000.0,
    "timestamp": "2026-03-17T14:35:00"
  },
  {
    "step": 200,
    "current_loss": 1.49,
    "baseline_loss": 1.42,
    "current_ppl": 4.45,
    "baseline_ppl": 4.16,
    "diff": 0.29,
    "allowed_delta": 0.208,
    "passed": false,
    "action": "demote_layers",
    "demoted": [1, 32],
    "fp8_layers": [2,3,...,31],
    "rollback_to_step": 100,
    "throughput": 61500.0,
    "timestamp": "2026-03-17T15:10:00"
  }
]
```

### state.json (crash recovery)

```json
{
  "current_step": 200,
  "lkg_step": 100,
  "promotion_round": 1,
  "layer_precision": {"1": "bf16", "2": "fp8", ..., "31": "fp8", "32": "bf16"},
  "fp8_layers": [2,3,4,...,31],
  "run_name": "ends_in_20260317_143000"
}
```

### report.md

The agent must maintain a `report.md` in the run directory. Update it after every check-in and at the end of training. It should contain:

- Run metadata: model name, num_layers, num_train_steps, tolerance, start time
- Final precision schedule: which layers ended in FP8 vs BF16
- Summary of all check-in results (table or list: step, baseline ppl, current ppl, pass/fail)
- Summary of all demotions and rollbacks
- Throughput comparison: average unpadded tokens/sec across the run vs. the BF16 baseline average
- Any observations from quant stats (research_guided only: layers with high underflow/overflow)
- Conclusion: how many layers stayed in FP8, overall accuracy vs baseline, throughput gain/loss

### Final Report

At the end of your session (or when training completes), you MUST produce a full markdown-based final report. Create a folder inside `$RESULTS_FOLDER` with the strategy name and date, and save all deliverables there:

```
$RESULTS_FOLDER/<strategy>_<YYYYMMDD_HHMMSS>/
  report.md                     # the final polished report
  graphs/                       # copies of all graphs referenced in the report
  history.json                  # copy of the run's history.json for reference
```

The folder name must be unique (strategy + timestamp) because these experiments will be run multiple times and results must not overwrite each other.

The final report should be a polished, self-contained document that includes everything from the run's `report.md` plus:

- A high-level executive summary (2-3 paragraphs) suitable for sharing with the team
- Graphs and visualizations (saved to graphs/ and referenced in the report), including at minimum:
  - **Perplexity over training steps**: baseline BF16 perplexity vs. this run's perplexity plotted on the same axes, with the tolerance band shaded. Mark any check-in failure points and demotion events on the plot.
  - **Throughput over training steps**: unpadded tokens/sec for this run vs. the BF16 baseline.
- Comparison against the BF16 baseline with concrete numbers
- Recommendations for next steps (e.g. try a different strategy, adjust tolerance, change layers_per_promotion)
- Any lessons learned or surprising findings during the run

The run-level `report.md` (inside the run directory) is a living document updated during training. The final report in `$RESULTS_FOLDER` is the cleaned-up deliverable produced at the end.

______________________________________________________________________

## Key Constraints

1. The agent is an **outer loop** around the training script. It configures precision, launches training, evaluates, and decides what to do next. Unless Milestone 3 (Code Intervention Mode) is explicitly enabled, the agent must ONLY control behavior via CLI arguments — do NOT modify the training script, config files, or any other source code.
2. All config (tolerance_pct, interval, baseline logfile path, layers_per_promotion) must be **passed via config** — not hardcoded.
3. All artifacts (checkpoints, logs, configs, graphs, history, report) go under `$WORKSPACE_ROOT/<run_name>/`. Final deliverables go under `$RESULTS_FOLDER/<strategy>_<YYYYMMDD_HHMMSS>/`.
4. WandB logging is desirable: training metrics should be published to WandB when available, but WandB failures should not block the agent loop.
5. The agent stops after reaching `$NUM_TRAIN_STEPS`. If all layers have been demoted to BF16 before that point, continue training in full BF16 for the remaining steps.
6. **Lepton-specific**: The agent must handle multi-node torchrun setup, NFS paths for checkpoints/data, and Lepton environment variables for distributed training.
