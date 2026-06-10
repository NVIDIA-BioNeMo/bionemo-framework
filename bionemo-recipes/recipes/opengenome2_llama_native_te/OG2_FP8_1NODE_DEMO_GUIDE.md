# FP8 Precision Agent — OpenGenome2 7B (1-Node Demo)

## RUN VARIABLES

```
TOLERANCE_PCT          = 3.0
BASELINE_LOGFILE       = ./baseline_bf16_1node.json
NUM_TRAIN_STEPS        = 6000
CHECKIN_INTERVAL       = 100
LAYERS_PER_PROMOTION   = 2
NUM_LAYERS             = 32
INITIAL_PRECISION      = bf16
PROMOTION_STRATEGY     = gradual
WORKSPACE_ROOT         = /data/savithas/agent_runs/demo_1node
CHECKPOINT_ROOT        = /data/savithas/checkpoints

# Training script & model
TRAINING_SCRIPT        = train_fsdp2.py
CONFIG_NAME            = og2_7b_bf16_1k_from_5k
NPROC_PER_NODE         = 8
MICRO_BATCH_SIZE       = 1
GRAD_ACC_STEPS         = 8
WANDB_PROJECT          = llama3-metagenome-7b
RESULTS_FOLDER         = /data/savithas/agent_runs/demo_1node/results
WARMUP_STEPS           = 500
```

______________________________________________________________________

## Project Overview

You are an autonomous agent managing a pretraining loop for an OpenGenome2 7B
Transformer model (Llama 3.1-8B architecture via BioNeMo/TransformerEngine).
The model resumes from a BF16 checkpoint at step 5000. Your objective is to
**maximize the number of layers running in FP8** while keeping pretraining
accuracy within tolerance of a BF16 baseline. You control a per-layer precision
schedule and monitor training metrics at regular intervals. Training runs from
step 5001 to step 6000 (1000 steps total).

### Architecture Summary

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

### Model Details

- **Architecture**: Llama 3.1-8B (7B params)
- **Layers**: 32 transformer block layers (1-32, 1-indexed)
- **Precision**: BF16 compute, FP32 master weights
- **Dataset**: OpenGenome2 metagenomes (JSON files on NFS)
- **GBS**: 64 (mbs=1 x grad_acc=8 x 8 GPUs)
- **Checkpoint**: Resumes from BF16 step 5000 at
  `/data/savithas/checkpoints/og2-7b-fp32mw-orig-ws-w1-b50k`

______________________________________________________________________

## Metrics

### Accuracy Metric: Loss (lower is better)

At every check-in, the agent compares current loss against the BF16 baseline
loss at the same step using a relative tolerance. A check-in **passes** if:
`current_loss - baseline_loss <= baseline_loss * (TOLERANCE_PCT / 100)`.

### Performance Metric: Unpadded Tokens Per Second (higher is better)

Measures actual training throughput. Throughput changes do NOT trigger
rollbacks — they are logged for informational purposes only.

______________________________________________________________________

## BF16 Warmup Phase

Since we resume from a BF16 checkpoint at step 5000, the model needs a warmup
period. The warmup phase covers steps 5001-5500 (the first 500 steps after
resume). During this phase all 32 layers run in BF16. The agent does NOT
perform check-ins or promotions during the warmup phase.

**Timeline:**

- Steps 5001-5500: BF16 warmup (all layers BF16). `fp8_config.enabled=False`.
  No agent intervention.
- Step 5500: Warmup ends. Training continues uninterrupted — agent begins
  monitoring at step 5600 (first check-in).
- Steps 5500-6000: Active phase. Agent monitors check-ins and adjusts layers.

______________________________________________________________________

## Base Training Command

Every launch uses this exact template. Only the three AGENT CONTROLS fields
change between launches — everything else is hardcoded.

```bash
cd /data/savithas/bionemo-framework/bionemo-recipes/recipes/opengenome2_llama_native_te

torchrun --nproc_per_node=8 train_fsdp2.py \
  --config-name og2_7b_bf16_1k_from_5k \
  dataset.micro_batch_size=1 \
  dataset.buffer_size=10000 \
  dataset.num_workers=8 \
  dataset.use_stateful_dataloader=true \
  num_train_steps=6000 \
  grad_acc_steps=8 \
  checkpoint.ckpt_dir=/data/savithas/checkpoints/<run_name> \
  checkpoint.save_every_n_steps=100 \
  checkpoint.resume_from_checkpoint=true \
  checkpoint.max_checkpoints=4 \
  checkpoint.save_final_model=true \
  checkpoint.async_save=false \
  logger.frequency=1 \
  fp8_config.enabled=<true|false> \
  fp8_config.fp8_recipe=transformer_engine.common.recipe.Float8BlockScaling \
  fp8_config.fp8_format=E4M3 \
  fp8_layers='<LAYER_LIST>' \
  wandb.project=llama3-metagenome-7b \
  wandb.name=<run_name> \
  +wandb.id=<run_name> \
  +wandb.resume=allow \
  hydra.run.dir=/data/savithas/agent_runs/demo_1node/<run_name>/hydra_outputs
```

**AGENT CONTROLS (change between launches):**

- `fp8_config.enabled` — `false` during BF16 warmup; `true` once any layers
  are in FP8
- `fp8_layers` — `'[]'` during warmup; e.g. `'[16,17]'` after first expansion

**FIXED (never change between launches):**

- `dataset.micro_batch_size=1` — always 1
- `dataset.buffer_size=10000` — always 10k
- `dataset.num_workers=8` — always 8
- `dataset.use_stateful_dataloader=true` — always true (see Data Integrity section below)
- `num_train_steps=6000` — always 6000
- `grad_acc_steps=8` — always 8 (GBS = 1 × 8 × 8 GPUs = 64)
- `checkpoint.ckpt_dir` — same directory for entire session
- `checkpoint.save_every_n_steps=100` — matches CHECKIN_INTERVAL
- `checkpoint.resume_from_checkpoint=true` — always true
- `checkpoint.async_save=false` — sync saves for reliability
- `wandb.name` — computed once at session start, never changes
- `+wandb.id` — same as `wandb.name` (WandB resumes by ID, not name)
- `+wandb.resume=allow` — resumes the same WandB run on relaunch
- `wandb.project=llama3-metagenome-7b` — fixed

______________________________________________________________________

## IMPORTANT: Initial Checkpoint Setup

The agent resumes from a pre-existing BF16 checkpoint. Before the first launch:

1. Create a unique checkpoint directory for this agent session:
   ```bash
   mkdir -p $CHECKPOINT_ROOT/<run_name>/train_fsdp2
   ```
2. Symlink the external 5k checkpoint:
   ```bash
   ln -s /data/savithas/checkpoints/og2-7b-fp32mw-orig-ws-w1-b50k/train_fsdp2/step_5000 \
     $CHECKPOINT_ROOT/<run_name>/train_fsdp2/step_5000
   ```
3. Verify:
   ```bash
   ls $CHECKPOINT_ROOT/<run_name>/train_fsdp2/step_5000/.metadata
   ```

The `<run_name>` must be unique. Use format: `gradual_fp8_<YYYYMMDD_HHMMSS>`.

______________________________________________________________________

## Layer Precision Schedule

The agent maintains a per-layer precision map for 32 transformer block layers
(1-32, 1-indexed).

During warmup (steps 5001-5500): all BF16.
After warmup (step 5500+): the agent uses the `gradual` strategy.

### Strategy: gradual

Post-warmup starting state: `fp8_layers=[]` (all BF16).

**On pass** — add `LAYERS_PER_PROMOTION` layers to FP8, expanding from the
center outward:

| Round | Layers Added to FP8 | FP8 Layers | FP8 Count |
| ----- | ------------------- | ---------- | --------- |
| Start | (none)              | (none)     | 0         |
| 1     | 16, 17              | 16-17      | 2         |
| 2     | 15, 18              | 15-18      | 4         |
| 3     | 14, 19              | 14-19      | 6         |
| 4     | 13, 20              | 13-20      | 8         |
| 5     | 12, 21              | 12-21      | 10        |
| 6     | 11, 22              | 11-22      | 12        |
| 7     | 10, 23              | 10-23      | 14        |
| 8     | 9, 24               | 9-24       | 16        |
| 9     | 8, 25               | 8-25       | 18        |
| 10    | 7, 26               | 7-26       | 20        |
| 11    | 6, 27               | 6-27       | 22        |
| 12    | 5, 28               | 5-28       | 24        |
| 13    | 4, 29               | 4-29       | 26        |
| 14    | 3, 30               | 3-30       | 28        |
| 15    | 2, 31               | 2-31       | 30        |
| 16    | 1, 32               | 1-32       | 32        |

**On fail** — remove the most recently added FP8 layers (demote back to BF16),
roll back to LKG checkpoint, and relaunch.

**Rationale**: Middle layers are the most tolerant to quantization. Edge layers
(closest to embedding input and output projection) are most sensitive. Expanding
from the center outward tests the safest layers first.

______________________________________________________________________

## Check-ins

Check-ins begin at step 5600 (first `CHECKIN_INTERVAL` after warmup ends at
5500).

Every `CHECKIN_INTERVAL` steps:

1. Record current loss from `wandb-history.jsonl`
2. Look up baseline loss for this step from `BASELINE_LOGFILE`
3. Compute tolerance: `allowed_delta = baseline_loss * (TOLERANCE_PCT / 100)`
4. **Pass** if: `current_loss - baseline_loss <= allowed_delta`
5. Log throughput (unpadded_tokens_per_second_per_gpu)
6. **Pass** → kill training, expand FP8 (add next center-outward pair), relaunch
7. **Fail** → kill training, demote last-added layers, rollback to LKG, relaunch

**IMPORTANT**: `CHECKIN_INTERVAL` must align with baseline logfile step
intervals. If the baseline has no entry for a check-in step, fail hard and
report the error.

### Metric Retrieval (WandB Local Files)

The agent monitors WandB's local log files:

```
<training_script_dir>/wandb/latest-run/files/wandb-history.jsonl
```

Each line is a JSON object:

```json
{"train/global_step": 5600, "train/loss": 1.19, "train/unpadded_tokens_per_second_per_gpu": 22000, ...}
```

**Agent monitoring loop:**

1. Launch training as a background process.
2. Poll `wandb-history.jsonl` periodically (every 30 seconds).
3. When `train/global_step` matches a check-in step, parse the metrics.
4. Compare loss against baseline (see Check-ins above).
5. Pass/fail → take appropriate action.

If multiple check-in steps appear between polls, process in order (lowest step
first). Kill on the FIRST failure.

### WandB Sync Before Killing

Before killing training, sync wandb data:

```bash
wandb sync <training_script_dir>/wandb/latest-run/
```

______________________________________________________________________

## Checkpointing & Resume

Checkpoints saved at `<ckpt_dir>/train_fsdp2/step_<N>/`. The training script
automatically finds the latest checkpoint and resumes.

**What gets restored:** model weights, optimizer state, LR scheduler, step
counter, epoch counter, dataloader position (`use_stateful_dataloader=true`).

All state — including where the dataloader left off in the dataset — is
restored from checkpoint.

**Key behavior:**

- `num_train_steps` is an absolute target (6000), not relative.
- Resuming at step 5400 with `num_train_steps=6000` trains steps 5401-6000.
- `checkpoint.resume_from_checkpoint=true` always — auto-finds latest checkpoint.

**CRITICAL — Data Integrity on Relaunch:**

The agent kills and relaunches training at every expansion (pass) and every
rollback (fail). With `dataset.use_stateful_dataloader=true`, the dataloader
position is saved in each checkpoint and restored on resume, so the model sees
each training batch exactly once — the same as a continuous baseline run. If
`use_stateful_dataloader` is NOT enabled, every relaunch resets the dataloader
to the start of the dataset. This causes the model to re-train on early batches
multiple times, artificially lowering training loss (overfitting to repeated
data) and invalidating comparisons against the BF16 baseline. The agent MUST
verify that `dataset.use_stateful_dataloader=true` is set in the training
command.

**CRITICAL — Checkpoint Safety Rules:**

- NEVER delete the checkpoint directory itself (`$CHECKPOINT_ROOT/<run_name>/`).
  Only delete individual `step_<N>/` subdirectories inside it.
- NEVER use `rm -rf` on any parent directory. Only `rm -rf` specific
  `step_<N>/` subdirectories that are newer than the LKG.
- Before deleting, always list the contents of the checkpoint directory first to
  confirm which checkpoints exist and which is the LKG.
- When in doubt, do NOT delete — relaunch and let the training script skip the
  corrupt checkpoint automatically.

### Recovery on Failed Check-in

1. Kill the current training process.
2. Delete any `step_<N>/` checkpoint newer than the LKG.
3. Remove the most recently added FP8 layers (demote to BF16).
4. Relaunch with updated precision schedule.

The agent discards all progress since the last successful check-in.

______________________________________________________________________

## Agent Workspace & Artifacts

All output under: `$WORKSPACE_ROOT/<run_name>/`

```
$CHECKPOINT_ROOT/<run_name>/           # model checkpoints
  train_fsdp2/step_<N>/

$WORKSPACE_ROOT/<run_name>/
  logs/                                # training stdout/stderr
  configs/                             # CLI invocations per segment
  graphs/                              # plots (perplexity/throughput vs baseline)
  history.json                         # structured log of all decisions
  state.json                           # agent state for crash recovery
  report.md                            # human-readable summary
```

### history.json

```json
[
  {
    "step": 5600,
    "current_loss": 1.19,
    "baseline_loss": 1.18,
    "diff": 0.01,
    "allowed_delta": 0.035,
    "passed": true,
    "action": "expand_fp8",
    "added_layers": [16, 17],
    "fp8_layers": [16, 17],
    "throughput": 22000.0,
    "timestamp": "2026-03-19T10:00:00"
  }
]
```

### state.json

```json
{
  "current_step": 5600,
  "lkg_step": 5600,
  "expansion_round": 1,
  "fp8_layers": [16, 17],
  "run_name": "gradual_fp8_20260319_100000",
  "warmup_complete": true
}
```

### report.md

Update after every check-in and at end of training. Include:

- Run metadata (model, layers, steps, tolerance, start time)
- Final precision schedule (which layers in FP8 vs BF16)
- Summary of all check-in results (table: step, baseline loss, current loss,
  pass/fail, action)
- Throughput comparison vs BF16 baseline
- Conclusion: how many layers ended in FP8, accuracy vs baseline

______________________________________________________________________

## WandB Run Naming & Resume

`wandb.name` and `+wandb.id` are both set to `<run_name>` (e.g.
`gradual_fp8_20260319_100000`). These are computed ONCE at session start and
NEVER change. WandB resumes by **run ID** (not name), so `+wandb.id` is
required. Combined with `+wandb.resume=allow`, every relaunch appends to the
same WandB run, producing a single continuous curve in the dashboard.

This means all training segments (warmup, expansions, rollbacks) appear as one
continuous line in WandB, making it easy to see the full training trajectory
including any loss spikes from precision changes.

______________________________________________________________________

## Key Constraints

1. The agent ONLY controls behavior via CLI arguments — do NOT modify training
   scripts or config files.
2. All artifacts go under `$WORKSPACE_ROOT/<run_name>/`. Checkpoints go under
   `$CHECKPOINT_ROOT/<run_name>/`.
3. The agent stops after reaching step 6000.
4. NEVER delete the checkpoint directory itself — only individual `step_<N>/`
   subdirectories.
5. Before deleting any checkpoint, always list its contents first.
6. If all 32 layers are successfully expanded to FP8, continue training in full
   FP8 for the remaining steps.
