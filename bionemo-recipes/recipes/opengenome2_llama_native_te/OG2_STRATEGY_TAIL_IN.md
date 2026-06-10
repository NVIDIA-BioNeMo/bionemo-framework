# Strategy: `tail_in` — Demote from Tail Toward Head

## Overview

Demote transformer block layers from FP8 to BF16 starting at the output end (layer 32)
and working backward toward layer 1. Each failed check-in demotes `LAYERS_PER_PROMOTION`
layers (default 2). There is no fixed cap — the agent decides how many total layers to
demote based on check-in results.

**Rationale**: The final layers of a transformer (closest to the output projection / lm_head)
are typically most sensitive to quantization error. Quantization noise in these layers passes
through fewer corrective transformations before reaching the output, making them the primary
source of perplexity degradation. Demoting from the tail inward addresses the most sensitive
layers first.

______________________________________________________________________

## Run Variables

```
TOLERANCE_PCT        = 5.0
CHECKIN_INTERVAL     = 100
LAYERS_PER_PROMOTION = 2      # layers demoted per failed check-in
NUM_LAYERS           = 32     # OG2-7B transformer block layers (1-indexed: 1..32)
```

______________________________________________________________________

## Demotion Order

Layers are demoted from the output end inward, in batches of `LAYERS_PER_PROMOTION`:

| Round | Layers Demoted to BF16 | Remaining FP8 Layers | FP8 Count |
| ----- | ---------------------- | -------------------- | --------- |
| Start | (none)                 | 1-32                 | 32        |
| 1     | 32, 31                 | 1-30                 | 30        |
| 2     | 30, 29                 | 1-28                 | 28        |
| 3     | 28, 27                 | 1-26                 | 26        |
| 4     | 26, 25                 | 1-24                 | 24        |
| 5     | 24, 23                 | 1-22                 | 22        |
| 6     | 22, 21                 | 1-20                 | 20        |
| 7     | 20, 19                 | 1-18                 | 18        |
| 8     | 18, 17                 | 1-16                 | 16        |
| 9     | 16, 15                 | 1-14                 | 14        |
| 10    | 14, 13                 | 1-12                 | 12        |
| 11    | 12, 11                 | 1-10                 | 10        |
| 12    | 10, 9                  | 1-8                  | 8         |
| 13    | 8, 7                   | 1-6                  | 6         |
| 14    | 6, 5                   | 1-4                  | 4         |
| 15    | 4, 3                   | 1-2                  | 2         |
| 16    | 2, 1                   | (none)               | 0         |

After round 16, all layers are BF16. Training continues in full BF16 for remaining steps.

The agent is not required to demote all the way to BF16. It demotes only when check-ins fail,
and stops demoting when check-ins pass. Most runs will stabilize well before reaching round 16.

______________________________________________________________________

## Pseudocode

```python
# --- Initialization ---

layer_precision = {i: "fp8" for i in range(1, NUM_LAYERS + 1)}
tail_ptr = NUM_LAYERS  # next layer to demote (starts at 32)
lkg_step = 0
promotion_round = 0
history = []

# --- Helpers ---


def get_fp8_layers():
    """Return sorted list of 1-indexed layers currently in FP8."""
    return sorted(k for k, v in layer_precision.items() if v == "fp8")


def demote_next_batch():
    """Demote LAYERS_PER_PROMOTION layers from the tail."""
    global tail_ptr, promotion_round
    demoted = []
    for _ in range(LAYERS_PER_PROMOTION):
        if tail_ptr < 1:
            break
        if layer_precision[tail_ptr] == "fp8":
            layer_precision[tail_ptr] = "bf16"
            demoted.append(tail_ptr)
        tail_ptr -= 1
    promotion_round += 1
    return demoted


# --- Main Agent Loop ---

current_step = 0

while current_step < NUM_TRAIN_STEPS:
    fp8_layers = get_fp8_layers()
    resume = current_step > 0

    cmd = build_torchrun_command(
        fp8_layers=fp8_layers,
        num_train_steps=NUM_TRAIN_STEPS,
        resume_from_checkpoint=resume,
        ckpt_dir=f"{WORKSPACE_ROOT}/{run_name}/checkpoints",
    )

    launch_training(cmd)

    next_checkin = current_step + CHECKIN_INTERVAL
    metrics = wait_for_step(next_checkin)

    # --- Check-in evaluation ---
    # Baseline lookup — fail hard if step is missing
    baseline_key = f"step_{next_checkin}"
    if baseline_key not in baseline:
        raise RuntimeError(
            f"Baseline logfile missing entry for {baseline_key}. "
            f"CHECKIN_INTERVAL={CHECKIN_INTERVAL} must align with baseline logfile steps."
        )

    import math

    current_ppl = math.exp(metrics["loss"])
    baseline_ppl = baseline[baseline_key]["perplexity"]
    allowed_delta = baseline_ppl * (TOLERANCE_PCT / 100)
    diff = current_ppl - baseline_ppl
    passed = diff <= allowed_delta

    entry = {
        "step": next_checkin,
        "baseline_ppl": baseline_ppl,
        "current_ppl": current_ppl,
        "diff": diff,
        "allowed_delta": allowed_delta,
        "passed": passed,
        "fp8_layers": fp8_layers,
        "throughput": metrics["unpadded_tokens_per_second_per_gpu"],
    }

    if passed:
        entry["action"] = "continue"
        lkg_step = next_checkin
        current_step = next_checkin
        history.append(entry)
        save_state()
        update_report()
        continue

    # --- FAIL ---
    stop_training()

    if not get_fp8_layers():
        # All layers already BF16 — nothing more to demote
        entry["action"] = "continue_bf16_exhausted"
        lkg_step = next_checkin
        current_step = next_checkin
        history.append(entry)
        save_state()
        update_report()
        continue

    demoted = demote_next_batch()
    entry["action"] = "demote_layers"
    entry["demoted"] = demoted
    entry["rollback_to_step"] = lkg_step
    history.append(entry)

    delete_checkpoints_after(lkg_step)
    current_step = lkg_step
    save_state()
    update_report()

# --- Training Complete ---
generate_final_report()
```

______________________________________________________________________

## Check-in Decision Tree

```
[Check-in at step N]
        |
        v
  baseline_key = f"step_{N}"
  baseline_key in baseline?
     /         \
   YES          NO → FATAL: misaligned CHECKIN_INTERVAL
    |
    v
  current_ppl = exp(loss)
  baseline_ppl = baseline[step_N]
  allowed_delta = baseline_ppl * (TOLERANCE_PCT / 100)
        |
  diff = current_ppl - baseline_ppl
        |
        v
  diff <= allowed_delta?
     /         \
   YES          NO
    |            |
    v            v
  PASS         FAIL
  LKG = N      |
  continue     Any FP8 layers left?
               /         \
             YES          NO
              |            |
              v            v
         demote tail_in   continue (all BF16)
         rollback to LKG  LKG = N
         resume from LKG
```

______________________________________________________________________

## Worked Example (32-layer OG2-7B)

### Setup

- `TOLERANCE_PCT = 5.0`
- `CHECKIN_INTERVAL = 100`
- `LAYERS_PER_PROMOTION = 2`

### Timeline

**Step 100 — Check-in 1**

- Baseline ppl: 6.04
- Current ppl: 6.15
- Allowed delta: 6.04 * 0.05 = 0.302
- Diff: 0.11 < 0.302 → **PASS**
- FP8 layers: [1..32] (all 32)

**Step 200 — Check-in 2**

- Baseline ppl: 4.16
- Current ppl: 4.45
- Allowed delta: 4.16 * 0.05 = 0.208
- Diff: 0.29 > 0.208 → **FAIL**
- Demote: layers 32, 31 (tail batch 1)
- Rollback to step 100 checkpoint
- FP8 layers: [1..30] (30 layers)

**Step 200 (retry) — Check-in 2b**

- Baseline ppl: 4.16
- Current ppl: 4.30
- Allowed delta: 0.208
- Diff: 0.14 < 0.208 → **PASS**
- FP8 layers: [1..30] (30 layers)

**Step 300 — Check-in 3**

- Baseline ppl: 4.09
- Current ppl: 4.35
- Allowed delta: 4.09 * 0.05 = 0.205
- Diff: 0.26 > 0.205 → **FAIL**
- Demote: layers 30, 29 (tail batch 2)
- Rollback to step 200 checkpoint
- FP8 layers: [1..28] (28 layers)

**Step 300 (retry) — Check-in 3b**

- Baseline ppl: 4.09
- Current ppl: 4.22
- Allowed delta: 0.205
- Diff: 0.13 < 0.205 → **PASS**
- FP8 layers: [1..28] (28 layers)

**Step 400 — Check-in 4**

- Baseline ppl: 3.95
- Current ppl: 4.05
- Allowed delta: 3.95 * 0.05 = 0.198
- Diff: 0.10 < 0.198 → **PASS**
- FP8 layers: [1..28] (28 layers, stable)

Training proceeds with 28 of 32 layers in FP8 — 87.5% quantized coverage.

______________________________________________________________________

## Comparison with `ends_in`

| Aspect             | `ends_in`                                              | `tail_in`                                         |
| ------------------ | ------------------------------------------------------ | ------------------------------------------------- |
| Demotion direction | Both ends inward                                       | Output end toward head only                       |
| Maximum rounds     | 16 (all layers)                                        | 16 (all layers)                                   |
| Minimum FP8        | 0                                                      | 0                                                 |
| Cap                | None (demotes until all BF16 or passes)                | None (demotes until all BF16 or passes)           |
| Best for           | Unknown sensitivity distribution                       | Output-sensitive models                           |
| Risk               | Over-demotion at input end if only output is sensitive | Under-demotion if input layers are also sensitive |

### When to Choose `tail_in`

- Prior evidence suggests the output-end layers are the primary source of quantization divergence.
- You want to preserve input-end FP8 layers as long as possible.
- The model shows stronger sensitivity near the output projection than near the embedding.

### When to Choose `ends_in`

- You have no prior knowledge of which layers are most sensitive.
- The model shows sensitivity at both the input and output ends.
- You want symmetric coverage: both ends get demoted at the same rate.

______________________________________________________________________

## Notes

- There is **no fixed cap** on demotions. The agent keeps demoting from the tail as long as check-ins fail.
- The `tail_ptr` never moves forward — demotions are permanent.
- The agent always resumes from the LKG checkpoint, never from the failed step.
- If the baseline logfile is missing an entry for the check-in step, the agent must **fail hard** and tell the user to fix `CHECKIN_INTERVAL` alignment.
