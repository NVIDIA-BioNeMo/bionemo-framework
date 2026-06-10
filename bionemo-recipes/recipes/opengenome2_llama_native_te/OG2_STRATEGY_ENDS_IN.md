# Strategy: `ends_in` — Demote from Both Ends Inward

## Overview

Demote transformer block layers from FP8 to BF16 starting at both the outermost positions
(layer 1 and layer 32) and working inward toward the middle. Each failed check-in demotes
`LAYERS_PER_PROMOTION` layers (default 2 — one from each end).

**Rationale**: Edge layers sit closest to the embedding input (layer 1) and the output
projection/lm_head (layer 32). These positions are most sensitive to quantization error
because:

- Layer 1 receives freshly embedded token representations that have not been refined by
  deeper layers — small rounding errors propagate through the entire stack.
- Layer 32 feeds directly into the language modeling head — quantization noise here
  has the most direct impact on loss.

Middle layers are the most tolerant because they receive and produce already-compressed
representations and their errors are attenuated by subsequent layers.

______________________________________________________________________

## Run Variables

```
TOLERANCE_PCT        = 5.0
CHECKIN_INTERVAL     = 100
LAYERS_PER_PROMOTION = 2      # layers demoted per failed check-in (1 from each end)
NUM_LAYERS           = 32     # OG2-7B transformer block layers (1-indexed: 1..32)
```

______________________________________________________________________

## Demotion Order

Each failure round demotes 2 layers: 1 from the bottom and 1 from the top.

| Round | Layers Demoted to BF16 | Remaining FP8 Layers | FP8 Count |
| ----- | ---------------------- | -------------------- | --------- |
| Start | (none)                 | 1-32                 | 32        |
| 1     | 1, 32                  | 2-31                 | 30        |
| 2     | 2, 31                  | 3-30                 | 28        |
| 3     | 3, 30                  | 4-29                 | 26        |
| 4     | 4, 29                  | 5-28                 | 24        |
| 5     | 5, 28                  | 6-27                 | 22        |
| 6     | 6, 27                  | 7-26                 | 20        |
| 7     | 7, 26                  | 8-25                 | 18        |
| 8     | 8, 25                  | 9-24                 | 16        |
| 9     | 9, 24                  | 10-23                | 14        |
| 10    | 10, 23                 | 11-22                | 12        |
| 11    | 11, 22                 | 12-21                | 10        |
| 12    | 12, 21                 | 13-20                | 8         |
| 13    | 13, 20                 | 14-19                | 6         |
| 14    | 14, 19                 | 15-18                | 4         |
| 15    | 15, 18                 | 16-17                | 2         |
| 16    | 16, 17                 | (none)               | 0         |

After round 16, all layers are BF16. Training continues in full BF16 for remaining steps.

______________________________________________________________________

## Pseudocode

```python
# --- Initialization ---

layer_precision = {i: "fp8" for i in range(1, NUM_LAYERS + 1)}
bottom_ptr = 1  # next layer to demote from bottom
top_ptr = NUM_LAYERS  # next layer to demote from top
lkg_step = 0  # last known good checkpoint step
promotion_round = 0
history = []

# --- Helpers ---


def get_fp8_layers():
    """Return sorted list of 1-indexed layers currently in FP8."""
    return sorted(k for k, v in layer_precision.items() if v == "fp8")


def demote_next_batch():
    """Demote LAYERS_PER_PROMOTION layers using ends_in pattern."""
    global bottom_ptr, top_ptr, promotion_round
    demoted = []
    pairs_to_demote = LAYERS_PER_PROMOTION // 2  # 2 layers = 1 pair
    for _ in range(pairs_to_demote):
        if bottom_ptr > top_ptr:
            break
        # Demote from bottom
        if layer_precision[bottom_ptr] == "fp8":
            layer_precision[bottom_ptr] = "bf16"
            demoted.append(bottom_ptr)
        bottom_ptr += 1
        # Demote from top
        if bottom_ptr <= top_ptr and layer_precision[top_ptr] == "fp8":
            layer_precision[top_ptr] = "bf16"
            demoted.append(top_ptr)
        top_ptr -= 1
    promotion_round += 1
    return demoted


# --- Main Agent Loop ---

current_step = 0

while current_step < NUM_TRAIN_STEPS:
    fp8_layers = get_fp8_layers()
    resume = current_step > 0

    # Build the torchrun command with current fp8_layers
    cmd = build_torchrun_command(
        fp8_layers=fp8_layers,
        num_train_steps=NUM_TRAIN_STEPS,
        resume_from_checkpoint=resume,
        ckpt_dir=f"{WORKSPACE_ROOT}/{run_name}/checkpoints",
    )

    # Launch training and monitor
    launch_training(cmd)

    # Wait for next check-in step
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
        # --- PASS ---
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

    # Rollback: delete checkpoints after LKG, resume from LKG
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
         demote ends_in   continue (all BF16)
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
- Demote: layers 1, 32
- Rollback to step 100 checkpoint
- FP8 layers: [2..31] (30 layers)

**Step 200 (retry) — Check-in 2b**

- Baseline ppl: 4.16
- Current ppl: 4.30
- Allowed delta: 0.208
- Diff: 0.14 < 0.208 → **PASS**
- FP8 layers: [2..31] (30 layers)

**Step 300 — Check-in 3**

- Baseline ppl: 4.09
- Current ppl: 4.20
- Allowed delta: 4.09 * 0.05 = 0.205
- Diff: 0.11 < 0.205 → **PASS**
- FP8 layers: [2..31] (30 layers, stable)

Training proceeds with 30 of 32 layers in FP8 — 93.75% quantized coverage.

______________________________________________________________________

## Notes

- Each round demotes exactly `LAYERS_PER_PROMOTION` layers (default 2).
- For odd `LAYERS_PER_PROMOTION`, the extra layer comes from the bottom side.
- The pointers (`bottom_ptr`, `top_ptr`) never move backward — demotions are permanent.
- After round 16, `bottom_ptr > top_ptr` and `get_fp8_layers()` returns `[]`.
- The agent always resumes from the LKG checkpoint, never from the failed step.
- If the baseline logfile is missing an entry for the check-in step, the agent must **fail hard** and tell the user to fix `CHECKIN_INTERVAL` alignment.
