---
name: operate-nemo-rl-phage
description: Use when launching, monitoring, resuming, relaunching, or selecting checkpoints from a NeMo-RL Evo2 prokaryotic-phage optimization run.
---

# Operate NeMo-RL Phage Training

Run the approved RL contract through adapt-phage-execution. Select checkpoints from declared comparable biological validation, not last step or training reward.

## Prepare

1. Search current and configured external result roots for compatible SFT runs. If choices exist, show evidence and ask reuse versus new.
2. Resolve exact SFT and prompt lineage using [the lineage contract](references/lineage-contract.md). Refuse ambiguous checkpoint, stage, split, base model, prompt manifest, or selection rationale.
3. Verify RL_OBJECTIVES.yaml, fixed validation-generation manifest, filter profile, policy/KL identities, runtime contract, hardware/disk, and execution plan. Resolve commands with [command resolution](references/command-resolution.md) and record them through [action traceability](references/action-traceability.md).
4. Predeclare primary metric, tie breakers, uncertainty/minimum change, comparable-event rules, validation cadence, maximum 500 steps, patience/rebound/collapse rules, and restart semantics.

## Launch and observe

- Create a new attempt with exact ordered script, script/command/config hashes, executor, host/job identity and URL, start/end/exit, logs, output hashes, environment, source state, lineage, and stable telemetry IDs.
- Monitor disk logs/config/checkpoints/source/free space and process/scheduler state. Use local TensorBoard by default; query W&B only when configured. Follow [the monitoring contract](references/monitoring-contract.md) for due-gated cadence, comparability, stopping, and rebound.
- Track primary validation and uncertainty, raw/full-QC and deduplicated passes, rewards, hard filters/OR branches, diversity, KL, entropy, lengths, invalids, throughput, NaN/Inf, OOM, and checkpoint integrity.
- Promote each comparable new best while preserving best/latest. Append observations/decisions; never rewrite history.
- Use [the historical case](references/historical-case.md) only for replication or interpretation of that run.

## Baselines and restarts

- Fresh RL initializes policy and KL reference from selected non-RL SFT.
- Exact resume restores optimizer/scheduler/RNG and retains original KL reference and prompt/validation manifests.
- A weights-only restart remains KL-anchored to selected SFT and is a new attempt.
- Prior RL weights are allowed only for explicit stagewise objective change. Record why; never silently make prior RL the new KL baseline.

## Report

Write attempt SUMMARY.md, OUTPUTS.yaml, and final monitor state with exact SFT-stage and prompt lineage, RL checkpoint/hash, max/stop/best steps, validation/filter manifests, metric uncertainty, tie breakers, stop/rebound rationale, stability, and risks. Include jump links to lineage, telemetry, checkpoint, and QC artifacts for root promotion. Distinguish selected checkpoint from later evidence-collection stop step.

Do not claim completion from a submitted job. Continue through the adapter monitor loop, or produce executable scripts plus a precise handoff and resume condition.

