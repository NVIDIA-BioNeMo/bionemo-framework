---
name: bionemo-phage-design-operate-nemo-rl
description: Use when launching, monitoring, resuming, relaunching, or selecting checkpoints from a NeMo-RL Evo2 phage optimization run.
---

# Operate NeMo-RL Phage Training

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Run the approved RL contract through bionemo-phage-design-adapt-execution, including its agent-session continuity contract. Select checkpoints from declared comparable biological validation, not last step or training reward.

## Prepare

1. Unless the user requires fresh/no reuse, search current and configured external result roots for compatible SFT runs. Use an already approved choice; otherwise ask only when candidates have materially different scientific consequences.
2. Resolve exact SFT and calibrated prompt/sampling lineage using [the lineage contract](references/lineage-contract.md). Require byte/token compatibility evidence, the training mixture, and the independent fixed validation manifest from bionemo-phage-design-calibrate-rl-sampling. Inspect durable artifacts and repair nonsemantic omissions; stop only when a biologically or reproducibly material ambiguity remains.
3. Verify `planning/DESIGN_SPEC.yaml`, RL_OBJECTIVES.yaml, fixed validation-generation manifest, filter profile, policy/KL identities, runtime contract, hardware/disk, and execution plan. For adapted therapeutic work, require the resolved intended-use applicability block and enabled design-relevant therapeutic-quality components; base-paper replication adds them only when requested. Reject a whole-genome project whose resolved generation or scoring path silently restricts mutation or evaluation to a locus/module. Resolve commands with [command resolution](references/command-resolution.md) and record them through [action traceability](references/action-traceability.md).
4. Confirm the approved RL context covers prompt plus the agreed target-family completion bound. Use the collected-genome distribution by default; change it only for a user-approved objective to expand or contract genome length. Never inherit a fixed context from a historical case.
5. Derive and pin the exact runtime capabilities required by enabled objectives. Behavior-test that same environment on the target phage: every score must be measurable and its expected direction reasoned from its semantics, without requiring every score to be 1. Include failure/no-signal controls; every objective must be independently observable on its intended denominator.
6. Predeclare primary metric, tie breakers, uncertainty/minimum change, comparable-event rules, validation cadence, maximum 500 steps, patience/rebound/collapse rules, and restart semantics.

## Launch and observe

- Create a new attempt with exact ordered script, script/command/config hashes, executor, host/job identity and URL, start/end/exit, logs, output hashes, environment, source state, lineage, and stable telemetry IDs.
- During the full-step smoke, attribute wall time by stage. For dominant scorers, tune outer workers, nested tool threads, batching, accelerator support for the invoked operation, and unconsumed work; require identical-input scores and stable resources. Pin the fastest verified setting, allowing monitored extrapolation/restart.
- Record concurrency scope and effective-global total (`workers/actor × actors/spawning rank × spawning ranks/node × nodes`), not a bare count; use DP degree as the global floor when resources permit.
- Monitor global steps (distinguish per-epoch banners), logs/config/checkpoints/source/free space, and process/scheduler state. Make unattended control single-instance, atomically stateful, tolerant of transient reads, heartbeat-visible, bounded-retry, and resumable; a running worker alone does not prove supervision. Keep local TensorBoard authoritative. Unless the user opted out, auto-enable W&B in an RL-specific project sharing the project-family prefix when the adapter's supported authentication probe succeeds; a checked-in false default is not an opt-out. If W&B remains unavailable, explicitly disable it in the attempt's resolved config, record the reason, and continue locally so telemetry cannot fail the run. Follow [the monitoring contract](references/monitoring-contract.md) for due-gated cadence, comparability, stopping, and rebound.
- Track every objective's reward and effective loss/advantage activity with its raw support, denominator, missingness, and hard-pass grounding, plus primary validation, uncertainty, diversity, KL, entropy, lengths, invalids, throughput, NaN/Inf, OOM, and checkpoint integrity. Audit gradient contribution against response length/termination so short low-reward samples are not silently discounted. Give noisy non-safety audit signals a 5–10-validation rebound window; pause only if they persist.
- Audit an unexplained missing or fixed-zero enabled metric immediately against stage reach, measurement availability, artifacts, and controls. A successful no-signal measurement is a biological zero; absent measurement is an environment/contract fault.
- Do not stop or drop an approved therapeutic-quality objective solely because initial support is sparse. Continue comparable evidence collection while healthy, and use only contract-compatible shaping, proposal-support work, or a recorded staged restart; never weaken final hard exclusions to improve aggregate reward.
- Promote each comparable new best while preserving best/latest. Append observations/decisions; never rewrite history.
- Use [the historical case](references/historical-case.md) only for replication or interpretation of that run.

## Baselines and restarts

- Before any resume or relaunch, independently verify through executor state and attempt markers that the prior process/job is absent or terminal; never duplicate a live submission.
- Fresh RL initializes policy and KL reference from selected non-RL SFT.
- Exact resume restores optimizer/scheduler/RNG and retains original KL reference and prompt/validation manifests.
- A material prompt serialization, temperature, prefix-mixture, or validation-distribution change starts a new SFT-anchored attempt.
- A weights-only restart remains KL-anchored to selected SFT and is a new attempt.
- Prior RL weights are allowed only for explicit stagewise objective change. Record why; never silently make prior RL the new KL baseline.

## Report

Write attempt SUMMARY.md, OUTPUTS.yaml, and final monitor state with exact SFT-stage and prompt lineage, RL checkpoint/hash, max/stop/best steps, validation/filter manifests, metric uncertainty, tie breakers, stop/rebound rationale, stability, and risks. Include jump links to lineage, telemetry, checkpoint, and QC artifacts for root promotion. Distinguish selected checkpoint from later evidence-collection stop step.

Do not claim completion from a submitted job. Continue through the adapter monitor loop, or produce executable scripts plus a precise handoff and resume condition.
