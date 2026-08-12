# SFT operation and checkpoint selection

## Base checkpoint gate

Confirm public resource visibility and license before download. Resolve downloader syntax at runtime; download_bionemo_data --source ngc is a semantic snapshot, not a complete command. Verify the downloaded manifest/hash and expected checkpoint format before conversion or training.

| Purpose            | Resource ID              | Notes                                                         |
| ------------------ | ------------------------ | ------------------------------------------------------------- |
| Paper/case study   | evo2/7b-8k:1.0           | Match the 7B short-context architecture used by the workflow. |
| Adapted design     | evo2/7b-1m:1.0           | Use when the chosen genome/context requires it.               |
| Cheap smoke        | evo2/1b-8k-bf16:1.0      | Validate data and runtime, not 7B biological quality.         |
| Requested scale-up | evo2/40b-1m-fp8-bf16:1.0 | Require explicit hardware/runtime fit.                        |

The phage recipe uses BF16; FP8 is not required. The Evo 2 7B 8K/1M family is broadly compatible. If the user requests a different base-model family, consult the current GPU/precision compatibility table in `<repository_root>/recipes/evo2_megatron/README.md` before selecting or downloading its checkpoint.

For a fallback, discover current converters, run focused tests/smoke, and retain source plus converted hashes. Never label converted bytes as the original artifact.

## Effective batch and cadence

Compute:

```text
data_parallel_size = world_size / (tensor_parallel × pipeline_parallel × context_parallel)
global_sequences = microbatch × accumulation × data_parallel_size
effective_tokens = global_sequences × sequence_length
```

Require integer-compatible topology and compare effective tokens with the value approved after context selection. Aim within 5%; if memory or sequence length prevents that, report the achieved value and consequence rather than quietly changing it.

After the final usable split and batch preflight, compute the training-budget proposal with loss-bearing rather than padded volume:

```text
planned_steps = ceil(
    target_effective_exposures
    × usable non-padding token mass
    / effective non-padding tokens per optimizer step
)
```

Define target effective exposures from the approved optimization evidence and objective, then record the inputs, arithmetic, and uncertainty. Treat 12,000 steps as a historical starting hypothesis, not a fixed maximum. A materially larger usable corpus—for example about 33,000 rather than 10,000 genomes—may justify a ceiling above 12,000 when preserving the intended exposure range; heavy redundancy or a smaller corpus may justify less. Recompute if the resolved batch or data mix changes.

Choose validation and checkpoint intervals that provide at least 30 opportunities by the calibrated ceiling and let the six-event patience rule act within a useful exposure-scale decision horizon. Align saves with validation when possible. Validate only from the explicit validation split; keep test sealed until final evaluation.

## Ordered action trace

Use the controller's composite namespace. Project-wide setup scripts live under planning/execution/scripts/. Stage scripts live under `sft/runs/<attempt>/scripts/` and use IDs such as `sft/<attempt>/010`. Maintain the root planning/execution/ACTIONS.yaml ledger. Each item records ID, intent, prerequisites, script and hashes, executor/host or scheduler/cloud job, start/end/exit, logs, outputs/hashes, and idempotence/resume guard. Never rely on chat-only commands or overwrite another attempt's IDs.

For Slurm or Lepton, preserve both portable stage scripts and resolved submission/job config. Record job ID/URL and which host writes logs/checkpoints or uploads durable artifacts.

## Monitoring state and cadence

At each validation event append step, timestamp, train-loss summary/window, validation loss, learning rate, grad norm, throughput, GPU health, checkpoint path/hash/status, disk free, telemetry source, and comparability evidence.

Use phase-aware due-gated monitoring: observe launch and early progress at intervals justified by expected events and site policy. After one or two healthy validation/checkpoint cycles—or equivalent progress evidence—derive each source's wall-clock cadence from measured step timing and a useful fraction of the next validation, checkpoint, or early-stop boundary, such as the duration of about 10 steps. A stable long phase may use 5–30 minutes. In timerless `/goal` loops, read `next_check_at` and return without querying when not due.

Track best_step, lowest finite validation loss, comparable events since best, latest four comparable losses, robust matched-window training-loss trend, checkpoint verification, and exact resume lineage.

## Variance-aware early stopping

Count patience only for complete evaluations with the same held-out split-manifest hash, sample set/denominator, loss mask and reduction, model/eval mode, tokenizer, and data preprocessing. An incomplete evaluation or changed composition is diagnostic and does not consume patience.

Estimate ordinary validation noise robustly after excluding isolated spikes/drops; treat those separately as health incidents. A plateau within that noise continues. Declare overfitting only when all hold:

1. six comparable validation events have occurred since the best;
2. a robust recent trend shows sustained degradation rather than extended flatness;
3. the latest four comparable losses are worse and their median is at least 1% above the best beyond measured noise; and
4. the robust training-loss trend over matched recent windows still decreases.

A transient rise or plateau is insufficient. If the latest comparable events show a meaningful rebound—by default recovering at least half of the best-to-trough loss increase and exceeding noise—allow one predeclared confirmation extension of at most two events, never beyond the calibrated ceiling. A genuine new best resets patience. Always preserve and select the lowest comparable validation-loss checkpoint.

Immediate hazards terminate safely: NaN/Inf, OOM, invalid data, corrupt/incomplete checkpoint, or disk below the safety floor. A changed topology, learning rate, or batch config creates a new attempt rather than silently changing provenance.

## Resume and summary

An exact resume retains data/split hashes, model and optimizer state, scheduler state, RNG/data-order state when supported, topology-compatible config, and original attempt identity with an appended resume event. Otherwise start a new attempt and label it weights-only or fresh.

The concise summary reports maximum steps, actual stop/reason, selected checkpoint step/path/artifact/hash, lowest validation loss, comparable events since best, trend/uncertainty evidence, base-model provenance, data/split/config hashes, hardware/effective token batch, telemetry links/logdirs, and why the checkpoint is fit for RL. Keep observations in monitor/events.jsonl and append-only RUNLOG.md.
