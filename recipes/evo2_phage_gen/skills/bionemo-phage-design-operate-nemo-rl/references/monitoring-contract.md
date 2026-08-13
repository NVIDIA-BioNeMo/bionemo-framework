# RL monitoring and checkpoint selection

## Predeclaration and comparability

Name one primary validation metric and exact filter-profile hash before launch. For the replication case, use the count/rate of full-QC passers after 99%-identity cluster deduplication; tie break by raw full-QC passers, total diversity, then stability. New goals require target-specific metrics from the approved objective contract.

Pin a validation-generation manifest containing prompt IDs, source-genome and prompt-manifest hashes, prompt-length strata, seeds, temperature and all sampling parameters, sample counts, canonicalization, filter/tool versions, and QC denominator. Prefer paired fixed-seed comparisons. If composition changes, compare within strata and reweight to the frozen target mix; an unadjusted aggregate decline is diagnostic and does not consume patience.

Report sampling uncertainty appropriate to the metric, such as paired bootstrap intervals for the fixed bank and Wilson intervals for simple pass-rate diagnostics. Estimate ordinary noise robustly after excluding isolated spikes/drops; handle those separately as incident/collapse evidence. Predeclare a practical minimum change.

The 500-step setting is a ceiling, not a target. Validate often enough to observe degradation, but consistently enough that events are comparable.

## Normal stopping

Promote a new best whenever the predeclared ordering improves. Count only complete, comparable validation events. Stop only when all hold:

1. no new best for six comparable validation events;
2. a robust recent trend shows sustained degradation rather than a plateau within ordinary noise;
3. the latest four comparable primary values are below the best and their median is at least 10% below it beyond uncertainty/minimum change; and
4. no meaningful tie-breaker improvement offsets the decline.

Otherwise—including extended flatness—continue to the 500-step ceiling. Always preserve and select the best checkpoint.

Allow one predeclared confirmation extension of at most two comparable events when a late rebound recovers at least half of the best-to-trough drop, exceeds expected noise, and no collapse trigger is active. Reset patience only for a genuine comparable new best. Do not extend beyond step 500.

## Collapse and health

Stop early and preserve best/latest when any declared collapse trigger fires:

- full hard-QC pass becomes zero for two comparable validations after having been nonzero;
- primary performance is below 25% of its best for three comparable validations;
- NaN/Inf, repeated OOM, corrupt checkpoints, exhausted disk, runaway KL, entropy collapse, invalid-output spike, or another predeclared instability makes continuation unsafe or uninformative.

Before changing learning rate, batch shape, reward weights, validation composition, or objectives, save evidence and create a decision entry. A semantic config change starts a new attempt. Use exact resume only when training semantics are unchanged.

## Per-objective exploitation audit

At each comparable validation, record every objective's reward trajectory and effective loss/advantage activity. Pair it with raw numerator/support, eligible denominator, missingness, stage/measurement availability, and a hard-pass or biological grounding metric. Enabled objectives must be measured independently rather than starved by earlier gates. The combined policy loss alone is insufficient.

Treat non-safety exploitation signals as pending until they persist for 5–10 further comparable events (default 8); recovery clears the streak. Then pause at a complete checkpoint, preserve best/latest, and diagnose. Missing required telemetry for three events and unsafe health failures retain their shorter stops. Change objectives or weights only in a new recorded attempt.

## Phase-aware observation cadence

Use the execution adapter's due-gated one-tick monitor. Observe launch and early progress at intervals justified by expected events and site policy. After one or two healthy validation/checkpoint cycles—or equivalent progress evidence—derive each source's wall-clock cadence from measured step timing and a useful fraction of the next validation, checkpoint, or stop-decision boundary, such as the duration of about 10 steps. A stable long phase may use 5–30 minutes. On timerless `/goal` re-entry, return without querying when `next_check_at` is not due.

Do not confuse observation cadence with validation cadence. A cheap log-health check may occur between validations, but checkpoint selection changes only on complete comparable validation events.

## Monitor state

Keep monitor/state.json current and append monitor/events.jsonl. Each validation event includes step/time, checkpoint hash, primary and tie-break values, uncertainty, validation-manifest hash, prompt strata, exact denominator/filter profile, reward components, counts remaining after each hard filter, OR branches, diversity, KL/entropy, lengths/invalids, system health, data source, comparability flag/reason, and best/stop decision.

If continuous automation is unavailable, emit a repeatable one-tick script and handoff stating next_check_at. Never invent a nonexistent CLI.
