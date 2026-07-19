# RL monitoring and checkpoint selection

## Predeclaration and comparability

Name one primary validation metric and exact filter-profile hash before launch. For the replication case, use the count/rate of full-QC passers after 99%-identity cluster deduplication; tie break by raw full-QC passers, total diversity, then stability. New goals require target-specific metrics from the approved objective contract.

Pin a validation-generation manifest containing prompt IDs, source-genome and prompt-manifest hashes, prompt-length strata, rollout ordinals, seeds, temperature and all sampling parameters, sample counts, canonicalization, filter/tool versions, and QC denominator. Prefer paired fixed-seed comparisons. Report every length stratum and a predeclared reweighted aggregate. If composition changes, compare within strata and reweight to the frozen target mix; an unadjusted aggregate decline is diagnostic and does not consume patience.

Report sampling uncertainty appropriate to the metric, such as paired bootstrap intervals for the fixed bank and Wilson intervals for simple pass-rate diagnostics. Predeclare a practical minimum change. A decline must exceed ordinary sampling/data variance as well as the percentage rule below.

The 500-step setting is a ceiling, not a target. Validate often enough to observe degradation, but consistently enough that events are comparable.

## Normal stopping

Promote a new best whenever the predeclared ordering improves. Count only complete, comparable validation events. Stop only when all hold:

1. no new best for six comparable validation events;
2. the latest four comparable primary values are all below the best;
3. their median is at least 10% below the best and the decline exceeds predeclared uncertainty/minimum change; and
4. no meaningful tie-breaker improvement offsets the decline.

Otherwise continue to the 500-step ceiling. The stopping step supplies evidence; select and preserve the earlier best checkpoint.

Allow one predeclared confirmation extension of at most two comparable events when a late rebound recovers at least half of the best-to-trough drop, exceeds expected noise, and no collapse trigger is active. Reset patience only for a genuine comparable new best. Do not extend beyond step 500.

## Collapse and health

Stop early and preserve best/latest when any declared collapse trigger fires:

- full hard-QC pass becomes zero for two comparable validations after having been nonzero;
- primary performance is below 25% of its best for three comparable validations;
- NaN/Inf, repeated OOM, corrupt checkpoints, exhausted disk, runaway KL, entropy collapse, invalid-output spike, or another predeclared instability makes continuation unsafe or uninformative.

Before changing learning rate, batch shape, reward weights, validation composition, or objectives, save evidence and create a decision entry. A semantic config change starts a new attempt. Use exact resume only when training semantics are unchanged.

## Phase-aware observation cadence

Use the execution adapter's due-gated one-tick monitor. Check somewhat more often through launch, the first few steps, first validation, and first verified checkpoint. After that healthy boundary, back off scheduler, disk, and telemetry sources independently to minutes or the validation cadence, with jitter. On timerless /goal re-entry, return without querying when next_check_at is not due.

Do not confuse observation cadence with validation cadence. A cheap log-health check may occur between validations, but checkpoint selection changes only on complete comparable validation events.

## Monitor state

Keep monitor/state.json current and append monitor/events.jsonl. Each validation event includes step/time, checkpoint hash, primary and tie-break values, uncertainty, validation-manifest hash, prompt strata, exact denominator/filter profile, reward components, hard-filter waterfall, OR branches, diversity, KL/entropy, lengths/invalids, system health, data source, comparability flag/reason, and best/stop decision.

For the PhiX174 target profile, record AAI (filter 8), the required-gene list,
and synteny/total-gene logic (filter 9) as required final-pass filters. Do not
label any of those three optional in summaries or pass-rate telemetry. Filter 7
remains a separately reported disabled diagnostic unless the attempt explicitly
declares a different target profile.

For vLLM attempts also retain rollout-versus-policy logprob deltas and ratios, finite
reward/advantage/loss/gradient checks, optimizer-step/parameter-change evidence, per-DP
request/global-index/seed ranges, post-update export/conversion/refit/sync evidence, and rollout,
reward/QC, policy/reference forward, backward, optimizer, refit/sync, barrier, validation, peak
memory, and total-step timings. A missing or stale refit, overlapping DP stream, or non-finite
numeric gate is a health failure, not a comparable validation event.

If continuous automation is unavailable, emit a repeatable one-tick script and handoff stating next_check_at. Never invent a nonexistent CLI.
