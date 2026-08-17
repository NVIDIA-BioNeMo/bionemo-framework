# Rollout guidance

Use prompts and seeds independent of calibration and RL validation. Generate the requested number of completed candidates, accounting for failed or duplicate attempts without silently shrinking the denominator.

Validate, deduplicate biological equivalents, run required QC, and cluster in a deterministic candidate order. Representative batching is acceptable only when record mapping remains complete and the representative result agrees with controls.

Report raw and clustered counts, PASS/FAIL/INDETERMINATE denominators, uncertainty when comparing yields, and whether generation or filtering saturated before forecasting a larger experiment.
