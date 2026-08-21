# Rollout guidance

Use prompts and seeds independent of calibration and RL validation. Generate the requested number of completed candidates, accounting for failed or duplicate attempts without silently shrinking the denominator.

Validate the raw denominator, retain raw-model scores when requested, deduplicate exact/circular/
reverse-complement biological equivalents, run required safety and hard QC on representatives, and
only then cluster passers in a deterministic candidate order. Representative batching is
acceptable only when record mapping remains complete and the representative result agrees with
controls.

Report raw, biological-representative, hard-QC, and post-QC-cluster counts,
PASS/FAIL/INDETERMINATE denominators, uncertainty when comparing yields, and whether generation or
filtering saturated before forecasting a larger experiment.
