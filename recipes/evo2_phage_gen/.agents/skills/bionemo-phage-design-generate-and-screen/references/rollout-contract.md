# Adaptive rollout planning

Use the checked-in [historical evidence snapshot](../../bionemo-phage-design/references/historical-evidence.md) when quoting the exact step-190 context below; ignored local logs are not required.

## Compatible pilot

Reuse a pilot only when checkpoint hash, prompts, generation parameters, topology/canonicalization, hard-QC profile/code, external tool/database versions, and clustering settings match. Otherwise generate a fresh seeded batch of 1,000.

For each batch and cumulatively report requested, attempted, valid, exact-unique, hard-QC-passing and unique passing-cluster counts; sequential/independent filters; any-branch overlap/dominance; new-cluster accumulation; failures; and compute/time.

## Order-target mode

Estimate a conservative lower bound on unique hard-QC-passing clusters per raw generation. Include uncertainty and declining new clusters per batch. A binomial interval can describe initial hard-pass uncertainty, but model or conservatively discount cluster saturation separately.

Predeclare ranking. It must be target-relevant, defensible, nonredundant with hard gates, and available for all passers.

- With meaningful ranking, target at least 3 * order_target passing clusters.
- Without it, target at least ceil(1.25 * order_target).

Project 1,000-design batches from the conservative yield. Show cost and ask before an expensive projection. Stop at reserve, budget, or demonstrated saturation. If pilot yield is zero or QC repeatedly fails, revisit generation/objectives/filters.

## Fixed modes

Exact-generation mode attempts the requested total. Proof of concept defaults to 1,000; extended is 10,000. Report shortfall, usable yield, and saturation without silently expanding count.

Historical context for the exact step-190 profile only:

- Offline Arc Sequential Final with Architecture Removal disabled: 358/1000 (35.80%).
- Corresponding Full branch with Architecture Removal enabled: 5/1000 (0.50%).

These values never replace a compatible pilot.
