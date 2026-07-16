# Historical checkpoint evidence

This reference prevents the documented case from becoming a generic recipe. Audit its exact historical numbers against the checked-in [evidence snapshot](../../phage-design/references/historical-evidence.md); ignored local logs are optional corroboration, not a portability requirement.

- Step 190 was the historical best, not a prescribed stopping point. Sustained-degradation logic would have continued for several validations—roughly to step 250—before selecting 190.
- Step-190 validation, reported separately: 50/96 raw full-QC passes (52.08%); 48/96 after 99%-identity cluster deduplication (50%).
- Step-190 Offline Arc Sequential Final with Architecture Removal disabled: 358/1000 (35.80%).
- Corresponding Full branch with Architecture Removal enabled: 5/1000 (0.50%).

The requested replication selects filters 1–6, 8, and 9; filter 7 is disabled. It retains the combined Arc synteny/total-gene accepted pairs (10,10), (10,11), (10,12), (11,12), and (12,12), plus resolved required-gene and DUST gates. Literal paper filter-number definitions remain diagnostic. The exact resolved filter tree, config/source hashes, and artifact—not filter numbers alone—are authoritative.

Never conflate online rewards, 96-design validation, offline 1,000-design analysis, or another selection metric. Report numerator, denominator, raw/clustered status, identity/coverage, profile ID, artifact, and checkpoint hash.
