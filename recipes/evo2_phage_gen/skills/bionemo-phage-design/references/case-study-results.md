# PhiX174 case-study results

These numbers are context for the documented PhiX174 experiments, not defaults for another target and not evidence of wet-lab viability.

## 2026-08-24 8xH100 rerun

- The final rollout generated and SFT-likelihood scored 1,000 designs; exact/circular/reverse-complement deduplication retained 1,000 biological representatives.
- Pre-safety nucleotide QC submitted 991 representatives and excluded 9.
- Safety screening reported 989 PASS, 0 FAIL, and 2 INDETERMINATE.
- Target hard QC retained 513 safety-PASS representatives.
- Post-QC 99%-identity clustering retained 511 clusters and accepted representatives.

## Earlier completed end-to-end run

- Cluster-held-out SFT used 14,266 training, 100 validation, and 100 test genomes.
- Training ran to a 12,000-step ceiling and selected step 5,600 by validation loss (0.750670); held-out test loss was 0.798180.
- Sampling calibration selected temperature 1.0 and a 50:50 mixture of 16- and 24-nucleotide prompts.
- GDPO ran to 500 steps and selected step 430 from full-QC validation.
- A 1,000-design rollout produced 610 target-profile passes with filter 7 disabled and 22 diagnostic passes with filter 7 enabled.

## Earlier released-SFT shortcut

An earlier RL-only run reused the published Microviridae SFT checkpoint and selected step 190. Its 1,000-design rollout produced 358 target-profile passes and 5 filter-7-enabled passes. Step 190 is historical evidence, not a prescribed stopping point.

The publication-era 15/110,000 result used a different screening pipeline and is not a controlled enrichment baseline for either run. Keep validation, final rollout, online reward rates, filter profiles, and checkpoint selections distinct when comparing results.
