# PhiX174 case-study results

These numbers are context for the documented PhiX174 experiments, not defaults for another target and not evidence of wet-lab viability.

## Current end-to-end run

- Cluster-held-out SFT used 14,266 training, 100 validation, and 100 test genomes.
- Training ran to a 12,000-step ceiling and selected step 5,600 by validation loss (0.750670); held-out test loss was 0.798180.
- Sampling calibration selected temperature 1.0 and a 50:50 mixture of 16- and 24-nucleotide prompts.
- GDPO ran to 500 steps and selected step 430 from full-QC validation.
- A 1,000-design rollout produced 610 target-profile passes with filter 7 disabled and 22 diagnostic passes with filter 7 enabled.

## Earlier released-SFT shortcut

An earlier RL-only run reused the published Microviridae SFT checkpoint and selected step 190. Its 1,000-design rollout produced 358 target-profile passes and 5 filter-7-enabled passes. Step 190 is historical evidence, not a prescribed stopping point.

The publication-era 15/110,000 result used a different screening pipeline and is not a controlled enrichment baseline for either run. Keep validation, final rollout, online reward rates, filter profiles, and checkpoint selections distinct when comparing results.
