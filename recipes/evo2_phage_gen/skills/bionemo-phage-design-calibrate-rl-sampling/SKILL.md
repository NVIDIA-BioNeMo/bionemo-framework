---
name: bionemo-phage-design-calibrate-rl-sampling
description: Use after selecting an Evo 2 phage SFT checkpoint and defining RL objectives to calibrate prompt serialization, temperature, prefix-length distribution, and fixed validation sampling.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Calibrate Phage RL Sampling

Work inside the recipe and result roots selected by the controller. Calibrate the selected SFT checkpoint rather than copying paper or earlier-run settings.

Reconstruct the actual SFT prompt serialization and tokenization, including conditioning, orientation, wrappers, BOS/EOS, padding/masking, and continuation boundary. Use only cues the SFT model saw.

Sweep a reasonable range of temperatures and prefix lengths with paired seeds and enough samples to compare uncertainty. Materialize the selected top-k/top-p, completion length, and prompt mixture in the commands. Score raw and cluster-deduplicated hard-QC yield, target/lifecycle evidence, complete-genome integrity, copying, diversity, and every enabled objective. Use positive and failure controls to confirm each score is measurable; diagnose unexplained missing or fixed-zero components rather than dropping them.

Choose a robust quality-diversity plateau rather than a noisy maximum. Prefer temperature 1.0 when it is practically equivalent, and retain multiple prompt strata only when they improve the frontier and tile the deployed batch shape cleanly.

Keep calibration samples separate from the fixed RL-validation bank and final rollout. Record prompt construction, seeds, sampling settings, sample counts, score summaries, uncertainty, chosen mixture, and rationale in the stage summary and `RUNLOG.md`.
