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

For the realized `examples/phix174_8xh100.sh` workflow, treat
`<result_root>/calibration/sampling-selection.yaml` as the durable selection handoff. On re-entry
after calibration generation or scoring, inspect `calibration/scoring/selection-evidence.csv` and
the underlying score/novelty artifacts, choose settings by the criteria above, and write the
canonical YAML with `temperature`, `top_k`, `top_p`, `max_new_tokens`, `prompt_lengths`, `rl_seed`,
`rollout_seed`, and `seed_stride`. The prompt lengths are an equal mixture and their count must tile
the training bank, validation bank, final rollout, and GPU count. Record the evidence and rationale,
then rerun the original top-level command with the same result root; the narrow completion markers
reuse completed calibration work and the existing canonical file skips the historical-default
hard check.

`--sampling-selection PATH` is an explicit operator-override path: the script validates and copies
that file to the canonical location whether supplied on the first invocation or a rerun. When the
skill is operating the demo autonomously, do not pass
`examples/default-sampling-selection.yaml` merely to get past a calibration stop. Make the
evidence-based decision and write the canonical file directly. Use the bundled default override
only when the user explicitly chooses historical settings despite the warning.
