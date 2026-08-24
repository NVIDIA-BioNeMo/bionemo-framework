---
name: bionemo-phage-design-calibrate-rl-sampling
description: Use after selecting an Evo 2 phage SFT checkpoint and defining RL objectives to calibrate prompt serialization, temperature, prefix-length distribution, and fixed validation sampling.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Calibrate Phage RL Sampling

Work inside the recipe and result roots selected by the controller. Calibrate the selected SFT checkpoint rather than copying paper or earlier-run settings.

Reconstruct the actual SFT prompt serialization and tokenization, including conditioning, orientation, wrappers, BOS/EOS, padding/masking, and continuation boundary. Use only cues the SFT model saw.

Sweep a reasonable range of temperatures and prefix lengths with paired seeds and enough samples to compare uncertainty. Materialize the selected top-k/top-p, completion length, and prompt mixture in the commands. Treat every prompt base as fixed rather than designed: consume the objective plan's exclusions for genes or regions intended to change and reject circular prompts—including origin-wrapping intervals—that overlap any intended-to-change bases. If neutral anchors cannot support the bank, revisit the prompt strategy.

Report prompt bases and fraction of the genome fixed. Start with the shortest workable prompt consistent with the selected SFT serialization and measured generation quality; do not scale prompt length linearly with genome size by default, because even a sub-percent prompt can be substantial on a small genome. Retain a longer prompt only with a model- and design-specific rationale supported by calibration evidence. Distribute eligible starts around circular genomes, shuffle or interleave strata, and make each global rollout batch representative rather than cycling through fixed cohorts. When step metrics oscillate, stratify by prompt position, length, and composition before attributing the pattern to the policy.

Score raw and cluster-deduplicated hard-QC yield, target/lifecycle evidence, complete-genome integrity, copying, diversity, and every enabled objective. A successful wrapper is insufficient: require every configured external measurement used for selection to be available, distinguish a biological no-hit from scorer failure, and diagnose unexplained missing or fixed-zero components rather than dropping them. Use positive and failure controls to confirm each score is measurable.

Choose a robust quality-diversity plateau rather than a noisy maximum. Prefer temperature 1.0 when it is practically equivalent, and retain multiple prompt strata only when they improve the frontier and tile the deployed batch shape cleanly.

Keep calibration samples separate from the fixed RL-validation bank and final rollout. Record prompt construction, seeds, sampling settings, sample counts, score summaries, uncertainty, chosen mixture, and rationale in the stage summary and `RUNLOG.md`.

For the realized PhiX workflow, read the
[example README](../../examples/README.md) before planning or rerunning it. Treat that document as
the source of truth for the current review stop, evidence paths, selection schema and handoff,
completion markers, and resume procedure. When the user delegates selection, make the
evidence-based choice described there and record its rationale; use the bundled historical choice
only when the user explicitly selects it.

The example shell script is a reference implementation, not a mandatory launcher. On a different
GPU or scheduler environment, inspect the available hardware and adapt topology, batch and worker
settings, or the launch method itself while preserving whole-genome context, effective batch,
sampling semantics, validation independence, and the durable selection record. Do not alter an
active run's sampling semantics in place; use a new result root and SFT-anchored RL attempt for a
material change, retaining the earlier run as evidence.
