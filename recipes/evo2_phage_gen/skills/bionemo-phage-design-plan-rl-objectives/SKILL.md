---
name: bionemo-phage-design-plan-rl-objectives
description: Use when converting a phage-design goal into target-specific RL rewards, validation criteria, and final QC filters, especially for a new reference phage or altered objective.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Design Phage RL Objectives

Work inside the recipe and result roots selected by the controller. Translate the user's biological goal into online rewards, validation criteria, and final hard filters. Default to whole-genome design unless the user explicitly requests a locus-, module-, or RBP-only task.

Cover complete-genome viability and the productive-infection lifecycle, not only host binding: adsorption/entry, defense and counter-defense, replication, morphogenesis/packaging, lysis/progeny, essential/key genes, regulation, synteny, topology, composition, similarity to viable references, desired host direction, diversity, and intended-use safety. A host-range model is one signal and does not prove productive infection.

When host-range or bootability rewards are sparse, do not rely on them as the only learning signal. Add biologically justified intermediate rewards—such as essential-gene completeness and reasonable synteny—with carefully calibrated partial credit toward evidence-supported ranges. Log them separately.

For whole-genome designs—including custom or adapted runs—include AMR, toxin, and lysogeny as separate RL objectives. An explicitly scoped locus or module edit may omit objectives that the edited region cannot affect.

Define components using the concise [objective guidance](references/objective-guidance.md). Each reward remains on `[0, 1]`: zero is a meaningful random/baseline or clearly unacceptable outcome, not merely the raw metric's numeric zero, and one is the supported target. Missing, invalid, empty, or failed measurements must not crash the portfolio or receive accidental positive credit.

For every component, identify a positive control that proves the calculation actually runs, plus baseline/random, boundary, no-hit/empty, too-short, missing-gene, and tool-failure cases as applicable. Specify what is logged so a silently skipped or fixed-zero component is visible. Keep required hard-QC filters independent of reward shaping.

Check the portfolio for duplicate signals, scale dominance, incompatible directions, easy gaming, and combinations that reward the wrong biological endpoint. Compare reference, baseline/random, desired, and counterexample designs. Preserve component-level validation when aggregate component sets differ.

For the PhiX174 case study, retain filters 1–6, 8, and 9 with filter 7 disabled for the target profile, and run filter 7 separately as a diagnostic. Add relevant safety objectives without pretending this computational profile establishes biological viability.

Write a concise `artifacts/RL_OBJECTIVES.yaml` or table containing component definitions, baselines/targets, missing-data behavior, controls, hard-QC relationship, validation/selection criteria, and unresolved biological choices. Summarize the decision in `RUNLOG.md`.
