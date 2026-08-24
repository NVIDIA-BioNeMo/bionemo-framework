---
name: bionemo-phage-design-implement-rl-objectives
description: Use when adding or changing Evo2 phage RL metrics, reward functions, filter logic, or validation criteria after an objective plan has been approved.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Implement Phage RL Objectives

Work inside the selected recipe checkout and follow its existing code and test patterns. Implement only the approved scientific behavior, keeping online rewards and final QC definitions aligned where they are intended to match.

Keep rewards bounded on `[0, 1]`, with the objective plan's meaningful baseline at zero and target at one. Handle short genomes, absent genes or ORFs, empty tool output, missing observations, invalid values, and tool failures explicitly. Empty or invalid genomes receive explicit zero credit across every applicable objective. In mixed valid and invalid batches, isolate invalid rows so valid rows still score; initialize numeric reward columns with stable numeric dtypes, distinguish a successful no-hit from measurement failure, and make results invariant to batch and row order. Do not let an exception, default value, or skipped branch create positive credit.

Preserve formulas, controls, record mapping, and verified scoring concurrency when refactoring or optimizing a scorer; compare the reference and optimized paths on the same cases.

For every new or modified function, run biological positive and negative controls—not only arithmetic unit tests. Use known or evidence-supported expected high- and low-scoring genomes or targeted counterfactuals, record their provenance and expected ordering, and exercise them through the exact installed scorer and combined portfolio.

For learned or stacked scorers, construct upstream features from deployment-matched out-of-fold predictions. Test ranking and calibration in the high-reward tail, scorer-training versus policy-candidate shift, and model/seed disagreement; treat unsupported candidates as missing or out of domain rather than extrapolating confidently.

Add focused tests that demonstrate:

- each enabled component runs and is logged for a known positive control;
- reference, random/baseline, desired, boundary, missing/empty/no-hit, too-short, missing-gene, and tool-failure behavior as applicable;
- the complete enabled portfolio runs on the reference, equivalent circular representations where relevant, targeted negatives, and mixed-validity batches;
- reordered mixed-validity batches preserve valid-row scores, zero invalid rows, and stable numeric dtypes;
- combined objectives do not hide a skipped or fixed-zero component and preserve intended ordering;
- online scoring and final QC agree where promised;
- the installed runtime accepts the expected names, shapes, dtypes, devices, and reductions; and
- a tiny deterministic RL smoke computes rewards, writes a checkpoint, and can resume.

Before declaring reward work complete, create or refresh `artifacts/RL_SCORE_DEFINITIONS.md` in the selected result root. Reconcile it against the implemented reward columns, formulas, resolved configuration, and focused boundary/failure tests. For every enabled objective, state the measured quantity and units, direction, exact formula and settings, zero-credit and full-credit regions or categorical states, both-side partial-credit behavior when applicable, missing/invalid/empty/no-hit/missing-gene/tool-failure behavior, biological rationale and citations, controls and telemetry, and relationship to final hard QC. This is an agent-produced implementation artifact, not a required stage of the fully scripted run. Use the **Current PhiX174 GDPO score definitions** section in `examples/README.md` as a worked format, not as target-independent scientific defaults.

Use the execution skill for a real installed-environment smoke test when local imports are not representative. Run the relevant focused tests after implementation and record the command, settings, results, and any scientific limitation in the stage summary and `RUNLOG.md`.
