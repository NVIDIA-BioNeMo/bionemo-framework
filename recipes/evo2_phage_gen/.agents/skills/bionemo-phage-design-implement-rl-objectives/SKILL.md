---
name: bionemo-phage-design-implement-rl-objectives
description: Use when adding or changing Evo2 phage RL metrics, reward functions, filter logic, or validation criteria after an objective contract has been approved.
---

# Implement Phage RL Objectives

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Implement the approved RL_OBJECTIVES.yaml without changing its biological meaning. Make online rewards, offline validation, and final filtering share tested primitives wherever possible.

## Preconditions

- Read the selected project's design spec, objective contract, and decision log, including approved scope, per-objective exploit analysis, lifecycle coverage, and joint portfolio analysis.
- Require the planning-owned intended-use and safety-applicability block, including every approved separate online component and its matching hard or experimental endpoint. Do not choose or reinterpret safety applicability during implementation; return a missing or contradictory decision to bionemo-phage-design-plan-rl-objectives.
- Require explicit formulas, [0,1] anchors, missing-data behavior, hard all/any tree, and expected runtime call shape. Return unresolved choices to bionemo-phage-design-plan-rl-objectives.
- Require complete-genome generation/evaluation when `mutable_scope: whole-genome`. Reject an implementation that converts similarity, synteny, host-range emphasis, or a regional predictor into an undeclared fixed backbone or locus-only path.
- Confirm a user-owned checkout or authorized branch/worktree/copy. Record branch, revision, existing dirty diff, and symlink/copied-file boundaries; preserve unrelated changes. Stop before mutation if safe source isolation is unavailable.

## TDD workflow

Optional engineering workflow plugins may help with planning, TDD, debugging, and review. [Superpowers](https://github.com/obra/superpowers) is available for Codex, Claude Code, Pi, and other compatible harnesses; it does not replace the approved objective contract, this recipe's TDD and tests, or result-root provenance. Scale additional verification to the biological and integration risk. The counterexample, boundary, missing-input, online/final-alignment, installed-runtime, and deterministic RL smoke tests below remain required.

01. Inspect nearest recipe implementation, tests, package metadata, and current NeMo-RL checkout. Do not infer an API from an example config.
02. Before code, check how each score could look good while the biological result is wrong. Try missing or default values, no observations, very few observations, deletion, truncation, duplication, alternate canonical forms, tool failure, and exact threshold edges. Then test all scores together for correlation, double counting, weight or scale dominance, conflicting directions, one OR branch dominating, and combinations that favor an unintended design. Write concrete reference, baseline/random, desired, and counterexample fixtures. Even when this review stops before code, state what will be recorded—numerator, denominator, observation count, each component, combined score, hard-QC result, and reason—and the test comparing online scoring with final QC. Also state the owning recipe `src/` and `tests/` RED/GREEN placement, installed-runtime name/order/dtype/device/shape/reduction checks, and a tiny deterministic RL smoke test covering reward calculation, logging, checkpoint writing, and restart metadata. Return unresolved meaning or a misleading combined score to the `bionemo-phage-design-plan-rl-objectives` skill.
03. Write focused tests from that audit before implementation. Run them and save expected RED output in the result attempt.
04. Add the smallest implementation under the owning recipe src/; mirror tests under that recipe tests/. Do not add cross-recipe imports or ad hoc symlinks. Preserve existing repository-authorized symlink boundaries. For intentionally identical copies, edit the authoritative source, update `SOURCE_TO_DESTINATION_MAP` when the mapping changes, and regenerate with `python ci/scripts/check_copied_files.py --fix`. Treat a new symlink or copy mapping as a deliberate repository-boundary change, not a shortcut around ownership.
05. Cover unit examples, exact boundaries, monotonicity/range properties, invalid or missing inputs, counterexamples, circular/strand equivalence where relevant, and explicit all/any behavior.
06. Run an offline fixture through online scoring and final QC. Verify agreement; document intentional proxy differences. Require the intended combined-score ordering across reference, baseline/random, desired, and counterexample designs; retain every component score and observation count.
07. Derive the required runtime capabilities from enabled objectives, pin exact paths, and test representative positive and failure/no-signal controls. For external computational tools, apply the shared [external-tool filtering and scoring policy](../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md#external-tool-filtering-and-scoring) before choosing workers, threads, batching, or acceleration. Measure each online objective independently on its intended denominator; hard-gate sequencing belongs in final QC and must not silently starve later rewards.
08. Score the reference and representative baseline-SFT generations before RL. For every enabled therapeutic-quality component, verify measurable support, partial-credit ordering when defined, and isolation from earlier gates. Diagnose fixed-zero or missing components; implement only contract-compatible shaping or a preapproved staged schedule, never removal of a hard endpoint merely to make the aggregate trainable.
09. Verify the pinned runtime contract with a tiny real batch, then a short RL smoke. Record GREEN commands/output, environment, source hash, and resolved config.
10. Put generated configs/reports in the result attempt. Do not mutate shared defaults for one run.

Use [the implementation checklist](references/implementation-contract.md) and [runtime adapter contract](references/runtime-contract.md).

## Required behavior

- Clip rewards to [0,1]; map named baseline/chance to 0 and target to 1; provide monotonic partial credit; give missing or invalid inputs zero credit with a machine-readable reason.
- Keep stable objective/filter IDs. Emit components, aggregate, hard pass, OR branches, and diversity separately.
- Implement AND as separate channels plus hard all. Implement OR as the approved aggregate—normally max(children)—plus hard any; retain children to detect dominance.
- Default diversity is `1 / cluster_size` at declared 99% identity unless the contract names a tested substitute.
- Use each approved operational threshold with its calibration population, model identity and version, uncertainty, comparator, applicability limits, and explicit handling of missing or invalid evidence. An operational threshold is not biological proof; return any proposed change in its meaning to planning.
- Emit design-spec identity and scope with every resolved config and score report. A regional metric is one named component; complete-genome and lifecycle hard checks remain independently observable.
- Keep every approved intended-use and safety component separately observable and bounded, using separate GDPO objectives where applicable. Do not combine them into one opaque pass/fail reward, let a sparse component zero unrelated rewards, or relax final harmful-cargo and other hard exclusions to improve reward support.

## Stop conditions

Diagnose and repair or compatibly adapt runtime, calibration, proposal support, and test-harness gaps within scope. Stop only when objective meaning remains unresolved, safe source isolation or new authority is unavailable, or no test can distinguish intended biology from an exploit without changing the contract. Low initial reward support alone is not a reason to drop an approved therapeutic objective or abandon the run. Record blockers; never silently change biology or reshape rewards to fit.
