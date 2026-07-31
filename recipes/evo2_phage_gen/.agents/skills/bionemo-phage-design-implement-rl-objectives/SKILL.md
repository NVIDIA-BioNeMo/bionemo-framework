---
name: bionemo-phage-design-implement-rl-objectives
description: Use when adding or changing Evo2 phage RL metrics, reward functions, filter logic, or validation criteria after an objective contract has been approved.
---

# Implement Phage RL Objectives

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Implement the approved RL_OBJECTIVES.yaml without changing its biological meaning. Make online rewards, offline validation, and final filtering share tested primitives wherever possible.

## Preconditions

- Read the selected project's objective contract and decision log, including approved per-objective exploit analysis and joint portfolio analysis.
- Require explicit formulas, [0,1] anchors, missing-data behavior, hard all/any tree, and expected runtime call shape. Return unresolved choices to bionemo-phage-design-plan-rl-objectives.
- Confirm a user-owned checkout or authorized branch/worktree/copy. Record branch, revision, existing dirty diff, and symlink/copied-file boundaries; preserve unrelated changes. Stop before mutation if safe source isolation is unavailable.

## TDD workflow

Optional engineering workflow plugins may help with planning, TDD, debugging, and review. [Superpowers](https://github.com/obra/superpowers) is available for both Codex and Claude Code; it does not replace the approved objective contract, this recipe's TDD and tests, or result-root provenance.

1. Inspect nearest recipe implementation, tests, package metadata, and current NeMo-RL checkout. Do not infer an API from an example config.
2. Before code, repeat the score-gaming brainstorm against the actual dataflow. For each component try missing/default values, empty support, denominator shrinkage, deletion, truncation, duplication, canonicalization, tool failure, and threshold edges. Then test the full portfolio for correlation, double counting, weight/scale dominance, conflicting directions, OR dominance, and A+B trajectories toward unintended C rather than desired D. Write concrete reference, baseline/random, desired, and adversarial fixtures. Even when the audit stops before code, state the proposed telemetry contract—raw numerator/denominator/support, every child score, aggregate, hard pass/reason—and the fixture/test that compares online scoring with final QC. Also state planned selected owning recipe `src/` and `tests/` RED/GREEN placement, installed-runtime name/order/dtype/device/shape/reduction checks, and a tiny deterministic RL smoke covering reward calculation, logging, checkpoint writing, and restart metadata. Return unresolved meaning or an exploitable aggregate to bionemo-phage-design-plan-rl-objectives.
3. Write focused tests from that audit before implementation. Run them and save expected RED output in the result attempt.
4. Add the smallest implementation under the owning recipe src/; mirror tests under that recipe tests/. Never cross-import another recipe.
5. Cover unit examples, exact boundaries, monotonicity/range properties, invalid/missing/support inputs, adversarial fixtures, circular/strand equivalence where relevant, and explicit all/any behavior.
6. Run an offline fixture through online scoring and final QC. Verify alignment; document intentional proxy differences. Require the intended aggregate ordering across reference, baseline/random, desired, and adversarial designs; retain every child score and support field.
7. Derive the required runtime capabilities from enabled objectives, pin exact paths, and test representative positive and failure/no-signal controls. Measure each online objective independently on its intended denominator; hard-gate sequencing belongs in final QC and must not silently starve later rewards.
8. Verify the pinned runtime contract with a tiny real batch, then a short RL smoke. Record GREEN commands/output, environment, source hash, and resolved config.
9. Put generated configs/reports in the result attempt. Do not mutate shared defaults for one run.

Use [the implementation checklist](references/implementation-contract.md) and [runtime adapter contract](references/runtime-contract.md).

## Required behavior

- Clip rewards to [0,1]; map named baseline/chance to 0 and target to 1; provide monotonic partial credit; fail closed with a machine-readable reason.
- Keep stable objective/filter IDs. Emit components, aggregate, hard pass, OR branches, and diversity separately.
- Implement AND as separate channels plus hard all. Implement OR as the approved aggregate—normally max(children)—plus hard any; retain children to detect dominance.
- Default diversity is 1 / cluster_size at declared 99% identity unless the contract names a tested substitute.
- Do not promote an empirical threshold to viability/bootability. Preserve calibration population, model identity, and uncertainty.

## Stop conditions

Diagnose and repair or compatibly adapt runtime, calibration, and test-harness gaps within scope. Stop only when objective meaning remains unresolved, safe source isolation or new authority is unavailable, or no test can distinguish intended biology from an exploit without changing the contract. Record blockers; never silently change biology or reshape rewards to fit.
