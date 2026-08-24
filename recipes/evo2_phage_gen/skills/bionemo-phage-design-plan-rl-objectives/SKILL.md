---
name: bionemo-phage-design-plan-rl-objectives
description: Use when converting a phage-design goal into target-specific RL rewards, validation criteria, and final QC filters, especially for a new reference phage or altered objective.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Design Phage RL Objectives

Work inside the recipe and result roots selected by the controller. Translating the user's desired final phage product into online rewards, validation criteria, and final hard filters is a core agentic capability. Default to whole-genome design unless the user explicitly requests a locus-, module-, or RBP-only task.

Prefer modifying tested rewards when their measurements, direction, and failure semantics scientifically transfer. Existing functions are starting points rather than a closed catalog: beyond a faithful experiment rerun, novel reward functions are expected for important requirements the current portfolio does not capture. For those gaps, creatively invent functions from literature, biological mechanisms, viable references, available tools, and prior or partial-run evidence, then apply the score-definition, control, telemetry, and final-QC requirements below.

Cover complete-genome viability and the productive-infection lifecycle, not only host binding: adsorption/entry, defense and counter-defense, replication, morphogenesis/packaging, lysis/progeny, essential/key genes, regulation, synteny, topology, composition, similarity to viable references, desired host direction, diversity, and intended-use safety. A host-range model is one signal and does not prove productive infection.

When host-range or bootability rewards are sparse, do not rely on them as the only learning signal. Add biologically justified intermediate rewards—such as essential-gene completeness and reasonable synteny—with carefully calibrated partial credit toward evidence-supported ranges. Log them separately.

For whole-genome designs—including custom or adapted runs—include AMR, toxin, and lysogeny as separate RL objectives. An explicitly scoped locus or module edit may omit objectives that the edited region cannot affect.

Before the sampling handoff, identify every gene, module, motif, or uncertain interval the model should remain free to redesign. Record their circular coordinates as origin-aware prompt-exclusion intervals, plus any defensible neutral anchor intervals. Prompt overlap fixes those bases even when it covers only part of a feature; if no neutral anchor exists, require calibration to revisit the prompting strategy rather than silently locking a design target.

Treat the approved objective portfolio as a scientific baseline, not an immutable contract. Reconcile the planned, configured, and emitted components, then continue autonomously: a discrepancy is evidence to diagnose and record, not a reason to pause a late-stage run. Dropping an agreed objective is a last resort; first try to repair or recover it, then, if omission is the strongest defensible path, record its evidence, scientific impact, scope, and restoration criteria while retaining complementary nonredundant evidence.

Proactively adding well-supported shaping objectives is important when literature review or partial-run evidence indicates they will better guide the project toward the desired endpoint. Calibrate and test the added term, version the active objective set and change point, and do not compare aggregate rewards across different sets as the same metric. Make these decisions autonomously, and summarize them in the next natural user update and whenever asked.

Define components using the concise [objective guidance](references/objective-guidance.md). Each reward remains on `[0, 1]`: zero is a meaningful random/baseline or clearly unacceptable outcome, not merely the raw metric's numeric zero, and one is the supported target. Missing, invalid, empty, or failed measurements must not crash the portfolio or receive accidental positive credit.

For every new or modified component, identify strong positive and negative controls: biologically relevant genomes or counterfactuals known or expected to score high and low, with provenance, rationale, and expected ordering. Also include baseline/random, boundary, no-hit/empty, too-short, missing-gene, and tool-failure cases as applicable. Specify what is logged so a silently skipped or fixed-zero component is visible. Keep required hard-QC filters independent of reward shaping.

Check the portfolio for duplicate signals, scale dominance, incompatible directions, easy gaming, and combinations that reward the wrong biological endpoint. Compare reference, baseline/random, desired, and counterexample designs. Preserve component-level validation when aggregate component sets differ.

For the PhiX174 case study, retain filters 1–6, 8, and 9 with filter 7 disabled for the target profile, and run filter 7 separately as a diagnostic. Add relevant safety objectives without pretending this computational profile establishes biological viability.

Write a concise `artifacts/RL_OBJECTIVES.yaml` or table containing component definitions, baselines/targets, missing-data behavior, controls, hard-QC relationship, validation/selection criteria, prompt-exclusion and neutral-anchor intervals, and unresolved biological choices.

Also write `artifacts/RL_SCORE_DEFINITIONS.md` in the selected result root for a scientist reading the run record. Give every enabled objective one compact row or subsection with its reward column, measured quantity and units, direction, exact formula and resolved target-specific settings, zero-credit region or state (lower and upper sides when applicable), full-credit region or state, and how partial credit changes on either side. State categorical and missing/invalid/empty/no-hit/missing-gene/tool-failure behavior directly when numeric ranges do not apply. Include the biological rationale, evidence citations, controls, telemetry, and relationship to final hard QC. This is an agent-produced objective-design artifact, not a required stage of the fully scripted run.

Use the **Current PhiX174 GDPO score definitions** section in `examples/README.md` as a concise worked example of this writeup, but resolve every setting and source for the selected target rather than copying PhiX-specific thresholds.

Summarize the decision and both artifact paths in `RUNLOG.md`.
