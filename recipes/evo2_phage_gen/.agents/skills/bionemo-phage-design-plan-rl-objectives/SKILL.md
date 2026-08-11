---
name: bionemo-phage-design-plan-rl-objectives
description: Use when converting a phage-design goal into target-specific RL rewards, validation criteria, and final QC filters, especially for a new reference phage or altered objective.
---

# Design Phage RL Objectives

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Turn the user's desired outcome into an evidence-backed contract connecting every online reward to validation and final selection. Treat complete-genome viability, lifecycle-wide productive infection, intended change, and diversity as distinct needs.

## Workflow

1. Read the project [whole-genome/lifecycle contract](../bionemo-phage-design/references/design-scope-and-viability.md) and `planning/DESIGN_SPEC.yaml`. Record reference phage/hash, target and original hosts, desired host-range vector, protected traits, complete-genome output/mutable scope, viable-reference set, ordering budget, acceptable uncertainty, and proposed prompt semantics. Default to whole-genome design. A locus-, module-, edit-count-, or fixed-backbone restriction requires its explicit approval record; host-range emphasis, similarity, synteny, or an excluded diversity metric never implies one. Honor fresh/no-reuse instructions. Otherwise use an approved SFT choice or ask only when candidates differ materially.
2. Translate intent into four columns: preserve complete-genome viability; enrich for lifecycle-wide productive infection/bootability; reward the intended host-range and other directional changes from the reference; retain diverse successful designs. Trace every objective/filter to at least one column and maintain a separate lifecycle coverage table so an adsorption or host-model score cannot stand for every post-entry hurdle.
3. For a new phage, assume multiple rewards and filters need redesign. Research adsorption/entry, target-strain defenses and phage counter-defense, takeover/replication, morphogenesis/packaging, lysis/progeny, [therapeutic suitability and safety-related exclusion criteria](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md) when applicable, essential/key genes and regulatory elements, synteny and its definition, positive/negative thresholds, topology/termini/length, composition, viable-reference similarity, host phenotype, novelty, CheckV, production-host-dependent DNA modification, and predictive-model applicability. Remove irrelevant case-study objectives.
4. Use primary literature plus distributions from the reference, related natural phages, baseline SFT generations, and labeled outcomes. At high-level planning, ask whether the user is supplying an operational threshold or wants an evidence-calibrated proposal. A user-supplied operational threshold may define the approved decision boundary, but it does not establish viability or bootability. Record its source, rationale, comparator, false-positive and false-negative consequences, calibration population, model identity, uncertainty, applicability limits, and validation plan. Build an evidence/assumption/validation table; mark unknowns and calibration experiments.
5. Adversarially brainstorm each proposed score before approval: how could training maximize it through missing/default values, denominator or support shrinkage, sequential-gate starvation, deletion, truncation, duplication, canonicalization, tool failure, boundary effects, or a proxy shortcut without achieving its biological intent? Define each online denominator, required runtime capabilities, fail-closed support, and behavioral controls.
6. Analyze the collection jointly. Check correlation/double counting, weight or scale dominance, incompatible gradients, OR-branch dominance, and whether optimizing A+B leads toward an unintended C instead of the user's D. Ordinary trade-offs are acceptable; an easy high-scoring wrong endpoint is not. Compare reference, baseline/random, desired, and adversarial designs; add or change rewards, hard filters, weights, or calibration when needed.
7. Create `rl/runs/<attempt>/` and write artifacts/RL_OBJECTIVES.yaml using [the objective contract](references/objective-contract.md), plus standard request, outputs, summary, and run log. Include the design-spec hash, scope-approval status, lifecycle coverage, hard-QC logic, online approximations, adversarial/portfolio analysis, calibration endpoints, selection metric, ablations, and unresolved decisions. Do not write the stable rl/RL_OBJECTIVES.yaml directly; the calibration skill owns the later fixed sampling and validation manifests.
8. Read [evidence and proxies](references/evidence-and-proxies.md). For replication only, read [the historical case](references/historical-case.md).
9. Present the proposed contract. After user approval and controller verification, let the controller promote its hash-addressed copy. Identify hard final QC, shaped online proxies, individual gaming risks, joint divergence risks, and mitigations.

## Intended-use therapeutic objectives

Unless the user clearly states another use, provisionally classify adapted-design work as
therapeutic and record the visible assumption. For adapted-design work with therapeutic intended use,
include the applicable design-relevant
[therapeutic guardrails](../bionemo-phage-design/references/design-scope-and-viability.md#apply-intended-use-therapeutic-guardrails)
in the RL score by default. Use the EMA draft's relevant details on [phage seed
lots](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#phage-seed-lots),
[genome characterisation](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#genome-characterisation),
[host range](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#host-range),
[potency](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#potency),
and [transducing
capacity](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md#transducing-capacity),
while excluding downstream manufacturing controls from sequence rewards.

For the recipe's default PhiX174 case-study replication, use the current customized replication
profile: enable filters 1–6, 8, and 9, keep filter 7 disabled, and add every applicable
design-relevant safety objective with a defensible measurable proxy by default. This is not the
paper-exact objective set. Preserve the historical component set and its profile identity, emit the
added components separately, version the active component set, and do not directly compare aggregate
rewards computed from different component sets. For an explicitly non-therapeutic adapted project,
record the intended-use rationale and the reason for every omitted therapy-specific item.

Represent applicable items as separate components, not one opaque safety score. Every component with
a defensible measurable proxy must emit a score even when it is also final hard QC; genuinely
unscorable items remain explicit experimental endpoints. Calibrate each component on the reference
and baseline SFT generations, measure it independently of earlier gates, and use monotonic partial
credit where justified. Predeclare how sparse support will be repaired through runtime fixes,
better-calibrated shaping, proposal-distribution work, or an approved staged schedule. Never make a
hard exclusion passable, delete a difficult component, or declare the therapeutic objective
impossible merely to obtain nonzero aggregate reward.

## Invariants

- Default to GDPO. Map every online component to \[0,1\]: named baseline/chance is 0, target is 1, partial credit is monotonic/clipped, and missing data fails closed.
- Keep generation and final evaluation whole-genome unless the design spec contains an explicit approved scope reduction. A regional host-range score may be one component, never an undeclared mutation or viability boundary.
- Include 1 / cluster_size at 99% identity unless another explicit mechanism enforces diversity.
- Express final logic as nested all/any. Keep AND conditions as separate rewards plus hard all. Usually wrap OR branches in one max-style reward plus hard any; report branch dominance.
- Define synteny units, orientation, circular equivalence, overlaps, partial credit, and thresholds. It may reward preservation or deliberate disruption according to the goal.
- Treat unknown genes as unknown, not dispensable. Do not make transferred or correlative evidence a hard essentiality constraint.
- Keep online rewards and final hard QC aligned; document deliberate approximations.
