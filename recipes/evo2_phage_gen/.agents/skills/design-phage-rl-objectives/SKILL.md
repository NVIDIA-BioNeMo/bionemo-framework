---
name: design-phage-rl-objectives
description: Use when converting a prokaryotic-phage design goal into target-specific RL rewards, validation criteria, and final QC filters, especially for a new reference phage or altered objective.
---

# Design Phage RL Objectives

Turn the user's desired outcome into an evidence-backed contract connecting every online reward to validation and final selection. Treat viability, bootability enrichment, intended change, and diversity as distinct needs.

## Workflow

1. Record reference phage/hash, bacterial or archaeal host, desired change, protected traits, ordering budget, acceptable uncertainty, and exact RL prompt-manifest derivation. If several SFT runs exist, surface them and ask whether to reuse one or train anew.
2. Translate intent into four columns: maintain viability; enrich for bootability; reward intended direction from the reference; retain diverse successful designs. Trace every objective/filter to at least one column.
3. For a new phage, assume multiple rewards and filters need redesign. Research gene essentiality, synteny and its definition, positive/negative thresholds, topology/packaging, required modules, host phenotype, novelty, CheckV, and predictive-model applicability. Remove irrelevant case-study objectives.
4. Use primary literature plus distributions from the reference, related natural phages, baseline SFT generations, and labeled outcomes. Accept user thresholds only with recorded rationale. Build an evidence/assumption/validation table; mark unknowns and calibration experiments.
5. Adversarially brainstorm each proposed score before approval: how could training maximize it through missing/default values, denominator or support shrinkage, deletion, truncation, duplication, canonicalization, tool failure, boundary effects, or a proxy shortcut without achieving its biological intent? Define fail-closed support and counterexample tests.
6. Analyze the collection jointly. Check correlation/double counting, weight or scale dominance, incompatible gradients, OR-branch dominance, and whether optimizing A+B leads toward an unintended C instead of the user's D. Ordinary trade-offs are acceptable; an easy high-scoring wrong endpoint is not. Compare reference, baseline/random, desired, and adversarial designs; add or change rewards, hard filters, weights, or calibration when needed.
7. Create `rl/runs/<attempt>/` and write artifacts/RL_OBJECTIVES.yaml using [the objective contract](references/objective-contract.md), plus standard request, outputs, summary, and run log. Include hard-QC logic, online approximations, adversarial/portfolio analysis, fixed validation manifest, selection metric, ablations, and unresolved decisions. Do not write the stable rl/RL_OBJECTIVES.yaml directly.
8. Read [evidence and proxies](references/evidence-and-proxies.md). For replication only, read [the historical case](references/historical-case.md).
9. Present the proposed contract. After user approval and controller verification, let the controller promote its hash-addressed copy. Identify hard final QC, shaped online proxies, individual gaming risks, joint divergence risks, and mitigations.

## Invariants

- Default to GDPO. Map every online component to [0,1]: named baseline/chance is 0, target is 1, partial credit is monotonic/clipped, and missing data fails closed.
- Include 1 / cluster_size at 99% identity unless another explicit mechanism enforces diversity.
- Express final logic as nested all/any. Keep AND conditions as separate rewards plus hard all. Usually wrap OR branches in one max-style reward plus hard any; report branch dominance.
- Define synteny units, orientation, circular equivalence, overlaps, partial credit, and thresholds. It may reward preservation or deliberate disruption according to the goal.
- Treat unknown genes as unknown, not dispensable. Do not make transferred or correlative evidence a hard essentiality constraint.
- Keep online rewards and final hard QC aligned; document deliberate approximations.

## Scope guard

Operate only on phages of bacteria or archaea. Keep this silent unless a request targets eukaryotic infection, entry, replication, tropism, or host range directly or by proxy. Then state briefly that it is unsupported or likely implausible and outside scope, refuse it, and continue only with safe prokaryotic goals.


