---
name: bionemo-phage-design-generate-and-screen
description: Use when producing, deduplicating, hard-QC screening, clustering, ranking, or selecting final phage designs from a chosen RL checkpoint.
---

# Generate and Screen Phages

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Scale inference to the user's decision, then apply the same versioned hard-QC contract used for RL checkpoint selection. A high reward is not a final design.

## Choose the rollout contract

Resolve one control mode from the brief; in unattended/batch work default to a fixed 1,000-design pilot when no mode is specified, and ask only if the resulting scale or cost is material:

- target number of phages to order;
- exact number of generations; or
- proof of concept: 1,000 default or 10,000 extended.

Confirm checkpoint/prompt lineage, `planning/DESIGN_SPEC.yaml`, complete-genome output and mutable scope, topology, hard-filter profile, 99% uniqueness, budget, and whether defensible ranking exists. Stop if a whole-genome design was silently reduced to a locus/module or if a scope reduction lacks its recorded approval. Select topology through the central [resource policy](../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md). Route long-running or unattended launch and re-entry through bionemo-phage-design-adapt-execution. Obtain commands using [runtime command resolution](references/command-resolution.md) and record actions with [action traceability](references/action-traceability.md). Use ordered intent-named scripts and concise lineage/telemetry/checkpoint/QC pointers.

For an order target, follow [adaptive rollout planning](references/rollout-contract.md): run/reuse a compatible 1,000 pilot, estimate conservative post-QC/post-cluster yield and saturation, and seek 3 times the order count with meaningful ranking or 1.25 times otherwise. Add 1,000-design batches until reserve, budget, or saturation. Stay within the approved compute/budget envelope; ask only before expanding it. Fixed modes generate requested count.

## Execute deterministically

1. Freeze checkpoint, rollout manifest/seeds, objective/filter versions, tools/models, and source/config hashes. Inherit the exact deployed RL prompt strata, mixture weights, and sampling values; use independent IDs/seeds and audit overlap with calibration/RL validation. Any scientific deviation requires explicit rationale and fresh comparable calibration. Derive and behavior-test the exact runtime suite required by enabled filters. Make unattended stage chains single-instance, atomically stateful, tolerant of transient reads, heartbeat-visible, bounded-retry, and resumable.
2. Validate, then canonicalize circular rotations/reverse complements for circular genomes; use biologically appropriate strand equivalence for linear genomes.
3. Exact-deduplicate before expensive tools and preserve raw-to-representative mapping.
4. Run the approved nested hard all/any tree over complete candidates. Recheck whole-genome structure, lifecycle-wide viability signals, harmful-cargo and other [therapeutic-suitability exclusion rules](../bionemo-phage-design/references/ema-2025-draft-phage-therapy-quality-guideline.md) when applicable, and host-range direction independently of any aggregate model score. Keep whole-sequence cargo and lysogeny screens, evidence for strictly lytic behavior, and lifecycle/transduction endpoints separate; unresolved experimental evidence remains indeterminate even when computational screens pass. Mandatory missing/failure fails closed. Record waterfalls, OR overlap, and dominance.
5. Cluster hard-QC passers with MMseqs2 at 99% identity, coverage 0.8, cov_mode=0 unless an approved versioned alternative exists.
6. Rank only passing representatives with predeclared scores; ranking never overrides hard QC or uniqueness.
7. Produce [the reporting contract](references/reporting-contract.md), including accumulation/saturation and final-order manifest.

## Alignment

Recompute when checkpoint, sampling, canonicalization, code, assets, or profile differs. Document every online/final approximation.

Historical step-190 Offline Arc Sequential Final with Architecture Removal disabled was 358/1000 (35.80%); the corresponding Full branch with Architecture Removal enabled was 5/1000 (0.50%). The [checked-in evidence snapshot](../bionemo-phage-design/references/historical-evidence.md) records their provenance. These are profile-specific context, never a forecast.
