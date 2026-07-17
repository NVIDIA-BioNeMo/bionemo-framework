---
name: bionemo-phage-design-generate-and-screen
description: Use when producing, deduplicating, hard-QC screening, clustering, ranking, or selecting final phage designs from a chosen RL checkpoint.
---

# Generate and Screen Phages

Scale inference to the user's decision, then apply the same versioned hard-QC contract used for RL checkpoint selection. A high reward is not a final design.

## Choose the rollout contract

Ask for one control mode:

- target number of phages to order;
- exact number of generations; or
- proof of concept: 1,000 default or 10,000 extended.

Confirm checkpoint/prompt lineage, topology, hard-filter profile, 99% uniqueness, budget, and whether defensible ranking exists. Obtain commands using [runtime command resolution](references/command-resolution.md) and record actions with [action traceability](references/action-traceability.md). Use ordered intent-named scripts and concise lineage/telemetry/checkpoint/QC pointers.

For an order target, follow [adaptive rollout planning](references/rollout-contract.md): run/reuse a compatible 1,000 pilot, estimate conservative post-QC/post-cluster yield and saturation, and seek 3 times the order count with meaningful ranking or 1.25 times otherwise. Add 1,000-design batches until reserve, budget, or saturation. Ask before expensive scaling. Fixed modes generate requested count.

## Execute deterministically

1. On the execution host, build/source the recipe environment and verify the
   selected RL checkpoint plus its MBridge inventory.
2. Export that exact selected policy to a fresh vLLM safetensors directory with
   `python -m bionemo.evo2.vllm.export`. Hash and retain `config.json`,
   `model.safetensors.index.json`, `manifest.json`, and all shards. Never
   generate from a stale base-policy export.
3. Discover assigned GPUs and capacity-test a supported profile. The measured
   two-H100 reference is TP2 MP+async O2/balanced with compilation mode 3,
   exact-batch `FULL_AND_PIECEWISE` graphs, and public processed chosen
   logprobs. On larger systems test the TP needed for model/context fit and
   throughput, including TP4/TP8 where supported, then use remaining devices as
   disjoint DP engine groups. Recompute local wave/capture sizes, partition
   prompts and seeds, and prove that no engine groups share a CUDA device.
4. Freeze checkpoint/export, prompt manifest, sampling, seeds, objective/filter versions, tools/models, and source/config hashes. Use stable design IDs.
5. Require exact requested output length, finite aligned chosen logprobs,
   allowed DNA alphabet/no EOS, and unique advancing request seeds before QC.
6. Validate, then canonicalize circular rotations/reverse complements for circular genomes; use biologically appropriate strand equivalence for linear genomes.
7. Exact-deduplicate before expensive tools and preserve raw-to-representative mapping.
8. Run approved nested hard all/any tree. Mandatory missing/failure fails closed. Record waterfalls, OR overlap, and dominance.
9. Cluster hard-QC passers with MMseqs2 at 99% identity, coverage 0.8, cov_mode=0 unless an approved versioned alternative exists.
10. Rank only passing representatives with predeclared scores; ranking never overrides hard QC or uniqueness.
11. Produce [the reporting contract](references/reporting-contract.md), including accumulation/saturation and final-order manifest.

## Alignment

Recompute when checkpoint, sampling, canonicalization, code, assets, or profile differs. Document every online/final approximation.

Historical step-190 Offline Arc Sequential Final with Architecture Removal disabled was 358/1000 (35.80%); the corresponding Full branch with Architecture Removal enabled was 5/1000 (0.50%). The [checked-in evidence snapshot](../bionemo-phage-design/references/historical-evidence.md) records their provenance. These are profile-specific context, never a forecast.
