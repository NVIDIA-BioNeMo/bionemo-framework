---
name: bionemo-phage-design-prepare-sft
description: Use when phage genomes must be deduplicated, clustered, split without near-duplicate leakage, manifested, and converted into explicit train, validation, and test inputs for Evo 2 supervised fine-tuning.
---

# Prepare Phage SFT

Create a deterministic, cluster-held-out dataset so validation loss can reveal overfitting instead of sequence leakage.

## Freeze inputs and policy

Create an SFT preparation attempt under sft/runs/ATTEMPT/. Read ../bionemo-phage-design/references/project-contract.md, references/split-contract.md, ../bionemo-phage-design/references/command-discovery.md, and ../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md. The resource policy is the single authority for full-genome context sizing, padding, OOM response, and parallelism; link to it rather than copying its volatile rules.

Record source FASTA and metadata hashes, target genome identity, eligibility rules, seed, tool versions, and mode before transforming data. Use a validated collection from bionemo-phage-design-collect-genomes. Keep the selected reference phage and its near-neighbor cluster in training by default because RL prompts derive from it; record any contrary scientific reason.

## Build a leakage-controlled split

1. Parse and validate stable IDs and full nucleotide sequences. Group exact biological equivalents, including circular rotations and reverse complements where topology supports that equivalence; retain provenance for every removed duplicate.
2. Cluster the full genomes with MMseqs2 using minimum sequence identity 0.98, coverage 0.8, and coverage mode 0. Record exact command, version, parameters, input order, and output hashes.
3. Assign entire clusters to approximately 98% train, 1% validation, and 1% test. Target at least 128 genomes in validation and test when dataset size and cluster structure permit, unless an approved replication/user contract fixes other sizes. Use the fixed seed and balance length, GC, and taxonomy without splitting a cluster.
4. Verify the target genome hash and its at-least-98%-identity cluster are in train. Produce explicit train, validation, and test FASTA and manifests.
5. Run an independent final full-genome leakage audit at the same identity and coverage semantics. Require zero validation or test-to-train matches at or above the boundary; fail preparation if any remain.

Do not allow the shared preprocessing command to choose or randomize the split. Pass explicit pre-split inputs downstream; the existing shared path may use nondeterministic seed or set behavior. In paper replication, preserve requested split sizes when feasible but assign whole clusters; leakage control overrides exact historical membership.

## Plan the training handoff

Train on full phage genomes and derive the intended sequence length through the central resource policy. Record collection length statistics, target length, resulting context, padding and loss masking, and any user override. The paper-scale reference effective batch is 32 times 10,240, or 327,680 tokens per optimizer step; leave hardware-specific microbatch, accumulation, and model-parallel resolution to bionemo-phage-design-operate-mbridge-sft. Set a 12,000-step maximum and enough validation and recoverable checkpoint opportunities for at least 30 of each by that maximum.

Write proposed SFT_PLAN.yaml and SPLIT_MANIFEST.yaml, explicit pre-split data, cluster assignments, exclusion table, leakage report, preprocessing outputs, and the exact training serialization contract (markers/annotations, orientation, tokenizer, BOS/EOS, wrappers, masking) under artifacts/. Finish the standard attempt metadata, OUTPUTS.yaml, SUMMARY.md, and RUNLOG.md. The controller verifies and promotes stable pointers; do not write them directly.
