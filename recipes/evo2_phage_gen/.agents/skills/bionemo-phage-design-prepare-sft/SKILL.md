---
name: bionemo-phage-design-prepare-sft
description: Use when phage genomes must be deduplicated, clustered, split without near-duplicate leakage, manifested, and converted into explicit train, validation, and test inputs for Evo 2 supervised fine-tuning.
---

# Prepare Phage SFT

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Create a deterministic, cluster-held-out dataset so validation loss can reveal overfitting instead of sequence leakage.

## Freeze inputs and policy

Create an SFT preparation attempt under sft/runs/ATTEMPT/. Read ../bionemo-phage-design/references/project-contract.md, references/split-contract.md, [target-conditioning.md](references/target-conditioning.md), ../bionemo-phage-design/references/command-discovery.md, and ../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md. The resource policy is the single authority for full-genome context sizing, padding, OOM response, and parallelism; link to it rather than copying its volatile rules.

Record source FASTA and metadata hashes, target genome identity, eligibility rules, seed, tool versions, and mode before transforming data. Use a validated collection from bionemo-phage-design-collect-genomes. Keep the selected reference phage and its near-neighbor cluster in training by default because RL prompts derive from it; record any contrary scientific reason.

## Build a leakage-controlled split

1. Parse and validate stable IDs and full nucleotide sequences. Group exact biological equivalents, including circular rotations and reverse complements where topology supports that equivalence; retain provenance for every removed duplicate.
2. When one target is unambiguous, apply the target-conditioning contract to the unprefixed collection by default; record an opt-out or alternate policy. Freeze its one-to-one identity/bucket/control-token assignments before splitting.
3. Cluster the unprefixed full genomes with MMseqs2 using minimum sequence identity 0.98, coverage 0.8, and coverage mode 0. Record exact command, version, parameters, input order, and output hashes.
4. Assign entire clusters to approximately 98% train, 1% validation, and 1% test. Target at least 128 genomes in validation and test when dataset size and cluster structure permit, unless an approved replication/user contract fixes other sizes. Use the fixed seed and balance length, GC, taxonomy, and conditioning buckets without splitting a cluster.
5. Verify the target genome hash and its at-least-98%-identity cluster are in train. Produce explicit train, validation, and test FASTA and manifests, then serialize the frozen control prefixes.
6. Run an independent final full-genome leakage audit at the same identity and coverage semantics. Require zero validation or test-to-train matches at or above the boundary; fail preparation if any remain.

Do not allow the shared preprocessing command to choose or randomize the split. Pass explicit pre-split inputs downstream; the existing shared path may use nondeterministic seed or set behavior. In paper replication, preserve requested split sizes when feasible but assign whole clusters; leakage control overrides exact historical membership.

## Plan the training handoff

After the training genomes are downloaded, apply the central resource policy: compute their tokenized length distribution and obtain agreement on the proposed context rule when none was supplied. Record p99/p99.5/p99.9/max, selected bound, tokenizer rate, worst-case control/prompt/EOD overhead, required alignment, final SFT and RL contexts, whole-genome coverage, padding/loss masking, and any explicit RL intent to expand or contract genome length. Otherwise do not assume a length shift. Leave hardware-specific microbatch, accumulation, effective token batch, and model parallelism to bionemo-phage-design-operate-mbridge-sft.

Once the usable leakage-controlled split is final, issue a training-budget feedback decision from unique usable genomes, loss-bearing token mass, intended effective exposures, and a provisional effective batch. Treat 12,000 steps as a historical starting hypothesis, not a fixed maximum: a materially larger corpus may justify a ceiling above 12,000, while redundancy or a smaller corpus may justify less. Record the arithmetic and uncertainty, let the operator recompute after hardware batch resolution, and plan at least 30 comparable validation and recoverable checkpoint opportunities by the proposed ceiling.

Write proposed SFT_PLAN.yaml and SPLIT_MANIFEST.yaml, `CONDITIONING.yaml`, `TARGET_SIMILARITY.tsv`, unprefixed split data, prefixed training records, cluster assignments, exclusion table, leakage report, preprocessing outputs, and the exact training serialization contract (markers/annotations, orientation, tokenizer, BOS/EOS, wrappers, masking) under artifacts/. Finish the standard attempt metadata, OUTPUTS.yaml, SUMMARY.md, and RUNLOG.md. The controller verifies and promotes stable pointers; do not write them directly.
