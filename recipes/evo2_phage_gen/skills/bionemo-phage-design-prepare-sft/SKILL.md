---
name: bionemo-phage-design-prepare-sft
description: Use when phage genomes must be deduplicated, clustered, split without near-duplicate leakage, and converted into explicit train, validation, and test inputs for Evo 2 supervised fine-tuning.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Prepare Phage SFT

Work inside the recipe and result roots selected by the controller. Create a deterministic cluster-held-out dataset so validation loss measures generalization rather than sequence leakage.

Record the source collection, target genome, eligibility rules, random seed, tool versions, and conditioning scheme. Follow the concise [dataset guidance](references/dataset-guidance.md). Group exact biological equivalents, including circular rotations and reverse complements when topology makes them equivalent, while keeping their source accessions.

Cluster unprefixed full genomes before splitting. The current default uses MMseqs2 at 0.98 minimum identity, 0.8 coverage, and coverage mode 0. Assign whole clusters to train, validation, and test; aim for roughly 98/1/1 and at least 128 validation and test genomes when the corpus permits, while respecting an approved replication size. Keep the reference phage and its near-neighbor cluster in training when RL prompts derive from it.

For one clear target, apply the documented target-similarity conditioning consistently and freeze assignments before splitting. Serialize prefixes only after the unprefixed split is fixed. Run an independent final leakage check with the same full-genome identity/coverage semantics and require zero validation/test matches to training at or above the boundary.

Keep conditioning-prefix tokens in the model input but exclude their next-token labels from SFT loss; verify that the following biological nucleotide remains loss-bearing at document and packed-sample boundaries.

Run the configured sequence-safety screen with its positive controls. Report PASS, FAIL, and INDETERMINATE counts separately, including the main failure classes and a count reconciliation to the input collection. A representative-only optimization must still distinguish unique-sequence counts from source-record counts. These are computational sequence-screening results, not clinical conclusions.

Choose SFT context from the tokenized genome-length distribution plus conditioning and end-of-sequence overhead. Set a provisional training budget from distinct usable genomes, loss-bearing tokens, effective batch, and expected exposures. Treat 12,000 steps as historical context rather than an automatic ceiling, and plan enough validation/checkpoint opportunities to detect under- and overfitting.

Write the explicit train/validation/test inputs, cluster assignments, conditioning settings, exclusions, leakage results, safety report, and serialization settings, plus concise `SUMMARY.md` and `RUNLOG.md`.
