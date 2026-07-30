# Target-similarity conditioning

Apply this during SFT curation when the project has a single unambiguous target phage. Default to building the signal, but let the user opt out or revise it; its usefulness is a testable hypothesis, not an assumed result. The method and replication mapping are described in the bundled [original paper](../../bionemo-phage-design/assets/literature/king-2025-generative-phage-design/paper.md) and [supplemental methods](../../bionemo-phage-design/assets/literature/king-2025-generative-phage-design/supplement.md).

1. Freeze the target FASTA, ID/accession, reference hash, topology, and orientation/rotation policy. Work from normalized, unprefixed genomes.
2. Measure every genome against the target with a reproducible whole-genome method. Record identity and coverage separately, including tool/version, parameters, and failure status; missing measurement is not a low-similarity bucket.
3. For case-study replication, reproduce the paper's pinned family token, identity buckets, and control-token mapping. For an adapted run, inspect the observed similarity distribution, propose monotonic bucket edges and tokenizer-valid control tokens with adequate training support, and obtain agreement instead of assuming the paper thresholds transfer.
4. Write `CONDITIONING.yaml` with the target identity, bucket edges, token mapping, method, and hashes. Write `TARGET_SIMILARITY.tsv` with one row per sequence ID/hash, identity, coverage, bucket, and token. Require a complete one-to-one join.
5. Perform exact deduplication, similarity clustering, split assignment, and leakage audits on unprefixed sequences. Only then prepend each frozen assignment to its train/validation/test record. Verify that removing the declared prefix restores the original sequence hash and that split counts and bucket counts reconcile.

Resolve current recipe commands before acting. If no checked-in utility accepts raw FASTA plus the target, implement a deterministic, tested curation command in the selected recipe/copy rather than a one-off shell transformation.

Include the worst-case conditioning prefix in context sizing. Export both conditioning artifacts with the SFT serialization contract; RL prompts must use the same bytes/tokens for the intended bucket. When signal value is uncertain, plan a bounded conditioned-versus-control diagnostic before claiming benefit.
