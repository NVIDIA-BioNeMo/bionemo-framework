# Leakage-controlled SFT split

## Deterministic preparation

Preserve source bytes and metadata. Parse records in a stable accession/version order and hash raw files, emitted record metadata, and uppercase sequence bytes. Reject or explicitly handle empty records, non-IUPAC symbols, fragments, and topology conflicts.

For exact deduplication, group byte-identical sequences and, for declared circular genomes, equivalent rotations and reverse-complement rotations. Keep a deterministic representative but map every source record to it. Do not synthesize rotations as additional genomes.

Cluster representative **whole nucleotide genomes**, not genes, proteins, sketches, or random fragments. Resolve current MMseqs2 syntax at runtime, using these invariant semantics:

```yaml
min_seq_id: 0.98
coverage: 0.8
cov_mode: 0
```

Record MMseqs2 version, command/help hash, sensitivity and other resolved defaults, threads, temporary path, input-order hash, representative/member output, and cluster-table hash. A changed tool version or material default creates a new split version.

## Cluster assignment

Assign clusters, never individual members. Use deterministic seeded optimization to approach 98/1/1 by genome count while preserving:

- at least 128 validation and 128 test genomes when attainable;
- target genome and all members of its cluster in train by default;
- comparable length, GC, and taxonomy distributions;
- no empty split and no single source batch isolated as the only validation signal.

If giant clusters make targets impossible, report achieved counts/ratios and the constraint; do not split clusters. Validation and test are distinct cluster sets. Record assignment algorithm/version and seed, plus before/after stratum summaries.

## Independent audit

Re-search every validation and test genome against train as whole nucleotide sequences using identity at least 0.98, coverage at least 0.8, and coverage mode 0. Do not merely trust the clustering membership output. Fail if any boundary-crossing match remains. Also report each held-out genome's closest train hit, identity, query/target coverage, and whether it passed the boundary.

Audit exact hashes and circular/reverse-complement equivalence separately. Report cluster overlap across all three splits and require zero overlap.

## Proposed `SPLIT_MANIFEST.yaml`

```yaml
schema_version: 1
mode: leakage-controlled       # or historical-split
source:
  collection_run: "..."
  fasta_path: "..."
  fasta_sha256: "..."
target:
  record_id: "..."
  accession_version: "..."
  sequence_sha256: "..."
  cluster_id: "..."
dedup:
  policy: exact-circular-rc-aware
  input_count: 0
  unique_count: 0
clustering:
  tool: mmseqs
  version: "..."
  min_seq_id: 0.98
  coverage: 0.8
  cov_mode: 0
  command_sha256: "..."
  assignments_sha256: "..."
split:
  seed: 0
  target_ratios: {train: 0.98, validation: 0.01, test: 0.01}
  counts: {train: 0, validation: 0, test: 0}
  cluster_counts: {train: 0, validation: 0, test: 0}
  files: {}
  file_sha256: {}
leakage_audit:
  tool_version: "..."
  boundary_matches: 0
  report_path: "..."
  report_sha256: "..."
```

Historical reproduction records the original split source and hashes, skips claims of leakage control, and reports any audit that can be performed. Never silently relabel it as the new cluster-held-out policy.
