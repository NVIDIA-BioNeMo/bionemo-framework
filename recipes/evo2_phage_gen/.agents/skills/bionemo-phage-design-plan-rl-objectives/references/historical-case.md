# Historical replication profile

Use these values only for faithful replication of the documented case. Never present them as defaults for another phage or goal. The checked-in [evidence snapshot](../../bionemo-phage-design/references/historical-evidence.md) is the portable audit source; ignored local logs are optional corroboration.

## Filter semantics

- Select filters 1–6, 8, and 9; disable filter 7 for this specific replication.
- Preserve the combined Arc synteny-plus-total-gene rule with accepted (syntenic_genes, total_genes) pairs (10,10), (10,11), (10,12), (11,12), and (12,12).
- Preserve resolved extra-required-gene and DUST gates plus exact config/source hashes.
- Treat literal paper filter-number definitions as diagnostics, not replacements for the resolved implementation.

The exact resolved filter tree, code/config hashes, and source artifact are authoritative; numbers alone are ambiguous.

## Distinct measurements

- Historical best checkpoint: step 190. This is evidence, not a generic stop target.
- Step-190 validation, reported separately: 50/96 raw full-QC passes (52.08%) and 48/96 after 99%-cluster deduplication (50%).
- Step-190 Offline Arc Sequential Final with Architecture Removal disabled: 358/1000 (35.80%).
- Corresponding Full branch with Architecture Removal enabled: 5/1000 (0.50%).

Never conflate validation, offline 1,000-design evaluation, online reward rates, or another checkpoint-selection metric.
