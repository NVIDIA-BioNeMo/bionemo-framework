---
name: bionemo-phage-design-collect-genomes
description: Use when an Evo 2 phage SFT project needs a reproducible Microviridae or new phage genome collection from NCBI, paper-linked repositories, supplements, or other public biological databases.
---

# Collect Phage Genomes

Use the controller's recorded colocated roots. If absent, apply the local [workspace contract](../bionemo-phage-design/references/workspace-contract.md) or stop; never invoke portable bootstrap.

Build a licensed, checksummed, provenance-rich collection of usable full genomes. Store dataset-scale FASTA only in the gitignored project result root.

## Choose the source path

Create `genomes/runs/<attempt>/` under the project result root and read the project contract at `../bionemo-phage-design/references/project-contract.md`.

1. For Microviridae replication, inspect the recipe's current entry points and use its existing downloader first. Resolve the command through `../bionemo-phage-design/references/command-discovery.md`; do not copy stale syntax.
2. For a new taxon, host, or paper-linked dataset, invoke `bionemo-phage-design-research-evidence`. Follow the primary paper's Data Availability statement to stable archive records and exact files. Do not select a repository hit from title similarity alone.
3. Discover specialist NCBI, bioRxiv, repository, or life-science skills before generic web/API use. Optional examples are OpenAI Life Science Research skills in Codex and Anthropic's [bio-research plugin](https://github.com/anthropics/knowledge-work-plugins) in Claude. Neither is required; if none fit, warn once and continue with public APIs and bounded search.
4. Freeze eligibility rules before downloading: taxon and host scope, molecule/completeness, length/ambiguity limits, included record status, exclusions, version policy, and license.

Use the bounded lookup in `references/collection-contract.md`. Identify through metadata and a capped prefix before transferring large payloads. Never substitute an HTML page, installer, analysis script, or similarly named record for the claimed FASTA.

If the available tools cannot range-fetch and decompress a candidate, emit the exact bounded validation command, mark the biological payload unverified, and stop before download or use. Metadata, MIME, filename, archive size, and a checksum never substitute for content validation.

## Acquire and validate

- Download to a staging path; preserve source filenames and archive versions. Retry safely, then verify expected size/checksum and format before atomic promotion.
- Normalize stable record IDs without discarding source accession/version. Validate FASTA headers, nucleotide alphabet, nonempty sequences, declared compression, sequence count, and per-sequence SHA-256.
- Record retrieval query/API request, source URL/record, timestamp, source-provided checksum, local checksum, license, tool version, and exclusions.
- Report total records, valid records, unique sequence hashes, taxonomic/host coverage, lengths, ambiguous bases, and rejection reasons. Do not hide duplicates; SFT preparation performs the definitive exact deduplication and cluster split.

Every response, including a read-only research answer, must include a concise provenance block with the exact retrieval query or API request, access date, article and repository versions, license, supplied source checksum, planned local checksum, eligibility rules, and exclusions. Do not defer those fields only to future artifacts.

Predeclare data-adequacy evidence for the intended model and taxon: distinct full-genome hashes and similarity clusters, total nucleotide/token mass, target-relevant taxonomic and similarity coverage, length/composition support, cluster-held-out validation/test support, model scale, and the planned regularization or reuse strategy. Fifteen thousand unique genomes is a useful broad-corpus heuristic from the historical workflow, not a universal biological gate; 10,000 is not a universal stop threshold. If the available collection is materially below the preregistered need, pause for an explicit go/no-go or a smaller-model/transfer-learning/data-expansion decision. Never inflate support with rotations, versions, exact duplicates, or interaction-matrix rows.

## Handoff

Write the unprefixed collection FASTA(s), metadata table, exclusions, query log, source manifest, and checksums beneath `artifacts/`; finish `OUTPUTS.yaml`, `SUMMARY.md`, and `RUNLOG.md`. When a target is selected, include its exact FASTA path, accession, topology, and hash for the preparation skill's [target-conditioning contract](../bionemo-phage-design-prepare-sft/references/target-conditioning.md). Collection does not add control prefixes. Include exact paths and hashes for `bionemo-phage-design-prepare-sft`, but do not update root stable pointers yourself.

Keep collection eligibility aligned with the approved target and host scope. Quarantine records whose host annotations conflict with frozen rules and report them rather than silently changing scope.
