---
name: collect-phage-genomes
description: Use when an Evo 2 phage SFT project needs a reproducible Microviridae or new prokaryotic-virus genome collection from NCBI, paper-linked repositories, supplements, or other public biological databases.
---

# Collect Phage Genomes

Build a licensed, checksummed, provenance-rich collection of usable full genomes. Store dataset-scale FASTA only in the gitignored project result root.

## Choose the source path

Create `genomes/runs/<attempt>/` under the project result root and read the project contract at `../phage-design/references/project-contract.md`.

1. For Microviridae replication, inspect the recipe's current entry points and use its existing downloader first. Resolve the command through `../phage-design/references/command-discovery.md`; do not copy stale syntax.
2. For a new taxon, host, or paper-linked dataset, invoke `research-phage-evidence`. Follow the primary paper's Data Availability statement to stable archive records and exact files. Do not select a repository hit from title similarity alone.
3. Discover specialist NCBI, bioRxiv, repository, or life-science skills before generic web/API use. If none fit, warn once and continue with public APIs and bounded search.
4. Freeze eligibility rules before downloading: taxon and prokaryotic host scope, molecule/completeness, length/ambiguity limits, included record status, exclusions, version policy, and license.

Use the bounded lookup in `references/collection-contract.md`. Identify through metadata and a capped prefix before transferring large payloads. Never substitute an HTML page, installer, analysis script, or similarly named record for the claimed FASTA.

## Acquire and validate

- Download to a staging path; preserve source filenames and archive versions. Retry safely, then verify expected size/checksum and format before atomic promotion.
- Normalize stable record IDs without discarding source accession/version. Validate FASTA headers, nucleotide alphabet, nonempty sequences, declared compression, sequence count, and per-sequence SHA-256.
- Record retrieval query/API request, source URL/record, timestamp, source-provided checksum, local checksum, license, tool version, and exclusions.
- Report total records, valid records, unique sequence hashes, taxonomic/host coverage, lengths, ambiguous bases, and rejection reasons. Do not hide duplicates; SFT preparation performs the definitive exact deduplication and cluster split.

Aim for at least 15,000 usable unique genomes. Explain any shortfall. If the validated unique count is 10,000 or fewer, stop before SFT and obtain explicit acceptance with a warning about coverage and overfitting risk. Never inflate counts with rotations, versions, or exact duplicates.

## Handoff

Write the collection FASTA(s), metadata table, exclusions, query log, source manifest, and checksums beneath `artifacts/`; finish `OUTPUTS.yaml`, `SUMMARY.md`, and `RUNLOG.md`. Include exact paths and hashes for `prepare-phage-sft`, but do not update root stable pointers yourself.

Limit collection to viruses of bacteria or archaea. Reject records with eukaryotic hosts when host scope is known; invoke the controller's safety boundary if the requested design goal crosses that scope.

