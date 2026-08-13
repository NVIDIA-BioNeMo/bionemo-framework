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
3. Prefer an already installed, reviewed NCBI, bioRxiv, repository, or life-science capability before
   generic web/API use. Installing an external plugin requires explicit user approval for that install,
   an allowlisted source, immutable revision, checksum, and recorded review before any acquired
   instructions execute. Never install or run mutable unpinned plugin code. Optional candidates include
   OpenAI's [Life Science Research](https://github.com/openai/plugins/tree/main/plugins/life-science-research)
   and Anthropic's [bio-research plugin](https://github.com/anthropics/knowledge-work-plugins); if no trusted
   installed or approved pinned capability fits, warn once and continue with public APIs/bounded search.
4. Freeze eligibility rules before downloading: taxon and host scope, molecule/completeness, length/ambiguity limits, included record status, exclusions, version policy, and license.

Collection is source-neutral: do not require a natural-origin attestation or categorically reject a
database, metagenomic, local, or generated sequence because of its origin. Preserve source and host
metadata separately. Missing host metadata is recorded and stratified—not invented—and is exclusionary
only when the approved corpus definition actually requires it; conflicting host claims remain
quarantined.

Use the bounded lookup in [the collection contract](references/collection-contract.md). Identify through metadata and a capped prefix before transferring large payloads. Never substitute an HTML page, installer, analysis script, or similarly named record for the claimed FASTA.

If the available tools cannot range-fetch and decompress a candidate, emit the exact bounded validation command, mark the biological payload unverified, and stop before download or use. Metadata, MIME, filename, archive size, and a checksum never substitute for content validation.

Route every long-running or unattended download, decompression, parsing, and validation operation through `bionemo-phage-design-adapt-execution`. Do not treat background launch as completion; keep its due-gated heartbeat active through verified terminal success or failure.

## Acquire and validate

- Download to a staging path; preserve source filenames and archive versions. Retry safely, then verify expected size/checksum and format before atomic promotion.
- Normalize stable record IDs without discarding source accession/version. Validate FASTA headers, nucleotide alphabet, nonempty sequences, declared compression, sequence count, and per-sequence SHA-256.
- Record retrieval query/API request, source URL/record, timestamp, source-provided checksum, local checksum, license, tool version, and exclusions.
- Report total records, valid records, unique sequence hashes, taxonomic/host coverage, lengths, ambiguous bases, and rejection reasons. Do not hide duplicates; SFT preparation performs the definitive exact deduplication and cluster split.
- Two valid modes of increasing the number of viable sequences include looking for other phages that target the same host, or sequence-similar genomes present in various databases.
- Search current life-science sources such as NCBI and, when useful, phage-specific databases. Candidates include [PhageScope](https://phagescope.deepomics.org/) (873,718 phage sequences, including 767,797 nonredundant sequences, with [API access](https://phagescope.deepomics.org/tutorial)), [INPHARED](https://millardlab.org/bacteriophage-genomics/inphared/) (a regularly updated collection of complete phage genomes with linked annotations and metadata useful for type-aware discovery), [PhagesDB.org](https://phagesdb.org/) (as of this writing limited to Actinobacteriophage), [SEA-PHAGES](https://seaphages.org/), [PhageDive](https://phagedive.dsmz.de/), and [OpenGenome2's plasmids/phages collection](https://huggingface.co/datasets/arcinstitute/opengenome2/tree/main/fasta/plasmids_phage). Treat catalog annotations as discovery evidence to verify against source accessions and recorded provenance, not as unquestioned ground truth. This list may age; search current resources and relevant publications when expanding a corpus. Publications from key labs such as the [Bollyky lab](https://bollykylab.com/), the [Phage Foundry](https://phagefoundry.org/team/), or other groups doing phage/host interaction work and phage genome sequencing can also be jumping-off points. The [parent skill's paper and supplement assets](../bionemo-phage-design/assets/literature/king-2025-generative-phage-design/supplement.md) show how prior work collected phage genomes.

Every response, including a read-only research answer, must include a concise provenance block with the exact retrieval query or API request, access date, article and repository versions, license, supplied source checksum, planned local checksum, eligibility rules, and exclusions. Do not defer those fields only to future artifacts.

Predeclare data-adequacy evidence for the intended model and taxon: distinct full-genome hashes and similarity clusters, total nucleotide/token mass, target-relevant taxonomic and similarity coverage, length/composition support, cluster-held-out validation/test support, model scale, and the planned regularization or reuse strategy. Fifteen thousand unique genomes is a useful broad-corpus heuristic from the historical workflow, not a universal biological gate; 10,000 is not a universal stop threshold. If the available collection is materially below the preregistered need (eg under 5000), pause for an explicit go/no-go or a smaller-model/transfer-learning/data-expansion decision. Never inflate support with rotations, versions, exact duplicates, or interaction-matrix rows.

## Handoff

Write the unprefixed collection FASTA(s), metadata table, exclusions, query log, source manifest, and checksums beneath `artifacts/`; finish `OUTPUTS.yaml`, `SUMMARY.md`, and `RUNLOG.md`. When a target is selected, include its exact FASTA path, accession, topology, and hash for the preparation skill's [target-conditioning contract](../bionemo-phage-design-prepare-sft/references/target-conditioning.md). Collection does not add control prefixes. Include exact paths and hashes for `bionemo-phage-design-prepare-sft`, but do not update root stable pointers yourself.

Keep collection eligibility aligned with the approved target and host scope. Quarantine records whose host annotations conflict with frozen rules and report them rather than silently changing scope.
