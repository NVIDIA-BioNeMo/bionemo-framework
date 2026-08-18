---
name: bionemo-phage-design-collect-genomes
description: Use when an Evo 2 phage SFT project needs a reproducible Microviridae or new phage genome collection from NCBI, paper-linked repositories, supplements, or other public biological databases.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Collect Phage Genomes

Work inside the recipe and result roots selected by the controller. Build a licensed, documented collection of usable full genomes.

Define the taxon and host scope, completeness/topology requirements, length and ambiguity limits, record-status policy, exclusions, and license constraints before collection. For the Microviridae replication, inspect and use the recipe's maintained downloader. For a new target, use the research skill and follow primary sources or database records to the relevant sequence files.

Validate and document the collection using the concise [collection guidance](references/collection-guidance.md). Normalize stable record IDs without losing their source IDs, and keep conflicting host annotations separate rather than guessing.

Do not stop collection merely because an arbitrary count was reached. Assess whether the corpus has enough distinct full genomes and similarity clusters, nucleotide/token mass, target-relevant coverage, length/composition support, and cluster-held-out validation/test support for the chosen model and training strategy. Expand the search, use transfer learning, or revise the model plan when the evidence is inadequate. Never inflate support with exact duplicates, circular rotations, accession versions, or interaction rows.

Write the unprefixed FASTA collection, metadata table, exclusions, and source notes under the stage artifacts, plus a concise `SUMMARY.md` and `RUNLOG.md`. When a target is selected, record its FASTA path, accession, and topology for SFT preparation.
