---
name: bionemo-phage-design-collect-genomes
description: Use when an Evo 2 phage SFT project needs a reproducible Microviridae or new phage genome collection from NCBI, paper-linked repositories, supplements, or other public biological databases.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Collect Phage Genomes

Work inside the recipe and result roots selected by the controller. Build a licensed, documented collection of usable full genomes.

Define the taxon and host scope, completeness/topology requirements, length and ambiguity limits, record-status policy, exclusions, and license constraints before collection. For the Microviridae replication, inspect and use the recipe's maintained downloader. For a new target, use the research skill and follow primary sources or database records to the relevant sequence files.

Determine termini class, not merely linear versus circular. GenBank `LOCUS` is submitter-set and biologically unvalidated: OY979411.1 declares `circular`; combined deposit and congener evidence supports a linear Drulisvirus deposited as a DTR-collapsed, circularly permuted repeat unit whose 127-nt overlap is an assembler k-mer artifact, not its roughly 250-nt DTR.

Given a reference genome or de novo assembly plus ligation-based, randomly fragmented reads (paired-end preferred), run [PhageTerm](https://pmc.ncbi.nlm.nih.gov/articles/PMC5557969/) (Garneau et al., *Scientific Reports* 7:8292, 2017) from its maintained [Pasteur GitLab releases](https://gitlab.pasteur.fr/vlegrand/ptv/-/releases). Check library preparation first: transposase/tagmentation preps such as Nextera destroy the termini signal, so “no termini” from those reads is uninformative rather than evidence for headful packaging or permission to rotate. Map reported vocabulary directly:

- Fixed ends; do not rotate: `COS (5')` (lambda) and `COS (3')` (HK97), whose overhang polarity must be preserved for synthesis; `DTR (short)` (T7/Drulisvirus) and `DTR (long)`, which require different length budgets.
- No fixed ends; rotation is valid: `Headful (pac)` (P22, defined pac site) and `Headful` (T4-like, no preferred site). Genuinely circular ssDNA Microviridae such as PhiX174 also permit rotation.
- `Mu-like`: host DNA at both packaged ends from a transposable temperate phage. Reject it for therapeutic design and record both integrative lifestyle and host-DNA carriage.
- `Multiple` and `Unknown` are inconclusive reported outcomes, not tool crashes or permission to assume rotation. Protein-primed inverted terminal repeats (phi29) are fixed ends but may be unresolved by PhageTerm, so retain the sequence ITR check.

Use canonical classifications directly; avoid broad research for standard targets and inspect sources only for ambiguous identities, atypical assemblies, or conflicting evidence.

If reads are absent, use a transposase-based prep, or PhageTerm is inconclusive, check sequence and deposit evidence, cheapest first:

1. Test direct `seq[:k] == seq[-k:]` and inverted `seq[:k] == reverse_complement(seq[-k:])` matches. Flag common assembler k-mers (21, 31, 33, 55, 77, 99, 111, 127), but do not classify from length alone: combine recurrence across unrelated sizes or families with mismatch-tolerant extension, where an abrupt cutoff supports artifact and degraded extension supports a biological repeat. Validate the detector on a published DTR.
2. With no credible repeat, compare relatives: different rotations support headful permutation; consistent endpoints support a fixed-end mechanism such as cos, resolved with lineage and annotation rather than endpoint pattern alone.
3. A collapsed deposit retains one internal DTR. Screen broadly for properly terminated congeners (only 31/166 Drulisvirus deposits retain real DTRs), then map their repeats to recover the DTR and true origin.
4. Record the molecular form: collapsed deposit `[DTR][unique]` is one circularly permuted concatemer repeat unit; packaged genome `[DTR][unique][DTR]` is one DTR longer. Use one form for length, coding density, coordinates, prompts, synteny, and rewards.
5. Established biology classifies canonical ssDNA Microviridae such as PhiX174 as genuinely circular; absence of a terminal repeat is consistent but not sufficient by itself. End CDS markers `<1..N` and `M..>L` with no origin-spanning `join()` support linear annotation.

Do not infer termini from a shared first gene; deposit origins and strands are arbitrary. An ORF-calling pseudo-prefix is representation, not termini evidence.

Validate and document the collection using the concise [collection guidance](references/collection-guidance.md). Normalize stable record IDs without losing their source IDs, and keep conflicting host annotations separate rather than guessing.

Do not stop collection merely because an arbitrary count was reached. Assess whether the corpus has enough distinct full genomes and similarity clusters, nucleotide/token mass, target-relevant coverage, length/composition support, and cluster-held-out validation/test support for the chosen model and training strategy. Expand the search, use transfer learning, or revise the model plan when the evidence is inadequate. Never inflate support with exact duplicates, circular rotations, accession versions, or interaction rows.

Write the unprefixed FASTA collection, metadata table, exclusions, and source notes under the stage artifacts, plus a concise `SUMMARY.md` and `RUNLOG.md`. When a target is selected, record its FASTA path, accession, termini class, and molecular form for SFT preparation.
