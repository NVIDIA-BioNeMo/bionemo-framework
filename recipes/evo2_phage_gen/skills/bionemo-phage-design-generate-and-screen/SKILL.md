---
name: bionemo-phage-design-generate-and-screen
description: Use when producing, deduplicating, hard-QC screening, clustering, ranking, or selecting final phage designs from a chosen RL checkpoint.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Generate and Screen Phages

Work inside the recipe and result roots selected by the controller. Generate complete genomes by default; use a locus/module/RBP-only rollout only when that narrower scope was explicitly requested.

Use the selected checkpoint and calibrated prompt mixture, sampling settings, target length, and independent prompt IDs/seeds, following the concise [rollout guidance](references/rollout-guidance.md). For the PhiX174 case study, generate exactly 1,000 designs for the final rollout.

Process candidates in this order:

1. retain and validate the complete raw-generation denominator;
2. score every raw design with the selected pre-RL SFT when that ranking evidence is requested;
3. remove exact biological duplicates, including circular and reverse-complement equivalents when applicable;
4. run every required external and internal QC component with its configured positive controls on the representatives;
5. treat missing required evidence or tool failure as INDETERMINATE rather than PASS;
6. apply the approved target hard-filter profile;
7. cluster only the safety-PASS hard-QC set at 99% identity for diversity reporting; and
8. rank cluster representatives only when the objective plan defines a defensible ranking.

Short genomes, no predicted genes/ORFs, missing measured genes, and header-only or empty tool results must follow their explicit scientific behavior rather than crash the rollout. Confirm that each enabled filter executed and report any unavailable component; never silently skip a gate.
When pre-safety QC filters the representative set, retain its exact safety-input FASTA: report excluded representatives separately, and still require one manifest row for every sequence actually submitted to safety.

For the PhiX174 target profile, apply filters 1–6, 8, and 9 with filter 7 disabled. Run the filter-7-enabled diagnostic separately so it cannot overwrite or be confused with the target result.
Disable Arc's internal pre-QC clustering for this final rollout. Run the final MMseqs clustering
after safety and target hard QC with 99% identity, 80% coverage, coverage mode 0, and cluster mode
0, and retain the complete candidate-to-cluster membership table.

As the final rollout/report step, score every generated design with the selected pre-RL SFT checkpoint and its intended conditioning prefix. Attach total and mean per-nucleotide log probability to each design, rank by the mean score, and report residual correlation between that score and sequence length. If a strong residual association makes the normalization unreliable, retain the scores and diagnostic but do not use likelihood to order accepted candidates. [Black et al.](https://doi.org/10.64898/2026.06.12.731871) found that Evo 2 likelihood enriched for experimentally bootable PhiX174 designs, supporting within-protocol ranking—not a transferable cutoff or proof of viability.

Write the generated FASTA, per-candidate scores/states, final passing sequences, cluster assignments, and a concise waterfall from generated through PASS/FAIL/INDETERMINATE. Record checkpoint, sampling settings, tool/database versions, commands, counts, selected candidates, and limitations in the stage summary and `RUNLOG.md`. State that computational screening does not establish bootability, host range, therapeutic safety, or efficacy.
For long final rollouts, keep independently validated completion markers for raw generation,
deduplication, likelihood, safety, target and diagnostic branches, final clustering, and reporting so
a restart reuses only scientifically complete evidence.
