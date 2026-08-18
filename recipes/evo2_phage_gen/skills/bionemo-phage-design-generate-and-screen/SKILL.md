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

1. validate nucleotide alphabet, completion, length, and composition;
2. remove exact biological duplicates, including circular equivalents when applicable;
3. run every required external and internal QC component with its configured positive controls;
4. treat missing required evidence or tool failure as INDETERMINATE rather than PASS;
5. apply the approved target hard-filter profile;
6. cluster at 99% identity for diversity reporting; and
7. rank only when the objective plan defines a defensible ranking.

Short genomes, no predicted genes/ORFs, missing measured genes, and header-only or empty tool results must follow their explicit scientific behavior rather than crash the rollout. Confirm that each enabled filter executed and report any unavailable component; never silently skip a gate.

For the PhiX174 target profile, apply filters 1–6, 8, and 9 with filter 7 disabled. Run the filter-7-enabled diagnostic separately so it cannot overwrite or be confused with the target result.

As the final rollout/report step, score every generated design with the selected pre-RL SFT checkpoint and its intended conditioning prefix. Attach total and mean per-nucleotide log probability to each design, rank by the mean score, and report residual correlation between that score and sequence length. If a strong residual association makes the normalization unreliable, retain the scores and diagnostic but do not use likelihood to order accepted candidates. [Black et al.](https://doi.org/10.64898/2026.06.12.731871) found that Evo 2 likelihood enriched for experimentally bootable PhiX174 designs, supporting within-protocol ranking—not a transferable cutoff or proof of viability.

Write the generated FASTA, per-candidate scores/states, final passing sequences, cluster assignments, and a concise waterfall from generated through PASS/FAIL/INDETERMINATE. Record checkpoint, sampling settings, tool/database versions, commands, counts, selected candidates, and limitations in the stage summary and `RUNLOG.md`. State that computational screening does not establish bootability, host range, therapeutic safety, or efficacy.
