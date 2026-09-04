---
name: bionemo-phage-design-generate-and-screen
description: Use when producing, deduplicating, hard-QC screening, clustering, ranking, or selecting final phage designs from a chosen RL checkpoint.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Generate and Screen Phages

Work inside the recipe and result roots selected by the controller. Generate complete genomes by default; use a locus/module/RBP-only rollout only when that narrower scope was explicitly requested.

Use the selected checkpoint and calibrated prompt mixture, sampling settings, target length, and independent prompt IDs/seeds, following the concise [rollout guidance](references/rollout-guidance.md). For the PhiX174 case study, generate exactly 1,000 designs for the final rollout.

For medium/long generation, use packed dynamic inference for both uniform and mixed-length batches; use static-Flash only when a target-length benchmark shows it wins for equal-length prompts. Pass `--ignore-eos --strict-generation` when an exact target length is required. Use packed `predict` for ragged SFT-likelihood batches. For uniform medium/long scoring, use `--no-sequence-packing` when the target-hardware benchmark favors the rectangular path or when CP is required; TP supports either layout. Do not add `--use-subquadratic-ops` to packed likelihood scoring. Dynamic generation ignores that legacy flag and keeps CUDA-graphed decode; it remains meaningful only for static-Flash generation and rectangular prediction/training.

The maintained GDPO config applies that same exact-length packed dynamic generation through `Evo2MegatronGenerationAdapter` on each data-parallel shard. Its fixed-row decode captures only the physical full/remainder request shapes actually used and reuses them while registered model storage remains stable; an unexpected parameter or buffer address change on any TP/PP/CP rank forces replica-wide recapture without crossing DP replicas, and quantized graphs recapture after every refit even when those addresses remain stable. For heterogeneous prompts, qualify multi-page packed parity, preserve both request IDs and `request_to_mamba_state_idx` across staggered KV-page rollovers, and compare generated selected-action log probabilities with a prediction forward plus reconstructed target-preserving top-k/top-p support, including page-boundary phase checks. Qualify persistence across at least two rollout/offload-refit cycles. Compare cold end-to-end, model-generation, and steady-decode metrics separately when benchmarking. Do not confuse it with NeMo-RL's `policy.sequence_packing`, which controls gradient-bearing policy/loss forwards and stays disabled until separately qualified for Evo2 correctness, memory, and throughput.

Keep BF16 as the portable precision baseline and use a low-precision backend only when the exact checkpoint, GPU family, endpoint, and generation length passed accuracy and CUDA-graph qualification. On Hopper, the 7B model may use regular Transformer Engine FP8 across all compatible linears with `--mixed-precision-recipe bf16_with_fp8_current_scaling_mixed --fp8-all-layers`; without the scope flag, that MBridge recipe retains BF16 first/last blocks. In native dynamic inference, global FP8/FP4 automatically uses layer CUDA graphs even when block scope is requested, because Transformer Engine's quantization state is not block-graph compatible. For globally quantized packed prediction, `--sequence-parallel-policy auto` retains TP but disables SP with current MCore because its padding shim double-reduces row outputs; BF16 TP keeps SP. Use `on` only to requalify a future pad-aware single-reduction implementation with aligned/ragged TP parity and performance A/B tests. After qualification, `phix174_8xh100.sh --hopper-fp8-inference` applies that pair consistently to calibration, final rollout, and likelihood endpoints without changing GDPO training precision. Keep decode capacity at a multiple of eight requests when practical so regular FP8 can use its native aligned-row path without per-layer pad/unpad; packed dynamic prefill may still mix prompt lengths within that batch. FP8-sensitive Vortex checkpoints instead require BF16 global precision plus `--vortex-style-fp8` delayed scaling.

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
