# Rollout guidance

Use prompts and seeds independent of calibration and RL validation. Generate the requested number of completed candidates, accounting for failed or duplicate attempts without silently shrinking the denominator.

Use packed dynamic inference for medium/long generation with either uniform or mixed prompt lengths; use static-Flash only when a target-length benchmark shows it wins for an equal-length batch. Require `--ignore-eos --strict-generation` for exact target lengths. Use packed prediction for ragged likelihood batches and preserve record mappings. For uniform medium/long scoring, benchmark `--no-sequence-packing`; the rectangular path also enables CP, while TP supports both layouts.

The recipe's GDPO adapter uses this packed dynamic path directly on every data-parallel rollout shard and suppresses EOS for exact-length whole genomes. Its `policy.sequence_packing` setting is intentionally separate and disabled: that setting changes gradient-bearing policy and loss execution and must not be enabled solely from inference benchmarks.

Treat precision as a qualified deployment choice, not a portable checkpoint property. Hopper supports regular native FP8 for all compatible Transformer Engine linears (appropriate for the 7B model) through `--mixed-precision-recipe bf16_with_fp8_current_scaling_mixed --fp8-all-layers`, but not native MXFP8/NVFP4. The MBridge current-scaling recipe otherwise retains BF16 first/last blocks. In native dynamic inference, global FP8/FP4 automatically resolves requested block CUDA graphs to layer graphs because Transformer Engine's quantization state is not block-graph compatible. For globally quantized packed prediction, `--sequence-parallel-policy auto` retains TP but disables SP with current MCore because its padding shim double-reduces row outputs; BF16 TP keeps SP. Use `on` only to requalify a future pad-aware single-reduction implementation with aligned/ragged TP parity and performance A/B tests. After qualification, the case-study's `--hopper-fp8-inference` option forwards that pair to calibration, final rollout, and packed likelihood scoring while leaving GDPO training precision alone. Keep decode capacity at a multiple of eight requests when practical so regular FP8 can use its native aligned-row path instead of per-layer pad/unpad; packed dynamic prefill may mix prompt lengths within that batch. Use Vortex delayed scaling only for checkpoints whose Hopper behavior requires it. Blackwell may use qualified native MXFP8 or NVFP4 prefill while retaining BF16 decode. Record the exact precision recipe and effective graph scope, and reject experimental full-projection FP4 or low-precision decode when its checkpoint/hardware accuracy evidence is absent.

Validate the raw denominator, retain raw-model scores when requested, deduplicate exact/circular/
reverse-complement biological equivalents, run required safety and hard QC on representatives, and
only then cluster passers in a deterministic candidate order. Representative batching is
acceptable only when record mapping remains complete and the representative result agrees with
controls.

Report raw, biological-representative, hard-QC, and post-QC-cluster counts,
PASS/FAIL/INDETERMINATE denominators, uncertainty when comparing yields, and whether generation or
filtering saturated before forecasting a larger experiment.
