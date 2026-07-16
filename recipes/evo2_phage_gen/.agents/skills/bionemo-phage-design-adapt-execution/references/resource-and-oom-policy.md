# Resource, sequence-length, and OOM policy

This is the single source of volatile memory/parallelism guidance for SFT, RL training, and generation. Re-check current launcher help, runtime tests, and hardware support before applying it; update this file when new endpoints change capabilities.

## Default utilization objective

After correctness and recoverability pass, tune each stage to maximize stable GPU compute and memory utilization on representative target-length data. Record device inventory/occupancy, peak allocated and reserved memory, throughput, step-time breakdown, host pressure, and safety margin. Increase microbatch or useful concurrency until the next tested setting fails or loses throughput, then use the highest repeatably stable setting with headroom for validation, checkpointing, and data-length variance. Optimize SFT, RL training, and rollout generation separately; a setting that fills memory in one path is not evidence for another.

Utilization never outranks full-genome coverage, the approved effective token batch, numerical health, unbiased sampling, checkpoint/resume integrity, or cluster fairness. Re-probe after any model, length, precision, topology, endpoint, kernel, or software-revision change.

## Preserve the biological workload

Train SFT on full phage genomes, not arbitrary 8K fragments. From the final related-genome training collection, compute the 99th-percentile tokenized genome length using the selected tokenizer. Default training sequence length to at least max(target-phage token length, collection p99), then add required soft prompts/special tokens. Allow an explicit user override with the truncation/coverage consequence recorded.

Pad batches to the configured length and mask pad tokens from loss. Report the length distribution, target coverage, fraction truncated, padding fraction, tokenizer/version, and circular-rotation policy. If p99 is infeasible, revisit model/context/hardware rather than silently fragmenting genomes.

## Fit memory in this order

Probe representative target-length examples separately for SFT, RL training, and RL generation; their memory ceilings differ. Capture hardware, model, precision, topology, sequence length, peak memory, and exact OOM before changing config.

1. Start with data parallelism only when the model and target length fit.
2. Maximize a stable microbatch size on target-length data, leaving safety margin.
3. On OOM, reduce microbatch size first and increase gradient accumulation to preserve the approved global sequence/token batch and optimizer-step semantics.
4. If still needed, add tensor parallelism plus data parallelism. Keep TP within the local GPUs per node by default (often 8); crossing nodes adds communication and requires explicit evidence.
5. Add context parallelism later when the current stage/runtime proves support. CP is normally simpler for SFT than RL.
6. Use activation recomputation/checkpointing, optimizer-state sharding, and a supported lower-memory precision when validated; record performance/numerical tradeoffs.

Do not solve OOM by silently shortening below the approved full-genome context or reducing global token batch. If preservation is impossible, create a decision entry and new resolved config.

## RL topology split

Prefer DP-only generation first, then TP+DP when the model does not fit per GPU. The current conservative assumption is that CP may work for RL training steps but not inference/generation. Therefore generation/rollouts use TP+DP, while training may use TP+CP+DP only after a capability smoke proves the policy, reference, optimizer, refit, and checkpoint paths agree.

If current inference tests prove CP generation support, disregard the conservative restriction and record the evidence. Do not assume training and generation must share a topology; require tested checkpoint/refit transitions and keep each topology in the resolved runtime contract.

## Measured and published reference points

Use these as planning anchors, not fit guarantees. Preserve the exact historical semantics and add future rows only with artifact/config hashes, hardware, software revision, target length, topology, batches, memory, throughput, and acceptance status.

- **Published Evo2 7B phage SFT:** 12,000 iterations on 32 H100 GPUs; sample batch 32; context 10,240; 327,680 tokens per optimizer step; one full genome per sample; masked padding; initial LR 1e-5, 5% linear warmup, then cosine decay to 1e-6. Source: `bionemo-phage-design/assets/literature/king-2025-generative-phage-design/supplement.md`. Adapt the effective token batch, not the sample count in isolation.
- **Historical approximately 6 kb GDPO:** one 2xH100 80 GB node; Evo2 7B Microviridae SFT initialization; bf16; TP2/PP1/CP1; maximum total sequence length 10,240; historical `max_new_tokens=5989`; train MBS1/GBS96; generation and prompt batch 96; logprob batch 1; 96-sample validation every 10 steps; 12 GDPO objectives; LR 1e-6; KL 0.001. A same-shape smoke observed about 68.3 GB/GPU in train generation and 70.9 GB/GPU in validation; the final run itself did not record GPU memory. Its monitor collected evidence through step 250 under a 500-step ceiling. The checked-in [historical evidence snapshot](../../bionemo-phage-design/references/historical-evidence.md) distinguishes empirical observations from configuration facts and preserves source checksums.

The 5,989-token setting describes the executed historical configuration, not a universal exact-total-length formula. Re-derive prompt plus completion accounting for each new prompt contract. A 45-50 kb run requires a fresh capacity probe and will usually need a smaller microbatch; do not extrapolate memory linearly or copy the 6 kb batch.

## Acceptance

Run a short target-length preflight before a long job. Accept only when the full workload completes, effective global batch matches the approved target (within its declared tolerance), memory has safety margin, throughput is recorded, and checkpoint plus resume/generation transitions pass. An OOM-driven semantic change creates a new attempt; an unchanged exact resume does not.
