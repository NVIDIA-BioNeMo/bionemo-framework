# Resource, sequence-length, and OOM policy

This is the single source of volatile memory/parallelism guidance for SFT, RL training, and generation. Re-check current launcher help, runtime tests, and hardware support before applying it; update this file when new endpoints change capabilities.

## Default utilization objective

After correctness and recoverability pass, tune each stage to maximize stable global valid tokens/sec on representative target-length data. Record device inventory/occupancy, peak allocated and reserved memory, throughput, step-time breakdown, host pressure, and safety margin. Prefer the least model-parallel layout, including pure DP, when it fits and is faster even with smaller per-device microbatches; add TP/PP/CP only for memory or measured throughput. Pure TP is mainly a single-request-latency choice or a fit requirement for long context, model, optimizer, policy/reference, or activation state. Use all approved devices through useful DP or concurrency. Increase microbatch or useful concurrency until the next tested setting fails or loses throughput, then use the highest repeatably stable setting with headroom for validation, checkpointing, and data-length variance. Optimize SFT, RL training, and rollout generation separately; fit or topology in one path is not evidence for another.

For CPU/bioinformatics pipelines, bound aggregate process, thread, memory, and I/O demand across the full nested task tree to measured host capacity and stable end-to-end throughput; limiting only outer workers is insufficient.

Utilization never outranks full-genome coverage, the approved effective token batch, numerical health, unbiased sampling, checkpoint/resume integrity, or cluster fairness. Re-probe after any model, length, precision, topology, endpoint, kernel, or software-revision change.

## Agree on context from the genome distribution

After the training genomes are downloaded, compute their tokenized length distribution. If the user has not supplied a context, default to p99.9 as the proposal, prefer the observed maximum when affordable, and obtain agreement before launch.

Calculate `context = align_up(tokens per nucleotide * chosen whole-genome length + worst-case serialization overhead, required alignment)`. Include conditioning/control tokens (including target-similarity controls), prompts, and EOD/special tokens in that overhead. Record the agreed rule, quantile/value, tokenizer rate, overhead, required alignment, final SFT context, final RL context, whole-genome coverage fraction, and model context limit.

Base RL completion length on that target-family distribution unless the user explicitly wants to expand or contract genome length; do not assume a length shift. Size the RL context for prompt plus the agreed completion bound. Select a model/configuration that supports both approved contexts, pad SFT batches to their configured length, and mask pad tokens from loss. Report padding fraction and circular-rotation policy.

## Fit memory in this order

Probe representative target-length examples separately for SFT, RL training, and RL generation; their memory ceilings differ. Capture hardware, model, precision, topology, sequence length, peak memory, and exact OOM before changing config.

1. Start with data parallelism only when the model and target length fit.
2. Maximize a stable microbatch size on target-length data, leaving safety margin.
3. On OOM, reduce microbatch size first and increase gradient accumulation to preserve the approved global sequence/token batch and optimizer-step semantics.
4. If still needed, add tensor parallelism plus data parallelism. Keep TP within the local GPUs per node by default (often 8); crossing nodes adds communication and requires explicit evidence.
5. Add context parallelism later when the current stage/runtime proves support. CP is normally simpler for SFT than RL.
6. Use activation recomputation/checkpointing, optimizer-state sharding, and a supported lower-memory precision when validated; record performance/numerical tradeoffs.

Do not solve OOM by silently shortening either agreed context or reducing the approved global token batch. If preservation is impossible, create a decision entry and new resolved config.

## RL topology split

Prefer DP-only generation when the model and context fit per GPU, otherwise use TP+DP. The current conservative assumption is that CP may work for RL training steps but not inference/generation. Training may use TP+CP+DP only after a capability smoke proves the policy, reference, optimizer, refit, and checkpoint paths agree.

If current inference tests prove CP generation support, disregard the conservative restriction and record the evidence. Do not assume training and generation must share a topology; require tested checkpoint/refit transitions and keep each topology in the resolved runtime contract.

Before full launch, assert the materialized and live topology: world size and TP/PP/CP/DP product, intended GPU allocation, global versus DP-local rollout batch, and unique deterministic seeds/order across DP ranks. Exercise every rank through generation, reward, logprobs, optimizer update, refit, checkpoint, and exact resume. A smaller diagnostic topology cannot silently become the full run.

## Historical evidence

The bundled paper supplement and [historical evidence snapshot](../../bionemo-phage-design/references/historical-evidence.md) preserve prior configurations for replication and interpretation. They are not context defaults. Re-derive context and effective token batch from the agreed genome-length rule, then add future measurements with artifact/config hashes, hardware, software revision, target length, topology, batches, memory, throughput, and acceptance status.

## Acceptance

Run a short target-length preflight before a long job. Accept only when the full workload completes, effective global batch matches the approved target (within its declared tolerance), memory has safety margin, throughput is recorded, and checkpoint plus resume/generation transitions pass. An OOM-driven semantic change creates a new attempt; an unchanged exact resume does not.
