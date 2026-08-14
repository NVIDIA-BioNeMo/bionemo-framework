# Resource, sequence-length, and OOM policy

This is the single source of volatile memory/parallelism guidance for SFT, RL training, and generation. Re-check current launcher help, runtime tests, and hardware support before applying it; update this file when new endpoints change capabilities.

## Default utilization objective

After correctness and recoverability pass, optimize SFT, RL training, generation, and external-tool
stages separately on representative workloads. A setting that fits or wins in one path is not evidence
for another. Re-probe after any material model, length, precision, topology, tool, database, kernel,
endpoint, hardware, or software-revision change.

### GPU training, RL, and generation

GPU memory occupancy is a constraint, not the objective. On representative target-length inputs, choose
the topology, precision, microbatch, accumulation, and useful concurrency that maximize stable useful
tokens or valid sequences per second while preserving numerical behavior, the approved context and
effective batch, and checkpoint and resume behavior. Record device inventory and occupancy, peak
allocated and reserved memory, throughput, step-time breakdown, communication, host and I/O pressure,
and memory headroom. Use available memory productively, but retain headroom for validation,
checkpointing, data-length variation, and transient allocations.

Prefer the least model-parallel layout, including pure DP, when it fits and is faster; add TP, PP, or CP
only for fit or measured throughput. Pure TP is mainly a latency choice or a fit requirement. Use all
approved devices through useful DP or concurrency, sweeping upward until the next setting fails,
becomes unstable, or loses end-to-end throughput. Select the highest repeatably stable setting, not
the setting with the fullest memory meter.

### External-tool filtering and scoring

For CPU or accelerated bioinformatics tools, collect end-to-end and per-stage timings on representative
target-length controls, both isolated and at the planned concurrency, with a small thread sweep when
useful. Include process or container startup, database loading, and cold versus warm cache behavior.
Choose record workers and per-tool threads by stable total throughput under load, not CPU occupancy or
one isolated call. Bound the full nested task tree against CPU, RAM, memory bandwidth, GPU, scratch,
I/O, database-loading, and concurrent-workload limits; limiting only outer workers is insufficient.
Preserve stable record IDs, ordering, and deterministic input/output mapping under concurrency.

Cache identical results by sequence and full tool/asset/policy/parser identity. Prefer warm, batched, or
persistent processes or containers when supported. Require byte or semantic parity for each accepted
thread, process, container, or accelerator setting. Admit a GPU or other accelerated path only when the
actual operation and database layout are supported, it improves deployment-scale end-to-end throughput,
and the applicable control panel agrees. Do not generalize one tool's result to another stage or the
whole pipeline: tune each measured bottleneck, then remeasure end to end.

For `evo2_phage_sequence_safety scan`, retain `--batch-size 1` until the exact deployed inputs pass
single-record versus proposed-topology equivalence and the complete six-control panel. In this mode,
`--record-workers` is the outer concurrency and `record_workers * --threads` must fit the CPU affinity
visible to the process. `--batch-workers`, `--phrogs-threads`, and `--phrogs-workers` require
`--batch-size > 1`; `--record-workers` is mutually exclusive with that batch mode. Within batch mode,
require `phrogs_workers <= batch_size` and admit only when
`max(batch_workers * threads, batch_workers * phrogs_workers * phrogs_threads)` fits the available CPU
affinity. The CLI may cap batch workers to the actual group count, and the recorded manifest must use
the resolved value. Treat `--timeout` as a bound for each shared command (AMRFinder/DIAMOND) and each
per-record PHROGs command, not one aggregate workflow deadline.
Before a full batched launch, make the equivalence preflight span at least two complete proposed
batches plus a non-empty partial final batch: use at least `2 * batch_size + 1` records unless the
entire input is smaller. Compare exact per-record semantics and exercise terminal manifest
serialization plus `validate-manifest`; detector-output parity from one batch does not test
scan-wide versus batch-local record-index reconciliation. Keep the six-control panel as a separate
required gate.

Keep scanner implementation, policy, inputs, asset manifest, and tool pins immutable from launch
through terminal manifest validation because the manifest binds those identities. Scan publication is
atomic: transient per-record staging directories are neither a supported progress tally nor a resume
surface. Exit 3 is also the CLI's operational-validation status, so classify it as a biological
INDETERMINATE result only when a terminal manifest exists and validates with that state. If publication
or reconciliation fails and no explicit versioned resume contract exists, preserve the failure log and
restart the full attempt; do not promote an ad-hoc replay of private staging artifacts.

Before the control panel, topology preflight, or full scan, resolve and run
`evo2_phage_pin_safety_asset_manifest` into a new attempt-owned directory. Use that pinned manifest for
every gate and retain `PINNING.json`; do not leave a long execution bound to a recipe path in a mutable
checkout. Confirm the pinned recipe and manifest digests again immediately before launch.

Batching shares exact AMRFinder and DIAMOND invocations while PHROGs remains independently parsed per
record. Cache reuse is valid only for identical normalized sequence plus complete asset, policy, tool,
parser, and source identities; preserve stable IDs and never publish split outputs before every shared
command and parser result authenticates. Validate scanner CLI version 2 and output manifest schema 2,
including each shared execution lifecycle `NOT_STARTED`, `FAILED`, or `COMPLETED_AND_PARSED`. A failed
or unstarted command may omit raw output, but its affected records remain `INDETERMINATE` and cannot be promoted. A candidate can pass only after
manifest validation/replay confirms completed parsed evidence for every required class.

NCBI BLAST+ 2.17 has reviewed x86_64 and aarch64 archives and may be benchmarked on either architecture
in isolation. Do not infer full ARM safety support from BLAST alone: the current AMRFinder, HMMER,
DIAMOND, and MMseqs bundle is not fully resolved for aarch64, so `--with-safety` deliberately refuses
non-x86_64 before filesystem mutation. Require a native full-bundle control-panel smoke before changing
that boundary.

The Python-only AMRFinder source-bin override does not establish ARM support or authenticated build
provenance. Until its checkout and build are independently verified, its manifest must label the
repository/revision as operator asserted and bind the staged bytes; do not call it a pinned source
build.

Utilization never outranks full-genome coverage, the approved effective token batch, numerical health,
unbiased sampling, checkpoint/resume integrity, deterministic record mapping, or cluster fairness.

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
