# Compute guidance

Measure the current hardware and run a bounded full-shape smoke before choosing topology. Compare plausible layouts by valid-token or candidate throughput while preserving the scientific semantics.

For SFT, RL, and generation:

- size context from full tokenized genomes plus conditioning/prompt/EOD overhead;
- preserve whole-genome coverage and the intended effective batch;
- keep deterministic record-to-output mapping and requested generation counts; and
- leave headroom for validation, checkpointing, and the transition from training to generation.

When an OOM occurs, first reproduce it and inspect competing processes and peak allocation. Then adjust microbatching, accumulation, parallelism, placement, activation checkpointing, or allocator settings based on measurements. Do not silently truncate genomes, crop prompts, change target length, drop loss-bearing tokens, or weaken QC to make a run fit.

Use an accelerator for an external filter only when the actual tool/database supports it, a realistic benchmark improves throughput, and a control panel agrees with the accepted path. Otherwise use the measured CPU path. Missing or failed required filters remain visible and cannot become a PASS.

For large multi-tool FASTA screens, separately size record or batch workers and each tool's internal threads. Include outer workers, nested subprocess counts, and tool threads in the concurrency budget, and set nested limits explicitly because tool defaults may claim all visible CPUs. Use bounded batches and parallelize serial preparation such as ORF calling when measured throughput improves, but cap the largest simultaneously active worker-times-thread combination to leave headroom. Apply the same accounting to RL environment actors; do not multiply per-batch settings by GPU ranks when a single environment actor performs scoring.

External detectors often alternate short parallel searches with serial setup and result handling, so their thread setting is a ceiling rather than expected constant utilization. When memory permits, make each safety batch large enough to amortize tool/database startup and cover at least one RL generation batch; use the scanner's per-phase timings to tune this on the current node.
