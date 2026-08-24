# RL monitoring guidance

Run training through a facility whose worker, status, and logs survive the chat process. Retain a stable job identifier, observe startup and early progress, then check meaningful progress until the facility reports terminal success or failure. Reattach to a live job after a new session instead of launching a duplicate.

Treat training rollouts as scientific on-policy observations, not merely optimizer diagnostics. When training and validation differ only by prompts or seeds, both describe candidate quality; the fixed validation bank provides stable longitudinal comparison rather than a biological-label holdout. Read both streams together.

At each training step and comparable validation event, report prompt composition and stratum counts, component means and support/denominators, measurement availability and failures, safety states, hard-QC yield, raw and 99%-clustered diversity, copying, and uncertainty. Inspect prompt-cohort cycles before attributing a sawtooth pattern to policy learning. A positive aggregate must not hide an inactive component.

Discover and record the emitted metric keys and task namespace from the installed runtime. Do not guess names or require a parent framework to know child-owned metrics before they are emitted.

Select checkpoints from sustained, meaningful fixed-bank improvement corroborated by training-rollout behavior. Continue long enough to distinguish noise, prompt composition, temporary degradation, rebound, and collapse, within the approved ceiling. Preserve the latest resumable checkpoint and the best scientific checkpoint.
