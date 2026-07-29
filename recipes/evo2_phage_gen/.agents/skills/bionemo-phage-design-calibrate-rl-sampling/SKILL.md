---
name: bionemo-phage-design-calibrate-rl-sampling
description: Use after selecting an Evo 2 phage SFT checkpoint and defining RL objectives to calibrate prompt serialization, temperature, prefix-length distribution, and fixed validation sampling.
---

# Calibrate Phage RL Sampling

Run a fresh selected-SFT calibration against the approved objective/QC contract before RL. Treat paper settings and prior RL runs as hypotheses, not defaults to copy.

## Verify prompt compatibility

Reconstruct actual SFT training bytes and token IDs: marker or annotation logic, orientation/rotation, tokenizer hash, BOS/EOS, wrappers, padding, masking, and first continuation position. Compare the shared serialized prompt through that boundary byte-for-byte and token-for-token. Use only cues that SFT learned, and carry deliberate SFT protocol deviations into RL. Resolve nonsemantic drift from artifacts; pause only for a biologically or reproducibly material ambiguity.

Create `rl/runs/sampling-calibration-ATTEMPT/` with standard metadata plus immutable `SFT_PROMPT_CONTRACT.yaml`, sweep manifest, selected sampling contract, and validation manifest.

## Sweep the selected SFT

Predeclare paired seeds, target-length generations, minimum cell size, uncertainty/equivalence rule, temperatures, prefix strata, fixed top-k/top-p, QC versions, and cell-completion evidence. Span under-conditioned through copy-prone prefixes and low- through high-entropy temperatures; adapt the frontier instead of assuming the paper range.

Follow the central resource policy to benchmark a bounded set of plausible TP/DP layouts at full length, or reuse a hash-compatible recent benchmark. Choose the highest global valid-token throughput that fits and preserves semantics. Prefer W&B in a calibration-specific project sharing the project-family prefix when authenticated, but never block on it.

Before scoring, derive the required executables, databases, models, and outputs from the enabled objective/QC contract. Pin their exact runtime paths and behavior-test the same scoring environment. Use the target phage as a positive control: every enabled score must be measurable, while its expected value/direction is reasoned from the objective (similarity should usually be strong; diversity or learned-model scores need not be 1). Include failure/no-signal controls. Treat an unexplained missing or fixed-zero metric as an environment/contract fault. Repair and rerun preflight; generation may continue, but selection may not.

Score raw and cluster-deduplicated hard-QC yield, target/tropism and required-function support, invalidity, exact/circular matches to target or training data, reference identity, architecture/synteny change, 99%-cluster diversity, entropy, uncertainty, and filter completeness. Cluster each cell independently when ranking settings; cross-cell clustering is only a proposed-mixture or final-order diagnostic.

## Select and freeze

Exclude cells that lose phage/target signal, collapse, or win through copying. Choose a robust plateau and supported distribution, not a noisy single-cell maximum. Prefer temperature 1.0 when statistically and practically comparable; require material evidence beyond uncertainty to move. Choose the shortest target-retaining prompt stratum plus complementary Pareto strata only when they improve the quality-diversity frontier.

Keep calibration samples separate from a fresh fixed RL-validation bank and final rollout; verify prompt-ID, seed, and generated-sequence-hash non-overlap. Freeze exact training mixture, validation strata, prompt IDs/bytes/tokens, seeds, sampling, completion limits, and hashes. Current Evo2 parallel inference requires one prefix length per decode microbatch: limit Pareto strata and weights to counts that tile the topology's local batches; pure TP may require one length per global validation batch. Group compatible batches without changing weights and smoke-test no fallback. Use the deployed mixture weights in validation and final rollout, and report strata separately.

Exact-resume prior RL only when prompt, sampling, and validation contracts remain compatible. A material change starts a new attempt from the selected non-RL SFT checkpoint. Persist per-cell progress deadlines and bounded retries; diagnose or adapt stalled work instead of waiting indefinitely.
