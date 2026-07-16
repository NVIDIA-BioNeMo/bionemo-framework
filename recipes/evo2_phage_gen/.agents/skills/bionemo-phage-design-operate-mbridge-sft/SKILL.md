---
name: bionemo-phage-design-operate-mbridge-sft
description: Use when launching, monitoring, stopping, resuming, or relaunching Evo 2 phage SFT with Megatron Bridge, or when selecting its best validation-loss checkpoint across local, SSH, scheduler, or cloud execution.
---

# Operate Megatron Bridge Phage SFT

Run SFT from an explicit leakage-controlled split and select the checkpoint with the lowest comparable validation loss. A step limit is a ceiling, not a target.

## Resolve the run

1. Create sft/runs/ATTEMPT/ and read the project contract, ../bionemo-phage-design/references/command-discovery.md, references/sft-operation.md, ../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md, and the proposed SFT and split manifests. Reject missing hashes or an unsuccessful leakage audit.
2. Invoke bionemo-phage-design-adapt-execution; use its environment, storage, ordered-script, monitoring, resume, and centralized resource and OOM contracts. Resolve current recipe entry points from pyproject.toml, --help, source, nearest config and tests, and README. Never launch from a chat-only or historical command.
3. Prefer a public NGC checkpoint through the recipe venv download_bionemo_data command with source ngc. Use evo2/7b-8k:1.0 for paper or case-study SFT, evo2/7b-1m:1.0 for adapted longer-context work, evo2/1b-8k-bf16:1.0 for smoke tests, and evo2/40b-1m-fp8-bf16:1.0 only when requested and hardware supports it. Confirm identifiers at runtime.
4. Treat -bf16 1B and 40B assets as the broadly portable choice for BF16 or FP8 execution; the standard 7B assets do not need an FP8-specific variant. If public NGC is unavailable, use a public Arc checkpoint and a tested NeMo2 or Vortex-to-Megatron-Bridge conversion. Record provider, ID and version, license, source hash, format, converter and version, output hash, and fallback reason.

## Preflight and launch

Run a model, data, and checkpoint smoke test before the full attempt. Validate GPU memory and count, topology, precision, full-genome sequence length, padding and loss masking, microbatch, accumulation, disk, mounts, checkpoint compatibility, and throughput on target-length examples. Two H100 80 GB GPUs are a known useful development configuration and more may help; do not make that a universal requirement.

Apply ../bionemo-phage-design-adapt-execution/references/resource-and-oom-policy.md when fitting memory. In particular, never silently shorten full-genome context or change the approved effective token batch to hide OOM. Record the tested shape and resolved data, tensor, context, and pipeline parallelism.

Target 327,680 effective tokens per optimizer step and choose global batch or accumulation within 5% when feasible. Document the nearest feasible value otherwise. Set at most 12,000 optimizer steps. Schedule at least 30 validation events and 30 recoverable checkpoint saves by that ceiling.

Use local TensorBoard by default. Enable W&B only after the user confirms entity and project; record its run ID and URL but never keys. Keep resolved config, exact ordered scripts, logs, metrics, checkpoints, and monitor events in the attempt.

## Monitor and decide

Follow references/sft-operation.md for validation comparability, sampling uncertainty, phase-aware due gating, patience, sustained-degradation evidence, and bounded rebound observation. Observe both on-disk artifacts and configured telemetry: train and validation loss, learning rate, gradient norm, throughput, GPU utilization and memory, failures, checkpoint integrity, and free space. Promote each new lowest comparable validation-loss checkpoint as best.

Do not early-stop on one blip or on incomparable validation events. Stop at the 12,000-step ceiling or after the monitoring contract establishes sustained overfitting beyond its bounded recovery window. Stop immediately for NaN or Inf, corrupt checkpoint or data, critical disk pressure, or unrecoverable resource failure. Treat OOM as a diagnosis and relaunch decision under the central resource policy rather than as permission to truncate genomes.

Preserve best and latest checkpoints and distinguish selected step from stopping step. Resume only a verified exact training state; changed data, config, topology-incompatible state, or weights-only recovery is a new attempt. Finish OUTPUTS.yaml, SUMMARY.md, and RUNLOG.md with checkpoint evidence and hashes suitable for downstream RL lineage.

