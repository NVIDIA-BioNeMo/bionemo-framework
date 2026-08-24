---
name: bionemo-phage-design-operate-mbridge-sft
description: Use when launching, monitoring, stopping, resuming, or relaunching Evo 2 phage SFT with Megatron Bridge, or when selecting its best validation-loss checkpoint across local, SSH, scheduler, or cloud execution.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Operate Megatron Bridge Phage SFT

Work inside the recipe and result roots selected by the controller. Use the execution skill to resolve current commands and infrastructure.

For a new phage-design project using the 7B family, start from the long-context NGC checkpoint `evo2/7b-1m:1.0` with model size `evo2_7b`; it was trained further than the 8K `_base` checkpoint and remains valid for shorter inputs. A compatible reused checkpoint may retain its recorded family, but do not switch an old or active run from `evo2_7b_base` to `evo2_7b` mid-run. Such a change starts a new result root and SFT-anchored attempt.

Use that approved base or reused checkpoint and the explicit leakage-controlled split. Follow the concise [training guidance](references/training-guidance.md). Size the run with a bounded full-context smoke test rather than reducing the scientific sequence length.

The smoke test should use real train and validation records and show finite loss, parameter updates, checkpoint writing, and restartability. For conditioned data, confirm that prefix targets are masked while the following biological targets remain active. Fix data, masking, or runtime problems before the full run.

Train to the evidence-based ceiling from SFT preparation. Interpret that ceiling in optimizer updates and examples or tokens seen, not epoch count alone: increasing global batch reduces the updates per epoch, so one fast epoch is not evidence of convergence. `max_steps` is a safety ceiling, not a target. At every comparable validation checkpoint, review the complete train/validation curve and persist a `continue | one_more | stop` decision before a supervisor extends or relaunches the run. If the latest comparable validation is still the best and has not rebounded, extend the run—potentially across several epochs—and collect enough validation points to distinguish improvement from a plateau or reversal. Do not choose a token run merely because it is cheaper or copy a publication step count without looking at planned exposure and the validation curve. Monitor training/validation loss, throughput, memory, and failures. Resume a compatible interrupted run from its latest usable checkpoint; start a new attempt when data or model semantics change.

When checkpoint storage is limited, keep the best validation candidates plus a recent resume point; Megatron's native `--most-recent-k` keeps only the newest checkpoints. Align save and validation cadence for metric retention. Missing matches normally warn and fall back to recent retention while validation history continues to be recorded; strict metric mode stops and reports the available raw metric keys. TensorBoard adds ` validation` to the raw loss key. The training guidance describes the built-in, metric-aware, and operator-managed options.

Select the checkpoint by validation loss and curve stability, not training loss or the final step. When metric-aware retention produced `checkpoint_metrics.json`, its `best_checkpoint` field is a convenient path after reviewing the curve; if it is absent, select from `validation_metrics.json` or TensorBoard. Use the held-out test set once to characterize the selected checkpoint, never to select it. Record the command, resolved settings, environment, data inputs, checkpoints, validation curve, selected step and rationale, test result, and interruptions in `RUNLOG.md` and a concise stage summary.

Keep that selected full-state checkpoint for compatible SFT continuation. Before NeMo-RL consumes it, run `evo2_phage_prepare_sft_checkpoint_for_rl --source-checkpoint SELECTED --output-dir RESULT_ROOT/rl/sft-checkpoint`. The idempotent preparation retains tensor and serialized model object state, rewrites the distributed payload without optimizer, scheduler, or RNG state, omits training-state files, nulls serialized process-local callbacks and timers, and records the source, output, sanitization, and sizes in `preparation-manifest.json`. Give RL the manifest's direct `iter_*` path; do not distribute the much larger resumable SFT checkpoint when model-only reuse is the purpose.
