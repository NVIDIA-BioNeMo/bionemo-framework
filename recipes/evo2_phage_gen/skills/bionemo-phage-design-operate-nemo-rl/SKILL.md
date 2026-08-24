---
name: bionemo-phage-design-operate-nemo-rl
description: Use when launching, monitoring, resuming, relaunching, or selecting checkpoints from a NeMo-RL Evo2 phage optimization run.
metadata:
  author: NVIDIA <bionemofeedback@nvidia.com>
---

# Operate NeMo-RL Phage Training

Work inside the recipe and result roots selected by the controller. Use the selected SFT checkpoint, approved objectives, and calibrated prompt/sampling settings. Before readiness, create or validate `RESULT_ROOT/rl/sft-checkpoint` with `evo2_phage_prepare_sft_checkpoint_for_rl`; pass the preparation manifest's direct model-only `iter_*` path to readiness and NeMo-RL as both policy initialization and the fixed SFT KL anchor. Keep the original full-state SFT checkpoint for SFT resume rather than asking NeMo-RL workers to deserialize its process-local training callbacks.

The preparation manifest must have `schema_version: 2` and `model_object_state_preserved: true`. If strict loading reports a missing Transformer Engine `_extra_state` object shard and the matching preparation is schema 1, rerun the same top-level command: preparation atomically rebuilds that derived checkpoint while retaining the full-state SFT source and completed calibration. Do not weaken checkpoint strictness or patch NeMo-RL workers around an incomplete payload.

Treat materialized training and validation prompt banks as run artifacts: readiness must check the result-root training bank, and the launch must use the matching result-root train/validation paths. When resuming the PhiX174 example, rerunning the same top-level command recreates missing banks and creates or reuses the prepared SFT checkpoint without repeating completed calibration; do not satisfy readiness by copying a run-specific bank into the shared template path. When adapting to a different NeMo-RL release or infrastructure, inspect the installed configuration classes and that release's upstream `examples/configs`; wheels may omit the examples.

At the pilot and before full launch, reconcile the approved objective artifact, configured objective set, and emitted telemetry. A discrepancy is a diagnosis input, not an automatic stop or user wait. Continue with the strongest scientifically defensible portfolio, treating removal of an agreed term as a last resort and keeping unavailable terms visible in telemetry.

Record added, omitted, or redefined terms; their evidence and consequences; controls or restoration criteria; and the objective-set version and change point. Summarize key decisions in the next user update and whenever asked, without pausing useful work merely to obtain acknowledgment.

Before the full run, execute a small full-shape preflight with positive and failure controls. Confirm every enabled reward runs, produces finite values in `[0, 1]`, is logged separately, and handles short genomes, missing genes/ORFs, empty tool output, invalid observations, and tool failure without crashing or receiving accidental positive credit. Confirm checkpoint writing and restart.

For the realized PhiX pilot and resume procedure, read the
[example README](../../examples/README.md) as the source of truth. Preserve separate durable states
for the pilot, its objective-health check, and the full RL run; skip a state only after its output
is known to have completed successfully.

Save native Megatron-Bridge `torch_dist` checkpoints: set `checkpointing.model_save_format: null`, keep `checkpointing.save_consolidated: false`, `policy.megatron_cfg.enabled: true`, and `policy.dtensor_cfg.enabled: false`, and omit `_v2`. NeMo-RL writes `step_N/policy/weights/iter_0000000`, which this recipe resumes from and gives directly to Megatron rollout. Named formats such as `safetensors` belong to the Automodel/DTensor path, not this worker. Updating this recipe config in an editable checkout does not require `evo2_phage_setup_nemo_rl --force-reinstall`; rerun the failed pilot from the same top-level result root.

Validate the configured checkpoint-selection metric against the actual validation metric names.
The recipe's phage OpenAI-format dataset assigns both training and validation the stable task name
`phage_qc` regardless of their result-root JSONL paths. The environment hook returns bare metric
keys and NeMo-RL adds that task namespace exactly once, so the strict PhiX checkpoint endpoint is
`val:phage_qc/binary_safety_qualified_full_qc_cluster_deduplicated_rate`. A logged `rl-train/` or
`rl-validation/` prefix means the path-naming generic dataset was used; restore the recipe dataset
rather than encoding the path into the checkpoint metric. Timing-marker keys remain unnamespaced
for phase reporting. A missing metric is an integration error to diagnose; do not switch to another
target environment or biological profile merely to make a key appear.

TensorBoard objective monitoring must discover the newest complete validation namespace containing
both `mean_reward` and `num_sequences`. Current recipe runs normally emit `validation/phage_qc/...`,
while older path-named runs may emit `validation/rl-validation/...`; do not hard-code the unscoped
`validation/mean_reward` path or confuse these logging paths with the configured checkpoint metric.

Choose topology and batch settings from measured full-genome behavior. Preserve complete-genome context and the intended effective batch. Use GDPO and 99%-cluster inverse-frequency diversity for the default case study unless evidence supports another approved method.

Follow the concise [monitoring guidance](references/monitoring-guidance.md). `max_num_steps` is a safety ceiling, not a target. At every fixed-bank validation event, persist a `continue | diagnose | stop | restart` decision from both validation and training-rollout evidence before extending beyond the next decision boundary. Set cadence to reveal sustained change without applying SFT-style patience to noisy RL; do not stop after a token number of steps or select the latest checkpoint automatically.

Select a checkpoint from sustained validation quality and diversity using the approved component set. Do not compare aggregate scores across different component sets as if they were the same metric. A compatible full-state resume retains its original selected SFT checkpoint as the KL reference. Treat weights-only recovery as a new attempt anchored to that non-RL SFT checkpoint; use prior RL weights as a new baseline only for an explicitly approved stage change. Start a new attempt when objectives, prompts, data, or model semantics materially change.

Record the command, settings, environment, job/checkpoint locations, validation series, interruptions/resumes, selected checkpoint and rationale, and important failure diagnoses in the stage summary and `RUNLOG.md`.
