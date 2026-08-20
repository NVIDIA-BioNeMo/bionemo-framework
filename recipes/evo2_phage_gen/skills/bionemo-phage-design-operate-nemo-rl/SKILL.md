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

Before the full run, execute a small full-shape preflight with positive and failure controls. Confirm every enabled reward runs, produces finite values in `[0, 1]`, is logged separately, and handles short genomes, missing genes/ORFs, empty tool output, invalid observations, and tool failure without crashing or receiving accidental positive credit. Confirm checkpoint writing and restart.

Choose topology and batch settings from measured full-genome behavior. Preserve complete-genome context and the intended effective batch. Use GDPO and 99%-cluster inverse-frequency diversity for the default case study unless evidence supports another approved method.

Follow the concise [monitoring guidance](references/monitoring-guidance.md). Set a training ceiling and validation/checkpoint cadence that can reveal improvement, collapse, or overfitting; do not stop after a token number of steps or select the latest checkpoint automatically.

Select a checkpoint from sustained validation quality and diversity using the approved component set. Do not compare aggregate scores across different component sets as if they were the same metric. A compatible full-state resume retains its original selected SFT checkpoint as the KL reference. Treat weights-only recovery as a new attempt anchored to that non-RL SFT checkpoint; use prior RL weights as a new baseline only for an explicitly approved stage change. Start a new attempt when objectives, prompts, data, or model semantics materially change.

Record the command, settings, environment, job/checkpoint locations, validation series, interruptions/resumes, selected checkpoint and rationale, and important failure diagnoses in the stage summary and `RUNLOG.md`.
