---
name: bionemo-phage-design-operate-nemo-rl
description: Use when launching, monitoring, resuming, relaunching, or selecting checkpoints from a NeMo-RL Evo2 phage optimization run.
---

# Operate NeMo-RL Phage Training

Work inside the recipe and result roots selected by the controller. Use the selected SFT checkpoint, approved objectives, and calibrated prompt/sampling settings.

Before the full run, execute a small full-shape preflight with positive and failure controls. Confirm every enabled reward runs, produces finite values in `[0, 1]`, is logged separately, and handles short genomes, missing genes/ORFs, empty tool output, invalid observations, and tool failure without crashing or receiving accidental positive credit. Confirm checkpoint writing and restart.

Choose topology and batch settings from measured full-genome behavior. Preserve complete-genome context and the intended effective batch. Use GDPO and 99%-cluster inverse-frequency diversity for the default case study unless evidence supports another approved method.

Follow the concise [monitoring guidance](references/monitoring-guidance.md). Set a training ceiling and validation/checkpoint cadence that can reveal improvement, collapse, or overfitting; do not stop after a token number of steps or select the latest checkpoint automatically.

Select a checkpoint from sustained validation quality and diversity using the approved component set. Do not compare aggregate scores across different component sets as if they were the same metric. A compatible full-state resume retains its original selected SFT checkpoint as the KL reference. Treat weights-only recovery as a new attempt anchored to that non-RL SFT checkpoint; use prior RL weights as a new baseline only for an explicitly approved stage change. Start a new attempt when objectives, prompts, data, or model semantics materially change.

Record the command, settings, environment, job/checkpoint locations, validation series, interruptions/resumes, selected checkpoint and rationale, and important failure diagnoses in the stage summary and `RUNLOG.md`.
