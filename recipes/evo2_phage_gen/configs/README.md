# Evo2 Phage Generation Configurations

This directory contains the supported configuration surface for a Microviridae end-to-end run.

## Runtime Entry Points

- `sft_microviridae_preprocess.yaml` prepares the released Microviridae SFT data.
- `sft_microviridae_dataset.yaml` defines the corresponding SFT dataset inputs.
- `gdpo_phage_megatron.yaml` is the production RL entry point used by the replication workflow.
- `arc_genome_design_filtering_local.yaml` runs Arc nucleotide QC on generated prompt-sweep or
  rollout FASTA files.
- `phage_safety_policy.yaml`, `phage_safety_assets.yaml`, and
  `phage_safety_reference_controls.yaml` define the mandatory sequence-safety policy and assets.

## RL Inheritance

`gdpo_phage_megatron.yaml` inherits from `grpo_phage_megatron.yaml`, which in turn inherits from
`nemo_rl_defaults/grpo_math_1B_megatron.yaml` and `nemo_rl_defaults/grpo_math_1B.yaml`. The GRPO
file is therefore part of the supported configuration stack and can also serve as the scalar-reward
alternative, but the replication workflow launches the GDPO entry point.

One-step smoke tests and historical diagnostics should use command-line overrides and a dedicated
results directory instead of adding permanent configs here. This keeps the visible configuration
surface focused on reproducible end-to-end use.

Full runs may materialize launch-specific overlays for absolute paths, hardware topology, prompt
artifacts, and resume offsets. Store those resolved overlays with the run metadata; they should
inherit from `gdpo_phage_megatron.yaml` rather than becoming additional canonical configs here.
