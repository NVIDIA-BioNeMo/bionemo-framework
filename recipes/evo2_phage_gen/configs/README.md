# Evo2 Phage Generation Configurations

This directory contains the maintained inputs for the Microviridae workflow.

- `sft_microviridae_preprocess.yaml` and `sft_microviridae_dataset.yaml` reproduce the
  publication-era random split. The current PhiX case study instead uses
  `evo2_phage_prepare_sft_split`, which writes a cluster-held-out preprocessing config and
  training dataset under the chosen run directory.
- `grpo_phage_megatron.yaml` is a self-contained Evo2/NeMo-RL base configuration.
  `gdpo_phage_megatron.yaml` inherits it and supplies the case-study objectives and training
  settings. Its generation adapter shards the 96-sequence rollout across data-parallel replicas
  and calls Evo2's packed dynamic prefill/batched recurrent decode directly. The separate
  `policy.sequence_packing` switch remains off because it controls gradient-bearing policy and
  loss execution, which requires its own qualification rather than inheriting inference results.
- `arc_genome_design_filtering_local.yaml` configures downstream Arc screening.
- The `phage_safety_*.yaml` files describe the sequence-safety policy, data sources, and
  reference controls.

Generic NeMo-RL examples are not copied into this recipe. When adapting the RL configuration,
identify the NeMo-RL version selected in `pyproject.toml` or the installed environment, then
consult that version's `examples/configs` and configuration classes. Installed wheels may omit
examples, in which case use the matching upstream source checkout. Keep the Evo2 generation
adapter, whole-genome sequence length, selected SFT checkpoint, and objective/QC behavior from
this recipe while adapting infrastructure-specific fields.

Smoke runs and hardware-specific launches should use command-line overrides and a separate result
directory rather than adding permanent example configs here. Record the resolved settings and
observed tool/database versions with the run.
