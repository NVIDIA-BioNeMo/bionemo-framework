# Hydra configs

We could expand this to include configs for full-scale convergence testing, partial conv
experiments, etc.

The `trainer` section gets mapped directly to the `TrainingArguments` class.

Notes:

- Specifying `bf16` in the `TrainingArguments` class will override fp8 settings given to accelerate.
  This causes issues with the `deepspeed` backend, since HF will check to make sure the bf16
  settings are the same between HF and Deepspeed settings.
