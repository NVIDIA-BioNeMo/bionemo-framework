# Maintained workflow scripts

Most reusable behavior lives in the `bionemo.evo2_phage_gen` package and its installed command-line
entry points. This directory retains two shell helpers for the parallel, resumable calibration grid:

- `calibration/run_sft_sampling_sweep.sh` generates the temperature and prefix-length cells for a
  selected SFT checkpoint.
- `calibration/run_sampling_calibration_scoring.sh` scores those cells, checks similarity to the
  reference and training corpus, and writes the evidence used to select sampling settings.

The [8×H100 PhiX174 example](../examples/README.md) invokes both helpers as part of the complete
agent-free workflow. Their shell tests mirror the directory layout under `tests/scripts/`.
