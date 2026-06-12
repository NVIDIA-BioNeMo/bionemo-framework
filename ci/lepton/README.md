# Lepton CI

This directory holds code required for triggering automated model convergence runs in Lepton.

## Layout

```text
ci/lepton
├── core
├── model_convergence
├── README.md
└── requirements.txt
```

- `core/`: shared logic for launching Lepton jobs.
- `model_convergence/`: configs and launchers for recipe convergence runs.

## Model Convergence

To run locally:

```bash
python ci/lepton/core/launch_job.py \
  --config-path="../model_convergence/configs" \
  --config-name="recipes/codonfm_ptl_te"
```

The GitHub Action is `.github/workflows/convergence-tests.yml`.
