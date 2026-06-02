# Modular Co-Design (MoCo) Interpolants

MoCo enables abstracted interpolants for building and sampling from a variety of popular generative model frameworks. Specifically, MoCo supports interpolants for both continuous and discrete data types.
[![PyPI version](https://badge.fury.io/py/bionemo-moco.svg)](https://pypi.org/project/bionemo-moco/)

### Continuous Data Interpolants

MoCo currently supports the following continuous data interpolants:

- DDPM (Denoising Diffusion Probabilistic Models)
- VDM (Variational Diffusion Models)
- CFM (Conditional Flow Matching)

### Discrete Data Interpolants

MoCo also supports the following discrete data interpolants:

- D3PM (Discrete Denoising Diffusion Probabilistic Models)
- MDLM (Masked Diffusion Language Models)
- DFM (Discrete Flow Matching)

### Useful Abstractions

MoCo also provides useful wrappers for customizable time distributions and inference time schedules.

### Extendible

If the desired interpolant or sampling method is not already supported, MoCo was designed to be easily extended.

## Installation

For Conda environment setup, please refer to the `environment` directory for specific instructions.

Once your environment is set up, you can install this project by running the following command:

```bash
pip install -e .
```

This will install the project in editable mode, allowing you to make changes and see them reflected immediately.

## Examples

Please see examples of all interpolants in the [examples directory](./examples).
