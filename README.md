<div align="center">
  <h1>BioNeMo Framework</h1>
  <h4>GPU-optimized recipes & toolkits for training transformer models at scale with biological data</h4>
</div>

<div align="left">

[![Click here to deploy.](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/launchable/deploy/now?launchableID=env-2pPDA4sJyTuFf3KsCv5KWRbuVlU)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/NVIDIA/bionemo-framework/pages/pages-build-deployment?label=docs-build)](https://nvidia-bionemo.github.io/bionemo-framework)
[![Test Status](https://github.com/NVIDIA-BioNeMo/bionemo-framework/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/NVIDIA-BioNeMo/bionemo-framework/actions/workflows/unit-tests.yml)
[![Latest Tag](https://img.shields.io/github/v/tag/NVIDIA/bionemo-framework?label=latest-version)](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework/tags)
[![codecov](https://codecov.io/gh/NVIDIA/bionemo-framework/branch/main/graph/badge.svg?token=XqhegdZRqB)](https://codecov.io/gh/NVIDIA/bionemo-framework)

<div align="left">

NVIDIA BioNeMo Framework is a comprehensive suite of programming tools, libraries, and models designed for digital biology. It accelerates the most time-consuming and costly stages of building and adapting biomolecular AI models by providing domain-specific, optimized model recipes and tooling that are easily integrated into GPU-based computational resources with state-of-the-art performance.

<p align="center">
  <img src="docs/docs/assets/images/esm2/esm2_low_precision/esm2_15b_grouped_bars.png" width="600">
  <br>
  <em> Training benchmarks for ESM-2, a well known protein sequence model using the BERT architecture.</em>
</p>

## ⚡ Quick Start

```bash
# Try BioNeMo Recipes in Google Colab (Recommend A100, may be too slow or run out of memory on T4)
# Copy paste into Google Colab cells

!git clone https://github.com/NVIDIA/bionemo-framework.git
cd bionemo-framework/bionemo-recipes/recipes/esm2_native_te/

# Install transformer_engine[pytorch] from source, it takes a long time to install from PYPI
!curl -L -o transformer_engine_torch-2.8.0-cp312-cp312-linux_x86_64.whl "https://drive.google.com/uc?export=download&id=1Oz6dkkIMahv3LN_fQhhQRolZ3m-sr9SF"
!pip install --no-build-isolation transformer-engine transformer_engine_torch-2.8.0-cp312-cp312-linux_x86_64.whl

# Install dependencies
!pip install -r requirements.txt

# Run ESM2 Native Recipes with TE
!python train_ddp.py
```

## Recent News

<p align="center">
  <img src="docs/docs/assets/images/sae.png" width="600">
  <br>
  <em><a href="https://research.nvidia.com/labs/dbr/blog/sae/">Sparse autoencoder feature dashboard for CodonFM 1B</a>, showing learned latent features and their activations on protein sequences.</em>
</p>

- 03/13/2026 [Sparse Autoencoders for model interpretability](bionemo-recipes/interpretability/sparse_autoencoders/) — train and analyze SAEs on biological foundation models. Includes recipes for ESM2 and CodonFM with interactive feature dashboards.
- 03/09/2026 [Qwen2.5 / Qwen3 model](bionemo-recipes/models/qwen/) with TE acceleration, FP8/MXFP8, KV-cache inference, and bidirectional HF checkpoint conversion.
- 03/05/2026 [ESM2 NVFP4 and MXFP8](bionemo-recipes/recipes/esm2_native_te/README.md#low-precision-performance-benchmarks) low-precision training — up to **2,367 TFLOPS/GPU** on NVIDIA B300 at 15B scale with per-layer precision control.
- 02/23/2026 [Mixtral MoE model](bionemo-recipes/models/mixtral/) with TE `GroupedLinear` for efficient parallel expert computation, FP8/FP4 support, and HF conversion.
- 02/13/2026 [ESM2 PEFT recipe](bionemo-recipes/recipes/esm2_peft_te/) for LoRA fine-tuning with sequence packing support.
- 01/14/2026 [Llama3 Context Parallelism](bionemo-recipes/recipes/llama3_native_te/README.md#performance-benchmarks) — scaling Llama 3 70B to 144K context on 36x GB300 NVL36 with ~65% MFU.
- 10/27/2025 [CodonFM recipe](https://github.com/NVIDIA-BioNeMo/bionemo-framework/tree/main/bionemo-recipes/recipes/codonfm_ptl_te) released! This is an accelerated version of the original [research codebase](https://github.com/NVIDIA-Digital-Bio/CodonFM) with [scientific preprint](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf).
- 09/01/2025 [bionemo-recipes](https://github.com/NVIDIA-BioNeMo/bionemo-framework/tree/main/bionemo-recipes) goes live! Lightweight and portable examples with state-of-the-art training performance you can riff on to meet your needs.

## Code Overview

A core use-case of the BioNeMo Framework is to help digital biology scientists accelerate and scale their model training onto a compute cluster. This repository is organized around two complementary areas:

1\. Self-contained models and recipes in [`bionemo-recipes`](./bionemo-recipes/). These examples show different training patterns for biological AI workloads, including native PyTorch, Hugging Face Accelerate, and [NVIDIA megatron-FSDP](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/distributed/fsdp/src), with [NVIDIA TransformerEngine (TE)](https://github.com/NVIDIA/TransformerEngine) acceleration where appropriate.

<details>
<summary><b>(Click to expand) <code>bionemo-recipes</code> support matrix </b></summary>
<small>

| Directory                                      | Description                                                                                                                   | Support Status | 5D Parallel | Megatron-FSDP | TE     | Sequence Packing | FP8    | Context Parallelism |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | -------------- | ----------- | ------------- | ------ | ---------------- | ------ | ------------------- |
| `models/`<br>`amplify`                         | TE accelerated protein BERT, pushed to HuggingFace                                                                            | ✅ Active      | ❌          | ✅            | ✅     | 🚧 WIP           | ✅     | 🚧 WIP              |
| `models/`<br>`esm2`                            | TE accelerated protein BERT, pushed to HuggingFace                                                                            | ✅ Active      | ❌          | ✅            | ✅     | ✅               | ✅     | ✅                  |
| `models/`<br>`llama3`                          | TE accelerated Llama 3                                                                                                        | ✅ Active      | ❌          | 🚧 WIP        | ✅     | ✅               | ✅     | ✅                  |
| `models/`<br>`geneformer`                      | TE accelerated single-cell BERT                                                                                               | 🚧 WIP         | ❌          | ✅            | 🚧 WIP | 🚧 WIP           | 🚧 WIP | 🚧 WIP              |
| `recipes/`<br>`codonfm_ptl_te`                 | Recipe for [CodonFM](https://research.nvidia.com/labs/dbr/assets/data/manuscripts/nv-codonfm-preprint.pdf)'s Encodon using TE | ✅ Active      | ❌          | 🚧 WIP        | ✅     | ✅               | 🚧 WIP | 🚧 WIP              |
| `recipes/`<br>`esm2_accelerate_te`             | Recipe for ESM2 TE + HF Accelerate                                                                                            | ✅ Active      | ❌          | 🚧 WIP        | ✅     | ❌               | ✅     | 🚧 WIP              |
| `recipes/`<br>`esm2_native_te`                 | Recipe for ESM2 TE + native PyTorch                                                                                           | ✅ Active      | ❌          | ✅            | ✅     | ✅               | ✅     | ✅                  |
| `recipes/`<br>`geneformer_native_te_mfsdp_fp8` | Recipe for Geneformer HF model                                                                                                | 🚧 WIP         | ❌          | ✅            | ✅     | ❌               | ✅     | 🚧 WIP              |
| `recipes/`<br>`llama3_native_te`               | Recipe for Llama 3 TE + native PyTorch                                                                                        | ✅ Active      | ❌          | 🚧 WIP        | ✅     | ✅               | ✅     | ✅                  |
| `models/`<br>`mixtral`                         | TE accelerated MoE model                                                                                                      | ✅ Active      | ❌          | 🚧 WIP        | ✅     | ✅               | ✅     | 🚧 WIP              |
| `models/`<br>`qwen`                            | TE accelerated Qwen2.5/Qwen3                                                                                                  | ✅ Active      | ❌          | 🚧 WIP        | ✅     | ✅               | ✅     | 🚧 WIP              |
| `recipes/`<br>`esm2_peft_te`                   | Recipe for ESM2 LoRA fine-tuning                                                                                              | ✅ Active      | ❌          | ❌            | ✅     | ✅               | 🚧 WIP | ❌                  |
| `recipes/`<br>`evo2_megatron`                  | Recipe for Evo2 via Megatron Bridge                                                                                           | 🚧 WIP         | ❌          | ❌            | ✅     | ❌               | ✅     | ❌                  |
| `recipes/`<br>`fp8_analysis`                   | FP8 training analyzer & heatmap tool                                                                                          | ✅ Active      | N/A         | N/A           | N/A    | N/A              | N/A    | N/A                 |
| `recipes/`<br>`vit`                            | Recipe for Vision Transformer                                                                                                 | 🚧 WIP         | ❌          | ✅            | ✅     | ❌               | ✅     | 🚧 WIP              |

</small>

</details>

2\. Reusable BioNeMo libraries in `sub-packages`. These packages are limited to utility functions and biological workflow support, such as shared core interfaces, dataset helpers, I/O, benchmarking, and recipe utilities. They are lightweight, individually installable, and may be used directly in `bionemo-recipes` or in standalone pipelines.

<details>
<summary><b>(Click to expand) <code>sub-packages</code> library overview</b></summary>
<small>

| Directory                     | Description                                                 | Typical Use                                   |
| ----------------------------- | ----------------------------------------------------------- | --------------------------------------------- |
| `bionemo-core`                | Core interfaces, shared data helpers, and PyTorch utilities | Shared foundation for BioNeMo libraries       |
| `bionemo-recipeutils`         | Shared, framework-agnostic utilities and CLIs for recipes   | Used by multiple recipes                      |
| `bionemo-moco`                | Molecular co-design utilities for generative workflows      | Standalone workflows and reusable components  |
| `bionemo-noodles`             | High-performance FASTA I/O wrapper around `noodles`         | Sequence I/O utilities                        |
| `bionemo-scdl`                | Single-cell dataset loading and conversion utilities        | Single-cell data workflows                    |
| `bionemo-scspeedtest`         | Benchmarking utilities for single-cell dataloaders          | Benchmarking and evaluation                   |
| `bionemo-size-aware-batching` | Memory-aware mini-batching utilities                        | Training and data-pipeline helper             |
| `bionemo-webdatamodule`       | `WebDataset` data module utilities                          | Data loading helper for workflows and recipes |

</small>
</details>

BioNeMo Framework is part of a larger ecosystem of NVIDIA Biopharma products. Get notified of new releases, bug fixes, critical security updates, and more for biopharma. [Subscribe.](https://www.nvidia.com/en-us/clara/biopharma/product-updates/)

## Documentation Resources

- **Official Documentation:** Guides, API references, and troubleshooting for the framework are documented on our [official documentation](https://docs.nvidia.com/bionemo-framework/latest/). Nightly builds of this documentation are available on [BioNeMo Framework GitHub Pages](https://nvidia-bionemo.github.io/bionemo-framework/)

- **🚧 In-Progress Documentation 🚧:** `bionemo-recipes` documentation is currently work in progress, however the recipes are meant to be self-documented and easy to understand—we suggest you throw them into your favorite genai code assistant!

## Local Development

Full documentation on using the BioNeMo Framework is provided in our documentation:
<https://docs.nvidia.com/bionemo-framework/latest/user-guide/>. We also publish a container image for the BioNeMo Framework on
[NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/containers/bionemo-framework). To launch a pre-built container, you can use the brev.dev launchable [![ Click here to deploy.](https://uohmivykqgnnbiouffke.supabase.co/storage/v1/object/public/landingpage/brevdeploynavy.svg)](https://console.brev.dev/launchable/deploy/now?launchableID=env-2pPDA4sJyTuFf3KsCv5KWRbuVlU) or execute the following command:

```bash
docker run --rm -it \
  --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/clara/bionemo-framework:nightly \
  /bin/bash
```

### Setting up a local development environment

#### Build the Docker Image Locally

With a locally cloned repository, build the BioNeMo container using:

```bash
docker buildx build . -t my-container-tag
```

If you see an error message like `No file descriptors available (os error 24)`, add the option `--ulimit nofile=65535:65535` to the docker build command.

#### VSCode Devcontainer for Interactive Debugging

We distribute a [development container](https://devcontainers.github.io/) configuration for vscode
(`.devcontainer/devcontainer.json`) that simplifies the process of local testing and development for both `bionemo-recipes` and `sub-packages`. Opening the
bionemo-framework folder with VSCode should prompt you to re-open the folder inside the devcontainer environment.

Packages under `sub-packages` are not installed into that environment automatically. When working on one of them, install it into the active environment with an editable install, for example:

```bash
uv pip install -e ./sub-packages/bionemo-core
uv pip install -e ./sub-packages/bionemo-scdl
uv pip install -e "./sub-packages/bionemo-recipeutils[basecamp]"
```

You can also use `pip install -e ...` if you prefer.

> [!NOTE]
> The first time you launch the devcontainer, it may take a long time to build the image. Building the image locally
> (using the command shown above) will ensure that most of the layers are present in the local docker cache.

### More Examples

See the [tutorials pages](https://docs.nvidia.com/bionemo-framework/latest/user-guide/examples/bionemo-esm2/pretrain/)
for example applications and getting started guides.
