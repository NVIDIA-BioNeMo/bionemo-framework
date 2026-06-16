# Frequently Asked Questions

## Is BioNeMo Recipes free to use?

Yes, BioNeMo Recipes is free to use. BioNeMo Recipes code is licensed under the Apache 2.0 License. The Apache 2.0
License is a permissive open-source license that allows users to freely use, modify, and distribute software. With this
license, users have the right to use the software for any purpose, including commercial use, without requiring royalties
or attribution. Overall, our choice of the Apache 2.0 License allows for wide adoption and use of BioNeMo Recipes,
while also providing a high degree of freedom and flexibility for users.

## How do I install BioNeMo Recipes?

You can install individual recipe directories from within BioNeMo Recipes by following the corresponding
README pages the [BioNeMo Recipes GitHub](https://github.com/NVIDIA-BioNeMo/bionemo-framework). Please note that this is a
beta feature and may require some additional effort to install seamlessly. We are actively working on testing this
functionality and expect it will be a fully supported feature in future releases. You can review our
[release notes](../about/releasenotes-fw.md) to stay up to
date on our releases.

## What are the system requirements for BioNeMo Recipes?

Generally, BioNeMo Recipes should run on any NVIDIA GPU with Compute Capability ≥8.0. For a full list of supported
hardware, refer to the [Hardware and Software Prerequisites](../getting-started/pre-reqs.md).

## Can I contribute code or models to BioNeMo Recipes?

Yes, BioNeMo Recipes is open source and we welcome contributions from organizations and individuals.
You can do so either by forking the repository and directly opening a PR against our `main` branch from your fork or by
[contacting us](https://www.nvidia.com/en-us/industries/healthcare/contact-sales/) for further assistance. BioNeMo
Recipes' mission is to stay extremely light weight and primarily support building blocks required for various AI
models. As such, we currently prioritize feature extensions, bug fixes, and new independent modules such as dataloaders,
tokenizers, custom architecture blocks, and other reusable features over end-to-end model implementations. We might
consider end-to-end model implementations on a case-by-case basis. If you're interested in this contribution of this
kind, we recommend [reaching out to us](https://www.nvidia.com/en-us/industries/healthcare/contact-sales/) first

For more information about external contributions, refer to the [Contributing](../contributing/contributing.md) and
[Code Review](../contributing/code-review.md) pages.

## How do I report bugs or suggest new features?

To report a bug or suggest a new feature, open an issue on the
[BioNeMo Recipes GitHub site](https://github.com/NVIDIA-BioNeMo/bionemo-framework/issues). For the fastest turnaround,
thoroughly describe your issue, including any steps and/or _minimal_ data sets necessary to reproduce (when possible),
as well as the expected behavior.

## Can I train models in Jupyter notebooks using BioNeMo Recipes?

Most BioNeMo recipes now use native PyTorch or Accelerate-based training loops that work
fine inside notebooks. Some Megatron-based recipes (such as `evo2_megatron`) still require
launching via a script due to process-group initialization requirements. Check the README of
the recipe you want to use for details on supported execution modes.
