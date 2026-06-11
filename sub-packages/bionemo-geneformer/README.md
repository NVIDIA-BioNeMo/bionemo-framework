# bionemo-geneformer

Geneformer is a foundational single-cell RNA (scRNA) language model using a BERT architecture trained on millions of single-cell RNA sequences. It captures gene co-expression patterns to learn cellular representations, enabling predictive tasks across biology and medicine. Geneformer is trained on a masked language model (MLM) objective, where expression rank-ordered "gene tokens" in single-cell RNA sequences are masked, replaced, or left unchanged, and the model learns to predict these masked genes based on context. This module provides Dataset classes, collators for expression rank ordering, and Config objects for constructing Geneformer-style models.

## Setup

To install, execute the following from this directory (or point the install to this directory):

```bash
pip install -e .
```

To run unit tests, execute:

```bash
pytest -v .
```

## Acquiring Data

Datasets are expected to be in the form of AnnData (.h5ad) objects such as those downloaded from [Cell x Gene | CZI](https://chanzuckerberg.github.io/cellxgene-census/). They are then pre-processed with `sub-packages/bionemo-scdl/src/bionemo/scdl/scripts/convert_h5ad_to_scdl.py`.

## Geneformer-nv 10M and 106M

Refer to the Dataset cards and Model cards to learn more about the pre-trained checkpoints provided for both 10M and 106M of Geneformer-nv.

## See Also

- [sc-DL pypi](https://pypi.org/project/bionemo-scdl/)
- [sc-DL github](https://github.com/NVIDIA/bionemo-framework/tree/main/sub-packages/bionemo-scdl)
