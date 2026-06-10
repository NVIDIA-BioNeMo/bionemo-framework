# Geneformer (TransformerEngine-Optimized) Overview

<!-- NOTE: This model was accelerated using TransformerEngine. What it means in practical terms is the team only modified the format of the checkpoint. No retraining / fine-tuning was done on the original model.-->

## Description:

Geneformer is a foundational transformer model pretrained on a large-scale corpus of single-cell transcriptomes to enable context-specific predictions in settings with limited data in network biology.

This version of the Geneformer model is optimized with NVIDIA's [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) library. It is based on the original Geneformer V1 model, and (within numerical precision) has identical weights and outputs.

This model is ready for commercial/non-commercial use.

## Third-Party Community Consideration

This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see link to Non-NVIDIA Model Card [Geneformer Model Card](https://huggingface.co/ctheodoris/Geneformer).

### License/Terms of Use:

Geneformer is licensed under the [Apache 2.0 license](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md).

### Deployment Geography:

Global

### Use Case:

Network biology and therapeutic discovery, particularly in data-limited settings such as rare diseases or diseases affecting hard-to-access tissues.

### Release Date:

Hugging Face XX/XX/2025 via [LINK](LINK)

## Reference(s):

- [Transfer learning enables predictions in network biology](https://www.nature.com/articles/s41586-023-06139-9.epdf?sharing_token=u_5LUGVkd3A8zR-f73lU59RgN0jAjWel9jnR3ZoTv0N2UB4yyXENUK50s6uqjXH69sDxh4Z3J4plYCKlVME-W2WSuRiS96vx6t5ex2-krVDS46JkoVvAvJyWtYXIyj74pDWn_DutZq1oAlDaxfvBpUfSKDdBPJ8SKlTId8uT47M%3D) - details of the original model trained on ~30 million transcriptomes in June 2021 and the initial report of the in silico perturbation and cell and gene classification strategies.
- [Quantized multi-task learning for context-specific representations of gene network dynamics](https://www.biorxiv.org/content/10.1101/2024.08.16.608180v1.full.pdf) - the expanded model, trained on ~104 million transcriptomes, and continual learning, multitask learning, and quantization strategies.
- See [geneformer.readthedocs.io](https://geneformer.readthedocs.io/) for documentation.

## Model Architecture:

**Architecture Type:** Transformer
**Network Architecture:** BERT

**This model was developed based on:** [Geneformer](https://huggingface.co/ctheodoris/Geneformer) <br>
**Number of model parameters:**

- Geneformer-V1-10M: 10 million parameters
- Geneformer-V2-104M: 104 million parameters
- Geneformer-V2-104M_CLcancer: 104 million parameters (continually pretrained on cancer cells)
- Geneformer-V2-316M: 316 million parameters

## Input:

**Input Type:** Number (Row represents cell, containing gene names and single cell expression counts) <br>
**Input Format:** Array [AnnData](https://anndata.readthedocs.io/en/latest/) <br>
**Input Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Input:**

- Geneformer-V1-10M: Context length of 2048, vocabulary of ~25K protein-coding or non-coding RNA genes
- Geneformer-V2 models (104M, 104M_CLcancer, 316M): Context length of 4096, vocabulary of ~20K protein-coding genes

The input represents rank value encodings where genes are ranked by their expression in each cell scaled by their expression across the entire pretraining corpus.

## Output:

**Output Type:** Dense Embedding Predictions <br>
**Output Format:** Vector <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:** Numeric floating point vector (fp16, bf16, or fp32). The output embeddings encode context-specific gene and cell state representations that can be used for downstream tasks such as cell type annotation, disease classification, and in silico perturbation analysis.

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:

**Runtime Engine(s):**

- Transformer Engine
- PyTorch

**Supported Hardware Microarchitecture Compatibility:**

- A100
- H100
- H200
- GB200

**Preferred/Supported Operating System(s):**

- Linux

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.

## Model Version(s):

- Geneformer-V1-10M
- Geneformer-V2-104M
- Geneformer-V2-104M_CLcancer
- Geneformer-V2-316M

## Training and Evaluation Datasets:

## Training Datasets:

**Link:** [Genecorpus-30M](https://huggingface.co/datasets/ctheodoris/Genecorpus-30M)

**Data Modality:**

- Text (Human single-cell transcriptomes)

**Text Training Data Size:**

- 1 Billion to 10 Trillion Tokens

**Data Collection Method by dataset:**

- Human

**Labeling Method by dataset:**

- N/A

**Properties:**

- **Geneformer-V1-10M:** Trained on Genecorpus-30M (~30 million human single-cell transcriptomes) in June 2021. The dataset excludes cells with high mutational burdens (e.g., malignant cells and immortalized cell lines).

- **Geneformer-V2 models (104M, 316M):** Trained on an expanded corpus of ~104 million human single-cell transcriptomes (non-cancer) in December 2024.

- **Geneformer-V2-104M_CLcancer:** Continually pretrained on ~14 million cancer transcriptomes to yield a cancer domain-tuned model.

The single-cell transcriptomes were assembled from a broad range of publicly available data sources including NCBI Gene Expression Omnibus (GEO), Human Cell Atlas, and Tumor Immune Single-cell Hub (TISCH), among others. Only droplet-based sequencing platforms were included to ensure data comparability. The raw data was converted into a uniform loom HDF5 file format.

## Evaluation Datasets:

**Link:** [A cross-disorder dosage sensitivity map of the human genome](https://zenodo.org/records/6347673)

**Data Collection Method by dataset:**

- Human

**Labeling Method by dataset:**

- Not Applicable <!-- there are no labels for this dataset -->

**Properties:** The data was collected by harmonizing and meta-analyzing rare copy-number variants (rCNVs) from nearly one million individuals across 54 different disorders. This approach created a genome-wide catalog of dosage sensitivity.

**Link:** [Single-cell Transcriptome Analysis Reveals Dynamic Cell Populations and Differential Gene Expression Patterns in Control and Aneurysmal Human Aortic Tissue](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155468)

**Data Collection Method by dataset:**

- Human

**Labeling Method by dataset:**

- Human

**Properties:** The data was collected by performing single-cell RNA sequencing (scRNA-seq) on human ascending aortic tissues. Tissues were obtained from 11 study participants, consisting of 8 patients with ascending thoracic aortic aneurysm (ATAA) and 3 control subjects.

**Link:** [Systematic Comparison of High-throughput Single-Cell and Single-Nucleus Transcriptomes during Cardiomyocyte Differentiation](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE129096)

**Data Collection Method by dataset:**

- Automated

**Labeling Method by dataset:**

- Human

**Properties:** The researchers used two different sequencing platforms to collect data from the same biological process: induced pluripotent stem cell (iPSC) differentiation into cardiomyocytes. The two platforms used were Drop-seq (single-cell) and DroNc-seq (single-nucleus). The study involved two iPSC lines and collected data over a 15-day time period.

**Link:** [A human cell atlas of fetal gene expression](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE156793)

**Data Collection Method by dataset:**

- Human

**Labeling Method by dataset:**

- Hybrid: Human, Automated

**Properties:** The data was collected by profiling the gene expression of millions of single cells from 15 different human fetal organs.

**Link:** [Single-nuclei profiling of human dilated and hypertrophic cardiomyopathy](https://singlecell.broadinstitute.org/single_cell/study/SCP1303/single-nuclei-profiling-of-human-dilated-and-hypertrophic-cardiomyopathy#study-summary)

**Data Collection Method by dataset:**

- Human

**Labeling Method by dataset:**

- Hybrid: Human, Automated

**Properties:** The data was collected by performing single-nucleus RNA sequencing (snRNA-seq) on left ventricle samples from human hearts. The study included samples from 11 hearts with dilated cardiomyopathy, 15 hearts with hypertrophic cardiomyopathy, and 16 non-failing hearts. In total, nearly 600,000 nuclei were sequenced.

## Inference:

**Acceleration Engine:** Transformer Engine, PyTorch

**Test Hardware:**

- A100
- H100
- H200
- GB200

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
