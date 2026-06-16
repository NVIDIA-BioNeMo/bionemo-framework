## Model Overview

### Description:

Evo 2 is a genomic foundation model that enables prediction and generation tasks from the molecular to genome scale. At 40 billion parameters, the model understands the genetic code for all domains of life and is the largest AI model for biology to date. Evo 2 was trained on a dataset of nearly 9 trillion nucleotides.

This model is ready for commercial use.

### Third-Party Community Consideration

This model is not owned or developed by NVIDIA. This model has been developed and built to a third-party's requirements for this application and use case; see [Arc Institute website](https://arcinstitute.org).

### License/Terms of Use:

GOVERNING TERMS: The NIM container is governed by the [NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/) and [Product-Specific Terms for AI Products](https://www.nvidia.com/en-us/agreements/enterprise-software/product-specific-terms-for-ai-products/). Use of this model is governed by the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/). ADDITIONAL INFORMATION: [Apache 2.0 License](https://github.com/ArcInstitute/evo2/blob/main/LICENSE).

You are responsible for ensuring that your use of NVIDIA AI Foundation Models complies with all applicable laws.

### Deployment Geography:

Global

### Use Case:

Evo is able to perform zero-shot function prediction for genes. Evo also can perform multi-element generation tasks, such as generating synthetic CRISPR-Cas molecular complexes. Evo 2 can also predict gene essentiality at nucleotide resolution and can generate coding-rich sequences up to 1 million in length. Advances in multi-modal and multi-scale learning with Evo provides a promising path toward improving our understanding and control of biology across multiple levels of complexity.

### Release Date:

2/19/2025

### Acknowledgements:

```
@article{nguyen2024sequence,
   author = {Eric Nguyen and Michael Poli and Matthew G. Durrant and Brian Kang and Dhruva Katrekar and David B. Li and Liam J. Bartie and Armin W. Thomas and Samuel H. King and Garyk Brixi and Jeremy Sullivan and Madelena Y. Ng and Ashley Lewis and Aaron Lou and Stefano Ermon and Stephen A. Baccus and Tina Hernandez-Boussard and Christopher Ré and Patrick D. Hsu and Brian L. Hie },
   title = {Sequence modeling and design from molecular to genome scale with Evo},
   journal = {Science},
   volume = {386},
   number = {6723},
   pages = {eado9336},
   year = {2024},
   doi = {10.1126/science.ado9336},
   URL = {https://www.science.org/doi/abs/10.1126/science.ado9336},
}
```

```
@article {merchant2024semantic,
   author = {Merchant, Aditi T and King, Samuel H and Nguyen, Eric and Hie, Brian L},
   title = {Semantic mining of functional de novo genes from a genomic language model},
   year = {2024},
   doi = {10.1101/2024.12.17.628962},
   publisher = {Cold Spring Harbor Laboratory},
   URL = {https://www.biorxiv.org/content/early/2024/12/18/2024.12.17.628962},
   journal = {bioRxiv}
}
```

### Model Architecture:

**Architecture Type:** Generative Neural Network

**Network Architecture:** StripedHyena

### Input:

**Input Type(s):** Text

**Input Format(s):** DNA Sequence (String)

**Input Parameters:** One-Dimensional (1D)

### Output:

**Output Type(s):** Text

**Output Format:** DNA Sequence (String)

**Output Parameters:** One-Dimensional (1D)

### Software Integration:

**Runtime Engine(s):**

- PyTorch
- Transformer Engine

**Supported Hardware Microarchitecture Compatibility:**

- NVIDIA Hopper

**[Preferred/Supported] Operating System(s):**

- Linux

### Model Versions:

The following model versions are available to download through our CLI:

- `evo2/1b-8k:1.0` is a NeMo2 format model converted from [arcinstitute/savanna_evo2_1b_base](https://huggingface.co/arcinstitute/savanna_evo2_1b_base) which is a 1B parameter
  Evo2 model pre-trained on 8K context genome data.
- `evo2/1b-8k-bf16:1.0` is a fine-tuned variant of `evo2/1b-8k:1.0` that performs well with BF16 precision.
- `evo2/7b-8k:1.0` is a NeMo2 format model converted from [arcinstitute/savanna_evo2_7b_base](https://huggingface.co/arcinstitute/savanna_evo2_7b_base)
  which is a 7B parameter Evo2 model pre-trained on 8K context genome data.
- `evo2/7b-1m:1.0` is a NeMo2 format model converted from [arcinstitute/savanna_evo2_7b](https://huggingface.co/arcinstitute/savanna_evo2_7b) which is a
  7B parameter Evo2 model further fine-tuned from [arcinstitute/savanna_evo2_7b_base](https://huggingface.co/arcinstitute/savanna_evo2_7b_base) to support 1M context
  lengths.

The following Savanna format checkpoints are also available on HuggingFace and may be converted to NeMo2 format following
the steps in our Evo2 fine-tuning example notebook:

- [arcinstitute/savanna_evo2_40b_base](https://huggingface.co/arcinstitute/savanna_evo2_40b_base) - A 40B model trained
  on 8K context data.
- [arcinstitute/savanna_evo2_40b](https://huggingface.co/arcinstitute/savanna_evo2_40b) - A 40B model fine-tuned from [arcinstitute/savanna_evo2_40b_base](https://huggingface.co/arcinstitute/savanna_evo2_40b_base)
  to support 1M context data.

## Training, Testing, and Evaluation Datasets:

Multiple datasets were used for training, testing and evaluation (see details below).

_OpenGenome_
Link: [Sequence modeling and design from molecular to genome scale with Evo](https://www.science.org/doi/10.1126/science.ado9336)
Data Collection Method: Automatic/Sensors/Human
Labeling Method by dataset: Automatic
The previously published OpenGenome dataset was used in its entirety as part of the training data for this study. This included representative prokaryotic genomes available through GTDB release v214.1, and curated phage and plasmid sequences retrieved through IMG/VR and IMG/PR.

_Updated GTDB prokaryotic genomes_
Link: [GTDB: an ongoing census of bacterial and archaeal diversity through a phylogenetically consistent, rank normalized and complete genome-based taxonomy](https://academic.oup.com/nar/advance-article/doi/10.1093/nar/gkab776/6370255)
Data Collection Method: Automatic/Sensors/Human
Labeling Method by dataset: Automatic
New prokaryotic reference genomes made available through the GTDB release 220.0 update were added to the training data for this study. New genomes were identified by selecting all species' reference genomes that had no previously published (release 214.1) genomes within their species cluster, resulting in 28,174 additional prokaryotic genomes.

_NCBI Eukaryotic reference genomes_
Link: [Mash: fast genome and metagenome distance estimation using MinHash](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0997-x)
Data Collection Method: Automatic/Sensors/Human
Labeling Method by dataset: Automatic
All available eukaryotic reference genomes were downloaded from NCBI on 05/32/2024, excluding atypical genomes, metagenome-assembled genomes, and genomes from large multi-isolate projects. This resulted in 16,704 genomes including an estimated ~10.7 trillion nucleotides. Only contigs that were annotated as 'Primary Assembly', 'non-nuclear', or 'aGasCar1.hap1' (an aberrant annotation that applied only to GCA_027917425.1) were retained. Mash sketch was run on each individual genome with the flag "-s 10000" and the mash distance was calculated between all genomes as an estimate for their pairwise 1-ANI (average nucleotide identity). All genomes with a mash distance < 0.01 were joined with edges in a graph, and clusters were identified by finding connected components. One representative genome per cluster was chosen, prioritizing genomes with a higher assembly level and genomes with longer total sequence length. This clustering resulted in 15,148 candidate genomes. Genomes were further filtered by removing ambiguous nucleotides at the termini of each contig, by removing regions annotated as "centromere" in an available GFF file, and by removing contigs that were less than 10 kb in total length. Finally, contigs that were composed of more than 5% ambiguous nucleotides were removed. This final filtered set included 15,032 genomes and 6.98 trillion nucleotides.

_Bridge Metagenomic Data_
Link: [Bridge RNAs direct programmable recombination of target and donor DNA](https://www.nature.com/articles/s41586-024-07552-4)
Data Collection Method: Automatic/Sensors/Human
Labeling Method by dataset: Automatic
A previously described metagenomics dataset was further curated as part of the training data. This included 41,253 metagenomes and metagenome-assembled genomes from NCBI, JGI IMG, MGnify, MG-RAST, Tara Oceans samples, and Youngblut et al. animal gut metagenomes. All contigs were split at consecutive stretches of ambiguous nucleotides of length 5 bp or longer, the split contigs were filtered by a minimum sequence length of 1 kb, and only contigs with at least one open reading frame as predicted by prodigal were kept. Contig-encoded proteins were previously clustered at 90% identity using MMseqs. To further remove redundant sequences, contigs were sorted by descending length, and each contig was only retained if at least 90% of its respective protein clusters were not already in the sequence collection (determined using a bloom filter).

_NCBI Organelle_
Link: [NCBI Organelle Genome Data Package](https://www.ncbi.nlm.nih.gov/datasets/organelle/?taxon=2759)
Data Collection Method: Automatic/Sensors/Human
Labeling Method by dataset: Automatic
Eukaryotic organelle genomes: (at the time of data query) 33,457 organelle genomes were identified and downloaded using the "NCBI Organelle" web resource. Ambiguous nucleotides at the terminal ends of the organelle genome sequences were removed. Sequences that had over 25 ambiguous nucleotides were removed. This resulted in 32,241 organelle genomes that were used for training, including 1,613 mitochondria, 12,856 chloroplasts, 1,751 plastids, 18 apicoplasts, 1 cyanelle, and 1 kinetoplast.

### Inference:

**Engine:** PyTorch, Transformer Engine

**Test Hardware:**

Evo2 NIM:

- H200 (1 and 2 GPU configurations, 144 GB each)
- H100 (2 GPU configuration, 80 GB each)

BioNeMo Framework:

- A100 (1, 8, ..., 1024 GPU configurations)
- H100 (1, 8, ..., 2048 GPU configurations)
- A6000 (1, 2 GPU configurations (standard developer environment))
- A5880 (1, 2 GPU configurations (alternative developer environment))
- L40S (1 GPU configuration (brev.dev standard configuration))

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Users are responsible for ensuring the physical properties of model-generated molecules are appropriately evaluated and comply with applicable safety regulations and ethical standards.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

## Benchmarking

### Performance vs Context Length

With the current implementation of Evo2, we do not have the heavily optimized kernels in place for convolution operators like we do for
attention layers in a model like Llama 2. Even with this shortcoming, we see that the benefit from including more convolutional layers
makes up for the earlier stage of optimization at around the 64K context length. Beyond that point we see an improvement
in performance even compared to a highly optimized transformer model.

![Evo2 becomes faster than Llama 2 beyond around 64K context length in this version](../assets/images/evo2/evo2_vs_llama2_performance_vs_context_length.png)

Comparing model sizes, our benchmarks show the 7B variant processes approximately 4.9x more tokens per step than the 40B variant across tested configurations. When scaling from 8K to 1M sequence length, throughput decreases as expected, with the 7B model processing 9.7x fewer tokens per step and the 40B model processing 8.9x fewer tokens per step at the longer context length:

![Evo2 7B vs 40B performance by sequence length](../assets/images/evo2/evo2_vs_7b_40b_performance_vs_context_length.png)

### Performance vs Cluster Size

Performance scales linearly with a very small overhead on a cluster with fast interconnects.
![Evo2 linear scaling with increasing number of GPUs](../assets/images/evo2/evo2_performance_by_cluster_size.png)

## Accuracy

## Zeroshot BRCA1 VEP

To evaluate Evo 2's accuracy, we replicated Arc's zero-shot variant effect prediction experiment on the BRCA1 gene using the [Findlay et al. (2018) dataset](https://www.nature.com/articles/s41586-018-0461-z) of 3,893 SNVs. The experiment tests the model's ability to predict if single nucleotide variants disrupt protein function (potentially increasing cancer risk) by analyzing experimentally determined function scores that categorize variants as LOF, INT, or FUNC based on their degree of functional disruption.

Evo 2 is used to score the likelihood probabilities of both reference and variant sequences for each single nucleotide variant:

![Evo2 zeroshot BRCA1 strip plot](../assets/images/evo2/evo2_zeroshot_brca1_stripplot.png)

Performance evaluation across multiple Evo 2 model variants was conducted by computing likelihood scores for reference and variant sequences of each single nucleotide variant (SNV), with AUROC scores shown in the following table:

| Model                                                                                                    | AUROC |
| -------------------------------------------------------------------------------------------------------- | ----- |
| [Arc Evo 2 1B](https://github.com/ArcInstitute/evo2/blob/main/notebooks/brca1/brca1_zero_shot_vep.ipynb) | 0.73  |
| BioNeMo Evo 2 1B                                                                                         | 0.76  |
| BioNeMo Evo 2 7B                                                                                         | 0.87  |

## Training diagnostics

### 7b training equivalence with NV model variant

For this test we demonstrate that our NV model variant has similar architecture, but uses gelus activations in the hyena
layers as was originally intended, as well as convolutional bias in the short hyena convolutions as well as the medium
and long layers. These changes result in a model that has similar training dynamics early in the process, but may
have improved stability.
![7b training with bionemo reaches 1.08 loss in 28k steps](../assets/images/evo2/evo2_bionemo_7bnv_28ksteps.png)

As a baseline we compared to the original training run of Evo2 7b in the Savanna codebase [here on W&B](https://api.wandb.ai/links/hyena/opmbhm1c)
![7b training with savanna reaches 1.075 loss in 28k steps](../assets/images/evo2/evo2_savanna_7b_28ksteps.png)

### 1b training equivalence (same setup)

We trained a 1b model with the same configuration as was used by savanna. We achieve a largely similar training curve
for the first 6950 steps.
![7b training with bionemo reaches 1.2 loss in 6,950 steps](../assets/images/evo2/evo2_bionemo_1b_6950steps.png)
As a baseline we compared to a to the original training run of Evo2 1b in the Savanna codebase [here on W&B](https://api.wandb.ai/links/hyena/yebyphwe).
Around step 6950 this run had a loss of between 1.18 and 1.2.
![7b training with savanna also reaches 1.2 loss in 6,950 steps](../assets/images/evo2/evo2_savanna_1b_6950steps.png)
