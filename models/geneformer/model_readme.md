---
library_name: transformers
license: apache-2.0
widget:
  - text: [<cls>, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, <eos>]
---

## Geneformer (TransformerEngine-optimized)

This version of the Geneformer model is optimized with NVIDIA's
[TransformerEngine](https://github.com/NVIDIA/TransformerEngine) library. It is based on the
[original Geneformer model](https://huggingface.co/ctheodoris/Geneformer) from Theodoris et al.,
and (within numerical precision) has identical weights and outputs.

Geneformer is a foundational transformer model pretrained on a large-scale corpus of single cell transcriptomes
representing a broad range of human tissues. It is suitable for fine-tuning on a wide range of tasks that take
gene expression data as input. For detailed information on the model architecture and training data, please refer
to the [accompanying paper](https://rdcu.be/ddrx0). You may also be interested in the
[documentation](https://geneformer.readthedocs.io) and [examples](https://huggingface.co/ctheodoris/Geneformer/tree/main/examples)
which demonstrate how to fine-tune Geneformer models on your tasks of interest.

Several Geneformer checkpoints are available in the Hub with varying sizes. Larger sizes generally have
somewhat better accuracy, but require much more memory and time to train:

| Checkpoint name                                                                                                   | Parameters | Input size | Vocabulary | Training data            |
| ----------------------------------------------------------------------------------------------------------------- | ---------- | ---------- | ---------- | ------------------------ |
| [Geneformer-V1-10M](https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V1-10M)                     | 10M        | 2048       | ~25K genes | ~30M cells               |
| [Geneformer-V2-104M](https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V2-104M)                   | 104M       | 4096       | ~20K genes | ~104M cells              |
| [Geneformer-V2-316M](https://huggingface.co/ctheodoris/Geneformer)                                                | 316M       | 4096       | ~20K genes | ~104M cells              |
| [Geneformer-V2-104M_CLcancer](https://huggingface.co/ctheodoris/Geneformer/tree/main/Geneformer-V2-104M_CLcancer) | 104M       | 4096       | ~20K genes | ~104M + 14M cancer cells |
