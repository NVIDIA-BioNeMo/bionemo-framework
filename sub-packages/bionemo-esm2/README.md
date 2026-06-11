# bionemo-esm2

ESM-2 is a protein language model with BERT architecture trained on millions of protein sequences from UniProt. ESM-2 learns the patterns and dependencies between amino acids that ultimately give rise to a protein’s structure. ESM-2 is pretrained on a masked language model (MLM) objective. During pretraining, 15% of the input sequence is perturbed, and within which 80% of the residues are replaced with a mask token, 10% are replaced with a random token, and 10% are left unchanged. The model is then trained to predict the original amino acids at the perturbed positions with the context of the surrounding amino acids.

Despite pretraining on an MLM objective, the sequence representation learned by ESM-2 is highly transferable to downstream tasks. ESM-2 can be fine-tuned on a variety of tasks, including secondary structure prediction as, and whole-sequence prediction on cellular localization, thermostability, solubility, and other protein properties.

### Setup

To install, execute the following:

```bash
pip install -e .
```

To run unit tests, execute:

```bash
pytest -v .
```
