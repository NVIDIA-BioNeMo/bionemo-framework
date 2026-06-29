# Eden Data

## ShardedEdenDataset

Eden uses the `ShardedEdenDataset` from Basecamp Research for training data,
backed by SQLite databases for fast windowed access to genomic sequences.
The core dataset implementation lives in the `bionemo.common` sub-package
(`bionemo.common.data.basecamp`), while the Megatron-specific
`ShardedEdenDatasetProvider` wrapper lives here.

See `sharded_eden_dataloader.md` for full documentation on the dataset
schema, directory structure, and pre-processing workflow.

## FASTA Dataset

`SimpleFastaDataset` provides a simple PyTorch dataset for loading FASTA
files for prediction/inference use cases.

## FASTA to JSONL Conversion

Convert FASTA files to JSONL format for inference:

```bash
bionemo_fasta_to_jsonl --input /path/to/input.fasta --output /path/to/output.jsonl
```

This CLI tool is provided by the `bionemo.common` package.
