# Evo2 Data Preparation

## Data Preprocessing

To streamline the process of preparing and building datasets for training Evo2 on DNA sequences, we provide a configurable preprocessing script (`preprocess.py`) that can preprocess and tokenize a collection of `.fasta` files and convert them into Megatron-compatible `IndexedDataset`.

```bash
preprocess_evo2 -c <CONFIG_PATH>
```

or if you are running the script outside of the BioNeMo container or you haven't pip-installed `bionemo-evo2`, then you can run the script directly:

```bash
python sub-packages/bionemo-evo2/src/bionemo/evo2/data/preprocess.py -c <CONFIG_PATH>
```

Configuration YAML parameters for the script can be found in `utils/config.py`:

```python
class Evo2PreprocessingConfig(BaseModel):
    """Pydantic model class specifying the configuration schema for a preprocessed IndexedDataset (.bin, .idx)."""

    # Collection of FASTA files to preprocess and wrap into a single IndexedDataset.
    datapaths: list[Path] = []
    # Output directory for the preprocessed dataset .bin/.idx.
    output_dir: None | Path = None
    # Output file prefix for identifying your datasets.
    output_prefix: None | str = None
    # Random Sequence-Level Datasplit
    train_split: float = 0.7
    valid_split: float = 0.2
    test_split: float = 0.1
    # Overwrite existing binaries. Otherwise, skip already preprocessed datasets.
    overwrite: bool = False
    # Raw Preprocessing Transforms
    # For every sequence, include a reverse-complemented copy of that sequence in the dataset. Doubles the size of the dataset.
    embed_reverse_complement: bool = False
    # For every sequence, randomly reverse complement the sequence with the specified probability instead of using the original sequence.
    random_reverse_complement: float = 0.0
    # For sequences associated with taxonomic lineages specified in `taxonomy_data`, randomly drop out nodes of the lineage with the specified probability. For instance: |d__KINGDOM;p__None;c__CLASS;o__None;f__None;g__None;s__None|
    random_lineage_dropout: float = 0.0
    # Transcribe (DNA -> RNA) or Back-Transcribe (RNA -> DNA) the sequence before tokenization.
    transcribe: None | Literal["transcribe", "back_transcribe"] = None
    # Force upper-case alphabetical characters in the `.fasta` sequences.
    force_uppercase: bool = False
    # Data type of the IndexedDataset. When using the byte-level tokenizer, uint8 is more than sufficient with a vocabulary size of 255 for ASCII.
    indexed_dataset_dtype: str = "uint8"
    # Tokenization Transforms
    # Append end-of-document token to the end of each sequence.
    append_eod: bool = False
    # Enforce the length of the sequence, by padding shorter sequences and raising exceptions when the length is exceeded.
    enforce_sample_length: None | int = None
    # Run ftfy on the sequence characters prior to tokenization to fix encoding issues.
    ftfy: bool = False
    # Tokenizer
    tokenizer_type: Literal[
        "Byte-Level",
        "HuggingFace",
        "SentencePiece",
        "Regex",
        "Megatron",
        "Tiktoken",
    ] = "Byte-Level"  # Recommended for DNA / RNA sequences. All other tokenizers have not been tested, and only supported here for experimentation!
    # For more information on the behavior of the following parameters, refer to NeMo:
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/modules/common/tokenizer_utils.py
    vocab_file: None | Path = None
    vocab_size: None | int = 512
    merges_file: None | Path = None
    tokenizer_model_name: None | str = None
    pretrained_tokenizer_model: None | str = None
    special_tokens: None | dict[str, str] = {}
    fast_hf_tokenizer: bool = False
    # Compute Configuration
    # NOTE: If preprocessing a large amount of short individual sequences (< 1000 bp), do NOT use
    # multiprocessing (workers > 1) because sequence-level parallel IPC will dominate the preprocessing time!
    workers: int = 1
    # Number of sequences to load into memory at any given time during preprocessing.
    # Prevents OOM while doing sequence-parallel.
    preproc_concurrency: int = 100000
    chunksize: int = 1
    # Data Filters
    drop_empty_sequences: bool = False
    # If `NNN` is detected in the sequence, drop it from the preprocessed dataset.
    nnn_filter: bool = False
    # RNG
    seed: None | int = None
    # Evo2 Taxonomic Lineage Tags
    # SeqID Sub-String Indexing: "ABC" will have taxonomy data from "A".
    taxonomy_data: dict[str, Evo2TaxonomyLineage] = {}
    # Periodicity of injecting phylogenetic lineage tags in the sequence prior to tokenization.
    prompt_spacer_length: int = 131072
```

Furthermore, the `taxonomy_data` field contains a map from sequence ID substrings to phylogenetic lineage data of the form:

```python
class Evo2TaxonomyLineage(BaseModel):
    """Pydantic model class that defines the source lineage of a DNA sequence."""

    kingdom: None | str = None
    phylum: None | str = None
    clazz: None | str = None
    order: None | str = None
    family: None | str = None
    genus: None | str = None
    species: None | str = None
```

which gets converted into a lineage string prior to tokenization as a prefix to the sequence:

```
# (Example) Escherichia coli
|d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__Escherichia coli|ATCGTACGTACATCTCTA...
```

In the Evo2 model, this special "token" is masked out in the loss function, so the model will learn to not generate tokens of this form.

### Testing

To test equivalence with the reference implementation we first downloaded source-of-truth preprocessed Megatron `IndexedDataset` containing promoters data:

```bash
$ ls -lah
-rwxr-xr-x  1 bionemo bionemo 1.2M Dec  4 00:56 data_promoters_test_text_CharLevelTokenizer_document.bin
-rwxr-xr-x  1 bionemo bionemo  20K Dec  4 00:56 data_promoters_test_text_CharLevelTokenizer_document.idx
-rwxr-xr-x  1 bionemo bionemo 392M Dec  4 00:56 data_promoters_train_text_CharLevelTokenizer_document.bin
-rwxr-xr-x  1 bionemo bionemo 6.6M Dec  4 00:56 data_promoters_train_text_CharLevelTokenizer_document.idx
-rwxr-xr-x  1 bionemo bionemo 1.2M Dec  4 00:56 data_promoters_valid_text_CharLevelTokenizer_document.bin
-rwxr-xr-x  1 bionemo bionemo  20K Dec  4 00:56 data_promoters_valid_text_CharLevelTokenizer_document.idx
```

Next we acquired the `.fasta` file that was used to generate this, and configured our scripts to preprocess the sequence data into equivalent Megatron `IndexedDataset`.

```yaml
# mmseqs_promotors_config.yaml
- datapaths: ["/workspace/bionemo2/data/mmseqs_results_rep_seq_distinct.fasta"]
  output_dir: "/workspace/bionemo2/data"
  output_prefix: promoters_uint8_distinct
  train_split: 1.0  # We're just going to dump everything into a single file and compare against the union of the 3 splits in the SoT.
  valid_split: 0.0
  test_split: 0.0
  overwrite: True
  embed_reverse_complement: true
  random_reverse_complement: 0.0
  random_lineage_dropout: 0.0
  include_sequence_id: false
  transcribe: "back_transcribe"
  force_uppercase: true
  indexed_dataset_dtype: "uint8"
  tokenizer_type: "Byte-Level"
  vocab_file: null
  vocab_size: null
  merges_file: null
  pretrained_tokenizer_model: null
  special_tokens: null
  fast_hf_tokenizer: true
  append_eod: true
  enforce_sample_length: null
  ftfy: false
  workers: 1
  preproc_concurrency: 100000
  chunksize: 25
  drop_empty_sequences: true
  nnn_filter: true
  seed: null  # Not relevant because we are not using random reverse complement or lineage dropout.
```

To run the preprocessing script, we ran the following command:

```bash
$ python preprocess.py -c mmseqs_promotors_config.yaml
```

To check equivalence of the two preprocessed datasets, we verify that we get the same elements out of our processed dataset as the original, but do not enforce ordering of the data. (`bionemo-noodles` does not sequentially read the `.fasta` file.)

```python
>>> from megatron.core.datasets.indexed_dataset import IndexedDataset
>>> ds_train_ref = IndexedDataset("./data_promoters_train_text_CharLevelTokenizer_document")
>>> ds_val_ref = IndexedDataset("./data_promoters_valid_text_CharLevelTokenizer_document")
>>> ds_test_ref = IndexedDataset("./data_promoters_test_text_CharLevelTokenizer_document")
>>> ds_train_ours = IndexedDataset("./promoters_uint8_distinct_byte-level_train")
>>> len(ds_train_ours) == len(ds_train_ref) + len(ds_test_ref) + len(ds_val_ref)
True
>>>  # Example of what one of these set elements looks like, it's just a string representation of the token list for an
>>>  #  element of the training dataset. We can then compare all of these to make sure that the two datasets have the
>>>  #  same set of samples.
>>> ','.join([str(t) for t in ds_train_ref[0]])
'67,84,71,71,65,71,67,67,84,71,65,67,67,65,84,65,65,71,84,65,71,84,71,71,67,84,65,84,65,65,67,71,65,71,71,65,65,71,65,65,71,65,84,71,65,65,71,65,71,65,84,84,65,71,65,71,65,65,65,65,84,71,65,65,84,71,84,84,67,84,84,71,65,65,71,84,65,71,67,67,65,84,84,71,84,84,71,84,65,71,84,84,71,84,84,71,84,71,84,71,84,71,84,65,84,71,84,84,71,65,71,65,84,71,84,84,84,84,71,71,71,71,84,84,84,71,84,84,65,84,65,84,65,71,65,71,65,71,65,71,65,84,71,84,65,71,84,84,84,71,71,84,71,65,65,71,65,71,84,65,71,71,65,84,84,67,84,67,84,84,65,67,84,65,71,84,71,84,71,65,65,71,65,84,84,65,84,84,65,67,84,65,71,71,84,65,65,67,84,65,65,65,84,71,65,71,65,84,84,67,84,65,84,67,65,65,67,84,65,65,71,84,67,65,84,84,65,71,65,71,65,84,84,71,71,65,65,65,84,71,84,84,84,67,84,84,84,84,65,71,71,84,84,84,65,65,84,65,65,65,71,84,84,84,71,84,84,84,71,65,65,84,84,71,65,71,65,65,65,71,65,71,65,71,65,71,71,65,71,65,71,65,67,65,84,84,71,67,84,84,84,71,65,65,71,71,71,65,71,65,71,84,84,84,71,71,71,84,71,71,71,84,71,65,71,71,65,84,84,71,65,65,65,65,84,71,65,65,65,65,65,84,71,65,65,67,84,71,65,65,65,65,65,71,71,84,71,84,84,65,84,65,71,84,71,65,67,67,84,71,84,67,65,65,65,65,65,65,71,67,84,71,84,71,65,65,71,65,65,71,84,71,84,84,65,84,67,67,65,65,71,65,65,65,84,65,84,71,71,65,84,84,71,67,84,65,65,84,67,65,84,65,67,84,65,67,84,71,84,84,67,65,84,84,65,84,71,65,84,84,84,84,65,84,71,84,71,84,67,65,84,71,84,71,84,71,84,71,67,67,84,65,84,67,65,84,67,65,84,84,67,67,84,84,65,84,65,84,84,84,84,65,71,84,84,71,71,67,65,65,65,65,65,65,65,65,65,65,65,71,65,67,84,84,71,71,65,65,71,84,65,84,84,71,65,65,65,65,67,67,65,65,65,84,67,84,71,65,84,67,84,67,65,65,67,67,84,65,71,65,67,65,65,71,84,67,71,65,84,84,65,65,65,71,67,84,65,65,65,67,67,71,65,65,65,65,67,67,71,65,65,84,67,67,67,71,65,67,67,71,71,84,84,65,65,84,84,71,65,65,65,65,67,67,71,65,84,67,67,65,0'
>>> # Create a set of all of these elements:
>>> all_ref_data = {','.join([str(t) for t in rec]) for ds in [ds_train_ref, ds_val_ref, ds_test_ref] for rec in ds}
>>> # Verify that there is no redundancy so we can do set equality safely
>>> len(all_ref_data) == len(ds_train_ours)
True
>>> len(all_ref_data)
343504
>>> all_our_data = {','.join([str(t) for t in rec]) for ds in [ds_train_ours] for rec in ds}
>>> len(all_our_data)
343504
>>> # Verify set equality to show that we have processed an identical dataset
>>> #  (ignoring shuffling order and train/test/val splits)
>>> all_our_data == all_ref_data
True
```

## Sequence Splicing & Stitching

Evo2 has also been trained on spliced DNA and mRNA sequences, where introns are removed leaving only the concatenated exons of the genome. Moreover, "stitched" variants of spliced transcripts have been introduced into Evo2's training dataset, which include 1024 bp of sequence from the promoter and 32 bp around each exon.

To perform splicing or "stitched" splicing on sequences in a FASTA file given an associated gene transfer format (GTF) file, execute the following command:

```bash
$ splice_evo2 --help
usage: splice_evo2 [-h] --fasta-path FASTA_PATH --gtf-path GTF_PATH [--output-path OUTPUT_PATH] [--transcript-type {default,stitched}] [--stitched-promoter STITCHED_PROMOTER] [--stitched-intron STITCHED_INTRON] [--stitched-overlap] [--only-longest-transcript] [-v]

Extract spliced transcripts from a FASTA and GTF.

options:
  -h, --help            show this help message and exit
  --fasta-path FASTA_PATH
                        Path to FASTA file to extract transcripts from.
  --gtf-path GTF_PATH   Path to gene transfer format (GTF) file associated with the FASTA.
  --output-path OUTPUT_PATH
                        Path to output FASTA file.
  --transcript-type {default,stitched}
                        Type of transcript to extract from the GTF and FASTA files for splicing. 'Stitched' transcripts include 1024 bp of sequence from the promoter and 32 bp around each exon.
  --stitched-promoter STITCHED_PROMOTER
                        Number of bp to include in the promoter region when --transcript-type=stitched is used. Defaults to 1024.
  --stitched-intron STITCHED_INTRON
                        Number of bp to include from neighboring introns when --transcript-type=stitched is used. Defaults to 32.
  --stitched-overlap    Allow overlap of neighboring intron windows when --transcript-type=stitched is used. Defaults to False, i.e. prevents overlap by shortening the intron windows for a contiguous splice.
  --only-longest-transcript
                        Only extract the longest transcript per gene.
  -v, --verbose         Turn on verbose log messages.
```
