In order to process data for training, we have to run two scripts.

1. First we need to run `data_scripts/data_curation/download_codons_to_csv.ipynb` which will download our raw data and save it to CSV files.
   After running this script you should see the following files. Note: There are some hardcoded file paths in the notebook, specifically a mount to the container of the shared drive under `/data`, please change them to your own paths. We also apply some filtering: `data_scripts/data_curation/taxids_to_remove_bac.json` lists bacterial taxids to exclude during curation.

archaea.csv fungi.csv plant.csv protozoa.csv vertebrate_other.csv
bacteria.csv invertebrate.csv Primates.csv vertebrate_mammalian.csv

2. Next, we need to execute the script `data_scripts/ncbi_memmap_dataset_creator.py` which will take those downloaded CSVs and create memmap files.
   You can run the script with the following command.

```bash
python data_scripts/ncbi_memmap_dataset_creator.py --data-path /data/downloads/process_grouped/ --save-path /data/downloads/postprocessed/ --chunk-size 1000000000
```

The `--chunk-size` parameter controls how many bases (nucleotides) or characters are loaded into memory at once, impacting both processing speed and memory use.

- A larger chunk size leads to faster I/O and fewer files, but requires more RAM during processing.
- A smaller chunk size uses less RAM but will create more output files and may be slower.

**As a rough rule of thumb:**

- For machines with 32GB or less of RAM, try chunk sizes of `500000000` (500 million).
- For machines with 64-128GB RAM, use `1000000000` (1 billion) or up to `2000000000` (2 billion).
- If you have more RAM available, you can increase this further, but keep in mind file system limits and the size of your largest input files.

Monitor your memory usage during the run. If the process runs out of memory (is killed or system becomes unresponsive), reduce the chunk size and try again.

After running this script you should see the following files.

```bash
chunks_metadata.json  data_processed  index_chunk0.mmap  metadata.json  sequences_chunk0.mmap
```

Except, in your case there will be a lot more, but the format is the same.

Optionally, if you want to cluster data for training you can proceed with the next step.
3\. Run the script `data_scripts/allseq_clustering_for_splits.ipynb` which will cluster the sequences and create splits.

Outputs from (1) and (2) are transformed during training using the file `src/data/preprocess` and then consumed by the `CodonMemmapDataset` pipeline. This happens automatically during training.
