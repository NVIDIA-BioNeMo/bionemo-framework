# OpenGenome2 Metagenome Dataset: Sharding, Shuffling, and Tokenization in the LlaMA 3 Recipe

This document describes how data is loaded, tokenized, and shuffled for the OpenGenome2 Llama
recipe, including our rationale and strategy for resharding and reshuffling the original OG2
metagenome dataset.

> **Recommended approach:** Download the metagenome JSONL files, reshard them into globally
> shuffled Parquet with DuckDB (see [Reshuffling and resharding](#reshuffling-and-resharding-with-duckdb)),
> and use the `og2_7b_thd_gqa_global_shuffle` config. This gives better shuffle quality and
> smoother and faster step times than streaming the original JSONL files directly.

## Downloading the dataset

The OpenGenome2 dataset is hosted on HuggingFace at
[arcinstitute/opengenome2](https://huggingface.co/datasets/arcinstitute/opengenome2).
To download the metagenome training and validation files:

```bash
pip install huggingface_hub[cli]

# Download metagenome training files (~80 JSONL shards)
huggingface-cli download arcinstitute/opengenome2 \
  --repo-type dataset \
  --include "pretraining_or_both_phases/metagenomes/data_metagenomics_train_*.jsonl.gz" \
  --local-dir /data/opengenome2

# Download metagenome validation file
huggingface-cli download arcinstitute/opengenome2 \
  --repo-type dataset \
  --include "pretraining_or_both_phases/metagenomes/data_metagenomics_valid_*.jsonl.gz" \
  --local-dir /data/opengenome2

# Download metagenome test file
huggingface-cli download arcinstitute/opengenome2 \
  --repo-type dataset \
  --include "pretraining_or_both_phases/metagenomes/data_metagenomics_test_*.jsonl.gz" \
  --local-dir /data/opengenome2
```

After downloading, we recommend resharding the data with DuckDB for better training performance
(see [below](#reshuffling-and-resharding-with-duckdb)) and dataset shuffling quality.

## The OpenGenome2 metagenome dataset

The training data is the metagenome subset of the OpenGenome2 dataset, originally stored as 80
compressed JSONL shards. Within each file, sequences are sorted from longest to shortest — as seen
in the figure below. The data was likely split by length or sequence similarity before being
partitioned into files.

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_data_sorting.png" alt="OpenGenome2 sequence length distribution across shards" width="80%" />
</p>

The median sequence length is ~2.2k bases and the mean is ~4k, but some sequences exceed 1M bases.
Because most sequences are much shorter than the 8192-token context window, a standard BSHD
dataloader wastes significant compute on padding. THD sequence packing avoids this by concatenating
multiple sequences into a single 8192-token batch entry, processing ~2–3× more useful tokens per
step on this dataset.

## How the reference baseline handles data: Megatron ShardedEden Dataloader

For context, the Megatron/NeMo baseline uses the **ShardedEden dataloader**, which achieves a true
global shuffle:

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_megatron_sharded_eden_dataloader.png" alt="Megatron ShardedEden dataloader" width="80%" />
</p>

1. **Precomputed window index** — All 80 shards are indexed offline to produce a mapping of
   `window_idx -> (sequence_id, position_in_sequence)`. With `seq_length=8192` and `stride=7992`,
   this produces ~232M indexed windows.
2. **Global shuffle** — A PyTorch `DistributedSampler` assigns ~4.8M random indices to each rank
   per epoch. Every rank can see windows from any sequence in any shard.
3. **THD packing** — Each micro-batch packs ~2 windows into 8192 tokens.

This recipe replaces that pipeline with the HuggingFace streaming API.

## How this recipe handles data: HuggingFace streaming buffer

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_hf_streaming_buffer_original.png" alt="HF streaming buffer with original dataset" width="80%" />
</p>

The pipeline has five stages:

1. **Shard assignment** — Each rank is assigned a disjoint subset of files. With 80 JSONL shards
   and 48 ranks, each rank sees only 1–2 files.
2. **Streaming tokenization** — Workers read sequences sequentially from their assigned shards,
   tokenize on the fly, and split into 8k windows (with `stride=200` for overlap). Note: `<BOS>`
   and `<EOS>` are added to every window (not just at true sequence boundaries). Both are masked
   from the loss, so the impact on training is minimal. See the README for details.
3. **Buffer shuffle** — Tokenized windows are shuffled within a reservoir buffer of `buffer_size`
   (default: 50,000). Each new window replaces a randomly chosen element in the buffer. Ordering is
   randomized only within this sliding window, not globally.
4. **THD packing** — The collator packs ~2 windows per micro-batch into 8192 tokens.
5. **Gradient accumulation** — n micro-steps are accumulated before each all-reduce and optimizer
   step (GA=n).

With only 1–2 shards per rank and the strong internal length-sorting of the original JSONL files,
the buffer shuffle only randomizes within a narrow slice of similar-length sequences on each rank. This limits
batch diversity and we hypothesized that it may slow convergence compared to the Megatron baseline's true global shuffle.

### Tuning parameters

| Parameter         | Effect                                       | Tradeoff                                                                                   |
| ----------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `buffer_size`     | Controls local shuffle quality               | Larger = better randomization, more CPU memory                                             |
| `num_workers`     | Controls data loading throughput             | More workers = better I/O overlap and shuffling, but more memory (each has its own buffer) |
| `prefetch_factor` | Batches queued ahead per worker (default: 4) | Higher = absorbs shard-transition stalls, more memory                                      |

**Step-time spikes:** When a worker finishes its shard and opens a new one, the GPU may stall
waiting for the buffer to refill. This causes occasional step-time spikes visible in WandB.
Increasing `prefetch_factor` or `buffer_size` can help absorb these stalls.

## Reshuffling and resharding with [DuckDB](https://duckdb.org/)

To address the limited batch diversity, we globally shuffle all sequences and reshard into many
more Parquet files using [DuckDB](https://duckdb.org/):

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_resharding_duckdb.png" alt="Resharding pipeline with DuckDB" width="60%" />
</p>

This produces 1,734 Parquet shards with sequences globally shuffled (no
length ordering within any file) and uniformly distributed across files:

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_original_vs_reshuffled.png" alt="Original vs reshuffled data distribution" width="80%" />
</p>

With 48 ranks now streaming from ~36 shards each (instead of 1–2), and 8 workers per rank each
reading from different shards, the effective shuffle pool becomes
`buffer_size × num_workers` across a much more diverse set of sequences. We also expect better performance because we can use more workers compared to the non sharded dataset.

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_hf_streaming_buffer_resharded.png" alt="HF streaming buffer with resharded dataset" width="80%" />
</p>

### Creating your own resharded dataset

1. Install DuckDB: `pip install duckdb` (or download from [duckdb.org](https://duckdb.org/))
2. Run the following from the directory containing your JSONL training file. Note that FILE_SIZE_BYTES dtermines the total number of shards you write lut (choose a smaller value for more shards) :

```bash
duckdb -c "
SET memory_limit = '100GB';
SET temp_directory = '/tmp/duckdb_tmp';
SET threads = 48;
SET preserve_insertion_order = false;
COPY (
  SELECT text
  FROM read_json('*train*.jsonl', format='newline_delimited')
  ORDER BY random()
)
TO 'output' (FORMAT PARQUET, PER_THREAD_OUTPUT true, FILE_SIZE_BYTES '200MB');
"
```

3. The output directory will contain Parquet shards (e.g. `output/data_0.parquet`, ...)
4. Update your Hydra config or override on the command line:

```yaml
dataset:
  load_dataset_kwargs:
    data_files: null
    path: "/path/to/your/resharded_parquet_dir"
    split: "train"
    streaming: true
```

Or via command line:

```bash
torchrun --nproc_per_node=8 train_fsdp2.py --config-name og2_7b_thd_gqa_global_shuffle \
  dataset.load_dataset_kwargs.path=/path/to/your/resharded_parquet_dir
```

## Summary of approaches

Overall, we recommend using the resharded dataset for best performance and batch diversity. If using the original dataset, we recommend using at least a 50k buffer (with 1 worker).

<p align="center">
  <img src="../../../docs/docs/assets/images/recipes/og2_summary_shuffling_approaches.png" alt="Summary of shuffling approaches" width="80%" />
</p>

## Config mapping

| Config                          | Data source                | Tokenization          | stride | buffer_size | Notes                   |
| ------------------------------- | -------------------------- | --------------------- | ------ | ----------- | ----------------------- |
| `og2_7b_thd_gqa`                | Streaming JSONL (original) | Windowed (on-the-fly) | 200    | 50,000      | Original 80 shards      |
| `og2_7b_thd_gqa_global_shuffle` | Streaming Sharded Parquet  | Windowed (on-the-fly) | 200    | 10,000      | Reshuffled 1,733 shards |

Implementation: [dataset.py](dataset.py) (`create_tokenized_dataset`, `create_thd_dataloader`,
`create_bshd_dataloader`).
