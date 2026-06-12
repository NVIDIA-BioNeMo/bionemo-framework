# Context Parallelism

## What is it

When training transformer-based models, context is everything. It's what tells the model: look at this token in relation to all these other tokens. The model's ability to establish context is paramount to many LLM tasks because it "grounds" the model and tells it to look at this thing in relation to its context.

But what happens when I can't make that context window any bigger? Enter Context Parallelism (CP). CP is used to parallelize context across multiple GPUs, such that sequences can be sharded and split up so that no single GPU has to hold the entire context in memory, they can share the load.

In short, Context Parallelism distributes sequences across devices. It's one of the "Ds" in what's known as 5D parallelism (Tensor Parallel, Pipeline Parallel, Data Parallel, Expert Parallel, Context Parallel).

CP acts very similarly to Data Parallelism, in that the activations for input tokens are distributed across devices. The key difference is that in CP, the activations across multiple devices are part of the same input sequence, whereas in Data Parallelism these split activations are for different sequences.

## How does it work?

The core idea behind CP is to partition the data into various chunks, with each chunk assigned to a different device (GPU in this case). During each forward pass, each device computes attention locally on a chunk while coordinating with other devices to access key-value pairs needed for the full attention computation.

## What does the data generation part look like?

In BioNeMo, we've created some abstractions to partition the data for you. There exists a [ContextParallelDataLoaderWrapper](esm2_native_te/collator.py) that will shard the CP data for you and send it to each device. This dataloader operates on Sequence Packed (THD) data [link](https://docs.nvidia.com/nemo-framework/user-guide/24.12/nemotoolkit/features/optimizations/sequence_packing.html). This `ContextParallelDataLoaderWrapper` will take as arguments your CP group and local CP rank. This dataloader wrapper will call its underlying dataloader to generate a unique piece of data and then shard those unique sequences across your CP groups. This is beneficial because you won't need to maintain a deterministic data pipeline because unique data is only being generated across the non CP groups, and it is replicated across the CP groups. More details below.

Alternatively, one could utilize any DataLoader such as the canonical [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader), however, you would have to ensure that your dataset is synchronized across CP ranks. In some cases, if you have a non-deterministic data pipeline, even if you attempt to get the same data from a dataloader it may be different due to non-deterministic preprocessing stages such as masking. For more information on preserving determinism in your datasets, please see [MegatronLMDataModule](../docs/docs/main/about/background/megatron_datasets.md).

### Context Parallelism Sharding Example

**Original packed sequences (2 seqs):**

```
┌─────────────────────────┐
│  1, 2, 3 | 5, 6         │
└─────────────────────────┘
```

**Pad to divisibility:**

```
┌────────────────────────────────────┐
│  1, 2, 3, <pad> | 5, 6, <pad>, <pad> │
└────────────────────────────────────┘
```

**Distributed across CP ranks:**

```
CP0: [1, <pad> | 5, <pad>]
CP1: [2, 3     | 6, <pad>]
```

In the example above, imagine that we have 2 CP groups (CP0 and CP1). The `ContextParallelDataLoaderWrapper` takes as an argument an `UnderlyingDataloader` which generates the unique sequences `1, 2, 3` and `5, 6`. In CP we need to pad these sequences so that they are divisible by `cp_size*2` to enable chunking for [Ring Attention](https://arxiv.org/abs/2310.01889). In this case with `cp_size=2` we need to make each sequence divisible by 4.

After we've padded the sequences, we distribute the shards across the CP ranks (CP0 and CP1). We can see that each CP rank takes a slice from the first and second sequence. For CP0 it takes the first and last token of both sequences while CP1 takes the middle tokens.

After ring attention, the activations will also be sharded across those CP groups so no device has to hold all of them!

### Resources

For more information related to Context Parallelism, please see our recipes:

- [esm2/train_ddp_cp.py](esm2_native_te/train_ddp_cp.py)
