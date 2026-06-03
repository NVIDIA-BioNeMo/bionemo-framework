# bionemo-size-aware-batching

To install, execute the following:

```bash
pip install -e .
```

To run unit tests, execute:

```bash
pytest -v .
```

# **Summary of Usage**

This package provides a simple way to create mini-batches in a memory consumption-aware (or size-aware) manner, making
it useful for tasks like training models on datasets with varying memory requirements. The usage typically consists of
the following steps:

1. Use the `collect_cuda_peak_alloc` function to collect CUDA peak memory
   allocation statistics for a user-defined workflow. It's expected that the
   user-defined workflow will return a list of features extracted from the data
   so that the memory model in the following step can use these features to
   predict the memory allocation.
2. User defines and trains a memory model using the features and memory allocation
   data from previous step. This memory model can then be used to predict memory
   consumption.
3. Use `SizeAwareBatchSampler` or `size_aware_batching` with the memory model
   prediction (from the previous step) to build batch of data so that the
   resulting mini-batches do not exceed a specified maximum total memory size.

In addition, this package provides one solution to create homogeneous mini-batches, which can be useful to reduce the
padding when aligning the shape of inputs when training or evaluating the models. This `BucketBatchSampler` can be used
in conjunction with `torch.utils.data.BatchSampler`, `SizeAwareBatchSampler` or other user-defined batch samplers.
This usage can leverage the `create_buckets` function and follow the steps below:

1. Gather the tensor sizes of elements in the dataset, which are the shapes of
   tensors in a specific dimension where you want to reduce the padding.
2. Provide your own bucket boundaries based on the tensor sizes, or create bucket
   boundaries with `create_buckets` function with the tensor sizes and bucket
   maximal width and the minimal bucket count. The `create_buckets` function
   will try to create buckets with smallest widths and element counts >= minimal
   bucket count, unless the maximal width or the boundary is reached.
3. Use `BucketBatchSampler` with base batch sampler like `torch.utils.data.BatchSampler` or
   `SizeAwareBatchSampler` for each bucket. The `BucketBatchSampler` will select a bucket each time
   and generate a mini-batch from this bucket using the base batch sampler for this bucket.
   As such, the padding necessary for the generated mini-batches will be always smaller
   than the width of buckets.

Refer to the later sections for the API documentation and examples on how to achieve each of the steps above.

### utils Module

______________________________________________________________________

- [**collect_cuda_peak_alloc**](#utils.collect_cuda_peak_alloc): A function that
  collects CUDA peak memory allocation statistics and features to be used for
  memory usage prediction for a given workflow.

- [**create_buckets**](#create_buckets): A function to create buckets for a
  list of integers with pre-defined maximal width of ranges and minimal
  bucket sizes.

### sampler Module

______________________________________________________________________

- [**size_aware_batching**](#sampler.size_aware_batching): A generator that batches elements from an iterable while
  ensuring that the total size of each batch does not exceed a specified maximum.
- [**SizeAwareBatchSampler**](#sampler.SizeAwareBatchSampler): A class that batches elements of varying sizes while
  ensuring that the total size of each batch does not exceed a specified maximum.
- [**BucketBatchSampler**](#sampler.BucketBatchSampler): A class that groups elements of varying sizes based on predefined
  bucket ranges, and create batches with elements from each bucket to ensure that each batch has elements with
  homogeneous sizes.

# API reference and examples

<a id="utils"></a>

## utils

<a id="utils.collect_cuda_peak_alloc"></a>

#### collect_cuda_peak_alloc

```python
def collect_cuda_peak_alloc(
    dataset: Iterable[Data],
    work: Callable[[Data], Feature],
    device: torch.device,
    cleanup: Optional[Callable[[], None]] = None
) -> Tuple[List[Feature], List[int]]
```

Collects CUDA peak memory allocation statistics for a given workflow.

This function iterates through the provided dataset, applies the given feature function to each data point,
and records the peak CUDA memory allocation during this process. The features extracted from the data points
are collected along with their corresponding memory usage statistics.

Note that the first few iterations of the workflow might result in smaller memory allocations due to uninitialized
data (e.g., internal PyTorch buffers). Therefore, users may want to skip these initial data points when analyzing the
results.

**Arguments**:

- `dataset` - An iterable containing the input data.
- `work` - A function that takes a data point and returns its corresponding feature. This is where
  the main computation happens and memory allocations are tracked.
- `device` - The target Torch CUDA device.
- `cleanup` - A function that is called after each iteration to perform any necessary cleanup.

**Returns**:

A tuple containing the collected features and their corresponding memory usage statistics.

**Raises**:

- `ValueError` - If the provided device is not a CUDA device.

  ______________________________________________________________________

**Examples**:

```python
>>> import torch
>>> from bionemo.size_aware_batching.utils import collect_cuda_peak_alloc


>>> # prepare dataset, model and other components of a workflow
>>> # for which the user want to collect CUDA peak memory allocation statistics
>>> dataset, model, optimizer = ...
>>> # Set the target Torch CUDA device.
>>> device = torch.device("cuda:0")
>>> model = model.to(device)

>>> # Define a function that takes an element of the dataset as input and
>>> # do a training step
>>> def work(data):
...     # example body of a training loop
...     optimizer.zero_grad()
...     output = model(data.to(device))
...     loss = compute_loss(output)
...     loss.backward()
...     optimizer.step()
...     # extract the feature for later to be modeled or analyzed
...     return featurize(data)

>>> # can optionally use a cleanup function to release the references
>>> # hold during the work(). This cleanup function will be called
>>> # at the end of each step before garbage collection and memory allocations measurement
>>> def cleanup():
...     model.zero_grad(set_to_none=True)

>>> # Collect features (i.e., model outputs) and memory usage statistics for the workflow.
>>> features, alloc_peaks = collect_cuda_peak_alloc(
...     dataset=batches,
...     work=work,
...     device=device,
...     cleanup=cleanup,
... )


>>> # use features and alloc_peaks as needed, e.g., fit a model
>>> # that can use these statistics to predict memory usage
>>> memory_model = ...
>>> memory_model.fit(features, alloc_peaks)
```

<a id="utils.create_buckets"></a>

#### create_buckets

```python
def create_buckets(sizes: torch.Tensor, max_width: int,
                   min_bucket_count: int) -> Buckets
```

Create buckets for a list of integers with pre-defined maximal width of interval and minimal bucket count.

It will return a named tuple containing the bucket boundaries and the actual bucket sizes.
e.g. torch.tensor([0, 5, 7]), torch.tensor([3,2]): specifies 2 buckets: one with range 0\<= sizes < 5, width=5 and 3 elements
and the other one with range 5 \<= sizes < 7, width=2 and 2 elements.

**Arguments**:

- `sizes` - An 1D tensor of integers.
- `max_width` - The maximum width of a bucket, should be a positive integer.
- `min_bucket_count` - The minimum count of a bucket, should be a positive integer.
  Bucket size may be smaller than min_bucket_count if its width reaches max_width.

**Raises**:

- `ValueError` - If the provided sizes is empty, or not integers.
- `ValueError` - If max_width is not a positive integer or min_bucket_count is not a positive integer.

**Returns**:

A named tuple containing bucket boundaries in ascending order and the number of elements in each bucket.

______________________________________________________________________

**Examples**:

```python
>>> import torch
>>> from bionemo.size_aware_batching.utils import create_buckets

>>> sizes = torch.tensor([1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 22, 22, 22, 22])
>>> buckets = create_buckets(sizes, max_width=5, min_bucket_count=10)
>>> # 5 buckets: 1 <= sizes < 6, 6 <= sizes < 11, 11 <= sizes < 16, 16 <= sizes < 21, 21 <= sizes < 23
>>> print(buckets.bucket_boundaries)
tensor([ 1,  6, 11, 16, 21, 23])

>>> # each with 12, 0, 0, 0, 4 elements respectively.
>>> print(buckets.bucket_sizes)
tensor([12,  0,  0,  0,  4])

>>> sizes = torch.arange(20)
>>> # min_bucket_count is used to control bucket size
>>> buckets = create_buckets(sizes, max_width=10, min_bucket_count=5)
>>> print(buckets.bucket_boundaries)
tensor([ 0,  5, 10, 15, 20])

>>> print(buckets.bucket_sizes)
tensor([5, 5, 5, 5])
```

<a id="sampler"></a>

## sampler

<a id="sampler.size_aware_batching"></a>

#### size_aware_batching

```python
def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: Real,
    collate_fn: Optional[Callable[[Iterable[Data]], BatchCollated]] = None,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None
) -> Iterator[Union[List[Data], BatchCollated]]
```

A generator that batches elements from an iterable while ensuring that the
total size of each batch does not exceed a specified maximum. Here the size
can be a measurement of memory consumption of the elements in the batch.
This can be useful for both indexible data or non-indexible but iterable data.

**Arguments**:

- `dataset` - The input iterable.
- `sizeof` - A function or mapping that returns the "size" of each element in `dataset`.
  E.g., this can be used to determine how much memory an element consumes. Its return
  type must be comparable with `max_total_size` and it must be addable (operator `+`).
- `max_total_size` - The maximum total "size" of each batch. The semantics of "size"
  is defined by the `sizeof` argument. The type of this value must be comparable
  with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
- `collate_fn` - An optional function to collate batches. Defaults to None, in which case
  each batch is a list of elements from the input dataset
- `info_logger` - A function to log info. Defaults to None.
- `warn_logger` - A function to log warnings. Defaults to None.

**Yields**:

A generator that yields batches from `dataset`.

______________________________________________________________________

Assumptions

1. Linear complexity. This function consumes the given Iterable of data (`dataset`) once,
   by going over the data item one by one to build a batch and yield it as soon as the
   addition of the next data item to the batch would exceed `max_total_size` or if the
   batch is the last one (end of iteration)
2. Additive size measurement. For the general usage case of building mini-batches with
   a threshold of the batch's memory consumption, it assumes that the size of the batch is
   the sum of all elements in the batch (additive property).
3. Comparable type of `max_total_size` and `sizeof`'s return. `sizeof`'s return values
   must be compared with `max_total_size` to threshold the size of batches

______________________________________________________________________

Caveat

- `1` - The generated batch sizes may have large variance

  - how to workaround: filter the output of this generator using a batch size threshold

- `2` - The number of batches may vary a lot across different epochs.

  - how to workaround: increase the number of steps that compose an epoch,
    e.g., in the Lightning training/validation loop, which effectively increases the input
    dataset size per epoch

  ______________________________________________________________________

  Example

```python
>>> import torch
>>> from torch.utils.data import default_collate
>>> from bionemo.size_aware_batching.sampler import size_aware_batching

>>> # Define a sample dataset with torch.tensor
>>> dataset = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]),
...            torch.tensor([7, 8]), torch.tensor([9, 10])]

>>> # Define a sizeof function that returns the size of each tensor
>>> def sizeof(x):
...     return x.numel()

>>> # Create a generator with max_total_size=4 and default_collate_fn
>>> gen = size_aware_batching(dataset, sizeof, 4, collate_fn=default_collate)
>>> batches = list(gen)
>>> print(batches)
    [tensor([[1, 2], [3, 4]]), tensor([[5, 6], [7, 8]]), tensor([[9, 10]])]
```

<a id="sampler.SizeAwareBatchSampler"></a>

## SizeAwareBatchSampler Objects

```python
class SizeAwareBatchSampler(Sampler[List[int]])
```

A sampler that batches elements of varying sizes while ensuring
that the total size of each batch does not exceed a specified maximum.

This is useful when dealing with datasets where each element has a
different size, such as graphs or sequences of varying lengths.
The sampler uses a provided `sizeof` function to determine the size
of each element in the dataset and ensures that the total size of
each batch does not exceed the specified `max_total_size`.

______________________________________________________________________

**Examples**:

```python
>>> import torch
>>> from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler


>>> # Define a sample dataset with torch.tensor
>>> dataset = [torch.tensor([1, 2]), torch.tensor([3, 4]), torch.tensor([5, 6]),
...            torch.tensor([7, 8]), torch.tensor([9, 10])]


>>> # Define a function that returns the size of each element in the dataset.
>>> def sizeof(index):
...     return dataset[index].numel()


>>> # Create a SizeAwareBatchSampler with a maximum total batch size of 10.
>>> batch_sampler = SizeAwareBatchSampler(
...     sampler=torch.utils.data.SequentialSampler(dataset),
...     sizeof=sizeof,
...     max_total_size=4
... )


>>> # Iterate over batches of indices that do not exceed the maximum total size.
>>> print(list(batch_sampler))
    [[0, 1], [2, 3], [4]]
```

<a id="sampler.SizeAwareBatchSampler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(sampler: Union[Sampler[List[int]], Iterable[int]],
             sizeof: Callable[[int], Real],
             max_total_size: Real,
             info_logger: Optional[Callable[[str], None]] = None,
             warn_logger: Optional[Callable[[str], None]] = None) -> None
```

Initializes the SizeAwareBatchSampler.

**Arguments**:

- `sampler` - The underlying sampler.
- `sizeof` - A function that returns the size at each index. E.g., this can used to
  determine how much memory an element consumes. Its return type must be
  comparable with `max_total_size` and it must be addable (operator `+`).
- `max_total_size` - The maximum total size of a mini-batch. The semantics of "size"
  is defined by the `sizeof` argument. The type of this value must be comparable
  with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
- `info_logger` - A function to log info. Defaults to None.
- `warn_logger` - A function to log warnings. Defaults None.

**Raises**:

- `TypeError` - If sampler is not an instance of Sampler or Iterable, or if sizeof is not a callable, dictionary, or sequence container.
- `ValueError` - If max_total_size is not a positive number.

<a id="sampler.SizeAwareBatchSampler.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Iterator[List[int]]
```

Iterate over batches of indices.

This function yields batches of indices that do not exceed the maximum total size.

**Yields**:

A batch of indices that do not exceed the maximum total size.

<a id="sampler.BucketBatchSampler"></a>

## BucketBatchSampler Objects

```python
class BucketBatchSampler(Sampler[List[int]])
```

A batch sampler to create batches with sizes of elements from each pre-defined bucket ranges.

Elements of the dataset are first grouped into each bucket based on the bucket ranges and the sizes of elements.
Then, a base batch sampler is used for each bucket to create mini-batches.

The bucket ranges are specified by `bucket_boundaries`, which will be first sorted internally and used to create
`len(bucket_boundaries) - 1` left-closed right-open intervals.
e.g. if bucket_boundaries tensor is [10, 5, 0, 16], it will be sorted as [0, 5, 10, 16] and 3 buckets will be created
with ranges: \[0, 5), \[5, 10), \[10, 16).

The base batch sampler will be created by passing the element indices in each bucket as the data source, and
`base_batch_sampler_shared_kwargs` and `base_batch_sampler_individual_kwargs`
to the constructor of the base batch sampler class specified as `base_batch_sampler_class`.
e.g. `base_batch_sampler_shared_kwargs = {'drop_last': True}` and `base_batch_sampler_individual_kwargs = {'batch_size': [8,10,12]}`
will be used to create 3 batch samplers with drop_last=True and batch_size=8, 10 and 12, and initialized like
`base_batch_sampler_class(bucket_element_indices[0], batch_size=8, drop_last=True)`.

In the `__iter__` method, if `shuffle` is `True`, the element indices in each bucket will be shuffled, and a bucket
is randomly selected each time to create a mini-batch. If `shuffle` is `False`, there is no shuffle on element indices,
and the bucket is selected in ascending order of its interval boundaries.

This class is used to create homogeneous batches of data for training or evaluation, and reduce the padding necessary to align the shape of elements.

Modified from https://github.com/rssrwn/semla-flow/blob/main/semlaflow/data/util.py

______________________________________________________________________

**Examples**:

```python
>>> import torch
>>> from bionemo.size_aware_batching.sampler import BucketBatchSampler

>>> # Define the sizes for a dataset
>>> sizes = torch.arange(25)
>>> # Define bucket ranges
>>> bucket_boundaries = torch.tensor([0, 6, 15, 25])

>>> # Create a bucket batch sampler with torch.utils.data.BatchSampler as base batch sampler
>>> # As there are 3 buckets, there will be 3 base batch samplers with batch sizes 2, 3, and 5.
>>> batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_boundaries=bucket_boundaries,
        base_batch_sampler_class=torch.utils.data.BatchSampler,
        base_batch_sampler_shared_kwargs={'drop_last': False},
        base_batch_sampler_individual_kwargs={'batch_size': [2,3,5]},
        shuffle=False,
    )

>>> # Iterate over batches of indices that lies in the same bucket and with different batch sizes.
>>> print(list(batch_sampler))
[[0, 1], [2, 3], [4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24]]

>>> # randomize the dataset and buckets
>>> batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_boundaries=bucket_boundaries,
        base_batch_sampler_class=torch.utils.data.BatchSampler,
        base_batch_sampler_shared_kwargs={'drop_last': False},
        base_batch_sampler_individual_kwargs={'batch_size': [2,3,5]},
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
>>> print(list(batch_sampler))
[[24, 17, 16, 22, 19], [2, 5], [12, 10, 11], [3, 0], [15, 18, 20, 21, 23], [7, 13, 6], [14, 9, 8], [1, 4]]
>>> print(list(batch_sampler))
[[14, 9, 13], [23, 16, 20, 21, 15], [5, 0], [8, 10, 11], [17, 24, 22, 18, 19], [12, 6, 7], [4, 2], [3, 1]]

>>> # Combine with SizeAwareBatchSampler to control the cost of each batch
>>> from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler
>>> item_costs = sizes.tolist()
>>> def cost_of_element(index):
        return item_costs[index]
>>> batch_sampler = BucketBatchSampler(
        sizes=sizes,
        bucket_boundaries=bucket_boundaries,
        base_batch_sampler_class=SizeAwareBatchSampler,
        base_batch_sampler_shared_kwargs={"sizeof": cost_of_element, "max_total_size": 40},
        base_batch_sampler_individual_kwargs={},
        shuffle=True,
        generator=torch.Generator().manual_seed(0),
    )
>>> print(list(iter(batch_sampler)))
[[24], [2, 5, 3, 0, 1, 4], [12, 10, 11, 7], [13, 6, 14], [17, 16], [22], [19, 15], [9, 8], [18, 20], [21], [23]]
```

<a id="sampler.BucketBatchSampler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(sizes: torch.Tensor,
             bucket_boundaries: torch.Tensor,
             base_batch_sampler_class: Type[Sampler],
             base_batch_sampler_shared_kwargs: Optional[Dict[str, Any]] = None,
             base_batch_sampler_individual_kwargs: Optional[Dict[
                 str, Iterable]] = None,
             shuffle: Optional[bool] = True,
             generator: Optional[torch.Generator] = None) -> None
```

Initializes the BucketBatchSampler.

**Arguments**:

- `sizes` - A 1D tensor of real numbers representing the size of each element in the dataset.
- `bucket_boundaries` - A 1D tensor of real numbers representing the boundaries of the bucket ranges.
  It will be first sorted and used to create `len(bucket_boundaries) - 1` left-closed right-open intervals as bucket ranges.
  It should not contain any duplicate values.
- `base_batch_sampler_class` - Base batch sampler class type, which will be used for each bucket, and initialized with the bucket element indices,
  `base_batch_sampler_shared_kwargs` and the corresponding `base_batch_sampler_individual_kwargs`.
- `base_batch_sampler_shared_kwargs` - Shared keyword argument dictionary used to initialize all base batch samplers for all buckets.
  Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_individual_kwargs`. Default to {}.
- `base_batch_sampler_individual_kwargs` - Keyword argument dictionary used to initialize
  each bucket batch sampler with the corresponding key value pairs.
  Length of each value in this dict must be equal to len(bucket_boundaries) - 1 (the number of buckets).
  Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_shared_kwargs`.
  Default to {}.
- `shuffle` - A boolean indicating whether to shuffle the dataset and buckets. Defaults to True.
- `generator` - Generator used in sampling. Defaults to None.

**Raises**:

- `ValueError` - If `sizes` is not a 1D tensor of real numbers.
- `ValueError` - If `bucket_boundaries` is not a 1D tensor of real numbers.
- `ValueError` - If `base_batch_sampler_individual_kwargs` or `base_batch_sampler_individual_kwargs` is not a keyword argument dictionary.
- `ValueError` - If the length of values in the dict of `base_batch_sampler_individual_kwargs` must be equal to len(bucket_boundaries) - 1.
- `RuntimeError` - If there is no elements with sizes inside the ranges specified by `bucket_boundaries`.

<a id="sampler.BucketBatchSampler.__len__"></a>

#### \_\_len\_\_

```python
def __len__() -> int
```

Get the number of batches.

Can only be called if the `base_batch_sampler_class` has __len__() implemented

**Returns**:

- `int` - Number of batches

<a id="sampler.BucketBatchSampler.__iter__"></a>

#### \_\_iter\_\_

```python
def __iter__() -> Iterator[List[int]]
```

Iterate over batches of indices.

This function yields batches of indices of elements with sizes from each bucket range.

**Yields**:

- `List[int]` - A batch of indices of elements with sizes from each bucket range.
