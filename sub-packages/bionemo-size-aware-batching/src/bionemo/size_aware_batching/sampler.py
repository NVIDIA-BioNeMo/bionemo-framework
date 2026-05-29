# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Type, TypeVar, Union

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


__all__: Sequence[str] = (
    "BucketBatchSampler",
    "Real",
    "SizeAwareBatchSampler",
    "size_aware_batching",
)

Data = TypeVar("Data")
BatchCollated = TypeVar("BatchCollated")
Real = Union[int, float]
TorchIntegerDataTypes = {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}  # type: ignore
S = TypeVar("S", bound=Sampler)


def size_aware_batching(
    dataset: Iterable[Data],
    sizeof: Callable[[Data], Real],
    max_total_size: Real,
    collate_fn: Optional[Callable[[Iterable[Data]], BatchCollated]] = None,
    info_logger: Optional[Callable[[str], None]] = None,
    warn_logger: Optional[Callable[[str], None]] = None,
) -> Iterator[Union[List[Data], BatchCollated]]:
    """Creates a batching iterator where each batch size varries (within a max limit) according to memory consumption.

    A generator that batches elements from an iterable while ensuring that the
    total size of each batch does not exceed a specified maximum. Here the size
    can be a measurement of memory consumption of the elements in the batch.
    This can be useful for both indexible data or non-indexible but iterable data.

    Args:
        dataset: The input iterable.
        sizeof: A function or mapping that returns the "size" of each element in `dataset`.
            E.g., this can used to determine how much memory an element consumes. Its return
            type must be comparable with `max_total_size` and it must be addable (operator `+`).
        max_total_size: The maximum total "size" of each batch. The semantics of "size"
            is defined by the `sizeof` argument. The type of this value must be comparable
            with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
        collate_fn: An optional function to collate batches. Defaults to None, in which case
            each batch is a list of elements from the input dataset
        info_logger: A function to log info. Defaults to None.
        warn_logger: A function to log warnings. Defaults to None.

    Yields:
        A generator that yields batches from `dataset`.

    -----------
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


    ------
    Caveat
    1: The generated batch sizes may have large variance
       - how to workaround: filter the output of this generator using a batch size threshold
    2: The number of batches may vary a lot across different epochs.
       - how to workaround: increase the number of steps that compose an epoch,
         e.g., in the Lightning training/validation loop, which effectively increases the input
         dataset size per epoch


    -------

    Example:
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

    """
    is_sizeof_callable = callable(sizeof)
    has_collate_fn = collate_fn is not None and callable(collate_fn)

    if not is_sizeof_callable:
        raise TypeError("sizeof must be a callable")

    batch_total_size = 0
    batch = []
    n_samples = 0
    n_samples_batched = 0
    n_batches = 0
    for data in dataset:
        n_samples += 1
        try:
            new_size = sizeof(data)
        except Exception as e:
            raise RuntimeError(f"sizeof raises error at data={data}: {e}") from e
        if new_size > max_total_size:
            if warn_logger is not None:
                warn_logger(f"Size of element {data} exceeds max_total_size ({new_size} > {max_total_size}), skipping")
            continue
        if new_size + batch_total_size > max_total_size:
            n_batches += 1
            if has_collate_fn:
                yield collate_fn(batch)
            else:
                yield batch
            batch_total_size = 0
            batch = []
        batch.append(data)
        n_samples_batched += 1
        batch_total_size += new_size

    # return the remaining batch if there is
    if len(batch) > 0:
        n_batches += 1
        if has_collate_fn:
            yield collate_fn(batch)
        else:
            yield batch

    if warn_logger is not None and n_samples_batched < n_samples:
        warn_logger(
            f"{n_samples_batched} samples were batched from {n_samples} "
            f"of the input data. Missing samples are due to exceeding max_total_size={max_total_size})"
        )

    if info_logger is not None:
        info_logger(
            f"Batched {n_samples_batched} samples into {n_batches} batches. "
            f"If this doesn't match the your expectation, consider adjusting "
            f"max_total_size or the sizeof functor"
        )


class SizeAwareBatchSampler(Sampler[List[int]]):
    """Varriying-size batching data sampler class that ensures batch size doesn't exceed maximum.

    A sampler that batches elements of varying sizes while ensuring
    that the total size of each batch does not exceed a specified maximum.

    This is useful when dealing with datasets where each element has a
    different size, such as graphs or sequences of varying lengths.
    The sampler uses a provided `sizeof` function to determine the size
    of each element in the dataset and ensures that the total size of
    each batch does not exceed the specified `max_total_size`.

    ---------

    Examples:
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
    """

    def __init__(
        self,
        sampler: Union[Sampler[List[int]], Iterable[int]],
        sizeof: Callable[[int], Real],
        max_total_size: Real,
        info_logger: Optional[Callable[[str], None]] = None,
        warn_logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Initializes the SizeAwareBatchSampler.

        Args:
            sampler: The underlying sampler.
            sizeof: A function that returns the size at each index. E.g., this can used to
                determine how much memory an element consumes. Its return type must be
                comparable with `max_total_size` and it must be addable (operator `+`).
            max_total_size: The maximum total size of a mini-batch. The semantics of "size"
                is defined by the `sizeof` argument. The type of this value must be comparable
                with the return type of sizeof, i.e., the operator `<` and `==` must be meaningful.
            info_logger: A function to log info. Defaults to None.
            warn_logger: A function to log warnings. Defaults None.

        Raises:
            TypeError: If sampler is not an instance of Sampler or Iterable, or if sizeof is not a callable, dictionary, or sequence container.
            ValueError: If max_total_size is not a positive number.

        """
        if not (isinstance(sampler, Sampler) or (isinstance(sampler, Iterable) and not isinstance(sampler, str))):
            raise TypeError("sampler should be an instance of torch.utils.data.Sampler or Iterable")

        if not isinstance(max_total_size, Real):
            raise ValueError(f"max_total_size should be int or float but got {type(max_total_size)}")

        self._info_logger = info_logger
        self._warn_logger = warn_logger

        self._is_sizeof_callable = callable(sizeof)

        if not self._is_sizeof_callable:
            raise TypeError("sizeof must be a callable")

        self._sampler = sampler
        self._sizeof = sizeof
        self._max_total_size = max_total_size

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches of indices.

        This function yields batches of indices that do not exceed the maximum total size.

        Yields:
            A batch of indices that do not exceed the maximum total size.
        """
        return size_aware_batching(
            self._sampler,
            self._sizeof,
            self._max_total_size,
            collate_fn=None,
            info_logger=self._info_logger,
            warn_logger=self._warn_logger,
        )


class BucketBatchSampler(Sampler[List[int]]):
    """A batch sampler to create batches with sizes of elements from each pre-defined bucket ranges.

    Elements of the dataset are first grouped into each bucket based on the bucket ranges and the sizes of elements.
    Then, a base batch sampler is used for each bucket to create mini-batches.

    The bucket ranges are specified by `bucket_boundaries`, which will be first sorted internally and used to create
    `len(bucket_boundaries) - 1` left-closed right-open intervals.
    e.g. if bucket_boundaries tensor is [10, 5, 0, 16], it will be sorted as [0, 5, 10, 16] and 3 buckets will be created
    with ranges: [0, 5), [5, 10), [10, 16).

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

    ---------

    Examples:
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
    """

    def __init__(
        self,
        sizes: torch.Tensor,
        bucket_boundaries: torch.Tensor,
        base_batch_sampler_class: Type[S],
        base_batch_sampler_shared_kwargs: Optional[Dict[str, Any]] = None,
        base_batch_sampler_individual_kwargs: Optional[Dict[str, Iterable]] = None,
        shuffle: Optional[bool] = True,
        generator: Optional[torch.Generator] = None,
        sampler: Optional[Union[Sampler[int], Iterable[int]]] = None,
        num_batches: Optional[int] = None,
    ) -> None:
        """Initializes the BucketBatchSampler.

        Args:
            sizes: A 1D tensor of real numbers representing the size of each element in the dataset.
            bucket_boundaries: A 1D tensor of real numbers representing the boundaries of the bucket ranges.
                It will be first sorted and used to create `len(bucket_boundaries) - 1` left-closed right-open intervals as bucket ranges.
                It should not contain any duplicate values.
            base_batch_sampler_class: Base batch sampler class type, which will be used for each bucket, and initialized with the bucket element indices,
                `base_batch_sampler_shared_kwargs` and the corresponding `base_batch_sampler_individual_kwargs`.
            base_batch_sampler_shared_kwargs: Shared keyword argument dictionary used to initialize all base batch samplers for all buckets.
                Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_individual_kwargs`. Default to  {}.
            base_batch_sampler_individual_kwargs: Keyword argument dictionary used to initialize
                each bucket batch sampler with the corresponding key value pairs.
                Length of each value in this dict must be equal to len(bucket_boundaries) - 1 (the number of buckets).
                Sufficient and valid arguments should be provided for `base_batch_sampler_class` with `base_batch_sampler_shared_kwargs`.
                Default to  {}.
            shuffle: A boolean indicating whether to shuffle the dataset and buckets. Defaults to True.
            generator: Generator used in sampling. Defaults to None.
            sampler: Sampler to get data sample indices for this batch sampler. When using data distributed parallel strategy for training models,
                the user should provide here the torch.utils.data.distributed.DistributedSampler or the equivalent to partition the data sample
                indices across devices. Defaults to None.
            num_batches: Number of mini-batches, when using data distributed parallel strategy for training models, this argument value should
                be consistent across training devices so that they get the same number of batches.. Defaults to None.

        Raises:
            TypeError: If `sizes` is not torch tensor.
            ValueError: If `sizes` is not a 1D array, or its entries are not real numbers.
            TypeError: If `bucket_boundaries` is not torch tensor.
            ValueError: If `bucket_boundaries` is not a 1D array, or its entries are not real numbers, or it has only 1 entry, or it has duplicate entries.
            TypeError: If `shuffle` is not bool.
            TypeError: If `generator` is not NoneType and not torch generator.
            TypeError: If `base_batch_sampler_class` is not a subclass of `torch.utils.data.Sampler`
            ValueError: If `base_batch_sampler_individual_kwargs` or `base_batch_sampler_individual_kwargs` is not a keyword argument dictionary.
            ValueError: If the length of values in the dict of `base_batch_sampler_individual_kwargs` is not equal to len(bucket_boundaries) - 1.
            TypeError: If `sampler` is not None and not an instance of `torch.utils.data.Sampler` or is not iterable.
            ValueError: if `num_batches` is None but distributed sampling is enabled.

        """
        if not torch.is_tensor(sizes):
            raise TypeError(f"sizes should be a torch tensor, but got sizes={sizes}")

        if sizes.ndim != 1:
            raise ValueError(f"sizes should be a 1D tensor, but got sizes with shape {sizes.shape}")

        if not torch.is_floating_point(sizes) and sizes.dtype not in TorchIntegerDataTypes:
            raise ValueError(
                f"sizes should contain only integers or floating point numbers, but got sizes.dtype={sizes.dtype}"
            )

        if not torch.is_tensor(bucket_boundaries):
            raise TypeError(
                f"bucket_boundaries should be a torch tensor, but got bucket_boundaries={bucket_boundaries}"
            )

        if bucket_boundaries.ndim != 1:
            raise ValueError(
                f"bucket_boundaries should be a 2D tensor, but got bucket_boundaries with shape {bucket_boundaries.shape}"
            )

        if len(bucket_boundaries) < 2:
            raise ValueError(
                f"bucket_boundaries should have at least 2 numbers, but got bucket_boundaries={bucket_boundaries.shape}"
            )

        if not torch.is_floating_point(bucket_boundaries) and bucket_boundaries.dtype not in TorchIntegerDataTypes:
            raise ValueError(
                f"bucket_boundaries should contain only integers or floating point numbers, but got bucket_boundaries.dtype={bucket_boundaries.dtype}"
            )

        bucket_boundaries = torch.sort(bucket_boundaries)[0]

        if torch.any(bucket_boundaries[:-1] >= bucket_boundaries[1:]):
            raise ValueError(
                f"bucket_boundaries should not have duplicate values, and should specify the lower endpoint of each interval smaller than the upper endpoint, but got sorted bucket_boundaries={bucket_boundaries}"
            )

        if not isinstance(shuffle, bool):
            raise TypeError(f"shuffle should be a boolean value, but got shuffle={shuffle}")

        if generator is not None and not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch generator or None value, but got generator={type(generator)}")

        self.sizes = sizes
        self.bucket_boundaries = bucket_boundaries
        self.num_buckets = len(bucket_boundaries) - 1

        num_outliers = (self.sizes < self.bucket_boundaries[0]).sum() + (
            self.sizes >= self.bucket_boundaries[-1]
        ).sum()
        if num_outliers > 0:
            warnings.warn(f"{num_outliers} elements are outside the buckets provided and will be skipped")

        self.shuffle = shuffle
        self.generator = generator

        if self.shuffle and self.generator is None:
            self.generator = torch.Generator().manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))

        if not issubclass(base_batch_sampler_class, Sampler):
            raise TypeError(
                f"base_batch_sampler_class should be a batch sampler class inherited from torch.utils.data.Sampler, but got base_batch_sampler_class={base_batch_sampler_class}"
            )

        base_batch_sampler_shared_kwargs = (
            {} if base_batch_sampler_shared_kwargs is None else base_batch_sampler_shared_kwargs
        )
        base_batch_sampler_individual_kwargs = (
            {} if base_batch_sampler_individual_kwargs is None else base_batch_sampler_individual_kwargs
        )
        if not isinstance(base_batch_sampler_shared_kwargs, dict):
            raise TypeError(
                f"base_batch_sampler_shared_kwargs should be a dictionary, but got base_batch_sampler_shared_kwargs={base_batch_sampler_shared_kwargs}"
            )

        if not all(isinstance(key, str) for key in base_batch_sampler_shared_kwargs.keys()):
            raise TypeError(
                f"base_batch_sampler_shared_kwargs should have string keys, but got keys={list(base_batch_sampler_shared_kwargs.keys())}"
            )

        if not isinstance(base_batch_sampler_individual_kwargs, dict):
            raise TypeError(
                f"base_batch_sampler_individual_kwargs should be a dictionary, but got base_batch_sampler_individual_kwargs={base_batch_sampler_individual_kwargs}"
            )

        if not all(isinstance(key, str) for key in base_batch_sampler_individual_kwargs.keys()):
            raise TypeError(
                f"base_batch_sampler_individual_kwargs should have string keys, but got keys={list(base_batch_sampler_individual_kwargs.keys())}"
            )

        if not all(len(list(value)) == self.num_buckets for value in base_batch_sampler_individual_kwargs.values()):
            raise ValueError(
                f"Each value in base_batch_sampler_individual_kwargs should have a length of {self.num_buckets}, "
                f"but got lengths {[len(list(value)) for value in base_batch_sampler_individual_kwargs.values()]}"
            )

        self.base_batch_sampler_class = base_batch_sampler_class
        self.base_batch_sampler_shared_kwargs = base_batch_sampler_shared_kwargs
        self.base_batch_sampler_individual_kwargs = base_batch_sampler_individual_kwargs

        self.bucket_sizes: torch.Tensor  # number of elements in each bucket
        self.bucket_element_indices: List[List[int]]  # List of elements' indices for each bucket

        # bucket index for each element
        self.element_bucket_indices = torch.bucketize(sizes, self.bucket_boundaries, right=True)

        if sampler is not None and not isinstance(sampler, (Sampler, Iterable)):  # type: ignore
            raise TypeError(
                f"sampler should be an instance of torch.utils.data.Sampler or an iterable, but got sampler={type(sampler)}"
            )

        self.sampler = sampler
        self.bucket_element_indices = [[] for _ in range(self.num_buckets)]

        if self.sampler is None:
            self._compute_bucket_element_indices()
        else:
            self._compute_bucket_element_indices(list(self.sampler))  # type: ignore

        self.base_batch_samplers = self._init_base_batch_samplers()

        if num_batches is not None and (not isinstance(num_batches, int) or num_batches <= 0):
            raise ValueError(f"num_batches should be a positive integer, but got num_batches = {num_batches}")

        if dist.is_initialized() and num_batches is None:
            raise ValueError("num_batches is not set but distributed sampling is enabled")

        self.num_batches = num_batches

    def setup_for_pytorch_lightning(self) -> Sampler:
        """Add additional attributes necessary to be compatible with pytorch lightning distributed sampling.

        When used for distributed sampling, Pytorch Lightning will use these attributes to re-init this batch sampler and
        replace the sampler to torch.utils.data.DistributedSampler if the lightning trainer is instantiated with
        use_distributed_sampler=True.

        Returns:
            Sampler: this batch sampler instance.
        """
        setattr(self, "__pl_saved_args", ())
        setattr(
            self,
            "__pl_saved_kwargs",
            {
                "sizes": self.sizes,
                "bucket_boundaries": self.bucket_boundaries,
                "base_batch_sampler_class": self.base_batch_sampler_class,
                "base_batch_sampler_shared_kwargs": self.base_batch_sampler_shared_kwargs,
                "base_batch_sampler_individual_kwargs": self.base_batch_sampler_individual_kwargs,
                "shuffle": self.shuffle,
                "generator": self.generator,
                "sampler": self.sampler,
                "num_batches": self.num_batches,
            },
        )
        setattr(self, "__pl_saved_default_kwargs", {})
        setattr(self, "__pl_saved_arg_names", ())
        return self

    def _compute_bucket_element_indices(self, indices: Optional[List[int]] = None):
        """Compute and update the bucket element indices with the given element indices.

        Args:
            indices: Data indices subset, will use all indices if it is None. Defaults to None.

        Raises:
            RuntimeError: If there is no elements from the given indices having sizes
            inside the ranges specified by `self.bucket_boundaries`.
        """
        if indices is None:
            element_bucket_indices = self.element_bucket_indices
            # element indices reordered for each bucket
            reordered_element_indices = torch.argsort(element_bucket_indices, stable=True)
        else:
            indices_subst: torch.Tensor = torch.tensor(indices)  # type: ignore
            element_bucket_indices = self.element_bucket_indices[indices_subst]
            # element indices reordered for each bucket
            reordered_element_indices = indices_subst[torch.argsort(element_bucket_indices, stable=True)]

        # bucket sizes, including the buckets for < bucket_boundaries[0] and >= bucket_boundaries[-1]
        bucket_sizes = torch.bincount(element_bucket_indices, minlength=len(self.bucket_boundaries) + 1)

        # bucket segments
        bucket_sizes_cumsum = torch.cumsum(bucket_sizes, dim=0)[:-1]

        # exclude the buckets for < bucket_boundaries[0] and >= bucket_boundaries[-1]
        for bucket_idx in range(self.num_buckets):
            self.bucket_element_indices[bucket_idx][:] = reordered_element_indices[
                bucket_sizes_cumsum[bucket_idx] : bucket_sizes_cumsum[bucket_idx + 1]
            ].tolist()

        self.bucket_sizes = bucket_sizes[1 : (self.num_buckets + 1)]

        self.num_samples = torch.sum(self.bucket_sizes).item()
        if self.num_samples == 0:
            raise RuntimeError(
                "The sizes of the elements with given indices in the dataset are outside the bucket ranges provided"
            )

    def _init_base_batch_samplers(self) -> list[Sampler[List[int]]]:
        """Initialize batch samplers for each bucket.

        Returns:
            List of batch samplers.
        """
        base_batch_samplers = []
        for k in range(self.num_buckets):
            individual_kwargs = {
                key: list(self.base_batch_sampler_individual_kwargs[key])[k]
                for key in self.base_batch_sampler_individual_kwargs
            }
            base_batch_samplers.append(
                self.base_batch_sampler_class(
                    self.bucket_element_indices[k],
                    **self.base_batch_sampler_shared_kwargs,
                    **individual_kwargs,
                )
            )
        return base_batch_samplers

    def __len__(self) -> int:
        """Get the number of batches.

        Can only be called if the `base_batch_sampler_class` has __len__() implemented if num_batches is not provided.

        Returns:
            int: Number of batches
        """
        if self.num_batches is None:
            self.num_batches = sum(len(sampler) for sampler in self.base_batch_samplers)  # type: ignore
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        """Iterate over batches of indices.

        This function yields batches of indices of elements with sizes from each bucket range.

        Yields:
            List[int]: A batch of indices of elements with sizes from each bucket range.
        """
        if self.num_batches is None:
            self.num_batches = len(self)

        if self.sampler is not None:
            self._compute_bucket_element_indices(list(self.sampler))  # type: ignore
        elif self.shuffle:
            for indices in self.bucket_element_indices:
                idx = torch.randperm(len(indices), generator=self.generator)
                indices[:] = torch.tensor(indices)[idx].tolist()

        num_yield_batches = 0
        while num_yield_batches < self.num_batches:
            base_batch_sampler_iters = [iter(batch_sampler) for batch_sampler in self.base_batch_samplers]
            bucket_remaining_elements = self.bucket_sizes.clone()
            total_remaining_elements = self.num_samples

            while total_remaining_elements > 0 and num_yield_batches < self.num_batches:
                if self.shuffle:
                    bucket_idx = torch.multinomial(
                        bucket_remaining_elements / total_remaining_elements, 1, generator=self.generator
                    )
                else:
                    bucket_idx = torch.argmax((bucket_remaining_elements > 0).to(int))  # type: ignore

                try:
                    batch = next(base_batch_sampler_iters[bucket_idx])
                    batch_size = min(len(batch), bucket_remaining_elements[bucket_idx].item())
                    bucket_remaining_elements[bucket_idx] -= batch_size
                    total_remaining_elements -= batch_size
                    num_yield_batches += 1
                    yield batch
                except StopIteration:
                    bucket_remaining_elements[bucket_idx] = 0
                    total_remaining_elements = torch.sum(bucket_remaining_elements)
                    continue

    def set_epoch(self, epoch: int):
        """Set the epoch for `self.sampler`.

        Args:
            epoch (int): Epoch number.
        """
        if self.sampler is not None:
            try:
                self.sampler.set_epoch(epoch)  # type: ignore
            except AttributeError:
                raise RuntimeError(
                    f"Trying to call set_epoch on {self.sampler.__class__} which does not have such method"
                )
