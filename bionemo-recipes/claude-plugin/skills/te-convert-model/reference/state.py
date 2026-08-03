# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""State dict conversion utilities for mapping weights between HF and TE model formats.

This module provides the transform system used by convert.py to map state dicts:

- ``mapping``: A dict of simple key renames (source_key -> target_key). Each source key is
  copied directly to the corresponding target key with no modification to the tensor values.

- ``transforms``: A list of ``StateDictTransform`` objects for multi-key merges and splits.
  These handle cases where multiple source keys must be combined into one target key
  (e.g., merging Q/K/V into fused QKV), or one source key must be split into multiple target keys.

  Important: When ``source_key`` is a tuple (many-to-one merge), the transform function's
  parameter names are used to map each source key to a function argument. This means ``*args``
  style parameters do not work; each parameter must be explicitly named
  (e.g., ``def fn(q, k, v)`` not ``def fn(*args)``).

Adapted from nemo.lightning.io.state.
"""

import inspect
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar, Union, overload

import numpy as np
import torch
from torch import nn


logger = logging.getLogger(__name__)

SourceModuleT = TypeVar("SourceModuleT", bound=nn.Module)
TargetModuleT = TypeVar("TargetModuleT", bound=nn.Module)
F = TypeVar("F", bound=Callable[..., Any])


@dataclass
class TransformCTX:
    """Context passed to every transform function.

    Attributes:
        source: The source nn.Module (provides .config for reading hyperparams).
        source_state: Flat dict of source model's state_dict entries.
        target: The target nn.Module (provides .config for reading hyperparams).
        target_state: Flat dict of target model's state_dict entries (mutated in-place).
    """

    source: nn.Module
    source_state: dict
    target: nn.Module
    target_state: dict


class _ModelState:
    """Helper to wrap a raw state_dict as a source model for apply_transforms."""

    def __init__(self, state_dict, config=None):
        self._state_dict = state_dict
        self.config = config

    def state_dict(self):
        return self._state_dict

    def to(self, dtype):
        for k, v in self._state_dict.items():
            if v.dtype != dtype:
                logger.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
            self._state_dict[k] = v.to(dtype)


@torch.no_grad
def apply_transforms(
    source: Union[nn.Module, _ModelState],
    target: TargetModuleT,
    mapping: Dict[str, str],
    transforms: Optional[List[Callable[[TransformCTX], TransformCTX]]] = None,
    state_dict_ignored_entries: Optional[List] = None,
    cast_dtype: Optional[torch.dtype] = None,
) -> TargetModuleT:
    """Transform source state dict to match target model structure.

    1. Applies simple key renames from ``mapping``
    2. Applies each transform (merge/split operations)
    3. Copies tensors into target model parameters and buffers
    4. Validates shapes and checks for leftover/missing keys

    Args:
        source: Source model or _ModelState wrapper.
        target: Target model (parameters will be replaced in-place).
        mapping: Simple key renames {source_key: target_key}. Use "*" as wildcard.
        transforms: List of StateDictTransform instances for complex operations.
        state_dict_ignored_entries: Target state dict keys to skip (e.g., tied weights).
        cast_dtype: If set, cast output model to this dtype.

    Returns:
        The target model with weights populated from source.

    Raises:
        ValueError: Shape mismatch between source and target parameters.
        RuntimeError: Unmapped keys remain in target state dict.
    """
    if transforms is None:
        transforms = []
    if state_dict_ignored_entries is None:
        state_dict_ignored_entries = []

    target_orig_dtypes = extract_dtypes(target.named_parameters())

    target_state = target.state_dict()
    ctx = TransformCTX(
        source=source,
        source_state=source.state_dict(),
        target=target,
        target_state=target_state,
    )

    # Step 1: Apply simple key renames
    for key, val in mapping.items():
        ctx = StateDictTransform(key, val)(ctx)

    # Step 2: Apply complex transforms (QKV merge, embedding padding, etc.)
    for transform in transforms:
        ctx = transform(ctx)

    # Step 3: Copy tensors into target model parameters
    _params: Dict[str, nn.Parameter] = {}
    for name, param in target.named_parameters():
        if name in target_state:
            target_param = target_state[name]
            if param.data.shape != target_param.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name}: target shape {param.shape} vs "
                    f"converted source shape {target_param.shape}"
                )
            _params[name] = nn.Parameter(target_param, requires_grad=param.requires_grad)
            target_state.pop(name)
        else:
            print(f"Unexpected key: {name} not in target model but is in source model.")

    for key, val in _params.items():
        _module, _key = target, key
        if "." in key:
            for part in key.split(".")[:-1]:
                _module = getattr(_module, part)
            _key = key.split(".")[-1]
        _module.register_parameter(_key, val)

    # Step 4: Copy buffers
    _buffers = {}
    for name, buffer in target.named_buffers():
        if name in target_state:
            if buffer.shape != target_state[name].shape:
                raise ValueError(f"Shape mismatch for buffer {name}: {buffer.shape} vs {target_state[name].shape}")
            _buffers[name] = nn.Parameter(target_state[name], requires_grad=False)
            target_state.pop(name)

    for key, val in _buffers.items():
        _module, _key = target, key
        if "." in key:
            for part in key.split(".")[:-1]:
                _module = getattr(_module, part)
            _key = key.split(".")[-1]
        _module.register_buffer(_key, val)

    # Step 5: Validate no unmapped keys remain
    keys = list(filter(lambda x: x is not None and not x.endswith("_extra_state"), target_state.keys()))
    keys = [key for key in keys if key not in state_dict_ignored_entries]
    if len(keys) != 0:
        raise RuntimeError(f"Additional keys: {keys} in target model but not in source model.")

    if hasattr(target, "tie_weights"):
        target.tie_weights()

    # Step 6: Verify no meta tensors remain (all weights were converted)
    meta_tensor_keys = []
    for name, param in target.named_parameters():
        if param.is_meta:
            meta_tensor_keys.append(name)
    assert not meta_tensor_keys, (
        f"{meta_tensor_keys}\nThere are meta tensors in the model after conversion."
        f"Did you forget to include these parameters in the mapping or transforms in `convert_state`?"
    )

    if cast_dtype:
        target.to(cast_dtype)
    else:
        target_new_dtypes = extract_dtypes(target.named_parameters())
        for key in target_orig_dtypes.keys():
            if key in target_new_dtypes:
                assert target_orig_dtypes[key] == target_new_dtypes[key], (
                    f"dtype mismatch for key {key}: {target_orig_dtypes[key]} vs {target_new_dtypes[key]}"
                )

    return target


def _default_transform(inp):
    return inp


class StateDictTransform(Generic[F]):
    """A transformation that maps keys between source and target state dicts.

    Supports wildcards (*) in key patterns for matching layer indices.
    Can handle 1:1 renames, N:1 merges, and 1:N splits.

    Args:
        source_key: Source key pattern(s). Use "*" for layer index wildcards.
        target_key: Target key pattern(s).
        transform: Callable that transforms tensor values. Receives TransformCTX
            as first arg (if it accepts 'ctx'), plus matched source tensors.
    """

    def __init__(
        self,
        source_key: Union[str, Tuple[str, ...], Dict[str, str]],
        target_key: Union[str, Tuple[str, ...]],
        transform: F = _default_transform,
    ):
        """Initialize StateDictTransform with source/target key patterns and transform function."""
        self.source_key = source_key
        self.target_key = target_key
        self.transform = transform

    def __call__(self, ctx: TransformCTX) -> TransformCTX:
        """Perform the transformation on the given context."""
        source_key = self.source_key
        target_key = self.target_key
        source_dict, target_dict = ctx.source_state, ctx.target_state
        np.set_printoptions(threshold=10)
        fn_params = dict(inspect.signature(self.transform).parameters)
        fn_params.pop("ctx", None)
        matched = False

        if isinstance(source_key, (dict, tuple)):
            # Multi-source merge: e.g., (q_proj, k_proj, v_proj) -> layernorm_qkv
            if isinstance(source_key, tuple):
                source_key_dict = {param: source_key[i] for i, param in enumerate(fn_params)}
            else:
                source_key_dict = source_key
            source_matches_dict = {k: _match_keys(list(source_dict.keys()), v) for k, v in source_key_dict.items()}
            target_matches = _match_keys(list(target_dict.keys()), target_key)
            param_names = list(filter(lambda x: x in source_matches_dict, fn_params))
            source_matches = [
                source_matches_dict[v] if source_matches_dict[v].ndim > 0 else [source_matches_dict[v].item()]
                for v in param_names
            ]
            target_matches = [target_matches if target_matches.ndim > 0 else [target_matches.item()]]
            for layer_names_group in zip(*(source_matches + target_matches)):
                if isinstance(layer_names_group[0], str):
                    layer_names_group = [[x] for x in layer_names_group]
                for layer_names in zip(*layer_names_group):
                    target_dict[layer_names[-1]] = self.call_transform(
                        ctx, **dict(zip(param_names, [source_dict[x] for x in layer_names[:-1]]))
                    )
                matched = True
        else:
            # Single-source: 1:1 rename or 1:N split
            source_keys = list(source_dict.keys())
            target_keys = list(target_dict.keys())

            source_matches = _match_keys(source_keys, source_key)
            if source_matches.size == 1 and source_matches == np.array(None):
                raise ValueError(f"No matches found for source key: {source_key}")

            if isinstance(target_key, str):
                target_matches = _match_keys(target_keys, target_key)
                if target_matches.size == 1 and target_matches == np.array(None):
                    raise ValueError(f"No matches found for target key: {target_key}")
            else:
                if isinstance(target_key, dict):
                    raise ValueError("Target key must be a string or a tuple of strings.")
                _matches = [_match_keys(target_keys, key) for key in target_key]
                target_matches = np.stack(_matches, axis=-1)

            multiple_sources = source_matches.ndim >= target_matches.ndim
            accepts_var_args = any(
                param.kind == param.VAR_POSITIONAL for param in inspect.signature(self.transform).parameters.values()
            )

            if multiple_sources:
                for target_index, target_match in np.ndenumerate(target_matches):
                    try:
                        source_match = source_matches[target_index]
                    except IndexError as e:
                        logger.error(f"Encountered IndexError during transform.\n{source_matches=}\n{target_matches=}")
                        raise e
                    if accepts_var_args:
                        source_values = [source_dict[k] for k in source_match]
                        target_dict[target_match] = self.call_transform(ctx, *source_values)
                    else:
                        _source_match_list = [source_match] if isinstance(source_match, str) else list(source_match)
                        if len(fn_params) != len(_source_match_list):
                            raise ValueError(
                                f"Mismatch between source and target keys: {source_match} vs {target_match}"
                            )
                        kwargs = {param: source_dict[k] for param, k in zip(fn_params, _source_match_list)}
                        target_dict[target_match] = self.call_transform(ctx, **kwargs)
                    matched = True
            else:
                for source_index, source_match in np.ndenumerate(source_matches):
                    target_match = target_matches[source_index]
                    source_values = (
                        [source_dict[source_match]]
                        if np.isscalar(source_match)
                        else [source_dict[k] for k in source_match]
                    )
                    if accepts_var_args:
                        outputs = self.call_transform(ctx, *source_values)
                    else:
                        kwargs = dict(zip(fn_params, source_values))
                        outputs = self.call_transform(ctx, **kwargs)

                    if isinstance(target_match, str):
                        target_dict[target_match] = outputs
                    else:
                        for i, t in enumerate(outputs):
                            target_dict[target_match[i]] = t
                    matched = True

        if not matched:
            logger.warning(f"No matches found for source key: {source_key=} {target_key=}")
        return ctx

    def call_transform(self, ctx: TransformCTX, *args, **kwargs):
        """Invoke transform fn, injecting ctx if the function accepts it."""
        func_params = inspect.signature(self.transform).parameters
        expected_num_args = len([p for p in func_params if p not in ["self", "ctx"]])
        provided_num_args = len(args) + len(kwargs)
        accepts_var_args = any(param.kind == param.VAR_POSITIONAL for param in func_params.values())

        if not accepts_var_args and provided_num_args != expected_num_args:
            raise ValueError(
                f"Expected {expected_num_args} arguments for the transformation function, but got {provided_num_args}."
            )

        if "ctx" in func_params:
            return self.transform(ctx, *args, **kwargs)

        return self.transform(*args, **kwargs)


def _match_keys(keys: List[str], pattern: str) -> np.ndarray:
    """Match state dict keys against a pattern with wildcards.

    Supports:
    - "*" matches a single path segment (e.g., layer index)
    - "**" matches any characters including dots

    Returns an ndarray where each dimension corresponds to a wildcard position.
    """
    escaped_pattern = ""
    i = 0
    wildcard_positions = []
    while i < len(pattern):
        if pattern[i : i + 2] == "**":
            escaped_pattern += r"(.+)"
            wildcard_positions.append("**")
            i += 2
        elif pattern[i] == "*":
            escaped_pattern += r"([^.]+)"
            wildcard_positions.append("*")
            i += 1
        else:
            if pattern[i] == ".":
                escaped_pattern += r"\."
            else:
                escaped_pattern += pattern[i]
            i += 1

    regex_pattern = re.compile("^" + escaped_pattern + "$")
    num_wildcards = len(wildcard_positions)
    wildcard_matches = [[] for _ in range(num_wildcards)]

    for key in filter(lambda x: x is not None, keys):
        match = regex_pattern.match(key)
        if match:
            for i, group in enumerate(match.groups()):
                if group not in wildcard_matches[i]:
                    wildcard_matches[i].append(group)

    for i in range(len(wildcard_matches)):
        wildcard_matches[i].sort(key=lambda x: int(x) if x.isdigit() else x)

    shape = [len(matches) for matches in wildcard_matches]
    if len(wildcard_matches) == 0:
        shape = [1]

    output_array = np.empty(shape, dtype=object)

    for key in filter(lambda x: x is not None, keys):
        match = regex_pattern.match(key)
        if match:
            indices = [wildcard_matches[i].index(group) for i, group in enumerate(match.groups())]
            output_array[tuple(indices)] = key

    return output_array


@overload
def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]],
    target_key: Union[str, Tuple[str, ...]],
) -> Callable[[F], StateDictTransform[F]]: ...


@overload
def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]], target_key: Union[str, Tuple[str, ...]], fn: F
) -> StateDictTransform[F]: ...


def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]],
    target_key: Union[str, Tuple[str, ...]],
    fn: Optional[F] = None,
):
    """Create a StateDictTransform. Can be used as a decorator or called directly.

    Usage as decorator:
        @state_transform(source_key="a.*.weight", target_key="b.*.weight")
        def my_transform(ctx, weight):
            return weight * 2

    Usage with fn argument (inline):
        state_transform(source_key=(...), target_key="...", fn=TransformFns.merge_qkv)
    """

    def wrapper(fn) -> StateDictTransform:
        return StateDictTransform(source_key, target_key, fn)

    if fn is None:
        return wrapper
    return wrapper(fn)


class TransformFns:
    """Common transform functions for state dict conversion."""

    @staticmethod
    def split_qkv(ctx: TransformCTX, linear_qkv: torch.Tensor):
        """Split interleaved fused QKV into separate Q, K, V tensors.

        Handles GQA by computing the correct slicing for grouped query heads.
        """
        target_config = ctx.target.config
        head_num = target_config.num_attention_heads
        num_query_groups = target_config.num_key_value_heads
        heads_per_group = head_num // num_query_groups
        hidden_size = target_config.hidden_size
        head_size = hidden_size // head_num
        qkv_total_dim = head_num + 2 * num_query_groups

        linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
        hidden_size = linear_qkv.size(-1)

        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
        k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
        v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()
        return q_proj, k_proj, v_proj

    @staticmethod
    def merge_qkv(ctx: TransformCTX, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """Merge separate Q, K, V into interleaved fused QKV tensor.

        Handles GQA: for each query group, interleaves Q heads with their K/V heads.
        Layout: [group0_q_heads, group0_k, group0_v, group1_q_heads, group1_k, group1_v, ...]
        """
        target_config = ctx.target.config
        head_num = target_config.num_attention_heads
        num_query_groups = target_config.num_key_value_heads
        heads_per_group = head_num // num_query_groups
        hidden_size = target_config.hidden_size
        head_size = hidden_size // head_num
        old_tensor_shape = q.size()
        new_q_tensor_shape = (head_num, head_size, *old_tensor_shape[1:])
        new_kv_tensor_shape = (num_query_groups, head_size, *old_tensor_shape[1:])

        q = q.view(*new_q_tensor_shape)
        k = k.view(*new_kv_tensor_shape)
        v = v.view(*new_kv_tensor_shape)

        qkv_weights_l = []
        for i in range(num_query_groups):
            qkv_weights_l.append(q[i * heads_per_group : (i + 1) * heads_per_group, :, :])
            qkv_weights_l.append(k[i : i + 1, :, :])
            qkv_weights_l.append(v[i : i + 1, :, :])
        qkv_weights = torch.cat(qkv_weights_l)
        assert qkv_weights.ndim == 3, qkv_weights.shape
        assert qkv_weights.shape[0] == (heads_per_group + 2) * num_query_groups, qkv_weights.shape
        assert qkv_weights.shape[1] == head_size, qkv_weights.shape

        qkv_weights = qkv_weights.reshape([head_size * (head_num + 2 * num_query_groups), hidden_size])
        return qkv_weights

    @staticmethod
    def merge_fc1(gate: torch.Tensor, up: torch.Tensor):
        """Merge gate and up projections into concatenated fc1 (for SwiGLU)."""
        return torch.cat((gate, up), dim=0)

    @staticmethod
    def split_fc1(linear_fc1: torch.Tensor):
        """Split concatenated fc1 back into gate and up projections."""
        gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)
        return gate_proj, up_proj

    @staticmethod
    def prune_padding(ctx: TransformCTX, embedding: torch.Tensor):
        """Prune embedding to original vocab_size (remove FP8 padding)."""
        return embedding[: ctx.target.config.vocab_size, :]


def extract_dtypes(ckpt):
    """Extract dtype for each parameter/tensor in a named iterator."""
    dtypes = {}
    for key, val in ckpt:
        if hasattr(val, "dtype"):
            dtypes[key] = val.dtype
        elif hasattr(val, "data") and hasattr(val.data, "dtype"):
            dtypes[key] = val.data.dtype
    return dtypes
