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

"""Streaming MBridge-compatible weight loading for the Evo2 vLLM backend."""

import re
from collections.abc import Iterable

import torch
from torch import nn


_IGNORED_SOURCE_SUFFIXES = (
    "._extra_state",
    ".filter.t",
    ".rotary_emb.inv_freq",
)
_DERIVED_SOURCE_SUFFIXES = (".filter.h", ".filter.decay", ".filter.p", ".filter.gamma")
_VORTEX_BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.(.+)$")


def load_tensor_parallel_weight(
    parameter: nn.Parameter,
    loaded_weight: torch.Tensor,
    *,
    shard_dim: int,
    tp_rank: int | None = None,
    tp_size: int | None = None,
) -> None:
    """Copy a full or already-local tensor into one contiguous TP shard."""
    if tp_rank is None or tp_size is None:
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
    if tp_size < 1 or not 0 <= tp_rank < tp_size:
        raise ValueError(f"invalid tensor-parallel rank/size: rank={tp_rank}, size={tp_size}")
    if loaded_weight.ndim != parameter.ndim:
        raise ValueError(f"loaded tensor rank {loaded_weight.ndim} does not match parameter rank {parameter.ndim}")
    shard_dim %= parameter.ndim

    if loaded_weight.shape == parameter.shape:
        local_weight = loaded_weight
    else:
        expected_shape = list(parameter.shape)
        expected_shape[shard_dim] *= tp_size
        if tuple(loaded_weight.shape) != tuple(expected_shape):
            raise ValueError(
                f"loaded tensor shape {tuple(loaded_weight.shape)} cannot shard to parameter shape "
                f"{tuple(parameter.shape)} on dimension {shard_dim} with TP={tp_size}"
            )
        shard_size = parameter.shape[shard_dim]
        local_weight = loaded_weight.narrow(shard_dim, tp_rank * shard_size, shard_size)

    with torch.no_grad():
        parameter.copy_(local_weight.to(device=parameter.device, dtype=parameter.dtype))


def _copy_weight(parameter: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    if parameter.shape != loaded_weight.shape:
        raise ValueError(
            f"loaded tensor shape {tuple(loaded_weight.shape)} does not match parameter shape {tuple(parameter.shape)}"
        )
    with torch.no_grad():
        parameter.copy_(loaded_weight.to(device=parameter.device, dtype=parameter.dtype))


def _target_candidates(source_name: str) -> tuple[str, ...]:
    name = source_name.removeprefix("module.")
    if name.startswith("model."):
        return name, name.removeprefix("model.")
    return name, f"model.{name}"


def _hybrid_pattern(model: nn.Module) -> str:
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        if pattern := getattr(owner, "hybrid_override_pattern", None):
            return pattern
        if (config := getattr(owner, "config", None)) is not None and (
            pattern := getattr(config, "hybrid_override_pattern", None)
        ):
            return pattern
    raise ValueError("native Vortex loading requires the Evo2 hybrid_override_pattern")


def _map_vortex_weight(
    model: nn.Module,
    source_name: str,
    loaded_weight: torch.Tensor,
    pending_fc1: dict[int, dict[str, torch.Tensor]],
) -> list[tuple[str, torch.Tensor]]:
    name = source_name.removeprefix("module.")
    if name in ("embedding_layer.weight", "unembed.weight"):
        return [("embedding.word_embeddings.weight", loaded_weight)]
    if name == "norm.scale":
        return [("decoder.final_norm.weight", loaded_weight)]

    match = _VORTEX_BLOCK_PATTERN.match(name)
    if match is None:
        return [(name, loaded_weight)]
    layer_index = int(match.group(1))
    suffix = match.group(2)
    pattern = _hybrid_pattern(model)
    if layer_index >= len(pattern):
        raise ValueError(f"Vortex weight references layer {layer_index}, but the pattern has {len(pattern)} layers")
    symbol = pattern[layer_index]
    prefix = f"decoder.layers.{layer_index}"

    if suffix == "pre_norm.scale":
        owner = "self_attention.linear_qkv" if symbol == "*" else "mixer.dense_projection"
        return [(f"{prefix}.{owner}.layer_norm_weight", loaded_weight)]
    if suffix == "post_norm.scale":
        return [(f"{prefix}.mlp.linear_fc1.layer_norm_weight", loaded_weight)]
    if suffix in ("mlp.l1.weight", "mlp.l2.weight"):
        part = "l1" if ".l1." in suffix else "l2"
        parts = pending_fc1.setdefault(layer_index, {})
        parts[part] = loaded_weight
        if parts.keys() >= {"l1", "l2"}:
            fused = torch.cat((parts.pop("l1"), parts.pop("l2")), dim=0)
            del pending_fc1[layer_index]
            return [(f"{prefix}.mlp.linear_fc1.weight", fused)]
        return []
    if suffix == "mlp.l3.weight":
        return [(f"{prefix}.mlp.linear_fc2.weight", loaded_weight)]

    if symbol == "*":
        attention_mapping = {
            "inner_mha_cls.Wqkv.weight": "self_attention.linear_qkv.weight",
            "inner_mha_cls.out_proj.weight": "self_attention.linear_proj.weight",
            "inner_mha_cls.out_proj.bias": "self_attention.linear_proj.bias",
            "inner_mha_cls.rotary_emb.inv_freq": "self_attention.rotary_emb.inv_freq",
        }
        if suffix in attention_mapping:
            return [(f"{prefix}.{attention_mapping[suffix]}", loaded_weight)]
    else:
        hyena_mapping = {
            "projections.weight": "mixer.dense_projection.weight",
            "out_filter_dense.weight": "mixer.dense.weight",
            "out_filter_dense.bias": "mixer.dense.bias",
        }
        if suffix in hyena_mapping:
            return [(f"{prefix}.{hyena_mapping[suffix]}", loaded_weight)]
        if suffix == "filter.short_filter_weight":
            if loaded_weight.ndim == 3 and loaded_weight.shape[1] == 1:
                loaded_weight = loaded_weight.squeeze(1)
            return [(f"{prefix}.mixer.hyena_proj_conv.short_conv_weight", loaded_weight)]
        if suffix == "filter.h" and symbol == "S":
            return [(f"{prefix}.mixer.mixer.short_conv.short_conv_weight", loaded_weight)]
        if suffix == "filter.D" and symbol in ("D", "H"):
            return [(f"{prefix}.mixer.mixer.conv_bias", loaded_weight)]
        if suffix == "filter.h" and symbol == "D":
            if loaded_weight.ndim == 3 and loaded_weight.shape[1] == 1:
                loaded_weight = loaded_weight.squeeze(1)
            return [
                (f"{prefix}.mixer.mixer.filter.h", loaded_weight),
                (f"{prefix}.mixer.mixer.filter.decay", torch.ones_like(loaded_weight, dtype=torch.float32)),
            ]
        if suffix == "filter.log_poles" and symbol == "H":
            log_poles = loaded_weight.squeeze(-1).float()
            if not torch.all(log_poles < 0):
                raise ValueError("Vortex modal log_poles must be strictly negative")
            return [
                (f"{prefix}.mixer.mixer.filter.p", torch.log(-log_poles)),
                (f"{prefix}.mixer.mixer.filter.gamma", torch.zeros_like(log_poles)),
            ]
        if suffix == "filter.residues" and symbol == "H":
            return [(f"{prefix}.mixer.mixer.filter.R", loaded_weight.float())]

    raise ValueError(f"Evo2 checkpoint contains unknown Vortex weight: {source_name}")


def _refresh_filter_module(module: nn.Module) -> bool:
    if all(hasattr(module, name) for name in ("h", "decay", "effective_filter")):
        effective_filter = module.effective_filter
        with torch.no_grad():
            effective_filter.copy_(
                module.h[..., : effective_filter.shape[-1]].float()
                * module.decay[..., : effective_filter.shape[-1]].float()
            )
        return True
    if all(hasattr(module, name) for name in ("p", "gamma", "modal_decay")):
        with torch.no_grad():
            module.modal_decay.copy_(torch.exp(-torch.exp(module.p.float() + module.gamma.float())))
        return True
    return False


def refresh_derived_filters(model: nn.Module, module_names: set[str] | None = None) -> set[str]:
    """Refresh HCM effective filters and HCL poles in place after refit."""
    refreshed = set()
    if module_names is None:
        modules = model.named_modules()
    else:
        modules = ((name, model.get_submodule(name)) for name in sorted(module_names))
    for name, module in modules:
        if _refresh_filter_module(module):
            refreshed.add(name)
    return refreshed


def _required_parameter_ids(model: nn.Module) -> dict[int, str]:
    required: dict[int, str] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if not getattr(parameter, "_evo2_optional_weight", False):
            required.setdefault(id(parameter), name)
    return required


def load_evo2_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    strict: bool = True,
    pending_fc1: dict[int, dict[str, torch.Tensor]] | None = None,
) -> set[str]:
    """Stream native or MBridge-named tensors into an Evo2 vLLM model.

    The model intentionally preserves MBridge parameter paths. A leading vLLM ``model.`` wrapper
    and a DDP ``module.`` wrapper are accepted without materializing a renamed state dictionary.
    Tensor-parallel and quantized parameters can provide vLLM's conventional ``weight_loader``
    callback; ordinary replicated parameters are copied directly. The returned names are canonical
    entries from ``model.named_parameters()`` because vLLM uses them for strict initialization
    accounting even when checkpoint source names differ.
    """
    parameters = dict(model.named_parameters(remove_duplicate=False))
    canonical_names = {id(parameter): name for name, parameter in model.named_parameters()}
    required = _required_parameter_ids(model)
    loaded_parameter_ids: set[int] = set()
    loaded_parameter_names: set[str] = set()
    if pending_fc1 is None:
        pending_fc1 = {}

    for source_name, loaded_weight in weights:
        for mapped_name, mapped_weight in _map_vortex_weight(model, source_name, loaded_weight, pending_fc1):
            target_name = next(
                (candidate for candidate in _target_candidates(mapped_name) if candidate in parameters), None
            )
            if target_name is None:
                if mapped_name.endswith(_IGNORED_SOURCE_SUFFIXES):
                    continue
                raise ValueError(f"Evo2 checkpoint contains unknown weight: {source_name}")
            parameter = parameters[target_name]
            weight_loader = getattr(parameter, "weight_loader", _copy_weight)
            weight_loader(parameter, mapped_weight)
            loaded_parameter_ids.add(id(parameter))
            loaded_parameter_names.add(canonical_names[id(parameter)])

            if target_name.endswith(_DERIVED_SOURCE_SUFFIXES):
                refresh_derived_filters(model, {target_name.rsplit(".", maxsplit=1)[0]})

    if strict:
        missing = sorted(name for parameter_id, name in required.items() if parameter_id not in loaded_parameter_ids)
        if missing:
            raise ValueError(f"Evo2 checkpoint is missing mandatory weights: {missing}")
    return loaded_parameter_names


def _pending_fc1_signature(pending_fc1: dict[int, dict[str, torch.Tensor]]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    return tuple((layer_index, tuple(sorted(parts))) for layer_index, parts in sorted(pending_fc1.items()))


class IncrementalEvo2WeightLoader:
    """Track complete Evo2 initialization and refit transactions across chunks."""

    def __init__(self, model: nn.Module) -> None:
        """Record the model's mandatory canonical parameter names."""
        self.model = model
        self.required_parameter_names = frozenset(_required_parameter_ids(model).values())
        self.completed_transactions = 0
        self._loaded_parameter_names: set[str] = set()
        self._pending_fc1: dict[int, dict[str, torch.Tensor]] = {}
        self._started = False
        self._complete = False
        self._consumed = False

    def _reset_for_next_transaction(self) -> None:
        self._loaded_parameter_names.clear()
        self._pending_fc1.clear()
        self._started = False
        self._complete = False
        self._consumed = False

    def load(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load one IPC/checkpoint chunk and complete only after every mandatory parameter arrives."""
        if self._complete and self._consumed:
            self._reset_for_next_transaction()

        pending_before = _pending_fc1_signature(self._pending_fc1)
        loaded = load_evo2_weights(
            self.model,
            weights,
            strict=False,
            pending_fc1=self._pending_fc1,
        )
        has_activity = bool(loaded) or _pending_fc1_signature(self._pending_fc1) != pending_before
        if not has_activity and not self._started:
            return loaded

        self._started = True
        self._loaded_parameter_names.update(loaded)
        if not self._complete and not self._pending_fc1 and self.required_parameter_names <= self._loaded_parameter_names:
            self._complete = True
            self.completed_transactions += 1
        return loaded

    def assert_ready_for_inference(self) -> None:
        """Reject inference after a partial load while permitting untouched dummy initialization."""
        if not self._started:
            return
        if not self._complete:
            phase = "initial" if self.completed_transactions == 0 else "refit"
            missing = sorted(self.required_parameter_names - self._loaded_parameter_names)
            if self._pending_fc1:
                missing.append("incomplete Vortex MLP fusion")
            raise RuntimeError(f"Evo2 {phase} weight load is incomplete; missing mandatory weights: {missing}")
        self._consumed = True
