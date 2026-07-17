# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Exact same-name MCore-to-vLLM weight bridge for Evo2 policy refits."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    ColumnParallelMapping,
    ReplicatedMapping,
    RowParallelMapping,
)

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.refit import IndexedSafetensorsLayout, indexed_safetensors_layout


_COLUMN_PARALLEL_PATTERNS = (
    "embedding.word_embeddings.weight",
    "decoder.layers.*.mixer.dense_projection.weight",
    "decoder.layers.*.mixer.hyena_proj_conv.short_conv_weight",
    "decoder.layers.*.mixer.mixer.conv_bias",
    "decoder.layers.*.mixer.mixer.filter.R",
    "decoder.layers.*.mixer.mixer.filter.decay",
    "decoder.layers.*.mixer.mixer.filter.gamma",
    "decoder.layers.*.mixer.mixer.filter.h",
    "decoder.layers.*.mixer.mixer.filter.p",
    "decoder.layers.*.mixer.mixer.short_conv.short_conv_weight",
    "decoder.layers.*.mlp.linear_fc1.weight",
    "decoder.layers.*.self_attention.linear_qkv.weight",
)

_ROW_PARALLEL_PATTERNS = (
    "decoder.layers.*.mixer.dense.weight",
    "decoder.layers.*.mlp.linear_fc2.weight",
    "decoder.layers.*.self_attention.linear_proj.weight",
)

_REPLICATED_PATTERNS = (
    "decoder.final_norm.weight",
    "decoder.layers.*.mixer.dense.bias",
    "decoder.layers.*.mixer.dense_projection.layer_norm_weight",
    "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
    "decoder.layers.*.self_attention.linear_proj.bias",
    "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
)


class Evo2RefitBridge(MegatronModelBridge):
    """Gather live Evo2 MCore tensors into the exact indexed vLLM names."""

    def __init__(
        self,
        *,
        config: Evo2Config,
        layout: IndexedSafetensorsLayout,
        transformer_config: Any,
    ) -> None:
        self._config = config
        self._layout = layout
        self.transformer_config = transformer_config

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str | Path,
        *,
        transformer_config: Any,
    ) -> Evo2RefitBridge:
        """Load only config and indexed metadata; no HF model is constructed."""
        root = Path(checkpoint).expanduser().resolve()
        config = Evo2Config.from_pretrained(root, local_files_only=True)
        layout = indexed_safetensors_layout(root)
        bridge = cls(
            config=config,
            layout=layout,
            transformer_config=transformer_config,
        )
        registry = bridge.mapping_registry()
        unsupported = sorted(
            name
            for name in bridge.expected_weight_names
            if registry.megatron_to_hf_lookup(name) is None
        )
        if unsupported:
            raise ValueError(f"unsupported Evo2 refit weights in indexed checkpoint: {unsupported}")
        return bridge

    @property
    def expected_weight_names(self) -> frozenset[str]:
        """Return the immutable generation-checkpoint tensor inventory."""
        return frozenset(spec.name for spec in self._layout.tensors)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Map identical names while gathering each tensor on its MCore TP axis."""
        mappings = []
        for pattern in _COLUMN_PARALLEL_PATTERNS:
            mappings.append(ColumnParallelMapping(megatron_param=pattern, hf_param=pattern))
        for pattern in _ROW_PARALLEL_PATTERNS:
            mappings.append(RowParallelMapping(megatron_param=pattern, hf_param=pattern))
        for pattern in _REPLICATED_PATTERNS:
            mappings.append(ReplicatedMapping(megatron_param=pattern, hf_param=pattern))
        return MegatronMappingRegistry(*mappings)

    def get_conversion_tasks(self, megatron_model, hf_path=None):
        """Build tasks and require an exact live-model/indexed-checkpoint join."""
        if hf_path is not None and Path(hf_path).expanduser().resolve() != self._layout.root:
            raise ValueError("Evo2 refit task path differs from the admitted vLLM checkpoint")
        models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
        raw_tasks = super().build_conversion_tasks(self._config, models)
        tasks = [task for task in raw_tasks if task is not None]
        actual_names = []
        for task in tasks:
            hf_param = task.mapping.hf_param
            if type(hf_param) is not str:
                raise RuntimeError(f"Evo2 refit task has a non-scalar output mapping: {hf_param!r}")
            actual_names.append(hf_param)
        self._require_exact_inventory(actual_names, label="conversion task")
        return tasks

    def _stream_converted_weights(
        self,
        model,
        *,
        cpu: bool,
        show_progress: bool,
        conversion_tasks,
        merge_adapter_weights: bool,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return super().stream_weights_megatron_to_hf(
            model,
            self._config,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
        )

    def export_hf_weights(
        self,
        model,
        cpu: bool = False,
        show_progress: bool = True,
        conversion_tasks=None,
        merge_adapter_weights: bool = True,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Stream every indexed tensor exactly once under its vLLM name."""
        if conversion_tasks is None:
            conversion_tasks = self.get_conversion_tasks(model)
        seen = []
        for name, tensor in self._stream_converted_weights(
            model,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=merge_adapter_weights,
        ):
            if name in seen:
                raise RuntimeError(f"duplicate Evo2 refit weight emitted: {name}")
            if name not in self.expected_weight_names:
                raise RuntimeError(f"foreign Evo2 refit weight emitted: {name}")
            seen.append(name)
            yield name, tensor
        self._require_exact_inventory(seen, label="streamed weight")

    def _require_exact_inventory(self, names: list[str], *, label: str) -> None:
        if len(names) != len(set(names)):
            raise RuntimeError(f"duplicate Evo2 {label} names")
        actual = frozenset(names)
        expected = self.expected_weight_names
        if actual != expected:
            raise RuntimeError(
                f"Evo2 {label} inventory differs from indexed checkpoint: "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )


__all__ = ["Evo2RefitBridge"]
