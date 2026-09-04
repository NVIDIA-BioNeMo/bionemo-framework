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


import logging
from dataclasses import dataclass, field
from functools import wraps
from typing import Set

import torch
from megatron.bridge.peft.base import ModelType
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.peft.utils import wildcard_match
from megatron.core.utils import unwrap_model
from torch import nn

from bionemo.evo2.models.megatron.hyena.hyena_block import HyenaStack


logger: logging.Logger = logging.getLogger(__name__)

_HYENA_RECOMPUTE_PATCHED: Set[int] = set()


def _enable_recompute_inputs_grad_for_hyena(model, patched_registry: Set[int] | None = None) -> Set[int]:
    """Enable grad on HyenaStack inputs when only adapters are trainable.

    This is the HyenaStack analogue of ``maybe_enable_recompute_inputs_grad`` from
    ``megatron.bridge.peft.recompute``, which only patches ``TransformerBlock``.
    HyenaStack is not a TransformerBlock subclass, so the upstream fix never fires
    for Evo2 models.

    When activation checkpointing is active (``recompute_granularity == "full"``),
    Megatron's ``CheckpointFunction.backward()`` is only invoked by PyTorch autograd
    when at least one *input* tensor to the checkpoint has ``requires_grad=True``.
    With PP=1 and a fully frozen base model the embedding outputs carry
    ``requires_grad=False``, so ``CheckpointFunction.backward()`` is never called
    and LoRA gradients inside the checkpoint are silently dropped.

    The fix: monkey-patch ``HyenaStack.forward`` to force
    ``hidden_states.requires_grad_(True)`` before the tensor enters the checkpointed
    region.  No parameters are unfrozen; only the autograd bookkeeping is corrected.
    """
    registry = patched_registry if patched_registry is not None else _HYENA_RECOMPUTE_PATCHED

    unwrapped = unwrap_model(model)
    if not isinstance(unwrapped, list):
        unwrapped = [unwrapped]

    for unwrapped_model in unwrapped:
        if unwrapped_model is None:
            continue

        cfg = getattr(unwrapped_model, "config", None)
        if cfg is None or getattr(cfg, "recompute_method", None) is None:
            continue

        if id(unwrapped_model) in registry:
            continue

        params = list(unwrapped_model.named_parameters())
        trainable_adapter = any(p.requires_grad and ".adapter." in n.lower() for n, p in params)
        trainable_base = any(
            p.requires_grad and ".to_wrap." not in n.lower() and ".adapter." not in n.lower() for n, p in params
        )

        if not (trainable_adapter and not trainable_base):
            continue

        patched_any = False
        for module in unwrapped_model.modules():
            if isinstance(module, HyenaStack):
                original_forward = module.forward

                @wraps(original_forward)
                def _patched_forward(hidden_states, *args, _orig=original_forward, **kwargs):
                    if (
                        torch.is_tensor(hidden_states)
                        and not hidden_states.requires_grad
                        and hidden_states.is_floating_point()
                    ):
                        hidden_states = hidden_states.detach().requires_grad_(True)
                    return _orig(hidden_states, *args, **kwargs)

                module.forward = _patched_forward
                patched_any = True

        if patched_any:
            registry.add(id(unwrapped_model))
            logger.info(
                "[Evo2LoRA+Recompute] Patched HyenaStack.forward to enable grad on "
                "hidden_states input. This ensures checkpoint backward is called when "
                "only adapters are trainable (PP=1 with frozen base model)."
            )

    return registry


@dataclass
class Evo2LoRA(LoRA):
    """LoRA variant that allows selectively skipping parameter freezing for specified modules.

    Extends LoRA with a ``skip_freeze_modules`` field that follows the same pattern-matching
    semantics as ``target_modules``:

    - Exact short name: ``"mixer"`` matches any module whose immediate name is ``"mixer"``,
      regardless of depth.
    - Wildcard on full path: ``"*.layers.0.*.mixer"`` matches using ``*`` as a substring
      wildcard anchored to the full dotted path.

    Args:
        skip_freeze_modules: List of module name patterns to exclude from freezing.
            Supports the same syntax as ``target_modules``. Modules whose short name or
            full path matches any pattern will remain trainable.
    """

    skip_freeze_modules: list[str] = field(default_factory=list)

    def __call__(self, model: ModelType, training: bool = True) -> ModelType:
        """Apply LoRA to the model, with HyenaStack-aware recompute patching."""
        self._validate_tied_weight_config(model)
        model = super().__call__(model, training=training)
        if training:
            _enable_recompute_inputs_grad_for_hyena(model)
        return model

    def _get_is_tied(self, model: ModelType) -> bool:
        """Return True if the model uses ``share_embeddings_and_output_weights``."""
        unwrapped = unwrap_model(model)
        if not isinstance(unwrapped, list):
            unwrapped = [unwrapped]
        return next(
            (
                m.share_embeddings_and_output_weights
                for m in unwrapped
                if m is not None and hasattr(m, "share_embeddings_and_output_weights")
            ),
            False,
        )

    def _validate_tied_weight_config(self, model: ModelType) -> None:
        """Raise early if ``target_modules`` or ``skip_freeze_modules`` are asymmetric for a weight-tied model.

        When ``share_embeddings_and_output_weights=True``, ``word_embeddings`` and
        ``output_layer`` share the same weight tensor.  Both lists must treat the
        tied pair as a unit: list both layers or neither.  Listing only one side
        raises ``ValueError``.

        The check walks the actual model so that wildcard patterns (e.g.
        ``"embedding.*"``) are evaluated against real module paths rather than
        synthetic names.
        """
        if not self._get_is_tied(model):
            return

        targeted_short_names: set[str] = set()
        skip_frozen_short_names: set[str] = set()

        def _collect(module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
            full_name = f"{prefix}.{name}" if prefix else (name or "")
            short_name = name or ""
            if self._matches_lora_target(short_name, full_name):
                targeted_short_names.add(short_name)
            if self._matches_skip_freeze(short_name, full_name):
                skip_frozen_short_names.add(short_name)
            return module

        self._walk_model(model, _collect)

        target_word_emb = "word_embeddings" in targeted_short_names
        target_output = "output_layer" in targeted_short_names
        if target_word_emb != target_output:
            raise ValueError(
                "share_embeddings_and_output_weights is enabled: target_modules must "
                "include both word_embeddings and output_layer, or neither. "
                f"word_embeddings matched: {target_word_emb}, output_layer matched: {target_output}."
            )

        skip_word_emb = "word_embeddings" in skip_frozen_short_names
        skip_output = "output_layer" in skip_frozen_short_names
        if skip_word_emb != skip_output:
            raise ValueError(
                "share_embeddings_and_output_weights is enabled: skip_freeze_modules must "
                "include both word_embeddings and output_layer, or neither. "
                f"word_embeddings matched: {skip_word_emb}, output_layer matched: {skip_output}."
            )

    def _matches_skip_freeze(self, name: str, full_name: str) -> bool:
        """Return True if a module matches any ``skip_freeze_modules`` pattern."""
        return any(name == p or wildcard_match(p, full_name) for p in self.skip_freeze_modules)

    def _matches_lora_target(self, name: str, full_name: str) -> bool:
        """Return True if a module matches any ``target_modules`` pattern."""
        return any(name == p or wildcard_match(p, full_name) for p in (self.target_modules or []))

    def freeze_model(self, model: ModelType, training: bool = True) -> None:
        """Freeze all model parameters except those matching ``skip_freeze_modules``.

        Raises ``ValueError`` if any module matches both ``skip_freeze_modules``
        and ``target_modules``, because LoRA replaces target modules with adapter
        wrappers which resets ``requires_grad`` on the original weights.

        Args:
            model: The model (or list of model chunks) to freeze.
            training: Whether the model is being used for training. When True, sets
                the model to training mode after freezing.
        """
        matched_patterns: set[str] = set()
        overlap_modules: list[str] = []

        def selective_freeze(module: nn.Module, name: str | None = None, prefix: str | None = None) -> nn.Module:
            full_name = f"{prefix}.{name}" if prefix else (name or "")
            short_name = name or ""
            skip = self._matches_skip_freeze(short_name, full_name)
            if skip and self._matches_lora_target(short_name, full_name):
                overlap_modules.append(full_name)
            if not skip:
                for param in module.parameters(recurse=False):
                    param.requires_grad = False
            else:
                matched = [p for p in self.skip_freeze_modules if short_name == p or wildcard_match(p, full_name)]
                matched_patterns.update(matched)
                logger.info(f"Evo2LoRA: Skipping freezing module: {full_name}.")
            return module

        self._walk_model(model, selective_freeze)

        if overlap_modules:
            raise ValueError(
                f"skip_freeze_modules and target_modules must not overlap. "
                f"LoRA replaces target modules with adapter wrappers, which resets "
                f"requires_grad on the original weights and defeats skip-freeze. "
                f"Overlapping modules: {sorted(overlap_modules)}"
            )

        for p in self.skip_freeze_modules:
            if p not in matched_patterns:
                logger.warning(f"Evo2LoRA: skip_freeze_modules pattern '{p}' did not match any module.")

        if training:
            if isinstance(model, list):
                for model_chunk in model:
                    model_chunk.train(mode=True)
            elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.train(mode=True)
            else:
                model.train(mode=True)
