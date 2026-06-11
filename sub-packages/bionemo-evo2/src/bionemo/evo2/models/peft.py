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

from copy import deepcopy
from typing import List, Optional

import lightning.pytorch as pl
from nemo.collections.llm.fn.mixin import FNMixin
from nemo.collections.llm.peft.lora import LoRA
from nemo.utils import logging
from torch import nn


class Evo2LoRA(LoRA):
    """LoRA adapter specifically for Evo2/Hyena models."""

    def __init__(
        self,
        peft_ckpt_path: Optional[str] = None,
        freeze_modules: List[str] = ["encoder", "embedding"],
        target_modules: List[str] = [
            "linear_qkv",
            "linear_proj",
            "linear_fc1",
            "linear_fc2",
            "short_filter",  # Short convolution filters
            "hyena_filter",  # Hyena layer filters
            "positional_encoding",  # ROPE or other position encodings
        ],
        *args,
        **kwargs,
    ):
        """Initialize the LoRA Adapter for Evo2.

        Args:
            peft_ckpt_path: Path to pre-trained LoRA checkpoint.
            freeze_modules: List of module names to freeze (Evo2-specific defaults).
            target_modules: Modules to apply LoRA to (uses Evo2 defaults if None).
            *args: placeholder.
            **kwargs:
                dim: LoRA rank dimension.
                alpha: LoRA scaling parameter.
                dropout: Dropout rate for LoRA layers.
                dropout_position: Where to apply dropout ('pre' or 'post').
                lora_A_init_method: Initialization for A matrix ('xavier', 'uniform', 'normal').
                lora_B_init_method: Initialization for B matrix ('zero', 'normal').
        """
        """Initialize the LoRA Adapter for Evo2."""
        super().__init__(target_modules=target_modules, *args, **kwargs)
        self.freeze_modules = freeze_modules
        self.peft_ckpt_path = peft_ckpt_path

        # CRITICAL: Set model_transform to self
        # The callback system expects this attribute
        self.model_transform = self

    def setup(self, trainer, pl_module, stage):
        """Setup callback - properly initialize transform."""
        super().setup(trainer, pl_module, stage)

        logging.info(f"Will attempt to apply to model if matches: \n{self.target_modules}")

        # Ensure model_transform is set
        if not hasattr(self, "model_transform") or self.model_transform is None:
            self.model_transform = self

        # Pass checkpoint path to wrapped IO if available
        if hasattr(self, "wrapped_io") and self.peft_ckpt_path:
            self.wrapped_io.adapter_ckpt_path = self.peft_ckpt_path

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Event hook.

        Apply transformations for prediction if needed.

        Args:
            trainer: The trainer object.
            pl_module: The LightningModule object.
        """
        self._maybe_apply_transform(trainer)

    def adapter_key_filter(self, key: str) -> bool:
        """Filter state dict keys to identify adapter parameters.

        Args:
            key: State dict key to check

        Returns:
            bool: True if key corresponds to an adapter parameter
        """
        if isinstance(key, tuple):
            return key[1].requires_grad

        if "_extra_state" in key:
            return False

        # Check if it's an adapter parameter or not in freeze list
        return (
            (not any(substring in key for substring in self.freeze_modules))
            or ".adapter." in key
            or key.endswith(".adapters")
            or "lora_A" in key
            or "lora_B" in key
        )

    def __call__(self, model: nn.Module) -> nn.Module:
        """Apply LoRA transformations to the model.

        Override to avoid fn.walk compatibility issues.
        """
        # First, manually freeze specified modules
        self._apply_selective_freeze(model)

        # Then apply LoRA transformations
        self._apply_lora_transform(model)

        # THEN freeze ALL base model parameters
        # This must happen AFTER LoRA is applied
        self._freeze_base_model_parameters(model)

        # Log summary
        self._log_lora_summary(model)

        return model

    def _apply_selective_freeze(self, model: nn.Module, prefix=""):
        """Manually walk model and freeze specified modules."""
        for name, child in model.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Check if this module should be frozen
            if name in self.freeze_modules:
                logging.info(f"Freezing module: {full_name}")
                for param in child.parameters():
                    param.requires_grad = False

            # Recursively apply to children
            self._apply_selective_freeze(child, full_name)

    def _freeze_base_model_parameters(self, model: nn.Module):
        """Freeze all parameters except LoRA adapters and critical layers."""
        logging.info("\nFreezing base model parameters...")
        frozen_count = 0
        kept_trainable = []

        for name, param in model.named_parameters():
            # Keep LoRA/adapter parameters trainable
            if any(adapter_term in name for adapter_term in ["adapter", "lora_A", "lora_B", "lora"]):
                param.requires_grad = True
                kept_trainable.append(name)
            # CRITICAL: Keep output layer trainable to maintain gradient flow
            elif "output_layer" in name or "lm_head" in name:
                param.requires_grad = True
                kept_trainable.append(name)
                logging.info(f"  Keeping output layer trainable: {name}")
            # CRITICAL: Keep final layer norm trainable
            elif "final_norm" in name or ("decoder" in name and "norm" in name and "24" in name):
                param.requires_grad = True
                kept_trainable.append(name)
                logging.info(f"  Keeping final norm trainable: {name}")
            else:
                param.requires_grad = False
                frozen_count += 1

        logging.info(f"Froze {frozen_count} parameter tensors")
        logging.info(f"Kept {len(kept_trainable)} parameters trainable")

    def _apply_lora_transform(self, model: nn.Module, prefix=""):
        """Apply LoRA with better tracking."""
        # Get all modules in a flat list first
        modules_to_transform = []

        for name, module in model.named_modules():
            # Skip if has children (not a leaf module)
            if list(module.children()):
                continue

            # Check if this matches our target modules
            module_type = name.split(".")[-1] if "." in name else name
            if module_type in self.target_modules:
                modules_to_transform.append((name, module))

        logging.info(f"\nFound {len(modules_to_transform)} modules to apply LoRA to")

        # Apply transformations
        for full_name, module in modules_to_transform:
            # Get parent and attribute name
            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            # Apply transform
            attr_name = parts[-1]
            transformed = self.transform(module, name=attr_name, prefix="")

            if transformed is not module:
                setattr(parent, attr_name, transformed)
                logging.info(f"Applied LoRA to: {full_name}")

                # Verify LoRA was applied
                if hasattr(transformed, "adapter") or hasattr(transformed, "lora_A"):
                    logging.info(f"  ✓ LoRA adapter confirmed on {full_name}")

    def selective_freeze(self, m: nn.Module, name=None, prefix=None):
        """Selectively freeze modules based on freeze_modules list.

        Args:
            m: Module to potentially freeze.
            name: Name of the module.
            prefix: Prefix for the module name.

        Returns:
            nn.Module: The module (frozen or not).
        """
        if name in self.freeze_modules:
            FNMixin.freeze(m)
            logging.info(f"Freezing module: {prefix}.{name}" if prefix else f"Freezing module: {name}")

        return m

    # Deepcopy compatibility
    def __deepcopy__(self, memo):
        """Custom deepcopy to handle unpickleable objects."""
        # Create a new instance with the same parameters
        cls = self.__class__
        result = cls.__new__(cls)

        # Copy all attributes except problematic ones
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k not in ["_metadata", "_fields"]:  # Skip dataclass internals
                try:
                    setattr(result, k, deepcopy(v, memo))
                except Exception:
                    # If deepcopy fails, just use the original reference
                    setattr(result, k, v)

        return result

    def __getstate__(self):
        """Prepare object for pickling."""
        state = self.__dict__.copy()
        # Remove unpickleable entries
        state.pop("_metadata", None)
        state.pop("_fields", None)
        return state

    def __setstate__(self, state):
        """Restore object from pickle."""
        self.__dict__.update(state)

    # Debug module
    def _log_lora_summary(self, model: nn.Module):
        """Log a summary of LoRA modifications."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        adapter_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "adapter" in n)

        logging.info(f"\n{'=' * 50}")
        logging.info("LoRA Summary:")
        logging.info(f"  Total parameters: {total_params:,}")
        logging.info(f"  Trainable parameters: {trainable_params:,}")
        logging.info(f"  Adapter parameters: {adapter_params:,}")  # Changed from "LoRA parameters"
        logging.info(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        logging.info(f"  Percentage adapters: {100 * adapter_params / total_params:.2f}%")
        logging.info(f"{'=' * 50}\n")
