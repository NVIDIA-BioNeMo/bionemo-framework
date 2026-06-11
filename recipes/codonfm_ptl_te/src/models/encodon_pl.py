# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint
from lightning import LightningModule
from peft import LoraConfig, PeftType, get_peft_model
from torchmetrics import MeanMetric
from transformers.utils import logging


logger = logging.get_logger(__name__)


from src.data.metadata import MetadataFields  # noqa: E402
from src.models.components.cross_attention import CrossAttention  # noqa: E402
from src.models.components.encodon import EnCodon  # noqa: E402
from src.models.components.encodon_config import EnCodonConfig  # noqa: E402
from src.models.utils import construct_pretrained_config, get_decay_parameter_names  # noqa: E402


class EncodonPL(LightningModule):
    """LightningModule wrapper around the EnCodon encoder and optional heads.

    Supports language modeling and an optional downstream cross-attention head
    for classification or regression. Optimizer and scheduler are provided via
    callables to enable configuration-time injection.

    Args:
        optimizer: Callable that constructs an optimizer when provided parameter groups.
        scheduler: Callable that constructs a scheduler when provided an optimizer.
        vocab_size: Vocabulary size.
        hidden_size: Transformer hidden size.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        intermediate_size: Feed-forward network hidden size.
        hidden_act: Activation function name or callable.
        hidden_dropout_prob: Dropout probability for hidden states.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        initializer_range: Weight initialization scale.
        layer_norm_eps: Epsilon for LayerNorm.
        pad_token_id: Padding token id used by the tokenizer.
        position_embedding_type: Position embedding variant (e.g., "rotary").
        classifier_dropout: Dropout for classification/regression heads.
        rotary_theta: Rotary embedding theta value.
        ignore_index: Label ignore index for language modeling.
        loss_type: "regression" or "classification" when using downstream head.
        lora: Whether to enable LoRA PEFT.
        lora_alpha: LoRA alpha parameter.
        lora_r: LoRA rank.
        lora_dropout: LoRA dropout.
        finetune_strategy: One of {"full", "head_only_pretrained", "head_only_random", "lora"}.
        num_classes: Number of classes for classification with downstream head.
        use_downstream_head: Whether to enable downstream cross-attention head.
        cross_attention_hidden_dim: Hidden size for downstream head.
        cross_attention_num_heads: Number of attention heads for downstream head.
        max_position_embeddings: Maximum supported sequence length.
        kwargs: Additional keyword arguments.
    """

    def __init__(  # noqa: D107
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        vocab_size: int = 69,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1.0e-12,
        pad_token_id: int = 3,  # - 3 is the padding token id in the tokenizer
        position_embedding_type: str = "rotary",
        classifier_dropout: float = 0.1,
        rotary_theta: float = 1.0e4,
        ignore_index: int = -100,
        loss_type: str = "regression",  # - regression, classification
        lora: bool = False,
        lora_alpha: float = 32.0,
        lora_r: int = 16,
        lora_dropout: float = 0.1,
        finetune_strategy: str = "full",
        num_classes: int = 2,  # For classification tasks
        use_downstream_head: bool = False,  # Whether to use downstream cross-attention head
        cross_attention_hidden_dim: int = 256,  # Hidden dimension for cross-attention classifier
        cross_attention_num_heads: int = 8,  # Number of attention heads for cross-attention
        max_position_embeddings: int = 2048,  # Maximum position embeddings
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.pretrained_config = construct_pretrained_config(self.hparams, EnCodonConfig)
        self.model = None
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Best validation loss tracking (simplified for basic loss monitoring)
        self.best_val_loss = float("inf")

        # Determine task type for metric calculation
        if use_downstream_head:
            if loss_type == "classification":
                self.task_type = "classification"
            else:
                self.task_type = "regression"
        else:
            # When downstream head is disabled, we use standard language modeling
            self.task_type = "language_modeling"

        if use_downstream_head:
            if loss_type == "classification":
                self.loss = torch.nn.CrossEntropyLoss()
            elif loss_type == "regression":
                self.loss = torch.nn.MSELoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}. Must be 'regression' or 'classification'.")
        else:
            # When downstream head is disabled, use cross-entropy for language modeling
            self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    def configure_model(self, state_dict: Optional[Dict[str, Any]] = None) -> None:
        """Configure the underlying model and optionally load weights.

        Sets up the base EnCodon model, attaches optional downstream heads, and
        applies PEFT (LoRA) when enabled. Loads weights from ``state_dict`` if
        provided.

        Args:
            state_dict: A state dictionary to load into the model.
        """
        if self.model is not None:
            # If model already exists and no state_dict provided, nothing to do
            if state_dict is None:
                return

        if self.model is None:
            self.model = EnCodon(self.pretrained_config)

            # Add cross-attention downstream heads only if enabled
            if self.hparams.use_downstream_head:
                # Cross-attention input projection
                self.model.cross_attention_input_proj = nn.Linear(
                    self.pretrained_config.hidden_size, self.hparams.cross_attention_hidden_dim
                )

                if self.hparams.loss_type == "classification":
                    # Cross-attention classification head
                    self.model.cross_attention_head = CrossAttention(
                        hidden_dim=self.hparams.cross_attention_hidden_dim,
                        n_out=self.hparams.num_classes,  # Output for classification
                        num_heads=self.hparams.cross_attention_num_heads,
                        dropout=self.hparams.classifier_dropout,
                    )
                else:  # regression
                    # Cross-attention regression head
                    self.model.cross_attention_head = CrossAttention(
                        hidden_dim=self.hparams.cross_attention_hidden_dim,
                        n_out=1,  # Single output for regression
                        num_heads=self.hparams.cross_attention_num_heads,
                        dropout=self.hparams.classifier_dropout,
                    )
                self.init_downstream_heads()
        has_peft_params = any("lora" in k for k in state_dict.keys()) if state_dict else False

        # Load base model weights if not loading a PEFT checkpoint
        if state_dict and not has_peft_params:
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[len("model.") :]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict, strict=False)

        if self.hparams.finetune_strategy in ["head_only_pretrained", "head_only_random"]:
            logger.info("Freezing backbone for head-only finetuning.")
            # The backbone consists of embeddings and encoder layers
            backbone = [self.model.embeddings, self.model.layers]
            for module in backbone:
                for param in module.parameters():
                    param.requires_grad = False

        if self.hparams.finetune_strategy == "head_only_random":
            logger.info("Randomly initializing lm_head.")
            self.model.reset_cls_parameters()
            self.init_downstream_heads()

        # Configure PEFT if LoRA is enabled or if loading a LoRA checkpoint
        if self.hparams.lora or has_peft_params:
            peft_config = LoraConfig(
                peft_type=PeftType.LORA,
                task_type="SEQ_CLS",
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=["query", "value", "intermediate_dense", "output_dense"],
                inference_mode=False,
            )
            self.model = get_peft_model(self.model, peft_config)

            #  Ensure classifier head `model.cls` is trainable after applying LoRA
            cls_trainable = any(param.requires_grad for param in self.model.cls.parameters())
            if not cls_trainable:
                for param in self.model.cls.parameters():
                    param.requires_grad = True

            # Ensure cross-attention heads are trainable after applying LoRA if they exist and downstream head is enabled
            if self.hparams.use_downstream_head:
                cross_attention_heads = ["cross_attention_input_proj", "cross_attention_head"]
                for head_name in cross_attention_heads:
                    if hasattr(self.model, head_name):
                        head = getattr(self.model, head_name)
                        head_trainable = any(param.requires_grad for param in head.parameters())
                        if not head_trainable:
                            for param in head.parameters():
                                param.requires_grad = True

            # Log LoRA parameter counts
            if self.hparams.lora and not has_peft_params:
                self.model.print_trainable_parameters()

        # Load the full state dict (including PEFT weights) if provided
        if state_dict and has_peft_params:
            self.load_state_dict(state_dict, strict=True)

    def init_downstream_heads(self):
        """Initialize downstream head parameters when enabled.

        Randomly initializes cross-attention input projection and prediction
        layers to ensure trainable heads after configuration or re-initialization.
        """
        if self.hparams.use_downstream_head:
            cross_attention_heads = ["cross_attention_input_proj", "cross_attention_head"]
            for head_name in cross_attention_heads:
                if hasattr(self.model, head_name):
                    logger.info(f"Randomly initializing {head_name} head.")
                    head = getattr(self.model, head_name)
                    for module in head.modules():
                        if isinstance(module, nn.Linear):
                            gain = self.pretrained_config.initializer_range * math.sqrt(
                                math.log(2 * self.pretrained_config.num_hidden_layers)
                            )
                            nn.init.xavier_normal_(module.weight, gain=gain)
                            if module.bias is not None:
                                module.bias.data.zero_()
                        elif isinstance(module, nn.LayerNorm):
                            module.bias.data.zero_()
                            module.weight.data.fill_(1.0)

    def forward(self, batch_data, return_hidden_states: bool = False):
        """Run a forward pass through the backbone model.

        Args:
            batch_data: Mapping containing `INPUT_IDS` and `ATTENTION_MASK`.
            return_hidden_states: If True, returns all hidden states.

        Returns:
            Model output with logits, last hidden state, and optionally all hidden states.
        """
        input_ids = batch_data[MetadataFields.INPUT_IDS]
        attention_mask = batch_data[MetadataFields.ATTENTION_MASK]
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_hidden_states=return_hidden_states,
        )
        return output

    def model_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss and predictions for a single training/validation step.

        Handles both language modeling and downstream head modes.

        Args:
            batch: Batch containing inputs and labels.

        Returns:
            A tuple of (loss, predictions, targets).
        """
        y = batch[MetadataFields.LABELS]

        if self.hparams.use_downstream_head:
            # Forward pass to get hidden states
            output = self.forward(batch)
            hidden_states = output.last_hidden_state  # [batch_size, seq_len, hidden_size]
            attention_mask = batch[MetadataFields.ATTENTION_MASK]  # [batch_size, seq_len]

            # Project hidden states to cross-attention dimension
            projected_states = self.model.cross_attention_input_proj(
                hidden_states
            )  # [batch_size, seq_len, cross_attn_hidden_dim]

            # Extract [CLS] token as query and use full sequence as key/value
            query_input = projected_states[:, 0, :]  # [batch_size, cross_attn_hidden_dim]
            key_value_input = projected_states  # [batch_size, seq_len, cross_attn_hidden_dim]

            # Pass through cross-attention head
            preds = self.model.cross_attention_head(query_input, key_value_input, attention_mask)

            if self.hparams.loss_type == "classification":
                # Classification: preds shape [batch_size, num_classes]
                y = y.view(-1).long()  # Ensure labels are long for classification
            else:
                # Regression: preds shape [batch_size, 1] -> squeeze to [batch_size]
                preds = preds.squeeze(-1)
                y = y.view(-1).float()
        else:
            # Use standard model output (logits from language modeling head)
            output = self.forward(batch)
            preds = output.logits
            # Check if INPUT_MASK exists, otherwise create from attention mask
            if MetadataFields.INPUT_MASK in batch:
                mask = batch[MetadataFields.INPUT_MASK]
            else:
                # Create mask from attention mask (True for valid tokens)
                mask = batch[MetadataFields.ATTENTION_MASK].bool()
            # Standard cross-entropy loss for token-level prediction
            preds = preds.view(-1, self.pretrained_config.vocab_size)
            # Mask out tokens that are not part of the loss calculation
            y = torch.where(mask.view(-1), y.view(-1), self.hparams.ignore_index)
            y = y.view(-1)

        # Compute loss
        loss = self.loss(preds, y)

        if preds.dtype in [torch.float16, torch.bfloat16]:
            loss = loss.to(torch.float32)
        return loss, preds, y

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Perform a training step and log training loss.

        Args:
            batch: Training batch.
            batch_idx: Index of the batch in the current epoch.

        Returns:
            Loss tensor used for optimization.
        """
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Configure optimizer and (optionally) learning rate scheduler.

        Returns:
            Dict with keys `optimizer` and optionally `lr_scheduler` for Trainer.
        """
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and (optionally) learning rate scheduler.

        Returns:
            Dict with keys `optimizer` and optionally `lr_scheduler` for Trainer.
        """
        if self.hparams.finetune_strategy in ["head_only_pretrained", "head_only_random"]:
            # For head-only finetuning, determine which head to optimize
            if self.hparams.use_downstream_head:
                # Optimize cross-attention heads
                param_source = nn.ModuleList([self.model.cross_attention_input_proj, self.model.cross_attention_head])
            else:
                # Optimize standard language modeling head
                param_source = self.model.cls
        else:
            param_source = self.trainer.model

        decay_parameters = get_decay_parameter_names(
            param_source,
            none_applicable_layer_types=[nn.LayerNorm, nn.Embedding],
            disallowed_layer_names=["bias", "layernorm", "rmsnorm", "rotary_emb"],
        )
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_source.named_parameters() if (n in decay_parameters and p.requires_grad)
                ],
            },
            {
                "params": [
                    p for n, p in param_source.named_parameters() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = self.hparams.optimizer(params=optimizer_grouped_parameters)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        """Configures gradient clipping for both FSDP and non-FSDP modes.

        Args:
            optimizer: Optimizer instance.
            gradient_clip_val: Maximum allowed norm or value for gradients.
            gradient_clip_algorithm: One of {"norm", "value", None}.

        Raises:
            ValueError: If arguments are invalid or unsupported by the strategy.
        """
        if gradient_clip_val is not None and gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be non-negative")

        if gradient_clip_algorithm not in ("norm", "value", None):
            raise ValueError("gradient_clip_algorithm must be one of: 'norm', 'value', None")

        if self.trainer.strategy.__class__.__name__ == "FSDPStrategy":
            if gradient_clip_algorithm not in ("norm", None):
                raise ValueError("FSDP only supports 'norm' gradient clipping")
            # For FSDP, use torch.nn.utils.clip_grad_norm_ directly
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_val)
        else:
            self.clip_gradients(
                optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm
            )

    def optimizer_step(self, *args, **kwargs):
        """Skip optimizer updates if gradients contain NaNs or infs.
        Checks gradients for invalid values prior to invoking the optimizer.
        """  # noqa: D205
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            logger.warning("Detected inf or nan values in gradients. Skipping model update.")
            self.zero_grad()

        super().optimizer_step(*args, **kwargs)
