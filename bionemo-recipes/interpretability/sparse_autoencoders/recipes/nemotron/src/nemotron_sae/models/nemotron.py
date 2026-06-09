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

"""Nemotron-3-Nano model wrapper for activation extraction.

Provides a simple interface for extracting activations from any layer
of the Nemotron-3-Nano hybrid (Mamba-2 / MoE / GQA) model.

Requires trust_remote_code=True and bf16 precision.
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class NemotronModel:
    """Wrapper for Nemotron-3-Nano to extract features.

    Loads the full CausalLM model (not just base) so it can be reused
    for loss_recovered evaluation without loading a second copy.

    Args:
        model_name: HuggingFace model name
        layer: Layer index to extract features from. If None, uses 3/4 depth.
        device: Device hint (ignored when device_map="auto" is used)
        max_length: Maximum context length

    Example:
        >>> model = NemotronModel(layer=39)
        >>> texts = ["Hello world!", "The quick brown fox"]
        >>> activations, mask = model.generate_activations(texts)
        >>> print(activations.shape)  # [2, max_seq_len, 2688]
    """

    DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        layer: Optional[int] = None,
        device: str = "cpu",
        max_length: int = 2048,
        device_map="auto",
    ):
        """Load Nemotron for activation extraction.

        Args:
            model_name: HuggingFace model name or local path.
            layer: Layer index to extract (None -> 3/4 depth).
            device: Device hint (used for tokenizer output placement).
            max_length: Max context length.
            device_map: HF device_map. "auto" shards across all visible GPUs.
                For data-parallel extraction, pin a replica to one GPU by passing
                an int or {"": idx} (e.g. device_map={"": 3} for cuda:3). The 30B
                model (~59 GB bf16) fits on a single 80 GB GPU.
        """
        self.device = device
        self.max_length = max_length
        self.model_name = model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model in bf16. device_map="auto" shards across GPUs; an int/{"":idx}
        # pins the whole model to one GPU (for parallel data-parallel extraction).
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        ).eval()

        # Model info
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size

        # Set extraction layer (default to 3/4 depth)
        if layer is None:
            self.layer = int(self.num_layers * 3 / 4)
        else:
            if layer < 0 or layer >= self.num_layers:
                raise ValueError(f"Layer {layer} out of range [0, {self.num_layers - 1}]")
            self.layer = layer

        print(
            f"Loaded {model_name} - Hidden size: {self.hidden_size}, "
            f"Layers: {self.num_layers}, Extracting layer: {self.layer}"
        )

    def forward_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from Nemotron model.

        Args:
            input_ids: Tokenized sequence ids [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Tuple of (hidden_states, attention_mask)
            hidden_states shape: [batch, seq_len, hidden_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # hidden_states[0] = embeddings, hidden_states[i+1] = output of block i.
            # Use self.layer + 1 so extraction matches the loss_recovered eval,
            # which reads hidden_states[layer_idx + 1] and hooks model.layers[layer_idx].
            return outputs.hidden_states[self.layer + 1], attention_mask

    def generate_activations(
        self,
        texts: List[str],
        batch_size: int = 4,
        return_tensors: str = "pt",
        show_progress: bool = True,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]]:
        """Extract activations from a list of text strings.

        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once (keep small for 30B model)
            return_tensors: 'pt' for torch tensors, 'np' for numpy arrays
            show_progress: Whether to show progress bar

        Returns:
            Tuple of (activations, masks) where:
            - activations: [n_texts, max_seq_len, hidden_dim]
            - masks: [n_texts, max_seq_len] indicating valid positions
        """
        all_embeddings = []
        all_masks = []

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting activations")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            encoding = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.model.device)
            attention_mask = encoding["attention_mask"].to(self.model.device)

            # Extract features
            embeddings, mask = self.forward_features(input_ids, attention_mask)

            # Always collect as float32 on CPU for downstream SAE training
            all_embeddings.append(embeddings.float().cpu())
            all_masks.append(mask.cpu())

        result_embeddings = torch.cat(all_embeddings, dim=0)
        result_masks = torch.cat(all_masks, dim=0)

        if return_tensors == "np":
            return result_embeddings.numpy(), result_masks.numpy()
        return result_embeddings, result_masks

    def stream_activations(
        self,
        texts: List[str],
        batch_size: int = 4,
        show_progress: bool = False,
    ):
        """Yield per-batch flattened activations for producer-consumer streaming.

        Tokenizes texts one batch at a time, runs the model, and yields only the
        valid (non-pad) token activations from the target layer as CPU float32
        tensors of shape [n_valid_tokens, hidden_size]. Pairs with
        sae.streaming.make_streaming_dataloader to train an SAE on the fly
        without persisting activations to disk.

        Args:
            texts: List of text strings.
            batch_size: Number of texts per forward pass (keep small for 30B).
            show_progress: Whether to show a tqdm progress bar.

        Yields:
            torch.Tensor of shape [n_valid_tokens, hidden_size] (CPU, float32).
        """
        try:
            in_device = self.model.get_input_embeddings().weight.device
        except Exception:
            in_device = next(self.model.parameters()).device

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Streaming activations")

        for i in iterator:
            batch_texts = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(in_device)
            attention_mask = enc["attention_mask"].to(in_device)
            hidden, mask = self.forward_features(input_ids, attention_mask)
            flat = hidden.float().cpu()[mask.bool().cpu()]
            if flat.shape[0] > 0:
                yield flat

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text and return token strings (for visualization)."""
        tokens = self.tokenizer.encode(text)
        return [self.tokenizer.decode([t]) for t in tokens]

    def __repr__(self) -> str:
        """Readable summary of the wrapper (model, layer, dims)."""
        return (
            f"NemotronModel(model={self.model_name}, layer={self.layer}, "
            f"hidden_size={self.hidden_size}, num_layers={self.num_layers})"
        )
