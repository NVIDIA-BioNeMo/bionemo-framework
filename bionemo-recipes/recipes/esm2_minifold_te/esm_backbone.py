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

"""ESM-2 backbone wrapper using HuggingFace transformers.

Loads a pretrained ESM-2 model and extracts:
- Per-residue embeddings from the last hidden layer
- Pairwise attention maps from all transformer layers

The backbone is intended to be frozen during folding head training.
"""

import logging
import os

import torch
import torch.nn as nn
from transformers import EsmConfig, EsmModel, EsmTokenizer


logger = logging.getLogger(__name__)
_TOKENIZER_FALLBACK = "facebook/esm2_t33_650M_UR50D"


def _load_tokenizer(model_name: str) -> EsmTokenizer:
    try:
        return EsmTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        try:
            return EsmTokenizer.from_pretrained(model_name)
        except OSError:
            logger.warning("Falling back to cached ESM-2 tokenizer %s for %s", _TOKENIZER_FALLBACK, model_name)
            return EsmTokenizer.from_pretrained(_TOKENIZER_FALLBACK, local_files_only=True)


def _load_model(model_name: str) -> EsmModel:
    try:
        return EsmModel.from_pretrained(model_name, attn_implementation="eager", local_files_only=True)
    except OSError as local_error:
        if os.environ.get("BIONEMO_ALLOW_CONFIG_ONLY_ESM2_INIT") == "1":
            logger.warning("Initializing %s from config only: %s", model_name, local_error)
            config = EsmConfig.from_pretrained(model_name)
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
            return EsmModel(config)
        return EsmModel.from_pretrained(model_name, attn_implementation="eager")


def load_esm2_backbone(model_name: str = "facebook/esm2_t33_650M_UR50D", device: str = "cuda"):
    """Load a pretrained ESM-2 model from HuggingFace.

    Args:
        model_name: HuggingFace model name. Common options:
            - "facebook/esm2_t6_8M_UR50D" (8M params, 6 layers, 20 heads)
            - "facebook/esm2_t12_35M_UR50D" (35M params, 12 layers, 20 heads)
            - "facebook/esm2_t30_150M_UR50D" (150M params, 30 layers, 20 heads)
            - "facebook/esm2_t33_650M_UR50D" (650M params, 33 layers, 20 heads)
            - "facebook/esm2_t36_3B_UR50D" (3B params, 36 layers, 40 heads)
            - "facebook/esm2_t48_15B_UR50D" (15B params, 48 layers, 40 heads)
        device: Device to load the model onto.

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = _load_tokenizer(model_name)
    model = _load_model(model_name).to(device)
    return model, tokenizer


class ESM2Backbone(nn.Module):
    """Frozen ESM-2 backbone that extracts embeddings and attention maps.

    This wraps a HuggingFace EsmModel and provides a forward pass that returns
    both per-residue embeddings and pairwise attention maps in the format
    expected by MiniFold's folding head.
    """

    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        super().__init__()
        self.model = _load_model(model_name)
        config = self.model.config

        self.embed_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.attn_dim = self.num_layers * self.num_heads

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        """Extract embeddings and attention maps from ESM-2.

        Args:
            input_ids: Token IDs (B, L). Use ESM-2 tokenizer to encode sequences.
            attention_mask: Optional attention mask (B, L). 1 = valid, 0 = padding.

        Returns:
            Dict with:
                "representations": Per-residue embeddings (B, L, embed_dim).
                "attentions": Pairwise attention maps (B, L, L, num_layers * num_heads).
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        # Per-residue embeddings from last hidden state
        representations = outputs.last_hidden_state  # (B, L, embed_dim)

        # Stack attention maps from all layers
        # Each layer returns (B, num_heads, L, L)
        # Stack to (num_layers, B, num_heads, L, L) then rearrange to (B, L, L, num_layers * num_heads)
        attn_stack = torch.stack(outputs.attentions, dim=0)  # (num_layers, B, H, L, L)
        B, H, L = attn_stack.shape[1], attn_stack.shape[2], attn_stack.shape[3]
        # Rearrange: (num_layers, B, H, L, L) -> (B, L, L, num_layers, H) -> (B, L, L, num_layers * H)
        attn_stack = attn_stack.permute(1, 3, 4, 0, 2)  # (B, L, L, num_layers, H)
        attentions = attn_stack.reshape(B, L, L, -1)  # (B, L, L, num_layers * num_heads)

        return {
            "representations": representations,
            "attentions": attentions,
        }


# ESM-2 model specs for reference
ESM2_MODELS = {
    "facebook/esm2_t6_8M_UR50D": {"layers": 6, "embed_dim": 320, "heads": 20},
    "facebook/esm2_t12_35M_UR50D": {"layers": 12, "embed_dim": 480, "heads": 20},
    "facebook/esm2_t30_150M_UR50D": {"layers": 30, "embed_dim": 640, "heads": 20},
    "facebook/esm2_t33_650M_UR50D": {"layers": 33, "embed_dim": 1280, "heads": 20},
    "facebook/esm2_t36_3B_UR50D": {"layers": 36, "embed_dim": 2560, "heads": 40},
    "facebook/esm2_t48_15B_UR50D": {"layers": 48, "embed_dim": 5120, "heads": 40},
}
