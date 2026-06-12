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

"""Loss recovered evaluation for SAEs on ESM2.

ESM2-specific wrapper around the general loss_recovered metric.
CE is computed over all non-special tokens (positions 1..length-2) per sequence,
matching InterPLM's convention for masked language models.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sae.eval import LossRecoveredResult, evaluate_loss_recovered


def evaluate_esm2_loss_recovered(
    sae: torch.nn.Module,
    model: torch.nn.Module,
    tokenizer,
    sequences: List[str],
    layer_idx: int,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 42,
    max_length: int = 1024,
) -> LossRecoveredResult:
    """Evaluate SAE loss recovered on protein sequences.

    Args:
        sae: Trained sparse autoencoder.
        model: ESM2 model with LM head (EsmForMaskedLM).
        tokenizer: ESM2 tokenizer.
        sequences: List of protein sequences.
        layer_idx: Which transformer layer to intervene on (0-indexed).
        batch_size: Batch size for evaluation.
        device: Device to run on.
        seed: Random seed.
        max_length: Max sequence length for truncation.

    Returns:
        LossRecoveredResult with loss_recovered score and CE breakdowns.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model.eval().to(device)

    if hasattr(model, "esm"):
        encoder = model.esm.encoder
    else:
        encoder = model.encoder

    # Facebook ESM2 uses .layer, NVIDIA ESM2 uses .layers
    if hasattr(encoder, "layer"):
        encoder_layers = encoder.layer
    elif hasattr(encoder, "layers"):
        encoder_layers = encoder.layers
    else:
        raise AttributeError(
            f"Cannot find encoder layers on {type(encoder).__name__}. Expected 'layer' or 'layers' attribute."
        )

    # Pre-tokenize into batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        enc = tokenizer(
            batch_seqs,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batches.append(
            {
                "input_ids": enc["input_ids"].to(device),
                "attention_mask": enc["attention_mask"].to(device),
            }
        )

    def get_hiddens(batch):
        outputs = model(
            batch["input_ids"],
            batch["attention_mask"],
            output_hidden_states=True,
        )
        return outputs.hidden_states[layer_idx + 1]

    def compute_ce(batch, hidden_override=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if hidden_override is None:
            logits = model(input_ids, attention_mask).logits
        else:
            logits = _forward_with_hidden(
                model,
                encoder_layers,
                layer_idx,
                input_ids,
                attention_mask,
                hidden_override,
            )

        return _esm2_sequence_ce(logits, input_ids, attention_mask)

    def get_recon_mask(batch):
        """Mask excluding CLS/EOS/padding — matches step1 extraction."""
        return _reconstruction_mask(batch["attention_mask"])

    return evaluate_loss_recovered(
        sae=sae,
        batches=batches,
        get_hiddens=get_hiddens,
        compute_ce=compute_ce,
        device=device,
        get_recon_mask=get_recon_mask,
    )


def _forward_with_hidden(
    model: torch.nn.Module,
    encoder_layers: torch.nn.ModuleList,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_override: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with the hidden state at layer_idx replaced."""

    def hook_fn(module, inputs, output):
        # Cast to match model dtype (e.g. SAE outputs float32 but model runs in bf16)
        ref = output[0] if isinstance(output, tuple) else output
        h = hidden_override.to(dtype=ref.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    handle = encoder_layers[layer_idx].register_forward_hook(hook_fn)
    try:
        outputs = model(input_ids, attention_mask)
        return outputs.logits
    finally:
        handle.remove()


def _reconstruction_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Bool mask of positions the SAE should reconstruct (excludes CLS, EOS, padding).

    Matches step1_15b_extract.py's special token removal logic.
    """
    B, L = attention_mask.shape
    mask = attention_mask.bool().clone()

    # Exclude CLS (position 0)
    if L > 0:
        mask[:, 0] = False

    # Exclude EOS (last real token per sequence)
    lengths = attention_mask.sum(dim=1)
    for i in range(B):
        eos = int(lengths[i].item()) - 1
        if eos > 0:
            mask[i, eos] = False

    return mask


def _esm2_sequence_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[float, int]:
    """CE for masked language models: score every non-special, non-padding token.

    Excludes start and end tokens. Returns (total_ce, n_tokens).
    """
    B, L, V = logits.shape

    valid_mask = attention_mask.clone()

    # Exclude position 0 (start token)
    if L > 0:
        valid_mask[:, 0] = 0

    # Exclude the last real token per sequence (end token)
    lengths = attention_mask.sum(dim=1)
    for i in range(B):
        end_pos = int(lengths[i].item()) - 1
        if 0 <= end_pos < L:
            valid_mask[i, end_pos] = 0

    ce = F.cross_entropy(
        logits.view(-1, V),
        labels.view(-1),
        reduction="none",
    ).view(B, L)

    total_ce = (ce * valid_mask.float()).sum().item()
    n_tokens = int(valid_mask.sum().item())

    return total_ce, n_tokens
