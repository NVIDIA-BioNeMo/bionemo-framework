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

"""Loss recovered evaluation for SAEs on CodonFM (Encodon).

CodonFM-specific wrapper around the general loss_recovered metric.
CE is computed over all non-special codon tokens (positions 1..length-2),
matching extract.py's CLS/SEP removal.
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sae.eval import LossRecoveredResult, evaluate_loss_recovered


def evaluate_codonfm_loss_recovered(
    sae: torch.nn.Module,
    inference,
    sequences: List[str],
    layer: int,
    context_length: int = 2048,
    batch_size: int = 8,
    device: str = "cuda",
    seed: int = 42,
) -> LossRecoveredResult:
    """Evaluate SAE loss recovered on codon sequences.

    Args:
        sae: Trained sparse autoencoder.
        inference: EncodonInference instance (already configured, on device).
        sequences: List of DNA sequences.
        layer: Layer index (negative indexing supported).
        context_length: Max context length for tokenization.
        batch_size: Batch size for evaluation.
        device: Device to run on.
        seed: Random seed.

    Returns:
        LossRecoveredResult with loss_recovered score and CE breakdowns.
    """
    from src.data.preprocess.codon_sequence import process_item

    np.random.seed(seed)
    torch.manual_seed(seed)

    # Resolve negative layer index
    num_layers = len(inference.model.model.layers)
    layer_idx = layer if layer >= 0 else num_layers + layer

    # The EnCodon model: inference.model -> EncodonPL -> .model -> EnCodon
    encodon_model = inference.model.model
    encoder_layers = encodon_model.layers

    # Pre-tokenize into batches
    batches = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        items = [process_item(s, context_length=context_length, tokenizer=inference.tokenizer) for s in batch_seqs]
        batch = {
            "input_ids": torch.tensor(np.stack([it["input_ids"] for it in items])).to(device),
            "attention_mask": torch.tensor(np.stack([it["attention_mask"] for it in items])).to(device),
        }
        batches.append(batch)

    def get_hiddens(batch):
        out = inference.model(batch, return_hidden_states=True)
        return out.all_hidden_states[layer_idx]

    def compute_ce(batch, hidden_override=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if hidden_override is None:
            out = inference.model(batch)
            logits = out.logits
        else:
            logits = _forward_with_hidden(
                encodon_model,
                encoder_layers,
                layer_idx,
                input_ids,
                attention_mask,
                hidden_override,
            )

        return _codonfm_sequence_ce(logits, input_ids, attention_mask)

    def get_recon_mask(batch):
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
    encodon_model,
    encoder_layers,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_override: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with the hidden state at layer_idx replaced.

    EncoderLayer.forward returns a single tensor (not a tuple),
    so the hook simply returns the replacement.
    """

    def hook_fn(module, inputs, output):
        return hidden_override.to(dtype=output.dtype)

    handle = encoder_layers[layer_idx].register_forward_hook(hook_fn)
    try:
        out = encodon_model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits
    finally:
        handle.remove()


def _reconstruction_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Bool mask excluding CLS (pos 0), SEP (last real pos), and padding.

    Matches extract.py's `layer_acts[j, 1:seq_len-1, :]` logic.
    """
    B, L = attention_mask.shape
    mask = attention_mask.bool().clone()

    # Exclude CLS (position 0)
    if L > 0:
        mask[:, 0] = False

    # Exclude SEP (last real token per sequence)
    lengths = attention_mask.sum(dim=1)
    for i in range(B):
        sep = int(lengths[i].item()) - 1
        if sep > 0:
            mask[i, sep] = False

    return mask


def _codonfm_sequence_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[float, int]:
    """CE over non-special, non-padding codon tokens.

    Excludes CLS (pos 0) and SEP (last real pos).
    Returns (total_ce, n_tokens).
    """
    B, L, V = logits.shape

    valid_mask = attention_mask.clone()

    # Exclude CLS
    if L > 0:
        valid_mask[:, 0] = 0

    # Exclude SEP
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
