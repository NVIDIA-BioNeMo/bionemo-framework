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

"""Loss recovered evaluation for SAEs on Nemotron-3-Nano (causal LM).

Uses next-token prediction cross-entropy: logits[t] predicts token[t+1].

The hook target is model.model.layers[layer_idx] — Nemotron's hybrid blocks
(Mamba-2 / MoE / GQA) may return a single tensor or a tuple depending on
layer type. The hook handles both cases.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F
from sae.eval import LossRecoveredResult, evaluate_loss_recovered


def evaluate_nemotron_loss_recovered(
    sae: torch.nn.Module,
    model,
    tokenizer,
    texts: List[str],
    layer_idx: int,
    batch_size: int = 4,
    device: str = "cuda",
    max_length: int = 2048,
) -> LossRecoveredResult:
    """Evaluate SAE loss recovered on text using Nemotron-3-Nano.

    Args:
        sae: Trained sparse autoencoder.
        model: Nemotron CausalLM model (AutoModelForCausalLM).
        tokenizer: Nemotron tokenizer.
        texts: List of text strings.
        layer_idx: Which transformer layer to intervene on (0-indexed).
        batch_size: Batch size for evaluation (keep small for 30B model).
        device: Device for SAE inference.
        max_length: Max sequence length for truncation.

    Returns:
        LossRecoveredResult with loss_recovered score and CE breakdowns.
    """
    model = model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Nemotron's transformer blocks: model.model.layers
    transformer_blocks = model.model.layers

    # Pre-tokenize into batches
    batches = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        batches.append(
            {
                "input_ids": enc["input_ids"].to(model.device),
                "attention_mask": enc["attention_mask"].to(model.device),
            }
        )

    def get_hiddens(batch):
        outputs = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
        )
        # hidden_states[0] = embeddings, [1] = layer 0 output, etc.
        return outputs.hidden_states[layer_idx + 1]

    def compute_ce(batch, hidden_override=None):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        if hidden_override is None:
            logits = model(input_ids, attention_mask=attention_mask).logits
        else:
            logits = _forward_with_hidden(
                model,
                transformer_blocks,
                layer_idx,
                input_ids,
                attention_mask,
                hidden_override,
            )

        return _causal_lm_ce(logits, input_ids, attention_mask)

    return evaluate_loss_recovered(
        sae=sae,
        batches=batches,
        get_hiddens=get_hiddens,
        compute_ce=compute_ce,
        device=device,
    )


def _forward_with_hidden(
    model,
    transformer_blocks: torch.nn.ModuleList,
    layer_idx: int,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hidden_override: torch.Tensor,
) -> torch.Tensor:
    """Forward pass with the hidden state at layer_idx replaced."""

    def hook_fn(module, inputs, output):
        # Hybrid blocks may return tuple (hidden_states, ...) or single tensor
        if isinstance(output, tuple):
            return (hidden_override,) + output[1:]
        return hidden_override

    handle = transformer_blocks[layer_idx].register_forward_hook(hook_fn)
    try:
        outputs = model(input_ids, attention_mask=attention_mask)
        return outputs.logits
    finally:
        handle.remove()


def _causal_lm_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[float, int]:
    """Next-token prediction CE for causal language models.

    logits[t] predicts input_ids[t+1]. Returns (total_ce, n_tokens).
    """
    # Shift: logits[:-1] predicts input_ids[1:]
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    B, L, V = shift_logits.shape

    ce = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        reduction="none",
    ).view(B, L)

    total_ce = (ce * shift_mask.float()).sum().item()
    n_tokens = int(shift_mask.sum().item())

    return total_ce, n_tokens
