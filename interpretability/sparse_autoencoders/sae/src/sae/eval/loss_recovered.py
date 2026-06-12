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

"""Loss recovered metric for evaluating SAEs on any transformer model.

    loss_recovered = 1 - (CE_sae - CE_original) / (CE_zero - CE_original)

Measures how well an SAE reconstruction preserves the model's downstream
predictions, normalized between zero-ablation (0) and perfect reconstruction (1).

Also known as "fidelity" (InterPLM) or "loss recovered" (Gao et al., 2024).

Usage:
    The general evaluate_loss_recovered() takes two model-specific callables:
    - get_hiddens(batch) -> hidden states at the SAE's target layer
    - compute_ce(batch, hidden_override) -> (total_ce, n_tokens)

    Each recipe provides these for its model (ESM2, GPT-2, etc.).
"""

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
from tqdm import tqdm


@dataclass
class LossRecoveredResult:
    """Result of loss recovered evaluation."""

    loss_recovered: float
    ce_original: float
    ce_sae: float
    ce_zero: float
    n_tokens: int

    def __repr__(self) -> str:
        """Return string representation of loss recovered results."""
        return (
            f"LossRecoveredResult(loss_recovered={self.loss_recovered:.1%}, "
            f"ce_orig={self.ce_original:.3f}, ce_sae={self.ce_sae:.3f}, "
            f"ce_zero={self.ce_zero:.3f}, n_tokens={self.n_tokens})"
        )


def compute_loss_recovered(ce_original: float, ce_sae: float, ce_zero: float) -> float:
    """Compute normalized loss recovered score.

    Returns value in [0, 1] where 1 = perfect reconstruction,
    0 = no better than zero ablation.
    """
    return 1.0 - (ce_sae - ce_original) / (ce_zero - ce_original + 1e-8)


def evaluate_loss_recovered(
    sae: torch.nn.Module,
    batches: Iterable,
    get_hiddens: Callable[[Any], torch.Tensor],
    compute_ce: Callable[[Any, Optional[torch.Tensor]], Tuple[float, int]],
    device: str = "cuda",
    show_progress: bool = True,
    get_recon_mask: Optional[Callable[[Any], torch.Tensor]] = None,
) -> LossRecoveredResult:
    """Evaluate loss recovered for an SAE on any transformer model.

    Args:
        sae: Trained sparse autoencoder. forward(x) must return (reconstruction, codes).
        batches: Iterable of batches (format is model-specific, passed through to callables).
        get_hiddens: fn(batch) -> hidden states at the SAE's target layer.
        compute_ce: fn(batch, hidden_override) -> (total_ce, n_tokens).
            When hidden_override is None, runs the model normally (clean pass).
            When hidden_override is a tensor, patches the target layer output.
        device: Device for SAE inference.
        show_progress: Whether to show tqdm progress bar.
        get_recon_mask: Optional fn(batch) -> bool mask [B, L] indicating which positions
            the SAE should reconstruct. Positions where mask=False keep original hidden
            states. Use this when the SAE was trained without special tokens (e.g. CLS/EOS).

    Returns:
        LossRecoveredResult with loss_recovered score and CE breakdowns.
    """
    sae = sae.eval().to(device)

    total_ce_orig = 0.0
    total_ce_sae = 0.0
    total_ce_zero = 0.0
    total_tokens = 0

    iterator = tqdm(batches, desc="Evaluating loss recovered") if show_progress else batches

    with torch.no_grad():
        for batch in iterator:
            ce_orig, n_tok = compute_ce(batch, None)

            hidden = get_hiddens(batch)

            # Build mask for which positions to replace
            if get_recon_mask is not None:
                mask = get_recon_mask(batch).unsqueeze(-1)  # [B, L, 1]
                # Zero ablation: only zero out positions the SAE would reconstruct
                zero_hidden = hidden.clone()
                zero_hidden[mask.expand_as(hidden)] = 0.0
                ce_zero, _ = compute_ce(batch, zero_hidden)
            else:
                ce_zero, _ = compute_ce(batch, torch.zeros_like(hidden))

            shape = hidden.shape
            recon = sae(hidden.reshape(-1, shape[-1]))[0].reshape(shape)

            if get_recon_mask is not None:
                # Blend: SAE reconstruction for masked positions, original for CLS/EOS/padding
                recon = torch.where(mask.expand_as(hidden), recon, hidden)

            ce_sae, _ = compute_ce(batch, recon)

            total_ce_orig += ce_orig
            total_ce_sae += ce_sae
            total_ce_zero += ce_zero
            total_tokens += n_tok

    avg_orig = total_ce_orig / max(1, total_tokens)
    avg_sae = total_ce_sae / max(1, total_tokens)
    avg_zero = total_ce_zero / max(1, total_tokens)

    lr = compute_loss_recovered(avg_orig, avg_sae, avg_zero)

    return LossRecoveredResult(
        loss_recovered=max(0.0, min(1.0, lr)),
        ce_original=avg_orig,
        ce_sae=avg_sae,
        ce_zero=avg_zero,
        n_tokens=total_tokens,
    )
