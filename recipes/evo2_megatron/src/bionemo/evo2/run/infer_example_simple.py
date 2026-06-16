# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

"""Simple autoregressive generation example for Evo2 models.

This module provides a straightforward implementation of autoregressive text generation
that directly calls the model forward pass without using the native dynamic-inference
infrastructure. This is useful for:

1. Understanding how autoregressive generation works at a low level
2. Debugging and testing direct model-forward behavior
3. Custom generation workflows that don't fit the MCore API

For production use, prefer the MCore-based inference in `bionemo.evo2.run.infer`.

Example:
    >>> from bionemo.evo2.run.infer_example_simple import generate_tokens_simple
    >>> from bionemo.evo2.models.evo2_provider import HyenaInferenceContext
    >>>
    >>> # Assuming model and tokenizer are already loaded
    >>> ctx = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
    >>> prompt_tokens = tokenizer.text_to_ids("ATCGATCG")
    >>> tokens = generate_tokens_simple(model, prompt_tokens, max_new_tokens=100, inference_context=ctx)
"""

from typing import List, Optional

import torch

from bionemo.evo2.models.evo2_provider import HyenaInferenceContext


@torch.inference_mode()
def generate_tokens_simple(
    model: torch.nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    inference_context: Optional[HyenaInferenceContext] = None,
) -> List[int]:
    """Generate tokens autoregressively using direct model forward passes.

    This function implements autoregressive generation by repeatedly calling
    the model's forward pass with the previously generated token. It properly
    manages the HyenaInferenceContext to cache SSM state between steps.

    Unlike the MCore-based inference in `bionemo.evo2.run.infer`, this function:
    - Directly calls model.forward() instead of using inference wrappers
    - Manually manages sequence_len_offset and decode_mode
    - Does not use the native dynamic-inference request lifecycle

    Args:
        model: The Evo2 model (typically Float16Module wrapped).
        prompt_tokens: Input prompt token IDs as a tensor of shape [1, seq_len].
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature. Higher values (e.g., 1.0) make output
            more random, lower values make it more deterministic. Default is 1.0.
        top_k: Top-k sampling parameter. If > 0, only the top k tokens are
            considered for sampling. Use top_k=1 for greedy decoding. Default is 0.
        inference_context: Hyena-specific context for SSM state caching.
            If provided, enables efficient autoregressive generation by caching
            filter states between decode steps.

    Returns:
        List of generated token IDs (excluding the prompt).

    Example:
        >>> # Setup
        >>> ctx = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
        >>> prompt = torch.tensor([[65, 84, 67, 71]], device="cuda")  # "ATCG"
        >>>
        >>> # Generate with greedy decoding
        >>> tokens = generate_tokens_simple(
        ...     model, prompt, max_new_tokens=10, top_k=1, inference_context=ctx
        ... )
        >>> print(tokens)  # [65, 84, 67, 71, ...]  (continues the pattern)

    Note:
        For production use, prefer `bionemo.evo2.run.infer.generate()` which uses the
        native MCore dynamic-inference path with proper sampling, request lifecycle, and
        distributed support.
    """
    device = prompt_tokens.device
    generated_tokens: List[int] = []
    prompt_len = prompt_tokens.shape[1]

    # Ensure context starts in prefill mode
    if inference_context is not None:
        inference_context.enable_prefill_mode()

    # Process the full prompt first (prefill phase)
    # This computes and caches the SSM states for all prompt tokens
    logits = model(
        input_ids=prompt_tokens,
        position_ids=None,
        attention_mask=None,
        labels=None,
        runtime_gather_output=True,
        inference_context=inference_context,
    )

    # Update sequence_len_offset after prefill (MCore wrapper does this automatically)
    if inference_context is not None:
        inference_context.increment_sequence_len_offset(prompt_len)
        # Switch to decode mode after prefill is complete
        inference_context.enable_decode_mode()

    # Get next token from last position logits
    next_token_logits = logits[0, -1, :].clone()

    # Generate tokens autoregressively (decode phase)
    for _ in range(max_new_tokens):
        # Apply temperature scaling
        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float("-inf")

        # Sample or argmax
        if temperature > 0 and top_k != 1:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = torch.argmax(next_token_logits).item()

        generated_tokens.append(next_token)

        # Prepare next input (single token)
        next_input = torch.tensor([[next_token]], dtype=torch.long, device=device)

        # Forward pass with cached state
        logits = model(
            input_ids=next_input,
            position_ids=None,
            attention_mask=None,
            labels=None,
            runtime_gather_output=True,
            inference_context=inference_context,
        )

        # Update sequence_len_offset after each decode step
        if inference_context is not None:
            inference_context.increment_sequence_len_offset(1)

        next_token_logits = logits[0, -1, :].clone()

    return generated_tokens
