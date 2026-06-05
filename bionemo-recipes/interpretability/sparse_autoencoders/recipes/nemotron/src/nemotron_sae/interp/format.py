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

"""Text formatting utilities for auto-interpretation.

Provides format_fn callback for the FeatureSampler that formats
text samples for LLM-based feature interpretation.
"""

from typing import Any, Callable, List, Optional


# Default prompt template for text LM features
TEXT_PROMPT_TEMPLATE = """You are analyzing features learned by a sparse autoencoder trained on Nemotron-3-Nano activations.

Below are text examples where feature {feature_idx} activates strongly:

{high_examples}

Below are text examples where feature {feature_idx} does NOT activate:

{low_examples}

Describe what this feature detects in 1 sentence. Be direct and specific (e.g., "Historical dates and years, often in contexts discussing past events or timelines.").

Description:"""


def create_text_formatter(
    tokenizer: Optional[Any] = None,
    max_chars: int = 500,
) -> Callable[[Any, float, List[int]], str]:
    """Create a format function for text data.

    Returns a function compatible with FeatureSampler's format_fn signature:
        format_fn(data_item, activation_value, active_indices) -> str

    Args:
        tokenizer: Optional tokenizer for token-level highlighting
        max_chars: Maximum characters to include (truncates longer texts)

    Returns:
        Callable that formats text with activation info
    """

    def format_text(
        data_item: str,
        activation_value: float,
        active_indices: List[int],
    ) -> str:
        text = str(data_item)

        if active_indices and tokenizer is not None:
            try:
                tokens = tokenizer.encode(text)
                token_strs = [tokenizer.decode([t]) for t in tokens]

                result_tokens = []
                for i, tok_str in enumerate(token_strs):
                    if i in active_indices:
                        result_tokens.append(f"[{tok_str}]")
                    else:
                        result_tokens.append(tok_str)

                text = "".join(result_tokens)
            except Exception:
                pass

        if len(text) > max_chars:
            return text[:max_chars] + "..."
        return text

    return format_text
