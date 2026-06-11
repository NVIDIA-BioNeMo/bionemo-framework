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

"""Token-level activation collector for SAE features.

Replaces the memory-intensive compute_feature_stats() with a streaming
approach that uses bounded min-heaps to track top-k token examples per
feature without accumulating all activations.

Example:
    ```python
    from sae import TokenActivationCollector

    def encode_fn(text):
        tokens, codes = run_model_and_sae(text)
        return tokens, codes  # (List[str], Tensor[n_tokens, n_features])

    collector = TokenActivationCollector(encode_fn, n_features=12288)
    result = collector.collect(texts[:5000])

    # result.feature_stats   - per-feature activation statistics
    # result.text_codes      - [n_texts, n_features] max-pooled activations
    # result.token_examples  - top-k token examples per feature
    ```
"""

import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .analysis import FeatureStats, _StatsAccumulator


@dataclass
class TokenExample:
    """A top-activating token example for a feature."""

    feature_id: int
    text_idx: int
    position: int
    token_label: str
    activation: float


@dataclass
class CollectorResult:
    """Output from TokenActivationCollector.collect()."""

    feature_stats: List[FeatureStats]
    text_codes: torch.Tensor  # [n_texts, n_features]
    token_examples: Dict[int, List[TokenExample]]  # per-feature top-k
    total_tokens: int
    total_texts: int
    _text_labels: List[List[str]] = field(default_factory=list, repr=False)
    _text_token_codes: Optional[List[torch.Tensor]] = field(default=None, repr=False)

    def get_text_labels(self, text_idx: int) -> List[str]:
        """Get token labels for a specific text (for context window lookups)."""
        return self._text_labels[text_idx]

    def get_text_codes(self, text_idx: int) -> Optional[torch.Tensor]:
        """Get full token codes for a text (only available if store_token_codes=True)."""
        if self._text_token_codes is None:
            return None
        return self._text_token_codes[text_idx]


class TokenActivationCollector:
    """Collect token-level activation statistics from an SAE.

    Uses bounded min-heaps to efficiently track top-k token examples per
    feature without storing all activations in memory.

    Args:
        encode_fn: Function that takes an input item and returns
            (token_labels, sae_codes) where codes is [n_tokens, n_features]
        n_features: Number of SAE features (hidden_dim)
        top_k_tokens: Number of top token examples to keep per feature
        store_token_codes: If True, store full token codes per text (for
            features.json export). Increases memory usage.
    """

    def __init__(
        self,
        encode_fn: Callable,
        n_features: int,
        top_k_tokens: int = 50,
        store_token_codes: bool = False,
    ):
        """Initialize the collector with an encoding function and feature count."""
        self.encode_fn = encode_fn
        self.n_features = n_features
        self.top_k_tokens = top_k_tokens
        self.store_token_codes = store_token_codes

    def collect(
        self,
        inputs: list,
        show_progress: bool = True,
    ) -> CollectorResult:
        """Iterate inputs, collect per-feature stats and top-k token examples.

        Args:
            inputs: Iterable of input items to pass to encode_fn
            show_progress: Whether to show a tqdm progress bar

        Returns:
            CollectorResult with feature_stats, text_codes, token_examples
        """
        n_features = self.n_features
        top_k = self.top_k_tokens

        acc = _StatsAccumulator(n_features)

        # Bounded min-heaps per feature: heap of (activation, text_idx, position, token_label)
        heaps: List[list] = [[] for _ in range(n_features)]

        # Per-text outputs
        text_codes_list: List[torch.Tensor] = []
        text_labels_list: List[List[str]] = []
        text_token_codes_list: Optional[List[torch.Tensor]] = [] if self.store_token_codes else None

        total_tokens = 0

        iterator = inputs
        if show_progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(inputs, desc="Collecting activations")
            except ImportError:
                pass

        for text_idx, item in enumerate(iterator):
            token_labels, codes = self.encode_fn(item)
            # codes: [n_tokens, n_features]
            if isinstance(codes, torch.Tensor):
                codes = codes.detach().cpu()
            else:
                codes = torch.tensor(codes)

            n_tokens = codes.shape[0]
            total_tokens += n_tokens

            # Store token labels
            text_labels_list.append(list(token_labels))

            # Optionally store full token codes
            if text_token_codes_list is not None:
                text_token_codes_list.append(codes)

            # Update stats
            acc.update(codes)

            # Max-pool across tokens for text-level codes
            text_code = codes.max(dim=0).values
            text_codes_list.append(text_code)

            # Update top-k heaps for active tokens
            # Only process tokens/features with nonzero activations
            active_positions = codes.nonzero(as_tuple=False)  # [N, 2] of (token_pos, feat_idx)
            for row in active_positions:
                pos = row[0].item()
                feat = row[1].item()
                act_val = codes[pos, feat].item()
                heap = heaps[feat]

                entry = (act_val, text_idx, pos, token_labels[pos])

                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif act_val > heap[0][0]:
                    heapq.heapreplace(heap, entry)

        # Build feature stats
        feature_stats = acc.build_stats()

        # Convert heaps to sorted TokenExample lists (descending by activation)
        token_examples: Dict[int, List[TokenExample]] = {}
        for feat_idx in range(n_features):
            heap = heaps[feat_idx]
            if heap:
                examples = sorted(heap, key=lambda x: -x[0])
                token_examples[feat_idx] = [
                    TokenExample(
                        feature_id=feat_idx,
                        text_idx=entry[1],
                        position=entry[2],
                        token_label=entry[3],
                        activation=entry[0],
                    )
                    for entry in examples
                ]

        # Stack text codes
        text_codes = torch.stack(text_codes_list) if text_codes_list else torch.zeros(0, n_features)

        return CollectorResult(
            feature_stats=feature_stats,
            text_codes=text_codes,
            token_examples=token_examples,
            total_tokens=total_tokens,
            total_texts=len(text_codes_list),
            _text_labels=text_labels_list,
            _text_token_codes=text_token_codes_list,
        )
