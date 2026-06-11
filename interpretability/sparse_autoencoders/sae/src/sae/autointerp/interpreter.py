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

"""Auto-interpretation pipeline using LLMs."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from .llm import LLMClient
from .sampler import FeatureExamples, FeatureSampler


DEFAULT_PROMPT_TEMPLATE = """You are analyzing features learned by a sparse autoencoder.

Below are examples where feature {feature_idx} activates strongly:

{high_examples}

Below are examples where feature {feature_idx} does NOT activate:

{low_examples}

Describe what this feature detects in 1-2 sentences. Start with "Fires on..." (e.g., "Fires on scientific terminology and technical jargon. Common in academic or research contexts.").

Description:"""


TOKEN_PROMPT_TEMPLATE = """You are analyzing features learned by a sparse autoencoder.

Feature {feature_idx} activates most strongly on these tokens:

{high_examples}
{logit_evidence}

Examples where this feature does NOT activate:
{low_examples}

Based on the activating tokens, their contexts, and the decoder weights, describe what this feature detects in 1 sentence. Be specific about the pattern (e.g., "Past tense verbs ending in -ed" or "References to quantities and measurements").

Description:"""


@dataclass
class FeatureInterpretation:
    """Interpretation result for a single feature."""

    feature_idx: int
    description: str
    model: str
    n_high_examples: int
    n_low_examples: int


class AutoInterpreter:
    """Auto-interpret SAE features using an LLM.

    Supports two modes:
    1. **Sampler-based**: Pass a FeatureSampler with text-level examples.
    2. **Collector-based**: Pass a CollectorResult with token-level examples,
       optional decoder logits, and context windows.

    Args:
        llm_client: LLM client to use for generating descriptions
        prompt_template: Custom prompt template (use {feature_idx}, {high_examples}, {low_examples})
        max_workers: Number of parallel workers for batch processing

    Example (collector-based):
        ```python
        from sae import TokenActivationCollector, AutoInterpreter, compute_feature_logits

        collector = TokenActivationCollector(encode_fn, n_features=sae.hidden_dim)
        result = collector.collect(texts)

        logits = compute_feature_logits(sae, unembedding, vocab)
        logits_map = {fl.feature_id: fl for fl in logits}

        interpreter = AutoInterpreter(llm_client=client)
        results = interpreter.interpret_features(
            collector=result,
            logits=logits_map,
            feature_indices=top_features,
        )
        ```
    """

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: Optional[str] = None,
        max_workers: int = 10,
    ):
        """Initialize the auto-interpreter with an LLM client and prompt template."""
        self.llm_client = llm_client
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.max_workers = max_workers

    def _build_sampler_prompt(self, examples: FeatureExamples) -> str:
        """Build prompt from sampler-based examples."""
        high_str = "\n".join(
            f"Example {i + 1} (activation={ex['activation']:.3f}):\n{ex['formatted']}"
            for i, ex in enumerate(examples.high_examples)
        )

        low_str = "\n".join(f"Example {i + 1}:\n{ex['formatted']}" for i, ex in enumerate(examples.low_examples))

        return self.prompt_template.format(
            feature_idx=examples.feature_idx,
            high_examples=high_str or "(no high activation examples)",
            low_examples=low_str or "(no low activation examples)",
        )

    def _build_token_prompt(
        self,
        feature_idx: int,
        collector_result: Any,
        logits: Optional[Dict[int, Any]] = None,
        n_examples: int = 5,
        context_window: int = 3,
    ) -> str:
        """Build prompt with token-level evidence from a CollectorResult."""
        token_examples = collector_result.token_examples.get(feature_idx, [])

        # Dedup by lowercased token string (keep highest activation per unique token)
        seen_tokens = set()
        unique_examples: List[Any] = []
        for ex in token_examples:
            tok_lower = ex.token_label.strip().lower()
            if tok_lower not in seen_tokens and len(unique_examples) < n_examples:
                seen_tokens.add(tok_lower)
                unique_examples.append(ex)

        # Format high-activation examples with context windows
        high_str = ""
        for ex in unique_examples:
            text_labels = collector_result.get_text_labels(ex.text_idx)
            start = max(0, ex.position - context_window)
            end = min(len(text_labels), ex.position + context_window + 1)

            ctx_parts = []
            for j in range(start, end):
                if j == ex.position:
                    ctx_parts.append(f"**[{text_labels[j]}]**")
                else:
                    ctx_parts.append(text_labels[j])
            context_str = "".join(ctx_parts)

            high_str += f'  \u2022 "{ex.token_label.strip()}" (activation={ex.activation:.2f})\n'
            high_str += f"    Context: ...{context_str}...\n"

        # Logit evidence
        logit_str = ""
        if logits and feature_idx in logits:
            fl = logits[feature_idx]
            pos_tokens = [t for t, v in fl.top_positive[:5]]
            neg_tokens = [t for t, v in fl.top_negative[:5]]
            logit_str = f"\nDecoder weight analysis:\n  Promotes: {pos_tokens}\n  Suppresses: {neg_tokens}"

        # Counter-examples: random zero-activation tokens with context
        non_activating = []
        n_texts = collector_result.total_texts
        sample_indices = random.sample(range(n_texts), min(50, n_texts))

        for text_idx in sample_indices:
            if len(non_activating) >= 3:
                break
            text_labels = collector_result.get_text_labels(text_idx)
            text_codes = collector_result.get_text_codes(text_idx)

            if text_codes is not None:
                acts = text_codes[:, feature_idx]
                zero_positions = (acts == 0).nonzero(as_tuple=True)[0]
                if len(zero_positions) > 0:
                    pos = zero_positions[torch.randint(len(zero_positions), (1,))].item()
                    start = max(0, pos - 2)
                    end = min(len(text_labels), pos + 3)
                    context = "".join(text_labels[start:end])
                    non_activating.append(f"  \u2022 ...{context}...")

        low_str = "\n".join(non_activating) if non_activating else "(no examples)"

        return TOKEN_PROMPT_TEMPLATE.format(
            feature_idx=feature_idx,
            high_examples=high_str or "(no high activation examples)",
            logit_evidence=logit_str,
            low_examples=low_str,
        )

    def interpret_feature(
        self,
        examples: FeatureExamples,
    ) -> FeatureInterpretation:
        """Interpret a single feature using sampler-based examples."""
        prompt = self._build_sampler_prompt(examples)
        response = self.llm_client.generate(prompt)

        return FeatureInterpretation(
            feature_idx=examples.feature_idx,
            description=response.text.strip(),
            model=response.model,
            n_high_examples=len(examples.high_examples),
            n_low_examples=len(examples.low_examples),
        )

    def interpret_features(
        self,
        # Existing sampler-based path (backward-compatible):
        sampler: Optional[FeatureSampler] = None,
        feature_indices: Optional[list[int]] = None,
        # New token-level path:
        collector: Optional[Any] = None,
        logits: Optional[Dict[int, Any]] = None,
        n_examples: int = 5,
        context_window: int = 3,
        show_progress: bool = True,
    ) -> list[FeatureInterpretation]:
        """Interpret multiple features in parallel.

        Two modes:
        1. **Sampler-based**: Pass ``sampler`` and ``feature_indices``.
        2. **Collector-based**: Pass ``collector`` (a CollectorResult) and
           ``feature_indices``, with optional ``logits`` dict.

        Args:
            sampler: FeatureSampler with activations and data (mode 1)
            feature_indices: List of feature indices to interpret
            collector: CollectorResult from TokenActivationCollector (mode 2)
            logits: Dict mapping feature_id -> FeatureLogits (mode 2)
            n_examples: Number of top examples per feature (mode 2)
            context_window: Tokens of context on each side (mode 2)
            show_progress: Whether to show progress bar

        Returns:
            List of FeatureInterpretation results
        """
        if feature_indices is None:
            raise ValueError("feature_indices is required")

        # Build (feature_idx, prompt, n_high, n_low) for each feature
        prompt_info: List[tuple] = []

        if collector is not None:
            for feat_idx in feature_indices:
                prompt = self._build_token_prompt(
                    feat_idx,
                    collector,
                    logits,
                    n_examples=n_examples,
                    context_window=context_window,
                )
                n_high = min(len(collector.token_examples.get(feat_idx, [])), n_examples)
                prompt_info.append((feat_idx, prompt, n_high, 3))

        elif sampler is not None:
            all_examples = sampler.sample_features(feature_indices)
            for ex in all_examples:
                prompt = self._build_sampler_prompt(ex)
                prompt_info.append(
                    (
                        ex.feature_idx,
                        prompt,
                        len(ex.high_examples),
                        len(ex.low_examples),
                    )
                )

        else:
            raise ValueError("Either 'sampler' or 'collector' must be provided")

        # Single generate_batch call
        prompts = [p for _, p, _, _ in prompt_info]
        responses = self.llm_client.generate_batch(
            prompts,
            max_workers=self.max_workers,
            show_progress=show_progress,
        )

        # Build results
        return [
            FeatureInterpretation(
                feature_idx=feat_idx,
                description=response.text.strip(),
                model=response.model,
                n_high_examples=n_high,
                n_low_examples=n_low,
            )
            for (feat_idx, _, n_high, n_low), response in zip(prompt_info, responses)
        ]

    @staticmethod
    def save_results(
        results: list[FeatureInterpretation],
        path: str | Path,
    ) -> None:
        """Save interpretation results to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = [asdict(r) for r in results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(results)} interpretations to {path}")

    @staticmethod
    def load_results(path: str | Path) -> list[FeatureInterpretation]:
        """Load interpretation results from JSON."""
        with open(path) as f:
            data = json.load(f)
        return [FeatureInterpretation(**d) for d in data]
