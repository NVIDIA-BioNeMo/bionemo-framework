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

"""Auto-interpretation pipeline for SAE features using LLMs.

Example usage:
    ```python
    from sae.autointerp import (
        AutoInterpreter,
        FeatureSampler,
        AnthropicClient,
        OpenAIClient,
        NIMClient,
    )

    # Define how to format your data for the prompt
    def format_example(data_item, activation, indices):
        return f"Text: {data_item}"

    # Setup sampler with your activations and data
    sampler = FeatureSampler(
        activations=sae_activations,  # [n_samples, hidden_dim]
        data=raw_data,                 # list of data items
        format_fn=format_example,
    )

    # Setup LLM client (uses env vars for API keys)
    client = AnthropicClient(model="claude-sonnet-4-20250514")

    # Run interpretation
    interpreter = AutoInterpreter(llm_client=client)
    results = interpreter.interpret_features(
        sampler=sampler,
        feature_indices=list(range(100)),  # interpret first 100 features
    )

    # Save results (joinable by feature_idx)
    interpreter.save_results(results, "interpretations.json")
    ```
"""

from .interpreter import (
    DEFAULT_PROMPT_TEMPLATE,
    TOKEN_PROMPT_TEMPLATE,
    AutoInterpreter,
    FeatureInterpretation,
)
from .llm import (
    AnthropicClient,
    LLMClient,
    LLMResponse,
    NIMClient,
    NVIDIAInternalClient,
    OpenAIClient,
    OpenAICompatibleClient,
)
from .sampler import (
    FeatureExamples,
    FeatureSampler,
)


__all__ = [
    "DEFAULT_PROMPT_TEMPLATE",
    "TOKEN_PROMPT_TEMPLATE",
    "AnthropicClient",
    "AutoInterpreter",
    "FeatureExamples",
    "FeatureInterpretation",
    "FeatureSampler",
    "LLMClient",
    "LLMResponse",
    "NIMClient",
    "NVIDIAInternalClient",
    "OpenAIClient",
    "OpenAICompatibleClient",
]
