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

"""LLM clients for auto-interpretation.

All clients use environment variables for API keys:
- ANTHROPIC_API_KEY for Anthropic
- OPENAI_API_KEY for OpenAI
- NIM_API_KEY and NIM_BASE_URL for NVIDIA NIM
"""

import concurrent.futures
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    """Response from an LLM."""

    text: str
    model: str
    usage: Optional[dict] = None


class LLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: list[str],
        max_workers: int = 10,
        show_progress: bool = True,
    ) -> list[LLMResponse]:
        """Generate responses for multiple prompts in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.generate, p) for p in prompts]

            if show_progress:
                try:
                    from tqdm import tqdm

                    futures_iter = tqdm(
                        concurrent.futures.as_completed(futures),
                        total=len(futures),
                        desc="Interpreting features",
                    )
                except ImportError:
                    futures_iter = concurrent.futures.as_completed(futures)
            else:
                futures_iter = concurrent.futures.as_completed(futures)

            # Wait for all to complete (for progress bar)
            for _ in futures_iter:
                pass

            # Get results in original order
            responses = [f.result() for f in futures]
        return responses


class AnthropicClient(LLMClient):
    """Client for Anthropic's Claude models."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
    ):
        """Initialize the Anthropic client with model and API key."""
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.max_tokens = max_tokens

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        try:
            import anthropic

            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Install anthropic: pip install anthropic")

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using the Anthropic API."""
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return LLMResponse(
            text=message.content[0].text,
            model=self.model,
            usage={
                "input_tokens": message.usage.input_tokens,
                "output_tokens": message.usage.output_tokens,
            },
        )


class OpenAICompatibleClient(LLMClient):
    """Client for any OpenAI-compatible API (OpenAI, NVIDIA NIM, vLLM, etc.).

    Args:
        model: Model name/ID
        api_key: API key (or set via api_key_env)
        api_key_env: Environment variable name for the API key
        base_url: API base URL (None for default OpenAI)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
    ):
        """Initialize the OpenAI-compatible client with model and API configuration."""
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env)
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not self.api_key:
            raise ValueError(f"{api_key_env} environment variable not set")

        try:
            import openai

            kwargs = {"api_key": self.api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self.client = openai.OpenAI(**kwargs)
        except ImportError:
            raise ImportError("Install openai: pip install openai")

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using the OpenAI-compatible API."""
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**kwargs)
        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            usage={
                "input_tokens": response.usage.prompt_tokens if response.usage else None,
                "output_tokens": response.usage.completion_tokens if response.usage else None,
            }
            if response.usage
            else None,
        )


# Backward-compatible aliases with sensible defaults


def OpenAIClient(model: str = "gpt-4o", api_key: Optional[str] = None, max_tokens: int = 1024):
    """Client for OpenAI models."""
    return OpenAICompatibleClient(
        model=model,
        api_key=api_key,
        api_key_env="OPENAI_API_KEY",  # pragma: allowlist secret
        max_tokens=max_tokens,
    )


def NIMClient(
    model: str = "meta/llama3-70b-instruct",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: int = 1024,
):
    """Client for NVIDIA NIM (uses OpenAI-compatible API)."""
    return OpenAICompatibleClient(
        model=model,
        api_key=api_key,
        api_key_env="NIM_API_KEY",  # pragma: allowlist secret
        base_url=base_url or os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        max_tokens=max_tokens,
    )


def NVIDIAInternalClient(
    model: str = "aws/anthropic/bedrock-claude-3-7-sonnet-v1",
    api_key: Optional[str] = None,
    base_url: str = "https://inference-api.nvidia.com",
    max_tokens: int = 1024,
    temperature: float = 1.0,
):
    """Client for NVIDIA internal inference API (Bedrock Claude via NVIDIA)."""
    return OpenAICompatibleClient(
        model=model,
        api_key=api_key,
        api_key_env="CLAUDE_SONNET_INFERENCE_API_KEY",  # pragma: allowlist secret
        base_url=base_url,
        max_tokens=max_tokens,
        temperature=temperature,
    )
