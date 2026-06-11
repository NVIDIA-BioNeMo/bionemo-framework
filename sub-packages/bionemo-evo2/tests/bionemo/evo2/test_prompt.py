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

from dataclasses import dataclass

import pytest
from megatron.core.inference.inference_request import InferenceRequest

from bionemo.core.data.load import load
from bionemo.evo2.run.infer import infer


RANDOM_SEED = 42
MAX_NEW_TOKENS = 500
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.0

# todo: figure out 1M checkpoints (or add to NGC)
CHECKPOINT_NAMES = [
    "evo2/1b-8k-bf16:1.0",
    # "evo2/7b-8k:1.0",
    # "evo2/7b-1m:1.0",
]


PROMPT_1 = "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGGGAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGG"

PROMPT_2 = "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTT"


@dataclass
class InferCofig:
    """Configuration for model inference parameters."""

    temperature: float = TEMPERATURE
    top_k: int = TOP_K
    top_p: float = TOP_P
    tensor_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    max_new_tokens: int = MAX_NEW_TOKENS
    ckpt_format: str = "torch_dist"
    seed: int = RANDOM_SEED
    flash_decode: bool = False


_checkpoint_cache = {}


@pytest.fixture(scope="session")
def load_checkpoint():
    """Factory function that returns a checkpoint loader with caching."""

    def _load_checkpoint(ckpt_name: str) -> str:
        if ckpt_name not in _checkpoint_cache:
            _checkpoint_cache[ckpt_name] = load(ckpt_name)
        return _checkpoint_cache[ckpt_name]

    return _load_checkpoint


def percent_equal_tokens(response1: list[InferenceRequest], response2: list[InferenceRequest]) -> float:
    """Percent of tokens that are equal between two responses."""
    num_equal = [i == j for i, j in zip(response1[0].generated_tokens, response2[0].generated_tokens)]
    return sum(num_equal) / len(num_equal)


# just a DRY wrapper for the infer function
def run_inference(prompt: str, checkpoint_path: str, config: InferCofig) -> list[InferenceRequest]:
    """Run model inference with given parameters.

    Args:
        prompt: Input prompt for the model
        checkpoint_path: Path to model checkpoint
        config: Inference configuration parameters

    Returns:
        Model response
    """
    return infer(
        prompt=prompt,
        ckpt_dir=checkpoint_path,
        temperature=config.temperature,
        top_k=config.top_k,
        top_p=config.top_p,
        max_new_tokens=config.max_new_tokens,
        tensor_parallel_size=config.tensor_parallel_size,
        pipeline_model_parallel_size=config.pipeline_model_parallel_size,
        context_parallel_size=config.context_parallel_size,
        output_file=None,
        ckpt_format=config.ckpt_format,
        seed=config.seed,
        flash_decode=config.flash_decode,
    )


@pytest.mark.parametrize("ckpt_name", CHECKPOINT_NAMES)
def test_identical_prompts_should_be_identical(load_checkpoint, ckpt_name):
    """Test that identical prompts produce identical sequences for temperature 1.0."""
    checkpoint_path = load_checkpoint(ckpt_name)

    # with clean_parallel_state_context():
    response_prompt1 = run_inference(PROMPT_1, checkpoint_path, InferCofig())
    response_prompt2 = run_inference(PROMPT_1, checkpoint_path, InferCofig())

    sequence_similarity = percent_equal_tokens(response_prompt1, response_prompt2)
    print(f"sequence similarity {ckpt_name} identical prompts: {sequence_similarity}")
    assert sequence_similarity == 1.0


@pytest.mark.parametrize("ckpt_name", CHECKPOINT_NAMES)
def test_different_prompts_too_similar(load_checkpoint, ckpt_name):
    """Test that different prompts for the same sequence are too similar.
    That is, different prompts should produce more varied sequences.
    """
    checkpoint_path = load_checkpoint(ckpt_name)

    similarity_threshold = 0.9

    # with clean_parallel_state_context():
    response_prompt1 = run_inference(PROMPT_1, checkpoint_path, InferCofig())
    response_prompt2 = run_inference(PROMPT_2, checkpoint_path, InferCofig())
    sequence_similarity = percent_equal_tokens(response_prompt1, response_prompt2)
    assert sequence_similarity <= similarity_threshold
