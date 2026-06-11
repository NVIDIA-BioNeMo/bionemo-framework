# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""Text generation (inference) workflow for Eden using Megatron Core.

This module provides autoregressive text generation for Eden models using the
MCore inference infrastructure (StaticInferenceEngine, TextGenerationController).

Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/examples/inference/gpt/gpt_static_inference.py

Usage (CLI, single prompt):
    torchrun --nproc_per_node 1 -m bionemo.eden.run.infer \
        --ckpt-dir /path/to/mbridge/checkpoint \
        --prompt "|d__Bacteria;p__Pseudomonadota|" \
        --max-new-tokens 100 \
        --output-file results.jsonl

Usage (CLI, batch from JSONL file):
    torchrun --nproc_per_node 1 -m bionemo.eden.run.infer \
        --ckpt-dir /path/to/mbridge/checkpoint \
        --prompt-file prompts.jsonl \
        --max-new-tokens 100 \
        --output-file results.jsonl

    Where prompts.jsonl contains one JSON object per line::

        {"id": "seq_001", "prompt": "ATCGATCG"}
        {"id": "seq_002", "prompt": "GCTAGCTA"}

    The output results.jsonl will contain::

        {"id": "seq_001", "prompt": "ATCGATCG", "completion": "...", "finish_reason": "length", "usage": {...}}
        {"id": "seq_002", "prompt": "GCTAGCTA", "completion": "...", "finish_reason": "stop", "usage": {...}}

Usage (Python API):
    from bionemo.eden.run.infer import setup_inference_engine, generate

    # Setup engine (loads model, creates inference components)
    components = setup_inference_engine(ckpt_dir)

    # Generate text
    results = generate(components, prompts=["ATCGATCG"], max_new_tokens=100)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from megatron.bridge.models.model_provider import ProcessGroupCollection
from megatron.bridge.training.checkpointing import _load_model_weights_from_checkpoint
from megatron.bridge.training.config import DistributedInitConfig, RNGConfig
from megatron.bridge.training.mixed_precision import get_mixed_precision_config
from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.utils.common_utils import get_world_size_safe
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.core import parallel_state
from megatron.core.inference.contexts import StaticInferenceContext
from megatron.core.inference.engines.static_engine import StaticInferenceEngine
from megatron.core.inference.model_inference_wrappers.abstract_model_inference_wrapper import (
    AbstractModelInferenceWrapper,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config

from bionemo.eden.run.predict import initialize_inference_distributed, resolve_checkpoint_path


_REPO_BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_HF_TOKENIZER_MODEL_PATH = str(_REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_256")


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# =============================================================================
# Hardware-Aware Defaults
# =============================================================================


def _get_gpu_info() -> tuple[int, int]:
    """Return ``(per_gpu_memory_gb, num_gpus)`` from CUDA device properties.

    Returns ``(0, 0)`` when CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        return (0, 0)
    mem_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
    num_gpus = torch.cuda.device_count()
    return (mem_gb, num_gpus)


def _infer_model_size(ckpt_dir: Path) -> str:
    """Infer model-size category from checkpoint path components.

    Returns one of ``"40b"``, ``"7b"``, or ``"small"`` (covers 1b / Eden / unknown).
    """
    path_lower = str(ckpt_dir).lower()
    if "40b" in path_lower:
        return "40b"
    if "7b" in path_lower:
        return "7b"
    return "small"


def _detect_max_seq_length(ckpt_dir: Path) -> int:
    """Auto-detect a conservative ``max_seq_length`` based on GPU memory and model size.

    The values are intentionally conservative and match the lookup tables used in
    NVIDIA's reference inference script.  Users can override via the
    ``EDEN_MAX_SEQ_LEN`` environment variable or the ``--max-seq-length`` CLI flag.

    Args:
        ckpt_dir: Checkpoint directory (used to infer model size).

    Returns:
        An integer suitable for ``--max-seq-length``.
    """
    mem_gb, num_gpus = _get_gpu_info()
    model_size = _infer_model_size(ckpt_dir)

    if model_size == "40b":
        if mem_gb > 120 and num_gpus >= 4:
            ret = 1_000_000
        elif mem_gb > 120 and num_gpus >= 2:
            ret = 100_000
        elif mem_gb > 120:
            ret = 20_000
        elif mem_gb > 60 and num_gpus >= 2:
            ret = 20_000
        else:
            ret = 10_000
    else:
        if mem_gb > 40:
            ret = 100_000
        else:
            ret = 20_000

    logger.info(
        f"Auto-detected max_seq_length={ret:,} (model_size={model_size}, gpu_mem={mem_gb}GB, num_gpus={num_gpus})"
    )
    return ret


def _resolve_int(cli_val: Optional[int], env_var: str, auto_default: Optional[int]) -> Optional[int]:
    """Resolve an integer setting with priority: CLI arg > env var > auto default.

    Args:
        cli_val: Value from argparse (``None`` when not supplied by user).
        env_var: Environment variable name to check.
        auto_default: Fallback value from hardware auto-detection.

    Returns:
        Resolved integer, or ``None`` when all three tiers are absent.
    """
    if cli_val is not None:
        return cli_val
    env = os.environ.get(env_var)
    if env is not None:
        resolved = int(env)
        logger.info(f"Using {env_var}={resolved} from environment")
        return resolved
    return auto_default


def _prune_caches() -> None:
    """Run ``gc.collect()`` and ``torch.cuda.empty_cache()`` to free fragmented memory.

    Called before model setup to maximise contiguous GPU memory available for
    weight loading and KV-cache allocation.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Pruned Python and CUDA caches")


# =============================================================================
# Eden Model Inference Wrapper
# =============================================================================


class EdenModelInferenceWrapper(AbstractModelInferenceWrapper):
    """Inference wrapper for Eden models.

    Extends the abstract wrapper to provide Eden-specific input preparation
    and forward pass handling for autoregressive text generation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        inference_wrapper_config: InferenceWrapperConfig,
        inference_context: Optional[StaticInferenceContext] = None,
    ):
        """Initialize the Eden inference wrapper.

        Args:
            model: The Eden model to wrap for inference.
            inference_wrapper_config: Configuration with hidden size, vocab size, etc.
            inference_context: Context for managing state and sequence offsets.
        """
        super().__init__(model, inference_wrapper_config, inference_context)

    def prep_inference_input(self, prompts_tokens: torch.Tensor) -> Dict[str, Any]:
        """Prepare the inference input data.

        Args:
            prompts_tokens: A tensor of shape [batch_size, max_seq_len]

        Returns:
            Dict with tokens, attention_mask, and position_ids
        """
        batch_size, seq_len = prompts_tokens.shape
        device = prompts_tokens.device

        # For Eden models, position_ids are sequential
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        # Eden uses causal attention - for flash attention backend, mask is None
        attention_mask = None

        return {
            "tokens": prompts_tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }

    def get_batch_for_context_window(
        self,
        inference_input: Dict[str, Any],
        context_start_position: int,
        context_end_position: int,
    ) -> Dict[str, Any]:
        """Extract batch for a specific context window.

        Called iteratively during autoregressive generation.

        Args:
            inference_input: Full inference input dict
            context_start_position: Start of context window
            context_end_position: End of context window

        Returns:
            Dict with sliced tokens, positions, and attention mask
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]

        tokens2use = tokens[:, context_start_position:context_end_position]
        positions2use = position_ids[:, context_start_position:context_end_position]

        if attention_mask is not None:
            attention_mask2use = attention_mask[
                ..., context_start_position:context_end_position, :context_end_position
            ]
        else:
            attention_mask2use = None

        return {
            "tokens": tokens2use,
            "position_ids": positions2use,
            "attention_mask": attention_mask2use,
        }

    def _forward(self, inference_input: Dict[str, Any]) -> torch.Tensor:
        """Run a forward pass of the model.

        Args:
            inference_input: The input data dict.

        Returns:
            The model output logits.
        """
        tokens = inference_input["tokens"]
        position_ids = inference_input["position_ids"]
        attention_mask = inference_input["attention_mask"]

        return self.model(
            tokens,
            position_ids,
            attention_mask,
            inference_context=self.inference_context,
            runtime_gather_output=True,
        )


# =============================================================================
# Inference Components Container
# =============================================================================


@dataclass
class EdenInferenceComponents:
    """Container for Eden inference components.

    This dataclass holds all the components needed for text generation,
    making it easy to pass around and reuse.
    """

    inference_engine: StaticInferenceEngine
    tokenizer: _HuggingFaceTokenizer
    inference_wrapper: EdenModelInferenceWrapper
    inference_context: StaticInferenceContext
    model: torch.nn.Module


# =============================================================================
# Public API: Setup and Generate Functions
# =============================================================================


def setup_inference_engine(
    ckpt_dir: Path,
    *,
    max_seq_length: int = 8192,
    max_batch_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    mixed_precision_recipe: Optional[str] = None,
    random_seed: int = 1234,
) -> EdenInferenceComponents:
    """Setup the Eden inference engine and related components.

    This function loads the model, creates the inference wrapper, and sets up
    all necessary components for text generation.

    Args:
        ckpt_dir: Path to MBridge checkpoint directory.
        max_seq_length: Maximum sequence length for generation.
        max_batch_size: Maximum batch size for inference.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        mixed_precision_recipe: Override mixed precision recipe.
        random_seed: Random seed for reproducibility.

    Returns:
        EdenInferenceComponents containing all inference components.

    Example:
        >>> components = setup_inference_engine(Path("/path/to/checkpoint"), max_batch_size=4)
        >>> results = generate(components, prompts=["ATCG", "GCTA"], max_new_tokens=100)
    """
    # -------------------------------------------------------------------------
    # Step 1: Load configuration from checkpoint
    # -------------------------------------------------------------------------
    resolved_ckpt_dir = resolve_checkpoint_path(ckpt_dir)
    logger.info(f"Loading configuration from checkpoint: {resolved_ckpt_dir}")

    run_config_filename = get_checkpoint_run_config_filename(str(resolved_ckpt_dir))
    if not file_exists(run_config_filename):
        raise FileNotFoundError(f"run_config.yaml not found at {run_config_filename}")

    run_config = read_run_config(run_config_filename)
    model_provider = instantiate(run_config["model"])
    logger.info(f"Instantiated model provider: {type(model_provider).__name__}")

    # -------------------------------------------------------------------------
    # Step 2: Configure parallelism and precision
    # -------------------------------------------------------------------------
    model_provider.tensor_model_parallel_size = tensor_parallel_size
    model_provider.pipeline_model_parallel_size = pipeline_model_parallel_size
    model_provider.context_parallel_size = context_parallel_size
    # Disable sequence parallelism for inference - Megatron's inference engine
    # does not support it for non-MoE models.
    model_provider.sequence_parallel = False

    # Use bf16_mixed for inference to avoid FP8 issues
    if mixed_precision_recipe is not None:
        mp_config = get_mixed_precision_config(mixed_precision_recipe)
    else:
        mp_config = get_mixed_precision_config("bf16_mixed")

    mp_config.finalize()
    mp_config.setup(model_provider)

    # -------------------------------------------------------------------------
    # Step 3: Load tokenizer
    # -------------------------------------------------------------------------
    tokenizer_dir = resolved_ckpt_dir / "tokenizer"
    if tokenizer_dir.exists():
        tokenizer = _HuggingFaceTokenizer(tokenizer_dir)
    else:
        tokenizer = _HuggingFaceTokenizer(DEFAULT_HF_TOKENIZER_MODEL_PATH)

    model_provider.vocab_size = tokenizer.vocab_size
    model_provider.should_pad_vocab = True

    # -------------------------------------------------------------------------
    # Step 4: Initialize distributed environment
    # -------------------------------------------------------------------------
    rng_config = instantiate(run_config.get("rng")) if run_config.get("rng") else RNGConfig(seed=random_seed)
    dist_config = instantiate(run_config.get("dist")) if run_config.get("dist") else DistributedInitConfig()

    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    world_size = get_world_size_safe()
    data_parallel_size = world_size // model_parallel_size

    initialize_inference_distributed(
        tensor_model_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        micro_batch_size=max_batch_size,
        global_batch_size=max_batch_size * data_parallel_size,
        rng_config=rng_config,
        dist_config=dist_config,
    )
    logger.info("Initialized distributed environment")

    # -------------------------------------------------------------------------
    # Step 5: Create model and load weights
    # -------------------------------------------------------------------------
    logger.info("Creating model...")
    model_provider.finalize()

    # _pg_collection is a dataclass field on GPTModelProvider (megatron.bridge);
    # setting it before provide() is the intended configuration pattern.
    model_provider._pg_collection = ProcessGroupCollection.use_mpu_process_groups()

    raw_model = model_provider.provide().eval().cuda()

    logger.info(f"Loading weights from: {resolved_ckpt_dir}")
    _load_model_weights_from_checkpoint(
        checkpoint_path=str(resolved_ckpt_dir),
        model=[raw_model],
        dist_ckpt_strictness="ignore_all",
    )
    logger.info("Weights loaded successfully")

    # Wrap with Float16Module
    model = Float16Module(model_provider, raw_model)

    # -------------------------------------------------------------------------
    # Step 6: Setup MCore inference infrastructure
    # -------------------------------------------------------------------------
    # Create inference wrapper config
    model_config = get_model_config(raw_model)
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=model_config.hidden_size,
        inference_max_requests=max_batch_size,
        inference_max_seq_length=max_seq_length,
        inference_batch_times_seqlen_threshold=max_seq_length * max_batch_size,
        params_dtype=torch.bfloat16,
        padded_vocab_size=tokenizer.vocab_size,
    )

    inference_context = StaticInferenceContext(
        max_batch_size=max_batch_size,
        max_sequence_length=max_seq_length,
    )
    inference_context.materialize_only_last_token_logits = False

    # Create the inference wrapper
    inference_wrapper = EdenModelInferenceWrapper(
        model=model,
        inference_wrapper_config=inference_wrapper_config,
        inference_context=inference_context,
    )

    # Create the text generation controller and inference engine.
    # Eden uses the static engine with legacy=False for paged KV cache and
    # built-in chunked prefill.
    text_generation_controller = TextGenerationController(
        inference_wrapped_model=inference_wrapper,
        tokenizer=tokenizer,
    )

    inference_engine = StaticInferenceEngine(
        text_generation_controller=text_generation_controller,
        max_batch_size=max_batch_size,
        random_seed=random_seed,
        legacy=False,
    )

    return EdenInferenceComponents(
        inference_engine=inference_engine,
        tokenizer=tokenizer,
        inference_wrapper=inference_wrapper,
        inference_context=inference_context,
        model=model,
    )


def generate(
    components: EdenInferenceComponents,
    prompts: List[str],
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    return_log_probs: bool = False,
) -> List[Any]:
    """Generate text using the Eden inference engine.

    Args:
        components: Inference components from setup_inference_engine.
        prompts: List of prompt strings to generate from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (0 = disabled, 1 = greedy).
        top_p: Nucleus sampling parameter (0 = disabled).
        return_log_probs: Whether to return log probabilities.

    Returns:
        List of inference result objects (InferenceRequest or
        DynamicInferenceRequestRecord depending on the engine backend).

    Example:
        >>> components = setup_inference_engine(ckpt_dir)
        >>> results = generate(components, ["ATCGATCG"], max_new_tokens=50, top_k=1)
        >>> print(_unwrap_result(results[0]).generated_text)
    """
    # Reset inference context before generation
    components.inference_context.reset()

    sampling_params = SamplingParams(
        temperature=temperature,
        top_k=max(0, top_k),
        top_p=top_p if top_p > 0 else 0.0,
        num_tokens_to_generate=max_new_tokens,
        return_log_probs=return_log_probs,
    )

    results = components.inference_engine.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Reset context after generation
    components.inference_context.reset()

    return results


# =============================================================================
# JSONL I/O Helpers
# =============================================================================


def _read_prompts_jsonl(path: Path) -> List[Dict[str, str]]:
    """Read prompts from a JSONL file.

    Each line must be a JSON object with at least a ``"prompt"`` field.
    An optional ``"id"`` field is echoed in the output; when absent it is
    auto-assigned from the line index.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of dicts, each with ``"id"`` and ``"prompt"`` keys.
    """
    entries: List[Dict[str, str]] = []
    with open(path) as f:
        for idx, raw_line in enumerate(f):
            stripped = raw_line.strip()
            if not stripped:
                continue
            obj = json.loads(stripped)
            if "prompt" not in obj:
                raise ValueError(f"Line {idx} in {path} is missing required 'prompt' field: {stripped}")
            entries.append({"id": str(obj.get("id", idx)), "prompt": obj["prompt"]})
    return entries


def _unwrap_result(result: Any) -> Any:
    """Unwrap a DynamicInferenceRequestRecord to its inner request if needed."""
    if hasattr(result, "requests"):
        return result.requests[-1]
    return result


def _result_to_jsonl_record(
    *,
    request_id: str,
    prompt: str,
    result: Any,
    max_new_tokens: int,
    return_log_probs: bool = False,
) -> Dict[str, Any]:
    """Convert an inference result into a JSONL-serialisable dict.

    Handles both legacy ``InferenceRequest`` objects and the newer
    ``DynamicInferenceRequestRecord`` wrappers returned by the dynamic engine.

    Output follows OpenAI Completions conventions where practical:
    ``id``, ``prompt``, ``completion``, ``finish_reason``, ``usage``, and
    optionally ``logprobs``.

    Args:
        request_id: User-supplied or auto-generated identifier.
        prompt: The original prompt text.
        result: Completed inference result from the engine.
        max_new_tokens: Configured generation limit (used to infer finish_reason).
        return_log_probs: Whether log-probs were requested.

    Returns:
        Dict ready for ``json.dumps``.
    """
    result = _unwrap_result(result)
    generated_text = result.generated_text or ""
    generated_length = result.generated_length or 0
    prompt_tokens_count = len(result.prompt_tokens) if result.prompt_tokens is not None else 0

    finish_reason = "length" if generated_length >= max_new_tokens else "stop"

    record: Dict[str, Any] = {
        "id": request_id,
        "prompt": prompt,
        "completion": generated_text,
        "finish_reason": finish_reason,
        "usage": {
            "prompt_tokens": prompt_tokens_count,
            "completion_tokens": generated_length,
            "total_tokens": prompt_tokens_count + generated_length,
        },
    }

    if return_log_probs and result.generated_log_probs is not None:
        log_probs = result.generated_log_probs
        if hasattr(log_probs, "tolist"):
            log_probs = log_probs.tolist()
        record["logprobs"] = {"completion_logprobs": log_probs}

    return record


# =============================================================================
# CLI: Full Inference Workflow
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Eden inference.

    Returns:
        Parsed arguments namespace
    """
    default_prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )

    ap = argparse.ArgumentParser(
        description="Generate text with Eden models using MCore inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        required=True,
        help="Path to MBridge checkpoint directory",
    )

    # Generation arguments
    ap.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt text for generation (ignored when --prompt-file is given)",
    )
    ap.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help='JSONL file with one {"id": "...", "prompt": "..."} object per line. '
        "The 'id' field is optional and will be auto-assigned if omitted. "
        "Overrides --prompt.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate")
    ap.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    ap.add_argument("--top-k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    ap.add_argument("--top-p", type=float, default=0.0, help="Top-p nucleus sampling (0 = disabled)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument(
        "--return-log-probs",
        action="store_true",
        default=False,
        help="Include per-token log probabilities in JSONL output",
    )

    # Parallelism arguments
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism")
    ap.add_argument("--pipeline-model-parallel-size", type=int, default=1, help="Pipeline parallelism")
    ap.add_argument("--context-parallel-size", type=int, default=1, help="Context parallelism")

    # Output arguments
    ap.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Save results as JSONL (one result object per line)",
    )

    # Precision arguments
    ap.add_argument("--mixed-precision-recipe", type=str, default=None, help="Override precision recipe")

    # Model arguments
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Max sequence length. When omitted, resolved as: "
        "EDEN_MAX_SEQ_LEN env var > auto-detected from GPU memory and model size.",
    )
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for inference. The inference engine pre-allocates GPU memory "
        "proportional to this value (KV caches, attention masks, internal buffers). "
        "For large models (e.g. 40b), only batch_size=1 may fit in memory.",
    )

    return ap.parse_args()


def infer(
    prompts: List[Dict[str, str]],
    ckpt_dir: Path,
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    seed: Optional[int] = None,
    return_log_probs: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    output_file: Optional[Path] = None,
    mixed_precision_recipe: Optional[str] = None,
    max_seq_length: int = 8192,
    max_batch_size: int = 1,
) -> List[Dict[str, Any]]:
    """Run autoregressive text generation with Eden using MCore inference.

    This is the main CLI entry point that sets up everything and runs inference.
    For programmatic usage, prefer setup_inference_engine + generate.

    Args:
        prompts: List of dicts, each with ``"id"`` and ``"prompt"`` keys.
        ckpt_dir: Path to MBridge checkpoint directory.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (0 = disabled).
        top_p: Nucleus sampling parameter (0 = disabled).
        seed: Random seed for reproducibility.
        return_log_probs: Whether to return per-token log probabilities.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        output_file: Optional path to save results as JSONL.
        mixed_precision_recipe: Override mixed precision recipe.
        max_seq_length: Maximum sequence length.
        max_batch_size: Maximum batch size for inference. The inference engine pre-allocates
            GPU memory proportional to this value. For large models, only 1 may fit.

    Returns:
        List of JSONL-serialisable result dicts.
    """
    random_seed = seed or 1234

    _prune_caches()
    torch.cuda.reset_peak_memory_stats()

    components = setup_inference_engine(
        ckpt_dir=ckpt_dir,
        max_seq_length=max_seq_length,
        max_batch_size=max_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        mixed_precision_recipe=mixed_precision_recipe,
        random_seed=random_seed,
    )

    mem_after_setup_gb = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info(f"[MEMORY] After model setup: peak={mem_after_setup_gb:.3f} GB")

    all_records: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    t_generate_start = time.perf_counter()

    for batch_start in range(0, len(prompts), max_batch_size):
        batch = prompts[batch_start : batch_start + max_batch_size]
        batch_prompts = [entry["prompt"] for entry in batch]
        batch_idx = batch_start // max_batch_size + 1

        logger.info(f"Generating batch {batch_idx} ({len(batch)} prompt(s))...")

        t_batch_start = time.perf_counter()
        results = generate(
            components,
            prompts=batch_prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_log_probs=return_log_probs,
        )
        t_batch_elapsed = time.perf_counter() - t_batch_start

        batch_completion_tokens = 0
        for entry, result in zip(batch, results):
            record = _result_to_jsonl_record(
                request_id=entry["id"],
                prompt=entry["prompt"],
                result=result,
                max_new_tokens=max_new_tokens,
                return_log_probs=return_log_probs,
            )
            all_records.append(record)
            batch_completion_tokens += record["usage"]["completion_tokens"]
            total_prompt_tokens += record["usage"]["prompt_tokens"]
            total_completion_tokens += record["usage"]["completion_tokens"]

        batch_tok_per_sec = batch_completion_tokens / t_batch_elapsed if t_batch_elapsed > 0 else 0
        logger.info(
            f"[PERF] Batch {batch_idx}: {batch_completion_tokens} tokens in "
            f"{t_batch_elapsed:.2f}s ({batch_tok_per_sec:.1f} completion tok/s)"
        )

    t_generate_elapsed = time.perf_counter() - t_generate_start
    total_tok_per_sec = total_completion_tokens / t_generate_elapsed if t_generate_elapsed > 0 else 0

    mem_after_generate_gb = torch.cuda.max_memory_allocated() / (1024**3)
    logger.info(
        f"[MEMORY] After generation: peak={mem_after_generate_gb:.3f} GB "
        f"(setup={mem_after_setup_gb:.3f} GB, generation delta="
        f"{mem_after_generate_gb - mem_after_setup_gb:.3f} GB)"
    )
    logger.info(
        f"[PERF] Total: {total_prompt_tokens} prompt tokens + {total_completion_tokens} "
        f"completion tokens in {t_generate_elapsed:.2f}s "
        f"({total_tok_per_sec:.1f} completion tok/s)"
    )

    is_rank_zero = parallel_state.get_data_parallel_rank() == 0

    if is_rank_zero:
        for record in all_records:
            print(
                f"\n=== [{record['id']}] Generated Text ===\n{record['completion']}\n",
                file=sys.stdout,
            )

        if output_file is not None:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                for record in all_records:
                    f.write(json.dumps(record) + "\n")
            logger.info(f"Saved {len(all_records)} result(s) to: {output_file}")

    logger.info("Inference complete!")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    return all_records


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for Eden text generation."""
    args = parse_args()

    # --- Resolve settings: CLI arg > env var > auto-detected default ---
    max_seq_length = _resolve_int(args.max_seq_length, "EDEN_MAX_SEQ_LEN", _detect_max_seq_length(args.ckpt_dir))

    if args.prompt_file is not None:
        prompts = _read_prompts_jsonl(args.prompt_file)
    else:
        prompts = [{"id": "0", "prompt": args.prompt}]

    infer(
        prompts=prompts,
        ckpt_dir=args.ckpt_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        return_log_probs=args.return_log_probs,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        output_file=args.output_file,
        mixed_precision_recipe=args.mixed_precision_recipe,
        max_seq_length=max_seq_length,
        max_batch_size=args.max_batch_size,
    )


if __name__ == "__main__":
    main()
