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

r"""Text generation (inference) workflow for Evo2 using Megatron Core.

This module provides autoregressive text generation for Evo2 models through the native mcore
dynamic-inference engine. It drives paged-KV attention with Hyena recurrent state packed into
mcore's two Mamba state slots. ``flash_decode`` and sequence parallelism are turned off
automatically, and each prompt is decoded through an Evo2-specific dynamic context that keeps a
single active request as one row.

Usage (CLI, single prompt):
    torchrun --nproc_per_node 1 -m bionemo.evo2.run.infer \
        --ckpt-dir /path/to/mbridge/checkpoint \
        --prompt "|d__Bacteria;p__Pseudomonadota|" \
        --max-new-tokens 100 \
        --output-file results.jsonl

Usage (CLI, batch from JSONL file):
    torchrun --nproc_per_node 1 -m bionemo.evo2.run.infer \
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
    from bionemo.evo2.run.infer import setup_inference_engine, generate

    # Setup engine (loads model, creates inference components)
    components = setup_inference_engine(ckpt_dir)

    # Generate text
    results = generate(components, prompts=["ATCGATCG"], max_new_tokens=100)
"""

import argparse
import contextlib
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
from megatron.bridge.training.checkpointing import (
    _generate_model_state_dict,
    _load_model_weights_from_checkpoint,
    apply_peft_adapter_filter_to_state_dict,
)
from megatron.bridge.training.config import DistributedInitConfig, RNGConfig
from megatron.bridge.training.mixed_precision import get_mixed_precision_config


try:
    from megatron.bridge.training.tokenizers.tokenizer import _HuggingFaceTokenizer
except ImportError:
    from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import (
        HuggingFaceTokenizer as _HuggingFaceTokenizer,
    )
from megatron.bridge.training.utils.checkpoint_utils import (
    file_exists,
    get_checkpoint_run_config_filename,
    read_run_config,
)
from megatron.bridge.utils.common_utils import get_world_size_safe
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.module import Float16Module

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH
from bionemo.evo2.models.evo2_provider import (
    bind_hyena_packed_views_to_dynamic_context,
    build_evo2_mamba_inference_state_config,
    compute_evo2_paged_kv_buffer_size_gb,
    make_evo2_dynamic_inference_context_cls,
)
from bionemo.evo2.models.megatron.hyena.subquadratic_safety import ensure_subquadratic_ops_supported
from bionemo.evo2.run.predict import initialize_inference_distributed, resolve_checkpoint_path


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _register_bionemo_target_prefix() -> None:
    try:
        from megatron.bridge.utils.instantiate_utils import register_allowed_target_prefix

        register_allowed_target_prefix("bionemo.")
    except ImportError:
        pass


def _adapt_tokenizer_for_generation(tokenizer: Any) -> Any:
    """Normalize tokenizer method names used by the dynamic-engine generation path.

    Different mcore tokenizer backends expose ``tokenize``/``detokenize`` (HF-style) or
    ``text_to_ids``/``ids_to_text``; :func:`_generate_native_dynamic` calls the former, so
    alias them when only the latter exist.
    """
    if not hasattr(tokenizer, "tokenize") and hasattr(tokenizer, "text_to_ids"):
        tokenizer.tokenize = tokenizer.text_to_ids
    if not hasattr(tokenizer, "detokenize") and hasattr(tokenizer, "ids_to_text"):
        tokenizer.detokenize = tokenizer.ids_to_text
    if not hasattr(tokenizer, "bos") and hasattr(tokenizer, "bos_id"):
        tokenizer.bos = tokenizer.bos_id
    return tokenizer


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
    ``EVO2_MAX_SEQ_LEN`` environment variable or the ``--max-seq-length`` CLI flag.

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
# Inference Components Container
# =============================================================================


@dataclass
class Evo2InferenceComponents:
    """Container for Evo2 inference components.

    This dataclass holds everything needed for text generation, making it easy to pass around
    and reuse. Generation is driven through the native mcore dynamic-inference engine (paged KV +
    Hyena state packed into mcore's Mamba slots); see :class:`Evo2NativeDynamicComponents` and
    :func:`_generate_native_dynamic`.
    """

    tokenizer: _HuggingFaceTokenizer
    model: torch.nn.Module
    native_dynamic: "Evo2NativeDynamicComponents"


@dataclass
class Evo2NativeDynamicComponents:
    """Components for driving Evo2 generation on the native mcore dynamic engine.

    Holds the dynamic-context subclass, Evo2 Mamba state config, and standalone
    HyenaModel used by text generation. The per-request lifecycle
    (``add_request`` -> :func:`bind_hyena_packed_views_to_dynamic_context` ->
    ``initialize_attention_state`` -> forward -> sample -> ``update_requests``) runs
    in :func:`_generate_native_dynamic`.
    """

    ctx_cls: type
    mamba_state_config: Any
    forward_model: torch.nn.Module
    hyena_model: torch.nn.Module
    max_seq_length: int
    cuda_graphs_enabled: bool
    cuda_graph_manager_count: int


# =============================================================================
# Native dynamic-inference engine wiring
# =============================================================================


def _unwrap_hyena_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying HyenaModel from a (possibly Float16Module-wrapped) model.

    The native-dynamic helpers (state-shape probing, view binding) need the real
    ``HyenaModel`` whose ``decoder`` exposes ``hyena_state_shapes_per_request`` /
    ``mamba_state_shapes_per_request`` and whose layer ``id(module)`` values match the
    modules the Hyena ops touch at runtime.
    """
    inner = getattr(model, "module", model)
    return inner


def _setup_native_dynamic_components(
    *,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    max_seq_length: int,
    cuda_graphs_enabled: bool,
) -> Evo2NativeDynamicComponents:
    """Prepare the standalone HyenaModel to decode on an Evo2 dynamic context.

    This disables sequence parallelism for the Evo2 model, builds the exact-rounding
    ``DynamicInferenceContext`` subclass, and creates the Mamba state config that lets mcore
    allocate Hyena recurrent state in its dynamic state buffers. Per-prompt contexts are built
    lazily in :func:`_generate_native_dynamic` so each request can be sized to its prompt and
    requested generation length.
    """
    rank = int(os.environ.get("RANK", "0"))
    hyena_model = _unwrap_hyena_model(model)

    # Sequence-parallel off keeps the context's single active request as one row.
    if getattr(hyena_model.config, "sequence_parallel", False):
        try:
            from megatron.core.transformer.utils import (
                set_model_to_sequence_parallel,  # lazy: heavy mcore import
            )

            set_model_to_sequence_parallel(hyena_model, False)
        except Exception as exc:  # pragma: no cover - defensive
            if rank == 0:
                logger.warning("[evo2-native] set_model_to_sequence_parallel failed: %r", exc)
        hyena_model.config.sequence_parallel = False

    ctx_cls = make_evo2_dynamic_inference_context_cls()
    mamba_cfg = build_evo2_mamba_inference_state_config(raw_model)
    cuda_graph_manager_count = sum(1 for module in hyena_model.modules() if hasattr(module, "cudagraph_manager"))
    if rank == 0:
        logger.info(
            "[evo2-native] standalone evo2 prepared for native dynamic decode "
            "(SP off, cuda_graphs=%s, graph_managers=%d).",
            cuda_graphs_enabled,
            cuda_graph_manager_count,
        )
    return Evo2NativeDynamicComponents(
        ctx_cls=ctx_cls,
        mamba_state_config=mamba_cfg,
        forward_model=model,
        hyena_model=hyena_model,
        max_seq_length=max_seq_length,
        cuda_graphs_enabled=cuda_graphs_enabled,
        cuda_graph_manager_count=cuda_graph_manager_count,
    )


def _configure_native_dynamic_cuda_graphs(model_provider: Any, *, rank: int) -> bool:
    """Enable mcore local CUDA graphs for Evo2 dynamic inference when supported.

    This mirrors Megatron's ``cuda_graph_impl=local`` setup, but applies it directly to the
    provider loaded from the checkpoint because this recipe does not use Megatron's global arg
    parser. Empty ``cuda_graph_scope`` means per-layer local graph capture for every graphable
    layer; Evo2's HyenaLayer follows the same convention as mcore's MambaLayer.
    """
    if not hasattr(model_provider, "cuda_graph_impl"):
        if rank == 0:
            logger.warning("[evo2-native-cg] model provider has no cuda_graph_impl; CUDA graphs disabled")
        return False

    model_provider.cuda_graph_impl = "local"
    model_provider.cuda_graph_scope = []
    os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")
    if rank == 0:
        logger.info("[evo2-native-cg] enabled mcore local per-layer CUDA graphs for dynamic decode")
    return True


def _seed_cudagraph_safe_rng(rng_config: Any) -> None:
    """Re-seed Megatron's CUDA RNG tracker in graph-safe mode before graphable layers build."""
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    seed = int(rng_config.seed) + (100 * parallel_state.get_pipeline_model_parallel_rank())
    if getattr(rng_config, "data_parallel_random_init", False):
        seed += 10 * parallel_state.get_data_parallel_rank()
    model_parallel_cuda_manual_seed(
        seed,
        getattr(rng_config, "te_rng_tracker", False),
        getattr(rng_config, "inference_rng_tracker", False),
        use_cudagraphable_rng=True,
        force_reset_rng=True,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("[evo2-native-cg] re-seeded graph-safe CUDA RNG tracker (seed=%d)", seed)


def _teardown_distributed_for_inference() -> None:
    """Release Megatron and torch distributed state for non-forced inference exits."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if parallel_state.model_parallel_is_initialized():
        parallel_state.destroy_model_parallel()
    if dist.is_initialized():
        dist.destroy_process_group()


def _force_exit_after_cuda_graph_inference() -> None:
    """Bypass torchrun/NCCL atexit teardown after CUDA graph inference."""
    logger.info("[evo2-native-cg] forcing process exit after CUDA graph inference")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


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
    vortex_style_fp8: bool = False,
    random_seed: int = 1234,
    use_subquadratic_ops: bool = False,
) -> Evo2InferenceComponents:
    """Setup the Evo2 native dynamic-inference engine and related components.

    Loads the model, wires it onto the native mcore dynamic-inference engine (paged-KV attention +
    Hyena recurrent state packed into mcore's two Mamba slots), and returns everything needed for
    text generation. ``flash_decode`` and sequence-parallel are turned off automatically (both
    required by the dynamic path).

    Args:
        ckpt_dir: Path to MBridge checkpoint directory.
        max_seq_length: Maximum sequence length for generation.
        max_batch_size: Maximum batch size for inference.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        mixed_precision_recipe: Override mixed precision recipe.
        vortex_style_fp8: Use vortex-style FP8 (applies FP8 only to projection layers).
            Needed for FP8-sensitive checkpoints from original evo2 training (1b, 40b).
        random_seed: Random seed for reproducibility.
        use_subquadratic_ops: Use fused subquadratic-ops kernels (b2b causal
            conv1d in prefill, fft_causal_conv1d / causal_conv1d in
            parallel_fir).

    Returns:
        Evo2InferenceComponents containing all inference components.

    Example:
        >>> components = setup_inference_engine(Path("/path/to/checkpoint"), max_batch_size=4)
        >>> results = generate(components, prompts=["ATCG", "GCTA"], max_new_tokens=100)
    """
    # -------------------------------------------------------------------------
    # Step 1: Load configuration from checkpoint
    # -------------------------------------------------------------------------
    _register_bionemo_target_prefix()

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

    # The native dynamic engine drives paged flash-attn-varlen itself and asserts NOT
    # static-batching, so flash_decode (which asserts static batching, attention.py) MUST be off.
    model_provider.flash_decode = False
    model_provider.use_subquadratic_ops = use_subquadratic_ops
    cuda_graphs_enabled = _configure_native_dynamic_cuda_graphs(model_provider, rank=int(os.environ.get("RANK", "0")))
    if cuda_graphs_enabled and getattr(model_provider, "recompute_granularity", None):
        logger.info("Disabling activation recompute for inference CUDA graphs")
        model_provider.recompute_granularity = None
    if getattr(model_provider, "fp32_residual_connection", False):
        logger.info("Disabling fp32_residual_connection for inference to keep TE activations in params_dtype")
        model_provider.fp32_residual_connection = False

    if vortex_style_fp8:
        model_provider.vortex_style_fp8 = True

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
    tokenizer = _adapt_tokenizer_for_generation(tokenizer)

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
    if cuda_graphs_enabled and torch.cuda.is_available():
        _seed_cudagraph_safe_rng(rng_config)
    if use_subquadratic_ops:
        ensure_subquadratic_ops_supported()

    # -------------------------------------------------------------------------
    # Step 5: Create model and load weights
    # -------------------------------------------------------------------------
    logger.info("Creating model...")
    model_provider.finalize()

    raw_model = model_provider.provide().eval().cuda()

    # A LoRA finetune checkpoint only contains adapter tensors; the base weights live in
    # run_config["checkpoint"]["pretrained_checkpoint"]. Detect via the top-level `peft:`
    # section (same signal `peft_pre_wrap_hook` uses during training).
    peft_node = run_config.get("peft")
    if peft_node is not None:
        # pretrained_checkpoint may point at a training-output parent containing iter_*; resolve.
        resolved_pretrained_dir = resolve_checkpoint_path(Path(run_config["checkpoint"]["pretrained_checkpoint"]))
        logger.info(f"PEFT checkpoint detected. Loading base weights from: {resolved_pretrained_dir}")
        _load_model_weights_from_checkpoint(
            checkpoint_path=str(resolved_pretrained_dir),
            model=[raw_model],
            dist_ckpt_strictness="ignore_all",
        )

        logger.info("Applying PEFT adapter structure to base model")
        peft_cfg = instantiate(peft_node)
        raw_model = peft_cfg(raw_model, training=False)

        logger.info(f"Loading adapter weights from: {resolved_ckpt_dir}")
        sharded_sd = apply_peft_adapter_filter_to_state_dict(_generate_model_state_dict([raw_model], {}), peft_cfg)
        loaded = dist_checkpointing.load(sharded_sd, str(resolved_ckpt_dir), strict="ignore_all")
        raw_model.load_state_dict(loaded["model"], strict=False)
    else:
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
    # Step 6: wire onto the native mcore dynamic-inference engine.
    # -------------------------------------------------------------------------
    # Wire the model onto mcore dynamic inference: paged-KV attention plus Hyena recurrent
    # state packed into mcore's two Mamba slots. The per-request lifecycle runs in
    # _generate_native_dynamic. flash_decode is already off above.
    native_components = _setup_native_dynamic_components(
        model=model,
        raw_model=raw_model,
        max_seq_length=max_seq_length,
        cuda_graphs_enabled=cuda_graphs_enabled,
    )
    return Evo2InferenceComponents(
        tokenizer=tokenizer,
        model=model,
        native_dynamic=native_components,
    )


def generate(
    components: Evo2InferenceComponents,
    prompts: List[str],
    *,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    return_log_probs: bool = False,
    enable_chunked_prefill: bool = False,
    inference_dynamic_batching_max_tokens: Optional[int] = None,
    inference_dynamic_batching_block_size: int = 256,
) -> List[Any]:
    """Generate text using the Evo2 native dynamic-inference engine.

    Drives generation through the native mcore dynamic-inference path (paged-KV attention +
    Hyena state packed into mcore's Mamba slots).

    Args:
        components: Inference components from setup_inference_engine.
        prompts: List of prompt strings to generate from.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (higher = more random).
        top_k: Top-k sampling parameter (0 = disabled, 1 = greedy).
        top_p: Nucleus sampling parameter (0 = disabled).
        return_log_probs: Whether to return log probabilities.
        enable_chunked_prefill: Split prompts across multiple prefill forwards when they exceed
            ``inference_dynamic_batching_max_tokens``. Disabled by default.
        inference_dynamic_batching_max_tokens: Optional dynamic-context per-step token budget.
            When set and chunking is disabled, each prompt must fit within this value.
        inference_dynamic_batching_block_size: KV-cache block size for the dynamic context. This is
            not the prefill chunk size.

    Returns:
        List of :class:`_NativeDynamicResult` objects (mirroring the
        ``generated_text`` / ``generated_length`` / ``prompt_tokens`` fields downstream reads).

    Example:
        >>> components = setup_inference_engine(ckpt_dir)
        >>> results = generate(components, ["ATCGATCG"], max_new_tokens=50, top_k=1)
        >>> print(_unwrap_result(results[0]).generated_text)
    """
    return _generate_native_dynamic(
        components,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=return_log_probs,
        enable_chunked_prefill=enable_chunked_prefill,
        inference_dynamic_batching_max_tokens=inference_dynamic_batching_max_tokens,
        inference_dynamic_batching_block_size=inference_dynamic_batching_block_size,
    )


@dataclass
class _NativeDynamicResult:
    """Minimal result object mirroring mcore's ``InferenceRequest`` fields used downstream.

    Only the attributes :func:`_result_to_jsonl_record` reads are populated:
    ``generated_text``, ``generated_length``, ``prompt_tokens``, ``generated_log_probs``.
    """

    generated_text: str
    generated_length: int
    prompt_tokens: List[int]
    generated_log_probs: Optional[List[float]] = None


def _sample_from_logits(
    last_token_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: torch.Generator,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Sample next-token ids from logits (greedy / top-k / top-p / temperature).

    Self-contained mcore-compatible sampler for the native dynamic path. Greedy
    (``top_k == 1``) returns the argmax; otherwise this applies standard top-k / top-p
    filtering followed by ``torch.multinomial`` with the provided RNG.

    Args:
        last_token_logits: Logits of shape ``[batch_size, vocab_size]``.
        temperature: Temperature scaling factor (applied only on the non-greedy path).
        top_k: Top-k filtering value (0 = disabled, 1 = greedy argmax).
        top_p: Top-p (nucleus) filtering value (0.0 = disabled).
        generator: RNG used by ``torch.multinomial``.
        vocab_size: When provided, clamps sampled ids to ``[0, vocab_size - 1]``.

    Returns:
        Sampled token ids of shape ``[batch_size]``.
    """
    assert isinstance(top_p, float)
    assert isinstance(top_k, int)
    assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"
    assert top_p <= 1.0, "top-p should be in (0,1]"

    def _modify_for_top_k(logits, k):
        filter_ = logits < torch.topk(logits, k)[0][..., -1, None]
        logits.masked_fill_(filter_, float("-Inf"))

    def _modify_for_top_p(logits, p):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        filter_ = cumulative_probs > p
        # Clone: filter_[:, 1:] and filter_[:, :-1] overlap; without it each write corrupts the read.
        filter_[:, 1:] = filter_[:, :-1].clone()
        filter_[..., 0] = 0
        filter_ = filter_.scatter(1, sorted_indices, filter_)
        logits.masked_fill_(filter_, float("-Inf"))

    if top_k == 1:
        return torch.argmax(last_token_logits, dim=-1)

    last_token_logits = last_token_logits.clone()  # .div_/.masked_fill_ below are in-place
    if temperature != 1.0:
        last_token_logits.div_(temperature)
    if top_k > 1:
        assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
        if vocab_size:
            assert top_k < vocab_size, "top-k is larger than vocab size."
        _modify_for_top_k(last_token_logits, top_k)
    elif top_p > 0.0:
        _modify_for_top_p(last_token_logits, top_p)

    probabilities = last_token_logits.softmax(dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).view(-1)
    if vocab_size:
        sampled = torch.clamp(sampled, min=0, max=(vocab_size - 1))
    return sampled


def _generate_native_dynamic(
    components: Evo2InferenceComponents,
    prompts: List[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    return_log_probs: bool,
    enable_chunked_prefill: bool,
    inference_dynamic_batching_max_tokens: Optional[int],
    inference_dynamic_batching_block_size: int,
) -> List[_NativeDynamicResult]:
    """Drive standalone Evo2 (text→DNA) generation through the native mcore dynamic engine.

    One mcore ``DynamicInferenceContext`` per prompt, driven through the request lifecycle:
    ``add_request`` -> :func:`bind_hyena_packed_views_to_dynamic_context` ->
    ``initialize_attention_state`` -> model forward -> sample on the LM-head logits ->
    ``update_requests``. Standalone Evo2 has its own output layer (``post_process=True``), so
    logits are read directly from the forward pass.

    By default, each prompt is prefilled in a single forward pass. When
    ``enable_chunked_prefill`` is set, prompts that exceed the dynamic context's ``max_tokens``
    budget are split across multiple prefill forwards, matching mcore's dynamic-inference
    scheduling behavior. The KV-cache ``block_size_tokens`` controls paged-KV granularity, not the
    prefill chunk length.
    """
    # lazy: heavy mcore imports — pull the full dynamic-inference stack only when generating.
    from megatron.core.inference.config import InferenceConfig
    from megatron.core.inference.contexts.dynamic_context import (
        BlockOverflowError,
        MaxSequenceLengthOverflowError,
        TokenOverflowError,
    )
    from megatron.core.inference.inference_request import DynamicInferenceRequest

    nd = components.native_dynamic
    forward_model = nd.forward_model
    hyena_model = nd.hyena_model
    tokenizer = components.tokenizer
    device = next(hyena_model.parameters()).device
    rank = int(os.environ.get("RANK", "0"))

    # Greedy unless temperature/top-k/top-p say otherwise. The stock sampler asserts NOT
    # (top_k>0 AND top_p>0); honor top-k first for compatibility with SamplingParams.
    eff_top_k = max(0, int(top_k))
    eff_top_p = float(top_p) if (top_p and top_p > 0 and eff_top_k == 0) else 0.0
    sampling_rng = torch.Generator(device=device)
    sampling_rng.manual_seed(int(getattr(tokenizer, "_evo2_seed", 0)) or 1234)

    results: List[_NativeDynamicResult] = []
    for prompt in prompts:
        prompt_token_ids = list(tokenizer.tokenize(prompt))
        n_prompt = len(prompt_token_ids)
        # Size the context to cover prompt + generation (+ small headroom). max_requests is kept
        # tp-divisible (max(tp,1)); the per-context exact rounder decodes the single request as ONE
        # row regardless. buffer_size_gb is auto-sized for the prompt plus generated tokens.
        tp = int(getattr(hyena_model.config, "tensor_model_parallel_size", 1) or 1)
        max_requests = max(tp, 1)
        msl = min(nd.max_seq_length, n_prompt + max_new_tokens + 8)
        if msl < n_prompt + 1:
            msl = n_prompt + max_new_tokens + 8
        block_size_tokens = int(inference_dynamic_batching_block_size)
        if block_size_tokens <= 0:
            raise ValueError(f"inference_dynamic_batching_block_size must be positive, got {block_size_tokens}")
        max_tokens = inference_dynamic_batching_max_tokens
        if max_tokens is not None:
            max_tokens = int(max_tokens)
            if max_tokens <= 0:
                raise ValueError(f"inference_dynamic_batching_max_tokens must be positive, got {max_tokens}")
            if n_prompt > max_tokens and not enable_chunked_prefill:
                raise ValueError(
                    f"Prompt has {n_prompt} tokens but inference_dynamic_batching_max_tokens={max_tokens}. "
                    "Increase --inference-dynamic-batching-max-tokens or pass --enable-chunked-prefill."
                )
        buf_gb = compute_evo2_paged_kv_buffer_size_gb(
            hyena_model.config,
            mamba_state_config=nd.mamba_state_config,
            max_sequence_length=msl,
            max_requests=max_requests,
            block_size_tokens=block_size_tokens,
            safety_blocks=2,
        )
        dyn_ctx = nd.ctx_cls(
            model_config=hyena_model.config,
            inference_config=InferenceConfig(
                max_sequence_length=msl,
                buffer_size_gb=buf_gb,
                mamba_inference_state_config=nd.mamba_state_config,
                max_requests=max_requests,
                max_tokens=max_tokens,
                block_size_tokens=block_size_tokens,
                unified_memory_level=0,
                enable_chunked_prefill=enable_chunked_prefill,
                num_cuda_graphs=1 if nd.cuda_graphs_enabled else None,
                use_cuda_graphs_for_non_decode_steps=False,
            ),
        )
        if n_prompt > dyn_ctx.max_tokens and not enable_chunked_prefill:
            raise ValueError(
                f"Prompt has {n_prompt} tokens but the dynamic context max token budget is {dyn_ctx.max_tokens}. "
                "Increase --inference-dynamic-batching-max-tokens or pass --enable-chunked-prefill."
            )
        dyn_ctx.materialize_only_last_token_logits = False
        dyn_ctx.initialize_all_tensors()

        generated_ids: List[int] = []
        generated_logprobs: List[float] = []

        def _forward_sample_update(*, count_generated: bool) -> bool:
            dyn_ctx.initialize_attention_state()
            input_ids, position_ids = dyn_ctx.current_input_and_position_ids()
            try:
                from megatron.core.inference.utils import InferenceMode

                inference_mode_context = InferenceMode.active()
            except ImportError:
                inference_mode_context = contextlib.nullcontext()
            with inference_mode_context:
                logits = forward_model(
                    input_ids,
                    position_ids,
                    None,
                    inference_context=dyn_ctx,
                    runtime_gather_output=True,
                )
            # HyenaModel returns [B, S, vocab]; last_token_logits expects [1, S, H] and
            # selects the per-request final position -> [num_requests, vocab]. Sample in fp32 so
            # stochastic filters and logprobs do not depend on the model activation dtype.
            last_logits = dyn_ctx.last_token_logits(logits).float()
            sampled = _sample_from_logits(
                last_logits,
                temperature=float(temperature),
                top_k=eff_top_k,
                top_p=eff_top_p,
                generator=sampling_rng,
                vocab_size=tokenizer.vocab_size,
            )
            if count_generated:
                next_tok_id = int(sampled[0].item())
                generated_ids.append(next_tok_id)
                if return_log_probs:
                    logprob = torch.log_softmax(last_logits[0].float(), dim=-1)[next_tok_id].item()
                    generated_logprobs.append(logprob)
            active_after_sample = torch.tensor(
                [not count_generated or len(generated_ids) < max_new_tokens], dtype=torch.bool, device=device
            )
            dyn_ctx.update_requests(active_after_sample, sampled.to(dtype=torch.int64, device=device))
            return bool(active_after_sample[0].item())

        try:
            with torch.inference_mode():
                req = DynamicInferenceRequest(
                    request_id=0,
                    prompt_tokens=torch.tensor(prompt_token_ids, dtype=torch.int64, device=device),
                    sampling_params=SamplingParams(num_tokens_to_generate=max_new_tokens, termination_id=-1),
                )
                if max_new_tokens > 0:
                    first_chunk = True
                    while req.remaining_prompt_length > 0:
                        chunk_len = req.remaining_prompt_length
                        is_partial_chunk = False
                        if enable_chunked_prefill and req.remaining_prompt_length > dyn_ctx.max_tokens:
                            chunk_len = dyn_ctx.max_tokens
                            final_chunk_len = req.remaining_prompt_length - chunk_len
                            if final_chunk_len == 1:
                                if chunk_len <= 1:
                                    raise ValueError(
                                        "Chunked prefill cannot split this prompt without leaving a one-token "
                                        "final prefill chunk. Increase --inference-dynamic-batching-max-tokens."
                                    )
                                chunk_len -= 1
                            is_partial_chunk = True
                        dyn_ctx.chunked_prefill_request_id = req.request_id if is_partial_chunk else -1
                        dyn_ctx.add_request(req, prefill_chunk_length=chunk_len)
                        if first_chunk:
                            slot = int(dyn_ctx.mamba_metadata.request_to_mamba_state_idx[0].item())
                            bind_hyena_packed_views_to_dynamic_context(hyena_model, dyn_ctx, request_slot=slot)
                            first_chunk = False
                        if rank == 0:
                            logger.info(
                                "[evo2-native] prompt prefill: chunk=%d/%d tokens, remaining=%d",
                                chunk_len,
                                n_prompt,
                                req.remaining_prompt_length - chunk_len,
                            )
                        _forward_sample_update(count_generated=not is_partial_chunk)
                        if not is_partial_chunk:
                            req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                            break
                        req.remaining_prompt_tokens = req.remaining_prompt_tokens[chunk_len:]
                        req.finished_chunk_token_count += chunk_len

                    while len(generated_ids) < max_new_tokens and dyn_ctx.has_unfinished_requests():
                        _forward_sample_update(count_generated=True)
        except (BlockOverflowError, TokenOverflowError, MaxSequenceLengthOverflowError) as exc:
            if rank == 0:
                logger.warning(
                    "[evo2-native] generation stopped early at %d tokens (context overflow: %s). "
                    "Increase --max-seq-length to cover prompt + max_new_tokens.",
                    len(generated_ids),
                    type(exc).__name__,
                )
        finally:
            dyn_ctx.reset()

        generated_text = tokenizer.detokenize(generated_ids) if generated_ids else ""
        results.append(
            _NativeDynamicResult(
                generated_text=generated_text,
                generated_length=len(generated_ids),
                prompt_tokens=prompt_token_ids,
                generated_log_probs=generated_logprobs if return_log_probs else None,
            )
        )

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
    """Parse command-line arguments for Evo2 inference.

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
        description="Generate text with Evo2 models using MCore inference",
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
    ap.add_argument(
        "--vortex-style-fp8",
        action="store_true",
        help="Use vortex-style FP8 (applies FP8 only to projection layers)",
    )

    # Model arguments
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Max sequence length. When omitted, resolved as: "
        "EVO2_MAX_SEQ_LEN env var > auto-detected from GPU memory and model size.",
    )
    ap.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help="Maximum batch size for inference. The inference engine pre-allocates GPU memory "
        "proportional to this value (KV caches, attention masks, internal buffers). "
        "For large models (e.g. 40b), only batch_size=1 may fit in memory.",
    )
    ap.add_argument(
        "--use-subquadratic-ops",
        action="store_true",
        default=False,
        help="Use fused subquadratic-ops CUDA kernels (b2b causal conv1d in prefill, "
        "fft_causal_conv1d / causal_conv1d in parallel_fir). Speeds up prompt processing "
        "but has no effect on per-token decode throughput.",
    )
    ap.add_argument(
        "--enable-chunked-prefill",
        action="store_true",
        default=False,
        help="Enable mcore-style chunked prefill when prompts exceed the dynamic context max-token budget.",
    )
    ap.add_argument(
        "--inference-dynamic-batching-max-tokens",
        type=int,
        default=None,
        help="Dynamic context per-step token budget. When set and --enable-chunked-prefill is not "
        "passed, each prompt must fit within this many tokens.",
    )
    ap.add_argument(
        "--inference-dynamic-batching-block-size",
        type=int,
        default=256,
        help="Paged-KV block size for dynamic inference. This is not the prefill chunk length.",
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
    vortex_style_fp8: bool = False,
    max_seq_length: int = 8192,
    max_batch_size: int = 1,
    use_subquadratic_ops: bool = False,
    enable_chunked_prefill: bool = False,
    inference_dynamic_batching_max_tokens: Optional[int] = None,
    inference_dynamic_batching_block_size: int = 256,
    force_exit_on_completion: bool = False,
) -> List[Dict[str, Any]]:
    """Run autoregressive text generation with Evo2 using MCore inference.

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
        vortex_style_fp8: Use vortex-style FP8 (applies FP8 only to projection layers).
            Needed for FP8-sensitive checkpoints from original evo2 training (1b, 40b).
        max_seq_length: Maximum sequence length.
        max_batch_size: Maximum batch size for inference. The inference engine pre-allocates
            GPU memory proportional to this value. For large models, only 1 may fit.
        use_subquadratic_ops: Use fused subquadratic-ops kernels in the inference path.
        enable_chunked_prefill: Split prompts across multiple prefill forwards when needed.
        inference_dynamic_batching_max_tokens: Optional dynamic-context per-step token budget.
        inference_dynamic_batching_block_size: Paged-KV block size for dynamic inference.
        force_exit_on_completion: For CLI use, immediately exit after successful CUDA-graph
            inference to avoid torchrun/NCCL atexit hangs with captured collectives.

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
        vortex_style_fp8=vortex_style_fp8,
        random_seed=random_seed,
        use_subquadratic_ops=use_subquadratic_ops,
    )
    # Thread the resolved seed into the native sampler RNG (read off the tokenizer handle in
    # _generate_native_dynamic).
    components.tokenizer._evo2_seed = random_seed

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
            enable_chunked_prefill=enable_chunked_prefill,
            inference_dynamic_batching_max_tokens=inference_dynamic_batching_max_tokens,
            inference_dynamic_batching_block_size=inference_dynamic_batching_block_size,
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

    if force_exit_on_completion and components.native_dynamic.cuda_graphs_enabled:
        # Megatron's CUDA graph inference examples force-exit here as well: captured
        # collectives can otherwise leave torchrun waiting in NCCL atexit teardown.
        _force_exit_after_cuda_graph_inference()

    _teardown_distributed_for_inference()

    return all_records


# =============================================================================
# Entry Point
# =============================================================================


def main() -> None:
    """CLI entry point for Evo2 text generation."""
    args = parse_args()

    # --- Resolve settings: CLI arg > env var > auto-detected default ---
    max_seq_length = _resolve_int(args.max_seq_length, "EVO2_MAX_SEQ_LEN", _detect_max_seq_length(args.ckpt_dir))

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
        vortex_style_fp8=args.vortex_style_fp8,
        max_seq_length=max_seq_length,
        max_batch_size=args.max_batch_size,
        use_subquadratic_ops=args.use_subquadratic_ops,
        enable_chunked_prefill=args.enable_chunked_prefill,
        inference_dynamic_batching_max_tokens=args.inference_dynamic_batching_max_tokens,
        inference_dynamic_batching_block_size=args.inference_dynamic_batching_block_size,
        force_exit_on_completion=True,
    )


if __name__ == "__main__":
    main()
