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
automatically. Prompt groups use one boundary-described ragged prefill, then advance every active
request in parallel with persistent Hyena recurrent state and paged attention state. Block-level
CUDA graphs are the default decode path; a layer-level compatibility scope and eager debugging
path remain available.

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
        --prompt-batch-size 16 \
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
import math
import os
import sys
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional

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
from megatron.bridge.utils.common_utils import get_rank_safe, get_world_size_safe
from megatron.bridge.utils.instantiate_utils import instantiate
from megatron.core import dist_checkpointing, parallel_state
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import Float16Module

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH
from bionemo.evo2.models.evo2_provider import (
    CONTEXT_PARALLEL_COMM_TYPES,
    ContextParallelCommType,
    bind_hyena_packed_views_to_dynamic_context,
    bind_hyena_packed_views_to_dynamic_context_batch,
    bind_hyena_packed_views_to_static_context,
    build_evo2_mamba_inference_state_config,
    compute_evo2_paged_kv_buffer_size_gb,
    configure_runtime_context_parallel_comm_type,
    make_evo2_dynamic_inference_context_cls,
    reset_hyena_packed_views_for_new_request,
)
from bionemo.evo2.models.megatron.hyena.hyena_mixer import warm_packed_hyena_caches
from bionemo.evo2.models.megatron.hyena.subquadratic_safety import ensure_subquadratic_ops_supported
from bionemo.evo2.run.low_precision import (
    configure_global_fp8_layer_scope,
    configure_quantized_parameter_storage,
    inference_parameter_storage,
    inference_precision_kind,
    prepare_model_for_quantized_inference,
    validate_inference_precision,
)
from bionemo.evo2.run.predict import initialize_inference_distributed, resolve_checkpoint_path


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Detailed phase evidence requires synchronized CUDA boundaries and allocator resets. Keep it
# opt-in; ordinary inference still records low-overhead wall time for each phase, but does not
# synchronize solely for measurement or report phase-local allocator peaks.
_CUDA_PHASE_EVIDENCE_ENABLED = os.environ.get("EVO2_EXACT_PHASE_EVIDENCE") == "1"

# Static generation normally needs one full-batch context plus one remainder
# context. More exact-shape contexts retain large KV/state allocations and their
# CUDA-graph runners, so replace the small cache instead of letting it grow.
_MAX_STATIC_FLASH_CONTEXTS = 2


@dataclass(frozen=True)
class _CudaPhaseStats:
    """Wall time for one CUDA phase, plus allocator peaks when exact evidence is enabled."""

    elapsed_s: float = 0.0
    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    performed: bool = False
    _ended_at_s: float = field(default=0.0, repr=False, compare=False)


def _begin_cuda_phase(
    *,
    already_synchronized: bool = False,
    boundary_time_s: Optional[float] = None,
) -> float:
    """Start a CUDA phase, optionally reusing the preceding exact-measurement boundary."""
    if not _CUDA_PHASE_EVIDENCE_ENABLED:
        return time.perf_counter()
    if not already_synchronized:
        torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter() if boundary_time_s is None else float(boundary_time_s)


def _finish_cuda_phase(started_at_s: float) -> _CudaPhaseStats:
    """Finish a CUDA phase, synchronizing and reading allocator peaks only when requested."""
    if not _CUDA_PHASE_EVIDENCE_ENABLED:
        ended_at_s = time.perf_counter()
        return _CudaPhaseStats(
            elapsed_s=ended_at_s - started_at_s,
            performed=False,
            _ended_at_s=ended_at_s,
        )
    torch.cuda.synchronize()
    ended_at_s = time.perf_counter()
    return _CudaPhaseStats(
        elapsed_s=ended_at_s - started_at_s,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
        performed=True,
        _ended_at_s=ended_at_s,
    )


def _record_phase_stats(
    timings: Dict[str, Any],
    memory: Dict[str, int],
    phase_name: str,
    stats: _CudaPhaseStats,
) -> None:
    """Attach one named phase's timing, performed flag, and memory peaks."""
    timings[f"{phase_name}_elapsed_s"] = float(stats.elapsed_s)
    timings[f"{phase_name}_performed"] = bool(stats.performed)
    memory[f"{phase_name}_peak_allocated_bytes"] = int(stats.peak_allocated_bytes)
    memory[f"{phase_name}_peak_reserved_bytes"] = int(stats.peak_reserved_bytes)


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


# Small slack added on top of (prompt_len + max_new_tokens) when auto-sizing the context, matching the
# headroom the per-prompt path historically used.
_AUTO_MAX_SEQ_LENGTH_HEADROOM = 8

# Default number of leading prompts scanned to auto-size max_seq_length when no manual value is given.
# Tokenizing this many prompt strings is cheap; prompts beyond it are validated lazily and error loudly
# (naming the --max-seq-length to set) if one needs a larger context. Pass 0 to scan every prompt.
_DEFAULT_AUTO_MAX_SEQ_LENGTH_NUM_PROMPTS = 50


def _auto_max_seq_length_for(prompt_token_count: int, max_new_tokens: int) -> int:
    """Context length needed to fully serve a prompt of ``prompt_token_count`` tokens + its generation."""
    return int(prompt_token_count) + int(max_new_tokens) + _AUTO_MAX_SEQ_LENGTH_HEADROOM


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
    # Engine sequence-length budget. ``None`` means "auto": resolve from the prompts (longest prompt
    # + max_new_tokens + headroom) on first use and rebuild/re-capture if a later call must grow it.
    # A concrete value is a manual cap that supersedes auto-sizing.
    max_seq_length: Optional[int]
    evo2_seed: int
    cuda_graphs_enabled: bool
    cuda_graph_manager_count: int
    cuda_graph_scope: str
    precision_kind: str
    precision_parameter_storage: str
    # MCore's training/inference toggle removes ``cudagraph_manager`` attributes while graphs are
    # disabled. Keep the exact module/manager pairs owned by this persistent inference engine so
    # the next rollout can restore them without discarding captured runners or their storage.
    cuda_graph_manager_bindings: tuple[tuple[torch.nn.Module, Any], ...] = field(default_factory=tuple)
    cuda_graph_runner_count: int = 0
    cuda_graph_recorded_count: int = 0
    cuda_graph_replay_verified: bool = False
    # Transformer Engine quantized graphs may capture derived weight state that changes after a
    # colocated refit even when registered tensor addresses remain stable. The generation adapter
    # sets this after a quantized refit; the next rollout clears it only after successful recapture.
    cuda_graph_force_recapture: bool = False
    # Registered model-tensor storage captured by the current graph runners. Optimizer updates may
    # change values in place, but an offload/reload can rebind the tensors to new allocations; in
    # that case every runner must be discarded before another replay.
    cuda_graph_model_storage_signature: Optional[tuple[tuple[Any, ...], ...]] = None
    # MCore wrapper used only when pipeline parallelism is active. It is bound lazily to the
    # current dynamic context because that context may be rebuilt when its capacity grows.
    inference_wrapper: Optional[Any] = None
    # Persistent dynamic context, built lazily on the first generate() call and reused across all
    # subsequent calls so the per-layer CUDA graphs (captured once during warmup) stay valid. Keyed
    # by the context-affecting generate() options so it is rebuilt only if those change.
    shared_dyn_ctx: Optional[Any] = None
    shared_dyn_ctx_key: Optional[tuple] = None
    # Sampling RNG is intentionally persistent across generate() calls. The CLI invokes generate()
    # once per prompt-file chunk, and reseeding each chunk would replay identical samples for
    # repeated prompts.
    sampling_rng: Optional[torch.Generator] = None
    # True when ``max_seq_length`` was auto-sized from prompts (vs a manual cap). Auto mode grows the
    # context when needed; manual mode retains the configured cap.
    max_seq_length_is_auto: bool = False
    # Engine setup is measured by infer(), then emitted exactly once on the first generated group.
    engine_setup_stats: _CudaPhaseStats = field(default_factory=_CudaPhaseStats)
    engine_setup_stats_pending: bool = False
    # Stable generation-call counter used by serialized timing group identifiers.
    generation_call_index: int = 0
    # Static FlashAttention contexts are keyed by (batch, max sequence length).
    # Reusing the exact object preserves graph-bound KV and Hyena state pointers.
    static_contexts: dict[tuple[int, int], Any] = field(default_factory=dict)
    # Standalone inference can use inference_mode. A colocated training model must use
    # no_grad so Transformer Engine does not retain inference tensors across the next update.
    use_torch_inference_mode: bool = True


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


def _ensure_native_dynamic_cuda_graph_managers(
    hyena_model: torch.nn.Module,
    *,
    cuda_graph_scope: str,
) -> int:
    """Late-install graph managers omitted when a colocated model was built for training.

    Standalone inference selects local graphs before model construction. Colocated RL normally
    constructs from a training config with graphs disabled and enables them only for rollout;
    MCore's runtime toggle restores existing managers but does not create missing ones.
    """
    normalized_scope = str(getattr(cuda_graph_scope, "name", cuda_graph_scope)).lower()
    if normalized_scope not in {"block", "layer"}:
        raise ValueError(f"Unsupported CUDA graph scope {cuda_graph_scope!r}; expected 'block' or 'layer'")

    graph_scope = InferenceCudaGraphScope[normalized_scope]
    modules = list(hyena_model.modules())
    for module in modules:
        config = getattr(module, "config", None)
        if config is None:
            continue
        config.cuda_graph_impl = "local"
        config.inference_cuda_graph_scope = graph_scope
        config.cuda_graph_scope = []

    managers = [manager for module in modules if (manager := getattr(module, "cudagraph_manager", None)) is not None]
    if managers:
        return len(managers)

    # Optional MCore machinery: import it only when the training-built model needs a manager.
    from megatron.core.transformer.cuda_graphs import CudaGraphManager

    decoder = getattr(hyena_model, "decoder", None)
    if decoder is None:
        raise RuntimeError("Evo2 CUDA graphs require a decoder graph owner")
    if normalized_scope == "block":
        decoder_config = getattr(decoder, "config", getattr(hyena_model, "config", None))
        if decoder_config is None:
            raise RuntimeError("Evo2 decoder has no TransformerConfig for CUDA graph construction")
        create_manager = getattr(decoder, "create_mcore_cudagraph_manager", None)
        if callable(create_manager):
            create_manager(decoder_config)
        else:
            decoder.cudagraph_manager = CudaGraphManager(decoder_config)
    else:
        for layer in getattr(decoder, "layers", ()):
            layer_config = getattr(layer, "config", getattr(hyena_model, "config", None))
            create_manager = getattr(layer, "create_mcore_cudagraph_manager", None)
            if callable(create_manager):
                create_manager(layer_config)
            elif layer_config is not None:
                layer.cudagraph_manager = CudaGraphManager(layer_config)

    manager_count = sum(
        1 for module in hyena_model.modules() if getattr(module, "cudagraph_manager", None) is not None
    )
    if manager_count == 0:
        raise RuntimeError(
            f"Could not create an Evo2 {normalized_scope}-scope CUDA graph manager for the training-built model"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(
            "[evo2-native-cg] late-installed %d %s-scope CUDA graph manager(s) for colocated inference",
            manager_count,
            normalized_scope,
        )
    return manager_count


def _cuda_graph_manager_bindings(
    model: torch.nn.Module,
) -> tuple[tuple[torch.nn.Module, Any], ...]:
    """Return the module/manager bindings currently installed on ``model``."""
    return tuple(
        (module, manager)
        for module in model.modules()
        if (manager := getattr(module, "cudagraph_manager", None)) is not None
    )


def _restore_native_dynamic_cuda_graph_managers(nd: Evo2NativeDynamicComponents) -> int:
    """Restore graph managers removed by MCore's training-mode graph toggle.

    Colocated rollout switches the shared model back to its training graph configuration between
    calls. MCore implements that transition by deleting each module's ``cudagraph_manager``
    attribute. The manager objects and captured runners remain valid as long as registered model
    storage is unchanged, so the persistent native engine retains and restores those exact objects
    before checking its storage fingerprint.
    """
    bindings = getattr(nd, "cuda_graph_manager_bindings", ())
    for module, manager in bindings:
        if getattr(module, "cudagraph_manager", None) is not manager:
            module.cudagraph_manager = manager
    if bindings:
        nd.cuda_graph_manager_count = len(bindings)
    return len(bindings)


def _native_torch_context(nd: Evo2NativeDynamicComponents):
    """Return the non-grad context appropriate for this engine's lifetime."""
    if getattr(nd, "use_torch_inference_mode", True):
        return torch.inference_mode()
    return torch.no_grad()


def _setup_native_dynamic_components(
    *,
    model: torch.nn.Module,
    raw_model: torch.nn.Module,
    max_seq_length: Optional[int],
    evo2_seed: int,
    cuda_graphs_enabled: bool,
    cuda_graph_scope: Optional[str] = None,
    precision_kind: Optional[str] = None,
    precision_parameter_storage: Optional[str] = None,
) -> Evo2NativeDynamicComponents:
    """Prepare the standalone HyenaModel to decode on an Evo2 dynamic context.

    This disables sequence parallelism for the Evo2 model, builds the exact-rounding
    ``DynamicInferenceContext`` subclass, and creates the Mamba state config that lets mcore
    allocate Hyena recurrent state in its dynamic state buffers. A single dynamic context is built
    lazily in :func:`_generate_native_dynamic`, sized to the longest prompt plus the requested
    generation length, and reused (reset) across prompts so CUDA-graph capture stays valid.
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

    if cuda_graph_scope is None:
        configured_scope = getattr(hyena_model.config, "inference_cuda_graph_scope", None)
        cuda_graph_scope = str(getattr(configured_scope, "name", configured_scope or "none"))
    cuda_graph_scope = str(getattr(cuda_graph_scope, "name", cuda_graph_scope)).lower()
    ctx_cls = make_evo2_dynamic_inference_context_cls()
    mamba_cfg = build_evo2_mamba_inference_state_config(raw_model)
    # Allocate modal pole-table storage before graph capture. Later in-place optimizer/refit
    # updates refresh its values during prefill without changing graph-captured addresses.
    warmed_modal_layers = warm_packed_hyena_caches(hyena_model)
    if cuda_graphs_enabled:
        cuda_graph_manager_count = _ensure_native_dynamic_cuda_graph_managers(
            hyena_model,
            cuda_graph_scope=cuda_graph_scope,
        )
    else:
        cuda_graph_manager_count = sum(
            1 for module in hyena_model.modules() if getattr(module, "cudagraph_manager", None) is not None
        )
    if precision_kind is None:
        precision_kind = inference_precision_kind(hyena_model.config)
    if precision_parameter_storage is None:
        precision_parameter_storage = inference_parameter_storage(hyena_model.config)
    if rank == 0:
        logger.info(
            "[evo2-native] standalone evo2 prepared for native dynamic decode "
            "(SP off, cuda_graphs=%s, graph_managers=%d, modal_pole_caches=%d).",
            cuda_graphs_enabled,
            cuda_graph_manager_count,
            warmed_modal_layers,
        )
    return Evo2NativeDynamicComponents(
        ctx_cls=ctx_cls,
        mamba_state_config=mamba_cfg,
        forward_model=model,
        hyena_model=hyena_model,
        max_seq_length=max_seq_length,
        evo2_seed=evo2_seed,
        cuda_graphs_enabled=cuda_graphs_enabled,
        cuda_graph_manager_count=cuda_graph_manager_count,
        cuda_graph_scope=str(cuda_graph_scope),
        precision_kind=str(precision_kind),
        precision_parameter_storage=str(precision_parameter_storage),
        cuda_graph_manager_bindings=_cuda_graph_manager_bindings(hyena_model),
        max_seq_length_is_auto=max_seq_length is None,
    )


def _configure_native_dynamic_cuda_graphs(
    model_provider: Any,
    *,
    rank: int,
    cuda_graph_impl: str = "local",
    cuda_graph_scope: str = "block",
) -> bool:
    """Enable mcore local CUDA graphs for Evo2 dynamic inference when supported.

    This mirrors Megatron's ``cuda_graph_impl=local`` setup, but applies it directly to the
    provider loaded from the checkpoint because this recipe does not use Megatron's global arg
    parser. ``cuda_graph_scope="block"`` captures the complete decoder stack in one graph per
    active request count, avoiding one host graph launch per layer. ``"layer"`` preserves the
    narrower per-layer MCore graph path as a compatibility fallback.

    ``cuda_graph_impl="none"`` disables graph capture entirely (decode runs eager) -- useful for
    debugging and for tests that need an un-graphed reference to compare against.
    """
    if not hasattr(model_provider, "cuda_graph_impl"):
        if rank == 0:
            logger.warning("[evo2-native-cg] model provider has no cuda_graph_impl; CUDA graphs disabled")
        return False

    if cuda_graph_scope not in {"block", "layer"}:
        raise ValueError(f"Unsupported CUDA graph scope {cuda_graph_scope!r}; expected 'block' or 'layer'")

    model_provider.cuda_graph_impl = cuda_graph_impl
    # A checkpoint can carry a scope selected by a different graph implementation. Store the
    # normalized enum explicitly because the provider has already run TransformerConfig.__post_init__.
    model_provider.inference_cuda_graph_scope = (
        InferenceCudaGraphScope.none if cuda_graph_impl == "none" else InferenceCudaGraphScope[cuda_graph_scope]
    )
    model_provider.cuda_graph_scope = []
    if cuda_graph_impl == "none":
        if rank == 0:
            logger.info("[evo2-native-cg] CUDA graphs disabled (cuda_graph_impl='none'); decode runs eager")
        return False

    os.environ.setdefault("NCCL_GRAPH_REGISTER", "0")
    if rank == 0:
        logger.info("[evo2-native-cg] enabled mcore local %s CUDA graphs for dynamic decode", cuda_graph_scope)
    return True


def _resolve_native_dynamic_cuda_graph_scope(
    requested_scope: str,
    *,
    cuda_graph_impl: str,
    fp8_enabled: bool,
    fp4_enabled: bool,
) -> str:
    """Select a CUDA-graph scope compatible with Transformer Engine quantization state."""
    if cuda_graph_impl == "local" and requested_scope == "block" and (fp8_enabled or fp4_enabled):
        return "layer"
    return requested_scope


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


def _resolve_inference_subquadratic_ops(
    use_subquadratic_ops: bool,
    inference_backend: Literal["dynamic", "static-flash"],
) -> bool:
    """Ignore the legacy kernels when native segmented dynamic inference owns both phases."""
    if inference_backend not in {"dynamic", "static-flash"}:
        raise ValueError(f"Unsupported inference backend {inference_backend!r}")
    if use_subquadratic_ops and inference_backend == "dynamic":
        if int(os.environ.get("RANK", "0")) == 0:
            logger.warning(
                "--use-subquadratic-ops is ignored by the dynamic inference backend: "
                "segmented packed prefill and fused recurrent decode already replace those kernels. "
                "Keeping CUDA graphs enabled."
            )
        return False
    return bool(use_subquadratic_ops)


def setup_inference_engine(
    ckpt_dir: Path,
    *,
    max_seq_length: Optional[int] = None,
    max_batch_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    context_parallel_comm_type: Optional[ContextParallelCommType] = None,
    mixed_precision_recipe: Optional[str] = None,
    quantized_param_storage: Literal["recipe", "bf16"] = "recipe",
    fp8_all_layers: bool = False,
    vortex_style_fp8: bool = False,
    random_seed: int = 1234,
    use_subquadratic_ops: bool = False,
    cuda_graph_impl: str = "local",
    cuda_graph_scope: str = "block",
    inference_backend: Literal["dynamic", "static-flash"] = "dynamic",
) -> Evo2InferenceComponents:
    """Setup the Evo2 native dynamic-inference engine and related components.

    Loads the model, wires it onto the native mcore dynamic-inference engine (paged-KV attention +
    Hyena recurrent state packed into mcore's two Mamba slots), and returns everything needed for
    text generation. ``flash_decode`` and sequence-parallel are turned off automatically (both
    required by the dynamic path).

    Args:
        ckpt_dir: Path to MBridge checkpoint directory.
        max_seq_length: Engine sequence-length budget for the persistent dynamic context. ``None``
            (default) auto-sizes it from the prompts at the first :func:`generate` call (longest
            prompt + ``max_new_tokens`` + headroom); a concrete value is a manual cap that supersedes
            auto-sizing. The context is CUDA-graph-pinned, so the budget cannot change in place — in
            auto mode a later prompt that needs more triggers a one-time rebuild + graph re-capture at
            a larger size; a manual cap never grows (an over-long prompt then just stops early).
        max_batch_size: Prompt-file chunk size and Megatron setup micro-batch metadata. This does
            not control the number of prompt-file generations or Evo2 native decode concurrency.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        context_parallel_comm_type: Runtime TE attention transport. ``None`` selects
            P2P; A2A can be selected for tighter BF16 parity. Checkpoint metadata
            does not control this execution choice.
        mixed_precision_recipe: Override mixed precision recipe.
        quantized_param_storage: Preserve the recipe's native quantized parameters (``"recipe"``)
            or retain BF16 parameters while using quantized GEMMs (``"bf16"``).
        fp8_all_layers: Remove BF16 first/last-block exclusions from the selected global TE FP8
            recipe. This is the regular full-scope Hopper FP8 path for Evo2 7B.
        vortex_style_fp8: Use vortex-style FP8 (applies FP8 only to projection layers).
            Needed for FP8-sensitive checkpoints from original evo2 training (1b, 40b).
        random_seed: Random seed for reproducibility.
        use_subquadratic_ops: Use fused Hyena convolution kernels for static-Flash
            rectangular/eager prefill compatibility. Ignored by the dynamic backend.
        cuda_graph_impl: ``"local"`` (default) captures MCore CUDA graphs for decode; ``"none"``
            disables graph capture (eager decode), mainly for debugging / reference runs.
        cuda_graph_scope: ``"block"`` (default) captures the complete decoder in one graph per
            request count; ``"layer"`` retains per-layer graphs as a compatibility fallback.
            Global FP8/FP4 recipes automatically resolve ``"block"`` to ``"layer"`` because
            Transformer Engine's quantization state is not compatible with block capture.
        inference_backend: Backend that will consume the configured model. The default dynamic
            backend ignores ``use_subquadratic_ops`` because its packed prefill and decode use
            separate native kernels.

    Returns:
        Evo2InferenceComponents containing all inference components.

    Example:
        >>> components = setup_inference_engine(Path("/path/to/checkpoint"), max_batch_size=4)
        >>> results = generate(components, prompts=["ATCG", "GCTA"], max_new_tokens=100)
    """
    use_subquadratic_ops = _resolve_inference_subquadratic_ops(use_subquadratic_ops, inference_backend)

    # The remaining use is static-Flash rectangular prefill. Keep its decode eager because an
    # unsupported fused-decode shape can fall back to a prebuilt subquadratic kernel, which cannot
    # be captured safely. Dynamic inference resolved the flag to False above and retains graphs.
    if use_subquadratic_ops and cuda_graph_impl != "none":
        logger.warning(
            "use_subquadratic_ops=True is incompatible with CUDA-graphed decode "
            "(cuda_graph_impl=%r): the prebuilt subquadratic_ops_torch kernels cannot be captured "
            "into a CUDA graph and crash with SIGSEGV during capture. Forcing cuda_graph_impl='none' "
            "(eager decode) so the requested subquadratic-ops can run. Prefer the default "
            "cuda_graph_impl='local' + use_subquadratic_ops=False unless you need subquadratic-ops "
            "for very long prefill.",
            cuda_graph_impl,
        )
        cuda_graph_impl = "none"

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
    configure_runtime_context_parallel_comm_type(model_provider, context_parallel_comm_type)
    # Disable sequence parallelism for inference - Megatron's inference engine
    # does not support it for non-MoE models.
    model_provider.sequence_parallel = False

    # The native dynamic engine drives paged flash-attn-varlen itself and asserts NOT
    # static-batching, so flash_decode (which asserts static batching, attention.py) MUST be off.
    model_provider.flash_decode = False
    model_provider.use_subquadratic_ops = use_subquadratic_ops
    if getattr(model_provider, "fp32_residual_connection", False):
        logger.info("Disabling fp32_residual_connection for inference to keep TE activations in params_dtype")
        model_provider.fp32_residual_connection = False

    if vortex_style_fp8:
        model_provider.vortex_style_fp8 = True

    # Keep BF16 as the portable default; global FP8 is an explicit checkpoint/hardware-qualified choice.
    if mixed_precision_recipe is not None:
        mp_config = get_mixed_precision_config(mixed_precision_recipe)
    else:
        mp_config = get_mixed_precision_config("bf16_mixed")

    configure_global_fp8_layer_scope(mp_config, all_layers=fp8_all_layers)
    configure_quantized_parameter_storage(mp_config, quantized_param_storage)
    validate_inference_precision(mp_config, vortex_style_fp8=vortex_style_fp8)
    precision_kind = inference_precision_kind(mp_config)
    precision_parameter_storage = inference_parameter_storage(mp_config)
    mp_config.finalize()
    mp_config.setup(model_provider)
    effective_cuda_graph_scope = _resolve_native_dynamic_cuda_graph_scope(
        cuda_graph_scope,
        cuda_graph_impl=cuda_graph_impl,
        fp8_enabled=getattr(model_provider, "fp8", None) is not None,
        fp4_enabled=getattr(model_provider, "fp4", None) is not None,
    )
    if effective_cuda_graph_scope != cuda_graph_scope and int(os.environ.get("RANK", "0")) == 0:
        logger.warning(
            "[evo2-native-cg] global Transformer Engine FP8/FP4 state is layer-scoped; "
            "using cuda_graph_scope='layer' instead of requested 'block'"
        )
    cuda_graph_scope = effective_cuda_graph_scope
    cuda_graphs_enabled = _configure_native_dynamic_cuda_graphs(
        model_provider,
        rank=int(os.environ.get("RANK", "0")),
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_scope=cuda_graph_scope,
    )
    if cuda_graphs_enabled and getattr(model_provider, "recompute_granularity", None):
        logger.info("Disabling activation recompute for inference CUDA graphs")
        model_provider.recompute_granularity = None
    logger.info("Inference precision: %s (parameter storage: %s)", precision_kind, precision_parameter_storage)

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

    # Globally quantized TE GEMMs require an aligned leading token dimension, while dynamic decode
    # presents one token per active request and flat packed prefill can have any total length. MCore's
    # historically FP8-named wrapper handles both active FP8 and FP4 contexts and delegates the exact
    # alignment to Transformer Engine. Regular FP8 bypasses the wrapper when the flattened
    # sequence-times-batch GEMM rows are already legal. Install both paths before graph capture.
    if prepare_model_for_quantized_inference(raw_model, mp_config):
        aligned_fast_path_modules = int(getattr(raw_model, "evo2_regular_fp8_aligned_fast_path_modules", 0))
        logger.info(
            "%s recipe active: alignment fallback installed; %d TE linears can bypass per-layer "
            "padding for legal flattened row counts",
            precision_kind,
            aligned_fast_path_modules,
        )

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
        evo2_seed=random_seed,
        cuda_graphs_enabled=cuda_graphs_enabled,
        cuda_graph_scope=cuda_graph_scope if cuda_graphs_enabled else "none",
        precision_kind=precision_kind,
        precision_parameter_storage=precision_parameter_storage,
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
    ignore_eos: bool = False,
    preserve_eos_token: bool = False,
    strict_generation: bool = False,
    enable_chunked_prefill: bool = False,
    inference_dynamic_batching_max_tokens: Optional[int] = None,
    inference_dynamic_batching_block_size: int = 256,
    evo2_batched_decode_size: int = 1,
    inference_backend: Literal["dynamic", "static-flash"] = "dynamic",
    result_callback: Optional[Callable[[int, Any], None]] = None,
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
        top_p: Nucleus sampling parameter (0 = disabled), applied after top-k when both are enabled.
        return_log_probs: Whether to return log probabilities.
        ignore_eos: Omit sampled EOS tokens and continue to max_new_tokens.
        preserve_eos_token: Include a sampled terminal EOS/EOD token and, when requested, its
            log-probability in the result before stopping. Has no effect when ``ignore_eos``
            suppresses stop tokens.
        strict_generation: Fail instead of returning short or fallback generation results.
        enable_chunked_prefill: Split prompts across multiple prefill forwards when they exceed
            ``inference_dynamic_batching_max_tokens``. Disabled by default.
        inference_dynamic_batching_max_tokens: Optional dynamic-context per-step token budget.
            When set and chunking is disabled, each prompt must fit within this value.
        inference_dynamic_batching_block_size: KV-cache block size for the dynamic context. This is
            not the prefill chunk size.
        evo2_batched_decode_size: Number of variable-length prompts to prefill and decode together.
        inference_backend: Use packed paged-KV dynamic inference or equal-length static FlashAttention.
        result_callback: Optional callback invoked with each prompt index and native result as it completes.

    Returns:
        List of :class:`_NativeDynamicResult` objects (mirroring the
        ``generated_text`` / ``generated_length`` / ``prompt_tokens`` fields downstream reads).

    Example:
        >>> components = setup_inference_engine(ckpt_dir)
        >>> results = generate(components, ["ATCGATCG"], max_new_tokens=50, top_k=1)
        >>> print(_unwrap_result(results[0]).generated_text)
    """
    if inference_backend == "static-flash":
        if enable_chunked_prefill:
            raise ValueError("Static FlashAttention inference does not support chunked prefill")
        return _generate_static_flash(
            components,
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_log_probs=return_log_probs,
            ignore_eos=ignore_eos,
            preserve_eos_token=preserve_eos_token,
            strict_generation=strict_generation,
            evo2_batched_decode_size=evo2_batched_decode_size,
            result_callback=result_callback,
        )
    if inference_backend != "dynamic":
        raise ValueError(f"Unsupported Evo2 inference backend {inference_backend!r}")
    return _generate_native_dynamic(
        components,
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=return_log_probs,
        ignore_eos=ignore_eos,
        preserve_eos_token=preserve_eos_token,
        strict_generation=strict_generation,
        enable_chunked_prefill=enable_chunked_prefill,
        inference_dynamic_batching_max_tokens=inference_dynamic_batching_max_tokens,
        inference_dynamic_batching_block_size=inference_dynamic_batching_block_size,
        evo2_batched_decode_size=evo2_batched_decode_size,
        result_callback=result_callback,
    )


@dataclass
class _NativeDynamicResult:
    """Minimal result object mirroring mcore's ``InferenceRequest`` fields used downstream.

    Carries ``generated_text``, ``generated_length``, ``prompt_tokens``, ``generated_tokens``,
    ``generated_log_probs``, ``finish_reason``, ``stopped_on_eos``, ``truncated``, ``timings``, and
    ``memory`` for output serialization and generation validation.
    """

    generated_text: str
    generated_length: int
    prompt_tokens: List[int]
    generated_tokens: Optional[List[int]] = None
    generated_log_probs: Optional[List[float]] = None
    finish_reason: str = "length"
    stopped_on_eos: bool = False
    truncated: bool = False
    timings: Optional[Dict[str, Any]] = None
    memory: Optional[Dict[str, int]] = None


def _sampling_log_probs_from_logits(
    last_token_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Return log-probs from the exact distribution used for generation sampling.

    Self-contained mcore-compatible sampler for the native dynamic path. Greedy
    (``top_k == 1``) is represented as a one-token support distribution; otherwise
    this applies standard temperature, top-k, and top-p filtering before
    renormalization.

    Args:
        last_token_logits: Logits of shape ``[batch_size, vocab_size]``.
        temperature: Temperature scaling factor (applied only on the non-greedy path).
        top_k: Top-k filtering value (0 = disabled, 1 = greedy argmax).
        top_p: Top-p (nucleus) filtering value (0.0 = disabled).
        vocab_size: When provided, validates ``top_k < vocab_size``. Sampled-id clamping belongs to
            :func:`_sample_from_log_probs`.

    Returns:
        Log probabilities of shape ``[batch_size, vocab_size]``.
    """
    assert isinstance(top_p, float)
    assert isinstance(top_k, int)
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

    last_token_logits = last_token_logits.clone()  # .div_/.masked_fill_ below are in-place
    if top_k == 1:
        argmax = torch.argmax(last_token_logits, dim=-1)
        deterministic_logits = torch.full_like(last_token_logits, float("-Inf"))
        deterministic_logits.scatter_(1, argmax.unsqueeze(1), 0.0)
        return torch.log_softmax(deterministic_logits, dim=-1)

    if temperature != 1.0:
        last_token_logits.div_(temperature)
    if top_k > 1:
        assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
        if vocab_size:
            assert top_k < vocab_size, "top-k is larger than vocab size."
        _modify_for_top_k(last_token_logits, top_k)
    if 0.0 < top_p < 1.0:
        _modify_for_top_p(last_token_logits, top_p)

    return torch.log_softmax(last_token_logits, dim=-1)


def _sample_from_log_probs(
    log_probs: torch.Tensor,
    *,
    top_k: int,
    generator: torch.Generator,
    vocab_size: Optional[int] = None,
) -> torch.Tensor:
    """Sample next-token ids from pre-filtered log-probabilities."""
    if top_k == 1:
        return torch.argmax(log_probs, dim=-1)

    probabilities = log_probs.exp()
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).view(-1)
    if vocab_size:
        sampled = torch.clamp(sampled, min=0, max=(vocab_size - 1))
    return sampled


def _selected_log_probs_for_sampled_tokens(log_probs: torch.Tensor, sampled_tokens: torch.Tensor) -> list[float]:
    """Return sampled-token log-probs with one tensor gather and one host transfer."""
    sampled_indices = sampled_tokens.to(device=log_probs.device, dtype=torch.long).view(-1, 1)
    return log_probs.gather(1, sampled_indices).squeeze(1).detach().cpu().tolist()


def _sampling_rng_for_native_dynamic(nd: Evo2NativeDynamicComponents, device: torch.device) -> torch.Generator:
    """Return the persistent sampling RNG for an inference engine.

    The native CLI processes prompt files in chunks but should sample those chunks as one continuous
    stream. Keeping the generator on ``nd`` avoids replaying the same RNG sequence when many chunks
    contain identical prompts.
    """
    rng = nd.sampling_rng
    if rng is not None:
        try:
            if torch.device(rng.device) == torch.device(device):
                return rng
        except RuntimeError:
            pass

    rng = torch.Generator(device=device)
    rng.manual_seed(int(nd.evo2_seed))
    nd.sampling_rng = rng
    return rng


def _extract_generation_logits(dyn_ctx, logits: torch.Tensor) -> torch.Tensor:
    """Return one vocab-logit row per active generation request."""
    if getattr(dyn_ctx, "materialize_only_last_token_logits", False):
        assert logits.size(0) == 1, f"logits.size(0) ({tuple(logits.shape)}) != 1"
        return logits.squeeze(0)[: dyn_ctx.num_last_token_logits].float()
    return dyn_ctx.last_token_logits(logits).float()


def _forward_native_dynamic_logits(
    nd: Evo2NativeDynamicComponents,
    dyn_ctx: Any,
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    """Run one native-dynamic forward and make PP logits available on every stage."""
    pp_group = getattr(dyn_ctx, "pipeline_parallel_group", None)
    if pp_group is None or pp_group.size() == 1:
        return nd.forward_model(
            input_ids,
            position_ids,
            None,
            inference_context=dyn_ctx,
            runtime_gather_output=True,
        )

    inference_wrapper = getattr(nd, "inference_wrapper", None)
    if inference_wrapper is None or inference_wrapper.inference_context is not dyn_ctx:
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )

        # HyenaModel follows the same inference forward/set_input_tensor contract as GPTModel.
        # MCore's wrapper owns the PP receive, set_input_tensor, forward, and send sequence.
        inference_wrapper = GPTInferenceWrapper(nd.forward_model, dyn_ctx)
        nd.inference_wrapper = inference_wrapper

    logits = inference_wrapper.run_one_forward_step(
        {"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}
    )

    from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage

    config = getattr(dyn_ctx, "config", None)
    materialize_only_last = getattr(
        dyn_ctx,
        "materialize_only_last_token_logits",
        getattr(config, "materialize_only_last_token_logits", False),
    )
    logits_seq_len = dyn_ctx.num_last_token_logits if materialize_only_last else input_ids.shape[1]
    return broadcast_from_last_pipeline_stage(
        [1, int(logits_seq_len), int(nd.hyena_model.vocab_size)],
        dtype=nd.hyena_model.config.params_dtype,
        tensor=logits,
        pp_group=pp_group,
    )


def _native_stop_token_ids(tokenizer: Any) -> set[int]:
    """Best-effort set of tokenizer EOD/EOS ids for fixed-shape native decode."""
    stop_token_ids: set[int] = set()

    def _add_token_id(token_id: Any) -> None:
        if token_id is None:
            return
        if isinstance(token_id, (list, tuple, set)):
            for item in token_id:
                _add_token_id(item)
            return
        if hasattr(token_id, "tolist"):
            _add_token_id(token_id.tolist())
            return
        if isinstance(token_id, str):
            mapped_id = None
            for token_lookup_owner in (tokenizer, getattr(tokenizer, "tokenizer", None)):
                token_to_id = getattr(token_lookup_owner, "token_to_id", None)
                if callable(token_to_id):
                    mapped_id = token_to_id(token_id)
                    if mapped_id is not None:
                        break
            if mapped_id is None and hasattr(tokenizer, "tokenize"):
                tokenized = tokenizer.tokenize(token_id)
                if isinstance(tokenized, int):
                    mapped_id = tokenized
                elif hasattr(tokenized, "tolist"):
                    tokenized = tokenized.tolist()
                if isinstance(tokenized, (list, tuple)) and len(tokenized) == 1:
                    mapped_id = tokenized[0]
            if mapped_id is not None:
                stop_token_ids.add(int(mapped_id))
            return
        stop_token_ids.add(int(token_id))

    for attr_name in ("eod", "eod_id", "eod_token", "eos", "eos_id", "eos_token", "eos_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is None and hasattr(tokenizer, "tokenizer"):
            token_id = getattr(tokenizer.tokenizer, attr_name, None)
        _add_token_id(token_id)
    for token_text in ("<EOS>", "<EOD>"):
        _add_token_id(token_text)
    return stop_token_ids


def _sampled_token_action(
    token_id: int,
    stop_token_ids: set[int],
    *,
    ignore_eos: bool,
    preserve_eos_token: bool = False,
) -> tuple[bool, bool]:
    """Return whether to append a sampled token and stop its request."""
    is_eos = token_id in stop_token_ids
    if not is_eos:
        return True, False
    if ignore_eos:
        return False, False
    return preserve_eos_token, True


def _physical_request_counts(prompt_count: int, batch_size: int) -> tuple[int, ...]:
    """Return the distinct physical decode shapes used by fixed-row prompt groups."""
    if prompt_count < 0:
        raise ValueError(f"prompt_count must be non-negative, got {prompt_count}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if prompt_count == 0:
        return ()
    _, remainder = divmod(prompt_count, batch_size)
    counts = {min(prompt_count, batch_size)}
    if remainder:
        counts.add(remainder)
    return tuple(sorted(counts))


def _stop_token_mask(logits: torch.Tensor, stop_token_ids: set[int]) -> Optional[torch.Tensor]:
    """Build a reusable device mask for forced-length generation."""
    valid_stop_token_ids = sorted(token_id for token_id in stop_token_ids if 0 <= token_id < logits.shape[-1])
    if not valid_stop_token_ids:
        return None
    if len(valid_stop_token_ids) == logits.shape[-1]:
        raise RuntimeError("Cannot ignore EOS because every tokenizer vocabulary entry is a stop token")

    stop_token_mask = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    stop_token_mask[valid_stop_token_ids] = True
    return stop_token_mask


def _suppress_stop_token_logits(
    logits: torch.Tensor,
    stop_token_ids: set[int],
    *,
    stop_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exclude stop-token IDs without a per-token device reduction or host-to-device index copy."""
    if stop_token_mask is None:
        stop_token_mask = _stop_token_mask(logits, stop_token_ids)
    if stop_token_mask is None:
        return logits
    if stop_token_mask.shape != (logits.shape[-1],) or stop_token_mask.device != logits.device:
        raise ValueError("stop-token mask must match the logits vocabulary and device")
    return logits.masked_fill(stop_token_mask, float("-inf"))


def _normalize_new_request_slots_for_packed_hyena(dyn_ctx: Any, request_count: int) -> torch.Tensor:
    """Normalize mcore's reverse-contiguous LIFO allocation before packed-state binding.

    Individually added requests receive fresh Mamba slots in descending order. Before any
    recurrent state is consumed, reverse that exact allocator order in the request mapping so
    request rows align with the ascending stable slice used by packed Hyena state. Leave every
    other order unchanged so the binding validation rejects arbitrary permutations.
    """
    request_slots = dyn_ctx.mamba_metadata.request_to_mamba_state_idx[:request_count]
    slots = [int(slot) for slot in request_slots.tolist()]
    expected_lifo_slots = list(range(slots[0], slots[0] - len(slots), -1)) if slots else []
    if slots and slots == expected_lifo_slots:
        # This intentionally updates mcore's canonical request-to-slot map, not just the
        # tensor passed to Hyena binding. mcore reads the map again in
        # initialize_attention_state() and update_requests(). Reassigning the same freshly
        # allocated slot set before the first forward keeps those later reads aligned with
        # the ascending packed Hyena views. Evo2 leaves prefix caching off, keeps the
        # batched requests active until they reset together, and reserves rollover KV blocks
        # before mcore can pause/reorder requests, so every physical row remains stable.
        request_slots.copy_(request_slots.flip(0))
    return request_slots


def _reserve_packed_decode_rollover_blocks(dyn_ctx: Any, request_count: int) -> int:
    """Reserve the next KV page without letting mcore permute packed request rows.

    ``DynamicInferenceContext.update_requests`` normally handles a full final KV block by
    temporarily pausing that request, moving paused rows to the left, and resuming them in LIFO
    order after allocating a block. If only part of a heterogeneous Evo2 batch reaches the page
    boundary, that changes the request-row order. Attention follows the moved bookkeeping, but
    packed Hyena recurrent state and the result lists are intentionally bound to stable physical
    rows, so the next decode step would combine one request's token with another request's state.

    The Evo2 context is right-sized for every active request's full sequence and has no paused
    block pool. Allocate boundary pages immediately before ``update_requests`` and mark their
    offsets as just before the new page. Mcore then performs its normal metadata update from -1
    to 0 without entering pause/resume or changing row order.

    Returns:
        Number of request rows for which a new KV block was reserved.
    """
    if request_count < 1:
        raise ValueError(f"request_count must be positive, got {request_count}")
    paused_request_count = int(getattr(dyn_ctx, "paused_request_count", 0))
    if paused_request_count != 0:
        raise RuntimeError("Packed Evo2 decode cannot reserve KV blocks while requests are paused")
    active_request_count = int(dyn_ctx.total_request_count) - paused_request_count
    if active_request_count != request_count:
        raise RuntimeError(
            f"Packed Evo2 decode expected {request_count} active request rows, found {active_request_count}"
        )
    if int(getattr(dyn_ctx, "num_speculative_tokens", 0)) != 0:
        raise RuntimeError("Packed Evo2 decode does not support speculative-token KV rollover")

    allocator = dyn_ctx.kv_block_allocator
    if int(getattr(allocator, "paused_count", 0)) != 0:
        raise RuntimeError("Packed Evo2 decode requires a dynamic context without a paused KV block pool")

    block_size_tokens = int(dyn_ctx.block_size_tokens)
    if block_size_tokens < 1:
        raise RuntimeError(f"Invalid dynamic-context KV block size: {block_size_tokens}")
    offsets = dyn_ctx.request_last_kv_block_offset[:request_count]
    rollover_rows = torch.nonzero(offsets >= block_size_tokens - 1, as_tuple=True)[0]
    rollover_count = int(rollover_rows.numel())
    if rollover_count == 0:
        return 0

    block_counts = dyn_ctx.request_kv_block_counts
    block_table = dyn_ctx.request_to_kv_block_ids
    rollover_columns = block_counts[rollover_rows]
    if bool((rollover_columns >= block_table.shape[1]).any().item()):
        raise RuntimeError("Packed Evo2 decode exhausted a request's paged-KV block table")
    if int(allocator.get_active_avail()) < rollover_count:
        raise RuntimeError(
            f"Packed Evo2 decode needs {rollover_count} rollover KV block(s), but the active pool "
            f"has only {allocator.get_active_avail()} available"
        )

    new_block_ids = allocator.allocate_memory_blocks(rollover_count)
    if new_block_ids is None or int(new_block_ids.numel()) != rollover_count:
        raise RuntimeError(f"Packed Evo2 decode failed to allocate {rollover_count} rollover KV block(s)")

    block_table[rollover_rows, rollover_columns] = new_block_ids
    block_counts[rollover_rows] += 1
    dyn_ctx.request_last_kv_block_id[rollover_rows] = new_block_ids
    offsets[rollover_rows] = -1
    return rollover_count


def _warmup_native_dynamic_cuda_graphs(
    nd: Evo2NativeDynamicComponents,
    dyn_ctx: Any,
    device: torch.device,
    *,
    request_counts: Collection[int],
) -> None:
    """Capture requested physical decode shapes up front on throwaway requests.

    mcore captures each per-layer decode CUDA graph lazily on the first decode step that matches the
    graph's batch dimensions, and that capture runs warmup iterations of the layer forward. For Evo2
    those warmup iterations advance the in-place Hyena recurrent state, so if capture happened on the
    first *real* prompt's decode it would corrupt that prompt's output (later prompts, whose decode
    just replays the captured graph, are unaffected). mcore's ``DynamicInferenceEngine`` avoids this
    by capturing graphs up front in ``create_cuda_graphs()`` with throwaway requests; the standalone
    Evo2 loop does the equivalent here.

    Unlike a plain attention model, the captured decode graph must read and write the Hyena recurrent
    state through the packed mamba-slot views, so the throwaway request is *prefilled* first (binding
    those views and seeding the recurrent state, which selects the decode code path) and then decoded
    a couple of steps to trigger and replay capture. The context is reset afterwards, discarding the
    throwaway state; the captured graph (held on the model's layers) is then reused by every real
    prompt. EOD stopping is logical rather than physical: every row in a prompt group remains active
    until that group completes. Consequently only the full and optional remainder group shapes used
    by the current call need runners; warming every size from one to the configured capacity wastes
    capture time and graph memory. Only the public context primitives the real decode loop already
    uses are exercised here, so this does not depend on mcore's internal graph-warmup helpers.
    """
    from megatron.core.inference.inference_request import DynamicInferenceRequest

    hyena_model = nd.hyena_model
    rank = int(os.environ.get("RANK", "0"))

    # A short throwaway prompt is enough: the decode CUDA graph shape is independent of prompt length.
    n_warmup_prompt_tokens = max(1, min(8, int(dyn_ctx.max_tokens)))
    normalized_request_counts = tuple(sorted({int(count) for count in request_counts}))
    if not normalized_request_counts or normalized_request_counts[0] < 1:
        raise ValueError("CUDA graph warmup request counts must contain only positive values")
    with _native_torch_context(nd):
        for warmup_request_count in normalized_request_counts:
            try:
                for request_id in range(warmup_request_count):
                    req = DynamicInferenceRequest(
                        request_id=request_id,
                        prompt_tokens=torch.zeros(n_warmup_prompt_tokens, dtype=torch.int64, device=device),
                        sampling_params=SamplingParams(num_tokens_to_generate=8, termination_id=-1),
                    )
                    dyn_ctx.add_request(req, prefill_chunk_length=n_warmup_prompt_tokens)
                slots = _normalize_new_request_slots_for_packed_hyena(dyn_ctx, warmup_request_count)
                bind_hyena_packed_views_to_dynamic_context_batch(hyena_model, dyn_ctx, request_slots=slots)
                dyn_ctx.evo2_batched_decode_enabled = warmup_request_count > 1
                # One prefill forward (eager; not graphed) seeds the Hyena recurrent state, then two decode
                # forwards: the first triggers graph capture, the second replays it so any capture/replay
                # mismatch surfaces here rather than on a user prompt.
                for _step in range(3):
                    dyn_ctx.initialize_attention_state()
                    input_ids, position_ids = dyn_ctx.current_input_and_position_ids()
                    try:
                        from megatron.core.inference.utils import InferenceMode

                        inference_mode_context = InferenceMode.active()
                    except ImportError:
                        inference_mode_context = contextlib.nullcontext()
                    with inference_mode_context:
                        _forward_native_dynamic_logits(nd, dyn_ctx, input_ids, position_ids)
                    dyn_ctx.update_requests(
                        torch.ones(warmup_request_count, dtype=torch.bool, device=device),
                        torch.zeros(warmup_request_count, dtype=torch.int64, device=device),
                    )
            finally:
                dyn_ctx.evo2_batched_decode_enabled = False
                dyn_ctx.reset()
    if rank == 0:
        logger.info(
            "[evo2-native-cg] warmed decode CUDA graph(s) for physical request count(s): %s",
            ",".join(str(count) for count in normalized_request_counts),
        )


def _validate_cuda_graph_capture(
    nd: Evo2NativeDynamicComponents,
    *,
    expected_request_counts: Collection[int],
) -> None:
    """Require every configured graph manager to own captured runners.

    The warmup performs one capture followed by one replay for every requested physical
    request count. Reaching this check therefore proves replay completed without falling back
    to eager execution; inspecting the runners additionally catches configurations
    which claimed graphs were enabled but never constructed a graphable module. Packed
    Hyena layers key graphs by active-request count, while paged-attention layers use a
    fixed max-request shape and therefore intentionally reuse one runner.
    """
    managers = [
        manager
        for module in nd.hyena_model.modules()
        if (manager := getattr(module, "cudagraph_manager", None)) is not None
    ]
    if not managers:
        raise RuntimeError("CUDA graphs were enabled but the Evo2 model has no CUDA graph manager")
    if nd.cuda_graph_scope == "block" and len(managers) != 1:
        raise RuntimeError(f"Block-scope CUDA graphs require exactly one Evo2 graph manager; found {len(managers)}")

    request_counts = frozenset(int(count) for count in expected_request_counts)
    if not request_counts or min(request_counts) < 1:
        raise ValueError("Expected CUDA graph request counts must contain only positive values")
    expected_shape_count = len(request_counts)
    expected_by_manager = {id(manager): expected_shape_count for manager in managers}
    if nd.cuda_graph_scope == "layer":
        decoder = getattr(nd.hyena_model, "decoder", None)
        for layer, layer_type in zip(
            getattr(decoder, "layers", ()),
            getattr(decoder, "layer_type_list", ()),
            strict=False,
        ):
            manager = getattr(layer, "cudagraph_manager", None)
            if manager is not None and layer_type == "*":
                expected_by_manager[id(manager)] = 1

    manager_runners = [list(getattr(manager, "cudagraph_runners", ())) for manager in managers]
    runners = [runner for owned_runners in manager_runners for runner in owned_runners]
    minimum_runner_count = sum(expected_by_manager[id(manager)] for manager in managers)
    recorded = [
        runner
        for runner in runners
        if bool(getattr(runner, "fwd_graph_recorded", False)) and bool(getattr(runner, "cudagraph_created", False))
    ]
    incomplete_manager = any(
        len(owned_runners) < expected_by_manager[id(manager)]
        for manager, owned_runners in zip(managers, manager_runners, strict=True)
    )
    if incomplete_manager or len(recorded) != len(runners):
        raise RuntimeError(
            "CUDA graph warmup was not fully captured: "
            f"scope={nd.cuda_graph_scope}, managers={len(managers)}, runners={len(runners)}, "
            f"recorded={len(recorded)}, expected_at_least={minimum_runner_count}"
        )

    nd.cuda_graph_manager_count = len(managers)
    nd.cuda_graph_runner_count = len(runners)
    nd.cuda_graph_recorded_count = len(recorded)
    nd.cuda_graph_replay_verified = True
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(
            "[evo2-native-cg] verified capture+replay: scope=%s managers=%d runners=%d",
            nd.cuda_graph_scope,
            len(managers),
            len(runners),
        )


def _model_storage_signature(model: torch.nn.Module) -> tuple[tuple[Any, ...], ...]:
    """Describe graph-visible registered tensor storage without tracking tensor values.

    CUDA graphs may safely observe optimizer updates made in place, but not a parameter or buffer
    rebound by CPU offload/reload. Address and layout distinguish those cases; tensor object identity
    and versions are deliberately excluded so harmless wrapper replacement and ordinary optimizer
    steps retain graph reuse.
    """
    signature = []
    seen: set[int] = set()
    for iterator_name in ("parameters", "buffers"):
        iterator = getattr(model, iterator_name, None)
        if not callable(iterator):
            continue
        for tensor in iterator():
            if not isinstance(tensor, torch.Tensor) or id(tensor) in seen:
                continue
            seen.add(id(tensor))
            signature.append(
                (
                    int(tensor.data_ptr()),
                    str(tensor.device),
                    tensor.dtype,
                    tuple(tensor.shape),
                    tuple(tensor.stride()),
                    int(tensor.storage_offset()),
                )
            )
    return tuple(signature)


def _record_cuda_graph_model_storage(nd: Evo2NativeDynamicComponents) -> None:
    """Record the model allocations used by the runners just captured and replay-verified."""
    nd.cuda_graph_model_storage_signature = _model_storage_signature(nd.hyena_model)
    nd.cuda_graph_force_recapture = False


def _graph_parallel_any(local_value: bool) -> bool:
    """OR a flag across this DP replica's TP, PP, and CP graph participants."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return bool(local_value)

    from megatron.core import parallel_state

    groups = []
    if (
        int(parallel_state.get_tensor_model_parallel_world_size())
        * int(parallel_state.get_pipeline_model_parallel_world_size())
        > 1
    ):
        # MCore's model-parallel group is TP x PP with DP and CP held fixed.
        groups.append(parallel_state.get_model_parallel_group())
    if int(parallel_state.get_context_parallel_world_size()) > 1:
        # Folding over CP after TP x PP propagates the OR over the full graph replica.
        groups.append(parallel_state.get_context_parallel_group())

    reduced_value = bool(local_value)
    for group in groups:
        backend = str(torch.distributed.get_backend(group)).lower()
        device = torch.device("cuda", torch.cuda.current_device()) if backend.endswith("nccl") else torch.device("cpu")
        value_tensor = torch.tensor([int(reduced_value)], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(value_tensor, op=torch.distributed.ReduceOp.MAX, group=group)
        reduced_value = bool(value_tensor.item())
    return reduced_value


def _invalidate_cuda_graphs_for_rebound_model_storage(nd: Evo2NativeDynamicComponents) -> bool:
    """Invalidate graph runners when captured model storage or quantized state changed."""
    if not nd.cuda_graphs_enabled:
        return False
    _restore_native_dynamic_cuda_graph_managers(nd)
    captured = getattr(nd, "cuda_graph_model_storage_signature", None)
    local_storage_changed = captured is not None and captured != _model_storage_signature(nd.hyena_model)
    local_recapture_required = local_storage_changed or bool(getattr(nd, "cuda_graph_force_recapture", False))
    if not _graph_parallel_any(local_recapture_required):
        return False

    _reset_layer_cuda_graphs(nd)
    if nd.shared_dyn_ctx is not None:
        nd.shared_dyn_ctx.evo2_warmed_cuda_graph_request_counts = frozenset()
    for context in nd.static_contexts.values():
        context.evo2_static_cuda_graph_warmed = False
        context.evo2_static_cuda_graph_replay_verified = False
    if int(os.environ.get("RANK", "0")) == 0:
        logger.info("[evo2-native-cg] captured model state changed; recapturing CUDA graphs before replay")
    return True


def _reset_layer_cuda_graphs(nd: Evo2NativeDynamicComponents) -> None:
    """Drop all captured per-layer CUDA graphs so the next warmup re-captures at the new context size.

    Needed to "grow" the dynamic context (mcore has no in-place resize): a larger context is a new
    object with a longer ``rotary_pos_emb``, so graphs captured against the previous one must go.
    mcore's module-level ``delete_cuda_graphs()`` resets the global record, each runner's recorded
    graph, and the shared mempool — but it does NOT clear each ``CudaGraphManager``'s per-instance
    runner list or custom-key lookup table, so a stale runner would still be found and replayed against
    the new context (raising "CUDA graph argument mismatch"). We clear the current and legacy lookup
    names defensively; with the global ``cudagraph_created`` flag also reset, the next decode creates a
    fresh runner and captures at the current shape.
    """
    for module in nd.hyena_model.modules():
        mgr = getattr(module, "cudagraph_manager", None)
        if mgr is None:
            continue
        if hasattr(mgr, "cudagraph_runners"):
            mgr.cudagraph_runners = []
        for lookup_name in ("custom_cudagraphs_lookup_table", "inference_cudagraphs_lookup_table"):
            lookup_table = getattr(mgr, lookup_name, None)
            if lookup_table is not None:
                lookup_table.clear()

    from megatron.core.transformer.cuda_graphs import delete_cuda_graphs

    delete_cuda_graphs()
    nd.cuda_graph_runner_count = 0
    nd.cuda_graph_recorded_count = 0
    nd.cuda_graph_replay_verified = False
    nd.cuda_graph_model_storage_signature = None


def _get_or_build_shared_dynamic_context(
    nd: Evo2NativeDynamicComponents,
    *,
    block_size_tokens: int,
    max_tokens: Optional[int],
    enable_chunked_prefill: bool,
    max_active_requests: int,
    cuda_graph_request_counts: Optional[Collection[int]] = None,
    device: torch.device,
) -> tuple[Any, _CudaPhaseStats, _CudaPhaseStats]:
    """Return the engine's persistent dynamic context, building (and graph-warming) it on first use.

    A single context is reused for the whole engine lifetime so the per-layer CUDA graphs captured
    during warmup stay valid across every prompt and every :func:`generate` call (mcore keys decode
    graphs by the context object plus a ``rotary_pos_emb`` tensor whose length equals
    ``max_sequence_length``, so both must stay constant). This mirrors mcore's
    ``DynamicInferenceEngine``, which holds one context and feeds many requests through it.

    It is rebuilt only when (a) the context-affecting options change, or (b) the engine budget
    ``nd.max_seq_length`` has grown beyond the cached context (auto mode grows on demand). mcore has
    no in-place resize, so "grow" means building a new, larger context; any graphs captured against
    the old one are dropped first via :func:`_reset_layer_cuda_graphs` and re-captured by the warmup.
    """
    from megatron.core.inference.config import InferenceConfig

    ctx_key = (
        int(block_size_tokens),
        None if max_tokens is None else int(max_tokens),
        bool(enable_chunked_prefill),
    )
    cached = nd.shared_dyn_ctx
    requested_graph_counts = frozenset(int(count) for count in (cuda_graph_request_counts or (max_active_requests,)))
    if not requested_graph_counts or min(requested_graph_counts) < 1:
        raise ValueError("CUDA graph request counts must contain only positive values")
    if max(requested_graph_counts) > int(max_active_requests):
        raise ValueError("CUDA graph request counts cannot exceed max_active_requests")
    _invalidate_cuda_graphs_for_rebound_model_storage(nd)
    if (
        cached is not None
        and nd.shared_dyn_ctx_key == ctx_key
        and int(cached.max_sequence_length) >= int(nd.max_seq_length)
        and int(cached.max_requests) >= int(max_active_requests)
    ):
        # Reuse the persistent context (it is big enough); reset() returns it to a clean state without
        # freeing the CUDA-graph-referenced buffers (it is explicitly designed for reuse-after-capture).
        cached.reset()
        capture_stats = _CudaPhaseStats()
        warmed_graph_counts = frozenset(getattr(cached, "evo2_warmed_cuda_graph_request_counts", ()))
        missing_graph_counts = requested_graph_counts - warmed_graph_counts
        if nd.cuda_graphs_enabled and missing_graph_counts:
            capture_started_at_s = _begin_cuda_phase()
            _warmup_native_dynamic_cuda_graphs(
                nd,
                cached,
                device,
                request_counts=missing_graph_counts,
            )
            warmed_graph_counts |= missing_graph_counts
            _validate_cuda_graph_capture(nd, expected_request_counts=warmed_graph_counts)
            _record_cuda_graph_model_storage(nd)
            cached.evo2_warmed_cuda_graph_request_counts = warmed_graph_counts
            capture_stats = _finish_cuda_phase(capture_started_at_s)
        return cached, _CudaPhaseStats(), capture_stats

    # First build, config change, or grow. Drop any graphs captured against the previous context
    # object so a stale graph can never be replayed against the new (larger) one.
    context_setup_started_at_s = _begin_cuda_phase()
    if nd.cuda_graphs_enabled and (cached is not None or nd.static_contexts):
        _reset_layer_cuda_graphs(nd)
        # Static contexts remember whether their graph was warmed. Once the shared
        # graph managers are reset, retaining one would make a later static call skip
        # capture and execute a real decode while MCore records against live state.
        nd.static_contexts.clear()

    hyena_model = nd.hyena_model
    # max_requests is kept at least tp-divisible and can be enlarged by the opt-in Evo2 batched
    # decode path. Size to the engine's full max_seq_length so the persistent context (and its
    # constant rotary length) can serve any prompt across any batch.
    tp = int(getattr(hyena_model.config, "tensor_model_parallel_size", 1) or 1)
    max_requests = max(tp, int(max_active_requests), 1)
    msl = int(nd.max_seq_length)
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
    dyn_ctx.materialize_only_last_token_logits = True
    dyn_ctx.evo2_max_batched_decode_requests = int(max_active_requests)
    dyn_ctx.evo2_warmed_cuda_graph_request_counts = frozenset()
    dyn_ctx.initialize_all_tensors()
    context_setup_stats = _finish_cuda_phase(context_setup_started_at_s)
    cuda_graph_capture_stats = _CudaPhaseStats()
    if nd.cuda_graphs_enabled:
        capture_started_at_s = _begin_cuda_phase(
            already_synchronized=True,
            boundary_time_s=context_setup_stats._ended_at_s,
        )
        _warmup_native_dynamic_cuda_graphs(
            nd,
            dyn_ctx,
            device,
            request_counts=requested_graph_counts,
        )
        _validate_cuda_graph_capture(nd, expected_request_counts=requested_graph_counts)
        _record_cuda_graph_model_storage(nd)
        dyn_ctx.evo2_warmed_cuda_graph_request_counts = requested_graph_counts
        cuda_graph_capture_stats = _finish_cuda_phase(capture_started_at_s)
    nd.shared_dyn_ctx = dyn_ctx
    nd.shared_dyn_ctx_key = ctx_key
    return dyn_ctx, context_setup_stats, cuda_graph_capture_stats


def _get_or_build_static_flash_context(
    nd: Evo2NativeDynamicComponents,
    *,
    batch_size: int,
    max_sequence_length: int,
    device: torch.device,
) -> tuple[Any, _CudaPhaseStats]:
    """Return a graph-pointer-stable static context for one exact batch shape."""
    from megatron.core.inference.contexts import StaticInferenceContext

    _invalidate_cuda_graphs_for_rebound_model_storage(nd)
    key = (int(batch_size), int(max_sequence_length))
    cached = nd.static_contexts.get(key)
    if cached is not None:
        # Bypass HyenaInferenceContext.reset(), which intentionally deletes ordinary
        # state dictionaries. These are persistent packed dictionaries whose registered
        # views must survive for CUDA-graph pointer stability.
        StaticInferenceContext.reset(cached)
        reset_hyena_packed_views_for_new_request(cached)
        return cached, _CudaPhaseStats()

    capacity_changed = any(cached_key[1] != key[1] for cached_key in nd.static_contexts)
    if capacity_changed or len(nd.static_contexts) >= _MAX_STATIC_FLASH_CONTEXTS:
        # Graph runners retain context-backed storage. Delete them before dropping
        # the only explicit context references, and force a shared dynamic context
        # to rebuild if callers switch inference backends on this model instance.
        if nd.cuda_graphs_enabled:
            _reset_layer_cuda_graphs(nd)
            nd.shared_dyn_ctx = None
            nd.shared_dyn_ctx_key = None
        nd.static_contexts.clear()

    started_at_s = _begin_cuda_phase()
    context = StaticInferenceContext(
        max_batch_size=int(batch_size),
        max_sequence_length=int(max_sequence_length),
    )
    context.materialize_only_last_token_logits = True
    bind_hyena_packed_views_to_static_context(
        nd.hyena_model,
        context,
        batch_size=int(batch_size),
        device=device,
    )
    context.evo2_static_cuda_graph_warmed = False
    context.evo2_static_cuda_graph_replay_verified = False
    nd.static_contexts[key] = context
    return context, _finish_cuda_phase(started_at_s)


def _static_hyena_state_snapshot(context: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Clone the small recurrent state so graph warmup cannot alter real prefill state."""
    return (
        context._evo2_hyena_conv_states.clone(),
        context._evo2_hyena_ssm_states.clone(),
    )


def _restore_static_hyena_state(
    context: Any,
    snapshot: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Restore graph-bound recurrent buffers without changing their addresses."""
    with torch.no_grad():
        context._evo2_hyena_conv_states.copy_(snapshot[0])
        context._evo2_hyena_ssm_states.copy_(snapshot[1])


def _static_graph_counts(nd: Evo2NativeDynamicComponents) -> tuple[int, int, int]:
    """Return manager, runner, and fully-recorded runner counts."""
    managers = [
        manager
        for module in nd.hyena_model.modules()
        if (manager := getattr(module, "cudagraph_manager", None)) is not None
    ]
    runners = [runner for manager in managers for runner in getattr(manager, "cudagraph_runners", ())]
    recorded = [
        runner
        for runner in runners
        if bool(getattr(runner, "fwd_graph_recorded", False)) and bool(getattr(runner, "cudagraph_created", False))
    ]
    return len(managers), len(runners), len(recorded)


def _generate_static_flash(
    components: Evo2InferenceComponents,
    prompts: List[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    return_log_probs: bool,
    ignore_eos: bool,
    preserve_eos_token: bool,
    strict_generation: bool,
    evo2_batched_decode_size: int,
    result_callback: Optional[Callable[[int, Any], None]],
) -> List[_NativeDynamicResult]:
    """Generate with static FlashAttention KV cache and fused Hyena decode.

    This backend is intentionally fixed-shape: every request in a decode group must
    have the same tokenized prompt length. It avoids dynamic scheduler bookkeeping,
    keeps sampled tokens on GPU for forced-length generation, and reuses one static
    context (including graph-bound KV/Hyena buffers) per batch/capacity shape.
    """
    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if not prompts:
        return []

    nd = components.native_dynamic
    tokenizer = components.tokenizer
    forward_model = nd.forward_model
    hyena_model = nd.hyena_model
    device = next(hyena_model.parameters()).device
    rank = int(os.environ.get("RANK", "0"))
    eff_top_k = max(0, int(top_k))
    eff_top_p = float(top_p) if top_p and top_p > 0 else 0.0
    sampling_rng = _sampling_rng_for_native_dynamic(nd, device)
    stop_token_ids = _native_stop_token_ids(tokenizer)
    tokenized_prompts = [list(tokenizer.tokenize(prompt)) for prompt in prompts]
    if any(not prompt for prompt in tokenized_prompts):
        raise ValueError("Static FlashAttention inference requires non-empty prompts")

    needed_max_sequence_length = max(
        _auto_max_seq_length_for(len(prompt), max_new_tokens) for prompt in tokenized_prompts
    )
    if nd.max_seq_length is None:
        nd.max_seq_length = needed_max_sequence_length
    elif nd.max_seq_length_is_auto and needed_max_sequence_length > nd.max_seq_length:
        nd.max_seq_length = needed_max_sequence_length
    if needed_max_sequence_length > int(nd.max_seq_length):
        raise ValueError(
            "Static FlashAttention generation exceeds max_seq_length: "
            f"needs {needed_max_sequence_length}, configured {nd.max_seq_length}"
        )

    engine_setup_stats = _CudaPhaseStats()
    if bool(getattr(nd, "engine_setup_stats_pending", False)):
        engine_setup_stats = getattr(nd, "engine_setup_stats", _CudaPhaseStats())
        nd.engine_setup_stats_pending = False
    generation_call_index = int(getattr(nd, "generation_call_index", 0))
    nd.generation_call_index = generation_call_index + 1
    batch_size = max(1, int(evo2_batched_decode_size))
    results: list[_NativeDynamicResult] = []
    previous_flash_decode = bool(getattr(hyena_model.config, "flash_decode", False))
    hyena_model.config.flash_decode = True

    try:
        for group_start in range(0, len(tokenized_prompts), batch_size):
            prompt_group = tokenized_prompts[group_start : group_start + batch_size]
            prompt_lengths = {len(prompt) for prompt in prompt_group}
            if len(prompt_lengths) != 1:
                raise ValueError(
                    "Static FlashAttention requires equal tokenized prompt lengths within each decode batch; "
                    f"group at index {group_start} has lengths {sorted(prompt_lengths)}"
                )
            prompt_length = next(iter(prompt_lengths))
            request_count = len(prompt_group)
            context, context_setup_stats = _get_or_build_static_flash_context(
                nd,
                batch_size=request_count,
                max_sequence_length=int(nd.max_seq_length),
                device=device,
            )
            prompt_tensor = torch.tensor(prompt_group, dtype=torch.long, device=device)
            timings: dict[str, Any] = {}
            memory: dict[str, int] = {}
            _record_phase_stats(
                timings,
                memory,
                "engine_setup",
                engine_setup_stats if group_start == 0 else _CudaPhaseStats(),
            )
            _record_phase_stats(
                timings,
                memory,
                "context_setup",
                context_setup_stats,
            )

            try:
                from megatron.core.inference.utils import InferenceMode

                inference_mode_context = InferenceMode.active
            except ImportError:  # pragma: no cover - installed MCore always provides it
                inference_mode_context = contextlib.nullcontext

            def _forward(input_ids: torch.Tensor) -> torch.Tensor:
                with inference_mode_context():
                    return forward_model(
                        input_ids,
                        None,
                        None,
                        inference_context=context,
                        runtime_gather_output=True,
                    )

            prefill_started_at_s = _begin_cuda_phase()
            with torch.inference_mode():
                logits = _forward(prompt_tensor)
            prefill_stats = _finish_cuda_phase(prefill_started_at_s)
            _record_phase_stats(timings, memory, "prefill", prefill_stats)
            context.increment_sequence_len_offset(prompt_length)
            context.enable_decode_mode()

            graph_capture_stats = _CudaPhaseStats()
            if nd.cuda_graphs_enabled and max_new_tokens > 1 and not context.evo2_static_cuda_graph_warmed:
                snapshot = _static_hyena_state_snapshot(context)
                managers_before, runners_before, _recorded_before = _static_graph_counts(nd)
                if managers_before == 0:
                    raise RuntimeError("Static FlashAttention CUDA graphs were enabled but no graph manager exists")
                if nd.cuda_graph_scope == "block" and managers_before != 1:
                    raise RuntimeError(f"Static block CUDA graphs require one manager, found {managers_before}")
                capture_started_at_s = _begin_cuda_phase()
                dummy_tokens = torch.zeros((request_count, 1), dtype=torch.long, device=device)
                with torch.inference_mode():
                    _forward(dummy_tokens)  # capture
                    _restore_static_hyena_state(context, snapshot)
                    _forward(dummy_tokens)  # replay proof
                    _restore_static_hyena_state(context, snapshot)
                graph_capture_stats = _finish_cuda_phase(capture_started_at_s)
                managers_after, runners_after, recorded_after = _static_graph_counts(nd)
                if runners_after <= runners_before or recorded_after != runners_after:
                    raise RuntimeError(
                        "Static FlashAttention CUDA graph capture/replay was not verified: "
                        f"managers={managers_after}, runners={runners_after}, recorded={recorded_after}, "
                        f"runners_before={runners_before}"
                    )
                context.evo2_static_cuda_graph_warmed = True
                context.evo2_static_cuda_graph_replay_verified = True
                _record_cuda_graph_model_storage(nd)
            _record_phase_stats(timings, memory, "cuda_graph_capture", graph_capture_stats)

            generated_ids: list[list[int]] = [[] for _ in range(request_count)]
            generated_logprobs: list[list[float]] = [[] for _ in range(request_count)]
            stopped = [False] * request_count
            decode_started_at_s = _begin_cuda_phase()
            if max_new_tokens > 0:
                next_logits = logits[:, -1, :].float()
                stop_token_mask = _stop_token_mask(next_logits, stop_token_ids) if ignore_eos else None
                # Forced-length generation never needs per-step host decisions. Keep
                # tokens/log-probs resident and perform one transfer after the loop.
                gpu_resident_results = bool(ignore_eos)
                token_buffer = (
                    torch.empty((max_new_tokens, request_count), dtype=torch.long, device=device)
                    if gpu_resident_results
                    else None
                )
                logprob_buffer = (
                    torch.empty((max_new_tokens, request_count), dtype=torch.float32, device=device)
                    if gpu_resident_results and return_log_probs
                    else None
                )
                completed_steps = 0

                for token_index in range(max_new_tokens):
                    sampling_logits = (
                        _suppress_stop_token_logits(
                            next_logits,
                            stop_token_ids,
                            stop_token_mask=stop_token_mask,
                        )
                        if ignore_eos
                        else next_logits
                    )
                    log_probs = _sampling_log_probs_from_logits(
                        sampling_logits,
                        temperature=float(temperature),
                        top_k=eff_top_k,
                        top_p=eff_top_p,
                        vocab_size=tokenizer.vocab_size,
                    )
                    sampled = _sample_from_log_probs(
                        log_probs,
                        top_k=eff_top_k,
                        generator=sampling_rng,
                        vocab_size=tokenizer.vocab_size,
                    ).to(dtype=torch.long)
                    completed_steps = token_index + 1

                    if gpu_resident_results:
                        token_buffer[token_index].copy_(sampled)
                        if logprob_buffer is not None:
                            logprob_buffer[token_index].copy_(log_probs.gather(1, sampled.view(-1, 1)).squeeze(1))
                    else:
                        sampled_ids = sampled.detach().cpu().tolist()
                        selected_log_probs = (
                            _selected_log_probs_for_sampled_tokens(log_probs, sampled)
                            if return_log_probs
                            else [0.0] * request_count
                        )
                        for request_index, token_id in enumerate(sampled_ids):
                            if stopped[request_index]:
                                continue
                            append_token, stop_request = _sampled_token_action(
                                int(token_id),
                                stop_token_ids,
                                ignore_eos=False,
                                preserve_eos_token=preserve_eos_token,
                            )
                            if append_token:
                                generated_ids[request_index].append(int(token_id))
                                if return_log_probs:
                                    generated_logprobs[request_index].append(float(selected_log_probs[request_index]))
                            stopped[request_index] = stop_request
                        if all(stopped):
                            break

                    if token_index + 1 < max_new_tokens:
                        with torch.inference_mode():
                            logits = _forward(sampled.unsqueeze(1))
                        context.increment_sequence_len_offset(1)
                        next_logits = logits[:, -1, :].float()

                if gpu_resident_results:
                    host_tokens = token_buffer[:completed_steps].transpose(0, 1).detach().cpu().tolist()
                    host_logprobs = (
                        logprob_buffer[:completed_steps].transpose(0, 1).detach().cpu().tolist()
                        if logprob_buffer is not None
                        else None
                    )
                    for request_index in range(request_count):
                        generated_ids[request_index] = [int(token) for token in host_tokens[request_index]]
                        if host_logprobs is not None:
                            generated_logprobs[request_index] = [
                                float(value) for value in host_logprobs[request_index]
                            ]

            decode_stats = _finish_cuda_phase(decode_started_at_s)
            _record_phase_stats(timings, memory, "decode", decode_stats)
            timings["generation_elapsed_s"] = prefill_stats.elapsed_s + decode_stats.elapsed_s
            manager_count, runner_count, recorded_count = _static_graph_counts(nd)
            timings.update(
                {
                    "timing_scope": "static_flash_generation_group",
                    "timing_group_id": (f"static-call-{generation_call_index:08d}-group-{group_start:08d}"),
                    "timing_request_count": request_count,
                    "precision_kind": nd.precision_kind,
                    "precision_parameter_storage": nd.precision_parameter_storage,
                    "cuda_graph_scope": nd.cuda_graph_scope,
                    "cuda_graph_manager_count": manager_count,
                    "cuda_graph_runner_count": runner_count,
                    "cuda_graph_recorded_count": recorded_count,
                    "cuda_graph_replay_verified": bool(context.evo2_static_cuda_graph_replay_verified),
                }
            )
            timings["total_elapsed_s"] = sum(
                float(timings.get(f"{phase}_elapsed_s", 0.0))
                for phase in ("engine_setup", "context_setup", "cuda_graph_capture", "prefill", "decode")
            )
            memory["generation_peak_allocated_bytes"] = max(
                int(memory.get("prefill_peak_allocated_bytes", 0)),
                int(memory.get("decode_peak_allocated_bytes", 0)),
            )
            memory["generation_peak_reserved_bytes"] = max(
                int(memory.get("prefill_peak_reserved_bytes", 0)),
                int(memory.get("decode_peak_reserved_bytes", 0)),
            )
            memory["total_peak_allocated_bytes"] = max(
                (value for key, value in memory.items() if key.endswith("_peak_allocated_bytes")),
                default=0,
            )
            memory["total_peak_reserved_bytes"] = max(
                (value for key, value in memory.items() if key.endswith("_peak_reserved_bytes")),
                default=0,
            )

            for request_index, prompt_tokens in enumerate(prompt_group):
                request_tokens = generated_ids[request_index]
                stopped_on_eos = stopped[request_index]
                if strict_generation and not stopped_on_eos and len(request_tokens) != max_new_tokens:
                    raise RuntimeError(
                        "Strict static FlashAttention generation ended short: "
                        f"request {group_start + request_index} produced "
                        f"{len(request_tokens)} / {max_new_tokens} tokens"
                    )
                result = _NativeDynamicResult(
                    generated_text=tokenizer.detokenize(request_tokens) if request_tokens else "",
                    generated_length=len(request_tokens),
                    prompt_tokens=prompt_tokens,
                    generated_tokens=request_tokens,
                    generated_log_probs=(generated_logprobs[request_index] if return_log_probs else None),
                    finish_reason="stop" if stopped_on_eos else "length",
                    stopped_on_eos=stopped_on_eos,
                    truncated=not stopped_on_eos and len(request_tokens) >= max_new_tokens,
                    timings=dict(timings),
                    memory=dict(memory),
                )
                results.append(result)
                if result_callback is not None:
                    result_callback(group_start + request_index, result)
            if rank == 0:
                logger.info(
                    "[evo2-static-flash] requests=%d prompt_tokens=%d generated_tokens=%d graph_replay=%s",
                    request_count,
                    prompt_length * request_count,
                    sum(len(tokens) for tokens in generated_ids),
                    context.evo2_static_cuda_graph_replay_verified,
                )
    finally:
        hyena_model.config.flash_decode = previous_flash_decode

    return results


def _generate_native_dynamic(
    components: Evo2InferenceComponents,
    prompts: List[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    return_log_probs: bool,
    ignore_eos: bool,
    preserve_eos_token: bool,
    strict_generation: bool,
    enable_chunked_prefill: bool,
    inference_dynamic_batching_max_tokens: Optional[int],
    inference_dynamic_batching_block_size: int,
    evo2_batched_decode_size: int,
    result_callback: Optional[Callable[[int, Any], None]],
) -> List[_NativeDynamicResult]:
    """Drive standalone Evo2 (text→DNA) generation through the native mcore dynamic engine.

    A single mcore ``DynamicInferenceContext`` is reused across prompts, driven through the request
    lifecycle once per prompt:
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
    from megatron.core.inference.contexts.dynamic_context import (
        BlockOverflowError,
        MaxSequenceLengthOverflowError,
        TokenOverflowError,
    )
    from megatron.core.inference.inference_request import DynamicInferenceRequest

    nd = components.native_dynamic
    hyena_model = nd.hyena_model
    tokenizer = components.tokenizer
    device = next(hyena_model.parameters()).device
    rank = int(os.environ.get("RANK", "0"))

    # Match the documented sampling pipeline: temperature, then top-k, then top-p.
    eff_top_k = max(0, int(top_k))
    eff_top_p = float(top_p) if top_p and top_p > 0 else 0.0
    sampling_rng = _sampling_rng_for_native_dynamic(nd, device)

    results: List[_NativeDynamicResult] = []
    if not prompts:
        return results

    # Tokenize every prompt up front so the token-budget checks below run before any generation and
    # so the shared context's max-token budget can be validated against the longest prompt.
    tokenized_prompts: List[List[int]] = [list(tokenizer.tokenize(prompt)) for prompt in prompts]
    max_n_prompt = max(len(toks) for toks in tokenized_prompts)
    batched_decode_size = max(1, int(evo2_batched_decode_size))
    batched_prefill_request_count = min(batched_decode_size, len(tokenized_prompts))
    cuda_graph_request_counts = _physical_request_counts(len(tokenized_prompts), batched_decode_size)
    batched_prefill_tokens = max(
        sum(
            len(prompt_tokens)
            for prompt_tokens in tokenized_prompts[group_start : group_start + batched_prefill_request_count]
        )
        for group_start in range(0, len(tokenized_prompts), batched_prefill_request_count)
    )

    block_size_tokens = int(inference_dynamic_batching_block_size)
    if block_size_tokens <= 0:
        raise ValueError(f"inference_dynamic_batching_block_size must be positive, got {block_size_tokens}")
    max_tokens = inference_dynamic_batching_max_tokens
    if max_tokens is not None:
        max_tokens = int(max_tokens)
        if max_tokens <= 0:
            raise ValueError(f"inference_dynamic_batching_max_tokens must be positive, got {max_tokens}")
        if batched_prefill_tokens > max_tokens and not enable_chunked_prefill:
            raise ValueError(
                f"Batched prefill requires {batched_prefill_tokens} tokens "
                f"across at most {batched_prefill_request_count} request(s), but the configured "
                f"max token budget is {max_tokens}. Increase --inference-dynamic-batching-max-tokens or pass "
                "--enable-chunked-prefill."
            )

    # Resolve the engine sequence-length budget. In auto mode (max_seq_length=None at setup) it is
    # sized from the prompts on first use and then GROWS on demand: a later prompt that needs more
    # triggers a one-time rebuild of the dynamic context at a larger size (mcore has no in-place
    # resize) with a CUDA-graph re-capture, instead of failing. A manual --max-seq-length is a fixed
    # cap that supersedes auto-sizing and never grows (an over-long prompt then just stops early, as
    # before). The CLI may pre-size nd.max_seq_length from a prompt sample (_resolve_prompt_auto...).
    needed_max_seq_length = _auto_max_seq_length_for(max_n_prompt, max_new_tokens)
    if nd.max_seq_length is None:
        nd.max_seq_length = needed_max_seq_length
        if rank == 0:
            logger.info(
                "[evo2-native] auto-sized max_seq_length=%d (longest prompt=%d + max_new_tokens=%d + headroom=%d)",
                nd.max_seq_length,
                max_n_prompt,
                max_new_tokens,
                _AUTO_MAX_SEQ_LENGTH_HEADROOM,
            )
    elif nd.max_seq_length_is_auto and needed_max_seq_length > nd.max_seq_length:
        # Grow to cover this prompt, rounded up to a whole KV block so a small bump doesn't re-trigger
        # a rebuild on the very next slightly-longer prompt. _get_or_build_... rebuilds + re-captures.
        grown_max_seq_length = -(-needed_max_seq_length // block_size_tokens) * block_size_tokens
        if rank == 0:
            logger.info(
                "[evo2-native] growing max_seq_length %d -> %d to fit a larger prompt (%d tokens); this "
                "rebuilds the dynamic context and re-captures CUDA graphs once. Pass --max-seq-length to "
                "pin a fixed size and avoid regrows.",
                nd.max_seq_length,
                grown_max_seq_length,
                max_n_prompt,
            )
        nd.max_seq_length = grown_max_seq_length

    # One persistent dynamic context for the whole engine, reused across every prompt AND every
    # generate() call (mcore's DynamicInferenceEngine pattern: one context fed many requests). This is
    # required for CUDA-graph correctness — the per-layer decode graph, captured once during warmup,
    # freezes the context object identity and the rotary_pos_emb shape (== max_sequence_length), so the
    # same object and shape must be presented on every later decode step regardless of prompt or batch.
    # reset() (called between prompts in the loop below, and on reuse) returns the context to a clean
    # state without freeing the graph-referenced buffers.
    dyn_ctx, context_setup_stats, cuda_graph_capture_stats = _get_or_build_shared_dynamic_context(
        nd,
        block_size_tokens=block_size_tokens,
        max_tokens=max_tokens,
        enable_chunked_prefill=enable_chunked_prefill,
        max_active_requests=batched_decode_size,
        cuda_graph_request_counts=cuda_graph_request_counts,
        device=device,
    )
    engine_setup_stats = _CudaPhaseStats()
    if bool(getattr(nd, "engine_setup_stats_pending", False)):
        engine_setup_stats = getattr(nd, "engine_setup_stats", _CudaPhaseStats())
        nd.engine_setup_stats_pending = False
    generation_call_index = int(getattr(nd, "generation_call_index", 0))
    nd.generation_call_index = generation_call_index + 1
    if batched_prefill_tokens > dyn_ctx.max_tokens and not enable_chunked_prefill:
        raise ValueError(
            f"Batched prefill requires {batched_prefill_tokens} tokens "
            f"across at most {batched_prefill_request_count} request(s), but the dynamic context "
            f"max token budget is {dyn_ctx.max_tokens}. Increase --inference-dynamic-batching-max-tokens or pass "
            "--enable-chunked-prefill."
        )

    def _run_single_prompt(prompt_token_ids: list[int]) -> _NativeDynamicResult:
        n_prompt = len(prompt_token_ids)

        generated_ids: List[int] = []
        generated_logprobs: List[float] = []
        stop_token_ids = _native_stop_token_ids(tokenizer)
        stopped_on_eos = False
        stop_token_mask: Optional[torch.Tensor] = None
        timings = {
            "prefill_elapsed_s": 0.0,
            "decode_elapsed_s": 0.0,
            "generation_elapsed_s": 0.0,
        }
        memory: Dict[str, int] = {}
        phase_stats = {
            "prefill": _CudaPhaseStats(),
            "decode": _CudaPhaseStats(),
        }
        _record_phase_stats(timings, memory, "prefill", phase_stats["prefill"])
        _record_phase_stats(timings, memory, "decode", phase_stats["decode"])
        prefill_start: Optional[float] = None
        decode_start: Optional[float] = None
        timing_complete = False

        def _complete_phase_timing() -> None:
            nonlocal timing_complete
            if prefill_start is None or timing_complete:
                return
            if decode_start is None:
                phase_stats["prefill"] = _finish_cuda_phase(prefill_start)
                _record_phase_stats(timings, memory, "prefill", phase_stats["prefill"])
            else:
                phase_stats["decode"] = _finish_cuda_phase(decode_start)
                _record_phase_stats(timings, memory, "decode", phase_stats["decode"])
            timings["generation_elapsed_s"] = phase_stats["prefill"].elapsed_s + phase_stats["decode"].elapsed_s
            timing_complete = True

        def _forward_sample_update(*, count_generated: bool) -> bool:
            nonlocal stop_token_mask, stopped_on_eos
            dyn_ctx.initialize_attention_state()
            input_ids, position_ids = dyn_ctx.current_input_and_position_ids()
            try:
                from megatron.core.inference.utils import InferenceMode

                inference_mode_context = InferenceMode.active()
            except ImportError:
                inference_mode_context = contextlib.nullcontext()
            with inference_mode_context:
                logits = _forward_native_dynamic_logits(nd, dyn_ctx, input_ids, position_ids)
            # HyenaModel returns [B, S, vocab]; last_token_logits expects [1, S, H] and
            # selects the per-request final position -> [num_requests, vocab]. Sample in fp32 so
            # stochastic filters and logprobs do not depend on the model activation dtype.
            last_logits = _extract_generation_logits(dyn_ctx, logits)
            if ignore_eos and stop_token_mask is None:
                stop_token_mask = _stop_token_mask(last_logits, stop_token_ids)
            sampling_logits = (
                _suppress_stop_token_logits(
                    last_logits,
                    stop_token_ids,
                    stop_token_mask=stop_token_mask,
                )
                if ignore_eos
                else last_logits
            )
            sampled_log_probs = _sampling_log_probs_from_logits(
                sampling_logits,
                temperature=float(temperature),
                top_k=eff_top_k,
                top_p=eff_top_p,
                vocab_size=tokenizer.vocab_size,
            )
            sampled = _sample_from_log_probs(
                sampled_log_probs,
                top_k=eff_top_k,
                generator=sampling_rng,
                vocab_size=tokenizer.vocab_size,
            )
            selected_log_probs = (
                _selected_log_probs_for_sampled_tokens(sampled_log_probs, sampled) if return_log_probs else []
            )
            sampled_cpu = sampled.to(dtype=torch.int64).detach().cpu()
            if count_generated:
                next_tok_id = int(sampled_cpu[0].item())
                append_token, stop_request = _sampled_token_action(
                    next_tok_id,
                    stop_token_ids,
                    ignore_eos=ignore_eos,
                    preserve_eos_token=preserve_eos_token,
                )
                if append_token:
                    generated_ids.append(next_tok_id)
                    if return_log_probs:
                        generated_logprobs.append(float(selected_log_probs[0]))
                if stop_request:
                    stopped_on_eos = True
            active_after_sample = torch.tensor(
                [not count_generated or (not stopped_on_eos and len(generated_ids) < max_new_tokens)], dtype=torch.bool
            )
            dyn_ctx.update_requests(active_after_sample, sampled_cpu)
            return bool(active_after_sample[0].item())

        try:
            with _native_torch_context(nd):
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
                        if prefill_start is None:
                            prefill_start = _begin_cuda_phase()
                        _forward_sample_update(count_generated=not is_partial_chunk)
                        if not is_partial_chunk:
                            phase_stats["prefill"] = _finish_cuda_phase(prefill_start)
                            _record_phase_stats(timings, memory, "prefill", phase_stats["prefill"])
                            decode_start = _begin_cuda_phase(
                                already_synchronized=True,
                                boundary_time_s=phase_stats["prefill"]._ended_at_s,
                            )
                            req.remaining_prompt_tokens = req.remaining_prompt_tokens.new_empty(0)
                            break
                        req.remaining_prompt_tokens = req.remaining_prompt_tokens[chunk_len:]
                        req.finished_chunk_token_count += chunk_len

                    while len(generated_ids) < max_new_tokens and dyn_ctx.has_unfinished_requests():
                        _forward_sample_update(count_generated=True)
                    _complete_phase_timing()
        except (BlockOverflowError, TokenOverflowError, MaxSequenceLengthOverflowError) as exc:
            _complete_phase_timing()
            if strict_generation:
                raise
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
        return _NativeDynamicResult(
            generated_text=generated_text,
            generated_length=len(generated_ids),
            prompt_tokens=prompt_token_ids,
            generated_tokens=generated_ids,
            generated_log_probs=generated_logprobs if return_log_probs else None,
            finish_reason="stop" if stopped_on_eos else "length",
            stopped_on_eos=stopped_on_eos,
            truncated=not stopped_on_eos and len(generated_ids) >= max_new_tokens,
            timings=timings,
            memory=memory,
        )

    def _run_batched_prompts(prompt_token_id_batch: list[list[int]]) -> list[_NativeDynamicResult]:
        batch_request_count = len(prompt_token_id_batch)
        timings = {
            "prefill_elapsed_s": 0.0,
            "decode_elapsed_s": 0.0,
            "generation_elapsed_s": 0.0,
        }
        memory: Dict[str, int] = {}
        _record_phase_stats(timings, memory, "prefill", _CudaPhaseStats())
        _record_phase_stats(timings, memory, "decode", _CudaPhaseStats())
        if batch_request_count <= 1:
            return [_run_single_prompt(prompt_token_id_batch[0])]
        if max_new_tokens <= 0:
            return [
                _NativeDynamicResult(
                    generated_text="",
                    generated_length=0,
                    prompt_tokens=prompt_token_ids,
                    generated_tokens=[],
                    generated_log_probs=[] if return_log_probs else None,
                    finish_reason="length",
                    stopped_on_eos=False,
                    truncated=False,
                    timings=timings,
                    memory=memory,
                )
                for prompt_token_ids in prompt_token_id_batch
            ]
        if enable_chunked_prefill:
            raise ValueError("Evo2 batched decode does not support chunked prefill")
        prompt_lengths = [len(prompt_ids) for prompt_ids in prompt_token_id_batch]
        max_prompt_length = max(prompt_lengths)
        total_prompt_tokens = sum(prompt_lengths)
        generated_ids: list[list[int]] = [[] for _ in range(batch_request_count)]
        generated_logprobs: list[list[float]] = [[] for _ in range(batch_request_count)]
        stop_token_ids = _native_stop_token_ids(tokenizer)
        stop_token_mask: Optional[torch.Tensor] = None
        stopped_on_eos = [False for _ in range(batch_request_count)]
        stable_request_ids: Optional[torch.Tensor] = None
        stable_request_slots: Optional[torch.Tensor] = None

        def _forward_sample_update(*, count_generated: bool) -> bool:
            nonlocal stop_token_mask
            dyn_ctx.initialize_attention_state()
            input_ids, position_ids = dyn_ctx.current_input_and_position_ids()
            try:
                from megatron.core.inference.utils import InferenceMode

                inference_mode_context = InferenceMode.active()
            except ImportError:
                inference_mode_context = contextlib.nullcontext()
            with inference_mode_context:
                logits = _forward_native_dynamic_logits(nd, dyn_ctx, input_ids, position_ids)
            last_logits = _extract_generation_logits(dyn_ctx, logits)
            if last_logits.shape[0] < batch_request_count:
                raise RuntimeError(
                    "Evo2 batched decode expected one logit row per active request; "
                    f"got {last_logits.shape[0]} rows for {batch_request_count} requests"
                )
            active_logits = last_logits[:batch_request_count]
            if ignore_eos and stop_token_mask is None:
                stop_token_mask = _stop_token_mask(active_logits, stop_token_ids)
            sampling_logits = (
                _suppress_stop_token_logits(
                    active_logits,
                    stop_token_ids,
                    stop_token_mask=stop_token_mask,
                )
                if ignore_eos
                else active_logits
            )
            active_log_probs = _sampling_log_probs_from_logits(
                sampling_logits,
                temperature=float(temperature),
                top_k=eff_top_k,
                top_p=eff_top_p,
                vocab_size=tokenizer.vocab_size,
            )
            sampled = _sample_from_log_probs(
                active_log_probs,
                top_k=eff_top_k,
                generator=sampling_rng,
                vocab_size=tokenizer.vocab_size,
            )
            selected_log_probs = (
                _selected_log_probs_for_sampled_tokens(active_log_probs, sampled) if return_log_probs else []
            )
            sampled_cpu = sampled.to(dtype=torch.int64).detach().cpu()
            sampled_ids = sampled_cpu.tolist()
            if count_generated:
                for request_idx, next_tok_id in enumerate(sampled_ids):
                    if stopped_on_eos[request_idx] or len(generated_ids[request_idx]) >= max_new_tokens:
                        continue
                    append_token, stop_request = _sampled_token_action(
                        next_tok_id,
                        stop_token_ids,
                        ignore_eos=ignore_eos,
                        preserve_eos_token=preserve_eos_token,
                    )
                    if append_token:
                        generated_ids[request_idx].append(next_tok_id)
                        if return_log_probs:
                            generated_logprobs[request_idx].append(float(selected_log_probs[request_idx]))
                    if stop_request:
                        stopped_on_eos[request_idx] = True

            # EOD completion is logical: keep every physical row active until the group finishes
            # and ignore later samples for stopped rows above. This preserves one fixed graph shape
            # and each request's recorded action/log-probability trajectory without compaction.
            keep_group_active = (not count_generated) or any(
                not stopped_on_eos[request_idx] and len(request_generated_ids) < max_new_tokens
                for request_idx, request_generated_ids in enumerate(generated_ids)
            )
            active_after_sample = torch.full((batch_request_count,), keep_group_active, dtype=torch.bool)
            rollover_count = (
                _reserve_packed_decode_rollover_blocks(dyn_ctx, batch_request_count) if keep_group_active else 0
            )
            dyn_ctx.update_requests(active_after_sample, sampled_cpu)
            if int(getattr(dyn_ctx, "paused_request_count", 0)) != 0:
                raise RuntimeError("Evo2 batched decode does not yet support paused dynamic requests")
            if rollover_count:
                request_ids_are_stable = stable_request_ids is not None and torch.equal(
                    dyn_ctx.request_ids[:batch_request_count], stable_request_ids
                )
                request_slots_are_stable = stable_request_slots is not None and torch.equal(
                    dyn_ctx.mamba_metadata.request_to_mamba_state_idx[:batch_request_count], stable_request_slots
                )
                if not request_ids_are_stable or not request_slots_are_stable:
                    raise RuntimeError(
                        "MCore reordered packed Evo2 request rows or Hyena state slots during paged-KV rollover"
                    )
            return keep_group_active

        try:
            with _native_torch_context(nd):
                for request_idx, prompt_token_ids in enumerate(prompt_token_id_batch):
                    req = DynamicInferenceRequest(
                        request_id=request_idx,
                        prompt_tokens=torch.tensor(prompt_token_ids, dtype=torch.int64, device=device),
                        sampling_params=SamplingParams(num_tokens_to_generate=max_new_tokens, termination_id=-1),
                    )
                    dyn_ctx.add_request(req, prefill_chunk_length=len(prompt_token_ids))

                slots = _normalize_new_request_slots_for_packed_hyena(dyn_ctx, batch_request_count)
                bind_hyena_packed_views_to_dynamic_context_batch(hyena_model, dyn_ctx, request_slots=slots)
                stable_request_ids = dyn_ctx.request_ids[:batch_request_count].clone()
                stable_request_slots = slots.clone()
                dyn_ctx.evo2_batched_decode_enabled = True
                if rank == 0:
                    logger.info(
                        "[evo2-native] batched prompt prefill: requests=%d, total_tokens=%d, max_length=%d, "
                        "layout=packed",
                        batch_request_count,
                        total_prompt_tokens,
                        max_prompt_length,
                    )

                prefill_started_at_s = _begin_cuda_phase()
                _forward_sample_update(count_generated=True)
                prefill_stats = _finish_cuda_phase(prefill_started_at_s)
                _record_phase_stats(timings, memory, "prefill", prefill_stats)
                decode_started_at_s = _begin_cuda_phase(
                    already_synchronized=True,
                    boundary_time_s=prefill_stats._ended_at_s,
                )
                while any(len(request_generated_ids) < max_new_tokens for request_generated_ids in generated_ids):
                    if not dyn_ctx.has_unfinished_requests():
                        break
                    _forward_sample_update(count_generated=True)
                decode_stats = _finish_cuda_phase(decode_started_at_s)
                _record_phase_stats(timings, memory, "decode", decode_stats)
                timings["generation_elapsed_s"] = prefill_stats.elapsed_s + decode_stats.elapsed_s
        finally:
            dyn_ctx.evo2_batched_decode_enabled = False
            dyn_ctx.reset()

        return [
            _NativeDynamicResult(
                generated_text=tokenizer.detokenize(request_generated_ids) if request_generated_ids else "",
                generated_length=len(request_generated_ids),
                prompt_tokens=prompt_token_ids,
                generated_tokens=request_generated_ids,
                generated_log_probs=generated_logprobs[request_idx] if return_log_probs else None,
                finish_reason="stop" if stopped_on_eos[request_idx] else "length",
                stopped_on_eos=stopped_on_eos[request_idx],
                truncated=not stopped_on_eos[request_idx] and len(request_generated_ids) >= max_new_tokens,
                timings=dict(timings),
                memory=dict(memory),
            )
            for request_idx, (prompt_token_ids, request_generated_ids) in enumerate(
                zip(prompt_token_id_batch, generated_ids)
            )
        ]

    if batched_decode_size > 1 and rank == 0:
        logger.info("[evo2-native] opt-in batched decode active: size=%d", batched_decode_size)

    def _append_results(group_results: list[_NativeDynamicResult], *, prompt_offset: int) -> None:
        group_timings = dict(group_results[0].timings or {})
        group_memory = dict(group_results[0].memory or {})
        generation_completion_tokens = sum(len(result.generated_tokens or ()) for result in group_results)
        prefill_completion_tokens = sum(bool(result.generated_tokens) for result in group_results)
        decode_completion_tokens = generation_completion_tokens - prefill_completion_tokens
        generation_elapsed_s = float(group_timings.get("generation_elapsed_s", 0.0))
        decode_elapsed_s = float(group_timings.get("decode_elapsed_s", 0.0))
        setup_phase_stats = {
            "engine_setup": engine_setup_stats if prompt_offset == 0 else _CudaPhaseStats(),
            "context_setup": context_setup_stats if prompt_offset == 0 else _CudaPhaseStats(),
            "cuda_graph_capture": cuda_graph_capture_stats if prompt_offset == 0 else _CudaPhaseStats(),
        }
        for phase_name, stats in setup_phase_stats.items():
            _record_phase_stats(group_timings, group_memory, phase_name, stats)
        group_memory["generation_peak_allocated_bytes"] = max(
            int(group_memory.get("prefill_peak_allocated_bytes", 0)),
            int(group_memory.get("decode_peak_allocated_bytes", 0)),
        )
        group_memory["generation_peak_reserved_bytes"] = max(
            int(group_memory.get("prefill_peak_reserved_bytes", 0)),
            int(group_memory.get("decode_peak_reserved_bytes", 0)),
        )
        group_timings.update(
            {
                "timing_scope": "native_generation_group",
                "timing_group_id": (f"native-call-{generation_call_index:08d}-group-{prompt_offset:08d}"),
                "timing_request_count": len(group_results),
                "precision_kind": nd.precision_kind,
                "precision_parameter_storage": nd.precision_parameter_storage,
                "cuda_graph_scope": nd.cuda_graph_scope,
                "cuda_graph_manager_count": nd.cuda_graph_manager_count,
                "cuda_graph_runner_count": nd.cuda_graph_runner_count,
                "cuda_graph_recorded_count": nd.cuda_graph_recorded_count,
                "cuda_graph_replay_verified": nd.cuda_graph_replay_verified,
                "phase_timing_exact": _CUDA_PHASE_EVIDENCE_ENABLED,
                "generation_completion_tokens": generation_completion_tokens,
                "decode_completion_tokens": decode_completion_tokens,
                "generation_completion_tokens_per_s": (
                    generation_completion_tokens / generation_elapsed_s if generation_elapsed_s > 0 else 0.0
                ),
                "decode_completion_tokens_per_s": (
                    decode_completion_tokens / decode_elapsed_s if decode_elapsed_s > 0 else 0.0
                ),
            }
        )
        group_timings["total_elapsed_s"] = sum(
            float(group_timings.get(f"{phase_name}_elapsed_s", 0.0))
            for phase_name in (
                "engine_setup",
                "context_setup",
                "cuda_graph_capture",
                "prefill",
                "decode",
            )
        )
        group_memory["total_peak_allocated_bytes"] = max(
            (
                value
                for key, value in group_memory.items()
                if key.endswith("_peak_allocated_bytes") and key != "total_peak_allocated_bytes"
            ),
            default=0,
        )
        group_memory["total_peak_reserved_bytes"] = max(
            (
                value
                for key, value in group_memory.items()
                if key.endswith("_peak_reserved_bytes") and key != "total_peak_reserved_bytes"
            ),
            default=0,
        )
        for local_idx, result in enumerate(group_results):
            prompt_idx = prompt_offset + local_idx
            result.timings = dict(group_timings)
            result.memory = dict(group_memory)
            if strict_generation:
                generated_token_count = len(result.generated_tokens or [])
                stopped_early_on_eos = bool(result.stopped_on_eos) and generated_token_count < max_new_tokens
                if generated_token_count != max_new_tokens and not stopped_early_on_eos:
                    raise RuntimeError(
                        "Strict Evo2 generation expected exactly "
                        f"{max_new_tokens} generated tokens or an explicit EOS stop for prompt {prompt_idx}, "
                        f"got {generated_token_count}"
                    )
                if return_log_probs and result.generated_log_probs is None:
                    raise RuntimeError(
                        f"Strict Evo2 generation is missing requested chosen-token log-probs for prompt {prompt_idx}"
                    )
                if result.generated_log_probs is not None and len(result.generated_log_probs) != generated_token_count:
                    raise RuntimeError(
                        "Strict Evo2 generation returned mismatched token/log-prob lengths for "
                        f"prompt {prompt_idx}: {generated_token_count} tokens != "
                        f"{len(result.generated_log_probs)} log-probs"
                    )
                for logprob_idx, logprob in enumerate(result.generated_log_probs or []):
                    try:
                        is_finite = math.isfinite(float(logprob))
                    except (TypeError, ValueError):
                        is_finite = False
                    if not is_finite:
                        raise RuntimeError(
                            "Strict Evo2 generation returned a non-finite chosen-token log-prob "
                            f"for prompt {prompt_idx} at generated token {logprob_idx}: {logprob!r}"
                        )
            results.append(result)
            if result_callback is not None:
                result_callback(prompt_idx, result)

    for group_start in range(0, len(tokenized_prompts), batched_decode_size):
        group = tokenized_prompts[group_start : group_start + batched_decode_size]
        if batched_decode_size <= 1 or len(group) <= 1:
            _append_results([_run_single_prompt(group[0])], prompt_offset=group_start)
            continue
        try:
            _append_results(_run_batched_prompts(group), prompt_offset=group_start)
        except Exception as exc:
            if strict_generation:
                raise
            if rank == 0:
                logger.exception(
                    "[evo2-native] batched decode group failed (%r); falling back to single-request decode",
                    exc,
                )
            fallback_results = [_run_single_prompt(prompt_token_ids) for prompt_token_ids in group]
            _append_results(fallback_results, prompt_offset=group_start)

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
    prompt_token_ids = result.prompt_tokens if result.prompt_tokens is not None else []
    if hasattr(prompt_token_ids, "tolist"):
        prompt_token_ids = prompt_token_ids.tolist()
    prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]
    completion_token_ids = getattr(result, "generated_tokens", None)
    if completion_token_ids is None:
        completion_token_ids = getattr(result, "generated_token_ids", None)
    if completion_token_ids is None:
        completion_token_ids = []
    if hasattr(completion_token_ids, "tolist"):
        completion_token_ids = completion_token_ids.tolist()
    completion_token_ids = [int(token_id) for token_id in completion_token_ids]

    finish_reason = getattr(result, "finish_reason", None)
    if finish_reason is None:
        finish_reason = "length" if generated_length >= max_new_tokens else "stop"

    record: Dict[str, Any] = {
        "id": request_id,
        "prompt": prompt,
        "completion": generated_text,
        "prompt_token_ids": prompt_token_ids,
        "completion_token_ids": completion_token_ids,
        "finish_reason": finish_reason,
        "timings": dict(getattr(result, "timings", None) or {}),
        "memory": dict(getattr(result, "memory", None) or {}),
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
    ap.add_argument("--top-p", type=float, default=0.0, help="Top-p nucleus sampling after top-k (0 = disabled)")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    ap.add_argument(
        "--return-log-probs",
        action="store_true",
        default=False,
        help="Include per-token log probabilities in JSONL output",
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        default=False,
        help="Omit sampled EOS tokens and generate exactly --max-new-tokens tokens",
    )
    ap.add_argument(
        "--preserve-eos-token",
        action="store_true",
        default=False,
        help="Include a sampled terminal EOS/EOD token (and its log-probability when requested)",
    )
    ap.add_argument(
        "--strict-generation",
        action="store_true",
        default=False,
        help="Fail on context overflow, batched fallback, or incomplete generation evidence",
    )

    # Parallelism arguments
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism")
    ap.add_argument("--pipeline-model-parallel-size", type=int, default=1, help="Pipeline parallelism")
    ap.add_argument("--context-parallel-size", type=int, default=1, help="Context parallelism")
    ap.add_argument(
        "--context-parallel-comm-type",
        choices=CONTEXT_PARALLEL_COMM_TYPES,
        default=None,
        help=(
            "Runtime TE context-parallel attention transport. P2P is the default; A2A offers tighter "
            "numerical parity. Any value serialized in the checkpoint is ignored."
        ),
    )

    # Output arguments
    ap.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Save results as JSONL (one result object per line)",
    )
    ap.add_argument(
        "--stream-output",
        action="store_true",
        default=False,
        help=(
            "When --output-file is set, write and flush each JSONL result as soon as it is generated. "
            "Strict generation writes <output-file>.partial and atomically promotes it after success."
        ),
    )

    # Precision arguments
    ap.add_argument("--mixed-precision-recipe", type=str, default=None, help="Override precision recipe")
    ap.add_argument(
        "--quantized-param-storage",
        choices=["recipe", "bf16"],
        default="recipe",
        help="For FP8/FP4 recipes, preserve native quantized parameter storage or retain BF16 parameters "
        "while quantizing GEMMs. BF16 storage reduces checkpoint-load peak memory and is a measured fallback.",
    )
    ap.add_argument(
        "--fp8-all-layers",
        action="store_true",
        help="Apply the selected global TE FP8 recipe to every compatible linear, including the first/last blocks",
    )
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
        help="Max sequence length (a manual cap; supersedes auto-sizing). When omitted, resolved as: "
        "EVO2_MAX_SEQ_LEN env var > auto-sized from the prompt token lengths (longest sampled prompt "
        "+ --max-new-tokens). The dynamic context is CUDA-graph-pinned and cannot grow once set.",
    )
    ap.add_argument(
        "--max-seq-length-num-prompts",
        type=int,
        default=_DEFAULT_AUTO_MAX_SEQ_LENGTH_NUM_PROMPTS,
        help="When --max-seq-length is auto (omitted), size the context from the longest of the first "
        f"N prompts (default {_DEFAULT_AUTO_MAX_SEQ_LENGTH_NUM_PROMPTS}; pass 0 to scan all prompts). A "
        "longer prompt beyond the first N grows the context on demand (one-time rebuild + CUDA-graph "
        "re-capture); set --max-seq-length to pin a fixed size and avoid regrows.",
    )
    ap.add_argument(
        "--prompt-batch-size",
        type=int,
        default=None,
        help="Number of prompt-file rows to decode concurrently in the Evo2 native path. This does "
        "not change the number of generations, which is exactly the number of prompt-file rows.",
    )
    ap.add_argument(
        "--evo2-batched-decode-size",
        dest="evo2_batched_decode_size",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--max-batch-size",
        dest="max_batch_size",
        type=int,
        default=None,
        help="Maximum prompt-file rows per generate() call (prompt-file chunk size). This does not set decode "
        "concurrency; use --prompt-batch-size for that.",
    )
    ap.add_argument(
        "--use-subquadratic-ops",
        action="store_true",
        default=False,
        help="Use legacy fused Hyena kernels for static-Flash rectangular/eager prefill. "
        "Ignored by the dynamic backend, whose segmented prefill and fused decode keep CUDA graphs enabled.",
    )
    ap.add_argument(
        "--cuda-graph-impl",
        choices=["none", "local"],
        default="local",
        help="CUDA-graph implementation for dynamic decode: 'local' (MCore graphs, default) or 'none' "
        "(eager decode, no graph capture). 'none' is mainly for debugging / un-graphed reference runs.",
    )
    ap.add_argument(
        "--inference-backend",
        choices=["dynamic", "static-flash"],
        default="dynamic",
        help="Decode backend: packed paged-KV dynamic inference (default) or static-batch "
        "FlashAttention. Static Flash requires equal tokenized prompt lengths within each decode batch.",
    )
    ap.add_argument(
        "--cuda-graph-scope",
        choices=["block", "layer"],
        default="block",
        help="MCore local CUDA-graph granularity: one complete decoder-block graph (default) or "
        "one graph per layer (compatibility fallback). Global FP8/FP4 recipes automatically use "
        "layer scope. Ignored with --cuda-graph-impl none.",
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


def _resolve_prompt_auto_max_seq_length(
    components: Evo2InferenceComponents,
    prompt_texts: List[str],
    *,
    max_new_tokens: int,
    num_prompts: Optional[int] = None,
) -> int:
    """Auto-size the engine's initial ``max_seq_length`` from prompt token lengths.

    Sizes the persistent dynamic context to cover the longest of the first ``num_prompts`` prompts
    (``None`` or ``<= 0`` = all) plus the generation budget, using the engine tokenizer. Only the
    sampled prompts are tokenized here. A prompt beyond the sample that needs more is NOT a failure:
    :func:`_generate_native_dynamic` grows the context on demand (rebuild + CUDA-graph re-capture).
    Scanning a small leading sample just keeps startup cheap on large files and usually picks the
    final size in one shot. Sets and returns ``nd.max_seq_length``.
    """
    nd = components.native_dynamic
    tokenizer = components.tokenizer
    scan_all = num_prompts is None or int(num_prompts) <= 0
    sample = prompt_texts if scan_all else prompt_texts[: int(num_prompts)]
    auto_msl = max(_auto_max_seq_length_for(len(tokenizer.tokenize(text)), max_new_tokens) for text in sample)
    nd.max_seq_length = auto_msl
    if int(os.environ.get("RANK", "0")) == 0:
        if scan_all or len(sample) >= len(prompt_texts):
            logger.info(
                "[evo2-native] auto-sized max_seq_length=%d from all %d prompt(s) "
                "(longest + max_new_tokens=%d + headroom=%d)",
                auto_msl,
                len(prompt_texts),
                max_new_tokens,
                _AUTO_MAX_SEQ_LENGTH_HEADROOM,
            )
        else:
            logger.info(
                "[evo2-native] auto-sized max_seq_length=%d from the first %d of %d prompt(s); a longer "
                "later prompt will grow the context on demand (set --max-seq-length to pin a fixed size, "
                "or --max-seq-length-num-prompts 0 to scan all prompts up front)",
                auto_msl,
                len(sample),
                len(prompt_texts),
            )
    return auto_msl


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
    ignore_eos: bool = False,
    preserve_eos_token: bool = False,
    strict_generation: bool = False,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    context_parallel_comm_type: Optional[ContextParallelCommType] = None,
    output_file: Optional[Path] = None,
    stream_output: bool = False,
    mixed_precision_recipe: Optional[str] = None,
    quantized_param_storage: Literal["recipe", "bf16"] = "recipe",
    fp8_all_layers: bool = False,
    vortex_style_fp8: bool = False,
    max_seq_length: Optional[int] = None,
    max_seq_length_num_prompts: int = _DEFAULT_AUTO_MAX_SEQ_LENGTH_NUM_PROMPTS,
    max_batch_size: int = 1,
    evo2_batched_decode_size: int = 1,
    use_subquadratic_ops: bool = False,
    cuda_graph_impl: str = "local",
    cuda_graph_scope: str = "block",
    inference_backend: Literal["dynamic", "static-flash"] = "dynamic",
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
        top_p: Nucleus sampling parameter (0 = disabled), applied after top-k when both are enabled.
        seed: Random seed for reproducibility.
        return_log_probs: Whether to return per-token log probabilities.
        ignore_eos: Omit sampled EOS tokens and continue to max_new_tokens.
        preserve_eos_token: Include a sampled terminal EOS/EOD token and, when requested, its
            log-probability in each result. Has no effect when ``ignore_eos`` suppresses stop tokens.
        strict_generation: Fail instead of returning short or fallback generation results.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        context_parallel_comm_type: Runtime TE attention transport. ``None`` selects
            P2P; A2A can be selected for tighter BF16 parity. Checkpoint metadata
            does not control this execution choice.
        output_file: Optional path to save results as JSONL.
        stream_output: Write and flush each result as soon as it is generated. Strict generation
            writes to ``<output_file>.partial`` and atomically promotes it after success.
        mixed_precision_recipe: Override mixed precision recipe.
        quantized_param_storage: Preserve native quantized parameters from the selected recipe or
            retain BF16 parameters while executing its quantized GEMMs.
        fp8_all_layers: Remove BF16 first/last-block exclusions from the selected global TE FP8
            recipe. This is the regular full-scope Hopper FP8 path for Evo2 7B.
        vortex_style_fp8: Use vortex-style FP8 (applies FP8 only to projection layers).
            Needed for FP8-sensitive checkpoints from original evo2 training (1b, 40b).
        max_seq_length: Manual sequence-length cap (supersedes auto-sizing; never grows). ``None``
            (default) auto-sizes the engine from the prompt token lengths and grows on demand.
        max_seq_length_num_prompts: When auto-sizing, size from the longest of the first N prompts
            (``<= 0`` = all). A longer later prompt grows the context on demand rather than erroring.
        max_batch_size: Prompt-file chunk size and Megatron setup micro-batch metadata. This does
            not control the number of prompt-file generations or Evo2 native decode concurrency.
        evo2_batched_decode_size: Number of variable-length Evo2 prompts to keep active for
            packed prefill and native Hyena next-token decode. ``1`` preserves single-request execution.
        use_subquadratic_ops: Use fused Hyena convolution kernels for static-Flash rectangular/eager
            prefill compatibility. The dynamic backend ignores this flag and keeps CUDA graphs enabled.
        cuda_graph_impl: ``"local"`` (default) uses MCore decode CUDA graphs; ``"none"`` runs
            decode eagerly (no graph capture) -- mainly for debugging / un-graphed reference runs.
        cuda_graph_scope: ``"block"`` (default) captures the complete decoder; ``"layer"`` keeps
            the per-layer compatibility path. Global FP8/FP4 recipes automatically resolve block
            scope to layer scope for Transformer Engine quantization-state compatibility.
        inference_backend: Use packed paged-KV dynamic inference or equal-length static FlashAttention.
        enable_chunked_prefill: Split prompts across multiple prefill forwards when needed.
        inference_dynamic_batching_max_tokens: Optional dynamic-context per-step token budget.
        inference_dynamic_batching_block_size: Paged-KV block size for dynamic inference.
        force_exit_on_completion: For CLI use, immediately exit after successful CUDA-graph
            inference to avoid torchrun/NCCL atexit hangs with captured collectives.

    Returns:
        List of JSONL-serialisable result dicts.
    """
    world_size = get_world_size_safe()
    model_parallel_size = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size

    # TODO: Add standalone/offline DP orchestration here: assign indexed prompt shards to DP
    # replicas while keeping each shard replicated across its TP/PP/CP ranks, then gather and
    # restore input order on global rank zero without weakening streaming or strict-output
    # semantics. NeMo-RL already performs that orchestration before calling
    # _generate_native_dynamic(), so the lower-level generation path must remain shard-local.
    if world_size > model_parallel_size:
        raise NotImplementedError(
            "Top-level Evo2 inference does not yet support data parallelism: "
            f"world_size={world_size} exceeds model_parallel_size={model_parallel_size} "
            f"(tensor_parallel_size={tensor_parallel_size}, "
            f"pipeline_model_parallel_size={pipeline_model_parallel_size}, "
            f"context_parallel_size={context_parallel_size}). "
            "Launch with world_size equal to model_parallel_size."
        )

    random_seed = seed or 1234

    use_subquadratic_ops = _resolve_inference_subquadratic_ops(use_subquadratic_ops, inference_backend)

    _prune_caches()
    if not _CUDA_PHASE_EVIDENCE_ENABLED:
        torch.cuda.reset_peak_memory_stats()
    engine_setup_started_at_s = _begin_cuda_phase()

    components = setup_inference_engine(
        ckpt_dir=ckpt_dir,
        max_seq_length=max_seq_length,
        max_batch_size=max_batch_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        context_parallel_size=context_parallel_size,
        context_parallel_comm_type=context_parallel_comm_type,
        mixed_precision_recipe=mixed_precision_recipe,
        quantized_param_storage=quantized_param_storage,
        fp8_all_layers=fp8_all_layers,
        vortex_style_fp8=vortex_style_fp8,
        random_seed=random_seed,
        use_subquadratic_ops=use_subquadratic_ops,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_scope=cuda_graph_scope,
        inference_backend=inference_backend,
    )
    engine_setup_stats = _finish_cuda_phase(engine_setup_started_at_s)
    if not engine_setup_stats.performed:
        engine_setup_stats = _CudaPhaseStats(
            elapsed_s=time.perf_counter() - engine_setup_started_at_s,
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=int(torch.cuda.max_memory_reserved()),
            performed=False,
        )
    components.native_dynamic.engine_setup_stats = engine_setup_stats
    components.native_dynamic.engine_setup_stats_pending = True
    mem_after_setup_gb = engine_setup_stats.peak_allocated_bytes / (1024**3)
    mem_reserved_after_setup_gb = engine_setup_stats.peak_reserved_bytes / (1024**3)
    logger.info(
        f"[MEMORY] After model setup: peak={mem_after_setup_gb:.3f} GB, "
        f"reserved={mem_reserved_after_setup_gb:.3f} GB, "
        f"engine_setup_elapsed_s={engine_setup_stats.elapsed_s:.6f}"
    )

    # Auto-size the engine's sequence-length budget from the prompts unless a manual value was given.
    # Manual --max-seq-length supersedes (setup stored it; this only runs in auto mode). We size from
    # the longest of the first --max-seq-length-num-prompts prompts here so the budget reflects the
    # whole run (not just the first batch); prompts beyond that sample are validated per-batch in
    # _generate_native_dynamic, which fails loudly with the exact --max-seq-length to set.
    if max_seq_length is None and prompts:
        _resolve_prompt_auto_max_seq_length(
            components,
            [entry["prompt"] for entry in prompts],
            max_new_tokens=max_new_tokens,
            num_prompts=max_seq_length_num_prompts,
        )

    all_records: List[Dict[str, Any]] = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_model_generation_elapsed_s = 0.0
    total_decode_elapsed_s = 0.0
    total_decode_completion_tokens = 0
    t_generate_start = time.perf_counter()
    # Every process runs the same unsharded prompt list. Use one global writer for the shared
    # output path: data-parallel rank zero is true once per model-parallel coordinate and can
    # therefore select multiple writers. get_rank_safe() uses the initialized process group
    # before falling back to launcher state.
    is_rank_zero = get_rank_safe() == 0
    stream_file = None
    streamed_record_count = 0
    strict_stream_partial_path: Optional[Path] = None

    if is_rank_zero and output_file is not None and stream_output:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        stream_path = output_file
        if strict_generation:
            strict_stream_partial_path = Path(f"{output_file}.partial")
            stream_path = strict_stream_partial_path
        stream_file = open(stream_path, "w")
        logger.info("Streaming JSONL results to: %s", stream_path)

    try:
        for batch_start in range(0, len(prompts), max_batch_size):
            batch = prompts[batch_start : batch_start + max_batch_size]
            batch_prompts = [entry["prompt"] for entry in batch]
            batch_idx = batch_start // max_batch_size + 1

            logger.info(f"Generating batch {batch_idx} ({len(batch)} prompt(s))...")
            streamed_records: Dict[int, Dict[str, Any]] = {}

            def _stream_result(prompt_idx: int, result: Any) -> None:
                nonlocal streamed_record_count
                if stream_file is None:
                    return
                entry = batch[prompt_idx]
                record = _result_to_jsonl_record(
                    request_id=entry["id"],
                    prompt=entry["prompt"],
                    result=result,
                    max_new_tokens=max_new_tokens,
                    return_log_probs=return_log_probs,
                )
                streamed_records[prompt_idx] = record
                stream_file.write(json.dumps(record) + "\n")
                stream_file.flush()
                streamed_record_count += 1

            t_batch_start = time.perf_counter()
            results = generate(
                components,
                prompts=batch_prompts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=return_log_probs,
                ignore_eos=ignore_eos,
                preserve_eos_token=preserve_eos_token,
                strict_generation=strict_generation,
                enable_chunked_prefill=enable_chunked_prefill,
                inference_dynamic_batching_max_tokens=inference_dynamic_batching_max_tokens,
                inference_dynamic_batching_block_size=inference_dynamic_batching_block_size,
                evo2_batched_decode_size=evo2_batched_decode_size,
                inference_backend=inference_backend,
                result_callback=_stream_result if stream_file is not None else None,
            )
            t_batch_elapsed = time.perf_counter() - t_batch_start

            batch_completion_tokens = 0
            batch_model_generation_elapsed_s = 0.0
            batch_decode_elapsed_s = 0.0
            batch_decode_completion_tokens = 0
            seen_timing_groups: set[tuple[str, str]] = set()
            for prompt_idx, (entry, result) in enumerate(zip(batch, results)):
                record = streamed_records.get(prompt_idx)
                if record is None:
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
                result_timings = record.get("timings", {})
                timing_group_id = result_timings.get("timing_group_id")
                if timing_group_id is not None:
                    timing_group_key = (
                        str(result_timings.get("timing_scope", "native_generation_group")),
                        str(timing_group_id),
                    )
                    if timing_group_key not in seen_timing_groups:
                        seen_timing_groups.add(timing_group_key)
                        batch_model_generation_elapsed_s += float(result_timings.get("generation_elapsed_s", 0.0))
                        batch_decode_elapsed_s += float(result_timings.get("decode_elapsed_s", 0.0))
                        batch_decode_completion_tokens += int(result_timings.get("decode_completion_tokens", 0))

            batch_tok_per_sec = batch_completion_tokens / t_batch_elapsed if t_batch_elapsed > 0 else 0
            logger.info(
                f"[PERF] Batch {batch_idx} end-to-end: {batch_completion_tokens} tokens in "
                f"{t_batch_elapsed:.2f}s ({batch_tok_per_sec:.1f} completion tok/s)"
            )
            if batch_model_generation_elapsed_s > 0:
                batch_generation_tok_per_sec = batch_completion_tokens / batch_model_generation_elapsed_s
                logger.info(
                    f"[PERF] Batch {batch_idx} model generation (excludes engine/context/graph setup): "
                    f"{batch_completion_tokens} tokens in {batch_model_generation_elapsed_s:.2f}s "
                    f"({batch_generation_tok_per_sec:.1f} completion tok/s)"
                )
            if batch_decode_elapsed_s > 0:
                batch_decode_tok_per_sec = batch_decode_completion_tokens / batch_decode_elapsed_s
                logger.info(
                    f"[PERF] Batch {batch_idx} steady decode: {batch_decode_completion_tokens} tokens in "
                    f"{batch_decode_elapsed_s:.2f}s ({batch_decode_tok_per_sec:.1f} completion tok/s)"
                )
            total_model_generation_elapsed_s += batch_model_generation_elapsed_s
            total_decode_elapsed_s += batch_decode_elapsed_s
            total_decode_completion_tokens += batch_decode_completion_tokens
    finally:
        if stream_file is not None:
            stream_file.close()

    if strict_stream_partial_path is not None:
        os.replace(strict_stream_partial_path, output_file)

    t_generate_elapsed = time.perf_counter() - t_generate_start
    total_tok_per_sec = total_completion_tokens / t_generate_elapsed if t_generate_elapsed > 0 else 0

    mem_after_generate_bytes = max(
        engine_setup_stats.peak_allocated_bytes,
        int(torch.cuda.max_memory_allocated()),
        *(int(record.get("memory", {}).get("total_peak_allocated_bytes", 0)) for record in all_records),
    )
    mem_reserved_after_generate_bytes = max(
        engine_setup_stats.peak_reserved_bytes,
        int(torch.cuda.max_memory_reserved()),
        *(int(record.get("memory", {}).get("total_peak_reserved_bytes", 0)) for record in all_records),
    )
    mem_after_generate_gb = mem_after_generate_bytes / (1024**3)
    mem_reserved_after_generate_gb = mem_reserved_after_generate_bytes / (1024**3)
    logger.info(
        f"[MEMORY] After generation: peak={mem_after_generate_gb:.3f} GB, "
        f"reserved={mem_reserved_after_generate_gb:.3f} GB "
        f"(setup={mem_after_setup_gb:.3f} GB, generation delta="
        f"{mem_after_generate_gb - mem_after_setup_gb:.3f} GB)"
    )
    logger.info(
        f"[PERF] Total end-to-end: {total_prompt_tokens} prompt tokens + {total_completion_tokens} "
        f"completion tokens in {t_generate_elapsed:.2f}s "
        f"({total_tok_per_sec:.1f} completion tok/s)"
    )
    if total_model_generation_elapsed_s > 0:
        logger.info(
            f"[PERF] Total model generation (excludes engine/context/graph setup): {total_completion_tokens} "
            f"completion tokens in {total_model_generation_elapsed_s:.2f}s "
            f"({total_completion_tokens / total_model_generation_elapsed_s:.1f} completion tok/s)"
        )
    if total_decode_elapsed_s > 0:
        logger.info(
            f"[PERF] Total steady decode: {total_decode_completion_tokens} completion tokens in "
            f"{total_decode_elapsed_s:.2f}s "
            f"({total_decode_completion_tokens / total_decode_elapsed_s:.1f} completion tok/s)"
        )

    if is_rank_zero:
        for record in all_records:
            print(
                f"\n=== [{record['id']}] Generated Text ===\n{record['completion']}\n",
                file=sys.stdout,
            )

        if output_file is not None and not stream_output:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                for record in all_records:
                    f.write(json.dumps(record) + "\n")
            logger.info(f"Saved {len(all_records)} result(s) to: {output_file}")
        elif output_file is not None and stream_output:
            logger.info(f"Streamed {streamed_record_count} result(s) to: {output_file}")

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
    # Manual --max-seq-length (or EVO2_MAX_SEQ_LEN) supersedes; otherwise None => auto-size from the
    # prompts in infer() (which is tighter than the GPU-memory heuristic for typical short prompts).
    max_seq_length = _resolve_int(args.max_seq_length, "EVO2_MAX_SEQ_LEN", None)

    if args.prompt_file is not None:
        prompts = _read_prompts_jsonl(args.prompt_file)
    else:
        prompts = [{"id": "0", "prompt": args.prompt}]

    prompt_batch_size = args.prompt_batch_size
    if prompt_batch_size is None:
        prompt_batch_size = args.evo2_batched_decode_size
    if prompt_batch_size is None:
        prompt_batch_size = 1
    prompt_file_chunk_size = args.max_batch_size if args.max_batch_size is not None else prompt_batch_size
    prompt_batch_size = min(prompt_batch_size, prompt_file_chunk_size)

    infer(
        prompts=prompts,
        ckpt_dir=args.ckpt_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        return_log_probs=args.return_log_probs,
        ignore_eos=args.ignore_eos,
        preserve_eos_token=args.preserve_eos_token,
        strict_generation=args.strict_generation,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        context_parallel_comm_type=args.context_parallel_comm_type,
        output_file=args.output_file,
        stream_output=args.stream_output,
        mixed_precision_recipe=args.mixed_precision_recipe,
        quantized_param_storage=args.quantized_param_storage,
        fp8_all_layers=args.fp8_all_layers,
        vortex_style_fp8=args.vortex_style_fp8,
        max_seq_length=max_seq_length,
        max_seq_length_num_prompts=args.max_seq_length_num_prompts,
        max_batch_size=prompt_file_chunk_size,
        evo2_batched_decode_size=prompt_batch_size,
        use_subquadratic_ops=args.use_subquadratic_ops,
        cuda_graph_impl=args.cuda_graph_impl,
        cuda_graph_scope=args.cuda_graph_scope,
        inference_backend=args.inference_backend,
        enable_chunked_prefill=args.enable_chunked_prefill,
        inference_dynamic_batching_max_tokens=args.inference_dynamic_batching_max_tokens,
        inference_dynamic_batching_block_size=args.inference_dynamic_batching_block_size,
        force_exit_on_completion=True,
    )


if __name__ == "__main__":
    main()
