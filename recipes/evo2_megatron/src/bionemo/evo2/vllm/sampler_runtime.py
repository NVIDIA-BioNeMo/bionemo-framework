# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Production runtime guard for Evo2's per-request seeded sampler route."""

from __future__ import annotations

import os
from typing import Any, Mapping


V1_MODEL_RUNNER_MODULE = "vllm.v1.worker.gpu_model_runner"
TOPK_SAMPLER_MODULE = "vllm.v1.sample.ops.topk_topp_sampler"
NATIVE_ROUTE = f"{TOPK_SAMPLER_MODULE}.TopKTopPSampler.forward_native"
NEMO_VLLM_ACTOR_FQN = "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker"


def sampler_runtime_environment_contract(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Pin the vLLM sampler switches whose alternatives lack accepted seed proof."""
    values = os.environ if environ is None else environ
    flashinfer_flag = values.get("VLLM_USE_FLASHINFER_SAMPLER")
    v2_flag = values.get("VLLM_USE_V2_MODEL_RUNNER")
    if flashinfer_flag not in (None, "0"):
        raise RuntimeError("FlashInfer sampling must remain disabled for per-request seeded Evo2 generation")
    if v2_flag not in (None, "0"):
        raise RuntimeError("vLLM V2 model runner is not covered by the accepted Evo2 sampler proof")
    return {
        "schema_version": 1,
        "vllm_model_runner": V1_MODEL_RUNNER_MODULE,
        "logprobs_mode": "processed_logprobs",
        "selected_route": NATIVE_ROUTE,
        "one_generator_per_active_row": True,
        "flashinfer_sampling_allowed": False,
        "environment": {
            "VLLM_USE_FLASHINFER_SAMPLER": flashinfer_flag,
            "VLLM_USE_V2_MODEL_RUNNER": v2_flag,
        },
    }


__all__ = [
    "NATIVE_ROUTE",
    "NEMO_VLLM_ACTOR_FQN",
    "TOPK_SAMPLER_MODULE",
    "V1_MODEL_RUNNER_MODULE",
    "sampler_runtime_environment_contract",
]
