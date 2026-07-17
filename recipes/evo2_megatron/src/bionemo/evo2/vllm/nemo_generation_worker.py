# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Thin NeMo-RL generation worker for the Evo2 vLLM plugin."""

import ray
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl


class Evo2NemoRlGenerationWorkerImpl(VllmGenerationWorkerImpl):
    """Register Evo2 before NeMo-RL performs Transformers model probes."""

    def __init__(self, *args, **kwargs) -> None:
        """Register the plugin before the base worker resolves the model."""
        from bionemo.evo2.vllm.plugin import register

        register()
        super().__init__(*args, **kwargs)


@ray.remote(runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")})
class Evo2NemoRlGenerationWorker(Evo2NemoRlGenerationWorkerImpl):
    """Ray actor used by the Evo2 NeMo-RL production generation path."""


__all__ = ["Evo2NemoRlGenerationWorker", "Evo2NemoRlGenerationWorkerImpl"]
