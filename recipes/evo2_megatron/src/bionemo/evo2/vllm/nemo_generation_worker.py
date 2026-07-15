# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""NeMo-RL outer generation worker with phase-local Evo2 execution proof."""

import os
import socket
from typing import Any

import ray
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

from bionemo.evo2.vllm.profile import resolved_config_snapshot
from bionemo.evo2.vllm.runner import CUDAGraphProofRecorder, summarize_cudagraph_observations


class Evo2NemoRlGenerationWorkerImpl(VllmGenerationWorkerImpl):
    """Attach scheduler proof to each production NeMo-RL vLLM engine."""

    def __init__(self, *args, **kwargs) -> None:
        """Register Evo2 before NeMo-RL performs Transformers model probes."""
        from bionemo.evo2.vllm.plugin import register

        register()
        super().__init__(*args, **kwargs)

    def post_init(self):
        """Finish NeMo-RL initialization and attach the engine-local recorder."""
        result = super().post_init()
        self._attach_evo2_proof_recorder()
        return result

    def _attach_evo2_proof_recorder(self) -> None:
        """Attach one persistent scheduler recorder to the outer vLLM engine."""
        manager = self.llm.llm_engine.logger_manager
        if manager is None:
            raise RuntimeError("vLLM stat logger manager is disabled; Evo2 proof is unavailable")
        if hasattr(self, "_evo2_cudagraph_recorder"):
            return
        self._evo2_cudagraph_recorder = CUDAGraphProofRecorder()
        self._evo2_proof_phase = "unlabeled"
        self._evo2_observation_start = 0
        self._evo2_scheduler_observation_start = 0
        manager.stat_loggers.append(self._evo2_cudagraph_recorder)

    def reset_evo2_proof_phase(self, phase: str) -> dict[str, Any]:
        """Reset inner route/memory counters and label subsequent scheduler events."""
        if not phase:
            raise ValueError("proof phase cannot be empty")
        self._attach_evo2_proof_recorder()
        self._evo2_proof_phase = phase
        self._evo2_cudagraph_recorder.start_phase(phase)
        self._evo2_observation_start = len(self._evo2_cudagraph_recorder.observations)
        self._evo2_scheduler_observation_start = len(self._evo2_cudagraph_recorder.scheduler_observations)
        worker_reset = self.llm.collective_rpc("reset_evo2_proof_state", args=())
        return {"phase": phase, "worker_reset": worker_reset}

    def snapshot_evo2_proof_phase(self, phase: str) -> dict[str, Any]:
        """Return untruncated outer graph events plus inner route/compile/memory state."""
        if phase != self._evo2_proof_phase:
            raise ValueError(f"requested proof phase {phase!r} does not match active {self._evo2_proof_phase!r}")
        observations = tuple(self._evo2_cudagraph_recorder.observations[self._evo2_observation_start :])
        scheduler_observations = tuple(
            self._evo2_cudagraph_recorder.scheduler_observations[self._evo2_scheduler_observation_start :]
        )
        worker_proof = self.llm.collective_rpc("snapshot_evo2_proof_state", args=())
        return {
            "phase": phase,
            "resolved_config": resolved_config_snapshot(self.llm.llm_engine.vllm_config),
            "cudagraph_observations": list(observations),
            "cudagraph_summary": summarize_cudagraph_observations(observations),
            "scheduler_observations": list(scheduler_observations),
            "worker_proof": worker_proof,
        }

    def reset_evo2_refit_phase(self, phase: str) -> dict[str, Any]:
        """Reset refit chunk telemetry on every internal TP worker."""
        worker_reset = self.llm.collective_rpc(
            "reset_evo2_refit_proof_state", args=(phase,)
        )
        return {"phase": phase, "worker_reset": worker_reset}

    def snapshot_evo2_refit_phase(self, phase: str) -> dict[str, Any]:
        """Return internal TP refit transactions plus stable actor/model identity."""
        worker_proof = self.llm.collective_rpc(
            "snapshot_evo2_refit_proof_state", args=(phase,)
        )
        device_uuids = self.llm.collective_rpc("report_device_id", args=())
        model_config = self.llm.llm_engine.model_config
        return {
            "phase": phase,
            "actor": {
                "ray_actor_id": ray.get_runtime_context().get_actor_id(),
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "model": {
                "model": str(model_config.model),
                "architectures": list(model_config.architectures),
                "dtype": str(model_config.dtype),
                "max_model_len": int(model_config.max_model_len),
            },
            "device_uuids": device_uuids,
            "worker_proof": worker_proof,
        }


@ray.remote(runtime_env={**get_nsight_config_if_pattern_matches("vllm_generation_worker")})
class Evo2NemoRlGenerationWorker(Evo2NemoRlGenerationWorkerImpl):
    """Ray actor used by the Evo2 NeMo-RL production generation path."""


__all__ = ["Evo2NemoRlGenerationWorker", "Evo2NemoRlGenerationWorkerImpl"]
