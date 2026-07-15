# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Named vLLM worker controls for Evo2 proof collection."""

from typing import Any


class Evo2VllmWorkerExtension:
    """Expose trusted string-RPC proof controls without pickle serialization."""

    def reset_evo2_proof_state(self, reset_prefix_sources: bool = True) -> dict[str, int | bool]:
        """Reset phase-local FIR and CUDA-memory telemetry."""
        from bionemo.evo2.vllm.runner import reset_vllm_worker_proof_state

        return reset_vllm_worker_proof_state(self, reset_prefix_sources)

    def snapshot_evo2_proof_state(self) -> dict[str, Any]:
        """Return route, compile, and CUDA-memory evidence for this worker."""
        from bionemo.evo2.vllm.runner import snapshot_vllm_worker_proof_state

        return snapshot_vllm_worker_proof_state(self)


__all__ = ["Evo2VllmWorkerExtension"]
