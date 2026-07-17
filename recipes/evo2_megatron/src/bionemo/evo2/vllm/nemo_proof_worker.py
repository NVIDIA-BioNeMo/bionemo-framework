# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Opt-in NeMo-RL worker telemetry for Evo2 proof runs."""

import hashlib
import time
from typing import Any

from bionemo.evo2.vllm.nemo_worker import Evo2NemoRlVllmWorkerExtension
from bionemo.evo2.vllm.worker import Evo2VllmWorkerExtension


class Evo2NemoRlProofVllmWorkerExtension(Evo2VllmWorkerExtension, Evo2NemoRlVllmWorkerExtension):
    """Combine the normal refit adapter with opt-in route and refit telemetry."""

    def reset_evo2_refit_proof_state(self, phase: str) -> dict[str, str]:
        """Label and reset transport-level refit chunk telemetry."""
        if not phase:
            raise ValueError("refit proof phase cannot be empty")
        self._evo2_refit_phase = phase
        self._evo2_refit_chunks: list[dict[str, Any]] = []
        return {"phase": phase}

    def _load_weights(self, weights):
        """Record every real NeMo-RL IPC/collective chunk before delegating."""
        items = list(weights)
        names = [name for name, _ in items]
        started = time.perf_counter()
        result = super()._load_weights(items)
        elapsed = time.perf_counter() - started
        if not hasattr(self, "_evo2_refit_chunks"):
            self.reset_evo2_refit_proof_state("unlabeled")
        self._evo2_refit_chunks.append(
            {
                "chunk_index": len(self._evo2_refit_chunks),
                "tensor_count": len(items),
                "tensor_bytes": sum(int(tensor.nbytes) for _, tensor in items),
                "first_name": names[0] if names else None,
                "last_name": names[-1] if names else None,
                "names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                "load_call_s": elapsed,
            }
        )
        return result

    def snapshot_evo2_refit_proof_state(self, phase: str) -> dict[str, Any]:
        """Return actual chunk calls and the Evo2 transaction completion state."""
        active_phase = getattr(self, "_evo2_refit_phase", None)
        if phase != active_phase:
            raise ValueError(f"requested refit phase {phase!r} does not match active {active_phase!r}")
        loader = self.model_runner.model._weight_loader
        chunks = list(getattr(self, "_evo2_refit_chunks", ()))
        return {
            "phase": phase,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "loader": {
                "completed_transactions": int(loader.completed_transactions),
                "loaded_parameter_count": len(loader._loaded_parameter_names),
                "required_parameter_count": len(loader.required_parameter_names),
                "pending_fc1_layer_count": len(loader._pending_fc1),
                "started": bool(loader._started),
                "complete": bool(loader._complete),
                "consumed": bool(loader._consumed),
            },
        }


__all__ = ["Evo2NemoRlProofVllmWorkerExtension"]
