# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""NeMo-RL refit controls combined with Evo2 proof telemetry."""

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension

from bionemo.evo2.vllm.worker import Evo2VllmWorkerExtension


class Evo2NemoRlVllmWorkerExtension(Evo2VllmWorkerExtension, VllmInternalWorkerExtension):
    """Expose NeMo-RL refit plus phase-local Evo2 route and graph evidence."""


__all__ = ["Evo2NemoRlVllmWorkerExtension"]
