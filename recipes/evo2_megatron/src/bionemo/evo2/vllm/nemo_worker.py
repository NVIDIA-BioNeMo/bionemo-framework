# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Thin NeMo-RL worker extension for Evo2 weight refit support."""

from nemo_rl.models.generation.vllm.vllm_backend import VllmInternalWorkerExtension


class Evo2NemoRlVllmWorkerExtension(VllmInternalWorkerExtension):
    """Retain stock NeMo-RL refit behavior without proof-time wrappers."""


__all__ = ["Evo2NemoRlVllmWorkerExtension"]
