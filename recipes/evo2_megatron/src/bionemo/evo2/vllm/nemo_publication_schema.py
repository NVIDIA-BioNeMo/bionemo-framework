# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Versioned wire contracts shared by NeMo-RL rank publication boundaries."""

NEMO_CALLER_PROMPT_ROOT_SCHEMA_VERSION = "evo2-nemo-caller-prompt-root/v1"
NEMO_RANK_EXECUTION_OCCURRENCE_SCHEMA_VERSION = (
    "evo2-nemo-rank-execution-occurrence/v1"
)
NEMO_RANK_PUBLICATION_SCHEMA_VERSION = "evo2-nemo-rank-publication/v1"
NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION = (
    "evo2-nemo-rank-publication-outcome/v1"
)
NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION = "evo2-nemo-rank-sidecar-row/v1"
NEMO_RANK_PUBLICATION_PROOF_SCHEMA_VERSION = (
    "evo2-nemo-rank-publication-proof/v1"
)


__all__ = [
    "NEMO_CALLER_PROMPT_ROOT_SCHEMA_VERSION",
    "NEMO_RANK_EXECUTION_OCCURRENCE_SCHEMA_VERSION",
    "NEMO_RANK_PUBLICATION_OUTCOME_SCHEMA_VERSION",
    "NEMO_RANK_PUBLICATION_PROOF_SCHEMA_VERSION",
    "NEMO_RANK_PUBLICATION_SCHEMA_VERSION",
    "NEMO_RANK_SIDECAR_ROW_SCHEMA_VERSION",
]
