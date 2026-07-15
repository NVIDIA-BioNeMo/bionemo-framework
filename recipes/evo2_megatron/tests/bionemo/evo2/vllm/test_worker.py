# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from bionemo.evo2.vllm import runner
from bionemo.evo2.vllm.worker import Evo2VllmWorkerExtension


def test_named_worker_extension_delegates_proof_state_without_callable_rpc(monkeypatch) -> None:
    worker = Evo2VllmWorkerExtension()
    monkeypatch.setattr(runner, "reset_vllm_worker_proof_state", lambda owner: {"owner": owner})
    monkeypatch.setattr(runner, "snapshot_vllm_worker_proof_state", lambda owner: {"snapshot": owner})

    assert worker.reset_evo2_proof_state() == {"owner": worker}
    assert worker.snapshot_evo2_proof_state() == {"snapshot": worker}
