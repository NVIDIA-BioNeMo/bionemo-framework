# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

from bionemo.evo2.vllm import plugin
from bionemo.evo2.vllm.nemo_generation_worker import Evo2NemoRlGenerationWorkerImpl


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_generation_worker_registers_evo2_before_parent_initialization(monkeypatch) -> None:
    events: list[object] = []

    monkeypatch.setattr(plugin, "register", lambda: events.append("register"))

    def parent_init(self, *args, **kwargs) -> None:
        events.append((args, kwargs))

    monkeypatch.setattr(VllmGenerationWorkerImpl, "__init__", parent_init)

    Evo2NemoRlGenerationWorkerImpl("worker", rank=1)

    _require(
        events == ["register", (("worker",), {"rank": 1})],
        f"unexpected worker initialization order: {events!r}",
    )


def test_generation_worker_keeps_proof_instrumentation_out_of_runtime_adapter() -> None:
    proof_methods = {
        "publish_evo2_generation_sidecar",
        "reset_evo2_proof_phase",
        "snapshot_evo2_proof_phase",
        "reset_evo2_refit_phase",
        "snapshot_evo2_refit_phase",
    }
    retained = proof_methods.intersection(Evo2NemoRlGenerationWorkerImpl.__dict__)
    _require(not retained, f"proof-only methods remain on the production worker: {sorted(retained)}")
