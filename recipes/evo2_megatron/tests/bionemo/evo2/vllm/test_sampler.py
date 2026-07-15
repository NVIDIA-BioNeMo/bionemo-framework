# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

import bionemo.evo2.vllm.sampler as sampler_module
from bionemo.evo2.vllm.sampler import (
    install_sampler_route_proof_hook,
    run_sampler_seed_behavioral_preflight,
    sampler_installation_contract,
    sampler_runtime_environment_contract,
    snapshot_sampler_route_proof,
    validate_sampler_proof_evidence,
)


class _NativeTopKSampler:
    __module__ = "vllm.v1.sample.ops.topk_topp_sampler"
    logprobs_mode = "processed_logprobs"

    def __init__(self) -> None:
        self.forward = self.forward_native

    def forward_native(self, logits, generators, k, p):
        del k, p
        q = torch.empty_like(logits)
        for row_index, generator in generators.items():
            q[row_index].exponential_(generator=generator)
        logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        return logprobs.exp().div_(q).argmax(dim=-1), logprobs

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class _SeedZeroTopKSampler(_NativeTopKSampler):
    __module__ = "vllm.v1.sample.ops.topk_topp_sampler"

    def forward_native(self, logits, generators, k, p):
        del k, p
        q = torch.empty_like(logits)
        seed_zero = generators[0]
        for row_index in range(logits.shape[0]):
            q[row_index].exponential_(generator=seed_zero)
        logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        return logprobs.exp().div_(q).argmax(dim=-1), logprobs


class _V1ModelRunner:
    __module__ = "vllm.v1.worker.gpu_model_runner"

    def __init__(self, topk_sampler) -> None:
        self.device = torch.device("cpu")
        self.sampler = SimpleNamespace(topk_topp_sampler=topk_sampler)


def _worker(topk_sampler):
    return SimpleNamespace(model_runner=_V1ModelRunner(topk_sampler))


def _fake_installation_provenance() -> dict:
    source_files = [
        {
            "path": f"/audited/{relative_path}",
            "relative_path": relative_path,
            "size_bytes": index + 1,
            "sha256": sha256,
        }
        for index, (relative_path, sha256) in enumerate(sampler_module._EXPECTED_SOURCE_SHA256.items())
    ]
    return {
        "schema_version": 1,
        "executing_python": {
            "invoked_path": "/nemo-rl/.venv/bin/python",
            "resolved_path": "/python3.13",
            "resolved_sha256": "a" * 64,
            "invoked_symlink_target": "/python3.13",
            "nemo_rl_py_executables_system": "1",
        },
        "launcher_selection": {
            "nemo_rl_py_executables_system": "1",
            "selected_environment": "system-runtime",
            "executing_python_path": "/nemo-rl/.venv/bin/python",
            "isolated_worker_environment": {
                "path": "/nemo-rl/venvs/bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker/bin/python",
                "exists": True,
                "status": "installed-but-bypassed",
                "attested_as_executing": False,
            },
        },
        "distributions": dict(sampler_module._EXPECTED_DISTRIBUTIONS),
        "distribution_metadata": {
            name: {"record_path": f"/audited/{name}.dist-info/RECORD", "record_sha256": str(index) * 64}
            for index, name in enumerate(sampler_module._EXPECTED_DISTRIBUTIONS, start=1)
        },
        "loaded_modules": {
            "vllm.v1.worker.gpu_model_runner": {
                "loaded": True,
                "path": "/audited/vllm/v1/worker/gpu_model_runner.py",
                "sha256": sampler_module._EXPECTED_SOURCE_SHA256["vllm/v1/worker/gpu_model_runner.py"],
            },
            "vllm.v1.sample.ops.topk_topp_sampler": {
                "loaded": True,
                "path": "/audited/vllm/v1/sample/ops/topk_topp_sampler.py",
                "sha256": sampler_module._EXPECTED_SOURCE_SHA256["vllm/v1/sample/ops/topk_topp_sampler.py"],
            },
            "flashinfer": {"loaded": False, "path": None, "sha256": None},
            "flashinfer.sampling": {
                "loaded": True,
                "path": "/audited/flashinfer/sampling.py",
                "sha256": sampler_module._EXPECTED_SOURCE_SHA256["flashinfer/sampling.py"],
            },
        },
        "required_sampler_modules_loaded": True,
        "source_files": source_files,
        "source_contract_sha256": hashlib.sha256(
            "".join(item["sha256"] for item in source_files).encode()
        ).hexdigest(),
        "flashinfer_per_row_seed_arrays_supported": False,
        "flashinfer_impact_on_selected_vllm_route": "none-route-bypassed",
        "passed": True,
    }


def _sampler_proof(monkeypatch) -> tuple[dict, dict, dict]:
    installation = _fake_installation_provenance()
    monkeypatch.setattr(sampler_module, "sampler_installation_provenance", lambda: installation)
    environment = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"}
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(worker, environ=environment)
    sampler = worker.model_runner.sampler.topk_topp_sampler
    sampler(
        torch.zeros((2, 8), dtype=torch.float32),
        {
            0: torch.Generator().manual_seed(11),
            1: torch.Generator().manual_seed(13),
        },
        torch.full((2,), 4, dtype=torch.int32),
        None,
    )
    return (
        snapshot_sampler_route_proof(
            worker,
            require_generation_observations=True,
            environ=environment,
        ),
        sampler_runtime_environment_contract(environment),
        sampler_installation_contract(installation),
    )


def _tamper_chosen_logprob_everywhere(proof: dict, *, request_id: str, step: int) -> None:
    preflight = proof["behavioral_preflight"]
    streams = [preflight["oracle_streams"], preflight["oracle_replay_streams"]]
    streams.extend(variant["streams"] for variant in preflight["variants"].values())
    for collection in streams:
        rows = (
            collection.items()
            if isinstance(collection, dict)
            else ((row.get("request_id"), row) for row in collection)
        )
        for row_id, row in rows:
            if row_id != request_id or len(row["chosen_logprobs"]) <= step:
                continue
            original = torch.tensor(row["chosen_logprobs"][step], dtype=torch.float32)
            tampered = torch.nextafter(original, torch.tensor(float("inf"), dtype=torch.float32)).item()
            row["chosen_logprobs"][step] = tampered
            row["chosen_logprob_fp32_hex"][step] = sampler_module._fp32_hex(tampered)


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({"VLLM_USE_FLASHINFER_SAMPLER": "1"}, "FlashInfer sampling"),
        ({"VLLM_USE_V2_MODEL_RUNNER": "1"}, "V2 model runner"),
    ),
)
def test_sampler_environment_rejects_unproven_optimized_routes(environment, message) -> None:
    with pytest.raises(RuntimeError, match=message):
        sampler_runtime_environment_contract(environment)


def test_sampler_installation_contract_attests_actual_runtime_and_marks_isolated_worker_bypassed() -> None:
    contract = sampler_installation_contract(_fake_installation_provenance())

    assert contract["executing_python"]["invoked_path"] == "/nemo-rl/.venv/bin/python"
    assert contract["launcher_selection"] == {
        "nemo_rl_py_executables_system": "1",
        "selected_environment": "system-runtime",
        "executing_python_path": "/nemo-rl/.venv/bin/python",
        "isolated_worker_environment": {
            "path": "/nemo-rl/venvs/bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker/bin/python",
            "exists": True,
            "status": "installed-but-bypassed",
            "attested_as_executing": False,
        },
    }


def test_production_sampler_preflight_proves_per_request_stream_invariance() -> None:
    evidence = run_sampler_seed_behavioral_preflight(
        _worker(_NativeTopKSampler()),
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )

    assert evidence["passed"] is True
    assert evidence["selected_route"].endswith("TopKTopPSampler.forward_native")
    assert evidence["request_seeds"] == [11, 1_000_014, 2_000_017, 3_000_020]
    assert set(evidence["variants"]) == {"batched", "reordered", "deactivated", "padded", "subwaves"}
    assert all(variant["passed"] is True for variant in evidence["variants"].values())
    assert len({tuple(stream["token_ids"]) for stream in evidence["oracle_streams"]}) == 4


def test_production_sampler_preflight_rejects_seed_zero_reuse() -> None:
    with pytest.raises(RuntimeError, match="per-request sampler invariance"):
        run_sampler_seed_behavioral_preflight(
            _worker(_SeedZeroTopKSampler()),
            environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
        )


def test_sampler_route_hook_preserves_one_native_draw_per_request() -> None:
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    hooked = worker.model_runner.sampler.topk_topp_sampler
    oracle = _NativeTopKSampler()
    logits = torch.zeros((2, 8), dtype=torch.float32)
    top_k = torch.full((2,), 4, dtype=torch.int32)
    hooked_generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
    }
    oracle_generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
    }

    hooked_result = hooked(logits, hooked_generators, top_k, None)
    oracle_result = oracle(logits, oracle_generators, top_k, None)

    assert torch.equal(hooked_result[0], oracle_result[0])
    assert torch.equal(hooked_result[1], oracle_result[1])
    assert all(
        torch.equal(hooked_generators[index].get_state(), oracle_generators[index].get_state())
        for index in hooked_generators
    )


def test_sampler_route_hook_rejects_missing_generator_and_retains_actual_route(monkeypatch) -> None:
    monkeypatch.setattr(sampler_module, "sampler_installation_provenance", lambda: {"passed": True})
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    logits = torch.zeros((2, 8), dtype=torch.float32)
    top_k = torch.full((2,), 4, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="one explicit generator per active row"):
        sampler(logits, {0: torch.Generator().manual_seed(11)}, top_k, None)

    sampler(
        logits,
        {
            0: torch.Generator().manual_seed(11),
            1: torch.Generator().manual_seed(13),
        },
        top_k,
        None,
    )
    proof = snapshot_sampler_route_proof(worker, require_generation_observations=True)

    assert proof["passed"] is True
    assert proof["flashinfer_sampling_calls"] == 0
    assert proof["generation_observations"] == [
        {
            "batch_rows": 2,
            "generator_count": 2,
            "generator_indices": [0, 1],
            "generator_seeds": [11, 13],
            "route": "vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native",
            "passed": True,
        }
    ]


def test_sampler_proof_validator_recomputes_chosen_values_from_retained_full_distribution(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch)
    validate_sampler_proof_evidence(
        proof,
        expected_environment=environment,
        expected_installation=installation,
        expected_seed_batches=((11, 13),),
        require_generation_observations=True,
    )

    tampered = copy.deepcopy(proof)
    _tamper_chosen_logprob_everywhere(tampered, request_id="request-0", step=0)

    with pytest.raises(AssertionError, match="full processed-logprob tensor"):
        validate_sampler_proof_evidence(
            tampered,
            expected_environment=environment,
            expected_installation=installation,
            expected_seed_batches=((11, 13),),
            require_generation_observations=True,
        )


def test_sampler_proof_validator_rejects_loaded_sampler_module_path_swap(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch)
    proof["installation"]["loaded_modules"]["vllm.v1.sample.ops.topk_topp_sampler"] = {
        "loaded": True,
        "path": "/foreign/topk_topp_sampler.py",
        "sha256": sampler_module._EXPECTED_SOURCE_SHA256["vllm/v1/sample/ops/topk_topp_sampler.py"],
    }

    with pytest.raises(AssertionError, match="loaded production sampler module"):
        validate_sampler_proof_evidence(
            proof,
            expected_environment=environment,
            expected_installation=installation,
            expected_seed_batches=((11, 13),),
            require_generation_observations=True,
        )


def test_initialized_sampler_proof_accepts_behavioral_preflight_without_generation_calls(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch)
    proof["generation_observations"] = []

    summary = validate_sampler_proof_evidence(
        proof,
        expected_environment=environment,
        expected_installation=installation,
        expected_seed_batches=(),
        require_generation_observations=False,
    )

    assert summary["generation_observation_count"] == 0
    assert summary["physical_seed_batches"] == []
