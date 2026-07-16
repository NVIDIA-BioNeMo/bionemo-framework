# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

import ast
import copy
import hashlib
import inspect
import textwrap
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


class _CountingGeneratorMap(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.proof_iteration_count = 0

    def __iter__(self):
        self.proof_iteration_count += 1
        return super().__iter__()


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


def _expected_request_generations(*, accepted_output_token_count: int = 1) -> tuple[dict, ...]:
    return (
        {
            "request_id": "audit-request-11",
            "seed": 11,
            "accepted_output_token_count": accepted_output_token_count,
        },
        {
            "request_id": "audit-request-13",
            "seed": 13,
            "accepted_output_token_count": accepted_output_token_count,
        },
    )


def _sampler_proof(monkeypatch, *, sampler_calls: int = 1) -> tuple[dict, dict, dict]:
    installation = _fake_installation_provenance()
    monkeypatch.setattr(sampler_module, "sampler_installation_provenance", lambda: installation)
    environment = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"}
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(worker, environ=environment)
    sampler = worker.model_runner.sampler.topk_topp_sampler
    generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
    }
    for _ in range(sampler_calls):
        sampler(
            torch.zeros((2, 8), dtype=torch.float32),
            generators,
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

    expected_launcher = {
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
    if contract["executing_python"]["invoked_path"] != "/nemo-rl/.venv/bin/python":
        raise AssertionError("sampler runtime attested the wrong executing interpreter")
    if contract["launcher_selection"] != expected_launcher:
        raise AssertionError("sampler runtime launcher selection drifted")


def test_production_sampler_preflight_proves_per_request_stream_invariance() -> None:
    evidence = run_sampler_seed_behavioral_preflight(
        _worker(_NativeTopKSampler()),
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )

    if evidence["passed"] is not True or not evidence["selected_route"].endswith(
        "TopKTopPSampler.forward_native"
    ):
        raise AssertionError("sampler behavioral proof did not use the native route")
    if evidence["request_seeds"] != [11, 1_000_014, 2_000_017, 3_000_020]:
        raise AssertionError("sampler behavioral proof request seeds drifted")
    if set(evidence["variants"]) != {"batched", "reordered", "deactivated", "padded", "subwaves"}:
        raise AssertionError("sampler behavioral proof schedule inventory drifted")
    if any(variant["passed"] is not True for variant in evidence["variants"].values()):
        raise AssertionError("one sampler adversarial schedule failed")
    if len({tuple(stream["token_ids"]) for stream in evidence["oracle_streams"]}) != 4:
        raise AssertionError("sampler behavioral oracle streams are not discriminating")


def test_production_sampler_preflight_rejects_seed_zero_reuse() -> None:
    with pytest.raises(RuntimeError, match="native forward did not advance|per-request sampler invariance"):
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

    if not torch.equal(hooked_result[0], oracle_result[0]) or not torch.equal(
        hooked_result[1], oracle_result[1]
    ):
        raise AssertionError("sampler proof hook changed native tokens or processed logprobs")
    if any(
        not torch.equal(hooked_generators[index].get_state(), oracle_generators[index].get_state())
        for index in hooked_generators
    ):
        raise AssertionError("sampler proof hook changed native generator advancement")


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

    generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
    }
    sampler(logits, generators, top_k, None)
    proof = snapshot_sampler_route_proof(worker, require_generation_observations=True)

    if proof["passed"] is not True or proof["flashinfer_sampling_calls"] != 0:
        raise AssertionError("sampler aggregate proof did not retain the native route")
    descriptors = proof["generation_batch_descriptors"]
    if len(descriptors) != 1:
        raise AssertionError("one physical sampler batch did not produce one aggregate descriptor")
    descriptor = descriptors[0]
    if (
        descriptor["batch_rows"] != 2
        or descriptor["generator_count"] != 2
        or descriptor["generator_indices"] != [0, 1]
        or descriptor["generator_seeds"] != [11, 13]
        or descriptor["sampler_call_count"] != 1
        or descriptor["route"]
        != "vllm.v1.sample.ops.topk_topp_sampler.TopKTopPSampler.forward_native"
        or descriptor["passed"] is not True
    ):
        raise AssertionError("sampler aggregate descriptor drifted from the physical batch")


def test_sampler_timed_hook_registers_batch_once_and_counts_decode_calls_in_o1() -> None:
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    generators = _CountingGeneratorMap(
        {
            0: torch.Generator().manual_seed(11),
            1: torch.Generator().manual_seed(13),
        }
    )
    logits = torch.zeros((2, 8), dtype=torch.float32)
    top_k = torch.full((2,), 4, dtype=torch.int32)

    sampler(logits.clone(), generators, top_k, None)
    sampler(logits.clone(), generators, top_k, None)

    if generators.proof_iteration_count != 1:
        raise AssertionError("sampler proof iterated the full generator batch after descriptor registration")
    descriptors = worker._evo2_sampler_batch_descriptors
    if len(descriptors) != 1 or descriptors[0]["sampler_call_count"] != 2:
        raise AssertionError("sampler proof did not retain one O(1) aggregate decode counter")


@pytest.mark.parametrize("mutation", ("replace", "reorder", "reseed"))
def test_sampler_phase_boundary_rejects_middle_row_mutation(monkeypatch, mutation: str) -> None:
    monkeypatch.setattr(sampler_module, "sampler_installation_provenance", lambda: {"passed": True})
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
        2: torch.Generator().manual_seed(17),
    }
    logits = torch.zeros((3, 8), dtype=torch.float32)
    top_k = torch.full((3,), 4, dtype=torch.int32)
    sampler(logits, generators, top_k, None)

    if mutation == "replace":
        generators[1] = torch.Generator().manual_seed(19)
    elif mutation == "reorder":
        generators[1], generators[2] = generators[2], generators[1]
    else:
        generators[1].manual_seed(19)

    with pytest.raises(RuntimeError, match="generator (identity|seed) changed"):
        snapshot_sampler_route_proof(worker, require_generation_observations=True)


def test_sampler_o1_probe_rejects_middle_row_replacement_on_next_decode() -> None:
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
        2: torch.Generator().manual_seed(17),
    }
    logits = torch.zeros((3, 8), dtype=torch.float32)
    top_k = torch.full((3,), 4, dtype=torch.int32)
    sampler(logits, generators, top_k, None)
    generators[1] = torch.Generator().manual_seed(19)

    with pytest.raises(RuntimeError, match="generator identity changed"):
        sampler(logits, generators, top_k, None)


def test_sampler_rejects_middle_mutation_followed_by_unverified_batch_transition() -> None:
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
        2: torch.Generator().manual_seed(17),
    }
    logits = torch.zeros((3, 8), dtype=torch.float32)
    top_k = torch.full((3,), 4, dtype=torch.int32)
    for _ in range(3):
        sampler(logits, generators, top_k, None)
    generators[1] = torch.Generator().manual_seed(19)
    generators.clear()
    generators.update(
        {
            0: torch.Generator().manual_seed(23),
            1: torch.Generator().manual_seed(29),
        }
    )

    with pytest.raises(RuntimeError, match="boundary verification.*transition"):
        sampler(
            torch.zeros((2, 8), dtype=torch.float32),
            generators,
            torch.full((2,), 4, dtype=torch.int32),
            None,
        )


def test_sampler_allows_batch_transition_after_full_row_boundary_snapshot(monkeypatch) -> None:
    monkeypatch.setattr(sampler_module, "sampler_installation_provenance", lambda: {"passed": True})
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    generators = {
        0: torch.Generator().manual_seed(11),
        1: torch.Generator().manual_seed(13),
        2: torch.Generator().manual_seed(17),
    }
    sampler(
        torch.zeros((3, 8), dtype=torch.float32),
        generators,
        torch.full((3,), 4, dtype=torch.int32),
        None,
    )
    snapshot_sampler_route_proof(worker, require_generation_observations=True)
    generators.clear()
    generators.update(
        {
            0: torch.Generator().manual_seed(23),
            1: torch.Generator().manual_seed(29),
        }
    )
    sampler(
        torch.zeros((2, 8), dtype=torch.float32),
        generators,
        torch.full((2,), 4, dtype=torch.int32),
        None,
    )

    proof = snapshot_sampler_route_proof(worker, require_generation_observations=True)

    descriptors = proof["generation_batch_descriptors"]
    if len(descriptors) != 2:
        raise AssertionError("verified physical batch transition did not retain two descriptors")
    if any(
        descriptor["full_row_identity_verification_mode"] != "generation-call-boundary-full-row"
        for descriptor in descriptors
    ):
        raise AssertionError("physical batch transition lacks full-row boundary verification")


def test_sampler_timed_hook_contains_no_generator_state_hash_or_host_copy() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(install_sampler_route_proof_hook)))
    monitored = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "monitored_forward"
        ),
        None,
    )
    if monitored is None:
        raise AssertionError("sampler timed proof hook is missing")
    forbidden = {"_generator_state_sha256", "get_state", "cpu", "numpy", "sha256"}
    used = {
        node.id
        for node in ast.walk(monitored)
        if isinstance(node, ast.Name)
    } | {
        node.attr
        for node in ast.walk(monitored)
        if isinstance(node, ast.Attribute)
    }
    if forbidden & used:
        raise AssertionError("sampler timed hook performs endpoint hashing or a host synchronization")


def test_sampler_route_hook_rejects_recreated_generator_with_same_seed() -> None:
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    logits = torch.zeros((1, 8), dtype=torch.float32)
    top_k = torch.full((1,), 4, dtype=torch.int32)
    sampler(logits, {0: torch.Generator().manual_seed(11)}, top_k, None)

    with pytest.raises(RuntimeError, match="persistent request-local generator"):
        sampler(logits, {0: torch.Generator().manual_seed(11)}, top_k, None)


def test_sampler_route_hook_rejects_empty_dummy_without_advancing_state() -> None:
    worker = _worker(_NativeTopKSampler())
    install_sampler_route_proof_hook(
        worker,
        environ={"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_USE_V2_MODEL_RUNNER": "0"},
    )
    sampler = worker.model_runner.sampler.topk_topp_sampler
    sentinel = torch.Generator().manual_seed(11)
    before = sentinel.get_state().clone()

    with pytest.raises(RuntimeError, match="empty dummy/capture"):
        sampler(
            torch.zeros((0, 8), dtype=torch.float32),
            {},
            torch.empty((0,), dtype=torch.int32),
            None,
        )
    if not torch.equal(sentinel.get_state(), before):
        raise AssertionError("rejected empty dummy/capture call advanced unrelated generator state")
    if worker._evo2_sampler_batch_descriptors or worker._evo2_sampler_generators_by_seed:
        raise AssertionError("rejected empty dummy/capture call retained sampler evidence")


def test_sampler_proof_binds_exact_request_advancement_to_accepted_tokens(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch, sampler_calls=2)
    summary = validate_sampler_proof_evidence(
        proof,
        expected_environment=environment,
        expected_installation=installation,
        expected_seed_batches=((11, 13),),
        expected_request_generations=_expected_request_generations(accepted_output_token_count=2),
        require_generation_observations=True,
    )

    expected = [
        {
            "request_id": "audit-request-11",
            "seed": 11,
            "accepted_output_token_count": 2,
            "observed_sampler_call_count": 2,
        },
        {
            "request_id": "audit-request-13",
            "seed": 13,
            "accepted_output_token_count": 2,
            "observed_sampler_call_count": 2,
        },
    ]
    if summary["request_generator_advancement"] != expected:
        raise AssertionError("sampler proof did not bind exact advancement to caller request IDs")


def test_sampler_proof_rejects_extra_dummy_request_advancement(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch, sampler_calls=3)

    with pytest.raises(AssertionError, match="accepted output token"):
        validate_sampler_proof_evidence(
            proof,
            expected_environment=environment,
            expected_installation=installation,
            expected_seed_batches=((11, 13),),
            expected_request_generations=_expected_request_generations(accepted_output_token_count=2),
            require_generation_observations=True,
        )


@pytest.mark.parametrize(
    "tamper",
    ("missing", "nonadvancing", "broken-continuity", "wrong-count"),
)
def test_sampler_proof_rejects_malformed_request_advancement(monkeypatch, tamper: str) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch)
    advancement = proof["generation_batch_descriptors"]
    if tamper == "missing":
        advancement.pop()
    elif tamper == "nonadvancing":
        advancement[0]["phase_end_state_sha256"][0] = advancement[0]["initial_reference_state_sha256"][0]
    elif tamper == "broken-continuity":
        advancement[0]["persistent_generator_identity"] = False
    else:
        advancement[0]["sampler_call_count"] = 2

    with pytest.raises(AssertionError, match="descriptor|endpoint|accepted output token|advancement"):
        validate_sampler_proof_evidence(
            proof,
            expected_environment=environment,
            expected_installation=installation,
            expected_seed_batches=((11, 13),),
            expected_request_generations=_expected_request_generations(),
            require_generation_observations=True,
        )


def test_sampler_proof_validator_recomputes_chosen_values_from_retained_full_distribution(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch)
    validate_sampler_proof_evidence(
        proof,
        expected_environment=environment,
        expected_installation=installation,
        expected_seed_batches=((11, 13),),
        expected_request_generations=_expected_request_generations(),
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
            expected_request_generations=_expected_request_generations(),
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
            expected_request_generations=_expected_request_generations(),
            require_generation_observations=True,
        )


def test_initialized_sampler_proof_accepts_behavioral_preflight_without_generation_calls(monkeypatch) -> None:
    proof, environment, installation = _sampler_proof(monkeypatch)
    proof["generation_batch_descriptors"] = []

    summary = validate_sampler_proof_evidence(
        proof,
        expected_environment=environment,
        expected_installation=installation,
        expected_seed_batches=(),
        expected_request_generations=(),
        require_generation_observations=False,
    )

    if summary["generation_batch_descriptor_count"] != 0 or summary["physical_seed_batches"] != []:
        raise AssertionError("initialized sampler proof retained generation descriptors")
