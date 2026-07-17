# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Fail-closed vLLM per-request sampling capability and behavioral proof."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from bionemo.evo2.vllm import sampler_runtime as _sampler_runtime


_V1_MODEL_RUNNER_MODULE = _sampler_runtime.V1_MODEL_RUNNER_MODULE
_TOPK_SAMPLER_MODULE = _sampler_runtime.TOPK_SAMPLER_MODULE
_NATIVE_ROUTE = _sampler_runtime.NATIVE_ROUTE
_NEMO_VLLM_ACTOR_FQN = _sampler_runtime.NEMO_VLLM_ACTOR_FQN
sampler_runtime_environment_contract = _sampler_runtime.sampler_runtime_environment_contract
_EXPECTED_DISTRIBUTIONS = {
    "vllm": "0.20.0",
    "flashinfer-python": "0.6.8.post1",
    "flashinfer-cubin": "0.6.8.post1",
    "flashinfer-jit-cache": "0.6.8.post1+cu130",
}
_EXPECTED_SOURCE_SHA256 = {
    "vllm/v1/worker/gpu_model_runner.py": "167ca8cb86bd481fa5241fea4a6c603f03301b42e249bace56282108b2e4aee2",
    "vllm/v1/sample/ops/topk_topp_sampler.py": "be6c3aa3ee35f4f03435a77653187028ceb0775c4100a1b23c905fea4505939f",
    "flashinfer/sampling.py": "be613d99da8a3cd591cddb4b6ec2c04641c57808d86ce7742f7ba74d4a2ae172",
    "flashinfer/data/include/flashinfer/sampling.cuh": "386259bda23ff1a80630f1f7b1e5bd377c726aa89a8a67755af80939b322a1e4",
}
_REQUEST_IDS = ("request-0", "request-1", "request-2", "request-3")
_REQUEST_SEEDS = (11, 1_000_014, 2_000_017, 3_000_020)
_STEPS = 8


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _launcher_selection_provenance(invoked_python: Path) -> dict[str, Any]:
    system_flag = os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM")
    if system_flag not in (None, "0", "1"):
        raise RuntimeError("NEMO_RL_PY_EXECUTABLES_SYSTEM must be unset, 0, or 1")
    nemo_spec = importlib.util.find_spec("nemo_rl")
    isolated_python = None
    if nemo_spec is not None and nemo_spec.origin is not None:
        nemo_repository = Path(nemo_spec.origin).resolve().parent.parent
        venv_root = Path(os.environ.get("NEMO_RL_VENV_DIR", nemo_repository / "venvs")).resolve()
        isolated_python = venv_root / _NEMO_VLLM_ACTOR_FQN / "bin" / "python"
    isolated_exists = isolated_python is not None and isolated_python.is_file()
    system_selected = system_flag == "1"
    if system_selected and isolated_python is not None and invoked_python == isolated_python.absolute():
        raise RuntimeError("system-selected runtime unexpectedly executed the isolated vLLM worker interpreter")
    return {
        "nemo_rl_py_executables_system": system_flag,
        "selected_environment": "system-runtime" if system_selected else "launcher-default-runtime",
        "executing_python_path": str(invoked_python),
        "isolated_worker_environment": {
            "path": None if isolated_python is None else str(isolated_python),
            "exists": isolated_exists,
            "status": (
                "installed-but-bypassed"
                if system_selected and isolated_exists
                else "not-installed-but-bypassed"
                if system_selected
                else "launcher-default-candidate"
            ),
            "attested_as_executing": False if system_selected else None,
        },
    }


def sampler_installation_provenance(*, require_loaded_modules: bool = True) -> dict[str, Any]:
    """Pin the exact vLLM/FlashInfer source and distribution metadata under test."""
    versions = {}
    metadata = {}
    for name, expected in _EXPECTED_DISTRIBUTIONS.items():
        distribution = importlib.metadata.distribution(name)
        if distribution.version != expected:
            raise RuntimeError(f"installed {name} version {distribution.version} is not audited {expected}")
        versions[name] = distribution.version
        record = Path(distribution._path) / "RECORD"
        if not record.is_file():
            raise RuntimeError(f"installed {name} distribution has no RECORD metadata")
        metadata[name] = {
            "record_path": str(record.resolve()),
            "record_sha256": _sha256_file(record),
        }

    roots = {}
    for package_name in ("vllm", "flashinfer"):
        spec = importlib.util.find_spec(package_name)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"installed {package_name} package could not be resolved")
        locations = tuple(Path(item).resolve() for item in spec.submodule_search_locations)
        if len(locations) != 1:
            raise RuntimeError(f"installed {package_name} package resolved to multiple roots")
        roots[package_name] = locations[0]

    site_root = roots["vllm"].parent
    if roots["flashinfer"].parent != site_root:
        raise RuntimeError("vLLM and FlashInfer must resolve from the same worker environment")
    source_files = []
    for relative_path, expected_sha256 in _EXPECTED_SOURCE_SHA256.items():
        path = site_root / relative_path
        if not path.is_file():
            raise RuntimeError(f"audited sampler source is missing: {path}")
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise RuntimeError(f"audited sampler source drifted: {relative_path}")
        source_files.append(
            {
                "path": str(path),
                "relative_path": relative_path,
                "size_bytes": path.stat().st_size,
                "sha256": observed_sha256,
            }
        )
    invoked_python = Path(sys.executable).absolute()
    resolved_python = invoked_python.resolve()
    launcher_selection = _launcher_selection_provenance(invoked_python)
    loaded_modules = {}
    for module_name in (
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.sample.ops.topk_topp_sampler",
        "flashinfer",
        "flashinfer.sampling",
    ):
        module = sys.modules.get(module_name)
        module_path = None if module is None else getattr(module, "__file__", None)
        resolved_module_path = None if module_path is None else Path(module_path).resolve()
        loaded_modules[module_name] = {
            "loaded": module is not None,
            "path": None if resolved_module_path is None else str(resolved_module_path),
            "sha256": (
                None
                if resolved_module_path is None or not resolved_module_path.is_file()
                else _sha256_file(resolved_module_path)
            ),
        }
    expected_loaded_paths = {
        "vllm.v1.worker.gpu_model_runner": site_root / "vllm/v1/worker/gpu_model_runner.py",
        "vllm.v1.sample.ops.topk_topp_sampler": site_root / "vllm/v1/sample/ops/topk_topp_sampler.py",
        "flashinfer.sampling": site_root / "flashinfer/sampling.py",
    }
    for module_name, expected_path in expected_loaded_paths.items():
        module = loaded_modules[module_name]
        if module["loaded"] is True and module["path"] != str(expected_path):
            raise RuntimeError(f"production sampler module loaded from an unaudited path: {module_name}")
    if require_loaded_modules:
        for required_module in (
            "vllm.v1.worker.gpu_model_runner",
            "vllm.v1.sample.ops.topk_topp_sampler",
        ):
            if loaded_modules[required_module]["loaded"] is not True:
                raise RuntimeError(f"production sampler module is not loaded: {required_module}")
    return {
        "schema_version": 1,
        "executing_python": {
            "invoked_path": str(invoked_python),
            "resolved_path": str(resolved_python),
            "resolved_sha256": _sha256_file(resolved_python),
            "invoked_symlink_target": (os.readlink(invoked_python) if invoked_python.is_symlink() else None),
            "nemo_rl_py_executables_system": os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM"),
        },
        "launcher_selection": launcher_selection,
        "distributions": versions,
        "distribution_metadata": metadata,
        "loaded_modules": loaded_modules,
        "required_sampler_modules_loaded": all(
            loaded_modules[module_name]["loaded"] is True
            for module_name in (
                "vllm.v1.worker.gpu_model_runner",
                "vllm.v1.sample.ops.topk_topp_sampler",
            )
        ),
        "source_files": source_files,
        "source_contract_sha256": hashlib.sha256(
            "".join(item["sha256"] for item in source_files).encode()
        ).hexdigest(),
        "flashinfer_per_row_seed_arrays_supported": False,
        "flashinfer_impact_on_selected_vllm_route": "none-route-bypassed",
        "passed": True,
    }


def sampler_installation_contract(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce sampler installation evidence to the proof/speed linkage identity."""
    executing_python = provenance.get("executing_python")
    launcher_selection = provenance.get("launcher_selection")
    distributions = provenance.get("distributions")
    metadata = provenance.get("distribution_metadata")
    source_files = provenance.get("source_files")
    if (
        not isinstance(executing_python, dict)
        or not isinstance(launcher_selection, dict)
        or not isinstance(distributions, dict)
        or not isinstance(metadata, dict)
        or not isinstance(source_files, list)
    ):
        raise ValueError("sampler installation provenance is incomplete")
    if distributions != _EXPECTED_DISTRIBUTIONS:
        raise ValueError("sampler distribution versions drifted")
    if launcher_selection.get("executing_python_path") != executing_python.get("invoked_path"):
        raise ValueError("sampler launcher selection does not name the executing interpreter")
    if launcher_selection.get("nemo_rl_py_executables_system") == "1":
        isolated = launcher_selection.get("isolated_worker_environment")
        if (
            launcher_selection.get("selected_environment") != "system-runtime"
            or not isinstance(isolated, dict)
            or isolated.get("status") not in {"installed-but-bypassed", "not-installed-but-bypassed"}
            or isolated.get("attested_as_executing") is not False
        ):
            raise ValueError("sampler system launcher provenance conflates the bypassed worker environment")
    source_identity = [
        {
            "relative_path": item.get("relative_path"),
            "size_bytes": item.get("size_bytes"),
            "sha256": item.get("sha256"),
        }
        for item in source_files
        if isinstance(item, dict)
    ]
    if (
        len(source_identity) != len(_EXPECTED_SOURCE_SHA256)
        or {item["relative_path"]: item["sha256"] for item in source_identity} != _EXPECTED_SOURCE_SHA256
    ):
        raise ValueError("sampler source identities drifted")
    return {
        "schema_version": 1,
        "executing_python": dict(executing_python),
        "launcher_selection": dict(launcher_selection),
        "distributions": dict(distributions),
        "distribution_record_sha256": {
            name: item.get("record_sha256") for name, item in sorted(metadata.items()) if isinstance(item, dict)
        },
        "source_files": source_identity,
        "source_contract_sha256": provenance.get("source_contract_sha256"),
        "flashinfer_per_row_seed_arrays_supported": False,
        "flashinfer_impact_on_selected_vllm_route": "none-route-bypassed",
    }


def _resolve_native_topk_sampler(worker: Any) -> tuple[Any, dict[str, Any]]:
    model_runner = getattr(worker, "model_runner", None)
    if model_runner is None or type(model_runner).__module__ != _V1_MODEL_RUNNER_MODULE:
        raise RuntimeError("Evo2 sampler proof requires vLLM's audited V1 GPU model runner")
    model_sampler = getattr(model_runner, "sampler", None)
    topk_sampler = getattr(model_sampler, "topk_topp_sampler", None)
    if topk_sampler is None or type(topk_sampler).__module__ != _TOPK_SAMPLER_MODULE:
        raise RuntimeError("Evo2 sampler proof could not resolve the production TopKTopPSampler")
    if getattr(topk_sampler, "logprobs_mode", None) != "processed_logprobs":
        raise RuntimeError("Evo2 sampler proof requires processed_logprobs mode")
    original = getattr(topk_sampler, "_evo2_sampler_original_forward", topk_sampler.forward)
    function = getattr(original, "__func__", original)
    if getattr(function, "__name__", None) != "forward_native":
        raise RuntimeError("Evo2 production sampler did not select the audited native route")
    return topk_sampler, {
        "model_runner_type": f"{type(model_runner).__module__}.{type(model_runner).__qualname__}",
        "sampler_type": f"{type(topk_sampler).__module__}.{type(topk_sampler).__qualname__}",
        "selected_route": _NATIVE_ROUTE,
    }


def _fp32_hex(value: float) -> str:
    return struct.pack(">f", value).hex()


def _generator_state_sha256(generator: Any) -> str:
    state = generator.get_state()
    raw = state.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _execute_schedule(
    sampler: Any,
    *,
    device: Any,
    request_seeds: Mapping[str, int],
    groups_by_step: Sequence[Sequence[Sequence[str]]],
) -> tuple[dict[str, dict[str, list[Any]]], list[str]]:
    import torch

    generators = {
        request_id: torch.Generator(device=device).manual_seed(seed) for request_id, seed in request_seeds.items()
    }
    streams = {
        request_id: {"token_ids": [], "chosen_logprobs": [], "chosen_logprob_fp32_hex": []}
        for request_id in request_seeds
    }
    base_logits = torch.full((512,), float("-inf"), dtype=torch.float32, device=device)
    base_logits[:4] = torch.tensor((0.4, 0.2, 0.0, -0.2), dtype=torch.float32, device=device)
    retained_processed_logprobs = None
    for groups in groups_by_step:
        seen = set()
        for raw_group in groups:
            group = tuple(raw_group)
            if not group or any(request_id not in generators for request_id in group):
                raise RuntimeError("sampler preflight schedule contains an unknown or empty group")
            if seen.intersection(group):
                raise RuntimeError("sampler preflight scheduled one request twice in a step")
            seen.update(group)
            logits = base_logits.repeat(len(group), 1)
            top_k = torch.full((len(group),), 4, dtype=torch.int32, device=device)
            local_generators = {index: generators[request_id] for index, request_id in enumerate(group)}
            before_states = {
                request_id: _generator_state_sha256(generators[request_id]) for request_id in group
            }
            sampled, processed_logprobs = sampler(logits, local_generators, top_k, None)
            if any(
                _generator_state_sha256(generators[request_id]) == before_states[request_id]
                for request_id in group
            ):
                raise RuntimeError("sampler behavioral preflight native forward did not advance generator state")
            if processed_logprobs is None or processed_logprobs.shape != logits.shape:
                raise RuntimeError("processed-logprob sampler route did not retain its chosen-policy distribution")
            for row in processed_logprobs.detach().cpu().tolist():
                row_bits = [_fp32_hex(float(value)) for value in row]
                if retained_processed_logprobs is None:
                    retained_processed_logprobs = row_bits
                elif row_bits != retained_processed_logprobs:
                    raise RuntimeError("sampler preflight processed-logprob rows unexpectedly differ")
            chosen = processed_logprobs.gather(1, sampled.view(-1, 1)).view(-1)
            token_ids = sampled.detach().cpu().tolist()
            chosen_values = chosen.detach().cpu().tolist()
            for request_id, token_id, logprob in zip(group, token_ids, chosen_values, strict=True):
                streams[request_id]["token_ids"].append(int(token_id))
                streams[request_id]["chosen_logprobs"].append(float(logprob))
                streams[request_id]["chosen_logprob_fp32_hex"].append(_fp32_hex(float(logprob)))
    if retained_processed_logprobs is None:
        raise RuntimeError("sampler preflight retained no processed-logprob tensor")
    return streams, retained_processed_logprobs


def _repeat_groups(groups: Sequence[Sequence[str]]) -> tuple[tuple[tuple[str, ...], ...], ...]:
    normalized = tuple(tuple(group) for group in groups)
    return tuple(normalized for _ in range(_STEPS))


def _require_matching_streams(
    name: str,
    observed: Mapping[str, dict[str, list[Any]]],
    expected: Mapping[str, dict[str, list[Any]]],
    *,
    truncated_request: str | None = None,
    truncated_steps: int = 0,
) -> None:
    for request_id in _REQUEST_IDS:
        expected_stream = expected[request_id]
        if request_id == truncated_request:
            expected_stream = {key: values[:truncated_steps] for key, values in expected_stream.items()}
        if observed[request_id] != expected_stream:
            raise RuntimeError(f"per-request sampler invariance failed for {name}/{request_id}")


def run_sampler_seed_behavioral_preflight(
    worker: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Exercise the actual production sampler under adversarial physical schedules."""
    environment = sampler_runtime_environment_contract(environ)
    sampler_object, route = _resolve_native_topk_sampler(worker)
    sampler = getattr(sampler_object, "_evo2_sampler_original_forward", sampler_object)
    device = getattr(worker.model_runner, "device", None)
    if device is None:
        raise RuntimeError("vLLM model runner did not expose its sampler device")
    seeds = dict(zip(_REQUEST_IDS, _REQUEST_SEEDS, strict=True))
    oracle, oracle_distribution = _execute_schedule(
        sampler,
        device=device,
        request_seeds=seeds,
        groups_by_step=_repeat_groups(tuple((request_id,) for request_id in _REQUEST_IDS)),
    )
    replay, replay_distribution = _execute_schedule(
        sampler,
        device=device,
        request_seeds=seeds,
        groups_by_step=_repeat_groups(tuple((request_id,) for request_id in _REQUEST_IDS)),
    )
    _require_matching_streams("independent-replay", replay, oracle)
    if replay_distribution != oracle_distribution:
        raise RuntimeError("sampler replay processed-logprob tensor differs from its oracle")
    if len({tuple(oracle[request_id]["token_ids"]) for request_id in _REQUEST_IDS}) != len(_REQUEST_IDS):
        raise RuntimeError("sampler preflight seeds did not produce discriminating independent streams")

    variants = {}
    schedules = {
        "batched": _repeat_groups((_REQUEST_IDS,)),
        "reordered": tuple(
            (tuple(reversed(_REQUEST_IDS)) if step % 2 else (_REQUEST_IDS[2:] + _REQUEST_IDS[:2]),)
            for step in range(_STEPS)
        ),
        "subwaves": _repeat_groups((_REQUEST_IDS[:2], _REQUEST_IDS[2:])),
    }
    for name, schedule in schedules.items():
        observed, distribution = _execute_schedule(
            sampler,
            device=device,
            request_seeds=seeds,
            groups_by_step=schedule,
        )
        _require_matching_streams(name, observed, oracle)
        if distribution != oracle_distribution:
            raise RuntimeError(f"sampler {name} processed-logprob tensor differs from its oracle")
        variants[name] = {
            "streams": observed,
            "processed_logprob_fp32_hex": distribution,
            "passed": True,
        }

    deactivated_schedule = tuple((_REQUEST_IDS,) if step < 2 else (_REQUEST_IDS[1:],) for step in range(_STEPS))
    deactivated, deactivated_distribution = _execute_schedule(
        sampler,
        device=device,
        request_seeds=seeds,
        groups_by_step=deactivated_schedule,
    )
    _require_matching_streams(
        "deactivated",
        deactivated,
        oracle,
        truncated_request=_REQUEST_IDS[0],
        truncated_steps=2,
    )
    if deactivated_distribution != oracle_distribution:
        raise RuntimeError("sampler deactivated processed-logprob tensor differs from its oracle")
    variants["deactivated"] = {
        "streams": deactivated,
        "processed_logprob_fp32_hex": deactivated_distribution,
        "passed": True,
    }

    padding_id = "padding-row"
    padded_seeds = {**seeds, padding_id: 4_000_023}
    padded, padded_distribution = _execute_schedule(
        sampler,
        device=device,
        request_seeds=padded_seeds,
        groups_by_step=tuple(
            ((padding_id, *_REQUEST_IDS),) if step % 2 else ((*_REQUEST_IDS, padding_id),) for step in range(_STEPS)
        ),
    )
    _require_matching_streams("padded", padded, oracle)
    if padded_distribution != oracle_distribution:
        raise RuntimeError("sampler padded processed-logprob tensor differs from its oracle")
    variants["padded"] = {
        "streams": {request_id: padded[request_id] for request_id in _REQUEST_IDS},
        "padding_stream": padded[padding_id],
        "processed_logprob_fp32_hex": padded_distribution,
        "passed": True,
    }

    return {
        "schema_version": 1,
        "environment_contract": environment,
        **route,
        "request_ids": list(_REQUEST_IDS),
        "request_seeds": list(_REQUEST_SEEDS),
        "steps_per_request": _STEPS,
        "oracle_processed_logprob_fp32_hex": oracle_distribution,
        "oracle_streams": [{"request_id": request_id, **oracle[request_id]} for request_id in _REQUEST_IDS],
        "oracle_replay_processed_logprob_fp32_hex": replay_distribution,
        "oracle_replay_streams": [{"request_id": request_id, **replay[request_id]} for request_id in _REQUEST_IDS],
        "oracle_replay_exact": replay == oracle,
        "variants": variants,
        "native_forward_state_advancement_proven": True,
        "flashinfer_sampling_calls": 0,
        "passed": True,
    }


def install_sampler_route_proof_hook(
    worker: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Require and record one explicit generator for every production sampling row."""
    environment = sampler_runtime_environment_contract(environ)
    sampler, route = _resolve_native_topk_sampler(worker)
    if not hasattr(sampler, "_evo2_sampler_original_forward"):
        original = sampler.forward
        sampler._evo2_sampler_original_forward = original

        def monitored_forward(logits, generators, k, p):
            rows = int(logits.shape[0])
            if rows <= 0:
                raise RuntimeError("production sampler rejects an empty dummy/capture batch")
            if len(generators) != rows:
                raise RuntimeError("production sampler requires one explicit generator per active row")
            first_generator = generators.get(0)
            final_generator = generators.get(rows - 1)
            if first_generator is None or final_generator is None:
                raise RuntimeError("production sampler generator rows are not contiguous")
            hot_signature = (
                id(generators),
                rows,
                id(first_generator),
                id(final_generator),
            )
            descriptor = worker._evo2_sampler_active_batch_descriptor
            if descriptor is None or descriptor["hot_signature"] != hot_signature:
                if descriptor is not None and (
                    descriptor["boundary_verified_sampler_call_count"]
                    != descriptor["sampler_call_count"]
                ):
                    raise RuntimeError(
                        "production sampler requires full-row boundary verification before a physical batch transition"
                    )
                indices = tuple(sorted(generators))
                if indices != tuple(range(rows)):
                    raise RuntimeError("production sampler generator rows are not contiguous")
                generator_objects = tuple(generators[index] for index in indices)
                seeds = tuple(int(generator.initial_seed()) for generator in generator_objects)
                if len(seeds) != len(set(seeds)):
                    raise RuntimeError("production sampler request generators must have unique seeds")
                retained_generators = worker._evo2_sampler_generators_by_seed
                for seed, generator in zip(seeds, generator_objects, strict=True):
                    retained = retained_generators.get(seed)
                    if retained is not None and retained is not generator:
                        raise RuntimeError(
                            "production sampler requires one persistent request-local generator"
                        )
                    retained_generators[seed] = generator
                descriptor = {
                    "descriptor_index": len(worker._evo2_sampler_batch_descriptors),
                    "hot_signature": hot_signature,
                    "generator_mapping": generators,
                    "generator_objects": generator_objects,
                    "generator_seeds": seeds,
                    "identity_probe_mask": 0,
                    "boundary_verified_sampler_call_count": None,
                    "sampler_call_count": 0,
                }
                worker._evo2_sampler_batch_descriptors.append(descriptor)
                worker._evo2_sampler_active_batch_descriptor = descriptor
            probe_index = descriptor["sampler_call_count"] % rows
            if generators.get(probe_index) is not descriptor["generator_objects"][probe_index]:
                raise RuntimeError("production request-local generator identity changed within a physical batch")
            descriptor["identity_probe_mask"] |= 1 << probe_index
            result = original(logits, generators, k, p)
            descriptor["sampler_call_count"] += 1
            return result

        sampler.forward = monitored_forward
    worker._evo2_sampler_batch_descriptors = []
    worker._evo2_sampler_active_batch_descriptor = None
    worker._evo2_sampler_generators_by_seed = {}
    return {"environment_contract": environment, **route}


def snapshot_sampler_route_proof(
    worker: Any,
    *,
    require_generation_observations: bool,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Retain behavioral and actual production-call sampler route evidence."""
    environment = sampler_runtime_environment_contract(environ)
    _, route = _resolve_native_topk_sampler(worker)
    preflight = getattr(worker, "_evo2_sampler_seed_preflight", None)
    if preflight is None:
        preflight = run_sampler_seed_behavioral_preflight(worker, environ=environ)
        worker._evo2_sampler_seed_preflight = preflight
    retained_descriptors = list(getattr(worker, "_evo2_sampler_batch_descriptors", ()))
    if require_generation_observations and not retained_descriptors:
        raise RuntimeError("production proof retained no sampler batch descriptors")
    descriptor_rows = []
    if retained_descriptors:
        import torch

        active_descriptor = getattr(worker, "_evo2_sampler_active_batch_descriptor", None)
        if active_descriptor is not retained_descriptors[-1]:
            raise RuntimeError("production sampler active batch descriptor is stale or missing")
        for descriptor_index, descriptor in enumerate(retained_descriptors):
            generator_objects = descriptor["generator_objects"]
            generator_seeds = descriptor["generator_seeds"]
            sampler_call_count = descriptor["sampler_call_count"]
            if (
                descriptor["descriptor_index"] != descriptor_index
                or not generator_objects
                or len(generator_objects) != len(generator_seeds)
                or sampler_call_count <= 0
            ):
                raise RuntimeError("production sampler batch descriptor is malformed")
            if descriptor is active_descriptor:
                generator_mapping = descriptor["generator_mapping"]
                current_indices = tuple(sorted(generator_mapping))
                if current_indices != tuple(range(len(generator_objects))):
                    raise RuntimeError("production sampler phase-boundary generator rows are not contiguous")
                current_objects = tuple(generator_mapping[index] for index in current_indices)
                if any(
                    current is not retained
                    for current, retained in zip(current_objects, generator_objects, strict=True)
                ):
                    raise RuntimeError("production request-local generator identity changed at phase boundary")
                if tuple(int(generator.initial_seed()) for generator in current_objects) != generator_seeds:
                    raise RuntimeError("production request-local generator seed changed at phase boundary")
                descriptor["boundary_verified_sampler_call_count"] = sampler_call_count
            else:
                if descriptor["boundary_verified_sampler_call_count"] != sampler_call_count:
                    raise RuntimeError(
                        "production sampler descriptor retired without full-row boundary verification"
                    )
            identity_verification_mode = "generation-call-boundary-full-row"
            initial_state_sha256 = []
            phase_end_state_sha256 = []
            for seed, generator in zip(generator_seeds, generator_objects, strict=True):
                if int(generator.initial_seed()) != seed:
                    raise RuntimeError("production request-local generator seed changed across the phase")
                initial_reference = torch.Generator(device=generator.device).manual_seed(seed)
                initial_hash = _generator_state_sha256(initial_reference)
                phase_end_hash = _generator_state_sha256(generator)
                if initial_hash == phase_end_hash:
                    raise RuntimeError("production request-local generator did not advance across the phase")
                initial_state_sha256.append(initial_hash)
                phase_end_state_sha256.append(phase_end_hash)
            descriptor_rows.append(
                {
                    "descriptor_index": descriptor_index,
                    "batch_rows": len(generator_objects),
                    "generator_count": len(generator_objects),
                    "generator_indices": list(range(len(generator_objects))),
                    "generator_seeds": list(generator_seeds),
                    "sampler_call_count": sampler_call_count,
                    "full_row_identity_verified": True,
                    "full_row_identity_verification_mode": identity_verification_mode,
                    "initial_reference_state_sha256": initial_state_sha256,
                    "phase_end_state_sha256": phase_end_state_sha256,
                    "persistent_generator_identity": True,
                    "state_changed_from_seeded_initial": True,
                    "route": route["selected_route"],
                    "passed": True,
                }
            )
    return {
        "schema_version": 4,
        "environment_contract": environment,
        **route,
        "installation": sampler_installation_provenance(),
        "behavioral_preflight": preflight,
        "generation_batch_descriptors": descriptor_rows,
        "flashinfer_sampling_calls": 0,
        "passed": preflight.get("passed") is True,
    }


def _validated_streams(
    rows: Any,
    *,
    request_ids: Sequence[str],
    expected_steps: Mapping[str, int],
    label: str,
) -> dict[str, dict[str, list[Any]]]:
    if not isinstance(rows, list) or len(rows) != len(request_ids):
        raise AssertionError(f"{label} stream rows do not cover the exact requests")
    streams = {}
    for row, request_id in zip(rows, request_ids, strict=True):
        if not isinstance(row, dict) or row.get("request_id") != request_id:
            raise AssertionError(f"{label} request ownership drifted")
        token_ids = row.get("token_ids")
        logprobs = row.get("chosen_logprobs")
        logprob_hex = row.get("chosen_logprob_fp32_hex")
        steps = expected_steps[request_id]
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != steps
            or any(
                isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < 512 for token in token_ids
            )
            or not isinstance(logprobs, list)
            or len(logprobs) != steps
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in logprobs
            )
            or not isinstance(logprob_hex, list)
            or logprob_hex != [_fp32_hex(float(value)) for value in logprobs]
        ):
            raise AssertionError(f"{label} token or chosen-logprob stream is malformed")
        streams[request_id] = {
            "token_ids": token_ids,
            "chosen_logprobs": logprobs,
            "chosen_logprob_fp32_hex": logprob_hex,
        }
    return streams


def _validated_processed_logprob_tensor(value: Any, *, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != 512
        or any(
            not isinstance(bits, str)
            or len(bits) != 8
            or any(character not in "0123456789abcdef" for character in bits)
            for bits in value
        )
    ):
        raise AssertionError(f"{label} full processed-logprob tensor is malformed")
    return value


def _require_chosen_values_gathered_from_tensor(
    streams: Mapping[str, Mapping[str, list[Any]]],
    distribution: Sequence[str],
    *,
    label: str,
) -> None:
    for request_id, stream in streams.items():
        gathered = [distribution[token_id] for token_id in stream["token_ids"]]
        if stream["chosen_logprob_fp32_hex"] != gathered:
            raise AssertionError(
                f"{label}/{request_id} chosen value was not gathered from the full processed-logprob tensor"
            )


def validate_sampler_proof_evidence(
    proof: Mapping[str, Any],
    *,
    expected_environment: Mapping[str, Any],
    expected_installation: Mapping[str, Any],
    expected_seed_batches: Sequence[Sequence[int]],
    expected_request_generations: Sequence[Mapping[str, Any]],
    require_generation_observations: bool,
) -> dict[str, Any]:
    """Recompute sampler safety from retained streams, route calls, and installation evidence."""
    if not isinstance(proof, dict) or proof.get("schema_version") != 4:
        raise AssertionError("sampler proof schema is unsupported")
    if proof.get("environment_contract") != expected_environment:
        raise AssertionError("sampler environment contract drifted")
    if proof.get("selected_route") != _NATIVE_ROUTE or proof.get("flashinfer_sampling_calls") != 0:
        raise AssertionError("sampler proof did not bypass FlashInfer sampling")
    installation = proof.get("installation")
    if not isinstance(installation, dict) or installation.get("required_sampler_modules_loaded") is not True:
        raise AssertionError("sampler proof did not bind loaded production modules")
    if sampler_installation_contract(installation) != expected_installation:
        raise AssertionError("sampler installation does not match the linked runtime")
    source_files = installation.get("source_files")
    loaded_modules = installation.get("loaded_modules")
    if not isinstance(source_files, list) or not isinstance(loaded_modules, dict):
        raise AssertionError("sampler proof omitted loaded-module source identities")
    source_by_relative_path = {
        item.get("relative_path"): item
        for item in source_files
        if isinstance(item, dict) and isinstance(item.get("relative_path"), str)
    }
    expected_loaded_sources = {
        "vllm.v1.worker.gpu_model_runner": "vllm/v1/worker/gpu_model_runner.py",
        "vllm.v1.sample.ops.topk_topp_sampler": "vllm/v1/sample/ops/topk_topp_sampler.py",
    }
    for module_name, relative_path in expected_loaded_sources.items():
        module = loaded_modules.get(module_name)
        source = source_by_relative_path.get(relative_path)
        if (
            not isinstance(module, dict)
            or module.get("loaded") is not True
            or not isinstance(source, dict)
            or module.get("path") != source.get("path")
            or module.get("sha256") != source.get("sha256")
        ):
            raise AssertionError(f"loaded production sampler module is unaudited: {module_name}")

    preflight = proof.get("behavioral_preflight")
    if not isinstance(preflight, dict) or preflight.get("schema_version") != 1:
        raise AssertionError("sampler behavioral preflight is missing")
    if (
        preflight.get("environment_contract") != expected_environment
        or preflight.get("selected_route") != _NATIVE_ROUTE
        or preflight.get("request_ids") != list(_REQUEST_IDS)
        or preflight.get("request_seeds") != list(_REQUEST_SEEDS)
        or preflight.get("steps_per_request") != _STEPS
        or preflight.get("native_forward_state_advancement_proven") is not True
        or preflight.get("flashinfer_sampling_calls") != 0
    ):
        raise AssertionError("sampler behavioral preflight contract drifted")
    full_steps = {request_id: _STEPS for request_id in _REQUEST_IDS}
    oracle = _validated_streams(
        preflight.get("oracle_streams"),
        request_ids=_REQUEST_IDS,
        expected_steps=full_steps,
        label="sampler oracle",
    )
    oracle_distribution = _validated_processed_logprob_tensor(
        preflight.get("oracle_processed_logprob_fp32_hex"),
        label="sampler oracle",
    )
    _require_chosen_values_gathered_from_tensor(
        oracle,
        oracle_distribution,
        label="sampler oracle",
    )
    replay = _validated_streams(
        preflight.get("oracle_replay_streams"),
        request_ids=_REQUEST_IDS,
        expected_steps=full_steps,
        label="sampler oracle replay",
    )
    replay_distribution = _validated_processed_logprob_tensor(
        preflight.get("oracle_replay_processed_logprob_fp32_hex"),
        label="sampler oracle replay",
    )
    _require_chosen_values_gathered_from_tensor(
        replay,
        replay_distribution,
        label="sampler oracle replay",
    )
    if (
        replay != oracle
        or replay_distribution != oracle_distribution
        or preflight.get("oracle_replay_exact") is not True
    ):
        raise AssertionError("sampler independent replay differs from its oracle")
    if len({tuple(stream["token_ids"]) for stream in oracle.values()}) != len(_REQUEST_IDS):
        raise AssertionError("sampler oracle streams are not discriminating")
    variants = preflight.get("variants")
    if not isinstance(variants, dict) or set(variants) != {
        "batched",
        "reordered",
        "deactivated",
        "padded",
        "subwaves",
    }:
        raise AssertionError("sampler adversarial schedule set drifted")
    for name in ("batched", "reordered", "subwaves"):
        variant = variants[name]
        if not isinstance(variant, dict) or variant.get("passed") is not True:
            raise AssertionError(f"sampler {name} schedule did not pass")
        rows = [{"request_id": request_id, **stream} for request_id, stream in variant.get("streams", {}).items()]
        observed = _validated_streams(rows, request_ids=_REQUEST_IDS, expected_steps=full_steps, label=name)
        distribution = _validated_processed_logprob_tensor(
            variant.get("processed_logprob_fp32_hex"),
            label=f"sampler {name}",
        )
        _require_chosen_values_gathered_from_tensor(observed, distribution, label=f"sampler {name}")
        if observed != oracle or distribution != oracle_distribution:
            raise AssertionError(f"sampler {name} schedule differs from independent streams")
    deactivated = variants["deactivated"]
    deactivated_steps = {**full_steps, _REQUEST_IDS[0]: 2}
    rows = [{"request_id": request_id, **stream} for request_id, stream in deactivated.get("streams", {}).items()]
    observed = _validated_streams(
        rows,
        request_ids=_REQUEST_IDS,
        expected_steps=deactivated_steps,
        label="deactivated",
    )
    deactivated_distribution = _validated_processed_logprob_tensor(
        deactivated.get("processed_logprob_fp32_hex"),
        label="sampler deactivated",
    )
    _require_chosen_values_gathered_from_tensor(
        observed,
        deactivated_distribution,
        label="sampler deactivated",
    )
    if (
        deactivated.get("passed") is not True
        or any(
            observed[request_id]
            != (
                {key: values[:2] for key, values in oracle[request_id].items()}
                if request_id == _REQUEST_IDS[0]
                else oracle[request_id]
            )
            for request_id in _REQUEST_IDS
        )
        or deactivated_distribution != oracle_distribution
    ):
        raise AssertionError("sampler deactivation perturbed a surviving request")
    padded = variants["padded"]
    rows = [{"request_id": request_id, **stream} for request_id, stream in padded.get("streams", {}).items()]
    observed = _validated_streams(rows, request_ids=_REQUEST_IDS, expected_steps=full_steps, label="padded")
    padding = padded.get("padding_stream")
    padded_distribution = _validated_processed_logprob_tensor(
        padded.get("processed_logprob_fp32_hex"),
        label="sampler padded",
    )
    _require_chosen_values_gathered_from_tensor(observed, padded_distribution, label="sampler padded")
    if (
        padded.get("passed") is not True
        or observed != oracle
        or padded_distribution != oracle_distribution
        or not isinstance(padding, dict)
    ):
        raise AssertionError("sampler padding perturbed a surviving request")
    padding_stream = _validated_streams(
        [{"request_id": "padding-row", **padding}],
        request_ids=("padding-row",),
        expected_steps={"padding-row": _STEPS},
        label="padding row",
    )
    _require_chosen_values_gathered_from_tensor(
        padding_stream,
        padded_distribution,
        label="padding row",
    )

    if type(expected_request_generations) not in (list, tuple):
        raise AssertionError("expected request generations must be an exact sequence")
    expected_requests = []
    for row in expected_request_generations:
        if type(row) is not dict or set(row) != {
            "request_id",
            "seed",
            "accepted_output_token_count",
        }:
            raise AssertionError("expected request generation row fields are not exact")
        if type(row["request_id"]) is not str or not row["request_id"]:
            raise AssertionError("expected request generation ID is malformed")
        if type(row["seed"]) is not int or row["seed"] < 0:
            raise AssertionError("expected request generation seed is malformed")
        if type(row["accepted_output_token_count"]) is not int or row["accepted_output_token_count"] <= 0:
            raise AssertionError("expected accepted output token count is malformed")
        expected_requests.append(dict(row))
    if len({row["request_id"] for row in expected_requests}) != len(expected_requests) or len(
        {row["seed"] for row in expected_requests}
    ) != len(expected_requests):
        raise AssertionError("expected request generation IDs or seeds are not unique")
    if require_generation_observations and not expected_requests:
        raise AssertionError("generation sampler proof requires exact expected requests")

    if type(expected_seed_batches) not in (list, tuple):
        raise AssertionError("expected sampler seed batches must be an exact sequence")
    expected_batch_rows = []
    for batch in expected_seed_batches:
        if type(batch) not in (list, tuple) or not batch:
            raise AssertionError("expected sampler seed batch is empty or malformed")
        normalized = tuple(batch)
        if any(type(seed) is not int or seed < 0 for seed in normalized) or len(normalized) != len(set(normalized)):
            raise AssertionError("expected sampler seed batch is malformed or non-unique")
        expected_batch_rows.append(normalized)
    if len(expected_batch_rows) != len(set(expected_batch_rows)):
        raise AssertionError("expected sampler seed batches are duplicated")
    expected_batches = set(expected_batch_rows)
    if require_generation_observations and not expected_batches:
        raise AssertionError("generation sampler proof requires exact expected seed batches")
    expected_by_seed = {row["seed"]: row for row in expected_requests}
    flattened_batched_seeds = [seed for batch in expected_batch_rows for seed in batch]
    if len(flattened_batched_seeds) != len(set(flattened_batched_seeds)):
        raise AssertionError("expected sampler batches reuse a caller request seed")
    batched_seeds = set(flattened_batched_seeds)
    if batched_seeds != set(expected_by_seed):
        raise AssertionError("expected sampler batches do not cover the caller request seeds")
    expected_batch_call_counts = {}
    for batch in expected_batch_rows:
        accepted_counts = {expected_by_seed[seed]["accepted_output_token_count"] for seed in batch}
        if len(accepted_counts) != 1:
            raise AssertionError("one physical sampler batch has inconsistent accepted output lengths")
        expected_batch_call_counts[batch] = next(iter(accepted_counts))

    descriptors = proof.get("generation_batch_descriptors")
    if not isinstance(descriptors, list) or (require_generation_observations and not descriptors):
        raise AssertionError("sampler generation batch descriptors are missing")
    observed_batch_call_counts: dict[tuple[int, ...], int] = {}
    observed_seed_call_counts = {seed: 0 for seed in expected_by_seed}
    descriptor_fields = {
        "descriptor_index",
        "batch_rows",
        "generator_count",
        "generator_indices",
        "generator_seeds",
        "sampler_call_count",
        "full_row_identity_verified",
        "full_row_identity_verification_mode",
        "initial_reference_state_sha256",
        "phase_end_state_sha256",
        "persistent_generator_identity",
        "state_changed_from_seeded_initial",
        "route",
        "passed",
    }
    for descriptor_index, descriptor in enumerate(descriptors):
        if type(descriptor) is not dict or set(descriptor) != descriptor_fields:
            raise AssertionError("sampler generation batch descriptor fields are not exact")
        rows = descriptor.get("batch_rows")
        seeds = descriptor.get("generator_seeds")
        sampler_call_count = descriptor.get("sampler_call_count")
        initial_states = descriptor.get("initial_reference_state_sha256")
        final_states = descriptor.get("phase_end_state_sha256")
        if (
            type(rows) is not int
            or rows <= 0
            or descriptor.get("generator_count") != rows
            or descriptor.get("generator_indices") != list(range(rows))
            or type(seeds) is not list
            or len(seeds) != rows
            or len(seeds) != len(set(seeds))
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or tuple(seeds) not in expected_batches
            or type(sampler_call_count) is not int
            or sampler_call_count <= 0
            or descriptor.get("full_row_identity_verified") is not True
            or descriptor.get("full_row_identity_verification_mode")
            != "generation-call-boundary-full-row"
            or type(initial_states) is not list
            or len(initial_states) != rows
            or type(final_states) is not list
            or len(final_states) != rows
            or descriptor.get("descriptor_index") != descriptor_index
            or descriptor.get("persistent_generator_identity") is not True
            or descriptor.get("state_changed_from_seeded_initial") is not True
            or descriptor.get("route") != _NATIVE_ROUTE
            or descriptor.get("passed") is not True
        ):
            raise AssertionError("sampler generation batch descriptor drifted from exact requests")
        if any(
            type(state) is not str
            or len(state) != 64
            or any(character not in "0123456789abcdef" for character in state)
            for state in (*initial_states, *final_states)
        ) or any(initial == final for initial, final in zip(initial_states, final_states, strict=True)):
            raise AssertionError("sampler generator endpoint state evidence is malformed")
        batch = tuple(seeds)
        if batch in observed_batch_call_counts:
            raise AssertionError("sampler proof repeated one physical batch descriptor")
        observed_batch_call_counts[batch] = sampler_call_count
        for seed in seeds:
            observed_seed_call_counts[seed] = sampler_call_count
    if require_generation_observations and observed_batch_call_counts != expected_batch_call_counts:
        raise AssertionError("sampler calls do not match accepted output token counts for every physical batch")
    for seed, expected in expected_by_seed.items():
        if observed_seed_call_counts[seed] != expected["accepted_output_token_count"]:
            raise AssertionError("request generator advancement differs from accepted output token accounting")
    if proof.get("passed") is not True or preflight.get("passed") is not True:
        raise AssertionError("sampler proof did not pass")
    return {
        "schema_version": 4,
        "selected_route": _NATIVE_ROUTE,
        "physical_seed_batches": [list(batch) for batch in expected_batch_rows],
        "physical_seed_batch_call_counts": [
            {
                "seeds": list(batch),
                "accepted_output_token_count": expected_batch_call_counts[batch],
                "observed_sampler_call_count": observed_batch_call_counts[batch],
            }
            for batch in expected_batch_rows
        ],
        "request_generator_advancement": [
            {
                **expected,
                "observed_sampler_call_count": observed_seed_call_counts[expected["seed"]],
            }
            for expected in expected_requests
        ],
        "generation_batch_descriptor_count": len(descriptors),
        "flashinfer_sampling_calls": 0,
        "passed": True,
    }


__all__ = [
    "install_sampler_route_proof_hook",
    "run_sampler_seed_behavioral_preflight",
    "sampler_installation_contract",
    "sampler_installation_provenance",
    "sampler_runtime_environment_contract",
    "snapshot_sampler_route_proof",
    "validate_sampler_proof_evidence",
]
