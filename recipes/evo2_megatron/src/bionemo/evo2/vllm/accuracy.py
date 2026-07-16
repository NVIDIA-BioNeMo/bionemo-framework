# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Fail-closed accuracy contracts for optimized Evo2 vLLM generation."""

from __future__ import annotations

import base64
import binascii
import csv
import hashlib
import io
import math
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from bionemo.evo2.vllm.artifact_io import ArtifactSnapshotError, read_byte_snapshot, read_jsonl_snapshot
from bionemo.evo2.vllm.benchmark import WorkloadManifest, WorkloadRequest, build_request_waves
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


CANONICAL_7B_CHECKPOINT = "evo2/7b-1m:1.0"
CANONICAL_7B_PROMPTS_SHA256 = "7e525370e8fb66ef20c0e8d7959f6a0f8e78e5e973819cf3db6f4d23b0e19c0c"

_EXPECTED_IDENTITIES = (97.60, 89.63, 80.03, 84.57)
_SEQUENCE_LENGTHS = (6538, 7056, 6160, 7616)
_PROMPT_LENGTHS = (3269, 3528, 3080, 3808)
_SEQUENCE_SHA256 = (
    "61aaa0e7af4d48931b628ef6ac9b2115e174e65acfd4ae1f949f43fac4bc6a81",
    "fe23adff7ab5242144c41cea922f7c70f04dad31a6681795590ff79743fbd2f3",
    "f014724b175ff228f6380163988add1a4484a52b625aaa868632dc5ee0fde73b",
    "c24df880cda48519107bdb89cf213b0c93ec5bf61ed229c779129845e3818d81",
)
_PROMPT_SHA256 = (
    "a92c212221dbae0b8c8afef6f4cf53ec247efe3c41ca50ce8a5084fe8b275d8e",
    "b3fc30040112d22c7bc8c699128c3848e56ab3e92fdc0d72dfdcf2fb7bf43db2",
    "acc63af126428db972b9eda5187db2e3daa5c26d4222365a70d50e00adb91738",
    "ba411e918b8a75de8be7853075df7e12efff5964bb28fe14e9b3af9cbedc0cbf",
)
_TARGET_SHA256 = (
    "40917dd09186f4ddf54a8963bb7165fb3c91b3141a569dc82682a766cffb500c",
    "fdf6b6006f176f2c9e8aea829d5360a39718d434cd359e4eb9b29a8832fa8f88",
    "48db0b8e5a58737820ee7839adfc7b4443990c790eb789c35e7da76127957a96",
    "29478edc8329aaf37faef18abc5e6c18457f96f9d796b72dd4cf22bebee8dd74",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


@dataclass(frozen=True)
class CanonicalIdentityCase:
    """One unchanged first-half/second-half Evo2 7B identity case."""

    case_index: int
    sequence: str
    prompt: str
    target: str
    expected_identity_percent: float
    midpoint_fraction: float = 0.5
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_k: int = 1
    seed: int = 42

    @property
    def sequence_length(self) -> int:
        """Return the full frozen sequence length."""
        return len(self.sequence)

    @property
    def prompt_length(self) -> int:
        """Return the frozen first-half prompt length."""
        return len(self.prompt)

    @property
    def target_length(self) -> int:
        """Return the comparison target length."""
        return len(self.target)

    @property
    def minimum_identity_percent(self) -> float:
        """Return the original test floor: 90 percent of its expected identity."""
        return 0.90 * self.expected_identity_percent

    @property
    def minimum_matches(self) -> int:
        """Return the exact integer match floor over the fixed 500-base target."""
        return math.ceil(self.minimum_identity_percent * 5)

    @property
    def sequence_sha256(self) -> str:
        """Return the full frozen sequence digest."""
        return _sha256_text(self.sequence)

    @property
    def prompt_sha256(self) -> str:
        """Return the frozen prompt digest."""
        return _sha256_text(self.prompt)

    @property
    def target_sha256(self) -> str:
        """Return the frozen target digest."""
        return _sha256_text(self.target)


@dataclass(frozen=True)
class CommonPrefixIdentityCase:
    """One frozen 2048-prefix, 500-target differential identity case."""

    case_index: int
    sequence: str
    prompt: str
    target: str
    max_new_tokens: int = 500
    temperature: float = 1.0
    top_k: int = 1
    seed: int = 42

    @property
    def prompt_length(self) -> int:
        """Return the common prompt length."""
        return len(self.prompt)

    @property
    def target_length(self) -> int:
        """Return the common comparison target length."""
        return len(self.target)

    @property
    def prompt_sha256(self) -> str:
        """Return the common prompt digest."""
        return _sha256_text(self.prompt)

    @property
    def target_sha256(self) -> str:
        """Return the common target digest."""
        return _sha256_text(self.target)


@dataclass(frozen=True)
class HomogeneousIdentitySchedule:
    """Exact global and per-engine physical shapes for one repeated prompt case."""

    topology: str
    request_count: int
    global_wave_size: int
    global_request_shapes: tuple[int, ...]
    engine_request_shapes: tuple[tuple[int, ...], ...]
    semantic_padding: bool = False
    mixed_case_batching: bool = False


@dataclass(frozen=True)
class MixedIdentitySchedule:
    """Exact physical shapes and case ownership for one interleaved four-case call."""

    topology: str
    request_count: int
    global_wave_size: int
    global_request_shapes: tuple[int, ...]
    engine_request_shapes: tuple[tuple[int, ...], ...]
    request_ids: tuple[str, ...]
    case_order: tuple[int, ...]
    engine_case_counts: tuple[tuple[tuple[int, int, int, int], ...], ...]
    semantic_padding: bool = False
    mixed_case_batching: bool = True


def _validate_engine_physical_shape(
    *,
    engine: dict[str, Any],
    observations: Sequence[dict[str, Any]],
    wave_phase: str,
    global_shape: int,
    engine_shape: int,
    require_exact_decode_dimension: bool,
) -> None:
    full_decode = engine.get("full_decode_proof")
    scheduler = engine.get("scheduler_capacity_proof")
    if not isinstance(full_decode, dict) or not isinstance(scheduler, dict):
        raise AssertionError("identity physical proof is missing decode or scheduler evidence")
    if (
        full_decode.get("batch_size") != engine_shape
        or full_decode.get("max_new_tokens") != 500
        or full_decode.get("maximum_full_batch") != engine_shape
        or full_decode.get("passed") is not True
    ):
        raise AssertionError("identity FULL decode did not prove the required physical request shape")
    if (
        scheduler.get("global_wave_size") != global_shape
        or scheduler.get("engine_request_count") != engine_shape
        or scheduler.get("maximum_running_requests") != engine_shape
        or scheduler.get("batch_fit_without_preemption") is not True
    ):
        raise AssertionError("identity scheduler did not prove the required physical request shape")
    full_observations = [
        observation
        for observation in observations
        if isinstance(observation, dict)
        and observation.get("phase") == wave_phase
        and str(observation.get("runtime_mode", "")).endswith("FULL")
    ]
    if not full_observations or any(
        observation.get("num_unpadded_tokens") != engine_shape
        or observation.get("num_padded_tokens") != engine_shape
        or observation.get("num_paddings") != 0
        for observation in full_observations
    ):
        raise AssertionError("identity CUDA graph replay did not use the required physical request shape")
    if require_exact_decode_dimension:
        if len(full_observations) != 499:
            raise AssertionError("mixed identity requires exactly 499 FULL decode graph replays")
        for observation in full_observations:
            dimensions = observation.get("request_dimensions")
            if (
                not isinstance(dimensions, dict)
                or dimensions.get("schema_version") != 1
                or dimensions.get("source") != "iteration-stats-bound-to-cudagraph-dispatch"
                or dimensions.get("prefill_req_count") != 0
                or dimensions.get("decode_req_count") != engine_shape
                or dimensions.get("token_count") != engine_shape
                or dimensions.get("prompt_token_count") != 0
                or dimensions.get("first_token_event_count") != 0
            ):
                raise AssertionError("mixed identity CUDA graph replay used the wrong singleton decode dimension")


def _validate_identity_phase_evidence(
    phase: dict[str, Any],
    *,
    schedule: HomogeneousIdentitySchedule | MixedIdentitySchedule,
    expected_execution_coordinates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Require one phase to execute the exact admitted global and engine shapes."""
    phase_name = phase.get("phase")
    if not isinstance(phase_name, str) or not phase_name:
        raise AssertionError("identity phase name is missing")
    sample = phase.get("sample")
    if not isinstance(sample, dict):
        raise AssertionError("identity phase sample is missing")
    if (
        sample.get("request_count") != schedule.request_count
        or sample.get("generated_tokens") != schedule.request_count * 500
        or sample.get("output_lengths") != [500] * schedule.request_count
    ):
        raise AssertionError("identity phase did not retain exact 500-token outputs for every request")

    direct_waves = phase.get("wave_proofs")
    nemo_waves = phase.get("waves")
    if direct_waves is not None and nemo_waves is not None and direct_waves != nemo_waves:
        raise AssertionError("identity phase retained conflicting direct and NeMo wave evidence")
    waves = direct_waves if direct_waves is not None else nemo_waves
    if not isinstance(waves, list) or len(waves) != len(schedule.global_request_shapes):
        raise AssertionError("identity phase physical wave count drifted")
    direct_observations = phase.get("cudagraph_observations_retained", [])
    if schedule.topology == "tp2" and not isinstance(direct_observations, list):
        raise AssertionError("identity TP2 phase is missing raw CUDA graph observations")

    start = 0
    for wave_index, (wave, global_shape, engine_shapes) in enumerate(
        zip(waves, schedule.global_request_shapes, schedule.engine_request_shapes, strict=True)
    ):
        stop = start + global_shape
        if not isinstance(wave, dict) or any(
            wave.get(key) != expected
            for key, expected in (
                ("wave_index", wave_index),
                ("start", start),
                ("stop", stop),
                ("request_count", global_shape),
            )
        ):
            raise AssertionError("identity global physical wave boundaries drifted")
        wave_phase = f"{phase_name}.wave-{wave_index:03d}"
        if schedule.topology == "tp2":
            _validate_engine_physical_shape(
                engine=wave,
                observations=direct_observations,
                wave_phase=wave_phase,
                global_shape=global_shape,
                engine_shape=engine_shapes[0],
                require_exact_decode_dimension=schedule.mixed_case_batching,
            )
        else:
            engines = wave.get("engines")
            if not isinstance(engines, list) or len(engines) != len(engine_shapes):
                raise AssertionError("identity DP2 physical engine count drifted")
            for dp_rank, (engine, engine_shape) in enumerate(zip(engines, engine_shapes, strict=True)):
                if (
                    not isinstance(engine, dict)
                    or engine.get("dp_rank") != dp_rank
                    or engine.get("request_count") != engine_shape
                ):
                    raise AssertionError("identity DP2 shard ownership or physical shape drifted")
                observations = engine.get("cudagraph_observations")
                if not isinstance(observations, list):
                    raise AssertionError("identity DP2 engine is missing raw CUDA graph observations")
                _validate_engine_physical_shape(
                    engine=engine,
                    observations=observations,
                    wave_phase=wave_phase,
                    global_shape=global_shape,
                    engine_shape=engine_shape,
                    require_exact_decode_dimension=schedule.mixed_case_batching,
                )
            inactive_engines = wave.get("inactive_engines")
            expected_inactive_ranks = tuple(range(len(engine_shapes), 2))
            if not isinstance(inactive_engines, list) or len(inactive_engines) != len(expected_inactive_ranks):
                raise AssertionError("identity DP2 inactive replica count drifted")
            for dp_rank, engine in zip(expected_inactive_ranks, inactive_engines, strict=True):
                if (
                    not isinstance(engine, dict)
                    or engine.get("dp_rank") != dp_rank
                    or engine.get("request_count") != 0
                    or engine.get("inactive") is not True
                    or engine.get("phase") != wave_phase
                    or engine.get("cudagraph_observations") != []
                    or engine.get("scheduler_observations") != []
                ):
                    raise AssertionError("identity DP2 inactive replica executed work or lost ownership evidence")
        if schedule.mixed_case_batching:
            if expected_execution_coordinates is None or len(expected_execution_coordinates) != schedule.request_count:
                raise AssertionError("mixed identity physical proof lacks caller-owned execution coordinates")
            executions = phase.get("request_executions")
            if not isinstance(executions, list) or len(executions) != schedule.request_count:
                raise AssertionError("mixed identity phase is missing caller-bound request execution rows")
            wave_executions = executions[start:stop]
            if len(wave_executions) != global_shape:
                raise AssertionError("mixed identity execution rows do not cover the physical wave")
            case_counts_by_rank = []
            case_by_request_id = dict(zip(schedule.request_ids, schedule.case_order, strict=True))
            for dp_rank, engine_shape in enumerate(engine_shapes):
                expected_positions = range(sum(engine_shapes[:dp_rank]), sum(engine_shapes[: dp_rank + 1]))
                actual_cases = []
                for position in expected_positions:
                    row = wave_executions[position]
                    expected_row = expected_execution_coordinates[start + position]
                    required_fields = (
                        "request_id",
                        "global_request_index",
                        "generation_round",
                        "dp_rank",
                        "call_index",
                        "seed",
                    )
                    if not isinstance(row, dict) or type(expected_row) is not dict:
                        raise AssertionError("mixed identity execution rows are malformed")
                    if any(
                        type(row.get(field)) is not type(expected_row.get(field))
                        or row.get(field) != expected_row.get(field)
                        for field in required_fields
                    ):
                        raise AssertionError("mixed identity execution row drifted from caller admission")
                    if row.get("dp_rank") != dp_rank:
                        raise AssertionError("mixed identity DP case ownership changed physical rank")
                    request_id = row.get("request_id")
                    if request_id not in case_by_request_id:
                        raise AssertionError("mixed identity execution row contains a foreign semantic request")
                    actual_cases.append(case_by_request_id[request_id])
                if len(actual_cases) != engine_shape:
                    raise AssertionError("mixed identity physical rank request count drifted")
                case_counts_by_rank.append(tuple(actual_cases.count(case_index) for case_index in range(4)))
            if tuple(case_counts_by_rank) != schedule.engine_case_counts[wave_index]:
                raise AssertionError("mixed identity per-rank case counts drifted")
        start = stop

    return {
        "schema_version": 1,
        "topology": schedule.topology,
        "request_count": schedule.request_count,
        "global_request_shapes": list(schedule.global_request_shapes),
        "engine_request_shapes": [list(shapes) for shapes in schedule.engine_request_shapes],
        "semantic_padding": False,
        "mixed_case_batching": schedule.mixed_case_batching,
        "passed": True,
    }


def validate_homogeneous_identity_phase_evidence(
    phase: dict[str, Any],
    *,
    schedule: HomogeneousIdentitySchedule,
) -> dict[str, Any]:
    """Require one phase to execute the exact homogeneous global and engine shapes."""
    if type(schedule) is not HomogeneousIdentitySchedule or schedule.mixed_case_batching is not False:
        raise TypeError("homogeneous identity evidence requires a homogeneous schedule")
    return _validate_identity_phase_evidence(phase, schedule=schedule)


def validate_mixed_identity_phase_evidence(
    phase: dict[str, Any],
    *,
    schedule: MixedIdentitySchedule,
    expected_execution_coordinates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Require one mixed phase to execute its one admitted scheduler/CUDA-graph shape."""
    if type(schedule) is not MixedIdentitySchedule or schedule.mixed_case_batching is not True:
        raise TypeError("mixed identity evidence requires a mixed schedule")
    return _validate_identity_phase_evidence(
        phase,
        schedule=schedule,
        expected_execution_coordinates=expected_execution_coordinates,
    )


def _validate_canonical_identity_output_rows(
    artifact: dict[str, Any],
    *,
    cases_by_request: Sequence[CanonicalIdentityCase],
    expected_request_ids: Sequence[str],
    expected_prompt_token_ids: Sequence[Sequence[int]] | None,
    expected_execution_coordinates: Sequence[Mapping[str, Any]] | None,
    decode_output_token_ids: Callable[[Sequence[int]], str],
) -> tuple[list[dict[str, Any]], int, str]:
    """Reopen and score canonical rows against caller-owned per-request cases."""
    path = Path(str(artifact.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"canonical identity output artifact is missing: {path}")
    try:
        snapshot = read_jsonl_snapshot(path, label="canonical identity output", compression="gzip")
    except ArtifactSnapshotError as error:
        raise AssertionError("canonical identity output could not be decoded") from error
    observed_digest = snapshot.sha256
    if artifact.get("sha256") != observed_digest:
        raise AssertionError("canonical identity output artifact SHA256 is invalid")
    if artifact.get("decoded_output_bytes_retained") is not True:
        raise AssertionError("canonical identity output artifact did not retain raw output bytes")

    rows = list(snapshot.values)
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AssertionError(f"identity output row {line_number} is not an object")
    expected_ids = tuple(expected_request_ids)
    if tuple(row.get("request_id") for row in rows) != expected_ids:
        raise AssertionError("canonical identity output requests are missing, duplicated, or reordered")
    expected_cases = tuple(cases_by_request)
    expected_prompts = None if expected_prompt_token_ids is None else tuple(tuple(row) for row in expected_prompt_token_ids)
    expected_coordinates = None if expected_execution_coordinates is None else tuple(expected_execution_coordinates)
    if (
        artifact.get("request_count") != len(rows)
        or len(rows) != len(expected_ids)
        or len(rows) != len(expected_cases)
        or (expected_prompts is not None and len(rows) != len(expected_prompts))
        or (expected_coordinates is not None and len(rows) != len(expected_coordinates))
    ):
        raise AssertionError("canonical identity output request count drifted")

    retained = []
    total_bytes = 0
    allowed_token_ids = {ord("A"), ord("C"), ord("G"), ord("T")}
    for row_index, (row, case) in enumerate(zip(rows, expected_cases, strict=True)):
        if type(case) is not CanonicalIdentityCase:
            raise TypeError("canonical identity cases must be exact CanonicalIdentityCase values")
        request_id = str(row["request_id"])
        token_ids = row.get("output_token_ids")
        logprobs = row.get("chosen_token_logprobs")
        if (
            type(token_ids) is not list
            or len(token_ids) != 500
            or any(type(token_id) is not int for token_id in token_ids)
            or type(logprobs) is not list
            or len(logprobs) != 500
            or any(
                type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value)
                for value in logprobs
            )
            or row.get("requested_new_tokens") != 500
            or row.get("observed_new_tokens") != 500
            or row.get("finish_reason") != "length"
            or row.get("stop_reason") is not None
            or row.get("stopped_on_eos") is not False
        ):
            raise AssertionError(f"request {request_id} lacks an exact finite 500-token completion")
        if any(token_id not in allowed_token_ids for token_id in token_ids):
            raise AssertionError(f"request {request_id} canonical identity output must contain exactly 500 ACGT bases")
        if expected_prompts is not None:
            prompt_token_ids = row.get("prompt_token_ids")
            if type(prompt_token_ids) is not list or tuple(prompt_token_ids) != expected_prompts[row_index]:
                raise AssertionError(f"request {request_id} prompt token IDs drifted from caller admission")
        if expected_coordinates is not None:
            expected_coordinate = expected_coordinates[row_index]
            required_fields = (
                "request_id",
                "global_request_index",
                "generation_round",
                "dp_rank",
                "call_index",
                "seed",
            )
            if type(expected_coordinate) is not dict or any(field not in expected_coordinate for field in required_fields):
                raise TypeError("expected execution coordinates must be exact dictionaries with all required fields")
            for field in required_fields:
                expected_value = expected_coordinate[field]
                observed_value = row.get(field)
                if type(observed_value) is not type(expected_value) or observed_value != expected_value:
                    raise AssertionError(f"request {request_id} execution coordinate {field} drifted")
        try:
            raw_bytes = base64.b64decode(row.get("output_text_utf8_base64", ""), validate=True)
        except (binascii.Error, ValueError) as error:
            raise AssertionError(f"request {request_id} raw output bytes are malformed") from error
        decoded = decode_output_token_ids(tuple(token_ids))
        if not isinstance(decoded, str) or decoded.encode("utf-8") != raw_bytes:
            raise AssertionError(f"request {request_id} raw bytes do not match retained output token IDs")
        if bytes(token_ids) != raw_bytes:
            raise AssertionError(f"request {request_id} token IDs do not match nucleotide bytes")
        if (
            len(raw_bytes) != 500
            or row.get("output_text_utf8_bytes") != len(raw_bytes)
            or row.get("output_text_sha256") != _sha256_bytes(raw_bytes)
        ):
            raise AssertionError(f"request {request_id} did not retain exactly 500 verified output bytes")
        try:
            output_text = raw_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise AssertionError(f"request {request_id} output is not ASCII nucleotide text") from error
        if any(base not in "ACGT" for base in output_text):
            raise AssertionError(f"request {request_id} canonical identity output must contain exactly 500 ACGT bases")
        matches = sum(observed == expected for observed, expected in zip(output_text, case.target, strict=True))
        identity_percent = 100.0 * matches / 500
        if matches < case.minimum_matches:
            raise AssertionError(
                f"request {request_id} case {case.case_index} identity {matches}/500 is below "
                f"{case.minimum_matches}/500"
            )
        retained.append(
            {
                "request_id": request_id,
                "case_index": case.case_index,
                "output_text_sha256": _sha256_bytes(raw_bytes),
                "output_text_utf8_bytes": len(raw_bytes),
                "matches": matches,
                "minimum_matches": case.minimum_matches,
                "identity_percent": identity_percent,
                "minimum_identity_percent": case.minimum_identity_percent,
                "passed": True,
            }
        )
        total_bytes += len(raw_bytes)
    if artifact.get("decoded_output_byte_count") != total_bytes:
        raise AssertionError("canonical identity decoded-byte aggregate is inconsistent")
    return retained, total_bytes, observed_digest


def validate_canonical_identity_output_artifact(
    artifact: dict[str, Any],
    *,
    case: CanonicalIdentityCase,
    expected_request_ids: Sequence[str],
    decode_output_token_ids: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    """Recompute every homogeneous canonical identity from retained exact nucleotide bytes."""
    expected_ids = tuple(expected_request_ids)
    retained, _total_bytes, observed_digest = _validate_canonical_identity_output_rows(
        artifact,
        cases_by_request=(case,) * len(expected_ids),
        expected_request_ids=expected_ids,
        expected_prompt_token_ids=None,
        expected_execution_coordinates=None,
        decode_output_token_ids=decode_output_token_ids,
    )
    return {
        "schema_version": 1,
        "case_index": case.case_index,
        "request_count": len(retained),
        "expected_identity_percent": case.expected_identity_percent,
        "minimum_identity_percent": case.minimum_identity_percent,
        "minimum_observed_identity_percent": min(item["identity_percent"] for item in retained),
        "raw_output_bytes_retained": True,
        "full_output_artifact_sha256": observed_digest,
        "requests": retained,
        "passed": True,
    }


def validate_mixed_canonical_identity_output_artifact(
    artifact: dict[str, Any],
    *,
    cases_by_request: Sequence[CanonicalIdentityCase],
    expected_request_ids: Sequence[str],
    expected_prompt_token_ids: Sequence[Sequence[int]],
    expected_execution_coordinates: Sequence[Mapping[str, Any]],
    decode_output_token_ids: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    """Score every interleaved mixed-case occurrence against its caller-owned target."""
    expected_cases = tuple(cases_by_request)
    retained, _total_bytes, observed_digest = _validate_canonical_identity_output_rows(
        artifact,
        cases_by_request=expected_cases,
        expected_request_ids=expected_request_ids,
        expected_prompt_token_ids=expected_prompt_token_ids,
        expected_execution_coordinates=expected_execution_coordinates,
        decode_output_token_ids=decode_output_token_ids,
    )
    case_counts = [sum(case.case_index == case_index for case in expected_cases) for case_index in range(4)]
    if any(count <= 0 for count in case_counts):
        raise AssertionError("mixed canonical output is missing one or more frozen cases")
    return {
        "schema_version": 1,
        "protocol": "mixed-four-case",
        "request_count": len(retained),
        "case_request_counts": case_counts,
        "minimum_observed_matches": min(item["matches"] for item in retained),
        "minimum_observed_identity_percent": min(item["identity_percent"] for item in retained),
        "raw_output_bytes_retained": True,
        "full_output_artifact_sha256": observed_digest,
        "requests": retained,
        "passed": True,
    }


def _validate_common_prefix_output_artifact(
    artifact: dict[str, Any],
    *,
    expected_request_ids: Sequence[str] | None,
    decode_output_token_ids: Callable[[Sequence[int]], str],
) -> tuple[list[dict[str, Any]], str]:
    path = Path(str(artifact.get("path", ""))).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"common-prefix identity output artifact is missing: {path}")
    try:
        snapshot = read_jsonl_snapshot(path, label="common-prefix identity output", compression="gzip")
    except ArtifactSnapshotError as error:
        raise AssertionError("common-prefix identity output could not be decoded") from error
    observed_digest = snapshot.sha256
    if artifact.get("sha256") != observed_digest:
        raise AssertionError("common-prefix identity output artifact SHA256 is invalid")
    if artifact.get("decoded_output_bytes_retained") is not True:
        raise AssertionError("common-prefix identity output artifact did not retain raw output bytes")

    rows = list(snapshot.values)
    for line_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise AssertionError(f"common-prefix output row {line_number} is not an object")

    if expected_request_ids is not None:
        expected_ids = tuple(expected_request_ids)
        if tuple(row.get("request_id") for row in rows) != expected_ids:
            raise AssertionError("common-prefix output requests are missing, duplicated, or reordered")
    if artifact.get("request_count") != len(rows):
        raise AssertionError("common-prefix output request count drifted")

    total_bytes = 0
    retained = []
    for row in rows:
        request_id = str(row.get("request_id", ""))
        prompt_token_ids = row.get("prompt_token_ids")
        token_ids = row.get("output_token_ids")
        logprobs = row.get("chosen_token_logprobs")
        if (
            not request_id
            or not isinstance(prompt_token_ids, list)
            or not prompt_token_ids
            or any(
                not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
                for token_id in prompt_token_ids
            )
            or not isinstance(token_ids, list)
            or len(token_ids) != 500
            or any(
                not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0 for token_id in token_ids
            )
            or not isinstance(logprobs, list)
            or len(logprobs) != 500
            or any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in logprobs)
            or row.get("requested_new_tokens") != 500
            or row.get("observed_new_tokens") != 500
            or row.get("finish_reason") != "length"
            or row.get("stop_reason") is not None
            or row.get("stopped_on_eos") is not False
        ):
            raise AssertionError(f"request {request_id or '<missing>'} lacks an exact finite 500-token completion")
        try:
            raw_bytes = base64.b64decode(row.get("output_text_utf8_base64", ""), validate=True)
        except (binascii.Error, ValueError) as error:
            raise AssertionError(f"request {request_id} raw output bytes are malformed") from error
        decoded = decode_output_token_ids(tuple(token_ids))
        if not isinstance(decoded, str) or decoded.encode("utf-8") != raw_bytes:
            raise AssertionError(f"request {request_id} raw bytes do not match retained output token IDs")
        if (
            len(raw_bytes) != 500
            or row.get("output_text_utf8_bytes") != len(raw_bytes)
            or row.get("output_text_sha256") != _sha256_bytes(raw_bytes)
        ):
            raise AssertionError(f"request {request_id} did not retain exactly 500 verified output bytes")
        try:
            output_text = raw_bytes.decode("ascii")
        except UnicodeDecodeError as error:
            raise AssertionError(f"request {request_id} output is not ASCII nucleotide text") from error
        if set(output_text) - set("ACGTN"):
            raise AssertionError(f"request {request_id} output contains invalid nucleotide characters")
        longest_run = 1
        current_run = 1
        for previous, current in pairwise(output_text):
            current_run = current_run + 1 if current == previous else 1
            longest_run = max(longest_run, current_run)
        if longest_run > 20:
            raise AssertionError(f"request {request_id} output has a homopolymer run longer than 20 bases")
        retained.append(
            {
                "request_id": request_id,
                "prompt_token_ids": tuple(prompt_token_ids),
                "output_text": output_text,
                "output_text_sha256": _sha256_bytes(raw_bytes),
                "output_text_utf8_bytes": len(raw_bytes),
                "maximum_homopolymer_run": longest_run,
            }
        )
        total_bytes += len(raw_bytes)
    if artifact.get("decoded_output_byte_count") != total_bytes:
        raise AssertionError("common-prefix decoded-byte aggregate is inconsistent")
    return retained, observed_digest


def validate_common_prefix_identity_output_artifacts(
    reference_artifact: dict[str, Any],
    candidate_artifact: dict[str, Any],
    *,
    case: CommonPrefixIdentityCase,
    expected_candidate_request_ids: Sequence[str],
    decode_output_token_ids: Callable[[Sequence[int]], str],
) -> dict[str, Any]:
    """Apply the unchanged common-2048 serial-vs-batched identity gate to every request."""
    reference_rows, reference_digest = _validate_common_prefix_output_artifact(
        reference_artifact,
        expected_request_ids=None,
        decode_output_token_ids=decode_output_token_ids,
    )
    if len(reference_rows) != 1:
        raise AssertionError("common-prefix serial reference must contain exactly one request")
    candidate_rows, candidate_digest = _validate_common_prefix_output_artifact(
        candidate_artifact,
        expected_request_ids=expected_candidate_request_ids,
        decode_output_token_ids=decode_output_token_ids,
    )
    if not candidate_rows:
        raise AssertionError("common-prefix candidate artifact cannot be empty")
    reference_prompt = reference_rows[0].pop("prompt_token_ids")
    if len(reference_prompt) != case.prompt_length:
        raise AssertionError("common-prefix serial reference did not use the exact 2048-token prompt")
    for row in candidate_rows:
        if row.pop("prompt_token_ids") != reference_prompt:
            raise AssertionError("common-prefix candidate prompt differs from the serial reference")

    def target_identity(output_text: str) -> float:
        matches = sum(observed == expected for observed, expected in zip(output_text, case.target, strict=True))
        return 100.0 * matches / case.target_length

    serial_identity = target_identity(reference_rows[0]["output_text"])
    minimum_identity = serial_identity - 5.0
    candidates = []
    for row in candidate_rows:
        identity = target_identity(row.pop("output_text"))
        if identity < minimum_identity:
            raise AssertionError(
                f"request {row['request_id']} identity {identity:.3f}% is below the serial-reference "
                f"bound {minimum_identity:.3f}%"
            )
        candidates.append(
            {
                **row,
                "target_identity_percent": identity,
                "minimum_target_identity_percent": minimum_identity,
                "passed": True,
            }
        )
    reference = dict(reference_rows[0])
    reference.pop("output_text")
    reference["target_identity_percent"] = serial_identity
    return {
        "schema_version": 1,
        "case_index": case.case_index,
        "serial_reference_request": reference,
        "serial_target_identity_percent": serial_identity,
        "candidate_request_count": len(candidates),
        "minimum_candidate_target_identity_percent": min(
            candidate["target_identity_percent"] for candidate in candidates
        ),
        "allowed_identity_drop_points": 5.0,
        "reference_output_artifact_sha256": reference_digest,
        "candidate_output_artifact_sha256": candidate_digest,
        "raw_output_bytes_retained": True,
        "candidate_requests": candidates,
        "passed": True,
    }


def load_canonical_7b_identity_cases(path: str | Path) -> tuple[CanonicalIdentityCase, ...]:
    """Load and validate the committed four-case Evo2 7B identity protocol."""
    source = Path(path).expanduser().resolve()
    snapshot = read_byte_snapshot(source, label="canonical prompts CSV")
    observed_sha256 = snapshot.sha256
    if observed_sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise ValueError(
            f"canonical prompts.csv SHA256 mismatch: expected {CANONICAL_7B_PROMPTS_SHA256}, "
            f"observed {observed_sha256}"
        )

    with io.StringIO(snapshot.payload.decode("utf-8"), newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4 or any("Sequence" not in row for row in rows):
        raise ValueError("canonical prompts.csv must contain exactly four Sequence rows")

    cases = []
    for case_index, (row, expected_identity) in enumerate(zip(rows, _EXPECTED_IDENTITIES, strict=True)):
        sequence = row["Sequence"]
        midpoint = int(0.5 * len(sequence))
        prompt = sequence[:midpoint]
        target = sequence[midpoint : midpoint + 500]
        case = CanonicalIdentityCase(
            case_index=case_index,
            sequence=sequence,
            prompt=prompt,
            target=target,
            expected_identity_percent=expected_identity,
        )
        expected = (
            _SEQUENCE_LENGTHS[case_index],
            _PROMPT_LENGTHS[case_index],
            500,
            _SEQUENCE_SHA256[case_index],
            _PROMPT_SHA256[case_index],
            _TARGET_SHA256[case_index],
        )
        observed = (
            case.sequence_length,
            case.prompt_length,
            case.target_length,
            case.sequence_sha256,
            case.prompt_sha256,
            case.target_sha256,
        )
        if observed != expected:
            raise ValueError(f"canonical identity case {case_index} no longer matches the frozen protocol")
        cases.append(case)
    return tuple(cases)


def load_common_prefix_identity_cases(path: str | Path) -> tuple[CommonPrefixIdentityCase, ...]:
    """Derive the unchanged common-2048 protocol from the frozen four-case source."""
    canonical_cases = load_canonical_7b_identity_cases(path)
    cases = tuple(
        CommonPrefixIdentityCase(
            case_index=case.case_index,
            sequence=case.sequence,
            prompt=case.sequence[:2048],
            target=case.sequence[2048:2548],
        )
        for case in canonical_cases
    )
    if any(case.prompt_length != 2048 or case.target_length != 500 for case in cases):
        raise ValueError("common-prefix identity cases must retain exact 2048/500 lengths")
    return cases


def build_common_prefix_identity_manifest(
    base: WorkloadManifest,
    *,
    case: CommonPrefixIdentityCase,
    prompts_csv: str | Path,
    tokenizer: SnapshotBoundTokenizer,
    request_count: int,
    request_id_prefix: str,
) -> WorkloadManifest:
    """Replace a base workload with one repeated 2048-token identity prompt."""
    if base.source_checkpoint != CANONICAL_7B_CHECKPOINT:
        raise ValueError(f"common-prefix identity requires source checkpoint {CANONICAL_7B_CHECKPOINT}")
    if request_count <= 0 or not request_id_prefix:
        raise ValueError("common-prefix identity requires a positive count and nonempty request prefix")
    cases = load_common_prefix_identity_cases(prompts_csv)
    if not 0 <= case.case_index < len(cases) or case != cases[case.case_index]:
        raise ValueError("common-prefix identity case does not belong to the frozen prompts.csv")

    if not isinstance(tokenizer, SnapshotBoundTokenizer):
        raise TypeError("common-prefix identity requires a SnapshotBoundTokenizer")
    prompt_token_ids = tokenizer.encode(case.prompt)
    tokenizer.verify_source()
    if len(prompt_token_ids) != 2048 or any(token_id < 0 for token_id in prompt_token_ids):
        raise ValueError("common-prefix identity prompt must tokenize to exactly 2048 nonnegative tokens")
    width = max(4, len(str(request_count - 1)))
    requests = tuple(
        WorkloadRequest(
            request_id=f"{request_id_prefix}-{index:0{width}d}",
            prompt_token_ids=prompt_token_ids,
        )
        for index in range(request_count)
    )
    manifest = replace(
        base,
        name=f"common-2048-7b-identity-case{case.case_index}-n{request_count}",
        requests=requests,
        max_new_tokens=500,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        seed=42,
        ignore_eos=True,
        stop_token_ids=(),
        prompt_source_path=str(Path(prompts_csv).expanduser().resolve()),
        prompt_source_sha256=CANONICAL_7B_PROMPTS_SHA256,
        prompt_tokenizer_path=str(tokenizer.path),
        prompt_tokenizer_sha256=tokenizer.source_sha256,
    )
    validate_common_prefix_identity_manifest(manifest, case=case, request_count=request_count)
    return manifest


def validate_common_prefix_identity_manifest(
    manifest: WorkloadManifest,
    *,
    case: CommonPrefixIdentityCase,
    request_count: int,
) -> None:
    """Require one 2048-token case repeated homogeneously without semantic padding."""
    if manifest.source_checkpoint != CANONICAL_7B_CHECKPOINT:
        raise AssertionError("common-prefix identity manifest uses the wrong source checkpoint")
    if len(manifest.requests) != request_count:
        raise AssertionError("common-prefix identity request count drifted")
    prompts = {request.prompt_token_ids for request in manifest.requests}
    if len(prompts) != 1 or len(next(iter(prompts))) != case.prompt_length:
        raise AssertionError("common-prefix identity manifest is not one homogeneous 2048-token prompt")
    if (
        manifest.max_new_tokens != 500
        or manifest.temperature != 1.0
        or manifest.top_p != 1.0
        or manifest.top_k != 1
        or manifest.seed != 42
        or manifest.ignore_eos is not True
        or manifest.stop_token_ids
    ):
        raise AssertionError("common-prefix identity sampling or exact-length settings drifted")
    if manifest.prompt_source_sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise AssertionError("common-prefix identity prompt source provenance drifted")


def build_canonical_identity_manifest(
    base: WorkloadManifest,
    *,
    case: CanonicalIdentityCase,
    prompts_csv: str | Path,
    tokenizer: SnapshotBoundTokenizer,
    request_count: int,
    request_id_prefix: str,
) -> WorkloadManifest:
    """Replace a base workload with repeated copies of one canonical prompt."""
    if base.source_checkpoint != CANONICAL_7B_CHECKPOINT:
        raise ValueError(f"canonical identity requires source checkpoint {CANONICAL_7B_CHECKPOINT}")
    if request_count <= 0:
        raise ValueError("canonical identity request_count must be positive")
    if not request_id_prefix:
        raise ValueError("canonical identity request_id_prefix cannot be empty")
    cases = load_canonical_7b_identity_cases(prompts_csv)
    if not 0 <= case.case_index < len(cases) or case != cases[case.case_index]:
        raise ValueError("canonical identity case does not belong to the frozen prompts.csv")

    if not isinstance(tokenizer, SnapshotBoundTokenizer):
        raise TypeError("canonical identity requires a SnapshotBoundTokenizer")
    prompt_token_ids = tokenizer.encode(case.prompt)
    tokenizer.verify_source()
    if len(prompt_token_ids) != case.prompt_length:
        raise ValueError(
            f"canonical case {case.case_index} prompt must tokenize to {case.prompt_length} tokens, "
            f"observed {len(prompt_token_ids)}"
        )
    if any(token_id < 0 for token_id in prompt_token_ids):
        raise ValueError("canonical identity prompt produced a negative token ID")

    width = max(4, len(str(request_count - 1)))
    requests = tuple(
        WorkloadRequest(
            request_id=f"{request_id_prefix}-{index:0{width}d}",
            prompt_token_ids=prompt_token_ids,
        )
        for index in range(request_count)
    )
    manifest = replace(
        base,
        name=f"canonical-7b-identity-case{case.case_index}-n{request_count}",
        requests=requests,
        max_new_tokens=case.max_new_tokens,
        temperature=case.temperature,
        top_p=1.0,
        top_k=case.top_k,
        seed=case.seed,
        ignore_eos=True,
        stop_token_ids=(),
        prompt_source_path=str(Path(prompts_csv).expanduser().resolve()),
        prompt_source_sha256=CANONICAL_7B_PROMPTS_SHA256,
        prompt_tokenizer_path=str(tokenizer.path),
        prompt_tokenizer_sha256=tokenizer.source_sha256,
    )
    validate_canonical_identity_manifest(manifest, case=case, request_count=request_count)
    return manifest


def validate_canonical_identity_manifest(
    manifest: WorkloadManifest,
    *,
    case: CanonicalIdentityCase,
    request_count: int,
) -> None:
    """Require one unchanged canonical case repeated without mixed prompts or padding."""
    if manifest.source_checkpoint != CANONICAL_7B_CHECKPOINT:
        raise AssertionError("canonical identity manifest uses the wrong source checkpoint")
    if len(manifest.requests) != request_count:
        raise AssertionError("canonical identity manifest request count drifted")
    prompts = {request.prompt_token_ids for request in manifest.requests}
    if len(prompts) != 1:
        raise AssertionError("canonical identity manifest must be physically homogeneous")
    prompt_token_ids = next(iter(prompts))
    if len(prompt_token_ids) != case.prompt_length:
        raise AssertionError("canonical identity prompt token length drifted")
    if (
        manifest.max_new_tokens != 500
        or manifest.temperature != 1.0
        or manifest.top_p != 1.0
        or manifest.top_k != 1
        or manifest.seed != 42
        or manifest.ignore_eos is not True
        or manifest.stop_token_ids
    ):
        raise AssertionError("canonical identity sampling contract drifted")
    if manifest.prompt_source_sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise AssertionError("canonical identity source provenance drifted")
    if not manifest.name.startswith(f"canonical-7b-identity-case{case.case_index}-"):
        raise AssertionError("canonical identity case index is not bound to the manifest")


def _require_frozen_mixed_cases(
    cases: Sequence[CanonicalIdentityCase],
    *,
    prompts_csv: str | Path,
) -> tuple[CanonicalIdentityCase, ...]:
    frozen = load_canonical_7b_identity_cases(prompts_csv)
    observed = tuple(cases)
    if observed != frozen or tuple(case.case_index for case in observed) != (0, 1, 2, 3):
        raise ValueError("mixed canonical identity requires all four frozen cases in canonical order")
    return observed


def build_mixed_canonical_identity_manifest(
    base: WorkloadManifest,
    *,
    cases: Sequence[CanonicalIdentityCase],
    prompts_csv: str | Path,
    tokenizer: SnapshotBoundTokenizer,
    request_count: int,
    request_id_prefix: str,
) -> WorkloadManifest:
    """Build one interleaved B4 or B96 workload from all four original midpoint prompts."""
    if base.source_checkpoint != CANONICAL_7B_CHECKPOINT:
        raise ValueError(f"mixed canonical identity requires source checkpoint {CANONICAL_7B_CHECKPOINT}")
    if type(request_count) is not int or request_count not in {4, 96}:
        raise ValueError("mixed canonical identity request_count must be exactly 4 or 96")
    if type(request_id_prefix) is not str or not request_id_prefix:
        raise ValueError("mixed canonical identity request_id_prefix cannot be empty")
    frozen_cases = _require_frozen_mixed_cases(cases, prompts_csv=prompts_csv)
    if not isinstance(tokenizer, SnapshotBoundTokenizer):
        raise TypeError("mixed canonical identity requires a SnapshotBoundTokenizer")

    prompt_token_ids = tuple(tokenizer.encode(case.prompt) for case in frozen_cases)
    tokenizer.verify_source()
    if tuple(len(tokens) for tokens in prompt_token_ids) != _PROMPT_LENGTHS:
        raise ValueError("mixed canonical prompts were padded, truncated, or tokenized inconsistently")
    if any(type(token_id) is not int or token_id < 0 for tokens in prompt_token_ids for token_id in tokens):
        raise ValueError("mixed canonical identity prompt produced an invalid token ID")

    requests = tuple(
        WorkloadRequest(
            request_id=f"{request_id_prefix}-case{index % 4}-occurrence{index // 4:04d}",
            prompt_token_ids=prompt_token_ids[index % 4],
        )
        for index in range(request_count)
    )
    manifest = replace(
        base,
        name=f"canonical-7b-identity-mixed-n{request_count}",
        requests=requests,
        max_new_tokens=500,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        seed=42,
        ignore_eos=True,
        stop_token_ids=(),
        prompt_source_path=str(Path(prompts_csv).expanduser().resolve()),
        prompt_source_sha256=CANONICAL_7B_PROMPTS_SHA256,
        prompt_tokenizer_path=str(tokenizer.path),
        prompt_tokenizer_sha256=tokenizer.source_sha256,
    )
    validate_mixed_canonical_identity_manifest(manifest, cases=frozen_cases)
    return manifest


def validate_mixed_canonical_identity_manifest(
    manifest: WorkloadManifest,
    *,
    cases: Sequence[CanonicalIdentityCase],
) -> None:
    """Require exact interleaved variable-length prompts with no semantic padding."""
    frozen_cases = tuple(cases)
    if manifest.source_checkpoint != CANONICAL_7B_CHECKPOINT:
        raise AssertionError("mixed canonical identity manifest uses the wrong source checkpoint")
    if len(frozen_cases) != 4 or tuple(case.case_index for case in frozen_cases) != (0, 1, 2, 3):
        raise AssertionError("mixed canonical identity cases are incomplete or reordered")
    if len(manifest.requests) not in {4, 96}:
        raise AssertionError("mixed canonical identity manifest must contain exactly 4 or 96 requests")
    expected_ids = tuple(
        f"{manifest.requests[0].request_id.rsplit('-case', 1)[0]}-case{index % 4}-occurrence{index // 4:04d}"
        for index in range(len(manifest.requests))
    )
    if tuple(request.request_id for request in manifest.requests) != expected_ids:
        raise AssertionError("mixed canonical request case/occurrence identities drifted")
    for index, request in enumerate(manifest.requests):
        case = frozen_cases[index % 4]
        if request.prompt_token_ids != tuple(case.prompt.encode("ascii")):
            raise AssertionError("mixed canonical prompt bytes were padded, truncated, or reordered")
    if (
        manifest.max_new_tokens != 500
        or manifest.temperature != 1.0
        or manifest.top_p != 1.0
        or manifest.top_k != 1
        or manifest.seed != 42
        or manifest.ignore_eos is not True
        or manifest.stop_token_ids
    ):
        raise AssertionError("mixed canonical identity sampling contract drifted")
    if manifest.prompt_source_sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise AssertionError("mixed canonical identity source provenance drifted")
    if manifest.name != f"canonical-7b-identity-mixed-n{len(manifest.requests)}":
        raise AssertionError("mixed canonical identity manifest name drifted")


def build_mixed_identity_schedule(
    *,
    topology: str,
    request_count: int,
    global_wave_size: int,
    request_id_prefix: str = "mixed",
) -> MixedIdentitySchedule:
    """Build one unpadded interleaved B4 or B96 TP2/DP2 physical call."""
    if topology not in {"tp2", "dp2"}:
        raise ValueError("mixed canonical identity topology must be tp2 or dp2")
    if type(request_count) is not int or request_count not in {4, 96}:
        raise ValueError("mixed canonical identity request_count must be exactly 4 or 96")
    if type(global_wave_size) is not int or global_wave_size != request_count:
        raise ValueError("mixed canonical identity requires one global physical call")
    if type(request_id_prefix) is not str or not request_id_prefix:
        raise ValueError("mixed canonical identity request_id_prefix cannot be empty")
    replica_count = 1 if topology == "tp2" else 2
    waves = build_request_waves(
        request_count=request_count,
        global_batch_size=global_wave_size,
        replica_count=replica_count,
    )
    if len(waves) != 1:
        raise AssertionError("mixed canonical identity schedule split its admitted physical call")
    case_order = tuple(range(4)) * (request_count // 4)
    request_ids = tuple(
        f"{request_id_prefix}-case{index % 4}-occurrence{index // 4:04d}" for index in range(request_count)
    )
    engine_case_counts = tuple(
        tuple(
            tuple(case_order[shard.start : shard.stop].count(case_index) for case_index in range(4))
            for shard in wave.shards
        )
        for wave in waves
    )
    return MixedIdentitySchedule(
        topology=topology,
        request_count=request_count,
        global_wave_size=global_wave_size,
        global_request_shapes=tuple(wave.request_count for wave in waves),
        engine_request_shapes=tuple(tuple(shard.request_count for shard in wave.shards) for wave in waves),
        request_ids=request_ids,
        case_order=case_order,
        engine_case_counts=engine_case_counts,
    )


def build_mixed_identity_admission_bundle(
    *,
    topology: str,
    base_seed: int,
    manifest_sha256_by_stage: Mapping[str, str] | None = None,
    request_id_prefix: str = "mixed",
) -> dict[str, Any]:
    """Bind B4 then B96 to advancing caller-owned call, request, and seed coordinates."""
    if topology not in {"tp2", "dp2"}:
        raise ValueError("mixed canonical identity topology must be tp2 or dp2")
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("mixed canonical identity base_seed must be a nonnegative built-in integer")
    if manifest_sha256_by_stage is not None:
        if type(manifest_sha256_by_stage) is not dict or set(manifest_sha256_by_stage) != {"mixed-b4", "mixed-b96"}:
            raise ValueError("mixed canonical manifest authority must contain exact B4 and B96 stage keys")
        if any(
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in manifest_sha256_by_stage.values()
        ):
            raise ValueError("mixed canonical stage manifest SHA256 values are malformed")
    replica_count = 1 if topology == "tp2" else 2
    attempts = []
    global_request_index_start = 0
    for global_call_index, (stage, request_count) in enumerate((("mixed-b4", 4), ("mixed-b96", 96))):
        stage_request_id_prefix = f"{request_id_prefix}-{'b4' if request_count == 4 else 'b96'}"
        schedule = build_mixed_identity_schedule(
            topology=topology,
            request_count=request_count,
            global_wave_size=request_count,
            request_id_prefix=stage_request_id_prefix,
        )
        wave = build_request_waves(
            request_count=request_count,
            global_batch_size=request_count,
            replica_count=replica_count,
        )[0]
        request_seeds = [None] * request_count
        execution_coordinates = [None] * request_count
        for shard in wave.shards:
            stream_seed = base_seed + (global_call_index * replica_count + shard.replica_index) * 1_000_003
            for local_ordinal, request_index in enumerate(range(shard.start, shard.stop)):
                request_seed = stream_seed + local_ordinal
                if request_seed >= 2**31:
                    raise ValueError("mixed canonical identity request seed exceeds the no-wrap domain")
                request_seeds[request_index] = request_seed
                execution_coordinates[request_index] = {
                    "request_id": schedule.request_ids[request_index],
                    "global_request_index": global_request_index_start + request_index,
                    "generation_round": global_call_index,
                    "dp_rank": shard.replica_index,
                    "call_index": global_call_index,
                    "seed": request_seed,
                }
        attempts.append(
            {
                "stage": stage,
                "manifest_sha256": (
                    None if manifest_sha256_by_stage is None else manifest_sha256_by_stage[stage]
                ),
                "global_call_index": global_call_index,
                "global_request_index_start": global_request_index_start,
                "request_count": request_count,
                "case_order": list(schedule.case_order),
                "engine_request_shapes": [list(shapes) for shapes in schedule.engine_request_shapes],
                "engine_case_counts": [
                    [list(case_counts) for case_counts in wave_counts]
                    for wave_counts in schedule.engine_case_counts
                ],
                "request_ids": list(schedule.request_ids),
                "request_seeds": request_seeds,
                "execution_coordinates": execution_coordinates,
            }
        )
        global_request_index_start += request_count
    if set(attempts[0]["request_seeds"]) & set(attempts[1]["request_seeds"]):
        raise AssertionError("mixed canonical B4/B96 request seed streams overlap")
    if set(attempts[0]["request_ids"]) & set(attempts[1]["request_ids"]):
        raise AssertionError("mixed canonical B4/B96 semantic request identities overlap")
    return {
        "schema_version": 1,
        "protocol": "mixed-b4-then-b96",
        "topology": topology,
        "base_seed": base_seed,
        "attempts": attempts,
    }


def build_homogeneous_identity_schedule(
    *,
    topology: str,
    request_count: int,
    global_wave_size: int,
) -> HomogeneousIdentitySchedule:
    """Build exact physical TP2 or TP1/DP2 shapes for one canonical case."""
    if topology not in {"tp2", "dp2"}:
        raise ValueError("canonical identity topology must be tp2 or dp2")
    replica_count = 1 if topology == "tp2" else 2
    waves = build_request_waves(
        request_count=request_count,
        global_batch_size=global_wave_size,
        replica_count=replica_count,
    )
    return HomogeneousIdentitySchedule(
        topology=topology,
        request_count=request_count,
        global_wave_size=global_wave_size,
        global_request_shapes=tuple(wave.request_count for wave in waves),
        engine_request_shapes=tuple(tuple(shard.request_count for shard in wave.shards) for wave in waves),
    )


def build_common_prefix_identity_contract(
    *,
    case: CommonPrefixIdentityCase,
    serial_schedule: HomogeneousIdentitySchedule,
    candidate_schedule: HomogeneousIdentitySchedule,
    prompts_csv: str | Path,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    """Return the immutable serial-vs-batched common-prefix accuracy contract."""
    source = Path(prompts_csv).expanduser().resolve()
    source_snapshot = read_byte_snapshot(source, label="common-prefix prompts CSV")
    if source_snapshot.sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise ValueError("common-prefix identity source SHA256 drifted")
    tokenizer = SnapshotBoundTokenizer.from_path(tokenizer_path)

    def schedule_payload(schedule: HomogeneousIdentitySchedule) -> dict[str, Any]:
        return {
            "topology": schedule.topology,
            "request_count": schedule.request_count,
            "global_wave_size": schedule.global_wave_size,
            "global_request_shapes": list(schedule.global_request_shapes),
            "engine_request_shapes": [list(shapes) for shapes in schedule.engine_request_shapes],
            "semantic_padding": schedule.semantic_padding,
            "mixed_case_batching": schedule.mixed_case_batching,
        }

    if serial_schedule.request_count != 1:
        raise ValueError("common-prefix serial reference schedule must contain exactly one request")
    if serial_schedule.topology != candidate_schedule.topology:
        raise ValueError("common-prefix serial and candidate schedules must use the same topology")
    return {
        "schema_version": 1,
        "protocol": "common-2048-serial-vs-batched",
        "checkpoint": CANONICAL_7B_CHECKPOINT,
        "prompts_csv_path": str(source),
        "prompts_csv_sha256": CANONICAL_7B_PROMPTS_SHA256,
        "tokenizer_path": str(tokenizer.path),
        "tokenizer_sha256": tokenizer.source_sha256,
        "case_index": case.case_index,
        "prompt_length": case.prompt_length,
        "prompt_sha256": case.prompt_sha256,
        "target_length": case.target_length,
        "target_sha256": case.target_sha256,
        "allowed_identity_drop_points": 5.0,
        "sampling": {
            "max_new_tokens": case.max_new_tokens,
            "temperature": case.temperature,
            "top_p": 1.0,
            "top_k": case.top_k,
            "seed": case.seed,
            "ignore_eos": True,
            "stop_token_ids": [],
        },
        "serial_schedule": schedule_payload(serial_schedule),
        "candidate_schedule": schedule_payload(candidate_schedule),
    }


def build_canonical_identity_contract(
    *,
    case: CanonicalIdentityCase,
    schedule: HomogeneousIdentitySchedule,
    prompts_csv: str | Path,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    """Return the exact model-quality contract linked across proof and speed lanes."""
    source = Path(prompts_csv).expanduser().resolve()
    source_snapshot = read_byte_snapshot(source, label="canonical prompts CSV")
    if source_snapshot.sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise ValueError("canonical identity source SHA256 drifted")
    tokenizer = SnapshotBoundTokenizer.from_path(tokenizer_path)
    return {
        "schema_version": 1,
        "checkpoint": CANONICAL_7B_CHECKPOINT,
        "prompts_csv_path": str(source),
        "prompts_csv_sha256": CANONICAL_7B_PROMPTS_SHA256,
        "tokenizer_path": str(tokenizer.path),
        "tokenizer_sha256": tokenizer.source_sha256,
        "case_index": case.case_index,
        "midpoint_fraction": case.midpoint_fraction,
        "sequence_length": case.sequence_length,
        "sequence_sha256": case.sequence_sha256,
        "prompt_length": case.prompt_length,
        "prompt_sha256": case.prompt_sha256,
        "target_length": case.target_length,
        "target_sha256": case.target_sha256,
        "expected_identity_percent": case.expected_identity_percent,
        "minimum_identity_percent": case.minimum_identity_percent,
        "sampling": {
            "max_new_tokens": case.max_new_tokens,
            "temperature": case.temperature,
            "top_p": 1.0,
            "top_k": case.top_k,
            "seed": case.seed,
            "ignore_eos": True,
            "stop_token_ids": [],
        },
        "schedule": {
            "topology": schedule.topology,
            "request_count": schedule.request_count,
            "global_wave_size": schedule.global_wave_size,
            "global_request_shapes": list(schedule.global_request_shapes),
            "engine_request_shapes": [list(shapes) for shapes in schedule.engine_request_shapes],
            "semantic_padding": schedule.semantic_padding,
            "mixed_case_batching": schedule.mixed_case_batching,
        },
    }


def build_mixed_canonical_identity_contract(
    *,
    cases: Sequence[CanonicalIdentityCase],
    schedule: MixedIdentitySchedule,
    stage_manifests: Mapping[str, WorkloadManifest],
    prompts_csv: str | Path,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    """Bind both mixed manifests and their ordered B4-then-B96 execution contract."""
    frozen_cases = _require_frozen_mixed_cases(cases, prompts_csv=prompts_csv)
    if type(schedule) is not MixedIdentitySchedule:
        raise TypeError("mixed canonical identity contract requires an exact mixed schedule")
    if type(stage_manifests) is not dict or set(stage_manifests) != {"mixed-b4", "mixed-b96"}:
        raise ValueError("mixed canonical identity contract requires exact B4 and B96 manifests")
    expected_counts = {"mixed-b4": 4, "mixed-b96": 96}
    manifest_sha256_by_stage = {}
    stage_prefixes = {}
    stages = []
    for stage, request_count in expected_counts.items():
        manifest = stage_manifests[stage]
        if type(manifest) is not WorkloadManifest or len(manifest.requests) != request_count:
            raise ValueError(f"{stage} manifest does not contain its exact request count")
        validate_mixed_canonical_identity_manifest(manifest, cases=frozen_cases)
        stage_prefix = manifest.requests[0].request_id.rsplit("-case", 1)[0]
        stage_prefixes[stage] = stage_prefix
        stage_schedule = build_mixed_identity_schedule(
            topology=schedule.topology,
            request_count=request_count,
            global_wave_size=request_count,
            request_id_prefix=stage_prefix,
        )
        if tuple(request.request_id for request in manifest.requests) != stage_schedule.request_ids:
            raise ValueError(f"{stage} manifest request identities drifted from its schedule")
        manifest_sha256_by_stage[stage] = manifest.sha256
        stages.append(
            {
                "stage": stage,
                "manifest_sha256": manifest.sha256,
                "request_count": request_count,
                "schedule": {
                    "global_request_shapes": list(stage_schedule.global_request_shapes),
                    "engine_request_shapes": [list(shapes) for shapes in stage_schedule.engine_request_shapes],
                    "engine_case_counts": [
                        [list(case_counts) for case_counts in wave_counts]
                        for wave_counts in stage_schedule.engine_case_counts
                    ],
                },
            }
        )

    if not stage_prefixes["mixed-b4"].endswith("-b4") or not stage_prefixes["mixed-b96"].endswith("-b96"):
        raise ValueError("mixed canonical stage manifests require stage-qualified request IDs")
    request_id_prefix = stage_prefixes["mixed-b4"][: -len("-b4")]
    if stage_prefixes["mixed-b96"][: -len("-b96")] != request_id_prefix:
        raise ValueError("mixed canonical B4 and B96 manifests use different request-ID roots")
    current_stage = "mixed-b4" if schedule.request_count == 4 else "mixed-b96"
    if schedule.request_ids != tuple(request.request_id for request in stage_manifests[current_stage].requests):
        raise ValueError("mixed canonical active schedule does not match its stage-qualified manifest")

    source = Path(prompts_csv).expanduser().resolve()
    source_snapshot = read_byte_snapshot(source, label="canonical prompts CSV")
    if source_snapshot.sha256 != CANONICAL_7B_PROMPTS_SHA256:
        raise ValueError("canonical identity source SHA256 drifted")
    tokenizer = SnapshotBoundTokenizer.from_path(tokenizer_path)
    admission_bundle = build_mixed_identity_admission_bundle(
        topology=schedule.topology,
        base_seed=42,
        manifest_sha256_by_stage=manifest_sha256_by_stage,
        request_id_prefix=request_id_prefix,
    )
    return {
        "schema_version": 1,
        "protocol": "mixed-b4-then-b96-single-engine",
        "checkpoint": CANONICAL_7B_CHECKPOINT,
        "prompts_csv_path": str(source),
        "prompts_csv_sha256": CANONICAL_7B_PROMPTS_SHA256,
        "tokenizer_path": str(tokenizer.path),
        "tokenizer_sha256": tokenizer.source_sha256,
        "case_order": list(schedule.case_order),
        "minimum_matches": [case.minimum_matches for case in frozen_cases],
        "sampling": {
            "max_new_tokens": 500,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "seed": 42,
            "ignore_eos": True,
            "stop_token_ids": [],
        },
        "schedule": {
            "topology": schedule.topology,
            "request_count": schedule.request_count,
            "global_wave_size": schedule.global_wave_size,
            "global_request_shapes": list(schedule.global_request_shapes),
            "engine_request_shapes": [list(shapes) for shapes in schedule.engine_request_shapes],
            "engine_case_counts": [
                [list(case_counts) for case_counts in wave_counts]
                for wave_counts in schedule.engine_case_counts
            ],
            "semantic_padding": False,
            "mixed_case_batching": True,
        },
        "stages": stages,
        "admission_bundle": admission_bundle,
        "qualification_requires_same_engine_dual_stage_physical_proof": True,
    }


__all__ = [
    "CANONICAL_7B_CHECKPOINT",
    "CANONICAL_7B_PROMPTS_SHA256",
    "CanonicalIdentityCase",
    "CommonPrefixIdentityCase",
    "HomogeneousIdentitySchedule",
    "MixedIdentitySchedule",
    "build_canonical_identity_contract",
    "build_canonical_identity_manifest",
    "build_common_prefix_identity_contract",
    "build_common_prefix_identity_manifest",
    "build_homogeneous_identity_schedule",
    "build_mixed_canonical_identity_contract",
    "build_mixed_canonical_identity_manifest",
    "build_mixed_identity_admission_bundle",
    "build_mixed_identity_schedule",
    "load_canonical_7b_identity_cases",
    "load_common_prefix_identity_cases",
    "validate_canonical_identity_manifest",
    "validate_canonical_identity_output_artifact",
    "validate_common_prefix_identity_manifest",
    "validate_common_prefix_identity_output_artifacts",
    "validate_homogeneous_identity_phase_evidence",
    "validate_mixed_canonical_identity_manifest",
    "validate_mixed_canonical_identity_output_artifact",
    "validate_mixed_identity_phase_evidence",
]
