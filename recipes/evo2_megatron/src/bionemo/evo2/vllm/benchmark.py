# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Reproducible benchmark schema and runner for Evo2 inference backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, token_id in enumerate(token_ids):
        if index:
            digest.update(b",")
        digest.update(str(int(token_id)).encode())
    digest.update(b"]")
    return digest.hexdigest()


@dataclass(frozen=True)
class WorkloadRequest:
    """One stable pretokenized generation request."""

    request_id: str
    prompt_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate request identity and token IDs."""
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.prompt_token_ids:
            raise ValueError(f"request {self.request_id} has an empty prompt")
        if any(token_id < 0 for token_id in self.prompt_token_ids):
            raise ValueError(f"request {self.request_id} has a negative token ID")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe request record."""
        return {"request_id": self.request_id, "prompt_token_ids": list(self.prompt_token_ids)}


@dataclass(frozen=True)
class RequestShard:
    """One exact contiguous request range assigned to a DP replica."""

    replica_index: int
    start: int
    stop: int

    @property
    def request_count(self) -> int:
        """Return the number of real, unpadded requests in this shard."""
        return self.stop - self.start


@dataclass(frozen=True)
class RequestWave:
    """One global generation wave split into exact DP replica shards."""

    wave_index: int
    start: int
    stop: int
    shards: tuple[RequestShard, ...]

    @property
    def request_count(self) -> int:
        """Return the number of real, unpadded requests in this wave."""
        return self.stop - self.start


def build_request_waves(
    *,
    request_count: int,
    global_batch_size: int,
    replica_count: int,
) -> tuple[RequestWave, ...]:
    """Partition requests into exact global waves and balanced contiguous DP shards."""
    if request_count <= 0 or global_batch_size <= 0 or replica_count <= 0:
        raise ValueError("request, global batch, and replica counts must be positive")
    if replica_count > global_batch_size:
        raise ValueError("replica_count cannot exceed global_batch_size")

    waves = []
    for wave_index, start in enumerate(range(0, request_count, global_batch_size)):
        stop = min(start + global_batch_size, request_count)
        wave_count = stop - start
        active_replicas = min(replica_count, wave_count)
        base_size, remainder = divmod(wave_count, active_replicas)
        shard_start = start
        shards = []
        for replica_index in range(active_replicas):
            shard_stop = shard_start + base_size + (replica_index < remainder)
            shards.append(RequestShard(replica_index, shard_start, shard_stop))
            shard_start = shard_stop
        waves.append(RequestWave(wave_index, start, stop, tuple(shards)))
    return tuple(waves)


@dataclass(frozen=True)
class WorkloadManifest:
    """Backend-neutral immutable generation workload."""

    schema_version: int
    name: str
    source_checkpoint: str
    checkpoint_manifest_sha256: str
    checkpoint_index_sha256: str
    tokenizer_sha256: str
    requests: tuple[WorkloadRequest, ...]
    max_new_tokens: int
    temperature: float
    top_p: float
    top_k: int
    seed: int
    dtype: str
    ignore_eos: bool
    stop_token_ids: tuple[int, ...]
    prompt_source_path: str | None = None
    prompt_source_sha256: str | None = None
    prompt_tokenizer_path: str | None = None
    prompt_tokenizer_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate workload and sampling invariants."""
        if self.schema_version != 1:
            raise ValueError(f"unsupported workload schema_version: {self.schema_version}")
        if not self.name:
            raise ValueError("workload name cannot be empty")
        request_ids = [request.request_id for request in self.requests]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("request IDs must be unique")
        if not self.requests:
            raise ValueError("workload must contain at least one request")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 or positive")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            raise ValueError(f"unsupported dtype: {self.dtype}")
        prompt_provenance = (
            self.prompt_source_path,
            self.prompt_source_sha256,
            self.prompt_tokenizer_path,
            self.prompt_tokenizer_sha256,
        )
        if any(value is not None for value in prompt_provenance) and not all(
            value is not None for value in prompt_provenance
        ):
            raise ValueError("prompt source and tokenizer provenance must be complete")
        for label, digest in (
            ("checkpoint_manifest_sha256", self.checkpoint_manifest_sha256),
            ("checkpoint_index_sha256", self.checkpoint_index_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
            ("prompt_source_sha256", self.prompt_source_sha256),
            ("prompt_tokenizer_sha256", self.prompt_tokenizer_sha256),
        ):
            if digest is None:
                continue
            if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
                raise ValueError(f"{label} must be a lowercase SHA256 digest")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadManifest:
        """Build a manifest from its JSON representation."""
        values = dict(data)
        values["requests"] = tuple(
            WorkloadRequest(
                request_id=request["request_id"],
                prompt_token_ids=tuple(request["prompt_token_ids"]),
            )
            for request in values["requests"]
        )
        values["stop_token_ids"] = tuple(values.get("stop_token_ids", ()))
        return cls(**values)

    @classmethod
    def from_path(cls, path: str | Path) -> WorkloadManifest:
        """Load and validate one JSON manifest."""
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-safe representation."""
        result = {
            "schema_version": self.schema_version,
            "name": self.name,
            "source_checkpoint": self.source_checkpoint,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "checkpoint_index_sha256": self.checkpoint_index_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "requests": [request.to_dict() for request in self.requests],
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "dtype": self.dtype,
            "ignore_eos": self.ignore_eos,
            "stop_token_ids": list(self.stop_token_ids),
        }
        if self.prompt_source_path is not None:
            result.update(
                {
                    "prompt_source_path": self.prompt_source_path,
                    "prompt_source_sha256": self.prompt_source_sha256,
                    "prompt_tokenizer_path": self.prompt_tokenizer_path,
                    "prompt_tokenizer_sha256": self.prompt_tokenizer_sha256,
                }
            )
        return result

    def constructor_kwargs(self) -> dict[str, Any]:
        """Return dataclass constructor values without JSON conversion."""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    @property
    def sha256(self) -> str:
        """Return a stable digest over canonical compact JSON."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    @property
    def max_total_tokens(self) -> int:
        """Return the longest prompt plus required generated length."""
        return max(len(request.prompt_token_ids) for request in self.requests) + self.max_new_tokens

    def with_max_new_tokens(self, max_new_tokens: int) -> WorkloadManifest:
        """Return an otherwise identical exact-length workload."""
        return replace(self, max_new_tokens=max_new_tokens)

    def with_prompt_jsonl(
        self,
        path: str | Path,
        *,
        tokenize: Callable[[str], Sequence[int]],
        tokenizer_path: str | Path,
        expected_sha256: str,
        expected_prompt_tokens: int | None = None,
        name: str | None = None,
    ) -> WorkloadManifest:
        """Replace requests with one hash-pinned, ID-preserving prompt JSONL source."""
        if expected_prompt_tokens is not None and expected_prompt_tokens <= 0:
            raise ValueError("expected_prompt_tokens must be positive")
        source = Path(path).resolve()
        payload = source.read_bytes()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        if source_sha256 != expected_sha256:
            raise ValueError(
                f"prompt source SHA256 mismatch: expected {expected_sha256}, observed {source_sha256}"
            )
        tokenizer = Path(tokenizer_path).resolve()
        tokenizer_sha256 = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        token_cache: dict[str, tuple[int, ...]] = {}
        requests = []
        for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
            if not line:
                raise ValueError(f"prompt source line {line_number} is blank")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"prompt source line {line_number} is not valid JSON") from error
            if not isinstance(row, dict) or set(row) != {"id", "prompt"}:
                raise ValueError(f"prompt source line {line_number} must contain exactly id and prompt")
            request_id = row["id"]
            prompt = row["prompt"]
            if not isinstance(request_id, str) or not isinstance(prompt, str):
                raise ValueError(f"prompt source line {line_number} id and prompt must be strings")
            if prompt not in token_cache:
                token_cache[prompt] = tuple(int(token_id) for token_id in tokenize(prompt))
            prompt_token_ids = token_cache[prompt]
            if expected_prompt_tokens is not None and len(prompt_token_ids) != expected_prompt_tokens:
                raise ValueError(
                    f"request {request_id} expected {expected_prompt_tokens} prompt tokens, "
                    f"observed {len(prompt_token_ids)}"
                )
            requests.append(WorkloadRequest(request_id=request_id, prompt_token_ids=prompt_token_ids))
        return replace(
            self,
            name=name or f"{self.name}-{source.stem}",
            requests=tuple(requests),
            prompt_source_path=str(source),
            prompt_source_sha256=source_sha256,
            prompt_tokenizer_path=str(tokenizer),
            prompt_tokenizer_sha256=tokenizer_sha256,
        )

    def with_request_count(
        self,
        request_count: int,
        *,
        request_id_prefix: str,
    ) -> WorkloadManifest:
        """Cycle real prompts into a deterministic workload of any positive size."""
        if request_count <= 0:
            raise ValueError("request_count must be positive")
        if not request_id_prefix:
            raise ValueError("request_id_prefix cannot be empty")
        width = max(4, len(str(request_count - 1)))
        requests = tuple(
            WorkloadRequest(
                request_id=f"{request_id_prefix}-{index:0{width}d}",
                prompt_token_ids=self.requests[index % len(self.requests)].prompt_token_ids,
            )
            for index in range(request_count)
        )
        return replace(self, name=f"{self.name}-n{request_count}", requests=requests)

    def with_uniform_prompt_length(
        self,
        prompt_length: int,
        *,
        request_count: int,
        request_id_prefix: str,
    ) -> WorkloadManifest:
        """Build exact synthetic pressure prompts by repeating real prompt tokens."""
        if prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        base = self.with_request_count(request_count, request_id_prefix=request_id_prefix)
        requests = []
        for request in base.requests:
            repetitions = math.ceil(prompt_length / len(request.prompt_token_ids))
            prompt_token_ids = (request.prompt_token_ids * repetitions)[:prompt_length]
            requests.append(replace(request, prompt_token_ids=prompt_token_ids))
        return replace(
            base,
            name=f"{self.name}-prompt{prompt_length}-n{request_count}",
            requests=tuple(requests),
        )

    def request_slice(self, start: int, stop: int) -> WorkloadManifest:
        """Return a deterministic request shard while preserving sampling settings."""
        return replace(self, requests=self.requests[start:stop], name=f"{self.name}[{start}:{stop}]")


@dataclass(frozen=True)
class GenerationRecord:
    """Tokens and chosen-token logprobs for one completed request."""

    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    output_logprobs: tuple[float, ...]
    requested_max_tokens: int
    finish_reason: str
    stop_reason: str | int | None
    stopped_on_eos: bool

    @property
    def output_sha256(self) -> str:
        """Return a compact stable output-token digest."""
        return _token_ids_sha256(self.output_token_ids)

    def summary_dict(self) -> dict[str, Any]:
        """Return exact lengths and digest without duplicating long token arrays."""
        return {
            "request_id": self.request_id,
            **exact_length_evidence(
                prompt_tokens=len(self.prompt_token_ids),
                generated_tokens=len(self.output_token_ids),
                requested_new_tokens=self.requested_max_tokens,
            ),
            "finish_reason": self.finish_reason,
            "stop_reason": self.stop_reason,
            "stopped_on_eos": self.stopped_on_eos,
            "output_sha256": self.output_sha256,
            "first_output_tokens": list(self.output_token_ids[:8]),
            "last_output_tokens": list(self.output_token_ids[-8:]),
            "all_logprobs_finite": all(math.isfinite(value) for value in self.output_logprobs),
        }


def exact_length_evidence(
    *,
    prompt_tokens: int,
    generated_tokens: int,
    requested_new_tokens: int,
) -> dict[str, int]:
    """Return explicit requested and observed prompt, generation, and total lengths."""
    return {
        "prompt_length": prompt_tokens,
        "output_length": generated_tokens,
        "requested_max_tokens": requested_new_tokens,
        "requested_prompt_tokens": prompt_tokens,
        "requested_new_tokens": requested_new_tokens,
        "requested_total_tokens": prompt_tokens + requested_new_tokens,
        "observed_prompt_tokens": prompt_tokens,
        "observed_new_tokens": generated_tokens,
        "observed_total_tokens": prompt_tokens + generated_tokens,
    }


@dataclass(frozen=True)
class BenchmarkSample:
    """One complete synchronized generation measurement."""

    sample_index: int
    generation_s: float
    request_count: int
    prompt_tokens: int
    generated_tokens: int
    ttft_s: tuple[float, ...]
    inter_token_latency_s: tuple[float, ...]
    output_lengths: tuple[int, ...]
    peak_device_memory_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject malformed timing samples."""
        if self.generation_s <= 0:
            raise ValueError("generation_s must be positive")
        if self.request_count <= 0:
            raise ValueError("request_count must be positive")
        if self.prompt_tokens <= 0 or self.generated_tokens <= 0:
            raise ValueError("sample token counts must be positive")
        if len(self.output_lengths) != self.request_count:
            raise ValueError("output_lengths must align with requests")

    @property
    def generated_tokens_per_s(self) -> float:
        """Return aggregate generated-token throughput."""
        return self.generated_tokens / self.generation_s

    @property
    def requests_per_s(self) -> float:
        """Return completed-request throughput."""
        return self.request_count / self.generation_s

    @property
    def batch_prefill_s(self) -> float:
        """Return wall time until every request in the batch emitted its first token."""
        return max(self.ttft_s)

    @property
    def batch_decode_s(self) -> float | None:
        """Return the longest request decode span after its first token."""
        if not self.inter_token_latency_s:
            return None
        return max(
            latency_s * (output_length - 1)
            for latency_s, output_length in zip(
                self.inter_token_latency_s,
                self.output_lengths,
                strict=False,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe sample including derived throughput."""
        result = asdict(self)
        result["generated_tokens_per_s"] = self.generated_tokens_per_s
        result["requests_per_s"] = self.requests_per_s
        result["batch_prefill_s"] = self.batch_prefill_s
        result["batch_decode_s"] = self.batch_decode_s
        return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + fraction * (ordered[upper] - ordered[lower]))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    median = float(statistics.median(values))
    deviations = [abs(value - median) for value in values]
    return {
        "median": median,
        "p95": _percentile(values, 0.95),
        "mad": float(statistics.median(deviations)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def aggregate_samples(samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
    """Aggregate independent raw samples without discarding outliers."""
    if not samples:
        raise ValueError("at least one benchmark sample is required")
    ttft = [value for sample in samples for value in sample.ttft_s]
    inter_token_latency = [value for sample in samples for value in sample.inter_token_latency_s]
    batch_decode = [value for sample in samples if (value := sample.batch_decode_s) is not None]
    peak_memory = [float(value) for sample in samples for value in sample.peak_device_memory_bytes]
    return {
        "sample_count": len(samples),
        "generation_s": _distribution([sample.generation_s for sample in samples]),
        "generated_tokens_per_s": _distribution([sample.generated_tokens_per_s for sample in samples]),
        "requests_per_s": _distribution([sample.requests_per_s for sample in samples]),
        "batch_prefill_s": _distribution([sample.batch_prefill_s for sample in samples]),
        "batch_decode_s": _distribution(batch_decode) if batch_decode else None,
        "ttft_s": _distribution(ttft),
        "inter_token_latency_s": _distribution(inter_token_latency) if inter_token_latency else None,
        "peak_device_memory_bytes": _distribution(peak_memory),
    }


def sampling_params_kwargs(manifest: WorkloadManifest) -> dict[str, Any]:
    """Return exact-length vLLM sampling settings matching the GDPO policy."""
    return {
        "temperature": manifest.temperature,
        "top_p": manifest.top_p,
        "top_k": manifest.top_k,
        "max_tokens": manifest.max_new_tokens,
        "min_tokens": manifest.max_new_tokens,
        "logprobs": 0,
        "stop_token_ids": list(manifest.stop_token_ids) or None,
        "ignore_eos": manifest.ignore_eos,
        "detokenize": False,
    }


def records_from_vllm_outputs(
    manifest: WorkloadManifest,
    outputs: Sequence[Any],
) -> tuple[GenerationRecord, ...]:
    """Adapt ordered offline vLLM outputs to the backend-neutral record schema."""
    if len(outputs) != len(manifest.requests):
        raise AssertionError("vLLM must return exactly one output per request")

    records = []
    for request, output in zip(manifest.requests, outputs, strict=True):
        if not output.finished or len(output.outputs) != 1:
            raise AssertionError(f"request {request.request_id} did not produce one finished completion")
        prompt_token_ids = tuple(output.prompt_token_ids or ())
        if prompt_token_ids != request.prompt_token_ids:
            raise AssertionError(f"request {request.request_id} prompt tokens changed or were reordered")

        completion = output.outputs[0]
        output_token_ids = tuple(int(token_id) for token_id in completion.token_ids)
        if completion.logprobs is None or len(completion.logprobs) != len(output_token_ids):
            raise AssertionError(f"request {request.request_id} is missing chosen-token logprobs")
        output_logprobs = []
        for token_id, position in zip(output_token_ids, completion.logprobs, strict=True):
            if position is None or token_id not in position:
                raise AssertionError(f"request {request.request_id} is missing a chosen-token logprob")
            output_logprobs.append(float(position[token_id].logprob))

        records.append(
            GenerationRecord(
                request_id=request.request_id,
                prompt_token_ids=prompt_token_ids,
                output_token_ids=output_token_ids,
                output_logprobs=tuple(output_logprobs),
                requested_max_tokens=manifest.max_new_tokens,
                finish_reason=str(completion.finish_reason),
                stop_reason=completion.stop_reason,
                stopped_on_eos=False,
            )
        )

    result = tuple(records)
    validate_generation_records(manifest, result)
    return result


def summarize_vllm_outputs(
    manifest: WorkloadManifest,
    outputs: Sequence[Any],
) -> tuple[dict[str, Any], ...]:
    """Validate and summarize vLLM outputs without copying complete token arrays."""
    if len(outputs) != len(manifest.requests):
        raise AssertionError("vLLM must return exactly one output per request")

    summaries = []
    for request, output in zip(manifest.requests, outputs, strict=True):
        if not output.finished or len(output.outputs) != 1:
            raise AssertionError(f"request {request.request_id} did not produce one finished completion")
        if tuple(output.prompt_token_ids or ()) != request.prompt_token_ids:
            raise AssertionError(f"request {request.request_id} prompt tokens changed or were reordered")

        completion = output.outputs[0]
        output_token_ids = completion.token_ids
        if len(output_token_ids) != manifest.max_new_tokens:
            raise AssertionError(
                f"request {request.request_id} must generate exactly {manifest.max_new_tokens} tokens"
            )
        if completion.logprobs is None or len(completion.logprobs) != len(output_token_ids):
            raise AssertionError(f"request {request.request_id} is missing chosen-token logprobs")
        all_logprobs_finite = True
        for token_id, position in zip(output_token_ids, completion.logprobs, strict=True):
            if position is None or token_id not in position:
                raise AssertionError(f"request {request.request_id} is missing a chosen-token logprob")
            all_logprobs_finite &= math.isfinite(float(position[token_id].logprob))
        if not all_logprobs_finite:
            raise AssertionError(f"request {request.request_id} has a non-finite logprob")
        finish_reason = str(completion.finish_reason)
        stop_reason = completion.stop_reason
        if finish_reason != "length" or stop_reason is not None:
            raise AssertionError(f"request {request.request_id} did not finish at the exact max-token limit")

        summaries.append(
            {
                "request_id": request.request_id,
                **exact_length_evidence(
                    prompt_tokens=len(request.prompt_token_ids),
                    generated_tokens=len(output_token_ids),
                    requested_new_tokens=manifest.max_new_tokens,
                ),
                "finish_reason": finish_reason,
                "stop_reason": stop_reason,
                "stopped_on_eos": False,
                "output_sha256": _token_ids_sha256(output_token_ids),
                "first_output_tokens": list(output_token_ids[:8]),
                "last_output_tokens": list(output_token_ids[-8:]),
                "all_logprobs_finite": True,
            }
        )
    return tuple(summaries)


def benchmark_sample_from_vllm_outputs(
    manifest: WorkloadManifest,
    outputs: Sequence[Any],
    *,
    sample_index: int,
    generation_s: float,
    peak_device_memory_bytes: tuple[int, ...],
    validated_summaries: Sequence[dict[str, Any]] | None = None,
) -> BenchmarkSample:
    """Build one synchronized sample from validated vLLM outputs and request metrics."""
    summaries = (
        summarize_vllm_outputs(manifest, outputs) if validated_summaries is None else tuple(validated_summaries)
    )
    if len(summaries) != len(outputs):
        raise AssertionError("validated summaries must align with vLLM outputs")
    ttft = []
    inter_token_latency = []
    for request, output, summary in zip(manifest.requests, outputs, summaries, strict=True):
        metrics = output.metrics
        if metrics is None:
            raise AssertionError(f"request {request.request_id} is missing vLLM timing metrics")
        if metrics.num_generation_tokens != summary["output_length"]:
            raise AssertionError(f"request {request.request_id} timing token count is inconsistent")
        ttft.append(float(metrics.first_token_latency))
        if summary["output_length"] > 1:
            decode_s = float(metrics.last_token_ts - metrics.first_token_ts)
            inter_token_latency.append(decode_s / (summary["output_length"] - 1))

    return BenchmarkSample(
        sample_index=sample_index,
        generation_s=generation_s,
        request_count=len(summaries),
        prompt_tokens=sum(summary["prompt_length"] for summary in summaries),
        generated_tokens=sum(summary["output_length"] for summary in summaries),
        ttft_s=tuple(ttft),
        inter_token_latency_s=tuple(inter_token_latency),
        output_lengths=tuple(summary["output_length"] for summary in summaries),
        peak_device_memory_bytes=peak_device_memory_bytes,
    )


def validate_compilation_proof(
    initialized: dict[str, int],
    after_warm_replay: dict[str, int],
) -> None:
    """Require Inductor CUDA graphs and reject eager or warm-run recompilation."""
    required = {
        "num_models_seen",
        "num_backend_compilations",
        "num_inductor_compiles",
        "num_eager_compiles",
        "num_gpu_runner_capture_triggers",
        "num_cudagraph_captured",
        "stock_torch_compile_count",
    }
    for label, snapshot in (("initialized", initialized), ("after_warm_replay", after_warm_replay)):
        missing = required - snapshot.keys()
        if missing:
            raise AssertionError(f"{label} compilation snapshot is missing {sorted(missing)}")

    if initialized["num_models_seen"] < 1:
        raise AssertionError("the Evo2 model was not seen by the compiler")
    if initialized["num_backend_compilations"] <= 0:
        raise AssertionError("no vLLM backend compilation was recorded")
    if initialized["num_inductor_compiles"] <= 0:
        raise AssertionError("no Inductor compilation was recorded")
    if initialized["num_eager_compiles"] != 0:
        raise AssertionError("eager compilation is forbidden")
    if initialized["num_gpu_runner_capture_triggers"] <= 0:
        raise AssertionError("CUDA graph capture was not triggered")
    if initialized["num_cudagraph_captured"] <= 0:
        raise AssertionError("no CUDA graphs were captured")
    if after_warm_replay["num_eager_compiles"] != 0:
        raise AssertionError("warm replay entered eager compilation")
    for field in required:
        if after_warm_replay[field] != initialized[field]:
            raise AssertionError(f"warm replay caused an unexpected recompile or graph recapture: {field}")


def validate_generation_records(
    manifest: WorkloadManifest,
    records: Sequence[GenerationRecord],
) -> None:
    """Require exact request, prompt, output-length, and logprob parity."""
    expected_ids = tuple(request.request_id for request in manifest.requests)
    actual_ids = tuple(record.request_id for record in records)
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise AssertionError("every request ID must appear exactly once")
    if actual_ids != expected_ids:
        raise AssertionError("generation records must preserve input request order")

    for request, record in zip(manifest.requests, records, strict=True):
        if record.prompt_token_ids != request.prompt_token_ids:
            raise AssertionError(f"request {request.request_id} prompt tokens changed")
        if len(record.output_token_ids) != manifest.max_new_tokens:
            raise AssertionError(
                f"request {request.request_id} must generate exactly {manifest.max_new_tokens} tokens"
            )
        if record.requested_max_tokens != manifest.max_new_tokens:
            raise AssertionError(f"request {request.request_id} requested max-token limit drifted")
        if record.finish_reason != "length" or record.stop_reason is not None:
            raise AssertionError(f"request {request.request_id} did not finish at the exact max-token limit")
        if record.stopped_on_eos:
            raise AssertionError(f"request {request.request_id} stopped on EOS during exact-length generation")
        if len(record.output_logprobs) != len(record.output_token_ids):
            raise AssertionError(f"request {request.request_id} token/logprob lengths differ")
        if not all(math.isfinite(value) for value in record.output_logprobs):
            raise AssertionError(f"request {request.request_id} has a non-finite logprob")


def build_parser() -> argparse.ArgumentParser:
    """Build the backend-neutral benchmark CLI parser."""
    parser = argparse.ArgumentParser(description="Benchmark Evo2 MCore or vLLM generation")
    parser.add_argument("--backend", choices=("mcore", "vllm"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--topology", choices=("tp2", "dp2"), required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, choices=(16_384, 32_768), required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, choices=(0.92, 0.95, 0.97), required=True)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--request-count", type=int)
    parser.add_argument("--uniform-prompt-length", type=int)
    parser.add_argument("--request-id-prefix", default="benchmark")
    parser.add_argument("--prompt-jsonl", type=Path)
    parser.add_argument("--prompt-jsonl-sha256")
    parser.add_argument("--prompt-tokenizer-json", type=Path)
    parser.add_argument("--expected-prompt-tokens", type=int)
    parser.add_argument("--load-format", choices=("safetensors", "dummy"), default="safetensors")
    parser.add_argument("--optimization-level", type=int, choices=(2, 3), default=2)
    parser.add_argument("--performance-mode", choices=("balanced", "throughput"), default="balanced")
    parser.add_argument("--generation-round", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--proof", action="store_true")
    parser.add_argument("--context-preflight-only", action="store_true")
    parser.add_argument("--async-scheduling", action="store_true")
    parser.add_argument("--max-concurrent-partial-prefills", type=int, default=1)
    parser.add_argument("--long-prefill-chunk-tokens", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


__all__ = [
    "BenchmarkSample",
    "GenerationRecord",
    "RequestShard",
    "RequestWave",
    "WorkloadManifest",
    "WorkloadRequest",
    "aggregate_samples",
    "benchmark_sample_from_vllm_outputs",
    "build_parser",
    "build_request_waves",
    "exact_length_evidence",
    "records_from_vllm_outputs",
    "sampling_params_kwargs",
    "summarize_vllm_outputs",
    "validate_compilation_proof",
    "validate_generation_records",
]
