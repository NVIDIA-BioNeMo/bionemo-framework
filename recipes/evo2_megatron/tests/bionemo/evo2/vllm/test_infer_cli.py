# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from bionemo.evo2.vllm.infer import (
    InferenceRequest,
    build_engine_kwargs,
    build_sampling_params_kwargs,
    load_export_identity,
    load_prompt_requests,
    records_from_public_outputs,
    resolve_tensor_parallel_size,
    require_vllm_runtime,
)
from bionemo.evo2.vllm.tokenizer_io import SnapshotBoundTokenizer


TOKENIZER_JSON = Path(__file__).resolve().parents[4] / "tokenizers/nucleotide_fast_tokenizer_512/tokenizer.json"
EVO2_RECIPE = Path(__file__).resolve().parents[4]
PHAGE_RECIPE = EVO2_RECIPE.parent / "evo2_phage_gen"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _export_root(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    root.mkdir()
    config_sha256 = _write_json(
        root / "config.json",
        {
            "architectures": ["Evo2ForCausalLM"],
            "hidden_size": 4096,
            "model_type": "evo2",
            "num_attention_heads": 32,
            "vocab_size": 512,
        },
    )
    index_sha256 = _write_json(
        root / "model.safetensors.index.json",
        {"metadata": {"total_size": 4}, "weight_map": {"weight": "model-00001-of-00001.safetensors"}},
    )
    (root / "model-00001-of-00001.safetensors").write_bytes(b"test")
    _write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "source_checkpoint": "/checkpoint/iter_0000001",
            "source_iteration": 1,
        },
    )
    return root


def _fake_output(
    *,
    request_id: str,
    prompt_token_ids: tuple[int, ...],
    output_token_ids: tuple[int, ...],
    logprobs: tuple[float, ...],
) -> SimpleNamespace:
    positions = [
        {token_id: SimpleNamespace(logprob=logprob)}
        for token_id, logprob in zip(output_token_ids, logprobs, strict=True)
    ]
    return SimpleNamespace(
        request_id=request_id,
        finished=True,
        prompt_token_ids=list(prompt_token_ids),
        outputs=[
            SimpleNamespace(
                token_ids=list(output_token_ids),
                logprobs=positions,
                finish_reason="length",
                stop_reason=None,
            )
        ],
    )


def test_evo2_packages_publish_vllm_inference_and_explicit_mcore_compatibility_commands() -> None:
    for recipe in (EVO2_RECIPE, PHAGE_RECIPE):
        with (recipe / "pyproject.toml").open("rb") as stream:
            scripts = tomllib.load(stream)["project"]["scripts"]
        _require(
            scripts.get("infer_evo2") == "bionemo.evo2.vllm.infer:main",
            f"{recipe.name} infer_evo2 is not vLLM-backed",
        )
        _require(
            scripts.get("infer_evo2_mcore") == "bionemo.evo2.run.infer:main",
            f"{recipe.name} did not preserve the explicit MCore compatibility command",
        )
        _require(
            scripts.get("evo2_export_mbridge_to_vllm") == "bionemo.evo2.vllm.export:main",
            "export command is missing",
        )


def test_inference_reexecs_into_the_locked_recipe_vllm_environment(monkeypatch, tmp_path) -> None:
    actor_root = tmp_path / "actor-venvs"
    actor_python = (
        actor_root
        / "bionemo.evo2.vllm.nemo_generation_worker.Evo2NemoRlGenerationWorker"
        / "bin"
        / "python"
    )
    actor_python.parent.mkdir(parents=True)
    actor_python.write_text("#!/bin/sh\n", encoding="utf-8")
    actor_python.chmod(0o755)
    monkeypatch.setenv("NEMO_RL_VENV_DIR", str(actor_root))
    monkeypatch.delenv("EVO2_VLLM_PYTHON", raising=False)
    monkeypatch.delenv("EVO2_VLLM_REEXEC", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "vllm" else object())

    captured: dict[str, object] = {}

    class ReexecObserved(Exception):
        pass

    def _capture_execve(executable: str, arguments: list[str], environment: dict[str, str]) -> None:
        captured.update(executable=executable, arguments=arguments, environment=environment)
        raise ReexecObserved

    monkeypatch.setattr(os, "execve", _capture_execve)
    with pytest.raises(ReexecObserved):
        require_vllm_runtime(argv=("--model", "/export", "--prompt", "AC"))

    _require(captured["executable"] == str(actor_python), "inference did not use the locked actor interpreter")
    _require(
        captured["arguments"]
        == [str(actor_python), "-m", "bionemo.evo2.vllm.infer", "--model", "/export", "--prompt", "AC"],
        "inference re-exec arguments changed",
    )
    environment = captured["environment"]
    _require(type(environment) is dict and environment.get("EVO2_VLLM_REEXEC") == "1", "re-exec guard is missing")
    _require(
        "PYTHONPATH" not in environment or environment["PYTHONPATH"] == os.environ.get("PYTHONPATH"),
        "handoff changed PYTHONPATH",
    )


def test_inference_keeps_an_already_capable_environment(monkeypatch) -> None:
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object() if name == "vllm" else None)
    monkeypatch.setattr(os, "execve", lambda *_args: pytest.fail("capable environment was replaced"))
    require_vllm_runtime(argv=("--help",))


@pytest.mark.parametrize(
    ("environment", "message"),
    (
        ({}, "not installed"),
        ({"NEMO_RL_VENV_DIR": "/missing"}, "does not exist"),
        ({"NEMO_RL_VENV_DIR": "/missing", "EVO2_VLLM_REEXEC": "1"}, "still cannot import"),
    ),
)
def test_inference_fails_closed_for_missing_or_broken_vllm_environment(monkeypatch, environment, message) -> None:
    for name in ("NEMO_RL_VENV_DIR", "EVO2_VLLM_PYTHON", "EVO2_VLLM_REEXEC"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "vllm" else object())
    with pytest.raises(RuntimeError, match=message):
        require_vllm_runtime(argv=("--model", "/export"))


def test_resolve_tensor_parallel_size_uses_every_visible_gpu_without_pinning_tp2(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5,6,7")
    _require(resolve_tensor_parallel_size("auto") == 4, "auto TP did not use all four visible GPUs")
    _require(resolve_tensor_parallel_size("2") == 2, "explicit TP2 changed")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    _require(resolve_tensor_parallel_size("auto") == 1, "single-GPU auto TP changed")

    with pytest.raises(ValueError, match="visible GPU"):
        resolve_tensor_parallel_size("2")


def test_engine_kwargs_enable_qualified_adaptive_performance_route() -> None:
    kwargs = build_engine_kwargs(
        model="/model",
        tensor_parallel_size=4,
        batch_size=96,
        max_model_len=6016,
        max_num_batched_tokens=16_384,
        gpu_memory_utilization=0.91,
        optimization_level=2,
        performance_mode="balanced",
        async_scheduling=True,
    )

    expected = {
        "model": "/model",
        "tensor_parallel_size": 4,
        "distributed_executor_backend": "mp",
        "async_scheduling": True,
        "optimization_level": 2,
        "performance_mode": "balanced",
        "max_num_seqs": 96,
        "max_model_len": 6016,
        "max_num_batched_tokens": 16_384,
        "logprobs_mode": "processed_logprobs",
        "enable_chunked_prefill": True,
        "enforce_eager": False,
    }
    for key, value in expected.items():
        _require(kwargs.get(key) == value, f"engine kwarg {key} changed")
    _require("worker_extension_cls" not in kwargs, "normal inference installed a proof worker extension")
    compilation = kwargs["compilation_config"]
    _require(compilation["mode"] == 3, "compilation mode changed")
    _require(compilation["backend"] == "inductor", "compilation backend changed")
    _require(compilation["cudagraph_mode"] == "FULL_AND_PIECEWISE", "CUDA graph mode changed")
    _require(96 in compilation["compile_sizes"], "exact batch compile size is missing")
    _require(96 in compilation["cudagraph_capture_sizes"], "exact batch capture size is missing")
    _require("bionemo_evo2::hyena_mixer" in compilation["splitting_ops"], "Hyena split point is missing")


def test_engine_kwargs_keep_async_and_exact_capture_for_single_gpu() -> None:
    kwargs = build_engine_kwargs(
        model="/model",
        tensor_parallel_size=1,
        batch_size=7,
        max_model_len=512,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.91,
        optimization_level=2,
        performance_mode="balanced",
        async_scheduling=True,
    )
    _require("distributed_executor_backend" not in kwargs, "single GPU must use vLLM's uniprocess executor")
    _require(kwargs["async_scheduling"] is True, "single-GPU async scheduling was disabled")
    _require(7 in kwargs["compilation_config"]["cudagraph_capture_sizes"], "exact B7 capture is missing")


def test_load_export_identity_binds_config_index_and_source_checkpoint(tmp_path) -> None:
    root = _export_root(tmp_path)
    identity = load_export_identity(root)

    _require(identity.root == root.resolve(), "export root changed")
    _require(identity.architecture == "Evo2ForCausalLM", "architecture changed")
    _require(identity.source_checkpoint == "/checkpoint/iter_0000001", "source checkpoint changed")
    _require(identity.source_iteration == 1, "source iteration changed")
    _require(identity.tensor_parallel_divisor == 32, "attention-head TP divisor changed")

    index = json.loads((root / "model.safetensors.index.json").read_text())
    index["weight_map"]["other"] = "model-00001-of-00001.safetensors"
    _write_json(root / "model.safetensors.index.json", index)
    with pytest.raises(RuntimeError, match="index digest"):
        load_export_identity(root)


def test_load_prompt_requests_uses_one_strict_jsonl_snapshot(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text('{"id":"a","prompt":"AC"}\n{"id":"b","prompt":"GT"}\n', encoding="utf-8")
    requests = load_prompt_requests(prompt=None, prompt_file=path)
    _require(
        requests == (InferenceRequest(request_id="a", prompt="AC"), InferenceRequest("b", "GT")),
        "prompts changed",
    )

    path.write_text('{"id":"a","prompt":"AC","prompt":"GT"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_prompt_requests(prompt=None, prompt_file=path)


def test_sampling_params_preserve_exact_length_chosen_logprobs_and_unique_seeds() -> None:
    first = build_sampling_params_kwargs(
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        top_k=4,
        seed=42,
    )
    second = build_sampling_params_kwargs(
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        top_k=4,
        seed=43,
    )
    _require(first["max_tokens"] == first["min_tokens"] == 4, "exact output length changed")
    _require(first["logprobs"] == 0, "chosen-only logprob mode changed")
    _require(first["allowed_token_ids"] == [65, 67, 71, 78, 84], "ACGTN policy changed")
    _require(first["ignore_eos"] is True, "EOS stopping was enabled")
    _require(first["seed"] == 42 and second["seed"] == 43, "request seeds are not unique")

    greedy = build_sampling_params_kwargs(
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        seed=42,
    )
    _require(greedy["temperature"] == 0.0, "top_k=1 did not use true greedy")
    _require(greedy["top_k"] == 0 and greedy["top_p"] == 1.0, "greedy aliases changed")


def test_public_outputs_preserve_legacy_jsonl_and_strict_generation_evidence() -> None:
    tokenizer = SnapshotBoundTokenizer.from_path(TOKENIZER_JSON)
    requests = (
        InferenceRequest("left", "AC"),
        InferenceRequest("right", "GT"),
    )
    prompt_ids = tuple(tokenizer.encode(request.prompt) for request in requests)
    outputs = (
        _fake_output(
            request_id="engine-9bc2",
            prompt_token_ids=prompt_ids[0],
            output_token_ids=(65, 67, 71, 84),
            logprobs=(-0.1, -0.2, -0.3, -0.4),
        ),
        _fake_output(
            request_id="engine-a18f",
            prompt_token_ids=prompt_ids[1],
            output_token_ids=(84, 71, 67, 65),
            logprobs=(-0.4, -0.3, -0.2, -0.1),
        ),
    )
    records = records_from_public_outputs(
        requests=requests,
        prompt_token_ids=prompt_ids,
        request_seeds=(42, 43),
        outputs=outputs,
        tokenizer=tokenizer,
        max_new_tokens=4,
    )

    _require([record["id"] for record in records] == ["left", "right"], "caller IDs changed")
    _require([record["completion"] for record in records] == ["ACGT", "TGCA"], "decoded output changed")
    _require(records[0]["usage"] == {"prompt_tokens": 2, "completion_tokens": 4, "total_tokens": 6}, "usage changed")
    _require(records[0]["seed"] == 42 and records[1]["seed"] == 43, "seeds changed")
    _require(records[0]["engine_request_id"] != records[1]["engine_request_id"], "engine IDs are not unique")
    _require(records[0]["token_ids"] == [65, 67, 71, 84], "raw output IDs changed")
    _require(records[0]["logprobs"]["completion_logprobs"] == [-0.1, -0.2, -0.3, -0.4], "chosen logprobs changed")


@pytest.mark.parametrize(
    ("output_token_ids", "logprobs", "message"),
    (
        ((65, 67, 0, 84), (-0.1, -0.2, -0.3, -0.4), "A/C/G/N/T"),
        ((65, 67, 71), (-0.1, -0.2, -0.3), "exactly 4"),
        ((65, 67, 71, 84), (-0.1, float("nan"), -0.3, -0.4), "finite"),
    ),
)
def test_public_outputs_reject_invalid_tokens_lengths_and_logprobs(
    output_token_ids: tuple[int, ...],
    logprobs: tuple[float, ...],
    message: str,
) -> None:
    tokenizer = SnapshotBoundTokenizer.from_path(TOKENIZER_JSON)
    request = InferenceRequest("request", "AC")
    prompt_ids = tokenizer.encode(request.prompt)
    output = _fake_output(
        request_id="engine",
        prompt_token_ids=prompt_ids,
        output_token_ids=output_token_ids,
        logprobs=logprobs,
    )
    with pytest.raises((AssertionError, ValueError), match=message):
        records_from_public_outputs(
            requests=(request,),
            prompt_token_ids=(prompt_ids,),
            request_seeds=(42,),
            outputs=(output,),
            tokenizer=tokenizer,
            max_new_tokens=4,
        )
