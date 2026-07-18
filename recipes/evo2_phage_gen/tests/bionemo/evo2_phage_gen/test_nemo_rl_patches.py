# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``bionemo.evo2_phage_gen.nemo_rl_patches``."""

from __future__ import annotations

import importlib.util
import subprocess
import types
from pathlib import Path

import pytest
import torch

from bionemo.evo2_phage_gen import nemo_rl_patches


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_patched_nemo_rl_prompt_digest_matches_portable_vector() -> None:
    from nemo_rl.models.generation.interfaces import (
        generation_prompt_token_ids_bytes,
        generation_prompt_token_ids_sha256,
    )

    expected_hex = "67656e65726174696f6e2e70726f6d70745f746f6b656e5f6964732e7631000000000000000001000000000000002b"
    expected_sha256 = "8fcfb284618fdd1c28d8a7022eee50831e44986fac86e48b396800bf5ba2c93b"

    _require(generation_prompt_token_ids_bytes([43]).hex() == expected_hex, "prompt digest bytes drifted")
    _require(
        generation_prompt_token_ids_sha256([43]) == expected_sha256,
        "prompt digest SHA256 drifted",
    )
    for invalid in ([True], [-1], [2**63], type("ListSubclass", (list,), {})([43])):
        with pytest.raises((TypeError, ValueError), match="prompt token IDs"):
            generation_prompt_token_ids_sha256(invalid)


def test_production_evo2_generation_worker_imports_with_maintained_patch() -> None:
    """The recipe worker must depend only on symbols shipped by the maintained patch."""
    from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorkerImpl

    from bionemo.evo2.vllm.nemo_generation_worker import Evo2NemoRlGenerationWorkerImpl

    _require(
        issubclass(Evo2NemoRlGenerationWorkerImpl, VllmGenerationWorkerImpl),
        "the production Evo2 worker must remain a thin NeMo-RL vLLM worker subclass",
    )


def test_patched_vllm_dp2_shards_every_prompt_group_and_restores_caller_order() -> None:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.vllm.vllm_generation import (
        _restore_generation_caller_order,
        _shard_generation_batch_by_prompt_group,
    )

    prompt_ids = torch.arange(8, dtype=torch.int64).repeat(12)
    caller_indices = torch.arange(1_000, 1_096, dtype=torch.int64)
    batch = BatchedDataDict(
        {
            "input_ids": prompt_ids.unsqueeze(1),
            "input_lengths": torch.ones(96, dtype=torch.int64),
            "generation_prompt_group_ids": [
                f"prompt-{prompt_id}" for prompt_id in prompt_ids.tolist()
            ],
            "generation_global_request_indices": caller_indices,
        }
    )

    shards = _shard_generation_batch_by_prompt_group(
        batch,
        dp_size=2,
        prompt_group_size=12,
    )

    _require([shard.size for shard in shards] == [48, 48], "DP2 shard sizes drifted")
    for rank, shard in enumerate(shards):
        observed = torch.bincount(shard["input_ids"][:, 0], minlength=8).tolist()
        _require(observed == [6] * 8, f"DP rank {rank} lost a prompt group: {observed}")

    rank_major = BatchedDataDict.from_batches(shards)
    restored = _restore_generation_caller_order(
        rank_major,
        expected_global_request_indices=caller_indices,
    )
    _require(
        restored["generation_global_request_indices"].tolist()
        == caller_indices.tolist(),
        "rank-major results were not restored to caller order",
    )
    _require(
        restored["input_ids"][:, 0].tolist() == prompt_ids.tolist(),
        "prompt rows detached from caller request identities",
    )

    duplicate = BatchedDataDict(dict(rank_major.data))
    duplicate["generation_global_request_indices"] = rank_major[
        "generation_global_request_indices"
    ].clone()
    duplicate["generation_global_request_indices"][0] = duplicate[
        "generation_global_request_indices"
    ][1]
    with pytest.raises(ValueError, match="caller request inventory"):
        _restore_generation_caller_order(
            duplicate,
            expected_global_request_indices=caller_indices,
        )


def test_patched_vllm_dp2_rejects_unsealed_prompt_groups() -> None:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.vllm.vllm_generation import (
        _shard_generation_batch_by_prompt_group,
    )

    base = {
        "input_ids": torch.arange(24, dtype=torch.int64).unsqueeze(1),
        "input_lengths": torch.ones(24, dtype=torch.int64),
        "generation_global_request_indices": torch.arange(24, dtype=torch.int64),
    }
    with pytest.raises(TypeError, match="generation_prompt_group_ids"):
        _shard_generation_batch_by_prompt_group(
            BatchedDataDict(base),
            dp_size=2,
            prompt_group_size=12,
        )

    invalid_type = dict(base)
    invalid_type["generation_prompt_group_ids"] = ["prompt-a"] * 23 + [1]
    with pytest.raises(TypeError, match="generation_prompt_group_ids"):
        _shard_generation_batch_by_prompt_group(
            BatchedDataDict(invalid_type),
            dp_size=2,
            prompt_group_size=12,
        )

    incomplete = dict(base)
    incomplete["generation_prompt_group_ids"] = ["prompt-a"] * 13 + ["prompt-b"] * 11
    with pytest.raises(ValueError, match="complete semantic prompt groups"):
        _shard_generation_batch_by_prompt_group(
            BatchedDataDict(incomplete),
            dp_size=2,
            prompt_group_size=12,
        )


def test_patched_rollout_retains_generation_coordinates_on_assistant_message() -> None:
    from nemo_rl.experience.rollouts import _attach_generation_request_metadata
    from nemo_rl.models.generation.interfaces import GENERATION_REQUEST_METADATA_KEYS

    outputs = {
        key: torch.tensor([value, value + 1], dtype=torch.int64)
        for key, value in zip(
            GENERATION_REQUEST_METADATA_KEYS,
            (42, 1_000, 7, 9, 0),
            strict=True,
        )
    }
    assistant_message: dict[str, object] = {"role": "assistant", "content": "AC"}

    _attach_generation_request_metadata(
        assistant_message,
        outputs,
        row_index=1,
        request_count=2,
    )

    _require(
        {key: assistant_message[key] for key in GENERATION_REQUEST_METADATA_KEYS}
        == {
            key: int(outputs[key][1].item()) for key in GENERATION_REQUEST_METADATA_KEYS
        },
        "assistant generation coordinates drifted",
    )

    partial = dict(outputs)
    partial.pop(GENERATION_REQUEST_METADATA_KEYS[-1])
    with pytest.raises(RuntimeError, match="incomplete generation request metadata"):
        _attach_generation_request_metadata(
            {"role": "assistant", "content": "AC"},
            partial,
            row_index=0,
            request_count=2,
        )


def test_patched_rollout_carries_semantic_prompt_groups_to_generation(monkeypatch) -> None:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.experience import rollouts

    expected_prompt_groups = ["prompt-4", "prompt-5", "prompt-4", "prompt-5"]
    captured_prompt_groups: list[list[str]] = []

    def fake_generate_responses(
        policy_generation,
        generation_input_data,
        batch,
        tokenizer,
        input_lengths,
        include_logprobs=True,
        greedy=False,
    ):
        captured_prompt_groups.append(generation_input_data["generation_prompt_group_ids"])
        generated_ids = [torch.tensor([65], dtype=torch.int64) for _ in input_lengths]
        return batch, generated_ids, {
            "mean_generation_length": 1.0,
            "total_generated_tokens": len(generated_ids),
        }

    def fake_calculate_rewards(batch, task_to_env):
        request_count = batch.size
        return types.SimpleNamespace(
            observations=[{"role": "environment", "content": ""}] * request_count,
            metadata=[None] * request_count,
            next_stop_strings=[None] * request_count,
            rewards=torch.ones(request_count, dtype=torch.float32),
            terminateds=torch.ones(request_count, dtype=torch.bool),
            answers=[None] * request_count,
        )

    class Tokenizer:
        pad_token_id = 0

        def __call__(self, text, return_tensors, add_special_tokens):
            return types.SimpleNamespace(input_ids=torch.empty((1, 0), dtype=torch.int64))

    monkeypatch.setattr(rollouts, "generate_responses", fake_generate_responses)
    monkeypatch.setattr(rollouts, "calculate_rewards", fake_calculate_rewards)
    input_batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": prompt_group,
                        "token_ids": torch.tensor([65], dtype=torch.int64),
                    }
                ]
                for prompt_group in expected_prompt_groups
            ],
            "extra_env_info": [
                {"prompt_id": prompt_group} for prompt_group in expected_prompt_groups
            ],
        }
    )

    rollouts.run_multi_turn_rollout(
        policy_generation=object(),
        input_batch=input_batch,
        tokenizer=Tokenizer(),
        task_to_env={},
        max_seq_len=32,
        max_rollout_turns=1,
    )

    _require(
        captured_prompt_groups == [expected_prompt_groups],
        "semantic prompt groups did not reach the production generation input",
    )


def test_patched_grpo_logs_prompt_rollout_identity_and_exact_generation_evidence() -> (
    None
):
    from nemo_rl.algorithms.grpo import (
        _rollout_log_fields,
        _stamp_repeated_rollout_ordinals,
    )
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.generation.interfaces import GENERATION_REQUEST_METADATA_KEYS

    repeated = BatchedDataDict(
        {
            "extra_env_info": [
                {"prompt_id": f"prompt-{prompt}", "length_stratum": prompt + 4}
                for prompt in range(2)
                for _ in range(3)
            ]
        }
    )
    _stamp_repeated_rollout_ordinals(repeated, num_generations_per_prompt=3)

    message_logs = []
    for row_index in range(6):
        generated = {
            "role": "assistant",
            "content": "AC",
            "token_ids": torch.tensor([65, 67], dtype=torch.int64),
            "generation_logprobs": torch.tensor([-0.25, -0.5], dtype=torch.float32),
        }
        for key, value in zip(
            GENERATION_REQUEST_METADATA_KEYS,
            (42 + row_index, 1_000 + row_index, 7, 9, row_index // 3),
            strict=True,
        ):
            generated[key] = value
        message_logs.append(
            [
                {
                    "role": "user",
                    "content": "+~A",
                    "token_ids": torch.tensor([43, 126, 65]),
                },
                generated,
            ]
        )

    fields = _rollout_log_fields(
        message_logs,
        repeated["extra_env_info"],
        include_output_evidence=True,
    )

    _require(
        fields["prompt_id"] == ["prompt-0"] * 3 + ["prompt-1"] * 3, "prompt IDs drifted"
    )
    _require(fields["length_stratum"] == [4] * 3 + [5] * 3, "length strata drifted")
    _require(
        fields["rollout_ordinal"] == [0, 1, 2, 0, 1, 2], "rollout ordinals drifted"
    )
    _require(
        fields["generation_request_seeds"] == [[seed] for seed in range(42, 48)],
        "request seeds drifted",
    )
    _require(fields["generated_token_ids"] == [[[65, 67]]] * 6, "generated IDs drifted")
    _require(
        fields["generated_chosen_logprobs"] == [[[-0.25, -0.5]]] * 6,
        "chosen logprob evidence drifted",
    )

    message_logs[0][1]["generation_logprobs"][1] = float("nan")
    with pytest.raises(ValueError, match="finite and token-aligned"):
        _rollout_log_fields(
            message_logs,
            repeated["extra_env_info"],
            include_output_evidence=True,
        )


def test_apply_nemo_rl_patch_applies_against_installed_package_root(
    tmp_path: Path, monkeypatch
) -> None:
    """The patch command should run from site-packages, not require a source checkout path."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text("diff --git a/nemo_rl/example.py b/nemo_rl/example.py\n")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd))
        return subprocess.CompletedProcess(["patch", *args], 0, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)

    assert result == f"patch applied to {source_root}"
    assert calls == [
        (["--batch", "--dry-run", "-p1", "-i", str(patch_file)], source_root),
        (["--batch", "-p1", "-i", str(patch_file)], source_root),
    ]


def test_apply_nemo_rl_patch_reports_already_applied(tmp_path: Path, monkeypatch) -> None:
    """Reverse dry-run success means the patch is already present."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text("diff --git a/nemo_rl/example.py b/nemo_rl/example.py\n")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return_code = 0 if "-R" in args else 1
        return subprocess.CompletedProcess(["patch", *args], return_code, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)

    assert result == f"patch already applied to {source_root}"


def test_default_nemo_rl_patch_series_is_ordered() -> None:
    _require(
        tuple(path.name for path in nemo_rl_patches.DEFAULT_PATCHES)
        == (
            "nemo-rl-evo2-policy.patch",
            "nemo-rl-evo2-vllm.patch",
            "nemo-rl-evo2-sampling.patch",
        ),
        "maintained NeMo-RL patch order drifted",
    )


def test_apply_nemo_rl_patch_applies_default_series_in_order(tmp_path: Path, monkeypatch) -> None:
    """The default CLI path should apply every maintained patch in declared order."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    patch_paths = (tmp_path / "policy.patch", tmp_path / "vllm.patch", tmp_path / "sampling.patch")
    for path in patch_paths:
        path.write_text(f"diff --git a/{path.stem} b/{path.stem}\n")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)
    calls: list[list[str]] = []

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(nemo_rl_patches, "DEFAULT_PATCHES", patch_paths)

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        _require(cwd == source_root, "patch series used the wrong source root")
        calls.append(args)
        return subprocess.CompletedProcess(["patch", *args], 0, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    result = nemo_rl_patches.apply_nemo_rl_patch()

    _require(
        result
        == (
            f"patch applied to {source_root}\n"
            f"patch applied to {source_root}\n"
            f"patch applied to {source_root}"
        ),
        "default series result drifted",
    )
    _require(
        [Path(args[-1]).name for args in calls]
        == [
            "policy.patch",
            "policy.patch",
            "vllm.patch",
            "vllm.patch",
            "sampling.patch",
            "sampling.patch",
        ],
        "default patch order drifted",
    )


def test_run_patch_uses_real_batch_dry_run(tmp_path: Path) -> None:
    """The maintained check path should use a real noninteractive patch dry-run."""
    source_file = tmp_path / "nemo_rl" / "example.py"
    source_file.parent.mkdir()
    source_file.write_text("old\n")
    patch_file = tmp_path / "change.patch"
    patch_file.write_text(
        "\n".join(
            [
                "diff --git a/nemo_rl/example.py b/nemo_rl/example.py",
                "index 1111111..2222222 100644",
                "--- a/nemo_rl/example.py",
                "+++ b/nemo_rl/example.py",
                "@@ -1 +1 @@",
                "-old",
                "+new",
                "",
            ]
        )
    )

    result = nemo_rl_patches._run_patch(["--batch", "--dry-run", "-p1", "-i", str(patch_file)], cwd=tmp_path)

    assert result.returncode == 0
    assert "--forward" in result.args
    assert source_file.read_text() == "old\n"


def test_apply_nemo_rl_patch_is_forward_only_and_idempotent(tmp_path: Path, monkeypatch) -> None:
    """Rerunning the patcher on an already-patched runtime must not reverse the patch."""
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    source_file = package_dir / "example.py"
    source_file.write_text("old\n")
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text(
        "\n".join(
            [
                "diff --git a/nemo_rl/example.py b/nemo_rl/example.py",
                "index 1111111..2222222 100644",
                "--- a/nemo_rl/example.py",
                "+++ b/nemo_rl/example.py",
                "@@ -1 +1 @@",
                "-old",
                "+new",
                "",
            ]
        )
    )
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)
    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)

    first_result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)
    second_result = nemo_rl_patches.apply_nemo_rl_patch(patch_file)

    assert first_result == f"patch applied to {source_root}"
    assert second_result == f"patch already applied to {source_root}"
    assert source_file.read_text() == "new\n"


def test_repair_install_exposes_new_editable_source_in_current_process(tmp_path: Path, monkeypatch) -> None:
    """The patch CLI must see an editable install created after interpreter startup."""
    source_root = tmp_path / "nemo-rl-source"
    (source_root / "nemo_rl").mkdir(parents=True)
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(
        nemo_rl_patches,
        "_ensure_pinned_nemo_rl_source",
        lambda *, force_reinstall: source_root,
    )
    monkeypatch.setattr(nemo_rl_patches, "_is_complete_nemo_rl_install", lambda: False)
    monkeypatch.setattr(
        nemo_rl_patches.sys,
        "path",
        [entry for entry in nemo_rl_patches.sys.path if entry != str(source_root)],
    )

    def fake_run(args: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        subprocess_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(nemo_rl_patches.subprocess, "run", fake_run)

    result = nemo_rl_patches.repair_nemo_rl_install(force_reinstall=True)

    _require(result == f"reinstalled editable nemo-rl from retained source {source_root}", "result drifted")
    _require(subprocess_calls and subprocess_calls[0][-1] == str(source_root), "pip used the wrong source")
    _require(nemo_rl_patches.sys.path[0] == str(source_root), "new editable source remained invisible")


def test_patch_sha256_reports_patch_content_hash(tmp_path: Path) -> None:
    """The launcher should be able to log the exact maintained patch content."""
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("patch contents\n")

    assert nemo_rl_patches.patch_sha256(patch_file) == (
        "3e21aed045526cbe401bb21136236cf0b768acfb13d71101e953f78792549fa1"
    )


def test_vllm_patch_excludes_request_timing_telemetry() -> None:
    """Stable phase timers, not optional vLLM request metrics, own timing evidence."""
    patch_text = (nemo_rl_patches.RECIPE_ROOT / "patches" / "nemo-rl-evo2-vllm.patch").read_text()

    forbidden = (
        "generation_first_token_latency_s",
        "generation_decode_s",
        "vLLM request timing metrics are missing or inconsistent",
        "metrics.first_token_latency",
        "metrics.last_token_ts",
    )
    _require(
        all(value not in patch_text for value in forbidden),
        "diagnostic request timing telemetry entered the dependency patch",
    )


def test_patched_vllm_worker_normalizes_top_k_one_to_true_greedy() -> None:
    """The public RL adapter must use the same true-greedy policy as inference."""
    from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

    worker = object.__new__(BaseVllmGenerationWorker)
    worker.cfg = {
        "top_k": 1,
        "top_p": 0.75,
        "temperature": 1.0,
        "max_new_tokens": 500,
        "num_logprobs": 0,
        "stop_token_ids": None,
        "ignore_eos": True,
        "allowed_token_ids": [65, 67, 71, 78, 84],
    }
    worker.SamplingParams = lambda **kwargs: kwargs

    sampling_params = worker._build_sampling_params(
        greedy=False,
        stop_strings=None,
        request_seeds=[42, 43],
    )

    _require(len(sampling_params) == 2, "one SamplingParams row was not built per request")
    for expected_seed, params in zip((42, 43), sampling_params, strict=True):
        _require(params["temperature"] == 0.0, "top_k=1 did not select true greedy temperature")
        _require(params["top_p"] == 1.0, "greedy policy retained stochastic top-p filtering")
        _require(params["top_k"] == 0, "top_k=1 retained the top-k sampler route")
        _require(params["seed"] == expected_seed, "request seed propagation changed")
        _require(
            params["allowed_token_ids"] == [65, 67, 71, 78, 84],
            "greedy normalization changed the allowed DNA support",
        )


def test_patched_nemo_rl_training_logprobs_apply_allowed_support_before_top_k() -> None:
    from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams, apply_top_k_top_p

    params = TrainingSamplingParams(top_k=2, top_p=1.0, allowed_token_ids=[1, 3, 5])
    logits = torch.tensor(
        [[[100.0, 4.0, 99.0, 6.0, 98.0, 2.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    filtered, keep_mask = apply_top_k_top_p(
        logits,
        top_k=params.top_k,
        top_p=params.top_p,
        allowed_token_ids=params.allowed_token_ids,
    )

    expected_keep = torch.tensor([[[False, True, False, True, False, False]]])
    _require(keep_mask is not None, "allowed support returned no gradient mask")
    torch.testing.assert_close(keep_mask, expected_keep)
    _require(
        torch.equal(torch.isfinite(filtered), expected_keep),
        "top-k was not evaluated within the generation-time allowed support",
    )
    torch.log_softmax(filtered, dim=-1)[..., 1].sum().backward()
    _require(logits.grad is not None, "chosen logprob produced no gradient")
    _require(
        torch.equal(logits.grad.ne(0), expected_keep),
        "forbidden or filtered logits received a gradient",
    )


def test_sampling_patch_excludes_upstream_vllm_and_diagnostic_telemetry() -> None:
    patch_text = (nemo_rl_patches.RECIPE_ROOT / "patches" / "nemo-rl-evo2-sampling.patch").read_text()

    _require("diff --git a/vllm/" not in patch_text, "sampling parity patch modified upstream vLLM")
    for required in (
        'allowed_token_ids=generation_cfg.get("allowed_token_ids")',
        "allowed_token_ids=saved_sampling_params.allowed_token_ids",
    ):
        _require(required in patch_text, f"sampling policy propagation is missing: {required}")
    for forbidden in ("generation_first_token_latency_s", "generation_decode_s", "proof", "telemetry"):
        _require(forbidden not in patch_text, f"diagnostic field entered sampling patch: {forbidden}")


def test_maintained_patches_are_applied_to_pinned_nemo_rl_source() -> None:
    """The retained source should contain every maintained patch in declared order."""
    source_root = nemo_rl_patches._nemo_rl_source_dir()
    if not source_root.exists():
        pytest.skip("Recipe-owned pinned NeMo-RL source has not been built")

    for patch_path in nemo_rl_patches.DEFAULT_PATCHES:
        result = nemo_rl_patches._run_patch(
            ["--batch", "--dry-run", "-R", "-p1", "-i", str(patch_path)],
            cwd=source_root,
        )
        _require(result.returncode == 0, result.stdout)


def test_maintained_patch_inventory_is_narrow_and_does_not_patch_vllm_core() -> None:
    """Only recipe-required NeMo-RL seams belong in the dependency patches."""
    paths = set()
    for patch_path in nemo_rl_patches.DEFAULT_PATCHES:
        for line in patch_path.read_text().splitlines():
            if line.startswith("diff --git a/"):
                paths.add(line.split()[2].removeprefix("a/"))

    _require(
        paths
        == {
            "nemo_rl/distributed/worker_groups.py",
            "nemo_rl/algorithms/logits_sampling_utils.py",
            "nemo_rl/algorithms/grpo.py",
            "nemo_rl/distributed/model_utils.py",
            "nemo_rl/experience/rollouts.py",
            "nemo_rl/models/generation/interfaces.py",
            "nemo_rl/models/generation/vllm/config.py",
            "nemo_rl/models/generation/vllm/vllm_generation.py",
            "nemo_rl/models/generation/vllm/vllm_worker.py",
            "nemo_rl/models/megatron/setup.py",
            "nemo_rl/models/policy/lm_policy.py",
            "nemo_rl/models/policy/workers/megatron_policy_worker.py",
        },
        f"dependency patch inventory drifted: {sorted(paths)}",
    )
    _require(
        all(not path.startswith("vllm/") for path in paths),
        "the recipe patch must not modify upstream vLLM core",
    )


def test_assert_nemo_rl_patch_runtime_requires_reverse_patch_match(tmp_path: Path, monkeypatch) -> None:
    """Runtime verification should prove the installed package matches the maintained patch."""
    patch_file = tmp_path / "evo2.patch"
    patch_file.write_text("diff --git a/nemo_rl/example.py b/nemo_rl/example.py\n")
    source_root = tmp_path / "site-packages"
    package_dir = source_root / "nemo_rl"
    package_dir.mkdir(parents=True)
    init_file = package_dir / "__init__.py"
    init_file.write_text("")
    spec = importlib.util.spec_from_file_location("nemo_rl", init_file)

    monkeypatch.setattr(nemo_rl_patches.importlib.util, "find_spec", lambda name: spec)
    monkeypatch.setattr(nemo_rl_patches, "assert_nemo_rl_patch_symbols", lambda: None)

    calls: list[tuple[list[str], Path]] = []

    def fake_run_patch(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd))
        return subprocess.CompletedProcess(["patch", *args], 0, stdout="ok")

    monkeypatch.setattr(nemo_rl_patches, "_run_patch", fake_run_patch)

    nemo_rl_patches.assert_nemo_rl_patch_runtime(patch_file)

    assert calls == [(["--batch", "--dry-run", "-R", "-p1", "-i", str(patch_file.resolve())], source_root)]


def test_assert_nemo_rl_patch_symbols_accepts_expected_runtime_symbols(monkeypatch) -> None:
    """Startup should accept a runtime with all expected patched symbols."""
    megatron_setup = types.SimpleNamespace(
        _apply_target_allowlist_prefixes=object(),
        NoRefitMegatronBridge=object(),
        _uses_colocated_megatron_generation=object(),
        _select_megatron_bridge=object(),
    )
    modules = {
        "nemo_rl.algorithms.logits_sampling_utils": types.SimpleNamespace(
            _canonical_allowed_token_ids=object()
        ),
        "nemo_rl.models.generation.interfaces": types.SimpleNamespace(generation_prompt_token_ids_sha256=object()),
        "nemo_rl.models.megatron.setup": megatron_setup,
        "nemo_rl.models.generation.vllm.config": types.SimpleNamespace(VllmActorExecutionConfig=object()),
        "nemo_rl.models.generation.vllm.vllm_generation": types.SimpleNamespace(_request_seeds_for_dp_stream=object()),
    }

    monkeypatch.setattr(nemo_rl_patches.importlib, "import_module", lambda name: modules[name])

    nemo_rl_patches.assert_nemo_rl_patch_symbols()
