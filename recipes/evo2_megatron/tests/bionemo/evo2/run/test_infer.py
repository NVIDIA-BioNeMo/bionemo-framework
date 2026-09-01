# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

"""Tests for Evo2 text generation (inference) using MBridge.

infer.py drives generation through the NATIVE mcore dynamic-inference engine (paged-KV attention +
Hyena recurrent state packed into mcore's two Mamba slots), which is the only engine here.
The general generation tests below
(test_infer_runs, test_infer_temperature, test_infer_top_k, test_infer_phylogenetic_prompt,
test_identical_prompts_should_be_identical, test_subquadratic_ops_matches_baseline,
test_different_prompts_produce_different_outputs, test_different_results_with_without_peft,
the batch-padding prefix-invariance test, and the parallel-accuracy tests) all exercise this
engine; they assert "infer.py generates valid DNA" rather than any engine-specific internal.
The native dynamic tests add edge-case coverage (full-prompt multi-block prefill, opt-in
chunked prefill, single-token decode, longer generation, short-prompt right-aligned seed, TP=2
batch=1).

The core forward pass (predict.py) and HyenaInferenceContext are tested
in test_evo2.py which has working test_forward_manual and test_forward_ckpt_conversion.
"""

import contextlib
import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import bionemo.evo2.run.infer as infer_module
from bionemo.common.data.load import load as bionemo_load
from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.models.evo2_provider import HyenaInferenceContext
from bionemo.evo2.run.infer import (
    _native_stop_token_ids,
    _NativeDynamicResult,
    _result_to_jsonl_record,
    _sample_from_log_probs,
    _sampled_token_action,
    _sampling_log_probs_from_logits,
    _selected_log_probs_for_sampled_tokens,
    parse_args,
)
from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import savanna_to_mbridge

from ..utils import check_fp8_support


# Capture environment at import time (consistent with test_predict.py)
PRETEST_ENV = copy.deepcopy(os.environ)

# Note: mbridge_checkpoint_path fixture is provided by conftest.py at session scope


def _xfail_if_unsupported_subquadratic_ops(result: subprocess.CompletedProcess, use_subquadratic_ops: bool) -> None:
    if use_subquadratic_ops and "failed a CUDA self-test" in result.stderr:
        pytest.xfail("subquadratic_ops_torch CUDA kernels are unsupported in this environment")


def _read_jsonl_results(output_file: Path) -> list[dict]:
    """Read JSONL output file and return parsed records."""
    records = []
    with open(output_file) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


@pytest.mark.parametrize(
    ("requested_impl", "checkpoint_impl", "expected_enabled"),
    [
        ("local", "none", True),
        ("none", "local", False),
    ],
)
def test_configure_native_dynamic_cuda_graphs_normalizes_checkpoint_state(
    requested_impl, checkpoint_impl, expected_enabled
):
    """The CLI choice must replace stale graph settings loaded from a checkpoint."""
    provider = SimpleNamespace(
        cuda_graph_impl=checkpoint_impl,
        cuda_graph_scope=None,
        inference_cuda_graph_scope="none" if requested_impl == "local" else "layer",
    )

    enabled = infer_module._configure_native_dynamic_cuda_graphs(provider, rank=1, cuda_graph_impl=requested_impl)

    assert enabled is expected_enabled
    assert provider.cuda_graph_impl == requested_impl
    assert provider.inference_cuda_graph_scope is None
    assert provider.cuda_graph_scope == []


def test_graph_reset_clears_current_mcore_runner_cache(monkeypatch):
    stale_runner = object()
    manager = SimpleNamespace(
        cudagraph_runners=[stale_runner],
        custom_cudagraphs_lookup_table={(1,): stale_runner},
    )
    nd = SimpleNamespace(
        hyena_model=SimpleNamespace(
            modules=lambda: [SimpleNamespace(cudagraph_manager=manager)],
        )
    )
    delete_calls = []
    monkeypatch.setattr(
        "megatron.core.transformer.cuda_graphs.delete_cuda_graphs",
        lambda: delete_calls.append(True),
    )

    infer_module._reset_layer_cuda_graphs(nd)

    assert manager.cudagraph_runners == []
    assert manager.custom_cudagraphs_lookup_table == {}
    assert delete_calls == [True]


def test_batched_binding_rejects_permuted_contiguous_request_slots():
    from bionemo.evo2.models.evo2_provider import bind_hyena_packed_views_to_dynamic_context_batch

    with pytest.raises(ValueError, match="contiguous request slots"):
        bind_hyena_packed_views_to_dynamic_context_batch(None, None, request_slots=[2, 0, 1])


def test_packed_hyena_uses_local_pp_layer_map():
    from bionemo.evo2.models.evo2_provider import bind_hyena_packed_views_to_dynamic_context_batch

    layer_shapes = [
        SimpleNamespace(conv_owner_id=101, ssm_owner_id=201, ssm_shape=(2, 3), ssm_kind="inner_fir"),
        SimpleNamespace(conv_owner_id=102, ssm_owner_id=202, ssm_shape=(3, 2), ssm_kind="iir"),
    ]
    layers = [
        SimpleNamespace(
            layer_number=17 + local_idx,
            mixer=SimpleNamespace(hyena_state_shapes_per_request=lambda: None),
        )
        for local_idx in range(2)
    ]
    decoder = SimpleNamespace(
        layers=layers,
        hyena_state_shapes_per_request=lambda: ((4, 5), (3, 3), layer_shapes),
    )
    model = SimpleNamespace(decoder=decoder)
    dyn_ctx = SimpleNamespace(
        mamba_conv_states=torch.zeros(2, 1, 4, 5),
        mamba_ssm_states=torch.zeros(2, 1, 3, 3),
        layer_map={0: 0, 1: 1},
    )

    packed_states = bind_hyena_packed_views_to_dynamic_context_batch(model, dyn_ctx, request_slots=[0])
    packed_by_kind = {state._kind: state for state in packed_states}
    packed_by_kind["fir"][101] = torch.full((1, 4, 5), 1.0)
    packed_by_kind["fir"][102] = torch.full((1, 4, 5), 2.0)
    packed_by_kind["inner_fir"][201] = torch.full((1, 2, 3), 3.0)
    packed_by_kind["iir"][202] = torch.full((1, 3, 2), 4.0)

    assert packed_by_kind["fir"][101].data_ptr() == dyn_ctx.mamba_conv_states[0, 0].data_ptr()
    assert packed_by_kind["fir"][102].data_ptr() == dyn_ctx.mamba_conv_states[1, 0].data_ptr()
    assert packed_by_kind["inner_fir"][201].data_ptr() == dyn_ctx.mamba_ssm_states[0, 0, :2, :3].data_ptr()
    assert packed_by_kind["iir"][202].data_ptr() == dyn_ctx.mamba_ssm_states[1, 0, :3, :2].data_ptr()
    assert torch.all(dyn_ctx.mamba_conv_states[0] == 1.0)
    assert torch.all(dyn_ctx.mamba_conv_states[1] == 2.0)
    assert torch.all(dyn_ctx.mamba_ssm_states[0, :, :2, :3] == 3.0)
    assert torch.all(dyn_ctx.mamba_ssm_states[1, :, :3, :2] == 4.0)


def test_native_pp_forward_broadcasts_last_stage_logits(monkeypatch):
    class _FakePPGroup:
        @staticmethod
        def size():
            return 2

    pp_group = _FakePPGroup()
    dyn_ctx = SimpleNamespace(
        pipeline_parallel_group=pp_group,
        config=SimpleNamespace(materialize_only_last_token_logits=True),
        num_last_token_logits=2,
    )
    wrapper_inputs = []

    class _FakeInferenceWrapper:
        inference_context = dyn_ctx

        @staticmethod
        def run_one_forward_step(inference_input):
            wrapper_inputs.append(inference_input)
            return None

    class _UnexpectedDirectForward:
        def __call__(self, *_args, **_kwargs):
            pytest.fail("pipeline-parallel inference must use MCore's inference wrapper")

    expected_logits = torch.ones((1, 2, 4), dtype=torch.bfloat16)
    broadcast_args = []

    def _broadcast(size, dtype, tensor=None, pp_group=None):
        broadcast_args.append((size, dtype, tensor, pp_group))
        return expected_logits

    from megatron.core.inference import communication_utils

    monkeypatch.setattr(communication_utils, "broadcast_from_last_pipeline_stage", _broadcast)
    nd = SimpleNamespace(
        forward_model=_UnexpectedDirectForward(),
        hyena_model=SimpleNamespace(vocab_size=4, config=SimpleNamespace(params_dtype=torch.bfloat16)),
        inference_wrapper=_FakeInferenceWrapper(),
    )
    input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    position_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)

    logits = infer_module._forward_native_dynamic_logits(nd, dyn_ctx, input_ids, position_ids)

    assert logits is expected_logits
    assert wrapper_inputs == [{"tokens": input_ids, "position_ids": position_ids, "attention_mask": None}]
    assert broadcast_args == [([1, 2, 4], torch.bfloat16, None, pp_group)]


@pytest.mark.parametrize(
    ("slots", "expected"),
    [
        ([7, 6, 5], [5, 6, 7]),
        ([2, 0, 1], [2, 0, 1]),
        ([7, 5, 3], [7, 5, 3]),
        ([0, 1, 2], [0, 1, 2]),
    ],
)
def test_normalize_new_request_slots_only_reverses_mcore_lifo_order(slots, expected):
    slot_tensor = torch.tensor(slots, dtype=torch.int32)
    context = SimpleNamespace(mamba_metadata=SimpleNamespace(request_to_mamba_state_idx=slot_tensor))

    normalized = infer_module._normalize_new_request_slots_for_packed_hyena(context, len(slots))

    assert normalized.tolist() == expected
    assert context.mamba_metadata.request_to_mamba_state_idx.tolist() == expected


def test_native_stop_token_ids_resolves_eos_text_token():
    """The Evo2 tokenizer uses token id 0 / <EOS> to mark generation end."""

    class _FakeBackendTokenizer:
        @staticmethod
        def token_to_id(token: str) -> int | None:
            return {"<EOS>": 0}.get(token)

    class _FakeTokenizer:
        eos = "<EOS>"
        tokenizer = _FakeBackendTokenizer()

    assert _native_stop_token_ids(_FakeTokenizer()) == {0}


def test_simple_generation_activates_mcore_inference_mode():
    from megatron.core.inference.utils import InferenceMode

    from bionemo.evo2.run.infer_example_simple import generate_tokens_simple

    inference_context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=16)

    class _InferenceModeCheckingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, input_ids, **kwargs):
            assert InferenceMode.is_active()
            assert kwargs["inference_context"] is inference_context
            self.calls += 1
            logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
            logits[..., 1] = 1.0
            return logits

    model = _InferenceModeCheckingModel()
    generated_tokens = generate_tokens_simple(
        model,
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=2,
        top_k=1,
        inference_context=inference_context,
    )

    assert generated_tokens == [1, 1]
    assert model.calls == 3
    assert not InferenceMode.is_active()


def test_sampled_eos_is_omitted_without_stopping_when_ignore_eos_is_enabled():
    assert _sampled_token_action(0, {0}, ignore_eos=True) == (False, False)


def test_sampled_eos_stops_and_is_omitted_by_default():
    assert _sampled_token_action(0, {0}, ignore_eos=False) == (False, True)


def test_exact_generation_cli_flags_default_false_and_enable_when_passed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt"])
    defaults = parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["infer", "--ckpt-dir", "/tmp/ckpt", "--ignore-eos", "--strict-generation"],
    )
    enabled = parse_args()

    assert defaults.ignore_eos is False
    assert defaults.strict_generation is False
    assert enabled.ignore_eos is True
    assert enabled.strict_generation is True


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], None), (["--context-parallel-comm-type", "p2p"], "p2p"), (["--context-parallel-comm-type", "a2a"], "a2a")],
)
def test_infer_context_parallel_comm_type_cli(monkeypatch, extra_args, expected):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt", *extra_args])

    assert parse_args().context_parallel_comm_type == expected


def test_max_batch_size_help_describes_prompt_file_chunking(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["infer", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--max-batch-size MAX_BATCH_SIZE" in help_text
    assert "prompt-file rows per generate() call" in help_text
    assert "--evo2-batched-decode-size" not in help_text


def test_main_clamps_decode_concurrency_to_prompt_file_chunk_size(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--prompt",
            "A",
            "--prompt-batch-size",
            "4",
            "--max-batch-size",
            "2",
        ],
    )
    monkeypatch.setattr(infer_module, "infer", lambda **kwargs: captured.update(kwargs))

    infer_module.main()

    assert captured["max_batch_size"] == 2
    assert captured["evo2_batched_decode_size"] == 2


def test_result_to_jsonl_record_honors_explicit_stop_reason():
    """EOS-stopped native results should not be reclassified as length-finished."""
    result = _NativeDynamicResult(
        generated_text="ACGT",
        generated_length=4,
        prompt_tokens=[43, 126, 71, 65, 71, 84],
        finish_reason="stop",
    )

    record = _result_to_jsonl_record(
        request_id="seq",
        prompt="+~GAGT",
        result=result,
        max_new_tokens=4,
    )

    assert record["completion"] == "ACGT"
    assert record["finish_reason"] == "stop"
    assert record["usage"]["completion_tokens"] == 4


def test_result_to_jsonl_record_serializes_complete_benchmark_evidence():
    result = _NativeDynamicResult(
        generated_text="AC",
        generated_length=2,
        prompt_tokens=[43, 126],
        generated_tokens=[65, 67],
        generated_log_probs=[-0.1, -0.2],
        timings={"prefill_elapsed_s": 0.25, "decode_elapsed_s": 0.75},
        memory={
            "prefill_peak_allocated_bytes": 1024,
            "prefill_peak_reserved_bytes": 2048,
            "generation_peak_allocated_bytes": 4096,
            "generation_peak_reserved_bytes": 8192,
        },
    )

    record = _result_to_jsonl_record(
        request_id="seq",
        prompt="+~",
        result=result,
        max_new_tokens=2,
        return_log_probs=True,
    )

    assert record["prompt_token_ids"] == [43, 126]
    assert record["completion_token_ids"] == [65, 67]
    assert record["logprobs"]["completion_logprobs"] == [-0.1, -0.2]
    assert record["timings"]["prefill_elapsed_s"] == 0.25
    assert record["timings"]["decode_elapsed_s"] == 0.75
    assert record["memory"]["prefill_peak_allocated_bytes"] == 1024
    assert record["memory"]["prefill_peak_reserved_bytes"] == 2048
    assert record["memory"]["generation_peak_allocated_bytes"] == 4096
    assert record["memory"]["generation_peak_reserved_bytes"] == 8192


def test_top_level_infer_rejects_data_parallel_before_engine_setup(monkeypatch):
    """Standalone inference must not duplicate an unsharded prompt list across DP replicas."""
    monkeypatch.setattr(infer_module, "get_world_size_safe", lambda: 4)
    monkeypatch.setattr(
        infer_module,
        "setup_inference_engine",
        lambda **_kwargs: pytest.fail("DP validation must run before inference-engine setup"),
    )
    monkeypatch.setattr(infer_module, "_prune_caches", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)

    with pytest.raises(
        NotImplementedError,
        match=r"Top-level Evo2 inference does not yet support data parallelism.*world_size=4.*model_parallel_size=2",
    ):
        infer_module.infer(
            prompts=[{"id": "seq", "prompt": "A"}],
            ckpt_dir=Path("/tmp/ckpt"),
            tensor_parallel_size=2,
        )


@pytest.mark.parametrize(
    ("phase_evidence_enabled", "expected_synchronizations"),
    [
        (False, []),
        (True, ["setup", "setup"]),
    ],
)
def test_infer_reports_setup_elapsed_and_peak_memory(
    monkeypatch, caplog, phase_evidence_enabled, expected_synchronizations
):
    synchronizations = []
    gib = 1024**3
    components = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenize=lambda text: [ord(char) for char in text]),
        native_dynamic=SimpleNamespace(cuda_graphs_enabled=False),
    )
    native_result = _NativeDynamicResult(
        generated_text="A",
        generated_length=1,
        prompt_tokens=[65],
        generated_tokens=[65],
        memory={
            "total_peak_allocated_bytes": 4 * gib,
            "total_peak_reserved_bytes": 5 * gib,
        },
    )

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(infer_module, "_CUDA_PHASE_EVIDENCE_ENABLED", phase_evidence_enabled)
    monkeypatch.setattr(infer_module, "_prune_caches", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: synchronizations.append("setup"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 2 * gib)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 3 * gib)
    monkeypatch.setattr(infer_module, "setup_inference_engine", lambda **kwargs: components)
    monkeypatch.setattr(infer_module, "generate", lambda *args, **kwargs: [native_result])
    monkeypatch.setattr(infer_module, "_teardown_distributed_for_inference", lambda: None)
    caplog.set_level("INFO", logger=infer_module.logger.name)

    records = infer_module.infer(
        prompts=[{"id": "seq", "prompt": "A"}],
        ckpt_dir=Path("/tmp/ckpt"),
        max_new_tokens=1,
        max_seq_length=16,
    )

    assert synchronizations == expected_synchronizations
    assert "[MEMORY] After model setup: peak=2.000 GB, reserved=3.000 GB, engine_setup_elapsed_s=" in caplog.text
    assert components.native_dynamic.engine_setup_stats.performed is phase_evidence_enabled
    assert components.native_dynamic.engine_setup_stats.peak_allocated_bytes == 2 * gib
    assert components.native_dynamic.engine_setup_stats.peak_reserved_bytes == 3 * gib
    assert "[MEMORY] After generation: peak=4.000 GB, reserved=5.000 GB" in caplog.text
    assert records[0]["completion_token_ids"] == [65]


def test_non_primary_global_rank_does_not_write_results(monkeypatch, tmp_path):
    """Only distributed global rank zero may perform output-file side effects."""
    output_file = tmp_path / "results.jsonl"
    components = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenize=lambda text: [ord(char) for char in text]),
        native_dynamic=SimpleNamespace(cuda_graphs_enabled=False),
    )
    native_result = _NativeDynamicResult(
        generated_text="A",
        generated_length=1,
        prompt_tokens=[65],
        generated_tokens=[65],
    )

    # The initialized process group is authoritative; a stale launcher environment must not
    # make another process believe it owns the single shared output file.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(infer_module, "get_world_size_safe", lambda: 1)
    monkeypatch.setattr(infer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(infer_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(infer_module, "_prune_caches", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(infer_module, "setup_inference_engine", lambda **kwargs: components)
    monkeypatch.setattr(infer_module, "generate", lambda *args, **kwargs: [native_result])
    monkeypatch.setattr(infer_module, "_teardown_distributed_for_inference", lambda: None)

    records = infer_module.infer(
        prompts=[{"id": "seq", "prompt": "A"}],
        ckpt_dir=Path("/tmp/ckpt"),
        max_new_tokens=1,
        max_seq_length=16,
        output_file=output_file,
    )

    assert records[0]["completion_token_ids"] == [65]
    assert not output_file.exists()


def test_strict_streaming_nonfinite_late_failure_leaves_only_named_partial_artifact(monkeypatch, tmp_path):
    output_file = tmp_path / "audit.jsonl"
    partial_file = tmp_path / "audit.jsonl.partial"
    components = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenize=lambda text: [ord(char) for char in text]),
        native_dynamic=SimpleNamespace(cuda_graphs_enabled=False),
    )
    first_result = _NativeDynamicResult(
        generated_text="A",
        generated_length=1,
        prompt_tokens=[65],
        generated_tokens=[65],
        generated_log_probs=[-0.1],
    )

    def _fail_after_first_result(_components, *, result_callback, **_kwargs):
        result_callback(0, first_result)
        raise RuntimeError("Strict Evo2 generation returned a non-finite chosen-token log-prob")

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(infer_module, "_prune_caches", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(infer_module, "setup_inference_engine", lambda **kwargs: components)
    monkeypatch.setattr(infer_module, "generate", _fail_after_first_result)
    monkeypatch.setattr(infer_module, "_teardown_distributed_for_inference", lambda: None)

    with pytest.raises(RuntimeError, match="non-finite chosen-token log-prob"):
        infer_module.infer(
            prompts=[{"id": "first", "prompt": "A"}, {"id": "second", "prompt": "C"}],
            ckpt_dir=Path("/tmp/ckpt"),
            max_new_tokens=1,
            max_seq_length=16,
            max_batch_size=2,
            return_log_probs=True,
            strict_generation=True,
            output_file=output_file,
            stream_output=True,
        )

    assert not output_file.exists()
    assert partial_file.exists()
    assert [record["id"] for record in _read_jsonl_results(partial_file)] == ["first"]


def test_strict_streaming_atomically_promotes_complete_partial_artifact(monkeypatch, tmp_path):
    output_file = tmp_path / "audit.jsonl"
    partial_file = tmp_path / "audit.jsonl.partial"
    components = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenize=lambda text: [ord(char) for char in text]),
        native_dynamic=SimpleNamespace(cuda_graphs_enabled=False),
    )
    results = [
        _NativeDynamicResult(
            generated_text=token,
            generated_length=1,
            prompt_tokens=[ord(token)],
            generated_tokens=[ord(token)],
            generated_log_probs=[-0.1],
        )
        for token in ("A", "C")
    ]
    replacements = []
    real_replace = os.replace
    serialized_results = []
    serialize_record = infer_module._result_to_jsonl_record

    def _serialize_once(**kwargs):
        serialized_results.append(kwargs["result"])
        return serialize_record(**kwargs)

    def _generate_all(_components, *, result_callback, **_kwargs):
        for result_idx, result in enumerate(results):
            result_callback(result_idx, result)
        return results

    def _record_replace(source, destination):
        replacements.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(infer_module, "_prune_caches", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(infer_module, "setup_inference_engine", lambda **kwargs: components)
    monkeypatch.setattr(infer_module, "generate", _generate_all)
    monkeypatch.setattr(infer_module, "_teardown_distributed_for_inference", lambda: None)
    monkeypatch.setattr(infer_module.os, "replace", _record_replace)
    monkeypatch.setattr(infer_module, "_result_to_jsonl_record", _serialize_once)

    infer_module.infer(
        prompts=[{"id": "first", "prompt": "A"}, {"id": "second", "prompt": "C"}],
        ckpt_dir=Path("/tmp/ckpt"),
        max_new_tokens=1,
        max_seq_length=16,
        max_batch_size=2,
        return_log_probs=True,
        strict_generation=True,
        output_file=output_file,
        stream_output=True,
    )

    assert replacements == [(partial_file, output_file)]
    assert output_file.exists()
    assert not partial_file.exists()
    assert [record["id"] for record in _read_jsonl_results(output_file)] == ["first", "second"]
    assert len(serialized_results) == len(results)
    assert all(actual is expected for actual, expected in zip(serialized_results, results))


def test_sampling_log_probs_use_temperature_scaled_top_k_support():
    """Recorded generation log-probs should match the filtered distribution used to sample."""
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    log_probs = _sampling_log_probs_from_logits(logits, temperature=2.0, top_k=2, top_p=0.0)
    expected = torch.log_softmax(torch.tensor([[2.0, 1.5]], dtype=torch.float32), dim=-1)

    torch.testing.assert_close(log_probs[0, :2], expected[0])
    assert torch.isneginf(log_probs[0, 2])
    assert torch.isneginf(log_probs[0, 3])


def test_sample_from_log_probs_uses_prefiltered_distribution():
    """Native decode should sample from the log-probs it already computed."""
    logits = torch.tensor([[1.0, 5.0, 4.0]], dtype=torch.float32)
    log_probs = _sampling_log_probs_from_logits(logits, temperature=1.0, top_k=1, top_p=0.0)

    sampled = _sample_from_log_probs(log_probs, top_k=1, generator=torch.Generator())

    assert sampled.tolist() == [1]


def test_selected_log_probs_for_sampled_tokens_gathers_batch_once():
    """Batched decode should avoid one Python scalar sync per request."""
    log_probs = torch.log_softmax(
        torch.tensor([[1.0, 2.0, 3.0], [6.0, 5.0, 4.0]], dtype=torch.float32),
        dim=-1,
    )
    sampled_tokens = torch.tensor([2, 0], dtype=torch.long)

    selected = _selected_log_probs_for_sampled_tokens(log_probs, sampled_tokens)

    assert selected == pytest.approx([log_probs[0, 2].item(), log_probs[1, 0].item()])


class _MockLoopTokenizer:
    vocab_size = 4
    eos_token_id = 0

    @staticmethod
    def tokenize(text: str) -> list[int]:
        if text in {"<EOS>", "<EOD>", "<|endoftext|>"}:
            return [0]
        return [3]

    @staticmethod
    def detokenize(token_ids: list[int]) -> str:
        return "".join(str(token_id) for token_id in token_ids)


class _MockLoopForwardModel(torch.nn.Module):
    def __init__(self, error: Exception | None = None, events: list[str] | None = None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.error = error
        self.events = events
        self.calls = 0

    def forward(self, *_args, **_kwargs):
        self.calls += 1
        if self.events is not None:
            self.events.append("forward")
        if self.error is not None:
            raise self.error
        return torch.zeros(1)


class _MockNativeDynamicContext:
    def __init__(self, *, stop_after_updates: int | None = None, events: list[str] | None = None):
        self.max_tokens = 128
        self.max_sequence_length = 128
        self.mamba_metadata = SimpleNamespace(request_to_mamba_state_idx=torch.arange(32))
        self.evo2_batched_decode_enabled = False
        self.paused_request_count = 0
        self.chunked_prefill_request_id = -1
        self.stop_after_updates = stop_after_updates
        self.events = events
        self.request_count = 0
        self.update_count = 0
        self.reset_count = 0
        self.active = False

    def add_request(self, _request, *, prefill_chunk_length: int):
        assert prefill_chunk_length > 0
        self.request_count += 1
        self.active = True

    @staticmethod
    def initialize_attention_state():
        return None

    def current_input_and_position_ids(self):
        shape = (max(1, self.request_count), 1)
        return torch.zeros(shape, dtype=torch.long), torch.zeros(shape, dtype=torch.long)

    def update_requests(self, active_after_sample, _sampled_tokens):
        if self.events is not None:
            self.events.append("update")
        self.update_count += 1
        self.active = bool(active_after_sample.any().item())
        if self.stop_after_updates is not None and self.update_count >= self.stop_after_updates:
            self.active = False

    def has_unfinished_requests(self) -> bool:
        return self.active

    def reset(self):
        self.reset_count += 1
        self.active = False
        self.request_count = 0


def test_cuda_graph_warmup_captures_every_active_request_count(monkeypatch):
    class _WarmupContext(_MockNativeDynamicContext):
        def add_request(self, request, *, prefill_chunk_length: int):
            super().add_request(request, prefill_chunk_length=prefill_chunk_length)
            request_idx = self.request_count - 1
            self.mamba_metadata.request_to_mamba_state_idx[request_idx] = 9 - request_idx

    context = _WarmupContext()
    context.evo2_max_batched_decode_requests = 3
    bound_slots = []
    batched_decode_enabled_when_bound = []

    def _capture_bound_slots(_model, _context, *, request_slots):
        bound_slots.append(request_slots.tolist())
        batched_decode_enabled_when_bound.append(context.evo2_batched_decode_enabled)

    forward_model = _MockLoopForwardModel()
    batch_sizes = []
    forward_model.register_forward_pre_hook(lambda _module, args: batch_sizes.append(args[0].shape[0]))
    native_dynamic = SimpleNamespace(forward_model=forward_model, hyena_model=forward_model)
    monkeypatch.setattr(
        infer_module,
        "bind_hyena_packed_views_to_dynamic_context_batch",
        _capture_bound_slots,
    )

    infer_module._warmup_native_dynamic_cuda_graphs(
        native_dynamic,
        context,
        torch.device("cpu"),
    )

    assert forward_model.calls == 9
    assert batch_sizes == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert bound_slots == [[9], [8, 9], [7, 8, 9]]
    assert batched_decode_enabled_when_bound == [False, False, False]
    assert context.reset_count == 3
    assert context.evo2_batched_decode_enabled is False


def _run_mock_native_generation(
    monkeypatch,
    *,
    sampled_steps: list[list[int]],
    prompts: list[str] | None = None,
    max_new_tokens: int = 3,
    return_log_probs: bool = True,
    ignore_eos: bool = False,
    strict_generation: bool = True,
    evo2_batched_decode_size: int = 1,
    stop_after_updates: int | None = None,
    forward_error: Exception | None = None,
    events: list[str] | None = None,
    perf_counter_values: list[float] | None = None,
    peak_allocated_values: list[int] | None = None,
    peak_reserved_values: list[int] | None = None,
    expected_suppressed_token_ids: set[int] | None = None,
    context_max_tokens: int = 128,
):
    from megatron.core.inference.utils import InferenceMode

    context = _MockNativeDynamicContext(stop_after_updates=stop_after_updates, events=events)
    context.max_tokens = context_max_tokens
    forward_model = _MockLoopForwardModel(error=forward_error, events=events)
    native_dynamic = SimpleNamespace(
        forward_model=forward_model,
        hyena_model=forward_model,
        max_seq_length=128,
        max_seq_length_is_auto=False,
        sampling_rng=None,
        evo2_seed=17,
        cuda_graphs_enabled=False,
        generation_call_index=0,
        engine_setup_stats=infer_module._CudaPhaseStats(),
        engine_setup_stats_pending=True,
    )
    components = SimpleNamespace(tokenizer=_MockLoopTokenizer(), native_dynamic=native_dynamic)
    sampled_step_iter = iter(sampled_steps)

    def _sample_step(log_probs, **_kwargs):
        for token_id in expected_suppressed_token_ids or set():
            assert torch.isneginf(log_probs[:, token_id]).all()
        sampled = next(sampled_step_iter)
        assert len(sampled) == log_probs.shape[0]
        return torch.tensor(sampled, dtype=torch.long)

    monkeypatch.setattr(InferenceMode, "active", staticmethod(contextlib.nullcontext))
    monkeypatch.setattr(
        infer_module,
        "_get_or_build_shared_dynamic_context",
        lambda *_args, **_kwargs: (
            context,
            infer_module._CudaPhaseStats(),
            infer_module._CudaPhaseStats(),
        ),
    )
    monkeypatch.setattr(infer_module, "bind_hyena_packed_views_to_dynamic_context", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        infer_module,
        "bind_hyena_packed_views_to_dynamic_context_batch",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        infer_module,
        "_extract_generation_logits",
        lambda *_args, **_kwargs: torch.zeros((context.request_count, _MockLoopTokenizer.vocab_size)),
    )
    monkeypatch.setattr(infer_module, "_sample_from_log_probs", _sample_step)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync") if events is not None else None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    allocated_iter = iter(peak_allocated_values or [])
    reserved_iter = iter(peak_reserved_values or [])
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda: next(allocated_iter) if peak_allocated_values is not None else 0,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda: next(reserved_iter) if peak_reserved_values is not None else 0,
    )
    if perf_counter_values is not None:
        perf_counter_iter = iter(perf_counter_values)
        monkeypatch.setattr(infer_module.time, "perf_counter", lambda: next(perf_counter_iter))

    results = infer_module._generate_native_dynamic(
        components,
        prompts=prompts or ["P"],
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_k=0,
        top_p=0.0,
        return_log_probs=return_log_probs,
        ignore_eos=ignore_eos,
        strict_generation=strict_generation,
        enable_chunked_prefill=False,
        inference_dynamic_batching_max_tokens=None,
        inference_dynamic_batching_block_size=16,
        evo2_batched_decode_size=evo2_batched_decode_size,
        result_callback=None,
    )
    return results, context, forward_model


def test_ignore_eos_suppresses_stop_tokens_before_sampling(monkeypatch):
    results, _context, forward_model = _run_mock_native_generation(
        monkeypatch,
        sampled_steps=[[1], [2], [3]],
        ignore_eos=True,
        expected_suppressed_token_ids={0},
    )

    assert results[0].generated_tokens == [1, 2, 3]
    assert results[0].generated_log_probs == pytest.approx([-math.log(3)] * 3)
    assert forward_model.calls == 3


def test_native_single_loop_omits_ignored_eos_and_reaches_exact_length(monkeypatch):
    results, _context, forward_model = _run_mock_native_generation(
        monkeypatch,
        sampled_steps=[[0], [1], [2], [3]],
        ignore_eos=True,
    )

    assert results[0].generated_tokens == [1, 2, 3]
    assert results[0].generated_log_probs == pytest.approx([-math.log(3)] * 3)
    assert forward_model.calls == 4


def test_native_batched_loop_omits_ignored_eos_and_reaches_exact_length(monkeypatch):
    results, _context, forward_model = _run_mock_native_generation(
        monkeypatch,
        prompts=["P", "Q"],
        sampled_steps=[[0, 0], [1, 2], [2, 1], [3, 3]],
        ignore_eos=True,
        evo2_batched_decode_size=2,
    )

    assert [result.generated_tokens for result in results] == [[1, 2, 3], [2, 1, 3]]
    for result in results:
        assert result.generated_log_probs == pytest.approx([-math.log(3)] * 3)
        assert result.timings["timing_scope"] == "native_generation_group"
        assert result.timings["timing_group_id"] == "native-call-00000000-group-00000000"
        assert result.timings["timing_request_count"] == 2
    assert forward_model.calls == 4
    assert results[0].timings is not results[1].timings
    assert results[0].memory is not results[1].memory
    results[0].timings["first_result_only"] = True
    results[0].memory["first_result_only"] = 1
    assert "first_result_only" not in results[1].timings
    assert "first_result_only" not in results[1].memory


def test_native_batched_prefill_enforces_total_token_budget(monkeypatch):
    with pytest.raises(ValueError, match=r"Batched prefill requires 2 tokens.*max token budget is 1"):
        _run_mock_native_generation(
            monkeypatch,
            prompts=["P", "Q"],
            sampled_steps=[[1, 1]],
            max_new_tokens=1,
            evo2_batched_decode_size=2,
            context_max_tokens=1,
        )


def test_native_single_loop_strict_overflow_reraises(monkeypatch):
    from megatron.core.inference.contexts.dynamic_context import TokenOverflowError

    with pytest.raises(TokenOverflowError, match="forced overflow"):
        _run_mock_native_generation(
            monkeypatch,
            sampled_steps=[],
            forward_error=TokenOverflowError(0, "forced overflow"),
        )


def test_native_batched_loop_strict_error_does_not_fall_back(monkeypatch):
    with pytest.raises(RuntimeError, match="forced batched failure"):
        _run_mock_native_generation(
            monkeypatch,
            prompts=["P", "Q"],
            sampled_steps=[],
            evo2_batched_decode_size=2,
            forward_error=RuntimeError("forced batched failure"),
        )


def test_native_strict_loop_rejects_short_output(monkeypatch):
    with pytest.raises(RuntimeError, match=r"expected exactly 3 generated tokens.*got 1"):
        _run_mock_native_generation(
            monkeypatch,
            sampled_steps=[[1]],
            stop_after_updates=1,
        )


def test_native_strict_loop_accepts_short_output_stopped_by_eos(monkeypatch):
    results, _context, _forward_model = _run_mock_native_generation(
        monkeypatch,
        sampled_steps=[[1], [0]],
    )

    assert results[0].generated_tokens == [1]
    assert results[0].finish_reason == "stop"
    assert results[0].stopped_on_eos is True
    assert results[0].truncated is False


def test_native_strict_loop_rejects_token_logprob_mismatch(monkeypatch):
    original_result_type = infer_module._NativeDynamicResult

    def _mismatched_result(**kwargs):
        result = original_result_type(**kwargs)
        result.generated_log_probs = result.generated_log_probs[:-1]
        return result

    monkeypatch.setattr(infer_module, "_NativeDynamicResult", _mismatched_result)

    with pytest.raises(RuntimeError, match="mismatched token/log-prob lengths"):
        _run_mock_native_generation(
            monkeypatch,
            sampled_steps=[[1], [2], [3]],
        )


def test_native_strict_loop_rejects_requested_but_missing_logprobs(monkeypatch):
    original_result_type = infer_module._NativeDynamicResult

    def _missing_logprobs_result(**kwargs):
        result = original_result_type(**kwargs)
        result.generated_log_probs = None
        return result

    monkeypatch.setattr(infer_module, "_NativeDynamicResult", _missing_logprobs_result)

    with pytest.raises(RuntimeError, match="missing requested chosen-token log-probs"):
        _run_mock_native_generation(
            monkeypatch,
            sampled_steps=[[1], [2], [3]],
        )


@pytest.mark.parametrize(
    "nonfinite_logprob",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "positive_infinity", "negative_infinity"],
)
def test_native_strict_loop_rejects_nonfinite_chosen_logprobs(monkeypatch, nonfinite_logprob):
    monkeypatch.setattr(
        infer_module,
        "_selected_log_probs_for_sampled_tokens",
        lambda _log_probs, sampled_tokens: [nonfinite_logprob] * sampled_tokens.numel(),
    )

    with pytest.raises(RuntimeError, match=r"non-finite chosen-token log-prob.*prompt 0"):
        _run_mock_native_generation(
            monkeypatch,
            sampled_steps=[[1], [2], [3]],
        )


@pytest.mark.parametrize(
    ("prompts", "batched_decode_size", "sampled_steps"),
    [
        (["P"], 1, [[1], [2], [3]]),
        (["P", "Q"], 2, [[1, 1], [2, 2], [3, 3]]),
    ],
)
def test_native_loop_synchronizes_only_at_phase_boundaries(
    monkeypatch,
    prompts,
    batched_decode_size,
    sampled_steps,
):
    events = []
    monkeypatch.setattr(infer_module, "_CUDA_PHASE_EVIDENCE_ENABLED", True)

    results, _context, _forward_model = _run_mock_native_generation(
        monkeypatch,
        prompts=prompts,
        sampled_steps=sampled_steps,
        evo2_batched_decode_size=batched_decode_size,
        events=events,
        perf_counter_values=[10.0, 12.0, 17.0],
        peak_allocated_values=[101, 303],
        peak_reserved_values=[202, 404],
    )

    assert events == [
        "sync",
        "forward",
        "update",
        "sync",
        "forward",
        "update",
        "forward",
        "update",
        "sync",
    ]
    assert results[0].timings["prefill_elapsed_s"] == 2.0
    assert results[0].timings["decode_elapsed_s"] == 5.0
    assert results[0].timings["total_elapsed_s"] == 7.0
    assert results[0].timings["context_setup_elapsed_s"] == 0.0
    assert results[0].timings["cuda_graph_capture_elapsed_s"] == 0.0
    for result in results:
        assert result.memory["prefill_peak_allocated_bytes"] == 101
        assert result.memory["prefill_peak_reserved_bytes"] == 202
        assert result.memory["decode_peak_allocated_bytes"] == 303
        assert result.memory["decode_peak_reserved_bytes"] == 404
        assert result.memory["generation_peak_allocated_bytes"] == 303
        assert result.memory["generation_peak_reserved_bytes"] == 404
        assert result.memory["total_peak_allocated_bytes"] == 303
        assert result.memory["total_peak_reserved_bytes"] == 404


def test_shared_dynamic_context_reports_cold_setup_and_reuses_larger_request_capacity(monkeypatch):
    events = []
    monkeypatch.setattr(infer_module, "_CUDA_PHASE_EVIDENCE_ENABLED", True)

    class _BuildContext:
        def __init__(self, *, model_config, inference_config):
            assert model_config.tensor_model_parallel_size == 1
            self.max_sequence_length = inference_config.max_sequence_length
            self.max_tokens = inference_config.max_tokens or inference_config.max_sequence_length
            self.max_requests = inference_config.max_requests
            self.reset_count = 0

        def initialize_all_tensors(self):
            events.append("initialize")

        def reset(self):
            self.reset_count += 1
            events.append("reset")

    nd = SimpleNamespace(
        shared_dyn_ctx=None,
        shared_dyn_ctx_key=None,
        cuda_graphs_enabled=True,
        hyena_model=SimpleNamespace(config=SimpleNamespace(tensor_model_parallel_size=1)),
        mamba_state_config=object(),
        max_seq_length=64,
        ctx_cls=_BuildContext,
    )
    perf_counter_values = iter([10.0, 12.0, 17.0])
    allocated_values = iter([101, 303])
    reserved_values = iter([202, 404])

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: events.append("reset_peak"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: next(allocated_values))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: next(reserved_values))
    monkeypatch.setattr(infer_module.time, "perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(infer_module, "compute_evo2_paged_kv_buffer_size_gb", lambda *_args, **_kwargs: 0.01)
    monkeypatch.setattr(
        infer_module,
        "_warmup_native_dynamic_cuda_graphs",
        lambda *_args, **_kwargs: events.append("capture"),
    )

    context, context_setup, graph_capture = infer_module._get_or_build_shared_dynamic_context(
        nd,
        block_size_tokens=16,
        max_tokens=64,
        enable_chunked_prefill=False,
        max_active_requests=2,
        device=torch.device("cpu"),
    )
    warm_context, warm_context_setup, warm_graph_capture = infer_module._get_or_build_shared_dynamic_context(
        nd,
        block_size_tokens=16,
        max_tokens=64,
        enable_chunked_prefill=False,
        max_active_requests=1,
        device=torch.device("cpu"),
    )

    assert warm_context is context
    assert warm_context.max_requests == 2
    assert context_setup == infer_module._CudaPhaseStats(
        elapsed_s=2.0,
        peak_allocated_bytes=101,
        peak_reserved_bytes=202,
        performed=True,
    )
    assert graph_capture == infer_module._CudaPhaseStats(
        elapsed_s=5.0,
        peak_allocated_bytes=303,
        peak_reserved_bytes=404,
        performed=True,
    )
    assert warm_context_setup == infer_module._CudaPhaseStats()
    assert warm_graph_capture == infer_module._CudaPhaseStats()
    assert events == [
        "sync",
        "reset_peak",
        "initialize",
        "sync",
        "reset_peak",
        "capture",
        "sync",
        "reset",
    ]


def test_infer_runs(mbridge_checkpoint_path, tmp_path):
    """Test that infer.py runs without errors and produces JSONL output."""
    output_file = tmp_path / "output.jsonl"

    # Use a longer DNA prompt to meet FP8 dimension requirements (divisible by 8)
    # 64 characters should be safe
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "10",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",  # Non-zero temperature required by MCore
        "--top-k",
        "1",  # Top-k=1 for greedy decoding
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) == 1, f"Expected 1 result, got {len(records)}"
    record = records[0]
    assert record["id"] == "0"
    assert record["prompt"] == prompt
    assert len(record["completion"]) > 0, "Generated text is empty"
    assert record["finish_reason"] in ("length", "stop")
    assert "usage" in record
    assert record["usage"]["prompt_tokens"] > 0
    assert record["usage"]["completion_tokens"] > 0


@pytest.mark.parametrize("temperature", [0.5, 1.0])
def test_infer_temperature(mbridge_checkpoint_path, tmp_path, temperature):
    """Test that different temperatures produce output."""
    output_file = tmp_path / f"output_temp_{temperature}.jsonl"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "5",
        "--temperature",
        str(temperature),
        "--output-file",
        str(output_file),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_top_k(mbridge_checkpoint_path, tmp_path):
    """Test top-k sampling."""
    output_file = tmp_path / "output_topk.jsonl"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "5",
        "--top-k",
        "4",  # Only sample from top 4 tokens (A, C, G, T)
        "--output-file",
        str(output_file),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_phylogenetic_prompt(mbridge_checkpoint_path, tmp_path):
    """Test generation with a phylogenetic lineage prompt.

    Evo2 is trained with phylogenetic tags, so generation should work
    well when conditioned on these tags. Using a longer prompt for FP8.
    """
    output_file = tmp_path / "output_phylo.jsonl"

    # Phylogenetic prompt (padded to be longer for FP8 compatibility)
    prompt = (
        "|d__Bacteria;"
        "p__Pseudomonadota;"
        "c__Gammaproteobacteria;"
        "o__Enterobacterales;"
        "f__Enterobacteriaceae;"
        "g__Escherichia;"
        "s__Escherichia|"
    )
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "20",
        "--temperature",
        "1.0",  # Non-zero temperature required by MCore
        "--top-k",
        "1",  # Top-k=1 for greedy decoding
        "--output-file",
        str(output_file),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) == 1
    assert len(records[0]["completion"]) > 0, "Generated text is empty"


# DNA prompts for reproducibility tests (from test_prompt.py)
PROMPT_1 = "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGGGAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGG"
PROMPT_2 = "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTT"


def run_infer_subprocess(
    mbridge_checkpoint_path,
    prompt: str,
    output_file,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    use_subquadratic_ops: bool = False,
    cuda_graph_impl: str | None = None,
    max_seq_length: int | None = None,
    block_size_tokens: int | None = None,
    return_log_probs: bool = False,
    extra_args: list[str] | None = None,
):
    """Helper function to run inference as a subprocess.

    Generation runs through the native mcore dynamic-inference engine (the only engine: paged-KV
    attention + Hyena state in mcore Mamba slots).

    Args:
        mbridge_checkpoint_path: Path to the MBridge checkpoint
        prompt: Input prompt for the model
        output_file: Path to write output (JSONL)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (1 for greedy)
        seed: Random seed for reproducibility
        use_subquadratic_ops: Pass --use-subquadratic-ops to the CLI.
        cuda_graph_impl: If set, pass --cuda-graph-impl ("local" = mcore per-layer decode graphs,
            "none" = eager decode). Defaults to the CLI default ("local") when None.
        max_seq_length: If set, pass --max-seq-length (caps the per-context allocation).
        block_size_tokens: If set, pass --inference-dynamic-batching-block-size (paged-KV block size).
            The CLI default is 256; pin it explicitly when a test depends on the block boundary.
        return_log_probs: Pass --return-log-probs (logprobs included in the JSONL record).
        extra_args: Additional CLI arguments appended to the infer command.

    Returns:
        The single JSONL result record (dict) for the prompt.
    """
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-file",
        str(output_file),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
    ]
    if use_subquadratic_ops:
        cmd.append("--use-subquadratic-ops")
    if cuda_graph_impl is not None:
        cmd.extend(["--cuda-graph-impl", str(cuda_graph_impl)])
    if max_seq_length is not None:
        cmd.extend(["--max-seq-length", str(max_seq_length)])
    if block_size_tokens is not None:
        cmd.extend(["--inference-dynamic-batching-block-size", str(block_size_tokens)])
    if return_log_probs:
        cmd.append("--return-log-probs")
    if extra_args:
        cmd.extend(extra_args)

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    _xfail_if_unsupported_subquadratic_ops(result, use_subquadratic_ops)
    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) == 1, f"Expected 1 JSONL record, got {len(records)}"
    return records[0]


def mid_point_split(*, seq, num_tokens: int | None = None, fraction: float = 0.5):
    """Split a sequence at a midpoint for prompt/target evaluation."""
    mid_point = int(fraction * len(seq))
    prompt = seq[:mid_point]
    if num_tokens is not None:
        target = seq[mid_point : mid_point + num_tokens]
    else:
        target = seq[mid_point:]
    return prompt, target


def calculate_sequence_identity(seq1: str, seq2: str) -> float | None:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))
    return (matches / min_length) * 100


def _recipe_root() -> Path:
    """Return the recipe root directory (evo2_megatron/)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def _infer_script_path() -> Path:
    """Return the path to the source infer.py script.

    Uses the source version directly (rather than the installed module via ``-m``)
    so that local fixes to infer.py are picked up without reinstalling the package.
    """
    return _recipe_root() / "src" / "bionemo" / "evo2" / "run" / "infer.py"


def _write_prompts_jsonl(prompt_file: Path, prompts: list[tuple[str, str]]) -> None:
    """Write a list of (id, prompt) pairs into a JSONL file."""
    with open(prompt_file, "w") as f:
        f.writelines(json.dumps({"id": prompt_id, "prompt": prompt_text}) + "\n" for prompt_id, prompt_text in prompts)


@pytest.fixture(
    params=[False, True],
    ids=["causal-conv1d", "subquadratic-ops"],
)
def infer_use_subquadratic_ops(request):
    """Whether infer should use subquadratic Hyena kernels."""
    return request.param


def _run_infer_prompt_file(
    *,
    mbridge_checkpoint_path: Path,
    prompt_file: Path,
    output_file: Path,
    max_batch_size: int,
    use_subquadratic_ops: bool,
) -> dict[str, dict]:
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        "1",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",
        "--top-k",
        "1",
        "--seed",
        "1234",
        "--max-batch-size",
        str(max_batch_size),
        "--max-seq-length",
        "512",
        "--return-log-probs",
    ]
    if use_subquadratic_ops:
        cmd.append("--use-subquadratic-ops")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=512,
        env=copy.deepcopy(PRETEST_ENV),
    )
    _xfail_if_unsupported_subquadratic_ops(result, use_subquadratic_ops)
    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    records = _read_jsonl_results(output_file)
    return {record["id"]: record for record in records}


def _completion_logprobs(record: dict) -> torch.Tensor:
    logprobs = record.get("logprobs", {}).get("completion_logprobs")
    assert logprobs is not None, f"Missing completion logprobs in record: {record}"
    tensor = torch.as_tensor(logprobs, dtype=torch.float32).flatten()
    assert tensor.numel() == 1
    return tensor


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_infer_evo2_short_prefill_is_prefix_invariant_across_batch_padding(
    mbridge_checkpoint_path,
    tmp_path,
    infer_use_subquadratic_ops: bool,
):
    """A short prefill should generate the same next token alone or in a padded batch.

    Routes through the native default engine. Native decodes each prompt as its own single-request
    context (no static batch padding), so the short prompt's completion + logprob must match whether
    it is submitted alone or alongside a longer prompt — the same "infer.py generates valid,
    batch-independent DNA" invariant, now exercised on the working path.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Inference prefill prefix-invariance test requires a GPU")

    short_prompt = "ACGTACGTAA"
    padding_prompt = ("GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTC" * 4)[:256]

    alone_prompt_file = tmp_path / "short_alone_prompts.jsonl"
    padded_prompt_file = tmp_path / "short_padded_prompts.jsonl"
    _write_prompts_jsonl(alone_prompt_file, [("short", short_prompt)])
    _write_prompts_jsonl(padded_prompt_file, [("padding", padding_prompt), ("short", short_prompt)])

    alone_records = _run_infer_prompt_file(
        mbridge_checkpoint_path=mbridge_checkpoint_path,
        prompt_file=alone_prompt_file,
        output_file=tmp_path / "alone_output.jsonl",
        max_batch_size=1,
        use_subquadratic_ops=infer_use_subquadratic_ops,
    )
    padded_records = _run_infer_prompt_file(
        mbridge_checkpoint_path=mbridge_checkpoint_path,
        prompt_file=padded_prompt_file,
        output_file=tmp_path / "padded_output.jsonl",
        max_batch_size=2,
        use_subquadratic_ops=infer_use_subquadratic_ops,
    )

    assert set(alone_records) == {"short"}
    assert set(padded_records) == {"padding", "short"}
    assert padded_records["short"]["prompt"] == short_prompt
    assert alone_records["short"]["completion"] == padded_records["short"]["completion"]

    torch.testing.assert_close(
        _completion_logprobs(alone_records["short"]),
        _completion_logprobs(padded_records["short"]),
        rtol=2e-2,
        atol=5e-2,
    )


def run_infer_subprocess_parallel(
    mbridge_checkpoint_path,
    prompt_file: Path,
    output_file: Path,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    max_batch_size: int | None = None,
    evo2_batched_decode_size: int | None = None,
    cuda_graph_impl: str | None = None,
    expected_log_substrings: tuple[str, ...] = (),
) -> list[dict]:
    """Run inference as a subprocess with model parallelism.

    Runs the source infer.py script directly (not the installed module) so that
    local fixes are picked up without reinstalling the package.  The caller is
    responsible for writing the JSONL prompt file beforehand.

    Args:
        mbridge_checkpoint_path: Path to the MBridge checkpoint.
        prompt_file: Path to an existing JSONL prompt file.
        output_file: Path to write JSONL output.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter (1 for greedy).
        seed: Random seed for reproducibility.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        max_batch_size: If set, pass --max-batch-size to the CLI.
        evo2_batched_decode_size: If set, pass --evo2-batched-decode-size to the CLI.
        cuda_graph_impl: If set, pass --cuda-graph-impl.
        expected_log_substrings: Strings that must appear in stdout or stderr.

    Returns:
        List of parsed JSONL result dicts.
    """
    nproc_per_node = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(nproc_per_node),
        "--nnodes",
        "1",
        str(_infer_script_path()),
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-file",
        str(output_file),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_model_parallel_size),
        "--context-parallel-size",
        str(context_parallel_size),
    ]
    if max_batch_size is not None:
        cmd.extend(["--max-batch-size", str(max_batch_size)])
    if evo2_batched_decode_size is not None:
        cmd.extend(["--evo2-batched-decode-size", str(evo2_batched_decode_size)])
    if cuda_graph_impl is not None:
        cmd.extend(["--cuda-graph-impl", str(cuda_graph_impl)])

    env = copy.deepcopy(PRETEST_ENV)
    # Prepend the source src/ directory to PYTHONPATH so that local model code
    # (hyena_mixer.py, hyena_utils.py, etc.) is used instead of the installed package.
    src_dir = str(_recipe_root() / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,  # 15 minutes for parallel configs
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    combined_output = f"{result.stdout}\n{result.stderr}"
    for substring in expected_log_substrings:
        assert substring in combined_output, (
            f"Expected infer output to contain {substring!r}.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    assert output_file.exists(), "Output file was not created"

    return _read_jsonl_results(output_file)


def test_identical_prompts_should_be_identical(mbridge_checkpoint_path, tmp_path):
    """Test that identical prompts produce identical sequences.

    With greedy decoding (top_k=1) and the same seed, identical prompts
    should produce identical outputs.
    """
    output_file_1 = tmp_path / "output_prompt1_run1.jsonl"
    output_file_2 = tmp_path / "output_prompt1_run2.jsonl"

    # Run inference twice with the same prompt
    generated_1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_1,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding for determinism
        seed=42,
    )

    generated_2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding for determinism
        seed=42,
    )

    assert len(generated_1) > 0, "First generation produced empty output"
    assert len(generated_2) > 0, "Second generation produced empty output"
    assert generated_1["completion"] == generated_2["completion"]
    assert generated_1["completion_token_ids"] == generated_2["completion_token_ids"]


@pytest.mark.parametrize("cuda_graph_impl", ["none", "local"])
@pytest.mark.parametrize("use_subquadratic_ops", [False, True])
def test_subquadratic_ops_with_cuda_graph_matches_baseline(
    mbridge_checkpoint_path, tmp_path, use_subquadratic_ops, cuda_graph_impl
):
    """Every (subq-ops x CUDA-graph) combination matches the eager, non-subq baseline.

    The reference is the simplest path: standard kernels with CUDA graphs OFF (``cuda_graph_impl=none``).
    Greedy decoding (top_k=1) + a fixed seed make generation deterministic, so each of the four
    combinations of {standard, subq-ops} x {eager, local CUDA graphs} must produce byte-identical output.

    subquadratic-ops kernels cannot be captured into a CUDA graph (they SIGSEGV during capture), so
    ``setup_inference_engine`` makes them mutually exclusive: requesting both forces eager decode
    (``cuda_graph_impl='none'``) with a warning. Hence the ``[True, 'local']`` case runs subq-ops
    eagerly rather than crashing, and must still match the baseline. The subq path uses guarded
    kernels: if this GPU cannot run them, ``run_infer_subprocess`` xfails (via the CUDA self-test
    guard) instead of producing invalid output.
    """
    baseline = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "output_baseline.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        use_subquadratic_ops=False,
        cuda_graph_impl="none",
    )
    variant = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "output_variant.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        use_subquadratic_ops=use_subquadratic_ops,
        cuda_graph_impl=cuda_graph_impl,
    )

    assert baseline["completion"], "Baseline generation produced empty output"
    assert variant["completion"], "Variant generation produced empty output"
    assert variant["completion"] == baseline["completion"], (
        f"subq_ops={use_subquadratic_ops}, cuda_graph_impl={cuda_graph_impl} diverged from the "
        f"eager non-subq baseline:\n  baseline={baseline['completion']!r}\n  variant ={variant['completion']!r}"
    )


def test_different_prompts_produce_different_outputs(mbridge_checkpoint_path, tmp_path):
    """Test that different prompts produce different sequences.

    Different input prompts should produce different outputs, demonstrating
    that the model is actually responding to the prompt content.
    """
    output_file_1 = tmp_path / "output_prompt1.jsonl"
    output_file_2 = tmp_path / "output_prompt2.jsonl"

    # Run inference with two different prompts
    generated_1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_1,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
    )

    generated_2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_2,
        output_file=output_file_2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
    )

    assert len(generated_1) > 0, "First generation produced empty output"
    assert len(generated_2) > 0, "Second generation produced empty output"

    # The outputs should be different since the prompts are different
    # We check that the generated portions (after the prompt) are not identical
    assert generated_1 != generated_2, (
        f"Different prompts produced identical outputs:\n"
        f"Prompt 1 output: {generated_1}\n"
        f"Prompt 2 output: {generated_2}"
    )


@pytest.fixture
def dna_sequences():
    """Load DNA sequences from prompts.csv test data."""
    prompts_csv = Path(__file__).resolve().parent.parent / "data" / "prompts.csv"
    with prompts_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row["Sequence"] for row in reader]


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp, cp",
    [
        # The 1b model only supports TP=1 through infer.py due to divisibility constraints
        # (15 attention heads and 128-width HyenaMixer). TP>1 requires the 7b model.
        pytest.param(1, 1, id="tp=1,cp=1"),
        pytest.param(
            1,
            2,
            id="tp=1,cp=2",
            marks=pytest.mark.xfail(reason="CP>1 is known broken for inference", strict=False),
        ),
    ],
)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy(mbridge_checkpoint_path, tmp_path, dna_sequences, tp, cp):
    """Test that parallel inference produces accurate generation results.

    Loads real DNA sequences, splits them in half, generates 500 tokens from the first half,
    and compares the generated tokens against the known second half using sequence identity.
    This mirrors the pattern in test_batch_generate_mbridge in test_evo2.py but exercises
    the subprocess-based infer.py CLI with parallelism.
    """
    num_gpus_required = tp * cp
    if torch.cuda.device_count() < num_gpus_required:
        pytest.skip(f"Not enough GPUs: need {num_gpus_required}, have {torch.cuda.device_count()}")

    num_tokens = 500
    # Expected sequence identity percentages for the 1b-8k-bf16 checkpoint (from test_evo2.py)
    expected_matchpercents = [96.8, 29.7, 76.6, 71.6]

    # Build a single JSONL prompt file with all sequences, keyed by id
    targets_by_id: dict[str, str] = {}
    expected_by_id: dict[str, float] = {}
    jsonl_entries = []
    for i, (seq, expected_mp) in enumerate(zip(dna_sequences, expected_matchpercents)):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)
        seq_id = f"seq_{i}"
        targets_by_id[seq_id] = target
        expected_by_id[seq_id] = expected_mp
        jsonl_entries.append((seq_id, prompt))

    prompt_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "outputs.jsonl"
    _write_prompts_jsonl(prompt_file, jsonl_entries)

    # Single inference call processes all prompts (batching handled internally)
    records = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=output_file,
        max_new_tokens=num_tokens,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
        tensor_parallel_size=tp,
        context_parallel_size=cp,
    )

    assert len(records) == len(dna_sequences), f"Expected {len(dna_sequences)} results, got {len(records)}"

    # Match results by id (output order is not guaranteed with dynamic engines)
    results_by_id = {r["id"]: r for r in records}
    match_percents = {}
    for seq_id, target in targets_by_id.items():
        assert seq_id in results_by_id, f"Missing result for {seq_id}"
        identity = calculate_sequence_identity(target, results_by_id[seq_id]["completion"])
        match_percents[seq_id] = identity

    matchperc_print = {k: f"{v:.2f}%" for k, v in match_percents.items()}
    matchperc_print_expected = {k: f"{v:.2f}%" for k, v in expected_by_id.items()}

    assert all(match_percents[sid] >= 0.90 * expected_by_id[sid] for sid in targets_by_id), (
        f"Expected at least 90% of {matchperc_print_expected}, got {matchperc_print}"
    )


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy_evo2_batched_decode_same_prefix_preserves_accuracy(
    mbridge_checkpoint_path,
    tmp_path,
    dna_sequences,
):
    """Same-prefix batched decode may diverge from serial, but should preserve target accuracy.

    The true Evo2 batched-decode path only accepts same-length prompts, so this uses a common prefix
    length across the DNA accuracy prompts for both serial and batched subprocess inference. Greedy
    serial-vs-batched completions can diverge after small numerical differences; when they do, the
    batched completion should remain similarly close to the real next-window target.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Batched decode inference accuracy test requires a GPU")

    num_tokens = 500
    prompt_len = min(len(seq) // 2 for seq in dna_sequences)
    prompt_len = min(prompt_len, 2048)
    batch_size = len(dna_sequences)

    targets_by_id: dict[str, str] = {}
    jsonl_entries = []
    for i, seq in enumerate(dna_sequences):
        seq_id = f"seq_{i}"
        targets_by_id[seq_id] = seq[prompt_len : prompt_len + num_tokens]
        jsonl_entries.append((seq_id, seq[:prompt_len]))

    serial_prompt_file = tmp_path / "serial_prompts.jsonl"
    serial_output_file = tmp_path / "serial_outputs.jsonl"
    batched_prompt_file = tmp_path / "batched_prompts.jsonl"
    batched_output_file = tmp_path / "batched_outputs.jsonl"
    _write_prompts_jsonl(serial_prompt_file, jsonl_entries)
    _write_prompts_jsonl(batched_prompt_file, jsonl_entries)

    serial_records = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=serial_prompt_file,
        output_file=serial_output_file,
        max_new_tokens=num_tokens,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_batch_size=1,
        evo2_batched_decode_size=1,
    )
    batched_records = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=batched_prompt_file,
        output_file=batched_output_file,
        max_new_tokens=num_tokens,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_batch_size=batch_size,
        evo2_batched_decode_size=batch_size,
        expected_log_substrings=(
            f"[evo2-native] opt-in batched decode active: size={batch_size}",
            f"[evo2-native] batched prompt prefill: requests={batch_size}",
        ),
    )

    serial_by_id = {r["id"]: r for r in serial_records}
    batched_by_id = {r["id"]: r for r in batched_records}
    assert set(serial_by_id) == set(batched_by_id) == set(targets_by_id)

    serial_match_percents: dict[str, float] = {}
    batched_match_percents: dict[str, float] = {}
    for seq_id, target in targets_by_id.items():
        serial_identity = calculate_sequence_identity(target, serial_by_id[seq_id]["completion"]) or 0.0
        batched_identity = calculate_sequence_identity(target, batched_by_id[seq_id]["completion"]) or 0.0
        serial_match_percents[seq_id] = serial_identity
        batched_match_percents[seq_id] = batched_identity

    serial_vs_batched_percents = {
        seq_id: calculate_sequence_identity(serial_by_id[seq_id]["completion"], batched_by_id[seq_id]["completion"])
        or 0.0
        for seq_id in targets_by_id
    }
    first_diffs = {
        seq_id: next(
            (
                idx
                for idx, (serial_base, batched_base) in enumerate(
                    zip(serial_by_id[seq_id]["completion"], batched_by_id[seq_id]["completion"])
                )
                if serial_base != batched_base
            ),
            None,
        )
        for seq_id in targets_by_id
    }

    exact_matches = {
        seq_id: serial_by_id[seq_id]["completion"] == batched_by_id[seq_id]["completion"] for seq_id in targets_by_id
    }

    def _max_homopolymer(sequence: str) -> int:
        best = 0
        current = 0
        previous = None
        for base in sequence:
            current = current + 1 if base == previous else 1
            best = max(best, current)
            previous = base
        return best

    batched_completion_stats = {
        seq_id: {
            "length": len(batched_by_id[seq_id]["completion"]),
            "valid_dna": set(batched_by_id[seq_id]["completion"]) <= {"A", "C", "G", "T", "N"},
            "max_homopolymer": _max_homopolymer(batched_by_id[seq_id]["completion"]),
        }
        for seq_id in targets_by_id
    }

    serial_match_print = {k: f"{v:.2f}%" for k, v in serial_match_percents.items()}
    batched_match_print = {k: f"{v:.2f}%" for k, v in batched_match_percents.items()}
    serial_vs_batched_print = {k: f"{v:.2f}%" for k, v in serial_vs_batched_percents.items()}
    exact_match_print = {k: str(v) for k, v in exact_matches.items()}
    assert all(stat["length"] == num_tokens and stat["valid_dna"] for stat in batched_completion_stats.values()), (
        f"Expected full-length DNA completions from batched decode, got {batched_completion_stats=}"
    )
    assert all(stat["max_homopolymer"] <= 20 for stat in batched_completion_stats.values()), (
        f"Expected non-degenerate batched DNA completions, got {batched_completion_stats=}"
    )
    assert all(batched_match_percents[sid] >= serial_match_percents[sid] - 5.0 for sid in targets_by_id), (
        "Expected batched decode to stay within 5 identity points of same-prefix serial target "
        f"accuracy, got {serial_match_print=}, {batched_match_print=}, "
        f"{serial_vs_batched_print=}, {exact_match_print=}, and {first_diffs=}"
    )


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_1m_path(tmp_path_factory) -> Path:
    """Create or load a MBridge checkpoint for 7b-1m model testing."""
    try:
        nemo2_checkpoint_path = bionemo_load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_checkpoint_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="evo2_7b",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp, pp, cp",
    [
        # The 7b model has 32 attention heads, supporting TP=1, 2, 4, 8
        # TP-only configs
        pytest.param(1, 1, 1, id="tp=1,pp=1,cp=1"),
        pytest.param(2, 1, 1, id="tp=2,pp=1,cp=1"),
        pytest.param(4, 1, 1, id="tp=4,pp=1,cp=1"),
        pytest.param(8, 1, 1, id="tp=8,pp=1,cp=1"),
        # PP-only configs
        pytest.param(1, 2, 1, id="tp=1,pp=2,cp=1"),
        pytest.param(1, 4, 1, id="tp=1,pp=4,cp=1"),
        pytest.param(1, 8, 1, id="tp=1,pp=8,cp=1"),
        # Combined TP+PP configs
        pytest.param(2, 2, 1, id="tp=2,pp=2,cp=1"),
        pytest.param(4, 2, 1, id="tp=4,pp=2,cp=1"),
        # CP>1 configs (known broken)
        pytest.param(
            1,
            1,
            2,
            id="tp=1,pp=1,cp=2",
            marks=pytest.mark.xfail(reason="CP>1 is known broken for inference", strict=False),
        ),
    ],
)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy_7b(mbridge_checkpoint_7b_1m_path, tmp_path, dna_sequences, tp, pp, cp):
    """Test that parallel inference with the 7b model produces accurate generation results.

    Uses the 7b-1m checkpoint which supports TP>1 (32 attention heads) and PP>1,
    enabling proper tensor and pipeline parallel accuracy testing.
    """
    num_gpus_required = tp * pp * cp
    if torch.cuda.device_count() < num_gpus_required:
        pytest.skip(f"Not enough GPUs: need {num_gpus_required}, have {torch.cuda.device_count()}")

    num_tokens = 500
    # Expected sequence identity percentages for the 7b model (from test_evo2.py)
    expected_matchpercents = [97.60, 89.63, 80.03, 84.57]

    # Build a single JSONL prompt file with all sequences, keyed by id
    targets_by_id: dict[str, str] = {}
    expected_by_id: dict[str, float] = {}
    jsonl_entries = []
    for i, (seq, expected_mp) in enumerate(zip(dna_sequences, expected_matchpercents)):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)
        seq_id = f"seq_{i}"
        targets_by_id[seq_id] = target
        expected_by_id[seq_id] = expected_mp
        jsonl_entries.append((seq_id, prompt))

    prompt_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "outputs.jsonl"
    _write_prompts_jsonl(prompt_file, jsonl_entries)

    # Single inference call processes all prompts (batching handled internally)
    records = run_infer_subprocess_parallel(
        mbridge_checkpoint_7b_1m_path,
        prompt_file=prompt_file,
        output_file=output_file,
        max_new_tokens=num_tokens,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
        tensor_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )

    assert len(records) == len(dna_sequences), f"Expected {len(dna_sequences)} results, got {len(records)}"

    # Match results by id (output order is not guaranteed with dynamic engines)
    results_by_id = {r["id"]: r for r in records}
    match_percents = {}
    for seq_id, target in targets_by_id.items():
        assert seq_id in results_by_id, f"Missing result for {seq_id}"
        identity = calculate_sequence_identity(target, results_by_id[seq_id]["completion"])
        match_percents[seq_id] = identity

    matchperc_print = {k: f"{v:.2f}%" for k, v in match_percents.items()}
    matchperc_print_expected = {k: f"{v:.2f}%" for k, v in expected_by_id.items()}

    assert all(match_percents[sid] >= 0.90 * expected_by_id[sid] for sid in targets_by_id), (
        f"Expected at least 90% of {matchperc_print_expected}, got {matchperc_print}"
    )


SAVANNA_7B_REPO = "arcinstitute/savanna_evo2_7b"


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_from_savanna(tmp_path_factory) -> Path:
    """Convert the ARC Savanna 7B checkpoint to MBridge and return the iteration directory.

    Downloads the savanna checkpoint from HuggingFace, converts it via
    ``savanna_to_mbridge``, and returns the ``iter_0000001`` path ready for
    inference.
    """
    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b_savanna")
    mbridge_ckpt_dir = savanna_to_mbridge(
        savanna_ckpt_path=SAVANNA_7B_REPO,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="evo2_7b",
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        seq_length=8192,
        te_enabled=True,
        mixed_precision_recipe="bf16_mixed",
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.skipif(
    not os.environ.get("LONG_TESTS"),
    reason="Set LONG_TESTS=1 to run (downloads ~30GB savanna checkpoint)",
)
def test_savanna_to_mbridge_inference_accuracy_7b(mbridge_checkpoint_7b_from_savanna, tmp_path, dna_sequences):
    """Validate the Savanna-to-MBridge conversion by running inference at TP=2.

    Downloads the ARC 7B savanna checkpoint, converts it to MBridge, generates
    500 tokens for each test sequence, and checks that sequence identity matches
    expected baselines within 90%.
    """
    tp = 2
    if torch.cuda.device_count() < tp:
        pytest.skip(f"Not enough GPUs: need {tp}, have {torch.cuda.device_count()}")

    num_tokens = 500
    expected_matchpercents = [97.60, 89.63, 80.03, 84.57]

    match_percents = []
    for i, seq in enumerate(dna_sequences):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)

        prompt_file = tmp_path / f"prompt_savanna7b_seq{i}.txt"
        output_file = tmp_path / f"output_savanna7b_seq{i}.txt"
        prompt_file.write_text(prompt)

        generated_text = run_infer_subprocess_parallel(
            mbridge_checkpoint_7b_from_savanna,
            prompt_file=prompt_file,
            output_file=output_file,
            max_new_tokens=num_tokens,
            temperature=1.0,
            top_k=1,
            seed=42,
            tensor_parallel_size=tp,
        )

        identity = calculate_sequence_identity(target, generated_text)
        match_percents.append(identity)

    matchperc_print = [f"{mp:.2f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.2f}%" for ep in expected_matchpercents]

    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_different_results_with_without_peft(tmp_path, mbridge_checkpoint_path, lora_finetune_checkpoint):
    """Top-k sample from the base ckpt vs. the LoRA ckpt and assert the logprobs differ."""
    env = copy.deepcopy(PRETEST_ENV)
    # 64-char prompt for FP8 divisibility.
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    def _run_infer(ckpt: Path, output_file: Path) -> dict:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            "1",
            "--nnodes",
            "1",
            "-m",
            "bionemo.evo2.run.infer",
            "--ckpt-dir",
            str(ckpt),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "10",
            "--temperature",
            "1.0",
            "--top-k",
            "2",  # top_k=1 makes chosen-token log-probs 0.0, so a base/LoRA comparison is vacuous.
            "--seed",
            "0",
            "--ignore-eos",
            "--return-log-probs",
            "--output-file",
            str(output_file),
        ]
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300, env=env)
        assert r.returncode == 0, f"infer_evo2 failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        with open(output_file) as f:
            return json.loads(f.readline())

    base = _run_infer(mbridge_checkpoint_path, tmp_path / "out_base.jsonl")
    lora = _run_infer(lora_finetune_checkpoint, tmp_path / "out_lora.jsonl")

    base_lp = base["logprobs"]["completion_logprobs"]
    lora_lp = lora["logprobs"]["completion_logprobs"]
    assert len(base_lp) == len(lora_lp), f"Different completion lengths: {len(base_lp)} vs {len(lora_lp)}"
    assert base_lp != lora_lp, "LoRA adapter had no effect on completion logprobs"


def test_hyena_inference_context_initialization():
    """Test that HyenaInferenceContext can be initialized."""
    context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
    assert context is not None
    assert context.max_batch_size == 1
    assert context.max_sequence_length == 8192


def test_hyena_inference_context_reset():
    """Test that context reset works without error."""
    context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
    # Add some fake filter state (simulating what hyena layers do)
    context.filter_state_dict_layer_0 = {"key": torch.zeros(10)}
    context.filter_state_dict_layer_1 = {"key": torch.ones(10)}

    # Verify the state was added
    assert hasattr(context, "filter_state_dict_layer_0")
    assert hasattr(context, "filter_state_dict_layer_1")

    # Reset should remove all filter_state_dict attributes
    context.reset()

    assert not hasattr(context, "filter_state_dict_layer_0")
    assert not hasattr(context, "filter_state_dict_layer_1")


def test_hyena_inference_context_materialize_logits_setting():
    """Test that materialize_only_last_token_logits can be configured."""
    context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)

    # Default should be True for efficiency
    # We can set it to False if we need full sequence logits
    context.materialize_only_last_token_logits = False
    assert context.materialize_only_last_token_logits is False

    context.materialize_only_last_token_logits = True
    assert context.materialize_only_last_token_logits is True


def test_hyena_inference_context_multiple_batches():
    """Test context with different batch sizes."""
    for batch_size in [1, 2, 4]:
        context = HyenaInferenceContext(max_batch_size=batch_size, max_sequence_length=4096)
        assert context.max_batch_size == batch_size
        context.reset()  # Should not error


def test_hyena_inference_context_different_sequence_lengths():
    """Test context with different max sequence lengths."""
    for seq_len in [1024, 8192, 16384]:
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=seq_len)
        assert context.max_sequence_length == seq_len
        context.reset()


# =============================================================================
# Native dynamic-inference engine edge-case tests
# =============================================================================
# These exercise the NATIVE mcore dynamic-inference path (paged-KV attention + Hyena recurrent
# state packed into mcore's two Mamba slots). They run against the small 1b-8k-bf16 fixture
# checkpoint (real weights, validates the mechanism + correctness, not just shapes). Edge cases
# cover full-prompt multi-block prefill (prompt > block_size_tokens), opt-in chunked prefill,
# single-token decode, longer generation, TP-non-divisible batch (batch=1 on TP=2), and
# prompt-shorter-than-the-medium-FIR-ring behavior. Greedy decoding (top_k=1) keeps the
# assertions deterministic.

# Paged-KV block size for the multi-block prefill test below. It also happens to be the CLI/engine
# default, but the test pins it explicitly (passing --inference-dynamic-batching-block-size) so the
# "prompt spans more than one block" premise cannot be silently broken by a future change to the default.
KV_BLOCK_SIZE_TOKENS = 256

# A long DNA prompt (> KV_BLOCK_SIZE_TOKENS) that forces a multi-block paged-KV prefill.
LONG_DNA_PROMPT = (
    "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGG"
    "GAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGGCGAGGCATTGCCTCACCTGGGAAGCGCAAGGGGTCAG"
    "GGAGTTCCCTTTCCGAGTCAAAGAAAGGGGTGACGGACGCACCTGGAAAATCGGGTCACTCCCACCCGAATATTGCGCTTTTCAGACCGGCTTAAGAA"
    "ACGGCGCACCACGAGACTATATCCCACAC"
)
assert len(LONG_DNA_PROMPT) > KV_BLOCK_SIZE_TOKENS, (
    f"LONG_DNA_PROMPT must exceed block_size_tokens={KV_BLOCK_SIZE_TOKENS} to cover >1 KV block"
)

DNA_BASES = set("ACGTacgtNn")


def _is_dna_completion(text: str) -> bool:
    """True when every character of ``text`` is a DNA base (Evo2's byte vocab)."""
    return len(text) > 0 and all(c in DNA_BASES for c in text)


def test_native_dynamic_runs(mbridge_checkpoint_path, tmp_path):
    """A short prompt generates a non-empty DNA completion through the native engine."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt="ACGTACGTAACCGGTTACGTACGTAACCGGTT",
        output_file=tmp_path / "native_runs.jsonl",
        max_new_tokens=10,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    assert record["usage"]["prompt_tokens"] > 0
    assert record["usage"]["completion_tokens"] == 10
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_full_prefill_multi_block(mbridge_checkpoint_path, tmp_path):
    """A prompt longer than the paged-KV block size prefills as one multi-block request.

    The block size is pinned explicitly (``--inference-dynamic-batching-block-size``) and the prompt
    exceeds it, so with no ``--enable-chunked-prefill`` the whole prompt is enqueued as a single
    prefill chunk whose KV spans ``ceil(n_prompt / block_size) >= 2`` paged blocks. The first forward
    processes all prompt tokens, and last_token_logits selects the true final position before decode.
    Pinning the block size (rather than relying on the default) is what makes this a multi-block test.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    block_size_tokens = KV_BLOCK_SIZE_TOKENS
    n_prompt_tokens = len(LONG_DNA_PROMPT)
    assert n_prompt_tokens > block_size_tokens, (
        f"LONG_DNA_PROMPT ({n_prompt_tokens} tokens) must exceed block_size_tokens={block_size_tokens} "
        "to span more than one KV block"
    )
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "native_full_prefill_multi_block.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
        block_size_tokens=block_size_tokens,
    )
    # The whole prompt must have been prefilled (KV spanning >1 block) and 20 tokens generated.
    assert record["usage"]["prompt_tokens"] == n_prompt_tokens, (
        f"prompt_tokens {record['usage']['prompt_tokens']} != {n_prompt_tokens}; multi-block "
        "prefill did not enqueue the full prompt"
    )
    assert record["usage"]["completion_tokens"] == 20
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_chunked_prefill_cli_multi_chunk(mbridge_checkpoint_path, tmp_path):
    """--enable-chunked-prefill allows prompts to exceed the per-step token budget."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    n_prompt_tokens = len(LONG_DNA_PROMPT)
    max_tokens = 256
    assert n_prompt_tokens > max_tokens
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "native_chunked_prefill.jsonl",
        max_new_tokens=4,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
        extra_args=[
            "--enable-chunked-prefill",
            "--inference-dynamic-batching-max-tokens",
            str(max_tokens),
        ],
    )
    assert record["usage"]["prompt_tokens"] == n_prompt_tokens
    assert record["usage"]["completion_tokens"] == 4
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_chunked_prefill_matches_full_prefill(mbridge_checkpoint_path, tmp_path):
    """Chunked prefill yields the same greedy continuation as single-shot (full) prefill.

    This is the prefix-invariance idea (same prompt -> same completion two ways) applied to chunked
    prefill: prefilling the whole prompt in one forward vs splitting it across multiple prefill
    forwards (``--enable-chunked-prefill`` with a per-step token budget below the prompt length) must
    produce identical tokens under greedy decoding, since chunked prefill is only a memory-bounded way
    to compute the same prefill. The existing chunked-prefill test only checks it runs and emits DNA;
    this one pins the equivalence to full prefill. It guards the Hyena chunked-prefill fix: the FIR/IIR
    recurrent state is threaded across chunks by stepping each chunk's tokens through step_fir/step_iir
    (hyena_utils.ParallelCausalDepthwiseConv1dWithState.forward / forward_long / forward_medium); before
    that fix, chunk 1+ was misclassified as a single decode step and the output degenerated.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")

    n_prompt_tokens = len(LONG_DNA_PROMPT)
    chunk_max_tokens = 128
    # Force at least two prefill chunks with a non-trivial final chunk (>1 token).
    assert n_prompt_tokens > 2 * chunk_max_tokens, (
        f"LONG_DNA_PROMPT ({n_prompt_tokens} tokens) must exceed 2*chunk_max_tokens={2 * chunk_max_tokens} "
        "to exercise multiple prefill chunks"
    )

    full = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "full_prefill.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # greedy -> deterministic
        seed=42,
        max_seq_length=512,
    )
    chunked = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "chunked_prefill.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
        extra_args=[
            "--enable-chunked-prefill",
            "--inference-dynamic-batching-max-tokens",
            str(chunk_max_tokens),
        ],
    )

    # Both prefilled the full prompt; chunked must reproduce the single-shot greedy continuation.
    assert full["usage"]["prompt_tokens"] == n_prompt_tokens == chunked["usage"]["prompt_tokens"]
    assert full["usage"]["completion_tokens"] == 20 == chunked["usage"]["completion_tokens"]
    assert _is_dna_completion(full["completion"]), f"non-DNA full-prefill completion: {full['completion']!r}"
    assert chunked["completion"] == full["completion"], (
        "chunked prefill diverged from full prefill:\n"
        f"  full   ={full['completion']!r}\n"
        f"  chunked={chunked['completion']!r}"
    )


def test_native_dynamic_full_fp8_runs_with_and_without_chunked_prefill(mbridge_checkpoint_path, tmp_path):
    """Full fp8 inference (fp8 on every TE linear) runs both with full and with chunked prefill.

    Confirms the fp8 token-padding path (``prepare_model_for_fp8_inference``, applied in
    ``setup_inference_engine`` when the recipe turns on fp8) coexists with (a) the multi-block /
    chunked-prefill Hyena block-step and (b) the CUDA-graphed single-token decode. Greedy full vs
    chunked fp8 completions need NOT be bit-identical: current-scaling fp8 derives each GEMM's scale
    from its own activation amax, which differs between a whole-prompt prefill and per-chunk prefills.
    So this pins that BOTH configurations run and emit a valid DNA completion of the requested length
    (not that they match) -- the bf16 equivalence above already pins the exact full==chunked behavior.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    if not is_fp8_supported:
        pytest.skip(f"FP8 not supported on {device_info} ({compute_capability})")

    n_prompt_tokens = len(LONG_DNA_PROMPT)
    chunk_max_tokens = 128
    assert n_prompt_tokens > 2 * chunk_max_tokens, (
        f"LONG_DNA_PROMPT ({n_prompt_tokens} tokens) must exceed 2*chunk_max_tokens={2 * chunk_max_tokens}"
    )
    fp8_args = ["--mixed-precision-recipe", "bf16_with_fp8_current_scaling_mixed"]

    full = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "fp8_full_prefill.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # greedy
        seed=42,
        max_seq_length=512,
        extra_args=fp8_args,
    )
    chunked = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "fp8_chunked_prefill.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
        extra_args=[
            *fp8_args,
            "--enable-chunked-prefill",
            "--inference-dynamic-batching-max-tokens",
            str(chunk_max_tokens),
        ],
    )
    for label, rec in (("full", full), ("chunked", chunked)):
        assert rec["usage"]["prompt_tokens"] == n_prompt_tokens, (
            f"{label} fp8 prefill enqueued {rec['usage']['prompt_tokens']} != {n_prompt_tokens}"
        )
        assert rec["usage"]["completion_tokens"] == 20, (
            f"{label} fp8 generated {rec['usage']['completion_tokens']} != 20 tokens"
        )
        assert _is_dna_completion(rec["completion"]), f"non-DNA {label} fp8 completion: {rec['completion']!r}"


def test_native_dynamic_single_token_decode(mbridge_checkpoint_path, tmp_path):
    """A single decode step (max_new_tokens=1) produces exactly one token after prefill."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt="ACGTACGTAACCGGTTACGTACGTAACCGGTT",
        output_file=tmp_path / "native_single.jsonl",
        max_new_tokens=1,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=256,
    )
    assert record["usage"]["completion_tokens"] == 1, "expected exactly one decoded token"
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_short_prompt_under_medium_ring(mbridge_checkpoint_path, tmp_path):
    """A prompt shorter than the medium-FIR ring (127) prefills via the right-aligned seed.

    The medium Hyena operator's recurrent FIR ring is 127 wide; a short prompt produces a seed
    shorter than the ring. The packed-slot path right-aligns that short seed into the fixed-width
    ring (numerically equivalent to the eager grow path for the flip-filter medium operator). This
    guards that fix: a ~16-token prompt must still generate a valid DNA completion.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt="ACGTACGTAACCGGTT",  # 16 tokens << 127 (medium ring width)
        output_file=tmp_path / "native_short.jsonl",
        max_new_tokens=10,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=256,
    )
    assert record["usage"]["completion_tokens"] == 10
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_long_generation(mbridge_checkpoint_path, tmp_path):
    """A longer generation (100 tokens) runs many decode steps without context overflow."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_long_gen.jsonl",
        max_new_tokens=100,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=1024,
    )
    assert record["usage"]["completion_tokens"] == 100, "long generation did not reach 100 tokens"
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_deterministic(mbridge_checkpoint_path, tmp_path):
    """Greedy decoding with the same prompt + seed is reproducible across runs."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    rec1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_det1.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    rec2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_det2.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    assert rec1["completion"] == rec2["completion"], (
        f"native greedy decode not deterministic:\n  run1: {rec1['completion']}\n  run2: {rec2['completion']}"
    )


def test_native_dynamic_different_prompts_differ(mbridge_checkpoint_path, tmp_path):
    """Different prompts produce different completions (the model responds to the prompt)."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    rec1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_diff1.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    rec2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_2,
        output_file=tmp_path / "native_diff2.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    assert rec1["completion"] != rec2["completion"], "different prompts produced identical completions"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_native_dynamic_tp2_batch1(mbridge_checkpoint_7b_1m_path, tmp_path):
    """TP=2 with a single request (batch=1) runs through decode-only CUDA graphs.

    Evo2 keeps sequence parallelism disabled for standalone inference and sizes each context to
    the active request count, while mcore pads decode graph dimensions only as needed for TP
    alignment. Needs the 7b checkpoint (32 heads, TP-divisible) + 2 GPUs.
    """
    tp = 2
    if torch.cuda.device_count() < tp:
        pytest.skip(f"TP={tp} requires {tp} GPUs, have {torch.cuda.device_count()}")
    output_file = tmp_path / "native_tp2.jsonl"
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(tp),
        "--nnodes",
        "1",
        str(_infer_script_path()),
        "--ckpt-dir",
        str(mbridge_checkpoint_7b_1m_path),
        "--prompt",
        "ACGTACGTAACCGGTTACGTACGTAACCGGTT",
        "--max-new-tokens",
        "10",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",
        "--top-k",
        "1",
        "--seed",
        "42",
        "--tensor-parallel-size",
        str(tp),
        "--max-seq-length",
        "256",
    ]
    env = copy.deepcopy(PRETEST_ENV)
    env["PYTHONPATH"] = str(_recipe_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, f"native TP=2 infer failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    records = _read_jsonl_results(output_file)
    assert len(records) == 1
    assert records[0]["usage"]["completion_tokens"] == 10
    assert _is_dna_completion(records[0]["completion"]), f"non-DNA completion: {records[0]['completion']!r}"
