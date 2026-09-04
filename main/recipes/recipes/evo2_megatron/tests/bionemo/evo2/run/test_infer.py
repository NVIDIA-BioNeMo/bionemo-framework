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
The generation tests below exercise this engine directly. A single mixed-prompt test reuses one
model load to cover the JSONL contract, ragged batched prefill, prompt sensitivity, and the short-
and long-prompt edge cases through 100 decode steps. Separate tests remain only for state
transitions that cannot share that run: full-vs-chunked prefill equivalence, LoRA, and model parallelism.

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
from megatron.core.transformer.enums import InferenceCudaGraphScope

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
    _stop_token_mask,
    _suppress_stop_token_logits,
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
    ("requested_impl", "requested_scope", "checkpoint_impl", "expected_enabled", "expected_scope"),
    [
        ("local", "block", "none", True, InferenceCudaGraphScope.block),
        ("local", "layer", "none", True, InferenceCudaGraphScope.layer),
        ("none", "block", "local", False, InferenceCudaGraphScope.none),
    ],
)
def test_configure_native_dynamic_cuda_graphs_normalizes_checkpoint_state(
    requested_impl, requested_scope, checkpoint_impl, expected_enabled, expected_scope
):
    """The CLI choice must replace stale graph settings loaded from a checkpoint."""
    provider = SimpleNamespace(
        cuda_graph_impl=checkpoint_impl,
        cuda_graph_scope=None,
        inference_cuda_graph_scope="none" if requested_impl == "local" else "layer",
    )

    enabled = infer_module._configure_native_dynamic_cuda_graphs(
        provider,
        rank=1,
        cuda_graph_impl=requested_impl,
        cuda_graph_scope=requested_scope,
    )

    assert enabled is expected_enabled
    assert provider.cuda_graph_impl == requested_impl
    assert provider.inference_cuda_graph_scope is expected_scope
    assert provider.cuda_graph_scope == []


@pytest.mark.parametrize(
    ("requested_scope", "cuda_graph_impl", "fp8_enabled", "fp4_enabled", "expected_scope"),
    [
        ("block", "local", False, False, "block"),
        ("block", "local", True, False, "layer"),
        ("block", "local", False, True, "layer"),
        ("layer", "local", True, False, "layer"),
        ("block", "none", True, False, "block"),
    ],
)
def test_resolve_native_dynamic_cuda_graph_scope_uses_layers_for_global_quantization(
    requested_scope: str,
    cuda_graph_impl: str,
    fp8_enabled: bool,
    fp4_enabled: bool,
    expected_scope: str,
) -> None:
    """Whole-stack MCore graphs cannot own the per-layer TE FP8/FP4 runtime state."""
    assert (
        infer_module._resolve_native_dynamic_cuda_graph_scope(
            requested_scope,
            cuda_graph_impl=cuda_graph_impl,
            fp8_enabled=fp8_enabled,
            fp4_enabled=fp4_enabled,
        )
        == expected_scope
    )


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


def test_graph_storage_signature_tracks_rebinding():
    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))
            self.register_buffer("scale", torch.ones(4))

    model = _Model()
    original = infer_module._model_storage_signature(model)

    weight_version = model.weight._version
    scale_version = model.scale._version
    with torch.no_grad():
        model.weight.add_(1)
        model.scale.mul_(2)
    assert model.weight._version > weight_version
    assert model.scale._version > scale_version
    assert infer_module._model_storage_signature(model) == original

    model.weight.data = model.weight.detach().clone()
    rebound_parameter = infer_module._model_storage_signature(model)
    assert rebound_parameter != original

    model.scale.data = model.scale.detach().clone()
    assert infer_module._model_storage_signature(model) != rebound_parameter


def test_persistent_graph_manager_binding_survives_training_toggle(monkeypatch):
    """A rollout restores the exact manager object removed while training graphs are off."""
    manager = SimpleNamespace(cudagraph_runners=[object()])
    layer = torch.nn.Linear(2, 2)
    layer.cudagraph_manager = manager
    nd = SimpleNamespace(
        cuda_graphs_enabled=True,
        hyena_model=layer,
        cuda_graph_model_storage_signature=infer_module._model_storage_signature(layer),
        cuda_graph_manager_bindings=((layer, manager),),
        cuda_graph_manager_count=0,
    )
    monkeypatch.setattr(infer_module, "_graph_parallel_any", bool)

    del layer.cudagraph_manager
    assert not infer_module._invalidate_cuda_graphs_for_rebound_model_storage(nd)

    assert layer.cudagraph_manager is manager
    assert nd.cuda_graph_manager_count == 1


@pytest.mark.parametrize("change_source", ["parameter", "buffer", "peer", "quantized-refit"])
def test_dynamic_graph_recaptures_after_model_state_change(monkeypatch, change_source):
    """Validation-first graphs must not survive changed captured model state."""

    class _Context:
        max_sequence_length = 64
        max_requests = 2
        evo2_warmed_cuda_graph_request_counts = frozenset({2})

        def reset(self):
            events.append("context-reset")

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4, 4))
            self.register_buffer("scale", torch.ones(4))

    model = _Model()
    context = _Context()
    static_context = SimpleNamespace(
        evo2_static_cuda_graph_warmed=True,
        evo2_static_cuda_graph_replay_verified=True,
    )
    nd = SimpleNamespace(
        shared_dyn_ctx=context,
        shared_dyn_ctx_key=(16, 64, False),
        static_contexts={(2, 64): static_context},
        cuda_graphs_enabled=True,
        cuda_graph_model_storage_signature=infer_module._model_storage_signature(model),
        hyena_model=model,
        max_seq_length=64,
    )
    events = []
    peer_storage_changed = {"value": False}
    monkeypatch.setattr(
        infer_module,
        "_graph_parallel_any",
        lambda local_value: bool(local_value or peer_storage_changed["value"]),
    )
    monkeypatch.setattr(infer_module, "_reset_layer_cuda_graphs", lambda _nd: events.append("graph-reset"))
    monkeypatch.setattr(
        infer_module,
        "_warmup_native_dynamic_cuda_graphs",
        lambda *_args, **_kwargs: events.append(("graph-capture", context.evo2_warmed_cuda_graph_request_counts)),
    )
    monkeypatch.setattr(
        infer_module,
        "_validate_cuda_graph_capture",
        lambda *_args, **_kwargs: events.append("graph-validate"),
    )
    monkeypatch.setattr(infer_module, "_begin_cuda_phase", lambda **_kwargs: 0.0)
    monkeypatch.setattr(infer_module, "_finish_cuda_phase", lambda *_args, **_kwargs: infer_module._CudaPhaseStats())

    with torch.no_grad():
        model.weight.add_(1)
    reused, _, _ = infer_module._get_or_build_shared_dynamic_context(
        nd,
        block_size_tokens=16,
        max_tokens=64,
        enable_chunked_prefill=False,
        max_active_requests=2,
        device=torch.device("cpu"),
    )
    assert reused is context
    assert events == ["context-reset"]

    if change_source == "peer":
        peer_storage_changed["value"] = True
    elif change_source == "quantized-refit":
        nd.cuda_graph_force_recapture = True
    else:
        rebound = model.weight if change_source == "parameter" else model.scale
        rebound.data = rebound.detach().clone()
    recaptured, _, _ = infer_module._get_or_build_shared_dynamic_context(
        nd,
        block_size_tokens=16,
        max_tokens=64,
        enable_chunked_prefill=False,
        max_active_requests=2,
        device=torch.device("cpu"),
    )
    assert recaptured is context
    assert events == [
        "context-reset",
        "graph-reset",
        "context-reset",
        ("graph-capture", frozenset()),
        "graph-validate",
    ]
    assert context.evo2_warmed_cuda_graph_request_counts == frozenset({2})
    assert not static_context.evo2_static_cuda_graph_warmed
    assert not static_context.evo2_static_cuda_graph_replay_verified
    assert nd.cuda_graph_model_storage_signature == infer_module._model_storage_signature(model)
    assert not nd.cuda_graph_force_recapture


@pytest.mark.parametrize("remote_group", ["model", "context"])
def test_graph_storage_change_consensus_spans_graph_parallel_groups(monkeypatch, remote_group):
    from megatron.core import parallel_state

    model_group = object()
    context_group = object()
    calls = []
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda _group: "gloo")
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(parallel_state, "get_model_parallel_group", lambda: model_group)
    monkeypatch.setattr(parallel_state, "get_context_parallel_group", lambda: context_group)

    def _all_reduce(value, *, op, group):
        assert op == torch.distributed.ReduceOp.MAX
        calls.append(group)
        if (remote_group == "model" and group is model_group) or (
            remote_group == "context" and group is context_group
        ):
            value.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", _all_reduce)

    assert infer_module._graph_parallel_any(False)
    assert calls == [model_group, context_group]


def test_graph_storage_change_consensus_skips_single_rank_collectives(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(parallel_state, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_pipeline_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda *_args, **_kwargs: pytest.fail("single-rank inference must not issue a consensus collective"),
    )

    assert not infer_module._graph_parallel_any(False)
    assert infer_module._graph_parallel_any(True)


def test_static_flash_context_cache_invalidates_on_sequence_capacity_growth(monkeypatch):
    created = []

    class _StaticContext:
        def __init__(self, *, max_batch_size, max_sequence_length):
            self.max_batch_size = max_batch_size
            self.max_sequence_length = max_sequence_length
            self.reset_count = 0
            created.append(self)

        def reset(self):
            self.reset_count += 1

    monkeypatch.setattr("megatron.core.inference.contexts.StaticInferenceContext", _StaticContext)
    monkeypatch.setattr(infer_module, "bind_hyena_packed_views_to_static_context", lambda *_args, **_kwargs: None)
    reset_graphs = []
    monkeypatch.setattr(infer_module, "_reset_layer_cuda_graphs", lambda _nd: reset_graphs.append(True))
    nd = SimpleNamespace(
        static_contexts={},
        cuda_graphs_enabled=True,
        hyena_model=object(),
        shared_dyn_ctx=object(),
        shared_dyn_ctx_key=(16, None, False),
    )

    first, _ = infer_module._get_or_build_static_flash_context(
        nd,
        batch_size=4,
        max_sequence_length=64,
        device=torch.device("cpu"),
    )
    grown, _ = infer_module._get_or_build_static_flash_context(
        nd,
        batch_size=4,
        max_sequence_length=128,
        device=torch.device("cpu"),
    )

    assert created == [first, grown]
    assert nd.static_contexts == {(4, 128): grown}
    assert reset_graphs == [True]
    assert nd.shared_dyn_ctx is None
    assert nd.shared_dyn_ctx_key is None


def test_static_flash_context_cache_is_bounded_to_full_and_remainder_shapes(monkeypatch):
    class _StaticContext:
        def __init__(self, *, max_batch_size, max_sequence_length):
            self.max_batch_size = max_batch_size
            self.max_sequence_length = max_sequence_length

        def reset(self):
            pass

    monkeypatch.setattr("megatron.core.inference.contexts.StaticInferenceContext", _StaticContext)
    monkeypatch.setattr(infer_module, "bind_hyena_packed_views_to_static_context", lambda *_args, **_kwargs: None)
    reset_graphs = []
    monkeypatch.setattr(infer_module, "_reset_layer_cuda_graphs", lambda _nd: reset_graphs.append(True))
    nd = SimpleNamespace(
        static_contexts={},
        cuda_graphs_enabled=True,
        hyena_model=object(),
        shared_dyn_ctx=None,
        shared_dyn_ctx_key=None,
    )

    old_contexts = [
        infer_module._get_or_build_static_flash_context(
            nd,
            batch_size=batch_size,
            max_sequence_length=64,
            device=torch.device("cpu"),
        )[0]
        for batch_size in (4, 3)
    ]
    newest, _ = infer_module._get_or_build_static_flash_context(
        nd,
        batch_size=2,
        max_sequence_length=64,
        device=torch.device("cpu"),
    )

    assert nd.static_contexts == {(2, 64): newest}
    assert all(context not in nd.static_contexts.values() for context in old_contexts)
    assert reset_graphs == [True]


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


@pytest.mark.parametrize(("use_inference_mode", "expected_inference_mode"), [(True, True), (False, False)])
def test_native_torch_context_selects_tensor_kind(use_inference_mode, expected_inference_mode):
    nd = SimpleNamespace(use_torch_inference_mode=use_inference_mode)

    with infer_module._native_torch_context(nd):
        assert torch.is_inference_mode_enabled() is expected_inference_mode
        assert torch.is_grad_enabled() is False


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
    assert _sampled_token_action(0, {0}, ignore_eos=True, preserve_eos_token=True) == (False, False)


def test_sampled_eos_stops_and_is_omitted_by_default():
    assert _sampled_token_action(0, {0}, ignore_eos=False) == (False, True)


def test_sampled_eos_stops_and_is_preserved_when_requested():
    assert _sampled_token_action(0, {0}, ignore_eos=False, preserve_eos_token=True) == (True, True)


def test_generation_control_cli_flags_default_false_and_enable_when_passed(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt"])
    defaults = parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--ignore-eos",
            "--preserve-eos-token",
            "--strict-generation",
        ],
    )
    enabled = parse_args()

    assert defaults.ignore_eos is False
    assert defaults.preserve_eos_token is False
    assert defaults.strict_generation is False
    assert enabled.ignore_eos is True
    assert enabled.preserve_eos_token is True
    assert enabled.strict_generation is True


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [([], None), (["--context-parallel-comm-type", "p2p"], "p2p"), (["--context-parallel-comm-type", "a2a"], "a2a")],
)
def test_infer_context_parallel_comm_type_cli(monkeypatch, extra_args, expected):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt", *extra_args])

    assert parse_args().context_parallel_comm_type == expected


def test_cuda_graph_cli_defaults_to_block_scope_with_layer_fallback(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt"])
    defaults = parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        ["infer", "--ckpt-dir", "/tmp/ckpt", "--cuda-graph-scope", "layer"],
    )
    layer = parse_args()

    assert defaults.cuda_graph_scope == "block"
    assert layer.cuda_graph_scope == "layer"


def test_inference_backend_cli_defaults_dynamic_and_accepts_static_flash(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt"])
    defaults = parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer", "--ckpt-dir", "/tmp/ckpt", "--inference-backend", "static-flash"],
    )
    static_flash = parse_args()

    assert defaults.inference_backend == "dynamic"
    assert static_flash.inference_backend == "static-flash"


def test_global_fp8_all_layers_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["infer", "--ckpt-dir", "/tmp/ckpt"])
    defaults = parse_args()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer",
            "--ckpt-dir",
            "/tmp/ckpt",
            "--mixed-precision-recipe",
            "bf16_with_fp8_current_scaling_mixed",
            "--fp8-all-layers",
        ],
    )
    fp8 = parse_args()

    assert defaults.fp8_all_layers is False
    assert fp8.fp8_all_layers is True


def test_generate_dispatches_explicit_static_flash_backend(monkeypatch):
    expected = [object()]
    calls = []

    def fake_static(*args, **kwargs):
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(infer_module, "_generate_static_flash", fake_static)

    result = infer_module.generate(
        SimpleNamespace(),
        ["ACGT"],
        max_new_tokens=2,
        top_k=3,
        top_p=0.7,
        preserve_eos_token=True,
        inference_backend="static-flash",
    )

    assert result is expected
    assert calls[0][1]["evo2_batched_decode_size"] == 1
    assert calls[0][1]["top_k"] == 3
    assert calls[0][1]["top_p"] == 0.7
    assert calls[0][1]["preserve_eos_token"] is True


def test_reset_cuda_graphs_clears_current_and_legacy_manager_caches(monkeypatch):
    """Context growth must not leave an installed-MCore custom-key runner alive."""
    manager = SimpleNamespace(
        cudagraph_runners=[object()],
        custom_cudagraphs_lookup_table={"current": object()},
        inference_cudagraphs_lookup_table={"legacy": object()},
    )
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(cudagraph_manager=manager)])
    deleted = []
    monkeypatch.setattr(
        "megatron.core.transformer.cuda_graphs.delete_cuda_graphs",
        lambda: deleted.append(True),
    )

    infer_module._reset_layer_cuda_graphs(SimpleNamespace(hyena_model=model))

    assert manager.cudagraph_runners == []
    assert manager.custom_cudagraphs_lookup_table == {}
    assert manager.inference_cudagraphs_lookup_table == {}
    assert deleted == [True]


def test_dynamic_context_rebuild_invalidates_warmed_static_contexts(monkeypatch):
    """A dynamic graph reset must not leave a static context marked as graph-warmed."""

    class _BuildContext:
        def __init__(self, *, model_config, inference_config):
            del model_config
            self.max_sequence_length = inference_config.max_sequence_length
            self.max_requests = inference_config.max_requests

        def initialize_all_tensors(self):
            pass

    stale_static = SimpleNamespace(
        evo2_static_cuda_graph_warmed=True,
        evo2_static_cuda_graph_replay_verified=True,
    )
    old_dynamic = SimpleNamespace(max_sequence_length=64, max_requests=1)
    nd = SimpleNamespace(
        shared_dyn_ctx=old_dynamic,
        shared_dyn_ctx_key=(16, 64, False),
        static_contexts={(1, 64): stale_static},
        cuda_graphs_enabled=True,
        hyena_model=SimpleNamespace(config=SimpleNamespace(tensor_model_parallel_size=1)),
        mamba_state_config=object(),
        max_seq_length=128,
        ctx_cls=_BuildContext,
    )
    reset_graphs = []
    monkeypatch.setattr(infer_module, "_reset_layer_cuda_graphs", lambda _nd: reset_graphs.append(True))
    monkeypatch.setattr(infer_module, "compute_evo2_paged_kv_buffer_size_gb", lambda *_args, **_kwargs: 0.01)
    monkeypatch.setattr(infer_module, "_begin_cuda_phase", lambda **_kwargs: 0.0)
    monkeypatch.setattr(infer_module, "_finish_cuda_phase", lambda *_args, **_kwargs: infer_module._CudaPhaseStats())
    monkeypatch.setattr(infer_module, "_warmup_native_dynamic_cuda_graphs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(infer_module, "_validate_cuda_graph_capture", lambda *_args, **_kwargs: None)

    rebuilt, _, _ = infer_module._get_or_build_shared_dynamic_context(
        nd,
        block_size_tokens=16,
        max_tokens=64,
        enable_chunked_prefill=False,
        max_active_requests=1,
        device=torch.device("cpu"),
    )

    assert rebuilt is not old_dynamic
    assert reset_graphs == [True]
    assert nd.static_contexts == {}


def test_dynamic_infer_ignores_subquadratic_ops_without_disabling_cuda_graphs(monkeypatch, caplog):
    """The dynamic backend must keep its segmented/graph path when the legacy flag is passed."""
    setup_kwargs = {}
    components = SimpleNamespace(
        tokenizer=SimpleNamespace(tokenize=lambda text: [ord(char) for char in text]),
        native_dynamic=SimpleNamespace(cuda_graphs_enabled=True),
    )
    result = _NativeDynamicResult(
        generated_text="A",
        generated_length=1,
        prompt_tokens=[65],
        generated_tokens=[65],
    )

    def setup(**kwargs):
        setup_kwargs.update(kwargs)
        return components

    monkeypatch.setattr(infer_module, "get_world_size_safe", lambda: 1)
    monkeypatch.setattr(infer_module, "get_rank_safe", lambda: 0)
    monkeypatch.setattr(infer_module, "_prune_caches", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    monkeypatch.setattr(infer_module, "setup_inference_engine", setup)
    monkeypatch.setattr(infer_module, "generate", lambda *_args, **_kwargs: [result])
    monkeypatch.setattr(infer_module, "_teardown_distributed_for_inference", lambda: None)
    caplog.set_level("WARNING", logger=infer_module.logger.name)

    infer_module.infer(
        prompts=[{"id": "seq", "prompt": "A"}],
        ckpt_dir=Path("/tmp/ckpt"),
        max_new_tokens=1,
        max_seq_length=16,
        use_subquadratic_ops=True,
        cuda_graph_impl="local",
        inference_backend="dynamic",
    )

    assert setup_kwargs["use_subquadratic_ops"] is False
    assert setup_kwargs["cuda_graph_impl"] == "local"
    assert "ignored by the dynamic inference backend" in caplog.text


def test_validate_cuda_graph_capture_records_every_block_graph_runner():
    """A configured graph path must expose captured runners before serving user prompts."""
    runners = [
        SimpleNamespace(fwd_graph_recorded=True, cudagraph_created=True),
        SimpleNamespace(fwd_graph_recorded=True, cudagraph_created=True),
    ]
    manager = SimpleNamespace(cudagraph_runners=runners)
    model = SimpleNamespace(modules=lambda: [SimpleNamespace(cudagraph_manager=manager)])
    native = SimpleNamespace(
        hyena_model=model,
        cuda_graph_scope="block",
        cuda_graph_manager_count=1,
        cuda_graph_runner_count=0,
        cuda_graph_recorded_count=0,
        cuda_graph_replay_verified=False,
    )

    infer_module._validate_cuda_graph_capture(native, expected_request_counts={1, 2})

    assert native.cuda_graph_runner_count == 2
    assert native.cuda_graph_recorded_count == 2
    assert native.cuda_graph_replay_verified is True


def test_late_graph_manager_for_block_scope(monkeypatch):
    """A model built for training gains the graph manager skipped during construction."""
    manager = SimpleNamespace(cudagraph_runners=[])
    config = SimpleNamespace(
        cuda_graph_impl="none",
        inference_cuda_graph_scope=InferenceCudaGraphScope.none,
        cuda_graph_scope=None,
    )
    decoder = SimpleNamespace(config=config)
    model = SimpleNamespace(config=config, decoder=decoder)
    model.modules = lambda: [model, decoder]

    from megatron.core.transformer import cuda_graphs

    monkeypatch.setattr(cuda_graphs, "CudaGraphManager", lambda received_config: manager)

    count = infer_module._ensure_native_dynamic_cuda_graph_managers(model, cuda_graph_scope="block")

    assert count == 1
    assert decoder.cudagraph_manager is manager
    assert config.cuda_graph_impl == "local"
    assert config.inference_cuda_graph_scope is InferenceCudaGraphScope.block
    assert config.cuda_graph_scope == []


def test_validate_layer_cuda_graph_capture_allows_one_fixed_attention_runner():
    """Paged attention keeps one max-request graph while packed Hyena keys each request count."""

    def manager(count):
        return SimpleNamespace(
            cudagraph_runners=[SimpleNamespace(fwd_graph_recorded=True, cudagraph_created=True) for _ in range(count)]
        )

    layer_managers = [manager(2), manager(1), manager(2)]
    layers = [SimpleNamespace(cudagraph_manager=item) for item in layer_managers]
    decoder = SimpleNamespace(layers=layers, layer_type_list=["M", "*", "M"])
    model = SimpleNamespace(
        decoder=decoder,
        modules=lambda: [SimpleNamespace(cudagraph_manager=item) for item in layer_managers],
    )
    native = SimpleNamespace(
        hyena_model=model,
        cuda_graph_scope="layer",
        cuda_graph_manager_count=0,
        cuda_graph_runner_count=0,
        cuda_graph_recorded_count=0,
        cuda_graph_replay_verified=False,
    )

    infer_module._validate_cuda_graph_capture(native, expected_request_counts={1, 2})

    assert native.cuda_graph_manager_count == 3
    assert native.cuda_graph_runner_count == 5
    assert native.cuda_graph_recorded_count == 5
    assert native.cuda_graph_replay_verified is True


def test_validate_layer_cuda_graph_capture_rejects_missing_hyena_request_shape():
    def manager(count):
        return SimpleNamespace(
            cudagraph_runners=[SimpleNamespace(fwd_graph_recorded=True, cudagraph_created=True) for _ in range(count)]
        )

    layer_managers = [manager(1), manager(1)]
    layers = [SimpleNamespace(cudagraph_manager=item) for item in layer_managers]
    decoder = SimpleNamespace(layers=layers, layer_type_list=["M", "*"])
    model = SimpleNamespace(
        decoder=decoder,
        modules=lambda: [SimpleNamespace(cudagraph_manager=item) for item in layer_managers],
    )
    native = SimpleNamespace(hyena_model=model, cuda_graph_scope="layer")

    with pytest.raises(RuntimeError, match="not fully captured"):
        infer_module._validate_cuda_graph_capture(native, expected_request_counts={1, 2})


@pytest.mark.parametrize(
    ("managers", "message"),
    [
        ([], "no CUDA graph manager"),
        (
            [SimpleNamespace(cudagraph_runners=[SimpleNamespace(fwd_graph_recorded=False, cudagraph_created=False)])],
            "not fully captured",
        ),
    ],
)
def test_validate_cuda_graph_capture_rejects_a_configured_but_inactive_graph_path(managers, message):
    modules = [SimpleNamespace(cudagraph_manager=manager) for manager in managers]
    model = SimpleNamespace(modules=lambda: modules)
    native = SimpleNamespace(
        hyena_model=model,
        cuda_graph_scope="block",
        cuda_graph_manager_count=len(managers),
        cuda_graph_runner_count=0,
        cuda_graph_recorded_count=0,
        cuda_graph_replay_verified=False,
    )

    with pytest.raises(RuntimeError, match=message):
        infer_module._validate_cuda_graph_capture(native, expected_request_counts={1})


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
    assert "[PERF] Batch 1 end-to-end:" in caplog.text
    assert records[0]["completion_token_ids"] == [65]


def test_low_overhead_phase_timing_reports_wall_time_without_cuda_sync(monkeypatch):
    monkeypatch.setattr(infer_module, "_CUDA_PHASE_EVIDENCE_ENABLED", False)
    perf_counter_values = iter([10.0, 17.5])
    monkeypatch.setattr(infer_module.time, "perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: pytest.fail("low-overhead timing must not synchronize CUDA"),
    )

    started_at_s = infer_module._begin_cuda_phase()
    stats = infer_module._finish_cuda_phase(started_at_s)

    assert stats.elapsed_s == 7.5
    assert stats.performed is False


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


def test_sampling_log_probs_compose_temperature_top_k_then_top_p():
    """Both filters compose, and log-probs use the final renormalized support."""
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    log_probs = _sampling_log_probs_from_logits(logits, temperature=2.0, top_k=3, top_p=0.7)
    expected = torch.log_softmax(torch.tensor([[2.0, 1.5]], dtype=torch.float32), dim=-1)

    torch.testing.assert_close(log_probs[0, :2], expected[0])
    assert torch.isneginf(log_probs[0, 2:]).all()


def test_sampling_top_p_one_skips_noop_vocab_sort(monkeypatch):
    """A full-mass nucleus must not add a vocabulary sort to every decode step."""
    logits = torch.tensor([[4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    monkeypatch.setattr(torch, "sort", lambda *_args, **_kwargs: pytest.fail("top-p=1.0 sorted logits"))

    actual = _sampling_log_probs_from_logits(logits, temperature=1.0, top_k=2, top_p=1.0)
    expected = torch.log_softmax(torch.tensor([[4.0, 3.0]], dtype=torch.float32), dim=-1)
    torch.testing.assert_close(actual[0, :2], expected[0])
    assert torch.isneginf(actual[0, 2:]).all()


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
        return [3] * len(text)

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


class _MockKVBlockAllocator:
    paused_count = 0

    def __init__(self):
        self.next_block_id = 1024

    @staticmethod
    def get_active_avail() -> int:
        return 1024

    def allocate_memory_blocks(self, count: int) -> torch.Tensor:
        block_ids = torch.arange(self.next_block_id, self.next_block_id + count, dtype=torch.int32)
        self.next_block_id += count
        return block_ids


class _MockNativeDynamicContext:
    def __init__(self, *, stop_after_updates: int | None = None, events: list[str] | None = None):
        self.max_tokens = 128
        self.max_sequence_length = 128
        self.block_size_tokens = 16
        self.num_speculative_tokens = 0
        self.mamba_metadata = SimpleNamespace(request_to_mamba_state_idx=torch.arange(128))
        self.kv_block_allocator = _MockKVBlockAllocator()
        self.request_ids = torch.full((128,), -1, dtype=torch.int32)
        self.request_last_kv_block_offset = torch.full((128,), -1, dtype=torch.int32)
        self.request_last_kv_block_id = torch.full((128,), -1, dtype=torch.int32)
        self.request_kv_block_counts = torch.zeros(128, dtype=torch.int32)
        self.request_to_kv_block_ids = torch.full((128, 16), -1, dtype=torch.int32)
        self.evo2_batched_decode_enabled = False
        self.paused_request_count = 0
        self.chunked_prefill_request_id = -1
        self.stop_after_updates = stop_after_updates
        self.events = events
        self.request_count = 0
        self.total_request_count = 0
        self.update_count = 0
        self.reset_count = 0
        self.active = False
        self.prefill_chunk_lengths = []

    def add_request(self, request, *, prefill_chunk_length: int):
        assert prefill_chunk_length > 0
        row = self.request_count
        block_count = math.ceil(prefill_chunk_length / self.block_size_tokens)
        block_ids = torch.arange(row * 16, row * 16 + block_count, dtype=torch.int32)
        self.request_ids[row] = request.request_id
        self.request_to_kv_block_ids[row, :block_count] = block_ids
        self.request_kv_block_counts[row] = block_count
        self.request_last_kv_block_id[row] = block_ids[-1]
        self.request_last_kv_block_offset[row] = (prefill_chunk_length - 1) % self.block_size_tokens
        self.prefill_chunk_lengths.append(prefill_chunk_length)
        self.request_count += 1
        self.total_request_count += 1
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
        if self.active:
            self.request_last_kv_block_offset[: self.request_count].add_(1).remainder_(self.block_size_tokens)
        if self.stop_after_updates is not None and self.update_count >= self.stop_after_updates:
            self.active = False

    def has_unfinished_requests(self) -> bool:
        return self.active

    def reset(self):
        self.reset_count += 1
        self.active = False
        self.request_count = 0
        self.total_request_count = 0
        self.request_ids.fill_(-1)
        self.request_last_kv_block_offset.fill_(-1)
        self.request_last_kv_block_id.fill_(-1)
        self.request_kv_block_counts.zero_()
        self.request_to_kv_block_ids.fill_(-1)


def test_graph_warmup_uses_only_physical_shapes(monkeypatch):
    class _WarmupContext(_MockNativeDynamicContext):
        def add_request(self, request, *, prefill_chunk_length: int):
            super().add_request(request, prefill_chunk_length=prefill_chunk_length)
            request_idx = self.request_count - 1
            self.mamba_metadata.request_to_mamba_state_idx[request_idx] = 127 - request_idx

    context = _WarmupContext()
    context.mamba_metadata.request_to_mamba_state_idx = torch.arange(128)
    context.evo2_max_batched_decode_requests = 96
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
        request_counts={96},
    )

    assert forward_model.calls == 3
    assert batch_sizes == [96, 96, 96]
    assert bound_slots == [list(range(32, 128))]
    assert batched_decode_enabled_when_bound == [False]
    assert context.reset_count == 1
    assert context.evo2_batched_decode_enabled is False


@pytest.mark.parametrize(
    ("prompt_count", "batch_size", "expected"),
    [
        (96, 96, (96,)),
        (100, 96, (4, 96)),
        (5, 2, (1, 2)),
    ],
)
def test_physical_request_shapes_include_only_full_and_remainder(prompt_count, batch_size, expected):
    assert infer_module._physical_request_counts(prompt_count, batch_size) == expected


def test_packed_decode_rollover_reserves_pages_in_place():
    """Staggered KV-page rollover must not hand request ordering to mcore's pause scheduler."""

    class _Allocator:
        paused_count = 0

        @staticmethod
        def get_active_avail() -> int:
            return 4

        @staticmethod
        def allocate_memory_blocks(count: int) -> torch.Tensor:
            assert count == 2
            return torch.tensor([90, 91], dtype=torch.int32)

    block_table = torch.full((3, 4), -1, dtype=torch.int32)
    block_table[0, 0] = 10
    block_table[1, :2] = torch.tensor([20, 21])
    block_table[2, :3] = torch.tensor([30, 31, 32])
    context = SimpleNamespace(
        paused_request_count=0,
        total_request_count=3,
        num_speculative_tokens=0,
        block_size_tokens=8,
        kv_block_allocator=_Allocator(),
        request_ids=torch.tensor([100, 101, 102], dtype=torch.int32),
        request_last_kv_block_offset=torch.tensor([3, 7, 7], dtype=torch.int32),
        request_last_kv_block_id=torch.tensor([10, 21, 32], dtype=torch.int32),
        request_kv_block_counts=torch.tensor([1, 2, 3], dtype=torch.int32),
        request_to_kv_block_ids=block_table,
    )
    original_request_ids = context.request_ids.clone()

    reserved = infer_module._reserve_packed_decode_rollover_blocks(context, request_count=3)

    assert reserved == 2
    assert torch.equal(context.request_ids, original_request_ids)
    assert context.request_last_kv_block_offset.tolist() == [3, -1, -1]
    assert context.request_last_kv_block_id.tolist() == [10, 90, 91]
    assert context.request_kv_block_counts.tolist() == [1, 3, 4]
    assert context.request_to_kv_block_ids[1].tolist() == [20, 21, 90, -1]
    assert context.request_to_kv_block_ids[2].tolist() == [30, 31, 32, 91]


def _run_mock_native_generation(
    monkeypatch,
    *,
    sampled_steps: list[list[int]],
    prompts: list[str] | None = None,
    max_new_tokens: int = 3,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    return_log_probs: bool = True,
    ignore_eos: bool = False,
    preserve_eos_token: bool = False,
    strict_generation: bool = True,
    evo2_batched_decode_size: int = 1,
    stop_after_updates: int | None = None,
    forward_error: Exception | None = None,
    events: list[str] | None = None,
    perf_counter_values: list[float] | None = None,
    peak_allocated_values: list[int] | None = None,
    peak_reserved_values: list[int] | None = None,
    expected_suppressed_token_ids: set[int] | None = None,
    generation_logits: torch.Tensor | None = None,
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
        cuda_graph_scope="none",
        cuda_graph_manager_count=0,
        cuda_graph_runner_count=0,
        cuda_graph_recorded_count=0,
        cuda_graph_replay_verified=False,
        precision_kind="bf16",
        precision_parameter_storage="bf16",
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
        lambda *_args, **_kwargs: (
            generation_logits.repeat(context.request_count, 1)
            if generation_logits is not None
            else torch.zeros((context.request_count, _MockLoopTokenizer.vocab_size))
        ),
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
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        return_log_probs=return_log_probs,
        ignore_eos=ignore_eos,
        preserve_eos_token=preserve_eos_token,
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


def test_suppress_stop_token_logits_does_not_reduce_device_logits(monkeypatch):
    """The forced-length decode path must not introduce a GPU-to-host sync per token."""

    def fail_if_reduced(_tensor):
        raise AssertionError("stop-token suppression reduced logits on the host")

    logits = torch.tensor([[1.0, 2.0, 3.0]])
    stop_token_mask = _stop_token_mask(logits, {0, 7})
    monkeypatch.setattr(torch, "isneginf", fail_if_reduced)

    filtered_logits = _suppress_stop_token_logits(logits, {0, 7}, stop_token_mask=stop_token_mask)

    assert filtered_logits.tolist() == [[float("-inf"), 2.0, 3.0]]


def test_suppress_stop_token_logits_rejects_suppressing_the_whole_vocabulary():
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    with pytest.raises(RuntimeError, match="every tokenizer vocabulary entry"):
        _suppress_stop_token_logits(logits, {0, 1, 2, 7})


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
        assert result.timings["generation_completion_tokens"] == 6
        assert result.timings["decode_completion_tokens"] == 4
        assert result.timings["generation_completion_tokens_per_s"] > 0
        assert result.timings["decode_completion_tokens_per_s"] > 0
    assert forward_model.calls == 4
    assert results[0].timings is not results[1].timings
    assert results[0].memory is not results[1].memory
    results[0].timings["first_result_only"] = True
    results[0].memory["first_result_only"] = 1
    assert "first_result_only" not in results[1].timings
    assert "first_result_only" not in results[1].memory


def test_native_batched_prefill_accepts_ragged_prompt_lengths(monkeypatch):
    results, context, forward_model = _run_mock_native_generation(
        monkeypatch,
        prompts=["P", "QQQ"],
        sampled_steps=[[1, 2]],
        max_new_tokens=1,
        evo2_batched_decode_size=2,
    )

    assert [result.generated_tokens for result in results] == [[1], [2]]
    assert context.prefill_chunk_lengths == [1, 3]
    assert forward_model.calls == 1


def test_native_sampler_composes_top_k_top_p_and_returns_filtered_logprob(monkeypatch):
    results, _context, _forward_model = _run_mock_native_generation(
        monkeypatch,
        sampled_steps=[[1]],
        max_new_tokens=1,
        temperature=2.0,
        top_k=3,
        top_p=0.7,
        generation_logits=torch.tensor([[4.0, 3.0, 2.0, 1.0]]),
    )

    expected = torch.log_softmax(torch.tensor([2.0, 1.5]), dim=-1)[1].item()
    assert results[0].generated_tokens == [1]
    assert results[0].generated_log_probs == pytest.approx([expected])


def test_native_batched_loop_preserves_terminal_eos_action_and_logprob(monkeypatch):
    results, _context, _forward_model = _run_mock_native_generation(
        monkeypatch,
        prompts=["P", "QQ"],
        sampled_steps=[[0, 1], [2, 2], [3, 3]],
        preserve_eos_token=True,
        evo2_batched_decode_size=2,
    )

    assert [result.generated_tokens for result in results] == [[0], [1, 2, 3]]
    assert results[0].generated_log_probs == pytest.approx([-math.log(4)])
    assert results[0].finish_reason == "stop"
    assert results[0].stopped_on_eos is True
    assert results[1].generated_log_probs == pytest.approx([-math.log(4)] * 3)


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
        preserve_eos_token=True,
    )

    assert results[0].generated_tokens == [1, 0]
    assert results[0].generated_log_probs == pytest.approx([-math.log(4), -math.log(4)])
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


def test_shared_dynamic_context_reports_cold_setup_and_reuses_same_request_shape(monkeypatch):
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
        static_contexts={},
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

    def validate_capture(_nd, *, expected_request_counts):
        assert expected_request_counts == frozenset({2})
        events.append("validate")

    monkeypatch.setattr(infer_module, "_validate_cuda_graph_capture", validate_capture)

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
        max_active_requests=2,
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
        "validate",
        "sync",
        "reset",
    ]


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


def _predict_script_path() -> Path:
    """Return the source prediction entry point used for teacher-forced replay."""
    return _recipe_root() / "src" / "bionemo" / "evo2" / "run" / "predict.py"


def _write_prompts_jsonl(prompt_file: Path, prompts: list[tuple[str, str]]) -> None:
    """Write a list of (id, prompt) pairs into a JSONL file."""
    with open(prompt_file, "w") as f:
        f.writelines(json.dumps({"id": prompt_id, "prompt": prompt_text}) + "\n" for prompt_id, prompt_text in prompts)


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
        "--evo2-batched-decode-size",
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
    if max_batch_size > 1:
        combined_output = f"{result.stdout}\n{result.stderr}"
        assert "[evo2-native] opt-in batched decode active" in combined_output
        assert "[evo2-native] batched prompt prefill: requests=2" in combined_output
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
):
    """A short prefill should be invariant when packed with a longer prompt.

    The two-prompt run explicitly enables ragged batched prefill and decode in one dynamic context.
    The short prompt's completion and log-prob must match whether it is submitted alone or alongside
    a 256-token prompt.
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
        use_subquadratic_ops=False,
    )
    padded_records = _run_infer_prompt_file(
        mbridge_checkpoint_path=mbridge_checkpoint_path,
        prompt_file=padded_prompt_file,
        output_file=tmp_path / "padded_output.jsonl",
        max_batch_size=2,
        use_subquadratic_ops=False,
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
    top_p: float = 0.0,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    max_batch_size: int | None = None,
    evo2_batched_decode_size: int | None = None,
    cuda_graph_impl: str | None = None,
    expected_log_substrings: tuple[str, ...] = (),
    extra_args: tuple[str, ...] = (),
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
        top_p: Top-p sampling parameter, applied after top-k.
        seed: Random seed for reproducibility.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.
        max_batch_size: If set, pass --max-batch-size to the CLI.
        evo2_batched_decode_size: If set, pass --evo2-batched-decode-size to the CLI.
        cuda_graph_impl: If set, pass --cuda-graph-impl.
        expected_log_substrings: Strings that must appear in stdout or stderr.
        extra_args: Additional CLI arguments appended to the infer command.

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
        "--top-p",
        str(top_p),
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
    cmd.extend(extra_args)

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


def _run_prediction_forward_replay(
    *,
    mbridge_checkpoint_path: Path,
    sequences: dict[str, str],
    work_dir: Path,
) -> dict[str, dict[str, torch.Tensor]]:
    """Run one rectangular prediction forward and return raw logits keyed by FASTA ID."""
    fasta_path = work_dir / "replay.fasta"
    fasta_path.write_text("".join(f">{sequence_id}\n{sequence}\n" for sequence_id, sequence in sequences.items()))
    output_dir = work_dir / "replay-output"
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        str(_predict_script_path()),
        "--fasta",
        str(fasta_path),
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--output-dir",
        str(output_dir),
        "--micro-batch-size",
        str(len(sequences)),
        "--write-interval",
        "epoch",
        "--no-sequence-packing",
    ]
    env = copy.deepcopy(PRETEST_ENV)
    env["PYTHONPATH"] = str(_recipe_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900, env=env)
    assert result.returncode == 0, (
        f"teacher-forced predict command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    prediction_files = sorted(output_dir.glob("predictions__rank_*__dp_rank_*.pt"))
    assert len(prediction_files) == 1, f"Expected one replay prediction file, found {prediction_files}"
    predictions = torch.load(prediction_files[0], map_location="cpu", weights_only=True)
    index_by_id = json.loads((output_dir / "seq_idx_map.json").read_text())
    id_by_index = {int(index): sequence_id for sequence_id, index in index_by_id.items()}

    replay_by_id = {}
    for row, original_index in enumerate(predictions["seq_idx"].tolist()):
        pad_mask = predictions["pad_mask"][row].bool()
        valid_length = int(pad_mask.sum().item())
        assert pad_mask[:valid_length].all() and not pad_mask[valid_length:].any()
        replay_by_id[id_by_index[int(original_index)]] = {
            "tokens": predictions["tokens"][row, :valid_length],
            "token_logits": predictions["token_logits"][row, :valid_length],
            "pad_mask": pad_mask,
        }
    assert set(replay_by_id) == set(sequences)
    return replay_by_id


def _target_preserving_selected_action_log_probs(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    stop_token_ids: set[int],
) -> torch.Tensor:
    """Reconstruct generation support, union sampled actions, and score those actions."""
    logits = _suppress_stop_token_logits(logits.float(), stop_token_ids)
    ordinary_log_probs = _sampling_log_probs_from_logits(
        logits,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    support = ordinary_log_probs.isfinite()
    support.scatter_(1, target_token_ids.long().unsqueeze(1), True)
    scaled_logits = logits / temperature if temperature != 1.0 else logits
    replay_log_probs = torch.log_softmax(scaled_logits.masked_fill(~support, float("-inf")), dim=-1)
    return replay_log_probs.gather(1, target_token_ids.long().unsqueeze(1)).squeeze(1)


def _run_segmented_parallel_infer_probe(
    *,
    checkpoint_path: Path,
    work_dir: Path,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    context_parallel_size: int = 1,
) -> tuple[dict[str, dict], list[dict]]:
    """Run unequal prompts together and return outputs plus per-rank kernel proof."""
    world_size = tensor_parallel_size * pipeline_parallel_size * context_parallel_size
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Packed parallel probe needs {world_size} GPUs, found {torch.cuda.device_count()}")

    prompt_file = work_dir / "prompts.jsonl"
    output_file = work_dir / "outputs.jsonl"
    probe_dir = work_dir / "kernel-proof"
    _write_prompts_jsonl(
        prompt_file,
        [
            ("short", "ACGTACGTACGTACGTA"),
            ("long", "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC"),
        ],
    )
    launcher = Path(__file__).with_name("packed_parallel_probe.py")
    command = [
        "torchrun",
        "--standalone",
        "--nproc_per_node",
        str(world_size),
        "--nnodes",
        "1",
        str(launcher),
        "infer",
        "--ckpt-dir",
        str(checkpoint_path),
        "--prompt-file",
        str(prompt_file),
        "--output-file",
        str(output_file),
        "--prompt-batch-size",
        "2",
        "--max-batch-size",
        "2",
        "--max-new-tokens",
        "20",
        "--max-seq-length",
        "128",
        "--temperature",
        "1",
        "--top-k",
        "1",
        "--seed",
        "42",
        "--ignore-eos",
        "--strict-generation",
        "--return-log-probs",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_parallel_size),
        "--context-parallel-size",
        str(context_parallel_size),
    ]
    env = copy.deepcopy(PRETEST_ENV)
    env["EVO2_PACKED_PROBE_DIR"] = str(probe_dir)
    env["PYTHONPATH"] = str(_recipe_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=900, env=env)
    assert result.returncode == 0, (
        f"Packed parallel infer probe failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )

    records = {record["id"]: record for record in _read_jsonl_results(output_file)}
    proofs = [json.loads((probe_dir / f"rank-{rank}.json").read_text()) for rank in range(world_size)]
    assert set(records) == {"short", "long"}
    assert {proof["rank"] for proof in proofs} == set(range(world_size))
    assert all(proof["calls"] > 0 for proof in proofs)
    assert all(proof["max_segments"] == 2 for proof in proofs)
    assert {operator for proof in proofs for operator in proof["operator_counts"]} == {
        "hyena",
        "hyena_medium_conv",
        "hyena_short_conv",
    }
    for record in records.values():
        assert record["usage"]["completion_tokens"] == 20
        assert record["timings"]["timing_request_count"] == 2
        assert record["timings"]["cuda_graph_scope"] == "block"
        assert record["timings"]["cuda_graph_manager_count"] == 1
        assert record["timings"]["cuda_graph_runner_count"] == 1
        assert record["timings"]["cuda_graph_recorded_count"] == record["timings"]["cuda_graph_runner_count"]
        assert record["timings"]["cuda_graph_replay_verified"] is True
    return records, proofs


def _assert_parallel_probe_matches_baseline(actual: dict[str, dict], expected: dict[str, dict]) -> None:
    assert set(actual) == set(expected)
    for request_id in expected:
        assert actual[request_id]["completion_token_ids"] == expected[request_id]["completion_token_ids"]
        torch.testing.assert_close(
            torch.tensor(actual[request_id]["logprobs"]["completion_logprobs"]),
            torch.tensor(expected[request_id]["logprobs"]["completion_logprobs"]),
            rtol=2e-2,
            atol=5e-2,
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
        pytest.param(1, 2, id="tp=1,cp=2"),
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

    This uses a common prefix length across the DNA accuracy prompts for both serial and batched
    subprocess inference. Greedy
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


@pytest.fixture(scope="module")
def segmented_infer_baseline_1b(mbridge_checkpoint_path, tmp_path_factory) -> dict[str, dict]:
    records, _ = _run_segmented_parallel_infer_probe(
        checkpoint_path=mbridge_checkpoint_path,
        work_dir=tmp_path_factory.mktemp("packed-infer-baseline-1b"),
    )
    return records


@pytest.mark.parametrize(
    ("pipeline_parallel_size", "context_parallel_size"),
    [
        pytest.param(1, 2, id="cp=2"),
        pytest.param(2, 1, id="pp=2"),
    ],
)
@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Packed CP/PP inference requires at least two GPUs")
def test_segmented_packed_infer_executes_on_every_cp_or_pp_rank(
    mbridge_checkpoint_path,
    segmented_infer_baseline_1b,
    tmp_path,
    pipeline_parallel_size: int,
    context_parallel_size: int,
) -> None:
    """CP/PP cannot silently replace ragged segmented prefill with a rectangular path."""
    records, _ = _run_segmented_parallel_infer_probe(
        checkpoint_path=mbridge_checkpoint_path,
        work_dir=tmp_path,
        pipeline_parallel_size=pipeline_parallel_size,
        context_parallel_size=context_parallel_size,
    )
    _assert_parallel_probe_matches_baseline(records, segmented_infer_baseline_1b)


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip 7b-1m checkpoint tests in CI due to disk space")
def test_segmented_packed_infer_executes_on_every_tp_rank(
    mbridge_checkpoint_7b_1m_path,
    tmp_path_factory,
) -> None:
    """Production TP must execute every segmented operator family on every rank."""
    baseline, _ = _run_segmented_parallel_infer_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("packed-infer-baseline-7b"),
    )
    parallel, _ = _run_segmented_parallel_infer_probe(
        checkpoint_path=mbridge_checkpoint_7b_1m_path,
        work_dir=tmp_path_factory.mktemp("packed-infer-tp2-7b"),
        tensor_parallel_size=2,
    )
    _assert_parallel_probe_matches_baseline(parallel, baseline)


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
        # CP-only config
        pytest.param(1, 1, 2, id="tp=1,pp=1,cp=2"),
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
# checkpoint (real weights, validates the mechanism + correctness, not just shapes). The mixed
# request covers packed short-to-multi-block prefill, prompt-dependent decode, and longer recurrence.
# Separate runs cover chunked-prefill equivalence, FP8, and TP-non-divisible batches.
# Greedy decoding (top_k=1) keeps the assertions deterministic.

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


@pytest.mark.timeout(900)
def test_native_staggered_kv_rollover_matches_serial(mbridge_checkpoint_path, tmp_path):
    """Heterogeneous packed decode stays request-exact across repeated 256-token rollovers.

    The second row begins exactly at a page boundary and rolls over one step before the first.
    Both requests then cross a second page during the continuation. Before Evo2 reserved pages in
    place, mcore's pause/resume bookkeeping permuted the two rows at the first staggered rollover,
    cross-wiring output ownership and positional Hyena state. Greedy packed output is compared with
    serial output. Combined top-k/top-p sampling retains a same-batch single-page trajectory
    reference, then every selected action is scored using a prediction forward plus reconstructed
    target-preserving support. That portable check catches row ownership defects that
    trajectory-only comparisons can miss; it is not an exact NeMo-RL worker replay.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")

    prompts = [
        ("near-boundary", "ACGT" * 63 + "ACG"),
        ("on-boundary", "TGCA" * 64),
    ]
    assert [len(prompt) for _, prompt in prompts] == [255, 256]
    prompt_file = tmp_path / "rollover_prompts.jsonl"
    _write_prompts_jsonl(prompt_file, prompts)
    common_args = (
        "--max-seq-length",
        "768",
        "--inference-dynamic-batching-block-size",
        str(KV_BLOCK_SIZE_TOKENS),
        "--ignore-eos",
        "--strict-generation",
        "--return-log-probs",
    )

    serial = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=tmp_path / "rollover_serial.jsonl",
        max_new_tokens=270,
        top_k=1,
        max_batch_size=1,
        evo2_batched_decode_size=1,
        extra_args=common_args,
    )
    packed = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=tmp_path / "rollover_packed.jsonl",
        max_new_tokens=270,
        top_k=1,
        max_batch_size=2,
        evo2_batched_decode_size=2,
        expected_log_substrings=("[evo2-native] batched prompt prefill: requests=2",),
        extra_args=common_args,
    )

    serial_by_id = {record["id"]: record for record in serial}
    packed_by_id = {record["id"]: record for record in packed}
    _assert_parallel_probe_matches_baseline(packed_by_id, serial_by_id)
    assert all(record["usage"]["completion_tokens"] == 270 for record in packed)

    top_p_reference_args = (
        "--max-seq-length",
        "768",
        "--inference-dynamic-batching-block-size",
        "1024",
        "--ignore-eos",
        "--strict-generation",
        "--return-log-probs",
    )
    top_p_reference = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=tmp_path / "top_p_single_page.jsonl",
        max_new_tokens=270,
        top_k=5,
        top_p=0.999,
        seed=1234,
        max_batch_size=2,
        evo2_batched_decode_size=2,
        extra_args=top_p_reference_args,
    )
    top_p_rollover = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=tmp_path / "top_p_rollover.jsonl",
        max_new_tokens=270,
        top_k=5,
        top_p=0.999,
        seed=1234,
        max_batch_size=2,
        evo2_batched_decode_size=2,
        extra_args=common_args,
    )

    top_p_reference_by_id = {record["id"]: record for record in top_p_reference}
    top_p_rollover_by_id = {record["id"]: record for record in top_p_rollover}
    _assert_parallel_probe_matches_baseline(top_p_rollover_by_id, top_p_reference_by_id)
    assert all(_is_dna_completion(record["completion"]) for record in top_p_rollover)
    replay_by_id = _run_prediction_forward_replay(
        mbridge_checkpoint_path=mbridge_checkpoint_path,
        sequences={record["id"]: record["prompt"] + record["completion"] for record in top_p_rollover},
        work_dir=tmp_path,
    )

    diagnostics = {}
    for request_id, record in top_p_rollover_by_id.items():
        prompt_ids = record["prompt_token_ids"]
        completion_ids = record["completion_token_ids"]
        full_ids = torch.tensor(prompt_ids + completion_ids, dtype=torch.long)
        replay = replay_by_id[request_id]
        torch.testing.assert_close(replay["tokens"], full_ids, rtol=0, atol=0)
        assert replay["pad_mask"][: full_ids.numel()].all()
        assert not replay["pad_mask"][full_ids.numel() :].any()

        action_positions = torch.arange(len(completion_ids), dtype=torch.long) + len(prompt_ids) - 1
        assert int(action_positions[0]) == len(prompt_ids) - 1
        assert int(action_positions[-1]) == full_ids.numel() - 2
        assert replay["token_logits"].shape[0] == full_ids.numel()
        selected_replay = _target_preserving_selected_action_log_probs(
            replay["token_logits"].index_select(0, action_positions),
            torch.tensor(completion_ids, dtype=torch.long),
            temperature=1.0,
            top_k=5,
            top_p=0.999,
            stop_token_ids={0},
        )
        generated = torch.tensor(record["logprobs"]["completion_logprobs"], dtype=torch.float32)
        delta = (selected_replay - generated).abs()
        absolute_positions = torch.arange(len(completion_ids), dtype=torch.long) + len(prompt_ids)
        residues = absolute_positions.remainder(KV_BLOCK_SIZE_TOKENS)
        residue_means = {
            residue: float(delta[residues == residue].mean().item())
            for residue in (1, 249, 9, 2, 10)
            if (residues == residue).any()
        }
        diagnostics[request_id] = {
            "median_abs_delta": float(delta.median().item()),
            "mean_abs_delta": float(delta.mean().item()),
            "max_abs_delta": float(delta.max().item()),
            "deltas_over_4": int((delta > 4.0).sum().item()),
            # NeMo-RL's sequence guard averages the per-token multiplicative error. Retain the
            # exp(mean(abs(delta))) form too so neither aggregation can hide a boundary-local spike.
            "mean_token_mult_prob_error": float(delta.exp().mean().item()),
            "exp_mean_abs_delta": float(delta.mean().exp().item()),
            "residue_mean_abs_delta": residue_means,
        }
        watched_phase_limit = max(0.1, 3.0 * diagnostics[request_id]["mean_abs_delta"])
        assert torch.isfinite(selected_replay).all(), diagnostics
        assert diagnostics[request_id]["deltas_over_4"] == 0, diagnostics
        assert diagnostics[request_id]["max_abs_delta"] <= 1.5, diagnostics
        assert diagnostics[request_id]["mean_token_mult_prob_error"] <= 1.5, diagnostics
        assert diagnostics[request_id]["exp_mean_abs_delta"] <= 1.5, diagnostics
        assert all(mean <= watched_phase_limit for mean in residue_means.values()), diagnostics

    print("ROLLOVER_SELECTED_ACTION_REPLAY " + json.dumps(diagnostics, sort_keys=True))


def test_native_mixed_prompt_contract(mbridge_checkpoint_path, tmp_path):
    """Cover independent native-engine prompt contracts with one model load.

    The ragged batch includes duplicate and different prompts, a prompt shorter than the medium-FIR
    ring, a prompt spanning multiple paged-KV blocks, and a phylogenetic prompt. This replaces
    separate subprocess tests whose repeated model setup dominated their runtime.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")

    prompts = [
        ("basic", "ATCG" * 16),
        ("same-a", PROMPT_1),
        ("same-b", PROMPT_1),
        ("different", PROMPT_2),
        ("short", "ACGTACGTAACCGGTT"),
        ("long", LONG_DNA_PROMPT),
        (
            "phylogenetic",
            "|d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;o__Enterobacterales;"
            "f__Enterobacteriaceae;g__Escherichia;s__Escherichia|",
        ),
    ]
    prompt_file = tmp_path / "mixed_prompts.jsonl"
    _write_prompts_jsonl(prompt_file, prompts)
    records = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=tmp_path / "mixed_outputs.jsonl",
        max_new_tokens=100,
        top_k=1,
        seed=42,
        max_batch_size=len(prompts),
        evo2_batched_decode_size=len(prompts),
        expected_log_substrings=("[evo2-native] batched prompt prefill: requests=7",),
        extra_args=(
            "--max-seq-length",
            "1024",
            "--inference-dynamic-batching-block-size",
            str(KV_BLOCK_SIZE_TOKENS),
            "--ignore-eos",
            "--strict-generation",
        ),
    )

    by_id = {record["id"]: record for record in records}
    assert set(by_id) == {prompt_id for prompt_id, _ in prompts}
    first_timing = records[0]["timings"]
    assert first_timing["cuda_graph_runner_count"] == 1
    assert first_timing["cuda_graph_recorded_count"] == 1
    assert first_timing["cuda_graph_replay_verified"] is True
    for prompt_id, prompt in prompts:
        record = by_id[prompt_id]
        assert record["prompt"] == prompt
        assert record["finish_reason"] == "length"
        assert record["usage"]["prompt_tokens"] == len(prompt)
        assert record["usage"]["completion_tokens"] == 100

    assert len(prompts[4][1]) < 127
    assert len(prompts[5][1]) > KV_BLOCK_SIZE_TOKENS
    for prompt_id in ("basic", "same-a", "same-b", "different", "short", "long"):
        assert _is_dna_completion(by_id[prompt_id]["completion"]), (
            f"non-DNA completion for {prompt_id}: {by_id[prompt_id]['completion']!r}"
        )
    assert by_id["phylogenetic"]["completion"]
    assert by_id["same-a"]["completion_token_ids"] == by_id["same-b"]["completion_token_ids"]
    assert by_id["same-a"]["completion"] != by_id["different"]["completion"]


def test_native_dynamic_chunked_prefill_matches_full_prefill(mbridge_checkpoint_path, tmp_path):
    """Chunked prefill yields the same greedy continuation as single-shot (full) prefill.

    This is the prefix-invariance idea (same prompt -> same completion two ways) applied to chunked
    prefill: prefilling the whole prompt in one forward vs splitting it across multiple prefill
    forwards (``--enable-chunked-prefill`` with a per-step token budget below the prompt length) must
    produce identical tokens under greedy decoding, since chunked prefill is only a memory-bounded way
    to compute the same prefill. This pins the equivalence to full prefill and guards the Hyena
    chunked-prefill fix: the FIR/IIR
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


def test_native_dynamic_fp8_chunked_prefill(mbridge_checkpoint_path, tmp_path):
    """Megatron FP8 inference runs through chunked prefill and graphed decode.

    Confirms the fp8 token-padding path (``prepare_model_for_fp8_inference``, applied in
    ``setup_inference_engine`` when the recipe turns on fp8) coexists with the multi-block Hyena
    block-step and CUDA-graphed decode. The BF16 test above already compares full and chunked prefill,
    while the broader model suite covers ordinary Megatron FP8 execution.
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
    fp8_args = [
        "--mixed-precision-recipe",
        "bf16_with_fp8_current_scaling_mixed",
        "--fp8-all-layers",
    ]

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
    assert chunked["usage"]["prompt_tokens"] == n_prompt_tokens
    assert chunked["usage"]["completion_tokens"] == 20
    assert _is_dna_completion(chunked["completion"]), f"non-DNA fp8 completion: {chunked['completion']!r}"


@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip 7b-1m checkpoint tests in single-GPU CI")
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
