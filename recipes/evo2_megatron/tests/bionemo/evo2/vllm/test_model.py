# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from contextlib import contextmanager
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bionemo.evo2.vllm.model as evo2_model
from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.hyena import Evo2HyenaDecoderLayer
from bionemo.evo2.vllm.layers import Evo2AttentionDecoderLayer
from bionemo.evo2.vllm.model import Evo2ForCausalLM
from bionemo.evo2.vllm.weights import refresh_derived_filters


DEVICE = "cuda"
DTYPE = torch.float32
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _config(pattern: str = "SDH", *, max_position_embeddings: int = 128) -> Evo2Config:
    return Evo2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=len(pattern),
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=max_position_embeddings,
        hybrid_override_pattern=pattern,
        short_conv_length=3,
        hcs_filter_length=7,
        hcm_filter_length=4,
        hcl_state_size=4,
        num_groups_hyena=16,
        num_groups_hyena_medium=4,
        num_groups_hyena_short=4,
        rms_norm_eps=1e-6,
    )


def _vllm_config(
    config: Evo2Config,
    model_directory: str,
    *,
    max_model_len: int | None = None,
):
    from vllm.config import CacheConfig, ModelConfig, VllmConfig

    from bionemo.evo2.vllm.plugin import register

    register()
    config.save_pretrained(model_directory)
    model_config = ModelConfig(
        model=model_directory,
        tokenizer=model_directory,
        dtype=DTYPE,
        max_model_len=config.max_position_embeddings if max_model_len is None else max_model_len,
        skip_tokenizer_init=True,
        model_impl="vllm",
    )

    cache_config = CacheConfig(
        block_size=16,
        enable_prefix_caching=False,
        mamba_cache_mode="none",
        mamba_cache_dtype="float32",
        mamba_ssm_cache_dtype="float32",
    )
    return VllmConfig(model_config=model_config, cache_config=cache_config)


@contextmanager
def _model_context(config: Evo2Config):
    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    with TemporaryDirectory() as temporary_directory:
        vllm_config = _vllm_config(config, temporary_directory)
        with set_current_vllm_config(vllm_config), set_default_torch_dtype(DTYPE):
            torch.cuda.set_device(0)
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temporary_directory}/distributed_init",
                local_rank=0,
                backend="nccl",
            )
            try:
                initialize_model_parallel(1, 1)
                yield vllm_config
            finally:
                destroy_model_parallel()
                destroy_distributed_environment()


def _metadata(starts: torch.Tensor, slots: torch.Tensor):
    from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata

    request_count = slots.numel()
    return Mamba1AttentionMetadata(
        num_prefills=request_count,
        num_prefill_tokens=int(starts[-1].item()),
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=request_count,
        has_initial_states_p=torch.zeros(request_count, device=DEVICE, dtype=torch.bool),
        query_start_loc_p=starts,
        num_computed_tokens_p=None,
        state_indices_tensor_p=slots,
        state_indices_tensor_d=None,
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        seq_lens=torch.diff(starts),
    )


def _bind_hyena_cache_and_metadata(model: Evo2ForCausalLM, starts: torch.Tensor, slots: torch.Tensor):
    metadata = {}
    for layer in model.model.decoder.layers:
        if not isinstance(layer, Evo2HyenaDecoderLayer):
            continue
        projection_shape, operator_shape = layer.mixer.get_state_shape()
        layer.mixer.kv_cache = (
            torch.zeros((3, *projection_shape), device=DEVICE, dtype=torch.float32),
            torch.zeros((3, *operator_shape), device=DEVICE, dtype=torch.float32),
        )
        metadata[layer.mixer.prefix] = _metadata(starts, slots)
    return metadata


def _randomize(model: torch.nn.Module, seed: int = 67) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("layer_norm_weight") or name.endswith("final_norm.weight"):
                parameter.copy_(
                    1 + 0.03 * torch.randn(parameter.shape, device=DEVICE, dtype=parameter.dtype, generator=generator)
                )
            else:
                parameter.copy_(
                    0.04 * torch.randn(parameter.shape, device=DEVICE, dtype=parameter.dtype, generator=generator)
                )
    refresh_derived_filters(model)


def test_model_state_contract_has_no_length_ceiling_and_copies_whole_ds_blocks(monkeypatch) -> None:
    config = _config(max_position_embeddings=10_240)
    monkeypatch.setenv("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    with TemporaryDirectory() as temporary_directory:
        vllm_config = _vllm_config(config, temporary_directory, max_model_len=1_000_000)
        length_contract = evo2_model.evo2_context_length_contract(vllm_config)
        assert Evo2ForCausalLM.get_mamba_state_shape_from_config(vllm_config) == ((48, 2), (16, 6))
        assert Evo2ForCausalLM.get_mamba_state_dtype_from_config(vllm_config) == (
            torch.float32,
            torch.float32,
        )
        assert length_contract["checkpoint_declared_max_position_embeddings"] == 10_240
        assert length_contract["resolved_max_model_len"] == 1_000_000
        assert length_contract["rotary_max_position_embeddings"] == 1_000_000
        assert length_contract["hyena_state_shapes"] == [[48, 2], [16, 6]]
        assert length_contract["hyena_state_length_dependent"] is False
        assert length_contract["position_clipping"] is False
    evo2_model.reset_mamba_state_copy_stats()
    projection_copy, operator_copy = Evo2ForCausalLM.get_mamba_state_copy_func()
    projection_state = torch.arange(3 * 48 * 2, dtype=torch.float32).view(3, 48, 2)
    operator_state = torch.arange(3 * 16 * 6, dtype=torch.float32).view(3, 16, 6)
    projection_spec = projection_copy(projection_state, [2], 0, 1)
    operator_spec = operator_copy(operator_state, [1], 0, 1)

    assert projection_spec.start_addr == projection_state[2].data_ptr()
    assert projection_spec.num_elements == projection_state[2].numel()
    assert operator_spec.start_addr == operator_state[1].data_ptr()
    assert operator_spec.num_elements == operator_state[1].numel()
    assert evo2_model.get_mamba_state_copy_stats() == {
        "copy_calls": 2,
        "copied_elements": projection_state[2].numel() + operator_state[1].numel(),
        "copied_bytes": (
            projection_state[2].numel() * projection_state.element_size()
            + operator_state[1].numel() * operator_state.element_size()
        ),
    }
    with pytest.raises(ValueError, match="speculative"):
        projection_copy(projection_state, [1, 2], 0, 2)
    evo2_model.reset_mamba_state_copy_stats()
    assert evo2_model.get_mamba_state_copy_stats() == {
        "copy_calls": 0,
        "copied_elements": 0,
        "copied_bytes": 0,
    }


def test_mamba_prefix_clone_hook_records_exact_request_scoped_fp32_copy_bytes() -> None:
    module = SimpleNamespace()

    def collect(
        copy_bufs,
        kv_cache_config,
        mamba_state_copy_funcs,
        mamba_group_ids,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
        req_state,
        forward_context,
    ):
        del kv_cache_config, mamba_state_copy_funcs, accept_token_bias
        block_ids = req_state.block_ids[mamba_group_ids[0]]
        for state in forward_context["layer"].kv_cache:
            source = state[block_ids[src_block_idx]]
            destination = state[block_ids[dest_block_idx]]
            copy_bufs.src_ptrs.np[copy_bufs.offset] = source.data_ptr()
            copy_bufs.dst_ptrs.np[copy_bufs.offset] = destination.data_ptr()
            copy_bufs.sizes.np[copy_bufs.offset] = source.numel() * source.element_size()
            copy_bufs.offset += 1

    def preprocess(
        scheduler_output,
        kv_cache_config,
        cache_config,
        mamba_state_idx,
        input_batch,
        requests,
        forward_context,
        mamba_state_copy_funcs,
        copy_bufs,
    ):
        del cache_config, mamba_state_idx
        for req_id in input_batch.req_ids:
            req_state = requests[req_id]
            if req_state.num_computed_tokens:
                module.collect_mamba_copy_meta(
                    copy_bufs,
                    kv_cache_config,
                    mamba_state_copy_funcs,
                    copy_bufs.mamba_group_ids,
                    0,
                    1,
                    0,
                    req_state,
                    forward_context,
                )

    module.collect_mamba_copy_meta = collect
    module.preprocess_mamba = preprocess
    evo2_model.install_mamba_prefix_clone_proof_hook(module)
    installed = module.preprocess_mamba
    evo2_model.install_mamba_prefix_clone_proof_hook(module)
    assert module.preprocess_mamba is installed

    states = (
        torch.zeros((3, 2, 3), dtype=torch.float32),
        torch.zeros((3, 5), dtype=torch.float32),
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                layer_names=["layer"],
                kv_cache_spec=SimpleNamespace(block_size=16),
            ),
            SimpleNamespace(
                layer_names=["attention.layer"],
                kv_cache_spec=SimpleNamespace(block_size=16),
            ),
        ]
    )
    copy_bufs = SimpleNamespace(
        offset=0,
        src_ptrs=SimpleNamespace(np=np.zeros(8, dtype=np.int64)),
        dst_ptrs=SimpleNamespace(np=np.zeros(8, dtype=np.int64)),
        sizes=SimpleNamespace(np=np.zeros(8, dtype=np.int32)),
        mamba_group_ids=[0],
        mamba_spec=SimpleNamespace(block_size=16),
    )
    requests = {
        "miss": SimpleNamespace(
            req_id="miss",
            num_computed_tokens=0,
            num_prompt_tokens=32,
            output_token_ids=[],
            block_ids=[[2, 1], [101, 102]],
        ),
        "clone": SimpleNamespace(
            req_id="clone",
            num_computed_tokens=16,
            num_prompt_tokens=32,
            output_token_ids=[],
            block_ids=[[0, 1], [101, 103]],
        ),
    }
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id="miss"),
            SimpleNamespace(req_id="clone"),
        ],
        num_scheduled_tokens={"miss": 32, "clone": 16},
    )

    evo2_model.reset_mamba_prefix_clone_stats()
    module.preprocess_mamba(
        scheduler_output,
        kv_cache_config,
        SimpleNamespace(),
        {},
        SimpleNamespace(req_ids=["miss", "clone"]),
        requests,
        {"layer": SimpleNamespace(kv_cache=states)},
        (object(), object()),
        copy_bufs,
    )

    stats = evo2_model.get_mamba_prefix_clone_stats()
    assert stats["cache_miss_count"] == 1
    assert stats["cache_miss_request_ids"] == ["miss"]
    assert len(stats["prefix_sources"]) == 1
    source = stats["prefix_sources"][0]
    assert source["request_id"] == "miss"
    assert source["prompt_tokens"] == 32
    assert len(source["snapshots"]) == 1
    source_snapshot = source["snapshots"][0]
    assert source_snapshot["snapshot_index"] == 0
    assert source_snapshot["num_computed_tokens_before_step"] == 0
    assert source_snapshot["num_scheduled_tokens"] == 32
    assert source_snapshot["directly_observed_prefix_tokens"] == 16
    assert source_snapshot["attention_kv_groups"] == [
        {
            "kv_cache_group_id": 1,
            "layer_names": ["attention.layer"],
            "block_size_tokens": 16,
            "physical_block_count": 1,
            "physical_block_ids": [101],
            "physical_block_ids_sha256": source_snapshot["attention_kv_groups"][0]["physical_block_ids_sha256"],
        }
    ]
    assert stats["clone_count"] == 1
    record = stats["requests"][0]
    assert record["request_id"] == "clone"
    assert record["source_miss_request_id"] == "miss"
    assert record["source_snapshot_index"] == 0
    assert record["attention_kv_identity_verified"] is True
    assert record["num_computed_tokens"] == 16
    assert record["prompt_tokens"] == 32
    assert record["block_size"] == 16
    assert record["copy_entries"] == 2
    assert record["copied_elements"] == 11
    assert record["copied_bytes"] == 44
    assert record["expected_copy_entries"] == 2
    assert record["expected_copied_elements"] == 11
    assert record["expected_copied_bytes"] == 44
    assert record["all_state_dtypes_fp32"] is True
    assert record["runtime_state_layout"] == [
        {
            "kv_cache_group_id": 0,
            "layer_name": "layer",
            "state_index": 0,
            "dtype": "torch.float32",
            "state_shape": [3, 2, 3],
            "block_shape": [2, 3],
            "copied_elements": 6,
            "copied_bytes": 24,
        },
        {
            "kv_cache_group_id": 0,
            "layer_name": "layer",
            "state_index": 1,
            "dtype": "torch.float32",
            "state_shape": [3, 5],
            "block_shape": [5],
            "copied_elements": 5,
            "copied_bytes": 20,
        },
    ]
    assert record["source_attention_kv_groups"] == source_snapshot["attention_kv_groups"]
    assert record["reused_attention_kv_groups"] == [
        {
            "kv_cache_group_id": 1,
            "layer_names": ["attention.layer"],
            "block_size_tokens": 16,
            "physical_block_count": 1,
            "physical_block_ids": [101],
            "physical_block_ids_sha256": record["reused_attention_kv_groups"][0]["physical_block_ids_sha256"],
        }
    ]
    assert record["state_copies"] == [
        {
            "kv_cache_group_id": 0,
            "layer_name": "layer",
            "state_index": 0,
            "dtype": "torch.float32",
            "state_shape": [3, 2, 3],
            "block_shape": [2, 3],
            "source_logical_block_index": 0,
            "destination_logical_block_index": 1,
            "source_physical_block_id": 0,
            "destination_physical_block_id": 1,
            "source_data_ptr": states[0][0].data_ptr(),
            "destination_data_ptr": states[0][1].data_ptr(),
            "copied_elements": 6,
            "copied_bytes": 24,
        },
        {
            "kv_cache_group_id": 0,
            "layer_name": "layer",
            "state_index": 1,
            "dtype": "torch.float32",
            "state_shape": [3, 5],
            "block_shape": [5],
            "source_logical_block_index": 0,
            "destination_logical_block_index": 1,
            "source_physical_block_id": 0,
            "destination_physical_block_id": 1,
            "source_data_ptr": states[1][0].data_ptr(),
            "destination_data_ptr": states[1][1].data_ptr(),
            "copied_elements": 5,
            "copied_bytes": 20,
        },
    ]

    evo2_model.reset_mamba_prefix_clone_stats(reset_prefix_sources=False)
    later_wave_stats = evo2_model.get_mamba_prefix_clone_stats()
    assert later_wave_stats["cache_miss_count"] == 0
    assert later_wave_stats["clone_count"] == 0
    assert later_wave_stats["prefix_sources"] == stats["prefix_sources"]

    evo2_model.reset_mamba_prefix_clone_stats()
    requests["clone"].block_ids[1][0] = 999
    with pytest.raises(AssertionError, match="physical block IDs do not exactly match"):
        module.preprocess_mamba(
            scheduler_output,
            kv_cache_config,
            SimpleNamespace(),
            {},
            SimpleNamespace(req_ids=["miss", "clone"]),
            requests,
            {"layer": SimpleNamespace(kv_cache=states)},
            (object(), object()),
            copy_bufs,
        )
    evo2_model.reset_mamba_prefix_clone_stats()


@CUDA_REQUIRED
def test_model_builds_hybrid_stack_with_native_checkpoint_paths() -> None:
    config = _config("SDH*")
    with _model_context(config) as vllm_config:
        model = Evo2ForCausalLM(vllm_config=vllm_config).to(DEVICE).eval()

        assert isinstance(model.model.decoder.layers[0], Evo2HyenaDecoderLayer)
        assert isinstance(model.model.decoder.layers[1], Evo2HyenaDecoderLayer)
        assert isinstance(model.model.decoder.layers[2], Evo2HyenaDecoderLayer)
        assert isinstance(model.model.decoder.layers[3], Evo2AttentionDecoderLayer)
        assert model.model.decoder.layers[0].mixer.operator_type == "S"
        assert model.model.decoder.layers[1].mixer.operator_type == "D"
        assert model.model.decoder.layers[2].mixer.operator_type == "H"
        assert model.lm_head.weight is model.model.embedding.word_embeddings.weight
        cache_spec = model.model.decoder.layers[0].mixer.get_kv_cache_spec(vllm_config)
        assert cache_spec is not None
        assert cache_spec.shapes == ((48, 2), (16, 6))
        assert cache_spec.dtypes == (torch.float32, torch.float32)
        assert cache_spec.block_size == 128
        assert cache_spec.mamba_cache_mode == "none"
        assert cache_spec.mamba_type == "mamba1"

        names = set(dict(model.named_parameters(remove_duplicate=False)))
        assert "model.embedding.word_embeddings.weight" in names
        assert "model.decoder.layers.0.mixer.hyena_proj_conv.short_conv_weight" in names
        assert "model.decoder.layers.1.mixer.mixer.filter.h" in names
        assert "model.decoder.layers.2.mixer.mixer.filter.R" in names
        assert "model.decoder.layers.3.self_attention.linear_qkv.weight" in names
        assert "model.decoder.final_norm.weight" in names
        logits = model.compute_logits(torch.randn((5, config.hidden_size), device=DEVICE, dtype=DTYPE))
        assert logits is not None
        assert logits.shape == (5, config.vocab_size)


@CUDA_REQUIRED
def test_all_hyena_model_forward_round_trips_native_weight_stream() -> None:
    from vllm.forward_context import set_forward_context

    config = _config("SDH")
    with _model_context(config) as vllm_config:
        model = Evo2ForCausalLM(vllm_config=vllm_config).to(DEVICE).eval()
        _randomize(model)
        source_weights = []
        for name, parameter in model.named_parameters():
            if not name.startswith("model."):
                continue
            checkpoint_tensor = parameter.detach().clone()
            if name == "model.embedding.word_embeddings.weight":
                checkpoint_tensor = checkpoint_tensor[: config.vocab_size]
            source_weights.append((name.removeprefix("model."), checkpoint_tensor))
        starts = torch.tensor([0, 2, 5], device=DEVICE, dtype=torch.int32)
        slots = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)
        metadata = _bind_hyena_cache_and_metadata(model, starts, slots)
        input_ids = torch.tensor([1, 2, 3, 4, 5], device=DEVICE, dtype=torch.long)
        positions = torch.tensor([0, 1, 0, 1, 2], device=DEVICE, dtype=torch.long)
        with set_forward_context(metadata, vllm_config, num_tokens=input_ids.numel()):
            expected = model(input_ids, positions)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        loaded = model.load_weights(iter(source_weights))
        assert loaded == set(dict(model.named_parameters()))
        metadata = _bind_hyena_cache_and_metadata(model, starts, slots)
        with set_forward_context(metadata, vllm_config, num_tokens=input_ids.numel()):
            actual = model(input_ids, positions)

        assert actual.shape == (5, config.hidden_size)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
def test_model_delegates_cuda_graph_state_buffers_to_vllm_cache_manager() -> None:
    config = _config("S")
    with _model_context(config) as vllm_config:
        model = Evo2ForCausalLM(vllm_config=vllm_config).to(DEVICE).eval()

        class CacheManager:
            def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
                return ("copy", input_buffers, kwargs)

            def get_seqlen_agnostic_capture_inputs(self, batch_size):
                return ("capture", batch_size)

        model.mamba_cache = CacheManager()
        assert model.copy_inputs_before_cuda_graphs({"state": 1}, foo=2) == (
            "copy",
            {"state": 1},
            {"foo": 2},
        )
        assert model.get_seqlen_agnostic_capture_inputs(96) == ("capture", 96)
