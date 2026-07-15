# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from contextlib import contextmanager
from itertools import pairwise
from tempfile import TemporaryDirectory

import pytest
import torch
from torch.nn import functional
from torch.profiler import ProfilerActivity, profile

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.hyena import Evo2HyenaDecoderLayer, Evo2HyenaMixer
from bionemo.evo2.vllm.weights import refresh_derived_filters


DEVICE = "cuda"
DTYPE = torch.float32
CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _config(symbol: str) -> Evo2Config:
    return Evo2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        hybrid_override_pattern=symbol,
        short_conv_length=3,
        hcs_filter_length=7,
        hcm_filter_length=4,
        hcl_state_size=4,
        num_groups_hyena=16,
        num_groups_hyena_medium=4,
        num_groups_hyena_short=4,
        rms_norm_eps=1e-6,
    )


@contextmanager
def _vllm_module_context():
    from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    cache_config = CacheConfig(
        block_size=16,
        enable_prefix_caching=False,
        mamba_block_size=16,
        mamba_cache_mode="none",
        mamba_cache_dtype="float32",
        mamba_ssm_cache_dtype="float32",
    )
    vllm_config = VllmConfig(cache_config=cache_config)
    with (
        TemporaryDirectory() as temporary_directory,
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(DTYPE),
    ):
        torch.cuda.set_device(0)
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temporary_directory}/distributed_init",
            local_rank=0,
            backend="nccl",
        )
        initialize_model_parallel(1, 1)
        try:
            yield vllm_config, cache_config
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def _metadata(
    *,
    query_start_loc_p: torch.Tensor | None = None,
    state_indices_tensor_p: torch.Tensor | None = None,
    has_initial_states_p: torch.Tensor | None = None,
    state_indices_tensor_d: torch.Tensor | None = None,
):
    from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata

    num_prefills = 0 if state_indices_tensor_p is None else state_indices_tensor_p.numel()
    num_prefill_tokens = 0 if query_start_loc_p is None else int(query_start_loc_p[-1].item())
    num_decodes = 0 if state_indices_tensor_d is None else state_indices_tensor_d.numel()
    num_decode_tokens = num_decodes
    seq_lens = torch.empty(num_decodes + num_prefills, device=DEVICE, dtype=torch.int32)
    if num_decodes:
        seq_lens[:num_decodes] = 1
    if num_prefills:
        seq_lens[num_decodes:] = torch.diff(query_start_loc_p)
    return Mamba1AttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_reqs=num_prefills + num_decodes,
        has_initial_states_p=has_initial_states_p,
        query_start_loc_p=query_start_loc_p,
        num_computed_tokens_p=None,
        state_indices_tensor_p=state_indices_tensor_p,
        state_indices_tensor_d=state_indices_tensor_d,
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        seq_lens=seq_lens,
    )


def _randomize(mixer: torch.nn.Module, seed: int = 7) -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    with torch.no_grad():
        for parameter in mixer.parameters():
            parameter.copy_(
                0.08 * torch.randn(parameter.shape, device=DEVICE, dtype=parameter.dtype, generator=generator)
            )
    refresh_derived_filters(mixer)


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return x.float() * torch.rsqrt(variance + eps) * weight.float()


def _mlp_reference(layer: Evo2HyenaDecoderLayer, hidden_states: torch.Tensor) -> torch.Tensor:
    gate_up = functional.linear(hidden_states, layer.mlp.linear_fc1.weight)
    gate, up = gate_up.chunk(2, dim=-1)
    return functional.linear(functional.silu(gate) * up, layer.mlp.linear_fc2.weight)


def _fir_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    state: torch.Tensor,
    starts: torch.Tensor,
    slots: torch.Tensor,
    has_initial: torch.Tensor,
    *,
    group_size: int,
    gated_bias: bool = False,
    flip_filter: bool = False,
) -> torch.Tensor:
    starts_list = starts.tolist()
    slots_list = slots.tolist()
    output = torch.empty_like(x)
    channel_weight = weight.repeat_interleave(group_size, dim=0).float()
    if flip_filter:
        channel_weight = channel_weight.flip(-1)
    channel_bias = None
    if bias is not None:
        channel_bias = (
            bias.float() if bias.shape[0] == x.shape[1] else bias.repeat_interleave(group_size, dim=0).float()
        )
    taps = channel_weight.shape[-1]

    for request_index, (start, end) in enumerate(pairwise(starts_list)):
        slot = slots_list[request_index]
        history = (
            state[slot, :, : taps - 1].clone()
            if slot and bool(has_initial[request_index])
            else torch.zeros((x.shape[1], taps - 1), device=x.device, dtype=torch.float32)
        )
        for token_index in range(start, end):
            current = x[token_index].float()
            value = channel_weight[:, -1] * current + (history * channel_weight[:, :-1]).sum(-1)
            if channel_bias is not None:
                value += channel_bias * current if gated_bias else channel_bias
            output[token_index] = value.to(x.dtype)
            history = torch.cat((history[:, 1:], current[:, None]), dim=-1)
        if slot and end > start:
            state[slot, :, : taps - 1] = history
    return output


def _hcl_reference(
    drive: torch.Tensor,
    gate: torch.Tensor,
    decay: torch.Tensor,
    residues: torch.Tensor,
    diagonal: torch.Tensor,
    state: torch.Tensor,
    starts: torch.Tensor,
    slots: torch.Tensor,
    has_initial: torch.Tensor,
) -> torch.Tensor:
    starts_list = starts.tolist()
    slots_list = slots.tolist()
    output = torch.empty_like(drive)
    state_size = decay.shape[-1]
    for request_index, (start, end) in enumerate(pairwise(starts_list)):
        slot = slots_list[request_index]
        recurrent = (
            state[slot, :, :state_size].clone()
            if slot and bool(has_initial[request_index])
            else torch.zeros_like(decay, dtype=torch.float32)
        )
        for token_index in range(start, end):
            current = drive[token_index].float()
            recurrent = decay.float() * recurrent + current[:, None]
            mixed = (residues.float() * recurrent).sum(-1) + diagonal.float() * current
            output[token_index] = (gate[token_index].float() * mixed).to(output.dtype)
        if slot and end > start:
            state[slot, :, :state_size] = recurrent
    return output


def _reference(
    mixer: Evo2HyenaMixer,
    hidden_states: torch.Tensor,
    projection_state: torch.Tensor,
    operator_state: torch.Tensor,
    starts: torch.Tensor,
    slots: torch.Tensor,
    has_initial: torch.Tensor,
) -> torch.Tensor:
    projected = functional.linear(hidden_states, mixer.dense_projection.weight)
    projected = _fir_reference(
        projected,
        mixer.hyena_proj_conv.short_conv_weight,
        None,
        projection_state,
        starts,
        slots,
        has_initial,
        group_size=1,
    )
    x1, x2, value = projected.view(projected.shape[0], mixer.local_hidden_size, 3).unbind(-1)
    drive = x2 * value
    if mixer.operator_type == "S":
        filtered = _fir_reference(
            drive,
            mixer.mixer.short_conv.short_conv_weight.squeeze(1),
            getattr(mixer.mixer, "conv_bias", None),
            operator_state,
            starts,
            slots,
            has_initial,
            group_size=mixer.operator_group_size,
            gated_bias=True,
        )
        mixed = x1 * filtered
    elif mixer.operator_type == "D":
        filtered = _fir_reference(
            drive,
            mixer.mixer.filter.h * mixer.mixer.filter.decay,
            mixer.mixer.conv_bias,
            operator_state,
            starts,
            slots,
            has_initial,
            group_size=mixer.operator_group_size,
            gated_bias=True,
            flip_filter=True,
        )
        mixed = x1 * filtered
    else:
        mixed = _hcl_reference(
            drive,
            x1,
            torch.exp(-torch.exp(mixer.mixer.filter.p + mixer.mixer.filter.gamma)),
            mixer.mixer.filter.R,
            mixer.mixer.conv_bias,
            operator_state,
            starts,
            slots,
            has_initial,
        )
    return functional.linear(mixed, mixer.dense.weight, mixer.dense.bias)


def _run(
    mixer: Evo2HyenaMixer,
    hidden_states: torch.Tensor,
    metadata,
    vllm_config,
) -> torch.Tensor:
    from vllm.forward_context import set_forward_context

    output = torch.full_like(hidden_states, torch.nan)
    with set_forward_context({mixer.prefix: metadata}, vllm_config, num_tokens=hidden_states.shape[0]):
        mixer(hidden_states, output)
    return output


@CUDA_REQUIRED
@pytest.mark.parametrize("symbol", ["S", "D", "H"])
def test_hyena_mixer_preserves_parameter_and_state_contract(symbol: str) -> None:
    config = _config(symbol)
    with _vllm_module_context() as (vllm_config, cache_config):
        prefix = f"decoder.layers.{symbol}.mixer"
        mixer = Evo2HyenaMixer(
            config,
            operator_type=symbol,
            cache_config=cache_config,
            prefix=prefix,
            params_dtype=DTYPE,
            disable_tp=True,
        )
        names = set(dict(mixer.named_parameters()))
        common = {
            "dense_projection.layer_norm_weight",
            "dense_projection.weight",
            "hyena_proj_conv.short_conv_weight",
            "dense.weight",
            "dense.bias",
        }
        operator_names = {
            "S": {"mixer.short_conv.short_conv_weight"},
            "D": {"mixer.conv_bias", "mixer.filter.h", "mixer.filter.decay"},
            "H": {"mixer.conv_bias", "mixer.filter.p", "mixer.filter.gamma", "mixer.filter.R"},
        }

        assert names == common | operator_names[symbol]
        assert mixer.get_state_shape() == config.local_state_shapes(1)
        assert mixer.get_state_dtype() == (torch.float32, torch.float32)
        assert mixer.mamba_type == "mamba1"
        assert len(mixer.kv_cache) == 2
        assert all(state.numel() == 0 for state in mixer.kv_cache)
        assert vllm_config.compilation_config.static_forward_context[prefix] is mixer


@CUDA_REQUIRED
@pytest.mark.parametrize("symbol", ["S", "D", "H"])
def test_hyena_mixer_packed_prefill_and_decode_match_independent_reference(symbol: str) -> None:
    config = _config(symbol)
    with _vllm_module_context() as (vllm_config, cache_config):
        mixer = (
            Evo2HyenaMixer(
                config,
                operator_type=symbol,
                cache_config=cache_config,
                prefix=f"decoder.layers.{symbol}.mixer",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(mixer)
        projection_shape, operator_shape = mixer.get_state_shape()
        projection_state = torch.zeros((4, *projection_shape), device=DEVICE, dtype=torch.float32)
        operator_state = torch.zeros((4, *operator_shape), device=DEVICE, dtype=torch.float32)
        mixer.kv_cache = (projection_state, operator_state)
        expected_projection_state = projection_state.clone()
        expected_operator_state = operator_state.clone()

        starts = torch.tensor([0, 2, 7, 10], device=DEVICE, dtype=torch.int32)
        slots = torch.tensor([1, 2, 3], device=DEVICE, dtype=torch.int32)
        no_initial = torch.zeros(3, device=DEVICE, dtype=torch.bool)
        generator = torch.Generator(device=DEVICE).manual_seed(23)
        prefill = torch.randn((10, config.hidden_size), device=DEVICE, dtype=DTYPE, generator=generator)
        expected_prefill = _reference(
            mixer,
            prefill,
            expected_projection_state,
            expected_operator_state,
            starts,
            slots,
            no_initial,
        )
        actual_prefill = _run(
            mixer,
            prefill,
            _metadata(
                query_start_loc_p=starts,
                state_indices_tensor_p=slots,
                has_initial_states_p=no_initial,
            ),
            vllm_config,
        )

        torch.testing.assert_close(actual_prefill, expected_prefill, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(projection_state, expected_projection_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(operator_state, expected_operator_state, rtol=2e-5, atol=2e-5)

        decode_starts = torch.arange(4, device=DEVICE, dtype=torch.int32)
        has_initial = torch.ones(3, device=DEVICE, dtype=torch.bool)
        decode = torch.randn((3, config.hidden_size), device=DEVICE, dtype=DTYPE, generator=generator)
        expected_decode = _reference(
            mixer,
            decode,
            expected_projection_state,
            expected_operator_state,
            decode_starts,
            slots,
            has_initial,
        )
        actual_decode = _run(
            mixer,
            decode,
            _metadata(state_indices_tensor_d=slots),
            vllm_config,
        )

        torch.testing.assert_close(actual_decode, expected_decode, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(projection_state, expected_projection_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(operator_state, expected_operator_state, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
@pytest.mark.parametrize("symbol", ["S", "D", "H"])
def test_hyena_mixer_preserves_decode_first_mixed_batch_order(symbol: str) -> None:
    config = _config(symbol)
    with _vllm_module_context() as (vllm_config, cache_config):
        mixer = (
            Evo2HyenaMixer(
                config,
                operator_type=symbol,
                cache_config=cache_config,
                prefix=f"decoder.layers.{symbol}.mixed.mixer",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(mixer, seed=31)
        projection_shape, operator_shape = mixer.get_state_shape()
        projection_state = torch.zeros((5, *projection_shape), device=DEVICE, dtype=torch.float32)
        operator_state = torch.zeros((5, *operator_shape), device=DEVICE, dtype=torch.float32)
        mixer.kv_cache = (projection_state, operator_state)
        expected_projection_state = projection_state.clone()
        expected_operator_state = operator_state.clone()

        generator = torch.Generator(device=DEVICE).manual_seed(37)
        hidden_states = torch.randn((7, config.hidden_size), device=DEVICE, dtype=DTYPE, generator=generator)
        decode_slots = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)
        decode_starts = torch.arange(3, device=DEVICE, dtype=torch.int32)
        decode_initial = torch.ones(2, device=DEVICE, dtype=torch.bool)
        prefill_slots = torch.tensor([3, 4], device=DEVICE, dtype=torch.int32)
        prefill_starts = torch.tensor([0, 3, 5], device=DEVICE, dtype=torch.int32)
        prefill_initial = torch.zeros(2, device=DEVICE, dtype=torch.bool)
        expected = torch.empty_like(hidden_states)
        expected[:2] = _reference(
            mixer,
            hidden_states[:2],
            expected_projection_state,
            expected_operator_state,
            decode_starts,
            decode_slots,
            decode_initial,
        )
        expected[2:] = _reference(
            mixer,
            hidden_states[2:],
            expected_projection_state,
            expected_operator_state,
            prefill_starts,
            prefill_slots,
            prefill_initial,
        )

        actual = _run(
            mixer,
            hidden_states,
            _metadata(
                query_start_loc_p=prefill_starts,
                state_indices_tensor_p=prefill_slots,
                has_initial_states_p=prefill_initial,
                state_indices_tensor_d=decode_slots[:, None],
            ),
            vllm_config,
        )

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(projection_state, expected_projection_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(operator_state, expected_operator_state, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
def test_hyena_mixer_profile_run_does_not_require_or_mutate_cache() -> None:
    from vllm.forward_context import set_forward_context

    config = _config("H")
    with _vllm_module_context() as (vllm_config, cache_config):
        mixer = (
            Evo2HyenaMixer(
                config,
                operator_type="H",
                cache_config=cache_config,
                prefix="decoder.layers.profile.mixer",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(mixer, seed=41)
        hidden_states = torch.randn((13, config.hidden_size), device=DEVICE, dtype=DTYPE)
        expected_values = functional.linear(hidden_states, mixer.dense_projection.weight).view(
            hidden_states.shape[0], config.hidden_size, 3
        )[..., 2]
        expected = functional.linear(expected_values, mixer.dense.weight, mixer.dense.bias)
        output = torch.full_like(hidden_states, torch.nan)

        with set_forward_context(None, vllm_config, num_tokens=hidden_states.shape[0]):
            mixer(hidden_states, output)

        torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-5)
        assert all(state.numel() == 0 for state in mixer.kv_cache)


@CUDA_REQUIRED
def test_hyena_mixer_custom_op_supports_fullgraph_compile() -> None:
    from vllm.forward_context import set_forward_context

    config = _config("H")
    with _vllm_module_context() as (vllm_config, cache_config):
        mixer = (
            Evo2HyenaMixer(
                config,
                operator_type="H",
                cache_config=cache_config,
                prefix="decoder.layers.compile.mixer",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(mixer, seed=43)
        projection_shape, operator_shape = mixer.get_state_shape()
        projection_state = torch.zeros((5, *projection_shape), device=DEVICE, dtype=torch.float32)
        operator_state = torch.zeros((5, *operator_shape), device=DEVICE, dtype=torch.float32)
        mixer.kv_cache = (projection_state, operator_state)
        expected_projection_state = projection_state.clone()
        expected_operator_state = operator_state.clone()
        hidden_states = torch.randn((4, config.hidden_size), device=DEVICE, dtype=DTYPE)
        starts = torch.arange(5, device=DEVICE, dtype=torch.int32)
        slots = torch.arange(1, 5, device=DEVICE, dtype=torch.int32)
        has_initial = torch.ones(4, device=DEVICE, dtype=torch.bool)
        expected = _reference(
            mixer,
            hidden_states,
            expected_projection_state,
            expected_operator_state,
            starts,
            slots,
            has_initial,
        )
        metadata = _metadata(state_indices_tensor_d=slots)

        def run(hidden: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
            mixer(hidden, output)
            return output

        compiled = torch.compile(run, fullgraph=True)
        output = torch.empty_like(hidden_states)
        with set_forward_context({mixer.prefix: metadata}, vllm_config, num_tokens=hidden_states.shape[0]):
            actual = compiled(hidden_states, output)

        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(projection_state, expected_projection_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(operator_state, expected_operator_state, rtol=2e-5, atol=2e-5)


@CUDA_REQUIRED
def test_hyena_mixer_full_decode_cuda_graph_replays_static_packed_buffers() -> None:
    from vllm.forward_context import set_forward_context

    config = _config("H")
    with _vllm_module_context() as (vllm_config, cache_config):
        mixer = (
            Evo2HyenaMixer(
                config,
                operator_type="H",
                cache_config=cache_config,
                prefix="decoder.layers.graph.mixer",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(mixer, seed=47)
        projection_shape, operator_shape = mixer.get_state_shape()
        projection_state = torch.zeros((3, *projection_shape), device=DEVICE, dtype=torch.float32)
        operator_state = torch.zeros((3, *operator_shape), device=DEVICE, dtype=torch.float32)
        projection_state[0].fill_(5)
        operator_state[0].fill_(7)
        mixer.kv_cache = (projection_state, operator_state)
        hidden_states = torch.randn((4, config.hidden_size), device=DEVICE, dtype=DTYPE)
        slots = torch.tensor([1, 2, 0, 0], device=DEVICE, dtype=torch.int32)
        starts = torch.arange(5, device=DEVICE, dtype=torch.int32)
        has_initial = torch.ones(4, device=DEVICE, dtype=torch.bool)
        metadata = _metadata(state_indices_tensor_d=slots)

        with set_forward_context({mixer.prefix: metadata}, vllm_config, num_tokens=hidden_states.shape[0]):
            warm_output = torch.empty_like(hidden_states)
            mixer(hidden_states, warm_output)
            torch.cuda.synchronize()
        projection_state.zero_()
        operator_state.zero_()
        projection_state[0].fill_(5)
        operator_state[0].fill_(7)
        expected_projection_state = projection_state.clone()
        expected_operator_state = operator_state.clone()
        _reference(
            mixer,
            hidden_states,
            expected_projection_state,
            expected_operator_state,
            starts,
            slots,
            has_initial,
        )
        expected_second = _reference(
            mixer,
            hidden_states,
            expected_projection_state,
            expected_operator_state,
            starts,
            slots,
            has_initial,
        )

        graph_output = torch.empty_like(hidden_states)
        graph = torch.cuda.CUDAGraph()
        with set_forward_context({mixer.prefix: metadata}, vllm_config, num_tokens=hidden_states.shape[0]):
            with torch.cuda.graph(graph):
                mixer(hidden_states, graph_output)
            output_pointer = graph_output.data_ptr()
            graph.replay()
            graph.replay()
        torch.cuda.synchronize()

        assert graph_output.data_ptr() == output_pointer
        torch.testing.assert_close(graph_output, expected_second, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(projection_state, expected_projection_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(operator_state, expected_operator_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(projection_state[0], torch.full_like(projection_state[0], 5))
        torch.testing.assert_close(operator_state[0], torch.full_like(operator_state[0], 7))


@CUDA_REQUIRED
def test_hyena_mixer_batch_96_uses_one_packed_fir_and_hcl_launch() -> None:
    config = _config("H")
    with _vllm_module_context() as (vllm_config, cache_config):
        mixer = (
            Evo2HyenaMixer(
                config,
                operator_type="H",
                cache_config=cache_config,
                prefix="decoder.layers.profile96.mixer",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(mixer, seed=53)
        projection_shape, operator_shape = mixer.get_state_shape()
        mixer.kv_cache = (
            torch.zeros((97, *projection_shape), device=DEVICE, dtype=torch.float32),
            torch.zeros((97, *operator_shape), device=DEVICE, dtype=torch.float32),
        )
        lengths = [4 + index % 9 for index in range(96)]
        starts_list = [0]
        for length in lengths:
            starts_list.append(starts_list[-1] + length)
        starts = torch.tensor(starts_list, device=DEVICE, dtype=torch.int32)
        slots = torch.arange(1, 97, device=DEVICE, dtype=torch.int32)
        has_initial = torch.zeros(96, device=DEVICE, dtype=torch.bool)
        hidden_states = torch.randn((starts_list[-1], config.hidden_size), device=DEVICE, dtype=DTYPE)
        metadata = _metadata(
            query_start_loc_p=starts,
            state_indices_tensor_p=slots,
            has_initial_states_p=has_initial,
        )

        _run(mixer, hidden_states, metadata, vllm_config)
        torch.cuda.synchronize()
        mixer.kv_cache[0].zero_()
        mixer.kv_cache[1].zero_()
        with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as profiler:
            _run(mixer, hidden_states, metadata, vllm_config)
            torch.cuda.synchronize()

        events = profiler.key_averages()
        fir_launches = sum(event.count for event in events if "packed_causal_fir_kernel" in event.key)
        hcl_launches = sum(event.count for event in events if "packed_modal_iir_kernel" in event.key)
        assert fir_launches == 1
        assert hcl_launches == 1


@CUDA_REQUIRED
def test_hyena_decoder_preserves_mbridge_names_and_residual_order() -> None:
    from vllm.forward_context import set_forward_context

    config = _config("H")
    with _vllm_module_context() as (vllm_config, cache_config):
        layer = (
            Evo2HyenaDecoderLayer(
                config,
                operator_type="H",
                cache_config=cache_config,
                prefix="decoder.layers.0",
                params_dtype=DTYPE,
                disable_tp=True,
            )
            .to(DEVICE)
            .eval()
        )
        _randomize(layer, seed=59)
        names = set(dict(layer.named_parameters()))
        assert names == {
            "mixer.dense_projection.layer_norm_weight",
            "mixer.dense_projection.weight",
            "mixer.hyena_proj_conv.short_conv_weight",
            "mixer.dense.weight",
            "mixer.dense.bias",
            "mixer.mixer.conv_bias",
            "mixer.mixer.filter.p",
            "mixer.mixer.filter.gamma",
            "mixer.mixer.filter.R",
            "mlp.linear_fc1.layer_norm_weight",
            "mlp.linear_fc1.weight",
            "mlp.linear_fc2.weight",
        }
        projection_shape, operator_shape = layer.mixer.get_state_shape()
        projection_state = torch.zeros((3, *projection_shape), device=DEVICE, dtype=torch.float32)
        operator_state = torch.zeros((3, *operator_shape), device=DEVICE, dtype=torch.float32)
        layer.mixer.kv_cache = (projection_state, operator_state)
        expected_projection_state = projection_state.clone()
        expected_operator_state = operator_state.clone()
        hidden_states = torch.randn((5, config.hidden_size), device=DEVICE, dtype=DTYPE)
        starts = torch.tensor([0, 2, 5], device=DEVICE, dtype=torch.int32)
        slots = torch.tensor([1, 2], device=DEVICE, dtype=torch.int32)
        has_initial = torch.zeros(2, device=DEVICE, dtype=torch.bool)
        first_norm = _rms_norm(
            hidden_states,
            layer.mixer.dense_projection.layer_norm_weight,
            config.rms_norm_eps,
        )
        expected_mixer = _reference(
            layer.mixer,
            first_norm,
            expected_projection_state,
            expected_operator_state,
            starts,
            slots,
            has_initial,
        )
        expected_residual = hidden_states + expected_mixer
        second_norm = _rms_norm(
            expected_residual,
            layer.mlp.linear_fc1.layer_norm_weight,
            config.rms_norm_eps,
        )
        expected_hidden = _mlp_reference(layer, second_norm)
        metadata = _metadata(
            query_start_loc_p=starts,
            state_indices_tensor_p=slots,
            has_initial_states_p=has_initial,
        )

        with set_forward_context({layer.mixer.prefix: metadata}, vllm_config, num_tokens=hidden_states.shape[0]):
            actual_hidden, actual_residual = layer(
                torch.arange(hidden_states.shape[0], device=DEVICE),
                hidden_states,
                None,
            )

        torch.testing.assert_close(actual_residual, expected_residual, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(actual_hidden, expected_hidden, rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(projection_state, expected_projection_state, rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(operator_state, expected_operator_state, rtol=2e-5, atol=2e-5)
