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

from types import SimpleNamespace

import pytest
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.megatron.megatron_generation import (
    MegatronGeneration,
    _adapter_requires_all_workers,
    _load_generation_adapter,
)

import bionemo.evo2_phage_gen.nemo_rl_evo2_generation as evo2_generation
from bionemo.evo2_phage_gen.nemo_rl_evo2_generation import (
    Evo2GenerationResult,
    Evo2MegatronGenerationAdapter,
    _PromptTokenProxy,
    should_use_evo2_native_batched_generation,
)


class _Tokenizer:
    vocab_size = 8

    def tokenize(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def test_prompt_token_proxy_preserves_nemo_rl_prompt_ids_and_delegates_other_text():
    tokenizer = _Tokenizer()
    proxy = _PromptTokenProxy(tokenizer, [[11, 12], [21, 22, 23]])

    assert proxy.tokenize(proxy.prompts[0]) == [11, 12]
    assert proxy.tokenize(proxy.prompts[1]) == [21, 22, 23]
    assert proxy.tokenize("AC") == [65, 67]
    assert proxy.detokenize([65, 67]) == "AC"
    assert proxy.vocab_size == 8


def test_nemo_rl_generation_adapter_loader_imports_configured_adapter():
    config = {
        "mcore_generation_config": {
            "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"),
            "generation_adapter_config": {"seed": 13},
        }
    }

    adapter = _load_generation_adapter(config)

    assert isinstance(adapter, Evo2MegatronGenerationAdapter)
    assert adapter.config["seed"] == 13
    assert _adapter_requires_all_workers(adapter, config)


def test_should_use_evo2_native_batched_generation_requires_evo2_batch_and_model():
    cfg = {
        "generation": {
            "mcore_generation_config": {
                "prompt_batch_size": 8,
                "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"),
            }
        }
    }
    evo2_model = SimpleNamespace(decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None))
    wrapped_evo2_model = SimpleNamespace(module=SimpleNamespace(module=evo2_model))
    non_evo2_model = SimpleNamespace(decoder=SimpleNamespace())

    assert not should_use_evo2_native_batched_generation(
        {"generation": {"mcore_generation_config": {"prompt_batch_size": 8}}},
        evo2_model,
        batch_size=8,
    )
    assert should_use_evo2_native_batched_generation(cfg, evo2_model, batch_size=8)
    assert should_use_evo2_native_batched_generation(cfg, wrapped_evo2_model, batch_size=8)
    assert should_use_evo2_native_batched_generation(cfg, evo2_model, batch_size=1)
    assert not should_use_evo2_native_batched_generation(
        {"generation": {"mcore_generation_config": {"prompt_batch_size": 1}}},
        evo2_model,
        batch_size=8,
    )
    assert not should_use_evo2_native_batched_generation(cfg, non_evo2_model, batch_size=8)


def test_evo2_adapter_rng_seed_advances_and_records_trace():
    adapter = Evo2MegatronGenerationAdapter({"seed": 17, "seed_stride": 101})
    worker = SimpleNamespace(rank=0, cfg={"generation": {"mcore_generation_config": {}}})

    assert adapter._next_seed(worker) == 17
    assert adapter._next_seed(worker) == 118
    assert worker._evo2_generation_rng_trace == [
        {
            "rank": 0,
            "data_parallel_rank": 0,
            "data_parallel_size": 1,
            "call_index": 0,
            "seed_index": 0,
            "seed": 17,
            "base_seed": 17,
            "seed_stride": 101,
        },
        {
            "rank": 0,
            "data_parallel_rank": 0,
            "data_parallel_size": 1,
            "call_index": 1,
            "seed_index": 1,
            "seed": 118,
            "base_seed": 17,
            "seed_stride": 101,
        },
    ]


def test_evo2_adapter_assigns_distinct_seed_streams_to_data_parallel_replicas():
    adapter = Evo2MegatronGenerationAdapter({"seed": 17, "seed_stride": 101})
    worker0 = SimpleNamespace(
        rank=0,
        data_parallel_rank=0,
        dp_size=2,
        cfg={"generation": {"mcore_generation_config": {}}},
    )
    worker1 = SimpleNamespace(
        rank=1,
        data_parallel_rank=1,
        dp_size=2,
        cfg={"generation": {"mcore_generation_config": {}}},
    )

    assert adapter._next_seed(worker0) == 17
    assert adapter._next_seed(worker1) == 118
    assert adapter._next_seed(worker0) == 219
    assert adapter._next_seed(worker1) == 320
    assert [entry["seed_index"] for entry in worker0._evo2_generation_rng_trace] == [0, 2]
    assert [entry["seed_index"] for entry in worker1._evo2_generation_rng_trace] == [1, 3]


def test_evo2_native_generation_reseeds_cached_sampling_rng_for_each_adapter_call():
    cached_rng = object()
    native_dynamic = SimpleNamespace(evo2_seed=17, sampling_rng=cached_rng)

    evo2_generation._reseed_evo2_native_dynamic(native_dynamic, 118)

    assert native_dynamic.evo2_seed == 118
    assert native_dynamic.sampling_rng is None


@pytest.mark.parametrize(
    ("configured_size", "tensor_parallel_size", "expected_size"),
    [(48, 1, 48), (48, 2, 48), (48, 5, 50), (96, 7, 98), (96, 8, 96)],
)
def test_evo2_native_decode_capacity_is_rounded_up_to_tensor_parallel_multiple(
    configured_size, tensor_parallel_size, expected_size
):
    worker = SimpleNamespace(
        cfg={
            "megatron_cfg": {"tensor_model_parallel_size": tensor_parallel_size},
            "generation": {"mcore_generation_config": {"prompt_batch_size": configured_size}},
        }
    )

    assert evo2_generation._evo2_native_batched_decode_size(worker) == expected_size


@pytest.mark.parametrize(
    ("request_count", "tensor_parallel_size", "expected_size"),
    [(48, 1, 48), (48, 5, 50), (96, 7, 98), (96, 8, 96)],
)
def test_nemo_worker_request_capacity_is_rounded_up_to_tensor_parallel_multiple(
    request_count, tensor_parallel_size, expected_size
):
    from nemo_rl.models.generation.megatron.megatron_worker import (
        _round_up_request_capacity,
    )

    assert _round_up_request_capacity(request_count, tensor_parallel_size) == expected_size


def test_evo2_adapter_emits_replicated_batched_data_from_every_model_parallel_rank(monkeypatch):
    adapter = Evo2MegatronGenerationAdapter({"seed": 17})
    data = SimpleNamespace(size=2)
    prompt_tokens = torch.tensor([[11, 12], [21, 22]])
    prompt_lengths = torch.tensor([2, 2])
    sampling_params = [SimpleNamespace(num_tokens_to_generate=2)]
    parsed = object()
    generated = [
        Evo2GenerationResult(
            prompt_tokens=prompt_tokens[0],
            generated_tokens=[65, 67],
            generated_log_probs=[-0.1, -0.2],
        ),
        Evo2GenerationResult(
            prompt_tokens=prompt_tokens[1],
            generated_tokens=[67, 65],
            generated_log_probs=[-0.3, -0.4],
        ),
    ]

    worker = SimpleNamespace(
        rank=1,
        cfg={
            "generation": {
                "mcore_generation_config": {
                    "prompt_batch_size": 8,
                    "generation_adapter": (
                        "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter"
                    ),
                }
            }
        },
        model=SimpleNamespace(decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None)),
        _prepare_data_for_generation=lambda _data, _greedy: (
            prompt_tokens,
            prompt_lengths,
            sampling_params,
        ),
        _parse_result_to_batched_data_dict=lambda _data, _result: parsed,
    )

    monkeypatch.setattr(
        evo2_generation,
        "generate_evo2_native_batched",
        lambda *args, **kwargs: generated,
    )

    assert adapter.generate_worker(worker, data=data, greedy=False) is parsed

    monkeypatch.setattr(
        evo2_generation,
        "generate_evo2_native_batched",
        lambda *args, **kwargs: generated[:1],
    )
    with pytest.raises(RuntimeError, match="returned 1 results for 2 prompts"):
        adapter.generate_worker(worker, data=data, greedy=False)


def test_megatron_generation_shards_adapter_input_across_dp_and_gathers_in_order():
    class _ShardingAnnotations:
        def get_axis_size(self, axis):
            assert axis == "data_parallel"
            return 2

    class _WorkerGroup:
        def __init__(self):
            self.sharding_annotations = _ShardingAnnotations()
            self.call = None
            self.future_bundle = object()
            self.outputs = [
                BatchedDataDict(
                    {
                        "output_ids": torch.tensor([[11, 12, 13], [21, 22, 23]]),
                        "generation_lengths": torch.tensor([1, 1]),
                        "unpadded_sequence_lengths": torch.tensor([3, 3]),
                        "logprobs": torch.tensor([[0.0, 0.0, -0.1], [0.0, 0.0, -0.2]]),
                        "truncated": torch.tensor([False, False]),
                    }
                ),
                BatchedDataDict(
                    {
                        "output_ids": torch.tensor([[31, 32, 33, 34], [41, 42, 43, 44]]),
                        "generation_lengths": torch.tensor([2, 2]),
                        "unpadded_sequence_lengths": torch.tensor([4, 4]),
                        "logprobs": torch.tensor([[0.0, 0.0, -0.3, -0.4], [0.0, 0.0, -0.5, -0.6]]),
                        "truncated": torch.tensor([False, False]),
                    }
                ),
            ]

        def run_all_workers_sharded_data(self, method_name, **kwargs):
            self.call = (method_name, kwargs)
            return self.future_bundle

        def get_all_worker_results(self, future_bundle):
            assert future_bundle is self.future_bundle
            return self.outputs

    worker_group = _WorkerGroup()
    generation = object.__new__(MegatronGeneration)
    generation.cfg = {
        "_pad_token_id": 99,
        "mcore_generation_config": {
            "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter")
        },
    }
    generation._owns_policy = False
    generation._policy = SimpleNamespace(worker_group=worker_group)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]]),
            "input_lengths": torch.tensor([2, 2, 2, 2]),
        }
    )

    result = generation.generate(data)

    assert worker_group.call is not None
    method_name, call = worker_group.call
    assert method_name == "generate_with_adapter"
    assert call["in_sharded_axes"] == ["data_parallel"]
    assert call["replicate_on_axes"] == [
        "context_parallel",
        "tensor_parallel",
        "pipeline_parallel",
    ]
    assert call["output_is_replicated"] == call["replicate_on_axes"]
    assert call["common_kwargs"] == {"greedy": False}
    assert call["data"][0]["input_ids"].tolist() == [[1, 2], [3, 4]]
    assert call["data"][1]["input_ids"].tolist() == [[5, 6], [7, 8]]
    assert result["output_ids"].tolist() == [
        [11, 12, 13, 99],
        [21, 22, 23, 99],
        [31, 32, 33, 34],
        [41, 42, 43, 44],
    ]
    assert result["generation_lengths"].tolist() == [1, 1, 2, 2]

    too_small = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )
    with pytest.raises(ValueError, match="batch size 1.*data-parallel size 2"):
        generation.generate(too_small)

    worker_group.outputs = worker_group.outputs[:1]
    with pytest.raises(RuntimeError, match="expected 2 data-parallel results, received 1"):
        generation.generate(data)


def test_megatron_generation_balances_uneven_dp_shards_without_empty_replicas():
    class _ShardingAnnotations:
        @staticmethod
        def get_axis_size(axis):
            assert axis == "data_parallel"
            return 4

    class _WorkerGroup:
        def __init__(self):
            self.sharding_annotations = _ShardingAnnotations()
            self.shards = None

        def run_all_workers_sharded_data(self, method_name, **kwargs):
            assert method_name == "generate_with_adapter"
            self.shards = kwargs["data"]
            return object()

        def get_all_worker_results(self, _future_bundle):
            outputs = []
            for shard in self.shards:
                input_ids = shard["input_ids"]
                shard_size = input_ids.size(0)
                outputs.append(
                    BatchedDataDict(
                        {
                            "output_ids": torch.cat(
                                [input_ids, torch.full((shard_size, 1), 9, dtype=torch.long)], dim=1
                            ),
                            "generation_lengths": torch.ones(shard_size, dtype=torch.long),
                            "unpadded_sequence_lengths": torch.full((shard_size,), 3, dtype=torch.long),
                            "logprobs": torch.zeros((shard_size, 3)),
                            "truncated": torch.zeros(shard_size, dtype=torch.bool),
                        }
                    )
                )
            return outputs

    worker_group = _WorkerGroup()
    generation = object.__new__(MegatronGeneration)
    generation.cfg = {
        "_pad_token_id": 99,
        "mcore_generation_config": {
            "generation_adapter": ("bionemo.evo2_phage_gen.nemo_rl_evo2_generation:Evo2MegatronGenerationAdapter")
        },
    }
    generation._owns_policy = False
    generation._policy = SimpleNamespace(worker_group=worker_group)
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[0, 10], [1, 11], [2, 12], [3, 13], [4, 14]]),
            "input_lengths": torch.full((5,), 2, dtype=torch.long),
        }
    )

    result = generation.generate(data)

    assert [shard.size for shard in worker_group.shards] == [2, 1, 1, 1]
    assert [shard["input_ids"][:, 0].tolist() for shard in worker_group.shards] == [[0, 1], [2], [3], [4]]
    assert result["output_ids"][:, 0].tolist() == [0, 1, 2, 3, 4]
