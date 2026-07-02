# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

from types import SimpleNamespace

import torch

import bionemo.evo2_phage_gen.nemo_rl_evo2_generation as evo2_generation
from bionemo.evo2_phage_gen.nemo_rl_evo2_generation import (
    Evo2GenerationResult,
    Evo2MegatronGenerationAdapter,
    _PromptTokenProxy,
    should_use_evo2_native_batched_generation,
)
from nemo_rl.models.generation.megatron.megatron_generation import (
    _adapter_requires_all_workers,
    _load_generation_adapter,
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
            "generation_adapter": (
                "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:"
                "Evo2MegatronGenerationAdapter"
            ),
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
                "generation_adapter": (
                    "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:"
                    "Evo2MegatronGenerationAdapter"
                ),
            }
        }
    }
    evo2_model = SimpleNamespace(
        decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None)
    )
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
    assert not should_use_evo2_native_batched_generation(
        cfg, non_evo2_model, batch_size=8
    )


def test_evo2_adapter_rng_seed_advances_and_records_trace():
    adapter = Evo2MegatronGenerationAdapter({"seed": 17, "seed_stride": 101})
    worker = SimpleNamespace(rank=0, cfg={"generation": {"mcore_generation_config": {}}})

    assert adapter._next_seed(worker) == 17
    assert adapter._next_seed(worker) == 118
    assert worker._evo2_generation_rng_trace == [
        {"rank": 0, "call_index": 0, "seed": 17, "base_seed": 17, "seed_stride": 101},
        {"rank": 0, "call_index": 1, "seed": 118, "base_seed": 17, "seed_stride": 101},
    ]


def test_evo2_adapter_only_return_rank_emits_batched_data(monkeypatch):
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
        )
    ]

    worker = SimpleNamespace(
        rank=1,
        cfg={
            "generation": {
                "mcore_generation_config": {
                    "prompt_batch_size": 8,
                    "generation_adapter": (
                        "bionemo.evo2_phage_gen.nemo_rl_evo2_generation:"
                        "Evo2MegatronGenerationAdapter"
                    ),
                }
            }
        },
        model=SimpleNamespace(
            decoder=SimpleNamespace(hyena_state_shapes_per_request=lambda: None)
        ),
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

    assert adapter.generate_worker(worker, data=data, greedy=False) is None

    worker.rank = 0
    assert adapter.generate_worker(worker, data=data, greedy=False) is parsed
