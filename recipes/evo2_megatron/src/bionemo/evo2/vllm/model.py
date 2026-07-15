# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2

"""Out-of-tree vLLM model implementation for Evo2 Vortex checkpoints."""

from collections.abc import Iterable
from itertools import islice

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import MambaCopySpec, MambaStateCopyFunc
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory, make_layers, maybe_prefix
from vllm.sequence import IntermediateTensors

from bionemo.evo2.vllm.config import Evo2Config
from bionemo.evo2.vllm.hyena import Evo2HyenaDecoderLayer
from bionemo.evo2.vllm.layers import Evo2AttentionDecoderLayer
from bionemo.evo2.vllm.weights import IncrementalEvo2WeightLoader


def _copy_whole_evo2_state_block(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Describe one contiguous channel-major Evo2 state block copy."""
    if num_accepted_tokens != 1:
        raise ValueError("Evo2 vLLM inference does not support speculative state copies")
    if not 0 <= cur_block_idx < len(block_ids):
        raise IndexError("Evo2 state copy block index is out of range")
    source = state[block_ids[cur_block_idx]]
    return MambaCopySpec(start_addr=source.data_ptr(), num_elements=source.numel())


def evo2_context_length_contract(vllm_config: VllmConfig) -> dict[str, object]:
    """Describe the resolved position and length-independent recurrent-state contract."""
    config = vllm_config.model_config.hf_config
    if not isinstance(config, Evo2Config):
        raise TypeError("Evo2 context length contract requires Evo2Config")
    resolved_max_model_len = int(vllm_config.model_config.max_model_len)
    if resolved_max_model_len <= 0:
        raise ValueError("resolved max_model_len must be positive")
    state_shapes = config.local_state_shapes(vllm_config.parallel_config.tensor_parallel_size)
    return {
        "checkpoint_declared_max_position_embeddings": int(config.max_position_embeddings),
        "resolved_max_model_len": resolved_max_model_len,
        "rotary_max_position_embeddings": resolved_max_model_len,
        "attention_kv_position_limit": resolved_max_model_len,
        "position_source": "vllm_config.model_config.max_model_len",
        "position_clipping": False,
        "hyena_state_shapes": [list(shape) for shape in state_shapes],
        "hyena_state_dtypes": ["float32", "float32"],
        "hyena_state_length_dependent": False,
        "hyena_decode_scratch_request_capacity": int(vllm_config.scheduler_config.max_num_seqs),
        "hyena_prefill_allocation_basis": "num_actual_tokens",
    }


class Evo2Embedding(nn.Module):
    """Vocab-parallel embedding with the native MBridge module path."""

    def __init__(
        self,
        config: Evo2Config,
        *,
        quant_config=None,
        prefix: str = "",
        params_dtype: torch.dtype | None = None,
    ) -> None:
        """Construct the checkpoint-compatible word embedding."""
        super().__init__()
        self.word_embeddings = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=f"{prefix}.word_embeddings",
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed one packed vLLM token tensor."""
        return self.word_embeddings(input_ids)


class Evo2Decoder(nn.Module):
    """Pipeline-aware hybrid Evo2 decoder with native checkpoint names."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str) -> None:
        """Construct the operator-pattern layers and final RMS normalization."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2ForCausalLM requires Evo2Config")
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        params_dtype = vllm_config.model_config.dtype
        self.context_length_contract = evo2_context_length_contract(vllm_config)
        max_position_embeddings = int(self.context_length_contract["rotary_max_position_embeddings"])
        self.config = config

        def make_layer(prefix: str) -> nn.Module:
            layer_index = int(prefix.rsplit(".", maxsplit=1)[-1])
            operator_type = config.hybrid_override_pattern[layer_index]
            if operator_type == "*":
                return Evo2AttentionDecoderLayer(
                    config,
                    cache_config=cache_config,
                    quant_config=quant_config,
                    prefix=prefix,
                    max_position_embeddings=max_position_embeddings,
                    params_dtype=params_dtype,
                )
            return Evo2HyenaDecoderLayer(
                config,
                operator_type=operator_type,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
                params_dtype=params_dtype,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            make_layer,
            prefix=f"{prefix}.layers",
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | IntermediateTensors:
        """Apply local pipeline layers and the final norm on the last rank."""
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        if not get_pp_group().is_last_rank:
            if residual is None:
                raise RuntimeError("Evo2 pipeline stage produced no residual")
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        if residual is None:
            return self.final_norm(hidden_states)
        normalized, _ = self.final_norm(hidden_states, residual)
        return normalized


@support_torch_compile
class Evo2Model(nn.Module):
    """Evo2 embedding and hybrid decoder compiled as one vLLM graph."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Construct the checkpoint-compatible Evo2 backbone."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2Model requires Evo2Config")
        self.config = config
        self.hybrid_override_pattern = config.hybrid_override_pattern
        self.embedding = Evo2Embedding(
            config,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "embedding"),
            params_dtype=vllm_config.model_config.dtype,
        )
        self.decoder = Evo2Decoder(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "decoder"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed packed token ids through the vocab-parallel table."""
        return self.embedding(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """Run one packed vLLM scheduler batch through the local pipeline stage."""
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("input_ids are required when inputs_embeds are absent")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            if intermediate_tensors is None:
                raise ValueError("non-first Evo2 pipeline ranks require intermediate tensors")
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        return self.decoder(hidden_states, positions, residual)


class Evo2ForCausalLM(nn.Module, HasInnerState, IsHybrid):
    """vLLM causal language model wrapper for Evo2 Vortex checkpoints."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Construct Evo2, tie its LM head, and expose vLLM state interfaces."""
        super().__init__()
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2ForCausalLM requires Evo2Config")
        self.config = config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = Evo2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )
        if not config.tie_word_embeddings:
            raise ValueError("Evo2 vLLM checkpoints require tied input and output embeddings")
        self.lm_head = self.model.embedding.word_embeddings
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self._weight_loader = IncrementalEvo2WeightLoader(self)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input ids through the tied Evo2 table."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        """Return hidden states for vLLM's logits and sampling pipeline."""
        del kwargs
        self._weight_loader.assert_ready_for_inference()
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return length-independent channel-major cache shapes for one TP rank."""
        config = vllm_config.model_config.hf_config
        if not isinstance(config, Evo2Config):
            raise TypeError("Evo2 state shapes require Evo2Config")
        return config.local_state_shapes(vllm_config.parallel_config.tensor_parallel_size)

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        """Keep both Evo2 recurrent states in fp32."""
        return (torch.float32, torch.float32)

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        """Return whole-block copy functions independent of vLLM's conv-layout setting."""
        return (_copy_whole_evo2_state_block, _copy_whole_evo2_state_block)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """Delegate graph input copies to vLLM's injected Mamba cache manager."""
        return self.mamba_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """Delegate state capture buffers to vLLM's injected Mamba cache manager."""
        return self.mamba_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        """Compute vocabulary-trimmed logits with the tied embedding table."""
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream native MBridge or Vortex names into the vLLM parameter tree."""
        return self._weight_loader.load(weights)
