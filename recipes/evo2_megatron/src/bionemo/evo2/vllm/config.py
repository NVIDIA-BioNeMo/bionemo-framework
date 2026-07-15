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

"""Serializable Transformers configuration for Evo2 vLLM models."""

from transformers import PreTrainedConfig


EVO2_7B_PATTERN = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*"
_LAYER_SYMBOLS = frozenset("SDH*")


class Evo2Config(PreTrainedConfig):
    """Configuration shared by exported Evo2 checkpoints and the vLLM backend."""

    model_type = "evo2"

    def __init__(
        self,
        *,
        vocab_size: int = 512,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int | None = None,
        max_position_embeddings: int = 8192,
        hybrid_override_pattern: str = EVO2_7B_PATTERN,
        short_conv_length: int = 3,
        hcs_filter_length: int = 7,
        hcm_filter_length: int = 128,
        hcl_state_size: int = 16,
        num_groups_hyena: int = 4096,
        num_groups_hyena_medium: int = 256,
        num_groups_hyena_short: int = 256,
        rms_norm_eps: float = 1e-6,
        rotary_base: float | None = None,
        use_short_conv_bias: bool = False,
        hidden_act: str = "gelu",
        gelu_approximate: str = "none",
        gated_linear_unit: bool = True,
        remove_activation_post_first_layer: bool = True,
        **kwargs,
    ) -> None:
        """Initialize an Evo2 checkpoint configuration."""
        kwargs.setdefault("architectures", ["Evo2ForCausalLM"])
        kwargs.setdefault("tie_word_embeddings", True)
        rope_theta = kwargs.pop("rope_theta", None)
        if rotary_base is None:
            rotary_base = 10000.0 if rope_theta is None else float(rope_theta)
        elif rope_theta is not None and float(rope_theta) != float(rotary_base):
            raise ValueError("rotary_base and rope_theta must match when both are provided")
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.hybrid_override_pattern = hybrid_override_pattern
        self.short_conv_length = short_conv_length
        self.hcs_filter_length = hcs_filter_length
        self.hcm_filter_length = hcm_filter_length
        self.hcl_state_size = hcl_state_size
        self.num_groups_hyena = num_groups_hyena
        self.num_groups_hyena_medium = num_groups_hyena_medium
        self.num_groups_hyena_short = num_groups_hyena_short
        self.rms_norm_eps = rms_norm_eps
        self.rotary_base = float(rotary_base)
        self.rope_theta = float(rotary_base)
        self.use_short_conv_bias = use_short_conv_bias
        self.hidden_act = hidden_act
        self.gelu_approximate = gelu_approximate
        self.gated_linear_unit = gated_linear_unit
        self.remove_activation_post_first_layer = remove_activation_post_first_layer

        self._validate()
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.operator_types = tuple(self.hybrid_override_pattern)
        self.layers_block_type = ["attention" if symbol == "*" else "mamba" for symbol in self.operator_types]

    def _validate(self) -> None:
        if len(self.hybrid_override_pattern) != self.num_hidden_layers:
            raise ValueError("hybrid_override_pattern length must equal num_hidden_layers")
        invalid_symbols = set(self.hybrid_override_pattern) - _LAYER_SYMBOLS
        if invalid_symbols:
            raise ValueError(f"unsupported Evo2 layer symbols: {sorted(invalid_symbols)}")
        if self.hidden_size <= 0 or self.num_attention_heads <= 0:
            raise ValueError("hidden size and attention heads must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if self.num_key_value_heads <= 0 or self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if self.hidden_act not in ("gelu", "silu", "identity"):
            raise ValueError(f"unsupported Evo2 MLP activation: {self.hidden_act}")
        if self.gelu_approximate not in ("none", "tanh"):
            raise ValueError(f"unsupported GELU approximation: {self.gelu_approximate}")
        if self.gated_linear_unit is not True:
            raise ValueError("Evo2 vLLM requires gated_linear_unit=true")
        if not isinstance(self.remove_activation_post_first_layer, bool):
            raise ValueError("remove_activation_post_first_layer must be boolean")
        if min(self.short_conv_length, self.hcs_filter_length, self.hcm_filter_length, self.hcl_state_size) < 2:
            raise ValueError("Evo2 recurrent filter and state lengths must be at least two")

    def local_state_shapes(self, tp_size: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the two uniform local recurrent-cache shapes used by vLLM."""
        if isinstance(tp_size, bool) or not isinstance(tp_size, int) or tp_size <= 0:
            raise ValueError("tensor parallel size must be a positive integer")
        if self.hidden_size % tp_size:
            raise ValueError("hidden size must be divisible by tensor parallel size")
        if self.num_attention_heads % tp_size:
            raise ValueError("attention heads must be divisible by tensor parallel size")

        local_hidden_size = self.hidden_size // tp_size
        operator_state_length = max(
            self.hcs_filter_length - 1,
            self.hcm_filter_length - 1,
            self.hcl_state_size,
        )
        return (
            (3 * local_hidden_size, self.short_conv_length - 1),
            (local_hidden_size, operator_state_length),
        )
