# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import List

import pytest
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.transformer.identity_op import IdentityOp

from bionemo.llm.model.biobert import transformer_specs
from bionemo.llm.model.layers import TELayerNorm


def test_enum_str_choices():
    options: List[str] = [o.value for o in transformer_specs.BiobertSpecOption]
    for o_str in options:
        # Make sure argparse will be happy with the string equality
        assert o_str == transformer_specs.BiobertSpecOption(o_str)
        if o_str != "random_string":
            # Show that some random string doesn't match
            assert "random_string" != transformer_specs.BiobertSpecOption(o_str)


def test_enum_equality():
    assert (
        transformer_specs.BiobertSpecOption("bert_layer_local_spec")
        == transformer_specs.BiobertSpecOption.bert_layer_local_spec
    )


def test_local_spec_sets_qk_ln():
    spec_with_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_local_spec_with_qk_ln, qk_layernorm=True
    )
    spec_no_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_local_spec_with_qk_ln, qk_layernorm=False
    )
    assert spec_with_qk.submodules.self_attention.submodules.q_layernorm == FusedLayerNorm
    assert (
        spec_with_qk.submodules.self_attention.submodules.q_layernorm
        == spec_with_qk.submodules.self_attention.submodules.k_layernorm
    )
    assert spec_no_qk.submodules.self_attention.submodules.q_layernorm == IdentityOp
    assert (
        spec_no_qk.submodules.self_attention.submodules.q_layernorm
        == spec_no_qk.submodules.self_attention.submodules.k_layernorm
    )


def test_te_spec_sets_qk_ln():
    spec_with_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_with_transformer_engine_and_qk_ln_spec, qk_layernorm=True
    )
    spec_no_qk = transformer_specs.get_biobert_spec(
        transformer_specs.BiobertSpecOption.bert_layer_with_transformer_engine_and_qk_ln_spec, qk_layernorm=False
    )
    assert spec_with_qk.submodules.self_attention.submodules.q_layernorm == TELayerNorm
    assert (
        spec_with_qk.submodules.self_attention.submodules.q_layernorm
        == spec_with_qk.submodules.self_attention.submodules.k_layernorm
    )
    assert spec_no_qk.submodules.self_attention.submodules.q_layernorm == IdentityOp
    assert (
        spec_no_qk.submodules.self_attention.submodules.q_layernorm
        == spec_no_qk.submodules.self_attention.submodules.k_layernorm
    )


def test_get_spec_bad_input():
    with pytest.raises(NotImplementedError):
        transformer_specs.get_biobert_spec("bad_input")
