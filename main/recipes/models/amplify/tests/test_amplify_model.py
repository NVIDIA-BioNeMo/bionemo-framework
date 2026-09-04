# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import transformer_engine
import transformer_engine.pytorch
from conftest import requires_fp8
from transformer_engine.common.recipe import DelayedScaling, Format

import amplify.amplify_hf as amp_hf
import amplify.amplify_te as amp_te
from amplify.state_dict_convert import convert_amplify_hf_to_te


def test_amplify_hf_model(config, input_data):
    model = amp_hf.AMPLIFY(config)
    model.to("cuda")
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(**input_data)


def test_amplify_te_model(config, input_data):
    config = amp_te.AMPLIFYConfig(**config.to_dict())
    model = amp_te.AMPLIFY(config)
    model.to("cuda")
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        model(**input_data)


@pytest.mark.fp8
@requires_fp8
def test_amplify_te_model_with_fp8(config, input_data):
    config = amp_te.AMPLIFYConfig(**config.to_dict())
    config.pad_vocab_size_to_multiple_of = 8
    model = amp_te.AMPLIFYForMaskedLM(config)
    model.to("cuda")
    model.eval()
    fp8_recipe = DelayedScaling(fp8_format=Format.HYBRID, amax_history_len=16, amax_compute_algo="max")
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        model(**input_data)


def test_te_model_has_all_te_layers(config):
    config = amp_te.AMPLIFYConfig(**config.to_dict())
    model = amp_te.AMPLIFY(config)
    for name, module in model.named_modules():
        assert not isinstance(module, nn.Linear), f"Vanilla linear layer found in {name}"
        assert not isinstance(module, nn.LayerNorm), f"Vanilla LayerNorm layer found in {name}"
        assert not isinstance(module, nn.RMSNorm), f"Vanilla RMSNorm layer found in {name}"


def test_models_have_identical_outputs(input_data):
    model_hf = amp_hf.AMPLIFY.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model_te = convert_amplify_hf_to_te(model_hf)
    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    model_hf.to("cuda", dtype=torch.bfloat16)
    model_te.to("cuda", dtype=torch.bfloat16)
    model_hf.eval()
    model_te.eval()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs_te = model_te(**input_data)
        outputs_hf = model_hf(**input_data)

    torch.testing.assert_close(outputs_hf.loss, outputs_te.loss, atol=1e-2, rtol=1e-3)


def test_converted_model_roundtrip(input_data, tmp_path):
    model_hf = amp_hf.AMPLIFY.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model_te = convert_amplify_hf_to_te(model_hf)

    model_te.save_pretrained(tmp_path / "AMPLIFY_120M")
    del model_te

    model_te = amp_te.AMPLIFYForMaskedLM.from_pretrained(tmp_path / "AMPLIFY_120M")

    input_data = {k: v.to("cuda") for k, v in input_data.items()}

    model_hf.to("cuda", dtype=torch.bfloat16)
    model_te.to("cuda", dtype=torch.bfloat16)
    model_hf.eval()
    model_te.eval()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs_te = model_te(**input_data)
        outputs_hf = model_hf(**input_data)

    torch.testing.assert_close(outputs_hf.loss, outputs_te.loss, atol=1e-2, rtol=1e-3)


def test_convert_state_dict():
    model_hf = amp_hf.AMPLIFY.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model_te = convert_amplify_hf_to_te(model_hf)

    from amplify.state_dict_convert import _pack_qkv_weight, _pad_bias, _pad_weights, mapping

    te_state_dict_keys = {
        k for k in model_te.state_dict().keys() if not k.endswith("_extra_state") and not k.endswith("inv_freq")
    }

    for k, v in mapping.items():
        if "*" in k:
            for i in range(model_hf.config.num_hidden_layers):
                k_sub = k.replace("*", str(i))
                v_sub = v.replace("*", str(i))
                torch.testing.assert_close(model_te.state_dict()[v_sub], model_hf.state_dict()[k_sub])
                te_state_dict_keys.remove(v_sub)
        else:
            torch.testing.assert_close(model_te.state_dict()[v], model_hf.state_dict()[k])
            te_state_dict_keys.remove(v)

    for i in range(model_hf.config.num_hidden_layers):
        k = f"amplify.transformer_encoder.{i}.self_attention.layernorm_qkv.weight"
        v = [
            f"transformer_encoder.{i}.q.weight",
            f"transformer_encoder.{i}.k.weight",
            f"transformer_encoder.{i}.v.weight",
        ]

        ctx_mock = MagicMock()
        ctx_mock.target.config.num_attention_heads = model_hf.config.num_attention_heads

        packed_weight = _pack_qkv_weight.transform(
            ctx_mock,
            model_hf.state_dict()[v[0]],
            model_hf.state_dict()[v[1]],
            model_hf.state_dict()[v[2]],
        )

        torch.testing.assert_close(packed_weight, model_te.state_dict()[k])
        te_state_dict_keys.remove(k)

    ctx_mock = MagicMock()
    ctx_mock.target.config.padded_vocab_size = model_te.config.padded_vocab_size

    torch.testing.assert_close(
        _pad_weights(ctx_mock, model_hf.state_dict()["encoder.weight"]),
        model_te.state_dict()["amplify.encoder.weight"],
    )
    torch.testing.assert_close(
        _pad_weights(ctx_mock, model_hf.state_dict()["decoder.weight"]), model_te.state_dict()["decoder.weight"]
    )
    torch.testing.assert_close(
        _pad_bias.transform(ctx_mock, model_hf.state_dict()["decoder.bias"]), model_te.state_dict()["decoder.bias"]
    )

    te_state_dict_keys.remove("amplify.encoder.weight")
    te_state_dict_keys.remove("decoder.weight")
    te_state_dict_keys.remove("decoder.bias")

    assert len(te_state_dict_keys) == 0


def test_hf_trained_model_loss(input_data):
    model = amp_hf.AMPLIFY.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model.to("cuda", dtype=torch.bfloat16)
    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**input_data)

    torch.testing.assert_close(output.loss.detach().cpu(), torch.tensor(2.4), atol=1e-1, rtol=1e-2)


def test_te_trained_model_loss(input_data):
    model_hf = amp_hf.AMPLIFY.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model = convert_amplify_hf_to_te(model_hf)
    model.to("cuda", dtype=torch.bfloat16)
    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**input_data)

    torch.testing.assert_close(output.loss.detach().cpu(), torch.tensor(2.4), atol=1e-1, rtol=1e-2)


def test_hf_reinitialized_model_loss(input_data):
    config = amp_hf.AMPLIFYConfig.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model = amp_hf.AMPLIFY(config)
    model.to("cuda", dtype=torch.bfloat16)
    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**input_data)

    loss = output.loss.detach().cpu()
    assert loss < 3.5, f"Loss is {loss}, expected less than 3.5"


def test_te_reinitialized_model_loss(input_data):
    config = amp_te.AMPLIFYConfig.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")
    model = amp_te.AMPLIFYForMaskedLM(config)
    model.to("cuda", dtype=torch.bfloat16)
    input_data = {k: v.to("cuda") for k, v in input_data.items()}
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(**input_data)

    loss = output.loss.detach().cpu()
    assert loss < 3.5, f"Loss is {loss}, expected less than 3.5"
