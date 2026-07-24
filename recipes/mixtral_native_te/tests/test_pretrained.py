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

import torch
from checkpoint import load_pretrained_state_dict
from modeling_mixtral_te import NVMixtralPreTrainedModel, localize_expert_state_dict


def test_localize_expert_state_dict():
    state_dict = {
        "dense.weight": torch.tensor(0),
        "model.layers.0.mlp.experts_gate_up_weight": torch.arange(8),
        "model.layers.0.mlp.experts_down_weight": torch.arange(8) + 10,
    }
    for module in ("experts_gate_up", "experts_down"):
        for expert in range(8):
            state_dict[f"model.layers.0.mlp.{module}.weight{expert}"] = torch.tensor(expert)

    localized = localize_expert_state_dict(state_dict, ep_rank=2, num_local_experts=2)

    assert localized["dense.weight"].item() == 0
    torch.testing.assert_close(
        localized["model.layers.0.mlp.experts_gate_up_weight"],
        torch.tensor([4, 5]),
    )
    torch.testing.assert_close(
        localized["model.layers.0.mlp.experts_down_weight"],
        torch.tensor([14, 15]),
    )
    for module in ("experts_gate_up", "experts_down"):
        assert localized[f"model.layers.0.mlp.{module}.weight0"].item() == 4
        assert localized[f"model.layers.0.mlp.{module}.weight1"].item() == 5
        assert f"model.layers.0.mlp.{module}.weight2" not in localized


def test_model_load_global_state_dict_localizes_experts():
    class Model:
        config = type("Config", (), {"expert_parallel_size": 4, "num_local_experts": 8})()

        def load_state_dict(self, state_dict, strict):
            self.loaded_state_dict = state_dict
            self.strict = strict
            return "incompatible"

        def named_parameters(self):
            return []

    model = Model()
    state_dict = {
        "dense.weight": torch.tensor(0),
        "mlp.experts_gate_up_weight": torch.arange(8),
        "mlp.experts_down_weight": torch.arange(8) + 10,
    }
    for expert in range(8):
        state_dict[f"mlp.experts_down.weight{expert}"] = torch.tensor(expert)

    result = NVMixtralPreTrainedModel.load_global_state_dict(model, state_dict, ep_rank=2)

    assert result == "incompatible"
    assert model.strict is False
    assert model.loaded_state_dict["dense.weight"].item() == 0
    torch.testing.assert_close(model.loaded_state_dict["mlp.experts_gate_up_weight"], torch.tensor([4, 5]))
    torch.testing.assert_close(model.loaded_state_dict["mlp.experts_down_weight"], torch.tensor([14, 15]))
    assert model.loaded_state_dict["mlp.experts_down.weight0"].item() == 4
    assert model.loaded_state_dict["mlp.experts_down.weight1"].item() == 5


def test_load_pretrained_state_dict_from_torch_file(tmp_path):
    path = tmp_path / "model.pt"
    expected = {"dense.weight": torch.arange(4)}
    torch.save(expected, path)

    actual = load_pretrained_state_dict(path)

    torch.testing.assert_close(actual["dense.weight"], expected["dense.weight"])


def test_load_pretrained_state_dict_rejects_directory(tmp_path):
    try:
        load_pretrained_state_dict(tmp_path)
    except FileNotFoundError as error:
        assert "export_hf_state_dict" in str(error)
    else:
        raise AssertionError("Expected a directory input to be rejected")
