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
import torch.nn as nn
import transformer_engine.pytorch as te

from loss import compute_plddt
from minifold_utils import init
from te_utils import te_layernorm_nd, te_linear_nd


class PerResidueLDDTCaPredictorTE(nn.Module):
    """TE version of PerResidueLDDTCaPredictor."""

    def __init__(self, no_bins, c_in, c_hidden, params_dtype=torch.float32):
        super().__init__()

        self.no_bins = no_bins
        self.c_in = c_in
        self.c_hidden = c_hidden

        self.layer_norm = te.LayerNorm(self.c_in, eps=1e-5, params_dtype=params_dtype)
        self.linear_1 = te.Linear(self.c_in, self.c_hidden, params_dtype=params_dtype)
        self.linear_2 = te.Linear(self.c_hidden, self.c_hidden, params_dtype=params_dtype)
        self.linear_3 = te.Linear(self.c_hidden, self.no_bins, params_dtype=params_dtype)

        init.he_normal_init_(self.linear_1.weight)
        init.he_normal_init_(self.linear_2.weight)
        init.final_init_(self.linear_3.weight)

        init.bias_init_zero_(self.linear_1.bias)
        init.bias_init_zero_(self.linear_2.bias)
        init.bias_init_zero_(self.linear_3.bias)

        self.relu = nn.ReLU()

    def forward(self, s):
        s = te_layernorm_nd(self.layer_norm, s)
        s = te_linear_nd(self.linear_1, s)
        s = self.relu(s)
        s = te_linear_nd(self.linear_2, s)
        s = self.relu(s)
        s = te_linear_nd(self.linear_3, s)
        return s


class AuxiliaryHeadsTE(nn.Module):
    """TE version of AuxiliaryHeads."""

    def __init__(self, config):
        super().__init__()
        self.plddt = PerResidueLDDTCaPredictorTE(
            **config["lddt"],
        )
        self.config = config

    def forward(self, outputs):
        aux_out = {}
        lddt_logits = self.plddt(outputs["sm"]["single"])
        aux_out["lddt_logits"] = lddt_logits
        aux_out["plddt"] = compute_plddt(lddt_logits)
        return aux_out
