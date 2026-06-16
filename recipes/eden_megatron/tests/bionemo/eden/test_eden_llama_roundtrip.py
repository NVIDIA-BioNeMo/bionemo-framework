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

"""Test mbridge -> HF -> mbridge roundtrip fidelity for Eden (Llama) models.

Verifies that an Eden mbridge checkpoint survives the mbridge -> HF -> mbridge
round trip with bit-exact weight preservation, using the pure state-dict
converters in ``eden_mbridge_hf``.

The checkpoint uses Megatron's stacked layer format where layer weights are
stored as ``decoder.layers.mlp.…`` with shape ``[num_layers, ...]`` rather
than per-layer indexed keys.  Both directions of the converter must
handle this format.
"""

import pytest
import torch

from bionemo.eden.utils.checkpoint.eden_mbridge_hf import (
    hf_to_mbridge_state_dict,
    mbridge_to_hf_state_dict,
)
from bionemo.eden.utils.checkpoint.mbridge_checkpoint_utils import load_mbridge_state_dict


NUM_LAYERS = 2
NUM_HEADS = 32
NUM_KV_HEADS = 8


@pytest.fixture(scope="module")
def eden_ckpt(mbridge_eden_checkpoint):
    """Module-scoped alias for the session-scoped Eden checkpoint."""
    return mbridge_eden_checkpoint


@pytest.fixture(scope="module")
def original_mbridge_sd(eden_ckpt):
    """Load the original mbridge state dict from DCP (stacked layer format)."""
    return load_mbridge_state_dict(eden_ckpt)


@pytest.fixture(scope="module")
def roundtripped_mbridge_sd(original_mbridge_sd):
    """Perform stacked mbridge -> HF -> stacked mbridge roundtrip.

    Uses the 2-layer eden_7b config that the ``mbridge_eden_checkpoint`` fixture creates.
    Both the input and output use Megatron's stacked layer format.
    """
    sd = {k: v.clone() for k, v in original_mbridge_sd.items() if isinstance(v, torch.Tensor)}

    hf_sd = mbridge_to_hf_state_dict(sd, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS)

    mbridge_sd = hf_to_mbridge_state_dict(hf_sd, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS)

    return mbridge_sd


def _model_weight_keys(sd: dict[str, object]) -> set[str]:
    """Return tensor keys that are model weights (excludes optimizer state)."""
    return {k for k, v in sd.items() if isinstance(v, torch.Tensor) and not k.startswith("optimizer.")}


@pytest.mark.slow
def test_roundtrip_mbridge_weight_equality(
    original_mbridge_sd: dict[str, torch.Tensor],
    roundtripped_mbridge_sd: dict[str, torch.Tensor],
):
    """Verify that stacked mbridge -> HF -> stacked mbridge produces identical weights."""
    orig_keys = _model_weight_keys(original_mbridge_sd)
    rt_keys = _model_weight_keys(roundtripped_mbridge_sd)

    assert orig_keys == rt_keys, (
        f"Key mismatch.\nOnly in original: {sorted(orig_keys - rt_keys)}\n"
        f"Only in roundtripped: {sorted(rt_keys - orig_keys)}"
    )

    for key in sorted(orig_keys):
        assert original_mbridge_sd[key].shape == roundtripped_mbridge_sd[key].shape, f"Shape mismatch for {key}"
        torch.testing.assert_close(
            original_mbridge_sd[key],
            roundtripped_mbridge_sd[key],
            atol=0,
            rtol=0,
            msg=lambda diff: f"Weight mismatch for {key}: {diff}",
        )
