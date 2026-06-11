# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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


import pytest
import torch

from bionemo.core.data.load import load
from bionemo.evo2.run.infer import infer
from bionemo.testing.megatron_parallel_state_utils import clean_parallel_state_context
from bionemo.testing.torch import check_fp8_support


RANDOM_SEED = 42


@pytest.mark.parametrize("fast", [True, False])
def test_run_infer(fast: bool):
    # Create PTL trainer.
    tensor_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    temperature = 1.0
    top_k = 0
    top_p = 0.0
    max_new_tokens = 1

    # Generation args.
    default_prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )
    try:
        checkpoint_path = load("evo2/1b-8k:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    is_fp8_supported, _, _ = check_fp8_support(torch.cuda.current_device())

    with clean_parallel_state_context():
        infer(
            prompt=default_prompt,
            ckpt_dir=checkpoint_path,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            vortex_style_fp8=is_fp8_supported,
            flash_decode=fast,
        )
