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

import nemo.lightning as nl
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm import generate

from bionemo.core.data.load import load
from bionemo.testing.megatron_parallel_state_utils import clean_parallel_state_context


RANDOM_SEED = 42


def test_infer_model_generates_expected_single_token_output():
    # Create PTL trainer.
    TENSOR_PARALLEL_SIZE = 1
    PIPELINE_MODEL_PARALLEL_SIZE = 1
    CONTEXT_PARALLEL_SIZE = 1
    NUM_GPUS = 1
    NUM_NODES = 1

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=TENSOR_PARALLEL_SIZE,
        pipeline_model_parallel_size=PIPELINE_MODEL_PARALLEL_SIZE,
        context_parallel_size=CONTEXT_PARALLEL_SIZE,
        pipeline_dtype=torch.bfloat16,
        ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
        ckpt_save_optimizer=False,
        ckpt_async_save=False,
        save_ckpt_format="torch_dist",
        ckpt_load_strictness="log_all",
    )
    trainer = nl.Trainer(
        accelerator="gpu",
        num_nodes=NUM_NODES,
        devices=NUM_GPUS,
        strategy=strategy,
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )
    temperature = 1.0
    top_k = 0
    top_p = 0.0
    max_new_tokens = 1
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
    with clean_parallel_state_context():
        results = generate(
            path=checkpoint_path,
            prompts=[prompt],
            trainer=trainer,
            inference_params=CommonInferenceParams(
                temperature,
                top_k,
                top_p,
                return_log_probs=False,
                num_tokens_to_generate=max_new_tokens,
            ),
            random_seed=RANDOM_SEED,
            text_only=True,
        )

        assert isinstance(results, list)
        assert results == ["T"]
