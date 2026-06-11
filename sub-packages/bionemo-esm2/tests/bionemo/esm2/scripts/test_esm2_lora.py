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

import glob
from pathlib import Path
from typing import get_args

import pytest
import torch

from bionemo.core.data.load import load
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.scripts.finetune_esm2 import train_model
from bionemo.esm2.scripts.infer_esm2 import infer_model
from bionemo.llm.lightning import batch_collator
from bionemo.llm.utils.callbacks import IntervalT
from bionemo.testing import megatron_parallel_state_utils


@pytest.mark.needs_gpu
@pytest.mark.parametrize("prediction_interval", get_args(IntervalT))
@pytest.mark.parametrize("precision", ["fp32", "bf16-mixed"])
def test_different_results_with_peft(
    tmp_path,
    dummy_data_per_token_classification_ft,
    dummy_protein_sequences,
    prediction_interval,
    precision,
    data_to_csv,
    n_steps_train=2,
    seed=42,
):
    checkpoint_path = Path(load("esm2/8m:2.0"))
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        lora_checkpoint_path, _, _ = train_model(
            train_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            experiment_name="finetune_new_head_token_classification",
            restore_from_checkpoint_path=checkpoint_path,
            num_steps=n_steps_train,
            num_nodes=1,
            num_gpus=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=10,
            lr=1e-5,
            scale_lr_layer="classification_head",
            lr_multiplier=1e2,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            task_type="classification",
            labels_mask_column="resolved",
            label_column="labels",
            encoder_frozen=True,
            dataset_class="InMemoryPerTokenValueDataset",
            config_class="ESM2FineTuneTokenConfig",
            lora_finetune=True,
            create_tensorboard_logger=False,
        )
    assert lora_checkpoint_path.exists(), "Could not find test results directory."
    data_path = data_to_csv(dummy_protein_sequences, tmp_path)
    result_dir_original = tmp_path / "results_original"
    min_seq_len = 1024  # Minimum length of the output batch; tensors will be padded to this length.
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        infer_model(
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            results_path=result_dir_original,
            min_seq_length=min_seq_len,
            prediction_interval=prediction_interval,
            include_hiddens=True,
            precision=precision,
            include_embeddings=True,
            include_input_ids=True,
            include_logits=True,
            micro_batch_size=3,  # dataset length (10) is not multiple of 3; this validates partial batch inference
            config_class=ESM2Config,
            lora_checkpoint_path=None,
        )
    assert result_dir_original.exists(), "Could not find test results directory."
    result_dir_peft = tmp_path / "results_peft"
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        infer_model(
            data_path=data_path,
            checkpoint_path=checkpoint_path,
            results_path=result_dir_peft,
            min_seq_length=min_seq_len,
            prediction_interval=prediction_interval,
            include_hiddens=True,
            precision=precision,
            include_embeddings=True,
            include_input_ids=True,
            include_logits=True,
            micro_batch_size=3,  # dataset length (10) is not multiple of 3; this validates partial batch inference
            config_class=ESM2Config,
            lora_checkpoint_path=lora_checkpoint_path,
        )

    if prediction_interval == "epoch":
        results_original = torch.load(f"{result_dir_original}/predictions__rank_0__dp_rank_0.pt")
        results_peft = torch.load(f"{result_dir_peft}/predictions__rank_0__dp_rank_0.pt")

    elif prediction_interval == "batch":
        results_original = batch_collator(
            [
                torch.load(f, map_location="cpu")
                for f in glob.glob(f"{result_dir_original}/predictions__rank_0__dp_rank_0__batch_*.pt")
            ]
        )
        results_peft = batch_collator(
            [
                torch.load(f, map_location="cpu")
                for f in glob.glob(f"{result_dir_peft}/predictions__rank_0__dp_rank_0__batch_*.pt")
            ]
        )
    assert (results_original["embeddings"] != results_peft["embeddings"]).any()
    assert (results_original["hidden_states"] != results_peft["hidden_states"]).any()
    assert (results_original["token_logits"] != results_peft["token_logits"]).any()
