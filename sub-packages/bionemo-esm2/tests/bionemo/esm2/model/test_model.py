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

import io
import tarfile
from unittest import mock

import pytest
import torch
from torch import Tensor
from transformers import AutoModelForMaskedLM

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.core.utils.random_utils import random_numpy_context
from bionemo.esm2.api import ESM2Config, ESM2Model
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.embedding import ESM2Embedding
from bionemo.esm2.testing.compare import assert_esm2_equivalence
from bionemo.llm.model.biobert.model import MegatronBioBertModel
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping
from bionemo.testing import megatron_parallel_state_utils


def test_esm2_model_initialized():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        tokenizer = get_tokenizer()
        config = ESM2Config()
        model = config.configure_model(tokenizer)

        assert isinstance(model, MegatronBioBertModel)
        assert isinstance(model, ESM2Model)
        assert isinstance(model.embedding, ESM2Embedding)


def test_esm2_nemo1_checkpoint():
    with tarfile.open(load("esm2/nv_650m:1.0"), "r") as ckpt, torch.no_grad():
        ckpt_file = ckpt.extractfile("./model_weights.ckpt")

        old_state_dict = torch.load(ckpt_file)
        # megatron is not registering inv_freq params anymore.
        # TODO: update Bionemo checkpoints
        old_state_dict.pop("model.language_model.rotary_pos_emb.inv_freq")

        with megatron_parallel_state_utils.distributed_model_parallel_state():
            tokenizer = get_tokenizer()
            config = ESM2Config()
            model = config.configure_model(tokenizer)
            new_state_dict = model.state_dict_for_save_checkpoint()

        # Set the new_model_prefix to "" since we are looking at the base megatron model and not the lightning module
        # which stores a copy of this model into self.module
        old_keys = {
            nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=True) for k in old_state_dict
        }
        assert len(old_keys) == len(old_state_dict), "Mapping unexpectedly discarded some keys."

        new_keys = set(new_state_dict)
        for k, v in old_state_dict.items():
            # Make sure the shapes of the weights match.
            assert (
                new_state_dict[nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=True)].shape
                == v.shape
            )

        extra_keys = new_keys.difference(old_keys)
        extra_non_null_keys = {
            k
            for k in extra_keys
            if not k.endswith("._extra_state")
            and new_state_dict[k] is not None
            and not isinstance(new_state_dict[k], io.BytesIO)
        }
        assert not extra_non_null_keys, "There are new keys that have state that is missing from the old checkpoint."

        missing_old_keys = old_keys.difference(new_keys)
        assert not missing_old_keys, "There are keys in the old checkpoint that are missing from the new model."


def _compute_loss(model, dataloader, vocab_size=None):
    loss = 0
    n = 0
    limit_batches = 10
    for i, batch in enumerate(dataloader):
        assert isinstance(batch, dict)
        result = model(input_ids=batch["text"].cuda(), attention_mask=batch["attention_mask"].cuda())

        # bionemo ESM2 vocab_size
        if vocab_size is not None:
            # token_logits is s,b and for simplicity here let's transpose to b,s. In general this reduces performance.
            logits = result["token_logits"].transpose(0, 1).contiguous()[..., :vocab_size]
        else:
            logits = result.logits

        loss_mask = batch["loss_mask"].cuda()
        target = batch["labels"].cuda()

        loss += torch.nn.functional.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
        n += loss_mask.sum()

        if limit_batches is not None and i + 1 >= limit_batches:
            break
    mean_loss: Tensor = loss / n
    return mean_loss


def test_esm2_loss(dummy_protein_dataset, dummy_parquet_train_val_inputs):
    hf_model_tag = "facebook/esm2_t6_8M_UR50D"
    nv_model_tag = "esm2/8m:2.0"
    # hf_model_tag = "facebook/esm2_t33_650M_UR50D"
    # nv_model_tag = "esm2/650m:2.0"

    train_cluster_path, valid_cluster_path = dummy_parquet_train_val_inputs

    seed: int = 42

    with (
        torch.inference_mode(),
        megatron_parallel_state_utils.distributed_model_parallel_state(seed),
        random_numpy_context(seed),
    ):
        tokenizer = get_tokenizer()

        # ESM2 model initialized with params
        model = ESM2Config(initial_ckpt_path=str(load(nv_model_tag))).configure_model(tokenizer).cuda()

        # Initialize the data module.
        data_module = ESMDataModule(
            train_cluster_path=train_cluster_path,
            train_database_path=dummy_protein_dataset,
            valid_cluster_path=valid_cluster_path,
            valid_database_path=dummy_protein_dataset,
            global_batch_size=4,
            micro_batch_size=2,
            min_seq_length=None,
            max_seq_length=1024,
            seed=seed,
            num_workers=1,
        )
        assert data_module is not None
        data_module.trainer = mock.Mock()
        data_module.trainer.max_epochs = 1
        data_module.trainer.max_steps = 10
        data_module.trainer.val_check_interval = 2
        data_module.trainer.limit_val_batches = 1

        data_module.setup()

        train_dataloader = data_module.train_dataloader()
        assert isinstance(train_dataloader, torch.utils.data.DataLoader)

        val_dataloader = data_module.val_dataloader()
        assert isinstance(val_dataloader, torch.utils.data.DataLoader)

        mean_loss = _compute_loss(model, train_dataloader, vocab_size=tokenizer.vocab_size)

        # HF model initialized with params
        hf_model = AutoModelForMaskedLM.from_pretrained(hf_model_tag, torch_dtype=get_autocast_dtype(32)).cuda()
        hf_mean_loss = _compute_loss(hf_model, train_dataloader)
        print(f"hf_mean_loss: {hf_mean_loss}")

        torch.testing.assert_close(mean_loss, hf_mean_loss, atol=1e-3, rtol=0.0)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16", "bf16-mixed"])
def test_model_equivalence_with_huggingface_8m(precision):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    ckpt_path = load("esm2/8m:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(ckpt_path, model_tag, precision=precision)


@pytest.mark.slow
def test_model_equivalence_with_huggingface_650m():
    model_tag = "facebook/esm2_t33_650M_UR50D"
    ckpt_path = load("esm2/650m:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(ckpt_path, model_tag, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_model_equivalence_with_huggingface_650m_bf16():
    model_tag = "facebook/esm2_t33_650M_UR50D"
    ckpt_path = load("esm2/650m:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(ckpt_path, model_tag, precision="bf16")


@pytest.mark.slow
@pytest.mark.skip(reason="This test triggers a large download from huggingface and requires considerable GPU memory.")
def test_model_equivalence_with_huggingface_3b():
    model_tag = "facebook/esm2_t36_3B_UR50D"
    ckpt_path = load("esm2/3b:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(ckpt_path, model_tag, atol=1e-4, rtol=1e-4)
