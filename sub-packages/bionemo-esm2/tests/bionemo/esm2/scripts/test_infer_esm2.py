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
from typing import get_args

import pandas as pd
import pytest
import torch

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.finetune.sequence_model import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.token_model import ESM2FineTuneTokenConfig
from bionemo.esm2.scripts.infer_esm2 import infer_model
from bionemo.llm.data import collate
from bionemo.llm.lightning import batch_collator
from bionemo.llm.utils.callbacks import IntervalT


@pytest.fixture
def dummy_protein_csv(tmp_path, dummy_protein_sequences):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame
    df = pd.DataFrame(dummy_protein_sequences, columns=["sequences"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def padded_tokenized_sequences(dummy_protein_sequences):
    tokenizer = get_tokenizer()
    tokenized_sequences = [
        tokenizer.encode(seq, add_special_tokens=True, return_tensors="pt") for seq in dummy_protein_sequences
    ]
    batch = [{"text": tensor.flatten()} for tensor in tokenized_sequences]
    collated_batch = collate.bert_padding_collate_fn(batch, padding_value=tokenizer.pad_token_id, min_length=1024)
    return collated_batch["text"]


@pytest.mark.needs_gpu
@pytest.mark.parametrize("prediction_interval", get_args(IntervalT))
@pytest.mark.parametrize("precision", ["fp32", "bf16-mixed"])
@pytest.mark.parametrize("config_class", [ESM2Config, ESM2FineTuneTokenConfig, ESM2FineTuneSeqConfig])
def test_infer_without_finetune_head(
    tmpdir,
    dummy_protein_csv,
    dummy_protein_sequences,
    precision,
    prediction_interval,
    padded_tokenized_sequences,
    config_class,
):
    # test inference of ESM2 without finetune head
    checkpoint_path = load("esm2/8m:2.0")
    data_path = dummy_protein_csv
    result_dir = tmpdir / "results"
    min_seq_len = 1024  # Minimum length of the output batch; tensors will be padded to this length.

    if config_class == ESM2FineTuneTokenConfig:
        initial_ckpt_skip_keys_with_these_prefixes = ["classification_head"]
    elif config_class == ESM2FineTuneSeqConfig:
        initial_ckpt_skip_keys_with_these_prefixes = ["classification_head", "regression_head"]
    elif config_class == ESM2Config:
        initial_ckpt_skip_keys_with_these_prefixes = []

    infer_model(
        data_path=data_path,
        checkpoint_path=checkpoint_path,
        results_path=result_dir,
        min_seq_length=min_seq_len,
        prediction_interval=prediction_interval,
        include_hiddens=True,
        precision=precision,
        include_embeddings=True,
        include_input_ids=True,
        include_logits=True,
        micro_batch_size=3,  # dataset length (10) is not multiple of 3; this validates partial batch inference
        config_class=config_class,
        initial_ckpt_skip_keys_with_these_prefixes=initial_ckpt_skip_keys_with_these_prefixes,
    )
    assert result_dir.exists(), "Could not find test results directory."

    if prediction_interval == "epoch":
        results = torch.load(f"{result_dir}/predictions__rank_0__dp_rank_0.pt")
    elif prediction_interval == "batch":
        results = batch_collator(
            [
                torch.load(f, map_location="cpu")
                for f in glob.glob(f"{result_dir}/predictions__rank_0__dp_rank_0__batch_*.pt")
            ]
        )
    assert isinstance(results, dict)
    keys_included = ["token_logits", "hidden_states", "embeddings", "binary_logits", "input_ids"]
    assert all(key in results for key in keys_included)
    assert results["binary_logits"] is None
    assert results["embeddings"].shape[0] == len(dummy_protein_sequences)
    assert results["embeddings"].dtype == get_autocast_dtype(precision)
    # hidden_states are [batch, sequence, hidden_dim]
    assert results["hidden_states"].shape[:-1] == (len(dummy_protein_sequences), min_seq_len)
    # input_ids are [batch, sequence]
    assert results["input_ids"].shape == (len(dummy_protein_sequences), min_seq_len)
    # token_logits are [sequence, batch, num_tokens]
    assert results["token_logits"].shape[:-1] == (min_seq_len, len(dummy_protein_sequences))

    # test 1:1 mapping between input sequence and results
    # this does not apply to "batch" prediction_interval mode since the order of batches may not be consistent
    # due distributed processing. To address this, we optionally include input_ids in the predictions, allowing
    # for accurate mapping post-inference.
    if prediction_interval == "epoch":
        assert torch.equal(padded_tokenized_sequences, results["input_ids"])
