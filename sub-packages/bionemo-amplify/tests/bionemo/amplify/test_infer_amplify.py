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

from typing import Dict, List, cast

import pandas as pd
import pytest
import torch

from bionemo.amplify.infer_amplify import main
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.llm.data import collate
from bionemo.llm.data.types import BertSample


@pytest.fixture
def mock_protein_sequences():
    return [
        "NLLQFGYMIRCANGRSRPVW",
        "GLLPKVKLVPEQISFILSTRENR",
        "MSAGVITGVLLVFLLLGYLVYALINAEAF",
        "MKFYTIKLPKFLGGIVRAMLGSFRKD",
        "IWLTALKFLGKHAAKHLAKQQLSKL",
        "GLLSVLGSVAKHVLPHVVPVIAEHL",
        "MRAKWRKKRMRRLKRKRRKMRQRSK",
        "LSAMLEDTENWLYEELSLTQDPVV",
        "MAPRGFSCLLLLTSEIDLPVKRRA",
        "MASRGALRRCLSPGLPRLLHLSRGLA",
    ]


@pytest.fixture
def mock_protein_csv(tmp_path, mock_protein_sequences):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame with sequence column
    df = pd.DataFrame({"sequences": mock_protein_sequences})

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def padded_tokenized_sequences(mock_protein_sequences):
    tokenizer = BioNeMoAMPLIFYTokenizer()
    # Convert to list of BertSample objects
    batch: List[BertSample] = []
    for seq in mock_protein_sequences:
        # Use the tokenizer's encode method which returns a list of integers
        input_ids = torch.tensor(tokenizer.encode(seq, add_special_tokens=True))
        batch.append(
            {
                "text": input_ids,
                "types": torch.zeros_like(input_ids, dtype=torch.int64),
                "attention_mask": torch.ones_like(input_ids, dtype=torch.int64),
                "labels": torch.full_like(input_ids, -1, dtype=torch.int64),
                "loss_mask": torch.zeros_like(input_ids, dtype=torch.int64),
                "is_random": torch.zeros_like(input_ids, dtype=torch.int64),
            }
        )
    collated_batch = collate.bert_padding_collate_fn(
        cast(List[BertSample], batch), padding_value=int(tokenizer.pad_token_id), min_length=1024
    )
    return collated_batch["text"]


def test_infer_epoch_mode(
    tmpdir,
    mock_protein_csv,
    mock_protein_sequences,
    padded_tokenized_sequences,
):
    data_path = mock_protein_csv
    result_dir = tmpdir / "results"
    seq_len = 1024  # Minimum length of the output batch; tensors will be padded to this length.

    # Run inference
    main(
        data_path=data_path,
        hf_model_name="chandar-lab/AMPLIFY_120M",
        results_path=result_dir,
        seq_length=seq_len,
        prediction_interval="epoch",
        micro_batch_size=3,
        include_embeddings=True,
        include_hiddens=True,
        include_input_ids=True,
        include_logits=True,
    )

    # Load and verify results
    results: Dict[str, torch.Tensor] = {}
    results = cast(Dict[str, torch.Tensor], torch.load(f"{result_dir}/predictions__rank_0__dp_rank_0.pt"))

    assert isinstance(results, dict)
    keys_included = ["token_logits", "hidden_states", "embeddings", "input_ids"]
    assert all(key in results for key in keys_included), f"Missing keys: {set(keys_included) - set(results.keys())}"
    assert results["embeddings"].shape[0] == len(mock_protein_sequences)
    assert results["embeddings"].dtype == get_autocast_dtype("bf16-mixed")
    # hidden_states are [batch, sequence, hidden_dim]
    assert results["hidden_states"].shape[:-1] == (len(mock_protein_sequences), seq_len)
    # input_ids are [batch, sequence]
    assert results["input_ids"].shape == (len(mock_protein_sequences), seq_len)
    # token_logits are [sequence, batch, num_tokens]
    assert results["token_logits"].shape[:-1] == (seq_len, len(mock_protein_sequences))

    # Test 1:1 mapping between input sequence and results
    # This does not apply to "batch" prediction_interval mode since the order of batches may not be consistent
    # due to distributed processing. To address this, we optionally include input_ids in the predictions, allowing
    # for accurate mapping post-inference.
    assert torch.equal(padded_tokenized_sequences, results["input_ids"])
