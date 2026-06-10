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

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformer_engine.pytorch.fp8 import check_fp8_support
from transformers import AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


# Custom pytest decorator
def requires_fp8(func):
    """Decorator to skip tests that require FP8 support."""
    fp8_available, reason = check_fp8_support()
    return pytest.mark.skipif(not fp8_available, reason=f"FP8 is not supported on this GPU: {reason}")(func)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("chandar-lab/AMPLIFY_120M", revision="d918a9e8")


@pytest.fixture
def config():
    config = AutoConfig.from_pretrained("chandar-lab/AMPLIFY_120M", trust_remote_code=True, revision="d918a9e8")
    config.dtype = torch.bfloat16
    return config


@pytest.fixture
def input_data(tokenizer):
    torch.manual_seed(42)

    test_proteins = [
        "MLSATEKLSDYISSLFASVSIINSISTEDLFFLKLTCQTFSKDSEEYKAAYRILRGVQRGKVQIIEEALVS",
        "MFVFFAGTLVNQDTLNFRDQLNINVVGTVRGIAQDASKYLEYAIDSV",
        "MAATGSLILSDEEQAELIALAVRIVLACAGGSQNKELAAQLGVIETTVGEWRRRFAQNRVEGLRDEARPGAPSDDQ",
        "MSAVLSAVASDDWTAFAKLVHPYVHWTADGITTRGRTRVMARLSGHDGVKPASSYELRDGQVYRWTS",
        "MSDPAAEPPADTSGIAWRKSSYSGPNGNCVELAQISGDHVGIRNSRDLHGSVLTCTRAEFAALLCDIKAGRFDSLIL",
        "MRRPKLRRSGVLMSHPARGQPIKDASTEAAAERRPHVTSSERQDVSDQDTR",
        "MQTITVAGGNLFQIAAQYLGDATQWIRIAQLNGLADPVLSGVVTLTIPQPNPLAGGGVVGQ",
        "MVFSLEQFVRGQGWQSITSNSDNEVPKPRQVYEVKAVCHPGAWRVKARVFGTSQGIPFDYSQASMERRVAQDECDRRPQ",
        "AGDGTGCNPTLSKAAGVELDNSDSGEVFVIYLHIIIAIIVLISINLIGFLYF",
        "MKVGVDPSVCEAHGACMSILPEVFDLDDDEVLQIRDGELAPSEEESAERAVASCPMGALRLSR",
        "MWISERPPSRMALGSQSQMSLPGIPARCLHS",
        "MIDNSIRLFDADDSELFSLAEVPLDNKPIQRDTDSLSQWGDTWLREIQHS",
        "MVKNLFFNKIKNATLKVANISRCYLPFPPPPCPPPEPLEPPEPPAPLEPAPDPPPLPPFPVPDILPAI",
        "MSYINDITQSNSSILNVNVKINDHNSDEMYRNETKWYGEQFRYQSNPRFSRSSTSKNEKGFVQKKT",
        "MQILILPIPDQLQNPNKISQHLICITFVSEQTLPI",
    ]

    dataset = Dataset.from_list([{"sequence": p} for p in test_proteins])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=1024,
        seed=42,
    )

    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"],
            truncation=True,
            padding="max_length",
            max_length=1024,
            return_tensors="pt",
        )

    tokenized_proteins = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["sequence"],
    )

    dataloader = DataLoader(
        tokenized_proteins,
        batch_size=len(tokenized_proteins),
        collate_fn=data_collator,
    )

    batch = next(iter(dataloader))
    batch = {k: v.to("cuda") for k, v in batch.items()}
    return batch
