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

import importlib
import os
import socket
import sys
from pathlib import Path

import pytest
import transformer_engine.pytorch
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling


sys.path.append(Path(__file__).parent.parent.as_posix())
sys.path.append(Path(__file__).parent.as_posix())


pytest_plugins = ["tests.common.fixtures"]


# Fix Triton UTF-8 decoding issue by setting CUDA library path
# This bypasses the problematic ldconfig -p call that contains non-UTF-8 characters
if not os.environ.get("TRITON_LIBCUDA_PATH"):
    # Set the path to CUDA libraries in the NVIDIA PyTorch container
    os.environ["TRITON_LIBCUDA_PATH"] = "/usr/local/cuda/lib64"


@pytest.fixture
def unused_tcp_port():
    """Find and return an unused TCP port for torchrun rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@pytest.fixture(autouse=True)
def use_te_debug(monkeypatch):
    monkeypatch.setenv("NVTE_DEBUG", "1")
    monkeypatch.setenv("NVTE_DEBUG_LEVEL", "2")
    importlib.reload(transformer_engine.pytorch)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("esm_fast_tokenizer")


@pytest.fixture
def test_proteins():
    return [
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


@pytest.fixture
def tokenized_proteins(tokenizer, test_proteins):
    return [tokenizer(p, truncation=True, max_length=1024) for p in test_proteins]


@pytest.fixture
def input_data(tokenizer, tokenized_proteins):
    """BSHD mock input data for forward pass tests."""

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=32,
        seed=42,
    )

    return data_collator(tokenized_proteins)


@pytest.fixture
def te_model_checkpoint(tmp_path):
    from convert import convert_esm_hf_to_te

    model_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D", revision="c731040f")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(tmp_path / "te_model_checkpoint")
    return tmp_path / "te_model_checkpoint"
