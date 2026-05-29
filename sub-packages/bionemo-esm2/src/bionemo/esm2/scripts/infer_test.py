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

from typing import Tuple

import torch

from bionemo.core.data.load import load
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.testing import megatron_parallel_state_utils


def infer(prot_seq: str, model_name: str = "esm2/650m:2.0") -> Tuple[torch.Tensor, torch.Tensor]:
    """Infer the ESM2 model on a given protein sequence."""
    tensor_model_parallel_size = 1
    pipeline_model_parallel_size = 1
    # micro_batch_size=2
    # num_nodes=1
    # devices=1
    # global_batch_size=1

    tokenizer = get_tokenizer()
    tokens = tokenizer(prot_seq, return_tensors="pt", padding=True, truncation=True).to("cuda")
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    dtype = torch.float32
    ckpt_path = load(model_name)
    nemo_config = ESM2Config(
        initial_ckpt_path=ckpt_path,
        include_embeddings=True,
        include_hiddens=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        initial_ckpt_skip_keys_with_these_prefixes=[],  # load everything from the checkpoint.
    )

    with megatron_parallel_state_utils.distributed_model_parallel_state():
        nemo_model = nemo_config.configure_model(tokenizer).to("cuda").eval()

        nemo_output = nemo_model(input_ids, attention_mask)

        nemo_logits = nemo_output["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]
        nemo_hidden_state = nemo_output["hidden_states"]

    return nemo_logits, nemo_hidden_state


if __name__ == "__main__":
    # MGYP003656012161
    seq_a = "MRVCIDIDGTICETDSSKEYAEVSPIRGARVALIALKQEGHHIILFTARHMKTCKGDADLAENKQRGTLEWWLGYHEIPYDELIFGKPYADIYIDDKGYKFEDWGDTLRDNF"
    # MGYP001468264167
    seq_b = (
        "MTYVIDIDGTLCSLTNGDYTSAKPFKERIAKVNRLYREGHTIILHTARGMGRFEGNRWKSYNQFYFFTERQLKKWDVQYHQLVMGKPSGDFYIDDKGIKDEDFFADETR"
    )
    # MGYP003968060607
    seq_c = "MAQKMPKGVLFDLDGTLLDSAPDFIVSLNTLLQKYNRPELDPEIIRSHVSDGSWKLVSLGFGIEESHDDCAQLREELLIEYEKNSLVYGSAFAGISNVLDYLLELKIPYGVVTNKPLRFAEPILQNEPAFKNCRTLVCPDHINKIKPNPEGILKGCEDLGISPSDCIYVGDHMKDLEAGINAGTRVIACYFGYSLKIGEHDKNIQGANHPIDLIDLIKA"

    _, hidden_a = infer(prot_seq=seq_a)
    _, hidden_b = infer(prot_seq=seq_b)
    _, hidden_c = infer(prot_seq=seq_c)

    distance_a_b = torch.cdist(hidden_a, hidden_b, p=2).mean()
    distance_a_c = torch.cdist(hidden_a, hidden_c, p=2).mean()
    distance_b_c = torch.cdist(hidden_b, hidden_c, p=2).mean()

    assert distance_a_b < distance_a_c, "distance_a_b should be smaller than distance_a_c"

    print(f"distance_a_b: {distance_a_b}")
    print(f"distance_a_c: {distance_a_c}")
