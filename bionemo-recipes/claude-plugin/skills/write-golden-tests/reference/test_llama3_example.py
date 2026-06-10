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

"""Reference: Llama3 test implementation (decoder pattern).

Shows how to implement golden value tests for a decoder/causal LM model.
Key differences from encoder tests:
- Uses text sequences instead of protein sequences
- DataCollatorForLanguageModeling with mlm=False (causal LM)
- Tests both BSHD and THD input formats
- Can test generation/KV-cache
"""

import torch
from convert import convert_llama_hf_to_te, convert_llama_te_to_hf
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class TestLlama3Model:
    """Test suite for Llama3 TE model."""

    upstream_model_id = "meta-llama/Llama-3.2-1B-Instruct"

    def get_test_input_data(self):
        tokenizer = AutoTokenizer.from_pretrained(self.upstream_model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        ]
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        batch = collator([tokenizer(text) for text in test_texts])
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def test_golden_values(self):
        """HF and TE models should produce matching outputs."""
        import transformers

        model_hf = transformers.LlamaForCausalLM.from_pretrained(self.upstream_model_id, dtype=torch.bfloat16).cuda()
        model_te = convert_llama_hf_to_te(model_hf).cuda()
        input_data = self.get_test_input_data()

        with torch.no_grad():
            hf_out = model_hf(**input_data)
            te_out = model_te(**input_data)

        torch.testing.assert_close(te_out.loss, hf_out.loss, atol=5e-3, rtol=0.01)
        torch.testing.assert_close(te_out.logits, hf_out.logits, atol=1.5, rtol=0.01)

    def test_roundtrip_conversion(self):
        """HF->TE->HF preserves weights."""
        import transformers

        model_hf = transformers.LlamaForCausalLM.from_pretrained(self.upstream_model_id).cuda()
        model_te = convert_llama_hf_to_te(model_hf)
        model_hf_back = convert_llama_te_to_hf(model_te)

        for (n1, p1), (n2, p2) in zip(model_hf.named_parameters(), model_hf_back.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Roundtrip mismatch: {n1}")
