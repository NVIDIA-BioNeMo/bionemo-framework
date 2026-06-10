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

"""Reference: ESM2 test implementation (encoder pattern).

Shows how to implement golden value tests for an encoder model.
Key patterns:
- Inheriting from a base test class
- Preparing protein sequence test data with MLM collation
- Setting model-specific tolerances
"""

import torch
from convert import convert_esm_hf_to_te, convert_esm_te_to_hf
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class TestESM2Model:
    """Test suite for ESM2 TE model."""

    upstream_model_id = "facebook/esm2_t6_8M_UR50D"

    def get_test_input_data(self):
        tokenizer = AutoTokenizer.from_pretrained("esm_fast_tokenizer")
        test_proteins = [
            "MLSATEKLSDYISSLFASVSIINSISTEDLFFLK",
            "MFVFFAGTLVNQDTLNFRDQLNINVVGTVRGIAQ",
        ]
        tokenized = [tokenizer(p, truncation=True, max_length=128) for p in test_proteins]
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        batch = collator(tokenized)
        return {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def test_golden_values(self):
        """HF and TE models should produce matching outputs."""
        from transformers.models.esm.modeling_esm import EsmForMaskedLM

        model_hf = EsmForMaskedLM.from_pretrained(self.upstream_model_id, dtype=torch.bfloat16).cuda()
        model_te = convert_esm_hf_to_te(model_hf).cuda()
        input_data = self.get_test_input_data()

        with torch.no_grad():
            hf_out = model_hf(**input_data)
            te_out = model_te(**input_data)

        # NOTE: These tolerances are model-specific. ESM2 needs slightly higher due to
        # numerical differences in TE's fused attention vs HF's unfused attention.
        torch.testing.assert_close(te_out.loss, hf_out.loss, atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(te_out.logits, hf_out.logits, atol=2.0, rtol=1e-4)

    def test_roundtrip_conversion(self):
        """HF->TE->HF should preserve all weights."""
        from transformers.models.esm.modeling_esm import EsmForMaskedLM

        model_hf = EsmForMaskedLM.from_pretrained(self.upstream_model_id).cuda()
        model_te = convert_esm_hf_to_te(model_hf)
        model_hf_back = convert_esm_te_to_hf(model_te)

        for (n1, p1), (n2, p2) in zip(model_hf.named_parameters(), model_hf_back.named_parameters()):
            torch.testing.assert_close(p1, p2, msg=f"Roundtrip mismatch: {n1}")

    def test_forward_backward(self):
        """Smoke test: forward + backward pass should work."""
        from transformers.models.esm.modeling_esm import EsmForMaskedLM

        model_hf = EsmForMaskedLM.from_pretrained(self.upstream_model_id, dtype=torch.bfloat16).cuda()
        model_te = convert_esm_hf_to_te(model_hf).cuda()
        input_data = self.get_test_input_data()

        output = model_te(**input_data)
        output.loss.backward()

        for name, param in model_te.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
