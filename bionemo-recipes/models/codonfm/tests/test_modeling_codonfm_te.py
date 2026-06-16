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

"""Tests for CodonFM model using the common test library.

This file provides comprehensive tests for the CodonFM model including:
- Meta device initialization tests
- FP8 tests
- Forward/backward pass smoke tests
- Golden value regression tests (generated from codonfm_ptl_te non-exact model)
- Model-specific tests

Golden values are generated from the codonfm_ptl_te recipe's non-exact
(standard TETransformerLayer) implementation, which is architecturally
identical to our native_te model. The generation script cross-validates
that both implementations produce identical logits given the same weights.
See generate_golden_values.py for details.

Conversion tests are skipped: CodonFM is natively TE (no HF variant exists).
"""

import gc
import json
from pathlib import Path
from typing import Callable, Dict, List, Literal, Type

import pytest
import torch
import transformer_engine.pytorch as te
from modeling_codonfm_te import MODEL_PRESETS, CodonFMConfig, CodonFMForMaskedLM
from tests.common import BaseModelTest, TestTolerances
from tokenizer import CodonTokenizer
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer


# Sample codon sequences for testing (each is a DNA string of 3-mer codons).
TEST_CODON_SEQUENCES = [
    "ATGCGTAAAGCTGTTCAGGATCTGAATGCCATCTATGCG",
    "ATGGATCGTACCGCTGAACAGCGTCTGATCAAAGCC",
    "ATGGCTACCGATCGTGAACTGGCTCAGGATAAAGCTACC",
    "ATGCGTGATCTGACCGAAGCTCAGAAAGTTGATCGTACC",
    "ATGACCGATGCTCGTAAAGCTCTGGAACAGATCGATGCT",
]


class TestCodonFMModel(BaseModelTest):
    """Model tester for CodonFM.

    This class provides CodonFM-specific configuration for the common test suite.
    Most tests are inherited from BaseModelTest. Conversion and golden value tests
    are skipped because there is no compatible upstream HF model.
    """

    def get_model_class(self) -> Type[PreTrainedModel]:
        """Return the CodonFM TE model class."""
        return CodonFMForMaskedLM

    def get_config_class(self) -> Type[PretrainedConfig]:
        """Return the CodonFM config class."""
        return CodonFMConfig

    def get_upstream_model_id(self) -> str:
        """Return upstream model ID (placeholder — no compatible upstream model exists yet)."""
        # TODO: Upload a native_te checkpoint to HF Hub and update this.
        return "nvidia/NV-CodonFM-Encodon-TE-80M-v1"

    def get_upstream_model_revision(self) -> str:
        """Return upstream model revision."""
        return "main"

    def get_upstream_model_class(self) -> Type[PreTrainedModel]:
        """Return upstream model class (self-reference since CodonFM is natively TE)."""
        return CodonFMForMaskedLM

    def get_layer_path(self, model: PreTrainedModel) -> List[nn.Module]:
        """Return the list of transformer layers."""
        return list(model.encoder.layers)

    def get_tokenizer(self) -> PreTrainedTokenizer:
        """Return tokenizer. CodonFM uses a plain Python tokenizer, not a HF PreTrainedTokenizer."""
        return None  # type: ignore

    def get_hf_to_te_converter(self) -> Callable:
        """Return identity converter (CodonFM is natively TE, no conversion needed)."""
        return lambda model, **kwargs: model

    def get_te_to_hf_converter(self) -> Callable:
        """Return identity converter (CodonFM is natively TE, no conversion needed)."""
        return lambda model, **kwargs: model

    def create_inference_params(self, config, batch_size=1, max_seq_len=256, num_beams=1):
        """Not applicable for non-autoregressive models."""
        pass

    def create_test_config(self, **kwargs) -> CodonFMConfig:
        """Create a test configuration using the encodon_200k preset.

        Args:
            **kwargs: Configuration parameters to override.

        Returns:
            CodonFMConfig instance.
        """
        preset = MODEL_PRESETS["encodon_200k"].copy()
        preset.update(kwargs)
        return CodonFMConfig(**preset)

    def get_test_input_data(
        self,
        format: Literal["bshd", "thd"] = "bshd",
        pad_to_multiple_of: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Prepare test input data (codon sequences).

        Args:
            format: Whether to use sequence packing (THD) or BSHD format.
            pad_to_multiple_of: Pad sequence length to a multiple of this value.

        Returns:
            Dictionary of input tensors.
        """
        tokenizer = CodonTokenizer()

        # Tokenize sequences
        encoded = [tokenizer.encode(seq) for seq in TEST_CODON_SEQUENCES]

        if format == "bshd":
            # Pad to same length
            max_len = max(len(e) for e in encoded)
            if pad_to_multiple_of:
                max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

            input_ids = []
            attention_mask = []
            labels = []
            for enc in encoded:
                pad_len = max_len - len(enc)
                ids = enc + [tokenizer.pad_token_id] * pad_len
                mask = [1] * len(enc) + [0] * pad_len

                # Create MLM labels: mask ~15% of non-special tokens
                lbl = [-100] * max_len
                torch.manual_seed(42)
                for i in range(1, len(enc) - 1):  # Skip CLS and SEP
                    if torch.rand(1).item() < 0.15:
                        lbl[i] = ids[i]
                        ids[i] = tokenizer.mask_token_id

                input_ids.append(ids)
                attention_mask.append(mask)
                labels.append(lbl)

            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cuda"),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device="cuda"),
                "labels": torch.tensor(labels, dtype=torch.long, device="cuda"),
            }
        else:
            # THD format: flatten all sequences with cu_seqlens
            all_ids = []
            all_labels = []
            cu_seqlens = [0]
            for enc in encoded:
                ids = list(enc)
                lbl = [-100] * len(enc)
                torch.manual_seed(42)
                for i in range(1, len(enc) - 1):
                    if torch.rand(1).item() < 0.15:
                        lbl[i] = ids[i]
                        ids[i] = tokenizer.mask_token_id
                all_ids.extend(ids)
                all_labels.extend(lbl)
                cu_seqlens.append(len(all_ids))

            cu_seqlens_t = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
            max_seqlen = max(cu_seqlens_t[1:] - cu_seqlens_t[:-1]).item()

            return {
                "input_ids": torch.tensor([all_ids], dtype=torch.long, device="cuda"),
                "labels": torch.tensor([all_labels], dtype=torch.long, device="cuda"),
                "cu_seq_lens_q": cu_seqlens_t,
                "cu_seq_lens_k": cu_seqlens_t,
                "max_length_q": max_seqlen,
                "max_length_k": max_seqlen,
            }

    def get_tolerances(self) -> TestTolerances:
        """Return CodonFM-specific test tolerances."""
        return TestTolerances(
            # MAGNETO init has different std than initializer_range, so relax init tolerances.
            init_std_atol=0.1,
            init_std_rtol=0.5,
        )

    def verify_model_parameters_initialized_correctly(
        self,
        model: PreTrainedModel,
        atol: float | None = None,
        rtol: float | None = None,
        should_be_fp8: bool = False,
    ) -> None:
        """Verify model parameters are initialized correctly.

        CodonFM uses MAGNETO initialization (xavier_normal with scaled gain),
        so the standard init check from BaseModelTest doesn't apply.

        Args:
            model: The model to verify.
            atol: Absolute tolerance for comparisons.
            rtol: Relative tolerance for comparisons.
            should_be_fp8: Whether to expect FP8 quantized weights.
        """
        import fnmatch

        from transformer_engine.pytorch import QuantizedTensor

        # Verify all parameters are on CUDA
        for name, parameter in model.named_parameters():
            assert str(parameter.device).startswith("cuda"), f"Parameter {name} is not on the cuda device"

        # Verify embeddings have correct std
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                torch.testing.assert_close(
                    module.weight.mean().item(),
                    0.0,
                    atol=0.01,
                    rtol=0.01,
                    msg=lambda x: f"Embedding {name} mean mismatch: {x}",
                )
                torch.testing.assert_close(
                    module.weight.std().item(),
                    model.config.initializer_range,
                    atol=0.01,
                    rtol=0.1,
                    msg=lambda x: f"Embedding {name} std mismatch: {x}",
                )

            # Verify FP8 quantization if requested
            if should_be_fp8 and isinstance(module, te.Linear):
                if f"{name}.weight" in set(model._tied_weights_keys or []):
                    continue
                if hasattr(model, "_do_not_quantize") and any(
                    fnmatch.fnmatch(name, pattern) for pattern in model._do_not_quantize
                ):
                    continue
                assert isinstance(module.weight, QuantizedTensor), f"Module {name} weight is not a QuantizedTensor"

    # ==================== Golden Value Tests ====================
    # Golden values are generated from the codonfm_ptl_te non-exact (standard
    # TETransformerLayer) implementation. The generate_golden_values.py script
    # cross-validates that both ptl_te and native_te produce identical logits.

    def test_golden_values(self):
        """Test golden values using pre-generated reference from codonfm_ptl_te non-exact model."""
        from safetensors.torch import load_file

        golden_dir = Path(__file__).parent
        golden_json = golden_dir / "golden_values.json"
        golden_sd_path = golden_dir / "golden_state_dict.safetensors"

        if not golden_json.exists() or not golden_sd_path.exists():
            pytest.skip("Golden values not generated. Run generate_golden_values.py first.")

        with open(golden_json) as f:
            golden = json.load(f)

        # Load the state dict generated from ptl_te non-exact model.
        golden_sd = load_file(golden_sd_path, device="cpu")

        # Create model with the same config used during generation.
        config = CodonFMConfig(**golden["config"])
        model = CodonFMForMaskedLM(config).cuda().to(torch.bfloat16)
        model.load_state_dict(golden_sd, strict=False)
        model.eval()

        # Prepare inputs from golden values.
        input_ids = torch.tensor(golden["input_ids"], dtype=torch.long, device="cuda")
        attention_mask = torch.tensor(golden["attention_mask"], dtype=torch.long, device="cuda")
        labels = torch.tensor(golden["labels"], dtype=torch.long, device="cuda")

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        expected_loss = torch.tensor(golden["loss"], dtype=torch.float32)
        expected_logits = torch.tensor(golden["logits"], dtype=torch.float32)
        tolerances = self.get_tolerances()

        torch.testing.assert_close(
            outputs.loss.float().cpu(),
            expected_loss,
            atol=tolerances.golden_value_loss_atol,
            rtol=tolerances.golden_value_loss_rtol,
            msg=lambda x: f"Golden value loss mismatch: {x}",
        )

        # Compare logits only at non-padded positions.
        mask = attention_mask.bool().cpu()
        torch.testing.assert_close(
            outputs.logits.float().cpu()[mask],
            expected_logits[mask],
            atol=tolerances.golden_value_logits_atol,
            rtol=tolerances.golden_value_logits_rtol,
            msg=lambda x: f"Golden value logits mismatch: {x}",
        )

    def test_golden_values_thd(self, te_attn_backend):
        """Test BSHD vs THD format equivalence with the same weights.

        Unlike the base class version (which requires an upstream model), this
        creates a model from golden values and verifies that BSHD and THD
        attention formats produce equivalent outputs.
        """
        from safetensors.torch import load_file

        if te_attn_backend == "fused_attn" and torch.cuda.get_device_capability()[0] == 8:
            pytest.xfail("On Ada and Ampere, no THD implementation is available for fused attn.")

        golden_dir = Path(__file__).parent
        golden_sd_path = golden_dir / "golden_state_dict.safetensors"
        golden_json = golden_dir / "golden_values.json"

        if not golden_sd_path.exists() or not golden_json.exists():
            pytest.skip("Golden values not generated. Run generate_golden_values.py first.")

        with open(golden_json) as f:
            golden = json.load(f)

        golden_sd = load_file(golden_sd_path, device="cpu")

        input_data_bshd = self.get_test_input_data(format="bshd")
        input_data_thd = self.get_test_input_data(format="thd")
        tolerances = self.get_tolerances()

        # Run BSHD model.
        config_bshd = CodonFMConfig(**golden["config"], attn_input_format="bshd")
        model_bshd = CodonFMForMaskedLM(config_bshd).cuda().to(torch.bfloat16)
        model_bshd.load_state_dict(golden_sd, strict=False)
        model_bshd.eval()
        with torch.no_grad():
            outputs_bshd = model_bshd(**input_data_bshd)
        bshd_loss = outputs_bshd.loss.detach().clone()
        bshd_logits = outputs_bshd.logits[input_data_bshd["attention_mask"].bool()].detach().clone()
        del model_bshd, outputs_bshd
        gc.collect()
        torch.cuda.empty_cache()

        # Run THD model with same weights.
        config_thd = CodonFMConfig(**golden["config"], attn_input_format="thd")
        model_thd = CodonFMForMaskedLM(config_thd).cuda().to(torch.bfloat16)
        model_thd.load_state_dict(golden_sd, strict=False)
        model_thd.eval()
        with torch.no_grad():
            outputs_thd = model_thd(**input_data_thd)

        torch.testing.assert_close(
            bshd_logits,
            outputs_thd.logits,
            atol=tolerances.golden_value_logits_atol,
            rtol=tolerances.golden_value_logits_rtol,
            msg=lambda x: f"BSHD vs THD logits mismatch: {x}",
        )
        torch.testing.assert_close(
            bshd_loss,
            outputs_thd.loss,
            atol=tolerances.golden_value_loss_atol,
            rtol=tolerances.golden_value_loss_rtol,
            msg=lambda x: f"BSHD vs THD loss mismatch: {x}",
        )

    # ==================== Skipped Tests ====================
    # Conversion tests: CodonFM is natively TE — no HF↔TE conversion needed.
    # THD padding tests: Not yet implemented for CodonFM's tokenizer.

    @pytest.mark.skip(reason="CodonFM is natively TE; no HF variant exists for conversion")
    def test_convert_hf_to_te(self):
        pass

    @pytest.mark.skip(reason="CodonFM is natively TE; no HF variant exists for conversion")
    def test_convert_te_to_hf(self):
        pass

    @pytest.mark.skip(reason="CodonFM is natively TE; no HF variant exists for conversion")
    def test_convert_te_to_hf_roundtrip(self):
        pass

    @pytest.mark.skip(reason="CodonFM is natively TE; no HF variant exists for conversion")
    def test_convert_config(self):
        pass

    @pytest.mark.skip(reason="THD padding not yet implemented for CodonFM tokenizer")
    def test_golden_values_thd_padded(self):
        pass

    @pytest.mark.skip(reason="THD padding not yet implemented for CodonFM tokenizer")
    def test_thd_padding_input_data_equivalence(self):
        pass
