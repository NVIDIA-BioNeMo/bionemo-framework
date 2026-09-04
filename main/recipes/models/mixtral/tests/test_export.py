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

import os

import pytest
import torch
from transformer_engine.pytorch import MultiheadAttention
from transformers import AutoModelForCausalLM, AutoTokenizer, MixtralConfig, MixtralForCausalLM

import export
from convert import convert_mixtral_hf_to_te, convert_mixtral_te_to_hf
from export import export_hf_checkpoint, export_hf_state_dict
from modeling_mixtral_te import NVMixtralForCausalLM, _ensure_fused_grouped_mlp_registered


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# Fused GroupedMLP and module GroupedLinear run different BF16 kernels/accumulation paths. Keep the
# parity tolerance explicit and pair each use with a same-tolerance negative control below so this
# cannot silently become a non-discriminating comparison.
LOGIT_PARITY_ATOL = 2e-2
LOGIT_PARITY_RTOL = 2e-2


def _fused_grouped_mlp_available() -> bool:
    if not torch.cuda.is_available():
        return False
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
    return _ensure_fused_grouped_mlp_registered()


def _tiny_mixtral_config(**overrides) -> MixtralConfig:
    defaults = {
        "vocab_size": 256,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 512,
        "router_jitter_noise": 0.0,
        "attn_implementation": "flash_attention_2",
        "torch_dtype": torch.bfloat16,
    }
    defaults.update(overrides)
    return MixtralConfig(**defaults)


def _make_tiny_hf_model(**config_overrides) -> MixtralForCausalLM:
    torch.manual_seed(42)
    config = _tiny_mixtral_config(**config_overrides)
    model = MixtralForCausalLM(config)
    model.to(torch.bfloat16)
    return model


def _make_tiny_inputs(batch_size: int = 2, seq_len: int = 256) -> dict:
    torch.manual_seed(0)
    input_ids = torch.randint(0, 256, (batch_size, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _assert_same_tolerance_rejects_shifted_logits(reference: torch.Tensor, actual: torch.Tensor) -> None:
    """Guard against over-loose tolerances by checking a token-logit shuffle fails."""
    assert actual.shape[-1] > 1, "negative control needs at least two vocabulary logits"
    shifted = torch.roll(actual, shifts=1, dims=-1)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(reference, shifted, atol=LOGIT_PARITY_ATOL, rtol=LOGIT_PARITY_RTOL)


@pytest.mark.skipif(os.getenv("CI", "false") == "true", reason="Skipping test in CI, requires Mini-Mixtral download.")
def test_export_mixtral_checkpoint(tmp_path):
    export_hf_checkpoint("NeuralNovel/Mini-Mixtral-v0.2", tmp_path / "checkpoint_export")

    _ = AutoTokenizer.from_pretrained(tmp_path / "checkpoint_export")
    model = AutoModelForCausalLM.from_pretrained(tmp_path / "checkpoint_export", trust_remote_code=True)
    assert "NVMixtralForCausalLM" in model.__class__.__name__
    assert "NVMixtralConfig" in model.config.__class__.__name__
    # Mixtral uses custom NVMixtralDecoderLayer with TE MultiheadAttention sub-modules
    assert isinstance(model.model.layers[0].self_attention, MultiheadAttention)


def test_export_hf_state_dict_writes_mmap_loadable_file(tmp_path, monkeypatch):
    model = torch.nn.Linear(4, 3, bias=False)
    expected = model.state_dict()["weight"].clone()
    monkeypatch.setattr(export, "_convert_hf_checkpoint", lambda *args, **kwargs: model)
    output_path = tmp_path / "converted.pt"

    export_hf_state_dict("unused", output_path, expert_ffn_mode="fused_grouped_mlp")

    state_dict = torch.load(output_path, map_location="cpu", weights_only=True, mmap=True)
    torch.testing.assert_close(state_dict["weight"], expected)


@requires_cuda
@pytest.mark.skipif(
    not _fused_grouped_mlp_available(),
    reason="fused CuteDSL kernel unavailable (needs sm_100 + cutlass-dsl 4.4.1 pin)",
)
def test_fused_grouped_mlp_hf_te_hf_roundtrip():
    """HF→TE→HF round-trip for fused_grouped_mlp preserves weights and logits."""
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    model_hf = _make_tiny_hf_model()
    original_state = {k: v.clone() for k, v in model_hf.state_dict().items()}

    te_kwargs = {
        "expert_ffn_mode": "fused_grouped_mlp",
        "attn_input_format": "bshd",
        "self_attn_mask_type": "causal",
    }
    model_te = convert_mixtral_hf_to_te(model_hf, **te_kwargs)
    assert isinstance(model_te, NVMixtralForCausalLM)
    model_te.to("cuda").eval()

    model_hf_roundtrip = convert_mixtral_te_to_hf(model_te)
    roundtrip_state = model_hf_roundtrip.state_dict()

    assert set(original_state.keys()) == set(roundtrip_state.keys())
    for key, original_param in original_state.items():
        converted = roundtrip_state[key].to(device="cpu", dtype=original_param.dtype)
        torch.testing.assert_close(
            original_param.cpu(),
            converted,
            atol=1e-5,
            rtol=1e-5,
            msg=lambda msg: f"Round-trip weight mismatch for {key}: {msg}",
        )

    inputs = _make_tiny_inputs()
    model_hf.cuda().eval()
    model_hf_rt = MixtralForCausalLM(model_hf.config).to(torch.bfloat16)
    model_hf_rt.load_state_dict({k: v.detach().cpu() for k, v in roundtrip_state.items()})
    model_hf_rt.to("cuda").eval()
    with torch.no_grad():
        ref_logits = model_hf(**inputs).logits
        rt_logits = model_hf_rt(**inputs).logits

    torch.testing.assert_close(ref_logits, rt_logits, atol=LOGIT_PARITY_ATOL, rtol=LOGIT_PARITY_RTOL)
    _assert_same_tolerance_rejects_shifted_logits(ref_logits, rt_logits)


@requires_cuda
def test_fused_grouped_mlp_save_pretrained_roundtrip(tmp_path):
    """Fused-mode save_pretrained/from_pretrained round-trips (shared-tensor dedup in state_dict).

    In ``fused_grouped_mlp`` mode the expert weights are aliased under both ``experts_gate_up`` /
    ``experts_down`` and the ``_experts_ffn_op`` fused ``Sequential``. Without deduping those
    aliases, ``save_pretrained``'s safetensors writer rejects the checkpoint as containing shared
    tensors. This does not require the CuteDSL kernel (no forward), only module construction + save.
    """
    import json
    import shutil

    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    from modeling_mixtral_te import AUTO_MAP

    model_hf = _make_tiny_hf_model()
    model_te = convert_mixtral_hf_to_te(
        model_hf,
        expert_ffn_mode="fused_grouped_mlp",
        attn_input_format="bshd",
        self_attn_mask_type="causal",
    ).cuda()

    state_dict = model_te.state_dict()
    assert not [k for k in state_dict if "._experts_ffn_op." in k], "duplicate fused-op keys leaked into state_dict"

    export_path = tmp_path / "checkpoint_export"
    model_te.save_pretrained(export_path)  # regression: previously raised on shared tensors

    with open(export_path / "config.json") as f:
        config = json.load(f)
    config["auto_map"] = AUTO_MAP
    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "..", "modeling_mixtral_te.py"), export_path / "modeling_mixtral_te.py"
    )

    reloaded = AutoModelForCausalLM.from_pretrained(export_path, trust_remote_code=True).cuda()
    reloaded_state = reloaded.state_dict()
    for key, value in state_dict.items():
        if ".experts_gate_up.weight" in key or ".experts_down.weight" in key or key == "lm_head.weight":
            torch.testing.assert_close(value, reloaded_state[key], atol=0.0, rtol=0.0, msg=key)


@requires_cuda
@pytest.mark.skipif(
    not _fused_grouped_mlp_available(),
    reason="fused CuteDSL kernel unavailable (needs sm_100 + cutlass-dsl 4.4.1 pin)",
)
def test_fused_vs_grouped_linear_logit_parity():
    """Same HF weights loaded via conversion into both modes produce matching logits."""
    os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "0"
    os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    model_hf = _make_tiny_hf_model()
    inputs = _make_tiny_inputs()

    common_te_kwargs = {
        "attn_input_format": "bshd",
        "self_attn_mask_type": "causal",
    }

    model_te_baseline = convert_mixtral_hf_to_te(model_hf, expert_ffn_mode="grouped_linear", **common_te_kwargs)
    model_te_baseline.to("cuda").eval()

    model_te_fused = convert_mixtral_hf_to_te(model_hf, expert_ffn_mode="fused_grouped_mlp", **common_te_kwargs)
    model_te_fused.to("cuda").eval()

    with torch.no_grad():
        logits_baseline = model_te_baseline(**inputs).logits
        logits_fused = model_te_fused(**inputs).logits

    max_diff = (logits_baseline - logits_fused).abs().max().item()
    print(f"max logit diff (grouped_linear vs fused_grouped_mlp): {max_diff}")

    torch.testing.assert_close(logits_baseline, logits_fused, atol=LOGIT_PARITY_ATOL, rtol=LOGIT_PARITY_RTOL)
    _assert_same_tolerance_rejects_shifted_logits(logits_baseline, logits_fused)


@requires_cuda
def test_grouped_linear_hf_te_hf_roundtrip_unchanged():
    """Default grouped_linear conversion round-trip still works (float32, no cast_dtype)."""
    torch.manual_seed(42)
    config = _tiny_mixtral_config(torch_dtype=torch.float32)
    model_hf = MixtralForCausalLM(config)
    original_state = {k: v.clone() for k, v in model_hf.state_dict().items()}

    te_kwargs = {
        "expert_ffn_mode": "grouped_linear",
        "attn_input_format": "bshd",
        "self_attn_mask_type": "causal",
    }
    model_te = convert_mixtral_hf_to_te(model_hf, **te_kwargs)

    model_hf_roundtrip = convert_mixtral_te_to_hf(model_te)
    roundtrip_state = model_hf_roundtrip.state_dict()

    for key, original_param in original_state.items():
        converted = roundtrip_state[key].to(device="cpu", dtype=original_param.dtype)
        torch.testing.assert_close(original_param.cpu(), converted, atol=1e-5, rtol=1e-5, msg=key)
