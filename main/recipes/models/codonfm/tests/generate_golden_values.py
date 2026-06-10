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

"""Generate golden values for CodonFM model regression testing.

This script creates a CodonFM model using the codonfm_ptl_te non-exact (standard
TETransformerLayer) implementation, runs a forward pass, and saves the outputs
as golden values. It also maps the state dict to the models/codonfm native_te
key format and verifies cross-model equivalence.

Usage (from the repo root):
    python bionemo-recipes/models/codonfm/tests/generate_golden_values.py
"""

import json
import os
import re
import sys
from pathlib import Path

import torch


# ── Path setup ──────────────────────────────────────────────────────────────────
# Import both the ptl_te recipe model and the native_te model.
REPO_ROOT = Path(__file__).resolve().parents[4]

# Force non-exact (standard TETransformerLayer) mode for ptl_te.
os.environ["CODON_FM_TE_IMPL"] = "nonexact"

PTL_TE_DIR = REPO_ROOT / "bionemo-recipes" / "recipes" / "codonfm_ptl_te"
NATIVE_TE_DIR = REPO_ROOT / "bionemo-recipes" / "models" / "codonfm"
TEST_DIR = Path(__file__).parent

sys.path.insert(0, str(PTL_TE_DIR))
sys.path.insert(0, str(PTL_TE_DIR / "src"))
sys.path.insert(0, str(NATIVE_TE_DIR))

# ── Constants ───────────────────────────────────────────────────────────────────
SEED = 42

# Use a small config that matches encodon_200k preset.
SMALL_CONFIG = {
    "hidden_size": 128,
    "intermediate_size": 512,
    "num_attention_heads": 4,
    "num_hidden_layers": 2,
    "vocab_size": 69,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,  # Disable dropout for deterministic output.
    "attention_probs_dropout_prob": 0.0,
    "layer_norm_eps": 1e-12,
    "pad_token_id": 3,
    "mask_token_id": 4,
    "max_position_embeddings": 2048,
}

TEST_CODON_SEQUENCES = [
    "ATGCGTAAAGCTGTTCAGGATCTGAATGCCATCTATGCG",
    "ATGGATCGTACCGCTGAACAGCGTCTGATCAAAGCC",
    "ATGGCTACCGATCGTGAACTGGCTCAGGATAAAGCTACC",
    "ATGCGTGATCTGACCGAAGCTCAGAAAGTTGATCGTACC",
    "ATGACCGATGCTCGTAAAGCTCTGGAACAGATCGATGCT",
]


# ── Key mapping: ptl_te non-exact → native_te ──────────────────────────────────
def map_ptl_te_to_native_te(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map state dict keys from ptl_te non-exact to native_te format.

    ptl_te (non-exact)          →  native_te (models/codonfm)
    embeddings.*                →  embeddings.*              (unchanged)
    layers.{i}.*                →  encoder.layers.{i}.*
    cls.0.{weight,bias}         →  lm_head.dense.{weight,bias}
    cls.2.{weight,bias}         →  lm_head.layer_norm_linear.{weight,bias}
    cls.2.layer_norm_{w,b}      →  lm_head.layer_norm_linear.layer_norm_{w,b}
    """
    mapped = {}
    for key, value in state_dict.items():
        # Skip TE internal bookkeeping.
        if "_extra_state" in key or "inv_freq" in key:
            continue

        new_key = key
        # layers.{i}.* → encoder.layers.{i}.*
        if re.match(r"layers\.\d+\.", key):
            new_key = "encoder." + key
        # cls.0.* → lm_head.dense.*
        elif key.startswith("cls.0."):
            new_key = key.replace("cls.0.", "lm_head.dense.")
        # cls.2.* → lm_head.layer_norm_linear.*
        elif key.startswith("cls.2."):
            new_key = key.replace("cls.2.", "lm_head.layer_norm_linear.")

        mapped[new_key] = value
    return mapped


def prepare_test_inputs(tokenizer):
    """Tokenize and prepare test input data in BSHD format."""
    encoded = [tokenizer.encode(seq) for seq in TEST_CODON_SEQUENCES]
    max_len = max(len(e) for e in encoded)

    input_ids, attention_mask, labels = [], [], []
    for enc in encoded:
        pad_len = max_len - len(enc)
        ids = enc + [tokenizer.pad_token_id] * pad_len
        mask = [1] * len(enc) + [0] * pad_len
        lbl = [-100] * max_len
        torch.manual_seed(SEED)
        for i in range(1, len(enc) - 1):
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


def generate():
    """Generate golden values from ptl_te non-exact model and verify cross-model equivalence."""
    from modeling_codonfm_te import CodonFMConfig, CodonFMForMaskedLM
    from models.components.encodon_config import EnCodonConfig
    from models.components.encodon_te import EnCodonTE
    from tokenizer import CodonTokenizer

    # ── 1. Create ptl_te non-exact model ────────────────────────────────────
    print("Creating ptl_te non-exact model...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # EnCodonConfig has a slightly different param set; filter to what it accepts.
    ptl_params = {
        k: v
        for k, v in SMALL_CONFIG.items()
        if k
        in (
            "vocab_size",
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "hidden_act",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "layer_norm_eps",
            "pad_token_id",
            "max_position_embeddings",
        )
    }
    ptl_config = EnCodonConfig(**ptl_params)
    ptl_model = EnCodonTE(ptl_config).cuda().to(torch.bfloat16)
    ptl_model.eval()

    # ── 2. Prepare inputs ───────────────────────────────────────────────────
    tokenizer = CodonTokenizer()
    inputs = prepare_test_inputs(tokenizer)

    # ── 3. Run ptl_te forward ───────────────────────────────────────────────
    print("Running ptl_te non-exact forward pass...")
    with torch.no_grad():
        ptl_out = ptl_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    ptl_logits = ptl_out.logits.float().cpu()

    # Compute loss the same way as our model.
    loss_fct = torch.nn.CrossEntropyLoss()
    ptl_loss = (
        loss_fct(
            ptl_logits.to("cuda").view(-1, SMALL_CONFIG["vocab_size"]),
            inputs["labels"].view(-1),
        )
        .float()
        .cpu()
    )

    print(f"  ptl_te loss: {ptl_loss.item():.6f}")
    print(f"  ptl_te logits shape: {list(ptl_logits.shape)}")

    # ── 4. Map state dict and verify with native_te ─────────────────────────
    print("Mapping state dict ptl_te → native_te...")
    ptl_sd = ptl_model.state_dict()
    native_sd = map_ptl_te_to_native_te(ptl_sd)

    native_config = CodonFMConfig(**SMALL_CONFIG)
    native_model = CodonFMForMaskedLM(native_config).cuda().to(torch.bfloat16)
    # Load the mapped weights.
    missing, unexpected = native_model.load_state_dict(native_sd, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    # Only rotary embeddings inv_freq should be missing (generated on the fly).
    unexpected_missing = {k for k in missing if not k.endswith(".inv_freq")}
    assert not unexpected_missing, f"Unexpected missing keys: {unexpected_missing}"
    if missing:
        print(f"  Missing keys (expected inv_freq): {missing}")

    native_model.eval()
    with torch.no_grad():
        native_out = native_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
    native_logits = native_out.logits.float().cpu()
    native_loss = native_out.loss.float().cpu()
    print(f"  native_te loss: {native_loss.item():.6f}")

    # ── 5. Cross-model equivalence check ────────────────────────────────────
    mask = inputs["attention_mask"].bool().cpu()
    torch.testing.assert_close(
        native_logits[mask],
        ptl_logits[mask],
        atol=1e-3,
        rtol=1e-3,
        msg=lambda x: f"Cross-model logits mismatch: {x}",
    )
    print("Cross-model equivalence verified (logits match).")

    # ── 6. Save golden values ───────────────────────────────────────────────
    from safetensors.torch import save_file

    # Save the mapped state dict (native_te key format) as safetensors.
    sd_path = TEST_DIR / "golden_state_dict.safetensors"
    save_file(native_sd, sd_path)
    print(f"Saved state dict to {sd_path} ({sd_path.stat().st_size / 1024:.0f} KB)")

    # Save outputs + inputs as JSON.
    golden = {
        "seed": SEED,
        "config": SMALL_CONFIG,
        "loss": native_loss.item(),
        "logits_shape": list(native_logits.shape),
        "logits": native_logits.tolist(),
        "input_ids": inputs["input_ids"].cpu().tolist(),
        "attention_mask": inputs["attention_mask"].cpu().tolist(),
        "labels": inputs["labels"].cpu().tolist(),
    }
    json_path = TEST_DIR / "golden_values.json"
    with open(json_path, "w") as f:
        json.dump(golden, f)
    print(f"Saved golden values to {json_path} ({json_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    generate()
