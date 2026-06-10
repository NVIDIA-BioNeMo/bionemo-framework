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

"""Reference: ESM2 export script for HuggingFace Hub.

Shows the complete export pipeline:
1. Load HF model -> convert to TE
2. Save model + tokenizer
3. Patch config with AUTO_MAP
4. Copy model code for trust_remote_code
5. Smoke test loading
"""

import gc
import json
import shutil
from pathlib import Path

import torch
from convert import convert_esm_hf_to_te
from modeling_esm_te import AUTO_MAP
from transformers import AutoModelForMaskedLM, AutoTokenizer


def export_hf_checkpoint(tag: str, export_path: Path):
    """Export a HuggingFace checkpoint to TE format for Hub distribution.

    Args:
        tag: HuggingFace model tag (e.g., "esm2_t6_8M_UR50D")
        export_path: Directory to save exported model
    """
    # NOTE: Load and convert
    model_hf = AutoModelForMaskedLM.from_pretrained(f"facebook/{tag}")
    model_te = convert_esm_hf_to_te(model_hf)
    model_te.save_pretrained(export_path / tag)

    # NOTE: Save tokenizer alongside model
    tokenizer = AutoTokenizer.from_pretrained("esm_fast_tokenizer")
    tokenizer.save_pretrained(export_path / tag)

    # NOTE: Patch config.json with AUTO_MAP for trust_remote_code loading
    with open(export_path / tag / "config.json", "r") as f:
        config = json.load(f)
    config["auto_map"] = AUTO_MAP
    with open(export_path / tag / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    # NOTE: Copy modeling file as the remote code file
    # The AUTO_MAP references "esm_nv.NVEsmForMaskedLM" so the file must be named esm_nv.py
    shutil.copy("modeling_esm_te.py", export_path / tag / "esm_nv.py")

    # NOTE: Copy license
    shutil.copy("LICENSE", export_path / tag / "LICENSE")

    # Clean up to free memory
    del model_hf, model_te
    gc.collect()
    torch.cuda.empty_cache()

    # NOTE: Smoke test - verify the exported model loads correctly
    model_te = AutoModelForMaskedLM.from_pretrained(
        export_path / tag,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    del model_te
    gc.collect()
    torch.cuda.empty_cache()
