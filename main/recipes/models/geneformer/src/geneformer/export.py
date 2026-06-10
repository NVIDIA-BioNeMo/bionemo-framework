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

import gc
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForMaskedLM

from geneformer import BertForMaskedLM
from geneformer.convert import convert_geneformer_hf_to_te, convert_geneformer_te_to_hf


def export_hf_checkpoint(model_name: str, export_path: Path):
    """Export a Geneformer checkpoint to a standardized format.

    Args:
        model_name: The name of the Geneformer model variant (e.g., "Geneformer-V2-316M", "Geneformer-V1-10M").
        export_path: The parent path to export the checkpoint to.
    """
    print(f"Loading Geneformer model: {model_name}")

    model_hf = AutoModelForMaskedLM.from_pretrained("ctheodoris/Geneformer", subfolder=model_name, revision="f45a6c7d")

    print(f"Loaded HF model with {len(list(model_hf.parameters()))} parameters")

    # TODO:tokenizer file ?

    model_te = convert_geneformer_hf_to_te(model_hf)

    export_path.mkdir(parents=True, exist_ok=True)

    # Filter out _extra_state parameters before saving
    # https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/faq.html#fp8-checkpoint-compatibility
    state_dict = model_te.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.endswith("_extra_state")}

    # Save the filtered state dict in pytorch_model.bin & model.safetensors
    torch.save(filtered_state_dict, export_path / "pytorch_model.bin")
    # Handle shared tensors by creating a copy of the filtered state dict
    # Ensures no memory sharing issues when saving to safetensors
    safetensors_state_dict = {}
    for k, v in filtered_state_dict.items():
        safetensors_state_dict[k] = v.clone().detach()
    save_file(safetensors_state_dict, export_path / "model.safetensors")

    model_te.config.save_pretrained(export_path)

    print(f"TE model saved to {export_path}")

    # Patch the config to include Geneformer-specific metadata
    config_path = export_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

        config["model_type"] = "bert"
        config["framework"] = "pytorch"
        # Add TE-specific auto_map
        config["auto_map"] = {
            "AutoConfig": "geneformer.TEBertConfig",
            "AutoModel": "geneformer.BertModel",
            "AutoModelForMaskedLM": "geneformer.BertForMaskedLM",
        }

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        print("Config updated with Geneformer metadata")

    # Copy the custom model file
    model_file_source = Path(__file__).parent / "modeling_bert_te.py"
    model_file_dest = export_path / "geneformer.py"
    shutil.copy(model_file_source, model_file_dest)
    print(f"Copied {model_file_source} to {model_file_dest}")

    shutil.copy("README.md", export_path / "README.md")
    shutil.copy("LICENSE", export_path / "LICENSE")
    shutil.copy("model_readme.md", export_path / "model_readme.md")
    print("Copied README.md, LICENSE, and model_readme.md")

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()

    # Smoke test that the TE model can be loaded
    print("Testing TE model loading...")
    test_model = AutoModelForMaskedLM.from_pretrained(
        export_path,
        trust_remote_code=True,
    )

    del test_model
    gc.collect()
    torch.cuda.empty_cache()
    print("TE model loading test successful!")


def export_te_checkpoint(te_checkpoint_path: str, output_path: str):
    """Export a Transformer Engine checkpoint back to the original HuggingFace Geneformer format.

    This function converts from the NVIDIA Transformer Engine (TE) format back to the
    weight format compatible with the original Geneformer checkpoints.
    The TE model is also a HuggingFace model (you can load it with AutoModel.from_pretrained),
    but this conversion ensures compatibility with the original Geneformer model format.

    Args:
        te_checkpoint_path (str): Path to the TE checkpoint
        output_path (str): Output path for the converted Geneformer format model
    """
    if not Path(te_checkpoint_path).exists():
        raise FileNotFoundError(f"TE checkpoint {te_checkpoint_path} not found")

    print(f"Converting {te_checkpoint_path} from TE format back to original HuggingFace Geneformer format")

    # Load the TE model

    model_te = BertForMaskedLM.from_pretrained(te_checkpoint_path)

    # Convert TE model to HF format

    model_hf = convert_geneformer_te_to_hf(model_te)
    model_hf.save_pretrained(output_path)

    # Update config to remove TE-specific settings
    config_path = Path(output_path) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        config.pop("auto_map", None)
        config["model_type"] = "bert"  # Geneformer uses BERT architecture
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    # Smoke test that the converted model can be loaded
    model_hf = AutoModelForMaskedLM.from_pretrained(
        output_path,
        trust_remote_code=False,
    )

    del model_hf
    gc.collect()
    torch.cuda.empty_cache()
    print("Geneformer model loading test successful!")
