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

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import gc
import json
import shutil
from pathlib import Path

import torch
from jinja2 import Template
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from amplify.state_dict_convert import convert_amplify_hf_to_te


BENCHMARK_SCORES = {
    "AMPLIFY_120M": {"CAMEO": "17.8±14.1", "CASP14": "12.4±11.3", "CASP15": "16.9±13.2"},
    "AMPLIFY_350M": {"CAMEO": "20.9±15.7", "CASP14": "16.6±13.6", "CASP15": "20.0±14.6"},
}


def format_parameter_count(num_params: int, sig: int = 1) -> str:
    """Format parameter count in scientific notation (e.g., 6.5 x 10^8).

    Args:
        num_params: Total number of parameters
        sig: Number of digits to include after the decimal point

    Returns:
        Formatted string in scientific notation
    """
    s = f"{num_params:.{sig}e}"
    base, exp = s.split("e")
    return f"{base} x 10^{int(exp)}"


def export_hf_checkpoint(tag: str, export_path: Path):
    """Export a Hugging Face checkpoint to a Transformer Engine checkpoint.

    Args:
        tag: The tag of the checkpoint to export.
        export_path: The parent path to export the checkpoint to.
    """
    model_hf = AutoModel.from_pretrained(f"chandar-lab/{tag}", trust_remote_code=True, revision="d918a9e8")
    model_te = convert_amplify_hf_to_te(model_hf)
    model_te.save_pretrained(export_path / tag)

    tokenizer = AutoTokenizer.from_pretrained(f"chandar-lab/{tag}", revision="d918a9e8")
    tokenizer.save_pretrained(export_path / tag)

    # Patch the config
    with open(export_path / tag / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "amplify_te.AMPLIFYConfig",
        "AutoModel": "amplify_te.AMPLIFY",
        "AutoModelForMaskedLM": "amplify_te.AMPLIFYForMaskedLM",
    }

    with open(export_path / tag / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy("src/amplify/amplify_te.py", export_path / tag / "amplify_te.py")
    shutil.copy("LICENSE", export_path / tag / "LICENSE")

    # Calculate model parameters and render README template
    num_params = sum(p.numel() for p in model_te.parameters())
    formatted_params = format_parameter_count(num_params)

    # Read and render the template
    with open("model_readme.template", "r", encoding="utf-8") as f:
        template_content = f.read()

    template = Template(template_content)
    rendered_readme = template.render(
        num_params=formatted_params,
        model_tag=tag,
        cameo_score=BENCHMARK_SCORES[tag]["CAMEO"],
        casp14_score=BENCHMARK_SCORES[tag]["CASP14"],
        casp15_score=BENCHMARK_SCORES[tag]["CASP15"],
    )

    # Write the rendered README
    with open(export_path / tag / "README.md", "w") as f:
        f.write(rendered_readme)

    del model_hf, model_te
    gc.collect()
    torch.cuda.empty_cache()

    # Smoke test that the model can be loaded.
    model_te = AutoModelForMaskedLM.from_pretrained(
        export_path / tag,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    del model_te
    gc.collect()
    torch.cuda.empty_cache()
