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
from jinja2 import Template
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from convert import convert_esm_hf_to_te
from modeling_esm_te import AUTO_MAP


BENCHMARK_RESULTS = {
    "esm2_t6_8M_UR50D": {"CAMEO": 0.48, "CASP14": 0.37},
    "esm2_t12_35M_UR50D": {"CAMEO": 0.56, "CASP14": 0.41},
    "esm2_t30_150M_UR50D": {"CAMEO": 0.65, "CASP14": 0.49},
    "esm2_t33_650M_UR50D": {"CAMEO": 0.70, "CASP14": 0.51},
    "esm2_t36_3B_UR50D": {"CAMEO": 0.72, "CASP14": 0.52},
    "esm2_t48_15B_UR50D": {"CAMEO": 0.72, "CASP14": 0.55},
}


ESM_TAGS = [
    "esm2_t6_8M_UR50D",
    "esm2_t12_35M_UR50D",
    "esm2_t30_150M_UR50D",
    "esm2_t33_650M_UR50D",
    "esm2_t36_3B_UR50D",
    "esm2_t48_15B_UR50D",
]


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
    model_hf_masked_lm = AutoModelForMaskedLM.from_pretrained(f"facebook/{tag}")
    model_hf = AutoModel.from_pretrained(f"facebook/{tag}")
    model_hf_masked_lm.esm.pooler = model_hf.pooler

    # Export without vocab padding so the checkpoint stores embeddings at the real
    # vocab_size.  This avoids shape-mismatch errors in vLLM's VocabParallelEmbedding,
    # which expects vocab_size-shaped weights.
    model_te = convert_esm_hf_to_te(model_hf_masked_lm, padded_vocab_size=None)
    model_te.save_pretrained(export_path / tag)

    tokenizer = AutoTokenizer.from_pretrained("esm_fast_tokenizer")  # Use our PreTrainedTokenizerFast implementation.
    tokenizer.save_pretrained(export_path / tag)

    # Patch the config
    with open(export_path / tag / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = AUTO_MAP

    with open(export_path / tag / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy("modeling_esm_te.py", export_path / tag / "esm_nv.py")

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
        cameo_score=BENCHMARK_RESULTS[tag]["CAMEO"],
        casp14_score=BENCHMARK_RESULTS[tag]["CASP14"],
    )

    # Write the rendered README
    with open(export_path / tag / "README.md", "w") as f:
        f.write(rendered_readme)

    shutil.copy("LICENSE", export_path / tag / "LICENSE")

    del model_hf, model_te, model_hf_masked_lm
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


def main():
    """Export the ESM2 models from Hugging Face to the Transformer Engine format."""
    # TODO (peter): maybe add a way to specify the model to export or option to export all models?
    for tag in ESM_TAGS:
        print(f"Converting {tag}...")
        export_hf_checkpoint(tag, Path("./checkpoint_export"))


if __name__ == "__main__":
    main()
