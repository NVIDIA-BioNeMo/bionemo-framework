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

"""Create a Mixtral TE checkpoint from a HuggingFace Mixtral model."""

import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import convert
from modeling_mixtral_te import AUTO_MAP


def _convert_hf_checkpoint(
    tag: str,
    *,
    torch_dtype: torch.dtype | None,
    low_cpu_mem_usage: bool,
    config_kwargs: dict,
):
    model_hf = AutoModelForCausalLM.from_pretrained(
        tag,
        dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model_te = convert.convert_mixtral_hf_to_te(model_hf, **config_kwargs)
    del model_hf
    return model_te


def export_hf_checkpoint(
    tag: str,
    export_path: Path,
    *,
    torch_dtype: torch.dtype | None = None,
    low_cpu_mem_usage: bool = False,
    **config_kwargs,
) -> None:
    """Export a Hugging Face checkpoint to a Transformer Engine checkpoint.

    Args:
        tag: The tag (or local path) of the Hugging Face checkpoint to export.
        export_path: The parent path to export the checkpoint to.
        torch_dtype: Optional dtype to load the source model in (e.g. ``torch.bfloat16`` for
            large models such as Mixtral-8x7B). ``None`` keeps the source checkpoint dtype.
        low_cpu_mem_usage: Forwarded to ``from_pretrained``; enable for large models to avoid
            materializing a second CPU copy of the weights during load.
        **config_kwargs: Extra ``NVMixtralConfig`` overrides forwarded to
            :func:`convert.convert_mixtral_hf_to_te`, e.g. ``expert_ffn_mode="fused_grouped_mlp"``,
            ``attn_input_format="bshd"``, ``self_attn_mask_type="causal"``.
    """
    model_te = _convert_hf_checkpoint(
        tag,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        config_kwargs=config_kwargs,
    )

    # save_pretrained works for every expert_ffn_mode: NVMixtralPreTrainedModel.state_dict() drops the
    # duplicate _experts_ffn_op.* aliases so the safetensors writer sees exactly one copy of each
    # (fused) expert weight.
    model_te.save_pretrained(export_path)

    tokenizer = AutoTokenizer.from_pretrained(tag)
    tokenizer.save_pretrained(export_path)

    # Patch the config
    with open(export_path / "config.json", "r") as f:
        config = json.load(f)

    config["auto_map"] = AUTO_MAP

    with open(export_path / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)

    shutil.copy(Path(__file__).parent / "modeling_mixtral_te.py", export_path / "modeling_mixtral_te.py")


def export_hf_state_dict(
    tag: str,
    output_path: Path,
    *,
    torch_dtype: torch.dtype | None = None,
    low_cpu_mem_usage: bool = False,
    **config_kwargs,
) -> None:
    """Convert a Hugging Face Mixtral checkpoint to one mmap-loadable TE training state dict.

    Unlike :func:`export_hf_checkpoint`, this API intentionally writes only the model state needed
    by native-TE training. A single ``torch.save`` file is the canonical handoff to the training
    recipe: it avoids Hugging Face serialization-format dispatch and can be mapped read-only across
    ranks on a node with ``torch.load(mmap=True)``.

    Args:
        tag: Hugging Face model tag or local checkpoint path.
        output_path: Destination file, conventionally ending in ``.pt``.
        torch_dtype: Optional source-model load dtype.
        low_cpu_mem_usage: Forwarded to ``from_pretrained``.
        **config_kwargs: Extra ``NVMixtralConfig`` conversion overrides.
    """
    model_te = _convert_hf_checkpoint(
        tag,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        config_kwargs=config_kwargs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_te.state_dict(), output_path)


if __name__ == "__main__":
    export_hf_checkpoint("NeuralNovel/Mini-Mixtral-v0.2", Path("checkpoint_export"))
