# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
from pathlib import Path
from typing import Callable

import torch
from megatron.core.transformer.module import Float16Module
from transformers import AutoModelForMaskedLM

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data.tokenizer import get_tokenizer
from bionemo.esm2.model.model import ESM2Config


def assert_esm2_equivalence(
    ckpt_path: Path | str,
    model_tag: str | Path,
    precision: PrecisionTypes = "fp32",
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Testing utility to compare the outputs of a NeMo2 checkpoint to the original HuggingFace model weights.

    Compares the cosine similarity of the logit and hidden state outputs of a NeMo2 model checkpoint to the outputs of
    the corresponding HuggingFace model.

    Args:
        ckpt_path: A path to a NeMo2 checkpoint for an ESM-2 model.
        model_tag: The HuggingFace model tag for the model to compare against.
        precision: The precision type to use for the comparison. Defaults to "fp32".
        rtol: The relative tolerance to use for the comparison. Defaults to None, which chooses the tolerance based on
            the precision.
        atol: The absolute tolerance to use for the comparison. Defaults to None, which chooses the tolerance based on
            the precision.
    """
    tokenizer = get_tokenizer()

    input_ids, attention_mask = get_input_tensors(tokenizer)

    nemo_logits, nemo_hidden_state = load_and_evaluate_nemo_esm2(ckpt_path, precision, input_ids, attention_mask)
    gc.collect()
    torch.cuda.empty_cache()
    hf_logits, hf_hidden_state = load_and_evaluate_hf_model(model_tag, precision, input_ids, attention_mask)

    # Rather than directly comparing the logit or hidden state tensors, we compare their cosine similarity. These
    # should be essentially 1 if the outputs are equivalent, but is less sensitive to small numerical differences.
    # We don't care about the padding tokens, so we only compare the non-padding tokens.
    assert_cosine_similarity(nemo_logits, hf_logits, attention_mask, rtol, atol)
    assert_cosine_similarity(nemo_hidden_state, hf_hidden_state, attention_mask, rtol, atol)


def get_input_tensors(tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """Get input tensors for testing.

    Args:
        tokenizer: A huggingface-like tokenizer object.

    Returns:
        A tuple of the input IDs and attention mask tensors.
    """
    test_proteins = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLA",
        "MKTVRQERLKSI<mask>RILERSKEPVSGAQLAEELS<mask>SRQVIVQDIAYLRSLGYN<mask>VATPRGYVLAGG",
    ]
    tokens = tokenizer(test_proteins, return_tensors="pt", padding=True, truncation=True)
    input_ids: torch.Tensor = tokens["input_ids"]  # type: ignore
    attention_mask: torch.Tensor = tokens["attention_mask"]  # type: ignore

    # Pad the input IDs and attention mask to be divisible by 8 so xformers doesn't fail.
    padded_shape = math.ceil(attention_mask.size(1) / 8)
    padded_input_ids = torch.full((input_ids.size(0), padded_shape * 8), tokenizer.pad_token_id, dtype=torch.long)
    padded_input_ids[: input_ids.size(0), : input_ids.size(1)] = input_ids

    padded_attention_mask = torch.zeros((attention_mask.size(0), padded_shape * 8), dtype=torch.bool)
    padded_attention_mask[: attention_mask.size(0), : attention_mask.size(1)] = attention_mask

    return padded_input_ids.to("cuda"), padded_attention_mask.to("cuda")


def load_and_evaluate_nemo_esm2(
    ckpt_path: Path | str,
    precision: PrecisionTypes,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a NeMo2 ESM-2 model and evaluate it on the given inputs.

    Args:
        ckpt_path: A path to a NeMo2 checkpoint for an ESM-2 model.
        precision: The precision type to use for the comparison.
        input_ids: The input IDs tensor to evaluate.
        attention_mask: The attention mask tensor to evaluate.

    Returns:
        A tuple of the logits and hidden states tensors calculated by the NeMo2 model, respectively.
    """
    tokenizer = get_tokenizer()

    dtype = get_autocast_dtype(precision)
    nemo_config = ESM2Config(
        initial_ckpt_path=str(ckpt_path),
        include_embeddings=True,
        include_hiddens=True,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        autocast_dtype=dtype,
        bf16=dtype is torch.bfloat16,
        fp16=dtype is torch.float16,
    )

    nemo_model = nemo_config.configure_model(tokenizer).to("cuda").eval()

    if dtype is torch.float16 or dtype is torch.bfloat16:
        nemo_model = Float16Module(nemo_config, nemo_model)

    nemo_output = nemo_model(input_ids, attention_mask)
    nemo_logits = nemo_output["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]
    nemo_hidden_state = nemo_output["hidden_states"]
    return nemo_logits, nemo_hidden_state


def load_and_evaluate_hf_model(
    model_tag: str | Path, precision: PrecisionTypes, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a HuggingFace model and evaluate it on the given inputs.

    Args:
        model_tag: The HuggingFace model tag for the model to compare against.
        precision: The precision type to use for the comparison.
        input_ids: The input IDs tensor to evaluate.
        attention_mask: The attention mask tensor to evaluate.

    Returns:
        A tuple of the logits and hidden states tensors calculated by the HuggingFace model, respectively.
    """
    hf_model = AutoModelForMaskedLM.from_pretrained(
        model_tag,
        torch_dtype=get_autocast_dtype(precision),
        trust_remote_code=True,
    )
    hf_model = hf_model.to("cuda").eval()
    hf_output_all = hf_model(input_ids, attention_mask, output_hidden_states=True)
    hf_hidden_state = hf_output_all.hidden_states[-1]
    return hf_output_all.logits, hf_hidden_state


def assert_cosine_similarity(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    mask: torch.Tensor,
    rtol: float | None = None,
    atol: float | None = None,
    magnitude_rtol: float = 1e-2,
    magnitude_atol: float = 1e-2,
    msg: str | None = None,
) -> None:
    """Assert that both the cosine similarity between two tensors is close to 1, and the ratio of their magnitudes is 1.

    Args:
        tensor1: The first tensor to compare.
        tensor2: The second tensor to compare.
        mask: A mask tensor to apply to the comparison.
        rtol: The relative tolerance to use for the comparison. Defaults to 1e-4.
        atol: The absolute tolerance to use for the comparison. Defaults to 1e-4.
        magnitude_rtol: The relative tolerance to use for the magnitude comparison. Defaults to 1e-2.
        magnitude_atol: The absolute tolerance to use for the magnitude comparison. Defaults to 1e-2.
        msg: An optional message to include in the assertion error.
    """
    assert tensor1.size() == tensor2.size()

    similarity = torch.nn.functional.cosine_similarity(tensor1, tensor2, dim=2)
    similarity = similarity[mask == 1]

    torch.testing.assert_close(
        similarity,
        torch.ones_like(similarity),
        rtol=rtol,
        atol=atol,
        msg=lambda x: f"{msg} (angle): {x}",
    )

    magnitude_similarity = torch.norm(tensor1, dim=2) / torch.norm(tensor2, dim=2)
    magnitude_similarity = magnitude_similarity[mask == 1]
    torch.testing.assert_close(
        magnitude_similarity,
        torch.ones_like(magnitude_similarity),
        rtol=magnitude_rtol,
        atol=magnitude_atol,
        msg=lambda x: f"{msg} (magnitude): {x}",
    )


TransformFn = Callable[
    [tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    torch.Tensor,
]


class ForwardHook:
    """A forward hook to extract a desired intermediate tensor for later comparison."""

    def __init__(self, transform_fn: TransformFn) -> None:
        """A forward hook to extract a desired intermediate tensor for later comparison.

        The resulting tensor is saved in the `data` attribute of the hook.

        Args:
            transform_fn: A function that maps the input and output tensors of the module to the desired tensor.
        """
        self._transform_fn = transform_fn
        self._data: torch.Tensor | None = None

    def __call__(self, module, module_in, module_out):
        """The forward hook function."""
        if not isinstance(module_out, tuple):
            module_out = (module_out,)
        if not isinstance(module_in, tuple):
            module_in = (module_in,)

        self._data = self._transform_fn(module_in, module_out).detach().cpu()

    @property
    def data(self) -> torch.Tensor:
        """The extracted tensor from the forward hook."""
        if self._data is None:
            raise ValueError("No data has been saved in this hook.")
        return self._data


class TestHook:
    """A test hook that just captures the raw inputs and outputs."""

    def __init__(self) -> None:
        """A test hook that just captures the raw inputs and outputs."""
        self.inputs: tuple[torch.Tensor, ...] | None = None
        self.outputs: tuple[torch.Tensor, ...] | None = None

    def __call__(self, module, inputs, outputs):
        """The forward hook function."""
        self.inputs = inputs
        self.outputs = outputs
