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

import sys
from pathlib import Path

import pytest
import torch
from megatron.core.transformer.module import Float16Module
from nemo.lightning import io
from transformers import AutoModel

from bionemo.amplify.convert import HFAMPLIFYImporter, maybe_mock_xformers  # noqa: F401
from bionemo.amplify.hf_rotary import apply_rotary_emb
from bionemo.amplify.model import AMPLIFYConfig
from bionemo.amplify.tokenizer import BioNeMoAMPLIFYTokenizer
from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.testing.compare import ForwardHook, assert_cosine_similarity, get_input_tensors
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import megatron_parallel_state_utils


def assert_amplify_equivalence(
    ckpt_path: str,
    model_tag: str,
    precision: PrecisionTypes = "fp32",
    rtol: float | None = None,
    atol: float | None = None,
    loss_only: bool = False,
) -> None:
    """Assert that the HF and NeMo models are equivalent.

    Args:
        ckpt_path: Path to the NeMo2 checkpoint.
        model_tag: HuggingFace model tag.
        precision: Precision type to use for the comparison.
        rtol: Relative tolerance for the comparison.
        atol: Absolute tolerance for the comparison.
        loss_only: Whether to only check the loss. For the BF16 precision, intermediate values are tough to compare with
            appropriate precision, so we just do a sanity check with the loss values.
    """

    tokenizer = BioNeMoAMPLIFYTokenizer()

    input_ids, attention_mask = get_input_tensors(tokenizer)
    hf_results = load_and_evaluate_hf_amplify(model_tag, precision, input_ids, attention_mask)
    nemo_results = load_and_evaluate_nemo_amplify(
        tokenizer,
        ckpt_path,
        precision,
        input_ids,
        attention_mask,
    )

    if loss_only:
        torch.testing.assert_close(hf_results["loss"], nemo_results["loss"], rtol=rtol, atol=atol)
        return

    torch.testing.assert_close(hf_results["embeddings"], nemo_results["embeddings"], rtol=rtol, atol=atol)
    torch.testing.assert_close(hf_results["query_post_rot"], nemo_results["query_post_rot"], rtol=rtol, atol=atol)
    torch.testing.assert_close(hf_results["key_post_rot"], nemo_results["key_post_rot"], rtol=rtol, atol=atol)
    torch.testing.assert_close(hf_results["value"], nemo_results["value"], rtol=rtol, atol=atol)

    assert_cosine_similarity(
        hf_results["attn_output"],
        nemo_results["attn_output"],
        attention_mask.cpu(),
        rtol=rtol,
        atol=atol,
        msg="Attn output",
    )

    assert_cosine_similarity(
        hf_results["attn_linear_output"],
        nemo_results["attn_linear_output"],
        attention_mask.cpu(),
        rtol=rtol,
        atol=atol,
        msg="Attn linear output",
    )

    for i, (hf_block_output, nemo_block_output) in enumerate(
        zip(hf_results["encoder_block_outputs"], nemo_results["encoder_block_outputs"], strict=True)
    ):
        assert_cosine_similarity(
            hf_block_output,
            nemo_block_output,
            attention_mask.cpu(),
            rtol=rtol,
            atol=atol,
            msg=f"Encoder block output {i}",
        )

    assert_cosine_similarity(
        nemo_results["logits"],
        hf_results["logits"],
        attention_mask,
        rtol,
        atol,
        msg="Output logits",
    )

    # We're not able to check the "hidden states" as easily, because the NeMo model returns a hidden state after the
    # final layer norm, while the HF model returns the output of the final encoder block (checked above). Something to
    # keep in mind in case we use those hidden states in any downstream tasks.


def load_and_evaluate_hf_amplify(
    model_tag: str, precision: PrecisionTypes, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Load a HuggingFace model and evaluate it on the given inputs.

    Args:
        model_tag: The HuggingFace model tag for the model to compare against.
        precision: The precision type to use for the comparison.
        input_ids: The input IDs tensor to evaluate.
        attention_mask: The attention mask tensor to evaluate.
    """
    hf_model = AutoModel.from_pretrained(
        model_tag,
        torch_dtype=get_autocast_dtype(precision),
        trust_remote_code=True,
    )

    embedding_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    hf_model.encoder.register_forward_hook(embedding_hook)

    query_pre_rot_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    hf_model.transformer_encoder[0].q.register_forward_hook(query_pre_rot_hook)

    key_pre_rot_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    hf_model.transformer_encoder[0].k.register_forward_hook(key_pre_rot_hook)

    value_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    hf_model.transformer_encoder[0].v.register_forward_hook(value_hook)

    # The output of the attention layer is the same as the output of the linear layer, but the actual attention function
    # isn't wrapped in a nn.Module.
    attn_output_hook = ForwardHook(lambda inputs, outputs: inputs[0])
    hf_model.transformer_encoder[0].wo.register_forward_hook(attn_output_hook)

    attn_linear_output_hook = ForwardHook(lambda inputs, outputs: outputs[0])
    hf_model.transformer_encoder[0].wo.register_forward_hook(attn_linear_output_hook)

    encoder_block_hooks = [
        ForwardHook(lambda inputs, outputs: outputs[0]) for _ in range(len(hf_model.transformer_encoder))
    ]
    for i, hook in enumerate(encoder_block_hooks):
        hf_model.transformer_encoder[i].register_forward_hook(hook)

    hf_model = hf_model.to("cuda").eval()

    # Attention mask here is a boolean tensor, but we need an additive attention mask.
    additive_attention_mask = torch.where(attention_mask, float(0.0), float("-inf")).to(get_autocast_dtype(precision))

    hf_output_all = hf_model(input_ids, additive_attention_mask, output_hidden_states=True)

    # These post-rotary embeddings are applied in the forward pass of the model, so we need to apply them here.
    xq = query_pre_rot_hook.data.view(
        input_ids.shape[0],
        input_ids.shape[1],
        hf_model.config.num_attention_heads,
        hf_model.transformer_encoder[0].d_head,
    )
    xk = key_pre_rot_hook.data.view(
        input_ids.shape[0],
        input_ids.shape[1],
        hf_model.config.num_attention_heads,
        hf_model.transformer_encoder[0].d_head,
    )
    xq, xk = apply_rotary_emb(xq, xk, hf_model.freqs_cis[: input_ids.shape[1]].cpu())

    hf_logits = hf_output_all.logits

    loss = torch.nn.functional.cross_entropy(hf_logits[attention_mask], input_ids[attention_mask])

    return {
        "embeddings": embedding_hook.data,
        "query_post_rot": xq.flatten(-2, -1),
        "key_post_rot": xk.flatten(-2, -1),
        "value": value_hook.data,
        "attn_output": attn_output_hook.data,
        "attn_linear_output": attn_linear_output_hook.data,
        "encoder_block_outputs": [hook.data for hook in encoder_block_hooks],
        "logits": hf_logits,
        "loss": loss,
    }


def load_and_evaluate_nemo_amplify(
    tokenizer: BioNeMoAMPLIFYTokenizer,
    ckpt_path: Path | str,
    precision: PrecisionTypes,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Load a AMPLIFY NeMo2 model checkpoint and evaluate it on the input tensors.

    It would be great to make this more ergonomic, i.e., how to create a model from a checkpoint and evaluate it.

    Args:
        tokenizer: Not sure why we need to pass a tokenizer to `configure_model`.
        ckpt_path: Path to the newly created NeMo2 converted checkpoint.
        precision: Precision type to use for the model.
        input_ids: Input tokens
        attention_mask: Input attention mask
    """

    dtype = get_autocast_dtype(precision)
    nemo_config = AMPLIFYConfig(
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

    embedding_hook = ForwardHook(lambda inputs, outputs: outputs[0].transpose(0, 1))
    nemo_model.embedding.register_forward_hook(embedding_hook)

    query_post_rot_hook = ForwardHook(lambda inputs, outputs: inputs[0].transpose(0, 1).flatten(-2, -1))
    nemo_model.encoder.layers[0].self_attention.core_attention.register_forward_hook(query_post_rot_hook)

    key_post_rot_hook = ForwardHook(lambda inputs, outputs: inputs[1].transpose(0, 1).flatten(-2, -1))
    nemo_model.encoder.layers[0].self_attention.core_attention.register_forward_hook(key_post_rot_hook)

    value_post_rot_hook = ForwardHook(lambda inputs, outputs: inputs[2].transpose(0, 1).flatten(-2, -1))
    nemo_model.encoder.layers[0].self_attention.core_attention.register_forward_hook(value_post_rot_hook)

    attn_output_hook = ForwardHook(lambda inputs, outputs: outputs[0].transpose(0, 1))
    nemo_model.encoder.layers[0].self_attention.core_attention.register_forward_hook(attn_output_hook)

    attn_linear_output_hook = ForwardHook(lambda inputs, outputs: outputs[0].transpose(0, 1))
    nemo_model.encoder.layers[0].self_attention.linear_proj.register_forward_hook(attn_linear_output_hook)

    encoder_block_hooks = [
        ForwardHook(lambda inputs, outputs: outputs[0].transpose(0, 1)) for _ in range(len(nemo_model.encoder.layers))
    ]
    for i, hook in enumerate(encoder_block_hooks):
        nemo_model.encoder.layers[i].register_forward_hook(hook)

    if dtype is torch.float16 or dtype is torch.bfloat16:
        nemo_model = Float16Module(nemo_config, nemo_model)

    nemo_output = nemo_model(input_ids, attention_mask)

    nemo_logits = nemo_output["token_logits"].transpose(0, 1).contiguous()[..., : tokenizer.vocab_size]

    loss = torch.nn.functional.cross_entropy(nemo_logits[attention_mask], input_ids[attention_mask])

    return {
        "embeddings": embedding_hook.data,
        "query_post_rot": query_post_rot_hook.data,
        "key_post_rot": key_post_rot_hook.data,
        "value": value_post_rot_hook.data,
        "attn_output": attn_output_hook.data,
        "attn_linear_output": attn_linear_output_hook.data,
        "encoder_block_outputs": [hook.data for hook in encoder_block_hooks],
        "logits": nemo_logits,
        "loss": loss,
    }


def test_convert_amplify_120M_smoke(tmp_path):
    maybe_mock_xformers()
    model_tag = "chandar-lab/AMPLIFY_120M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")


@pytest.mark.skipif(
    not sys.modules.get("xformers"),
    reason="AMPLIFY golden value tests require xformers. Run `pip install -v -U "
    "git+https://github.com/facebookresearch/xformers.git@v0.0.29.post1#egg=xformers` to enable.",
)
def test_convert_amplify_120M(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_120M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_amplify_equivalence(tmp_path / "nemo_checkpoint", model_tag, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(
    not sys.modules.get("xformers"),
    reason="AMPLIFY golden value tests require xformers. Run `pip install -v -U "
    "git+https://github.com/facebookresearch/xformers.git@v0.0.29.post1#egg=xformers` to enable.",
)
def test_convert_amplify_120M_bf16(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_120M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_amplify_equivalence(
            tmp_path / "nemo_checkpoint",
            model_tag,
            precision="bf16",
            loss_only=True,
        )


@pytest.mark.skipif(
    not sys.modules.get("xformers"),
    reason="AMPLIFY golden value tests require xformers. Run `pip install -v -U "
    "git+https://github.com/facebookresearch/xformers.git@v0.0.29.post1#egg=xformers` to enable.",
)
def test_convert_amplify_350M(tmp_path):
    model_tag = "chandar-lab/AMPLIFY_350M"
    module = biobert_lightning_module(config=AMPLIFYConfig())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_amplify_equivalence(tmp_path / "nemo_checkpoint", model_tag, atol=1e-4, rtol=1e-4)
