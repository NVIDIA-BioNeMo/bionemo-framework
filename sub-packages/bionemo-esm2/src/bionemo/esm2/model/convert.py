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


from pathlib import Path

import torch
import typer
from megatron.core.dist_checkpointing.validation import StrictHandling
from nemo.lightning import MegatronStrategy, Trainer, io, teardown
from nemo.lightning.pytorch.utils import dtype_from_hf
from transformers import AutoConfig as HFAutoConfig
from transformers import AutoModelForMaskedLM
from transformers.modeling_utils import no_init_weights
from transformers.models.esm.configuration_esm import EsmConfig as HFEsmConfig
from transformers.models.esm.modeling_esm import EsmForMaskedLM

from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.esm2.model.model import ESM2Config
from bionemo.llm.lightning import BionemoLightningModule
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


@io.model_importer(BionemoLightningModule, "hf")
class HFESM2Importer(io.ModelConnector[AutoModelForMaskedLM, BionemoLightningModule]):
    """Converts a Hugging Face ESM-2 model to a NeMo ESM-2 model."""

    def init(self) -> BionemoLightningModule:
        """Initialize the converted model."""
        return biobert_lightning_module(self.config, tokenizer=self.tokenizer)

    def apply(self, output_path: Path) -> Path:
        """Applies the transformation.

        Largely inspired by
        https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/hf-integration.html
        """
        source = AutoModelForMaskedLM.from_pretrained(str(self), trust_remote_code=True, torch_dtype="auto")
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(source, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted ESM-2 model to Nemo, model saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    def convert_state(self, source, target):
        """Converting HF state dict to NeMo state dict."""
        mapping = {
            # "esm.encoder.layer.0.attention.self.rotary_embeddings.inv_freq": "rotary_pos_emb.inv_freq",
            "esm.encoder.layer.*.attention.output.dense.weight": "encoder.layers.*.self_attention.linear_proj.weight",
            "esm.encoder.layer.*.attention.output.dense.bias": "encoder.layers.*.self_attention.linear_proj.bias",
            "esm.encoder.layer.*.attention.LayerNorm.weight": "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "esm.encoder.layer.*.attention.LayerNorm.bias": "encoder.layers.*.self_attention.linear_qkv.layer_norm_bias",
            "esm.encoder.layer.*.intermediate.dense.weight": "encoder.layers.*.mlp.linear_fc1.weight",
            "esm.encoder.layer.*.intermediate.dense.bias": "encoder.layers.*.mlp.linear_fc1.bias",
            "esm.encoder.layer.*.output.dense.weight": "encoder.layers.*.mlp.linear_fc2.weight",
            "esm.encoder.layer.*.output.dense.bias": "encoder.layers.*.mlp.linear_fc2.bias",
            "esm.encoder.layer.*.LayerNorm.weight": "encoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "esm.encoder.layer.*.LayerNorm.bias": "encoder.layers.*.mlp.linear_fc1.layer_norm_bias",
            "esm.encoder.emb_layer_norm_after.weight": "encoder.final_layernorm.weight",
            "esm.encoder.emb_layer_norm_after.bias": "encoder.final_layernorm.bias",
            "lm_head.dense.weight": "lm_head.dense.weight",
            "lm_head.dense.bias": "lm_head.dense.bias",
            "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
        }

        return io.apply_transforms(
            source,
            target,
            mapping=mapping,
            transforms=[_pad_embeddings, _pad_bias, _import_qkv_weight, _import_qkv_bias],
        )

    @property
    def tokenizer(self) -> BioNeMoESMTokenizer:
        """We just have the one tokenizer for ESM-2."""
        return get_tokenizer()

    @property
    def config(self) -> ESM2Config:
        """Returns the transformed ESM-2 config given the model tag."""
        source = HFAutoConfig.from_pretrained(str(self), trust_remote_code=True)
        output = ESM2Config(
            num_layers=source.num_hidden_layers,
            hidden_size=source.hidden_size,
            ffn_hidden_size=source.intermediate_size,
            position_embedding_type="rope",
            num_attention_heads=source.num_attention_heads,
            seq_length=source.max_position_embeddings,
            fp16=(dtype_from_hf(source) == torch.float16),
            bf16=(dtype_from_hf(source) == torch.bfloat16),
            params_dtype=dtype_from_hf(source),
        )

        return output


@io.model_exporter(BionemoLightningModule, "hf")
class HFESM2Exporter(io.ModelConnector[BionemoLightningModule, EsmForMaskedLM]):
    """Exporter Connector for converting NeMo ESM-2 Model to HF."""

    def init(self, dtype: torch.dtype = torch.bfloat16) -> EsmForMaskedLM:
        """Initialize the target model."""
        with no_init_weights():
            return EsmForMaskedLM._from_config(self.config, torch_dtype=dtype)

    def apply(self, output_path: Path) -> Path:
        """Applies the transformation."""
        cpu = not torch.distributed.is_initialized()
        trainer = Trainer(
            devices=1,
            accelerator="cpu" if cpu else "gpu",
            strategy=MegatronStrategy(
                ddp="pytorch", setup_optimizers=False, ckpt_load_strictness=StrictHandling.LOG_UNEXPECTED
            ),
        )
        source, _ = self.nemo_load(self, trainer=trainer, cpu=cpu)

        dtype = source.dtype

        # Not sure why we need to do this, for some reason lm_head stays as fp32
        source.module.lm_head.to(dtype)

        target = self.init(dtype)
        target = self.convert_state(source, target)

        target = target.cpu()
        target.esm.contact_head.regression.reset_parameters()  # We don't initialize this head, so we need to reset.
        target.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return output_path

    @property
    def tokenizer(self):
        """Retrieve Tokenizer from HF."""
        return get_tokenizer()

    def convert_state(self, nemo_module, target):
        """Convert NeMo state dict to HF style."""
        mapping = {
            "encoder.final_layernorm.weight": "esm.encoder.emb_layer_norm_after.weight",
            "encoder.final_layernorm.bias": "esm.encoder.emb_layer_norm_after.bias",
            "encoder.layers.*.self_attention.linear_proj.weight": "esm.encoder.layer.*.attention.output.dense.weight",
            "encoder.layers.*.self_attention.linear_proj.bias": "esm.encoder.layer.*.attention.output.dense.bias",
            "encoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "esm.encoder.layer.*.attention.LayerNorm.weight",
            "encoder.layers.*.self_attention.linear_qkv.layer_norm_bias": "esm.encoder.layer.*.attention.LayerNorm.bias",
            "encoder.layers.*.mlp.linear_fc1.weight": "esm.encoder.layer.*.intermediate.dense.weight",
            "encoder.layers.*.mlp.linear_fc1.bias": "esm.encoder.layer.*.intermediate.dense.bias",
            "encoder.layers.*.mlp.linear_fc2.weight": "esm.encoder.layer.*.output.dense.weight",
            "encoder.layers.*.mlp.linear_fc2.bias": "esm.encoder.layer.*.output.dense.bias",
            "encoder.layers.*.mlp.linear_fc1.layer_norm_weight": "esm.encoder.layer.*.LayerNorm.weight",
            "encoder.layers.*.mlp.linear_fc1.layer_norm_bias": "esm.encoder.layer.*.LayerNorm.bias",
            "lm_head.dense.weight": "lm_head.dense.weight",
            "lm_head.dense.bias": "lm_head.dense.bias",
            "lm_head.layer_norm.weight": "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias": "lm_head.layer_norm.bias",
        }

        return io.apply_transforms(
            nemo_module,
            target,
            mapping=mapping,
            transforms=[_export_qkv_weight, _export_qkv_bias, _export_embedding, _export_bias],
        )

    @property
    def config(self) -> HFEsmConfig:
        """Generate HF Config based on NeMo config."""
        source: ESM2Config = io.load_context(Path(str(self)), subpath="model.config")

        return HFEsmConfig(
            attention_probs_dropout_prob=float(source.attention_dropout),
            emb_layer_norm_before=False,
            hidden_act="gelu",
            hidden_dropout_prob=float(source.hidden_dropout),
            hidden_size=int(source.hidden_size),
            initializer_range=float(source.init_method_std),
            intermediate_size=int(source.ffn_hidden_size),
            is_folding_model=False,
            layer_norm_eps=float(source.layernorm_epsilon),
            mask_token_id=32,
            max_position_embeddings=int(source.seq_length),
            model_type="esm",
            num_attention_heads=int(source.num_attention_heads),
            num_hidden_layers=int(source.num_layers),
            pad_token_id=1,
            position_embedding_type="rotary",
            token_dropout=True,
            torch_dtype=torch.bfloat16,
            use_cache=True,
            vocab_size=self.tokenizer.vocab_size,
        )


@io.state_transform(
    source_key="encoder.layers.*.self_attention.linear_qkv.weight",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
)
def _export_qkv_weight(ctx: io.TransformCTX, linear_qkv):
    """Convert NeMo QKV weights to HF format."""
    megatron_config = ctx.target.config
    num_heads = megatron_config.num_attention_heads
    head_size = megatron_config.hidden_size // num_heads

    reshaped_qkv = linear_qkv.reshape(3 * num_heads, head_size, megatron_config.hidden_size)
    query = reshaped_qkv[::3, :, :].reshape(-1, megatron_config.hidden_size)
    key = reshaped_qkv[1::3, :, :].reshape(-1, megatron_config.hidden_size)
    value = reshaped_qkv[2::3, :, :].reshape(-1, megatron_config.hidden_size)

    return query, key, value


@io.state_transform(
    source_key="encoder.layers.*.self_attention.linear_qkv.bias",
    target_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
)
def _export_qkv_bias(ctx: io.TransformCTX, qkv_bias):
    """Export nemo qkv biases to HF format."""
    megatron_config = ctx.target.config
    num_heads = megatron_config.num_attention_heads
    head_size = megatron_config.hidden_size // num_heads

    reshaped_bias = qkv_bias.reshape(-1, head_size)

    q_bias = reshaped_bias[::3].reshape(-1)
    k_bias = reshaped_bias[1::3].reshape(-1)
    v_bias = reshaped_bias[2::3].reshape(-1)

    return q_bias, k_bias, v_bias


@io.state_transform(
    source_key="embedding.word_embeddings.weight",
    target_key="esm.embeddings.word_embeddings.weight",
)
def _export_embedding(ctx: io.TransformCTX, embedding):
    """Convert NeMo embeddings to HF format."""
    # prune padding
    return embedding[: ctx.target.config.vocab_size, :]


@io.state_transform(
    source_key="output_layer.bias",
    target_key="lm_head.bias",
)
def _export_bias(ctx: io.TransformCTX, bias):
    """Convert NeMo embeddings to HF format."""
    # prune bias
    return bias[: ctx.target.config.vocab_size]


@io.state_transform(
    source_key="esm.embeddings.word_embeddings.weight",
    target_key="embedding.word_embeddings.weight",
)
def _pad_embeddings(ctx: io.TransformCTX, source_embed):
    """Pad the embedding layer to the new input dimension."""
    nemo_embedding_dimension = ctx.target.config.make_vocab_size_divisible_by
    hf_embedding_dimension = source_embed.size(0)
    num_padding_rows = nemo_embedding_dimension - hf_embedding_dimension
    padding_rows = torch.zeros(num_padding_rows, source_embed.size(1))
    return torch.cat((source_embed, padding_rows), dim=0)


@io.state_transform(
    source_key="lm_head.bias",
    target_key="output_layer.bias",
)
def _pad_bias(ctx: io.TransformCTX, source_bias):
    """Pad the embedding layer to the new input dimension."""
    nemo_embedding_dimension = ctx.target.config.make_vocab_size_divisible_by
    hf_embedding_dimension = source_bias.size(0)
    output_bias = torch.zeros(nemo_embedding_dimension, dtype=source_bias.dtype, device=source_bias.device)
    output_bias[:hf_embedding_dimension] = source_bias
    return output_bias


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.weight",
        "esm.encoder.layer.*.attention.self.key.weight",
        "esm.encoder.layer.*.attention.self.value.weight",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.weight",
)
def _import_qkv_weight(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_weights = torch.cat((query, key, value), dim=0)
    input_shape = concat_weights.size()
    np = ctx.target.config.num_attention_heads
    # transpose weights
    # [sequence length, batch size, num_splits_model_parallel * attention head size * #attention heads]
    # --> [sequence length, batch size, attention head size * num_splits_model_parallel * #attention heads]
    concat_weights = concat_weights.view(3, np, -1, query.size()[-1])
    concat_weights = concat_weights.transpose(0, 1).contiguous()
    concat_weights = concat_weights.view(*input_shape)
    return concat_weights


@io.state_transform(
    source_key=(
        "esm.encoder.layer.*.attention.self.query.bias",
        "esm.encoder.layer.*.attention.self.key.bias",
        "esm.encoder.layer.*.attention.self.value.bias",
    ),
    target_key="encoder.layers.*.self_attention.linear_qkv.bias",
)
def _import_qkv_bias(ctx: io.TransformCTX, query, key, value):
    """Pad the embedding layer to the new input dimension."""
    concat_biases = torch.cat((query, key, value), dim=0)
    input_shape = concat_biases.size()
    np = ctx.target.config.num_attention_heads
    # transpose biases
    # [num_splits_model_parallel * attention head size * #attention heads]
    # --> [attention head size * num_splits_model_parallel * #attention heads]
    concat_biases = concat_biases.view(3, np, -1)
    concat_biases = concat_biases.transpose(0, 1).contiguous()
    concat_biases = concat_biases.view(*input_shape)
    return concat_biases


app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def convert_nemo_to_hf(nemo_path: str, output_path: str, overwrite: bool = True):
    """Convert a NeMo ESM-2 checkpoint to a HuggingFace checkpoint.

    Args:
        nemo_path: Path to the NeMo checkpoint.
        output_path: Path to the output HuggingFace checkpoint.
        overwrite: Whether to overwrite the output path if it already exists.
    """
    io.export_ckpt(
        Path(nemo_path),
        "hf",
        Path(output_path),
        overwrite=overwrite,
        load_connector=lambda path, ext: BionemoLightningModule.exporter(ext, path),
    )


@app.command()
def convert_hf_to_nemo(hf_tag_or_path: str, output_path: str, overwrite: bool = True):
    """Convert a HuggingFace ESM-2 checkpoint to a NeMo ESM-2 checkpoint.

    Args:
        hf_tag_or_path: Tag or path to the HuggingFace checkpoint.
        output_path: Path to the output NeMo checkpoint.
        overwrite: Whether to overwrite the output path if it already exists.
    """
    module = biobert_lightning_module(config=ESM2Config(), post_process=True)
    io.import_ckpt(module, f"hf://{hf_tag_or_path}", Path(output_path), overwrite=overwrite)


if __name__ == "__main__":
    app()
