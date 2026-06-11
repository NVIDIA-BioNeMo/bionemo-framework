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

import math
import tarfile
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple
from unittest import mock

import pytest
import torch
import torch.utils.data
from lightning.pytorch.loggers import TensorBoardLogger
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.transformer.module import Float16Module
from nemo import lightning as nl
from nemo.collections import llm as nllm
from nemo.lightning import io, resume
from nemo.lightning.nemo_logger import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks.model_transform import ModelTransform
from nemo.lightning.pytorch.callbacks.peft import PEFT
from nemo.lightning.pytorch.optim.lr_scheduler import WarmupPolicyScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from torch.nn import functional as F
from tqdm import tqdm

from bionemo.core.data.load import load
from bionemo.core.utils.batching_utils import pad_token_ids
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.core.utils.random_utils import random_numpy_context
from bionemo.geneformer.api import GeneformerConfig, GeneformerModel
from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.geneformer.model.finetune_token_regressor import (
    FineTuneSeqLenBioBertConfig,
    LoRAForGeneFormerTokenRegressor,
)
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker
from bionemo.testing.utils import (
    assert_matrix_correlation_above_value,
    assert_matrix_mape_below_value,
)


nemo1_checkpoint_path: Path = load("geneformer/qa")
nemo1_release_checkpoint_path: Path = load("geneformer/10M_240530:1.0")
nemo2_release_checkpoint_path: Path = load("geneformer/10M_240530:2.0")
nemo_1_per_layer_outputs_path: Path = load("single_cell/nemo1-geneformer-per-layer-outputs")
nemo_1_expected_values_path: Path = load("single_cell/nemo1-geneformer-golden-vals")
data_path: Path = load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl"


CELLS_FOR_TEST: List[List[str]] = [
    [
        "ENSG00000288623",
        "ENSG00000288658",
        "ENSG00000288681",
        "ENSG00000288698",
        "ENSGR0000002586",
        "ENSGR0000124333",
        "ENSGR0000124334",
        "ENSGR0000167393",
        "ENSGR0000168939",
        "ENSGR0000169084",
    ],
    [
        "ENSG00000259900",
        "ENSG00000259916",
        "ENSG00000259956",
        "ENSG00000259958",
        "ENSG00000259991",
        "ENSG00000260001",
        "ENSG00000260007",
        "ENSG00000260027",
        "ENSG00000260040",
        "ENSG00000260045",
        "ENSG00000260092",
        "ENSG00000260099",
        "ENSG00000260119",
    ],
    [
        "ENSG00000269743",
        "ENSG00000269746",
        "ENSG00000269748",
        "ENSG00000269753",
        "ENSG00000269754",
        "ENSG00000269755",
        "ENSG00000269759",
        "ENSG00000269766",
        "ENSG00000269773",
        "ENSG00000269781",
        "ENSG00000269782",
        "ENSG00000269783",
        "ENSG00000269790",
        "ENSG00000269791",
        "ENSG00000269795",
    ],
]

MODEL_PRECISION: str = "bf16-mixed"
USE_TE: bool = True
TARGET_MEAN_LOSS: float = 2.442256450653076


@pytest.fixture()
def cells() -> List[List[str]]:
    return deepcopy(CELLS_FOR_TEST)


@pytest.fixture
def geneformer_config():
    autocast_dtype = get_autocast_dtype(MODEL_PRECISION)
    return GeneformerConfig(
        model_cls=GeneformerModel,
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=2048,
        fp16=autocast_dtype == torch.float16,  # normally handled by ptl precision plugin
        bf16=autocast_dtype == torch.bfloat16,  # normally handled by ptl precision plugin
        params_dtype=autocast_dtype,
        pipeline_dtype=autocast_dtype,
        autocast_dtype=autocast_dtype,  # setting this speeds things up a lot
        nemo1_ckpt_path=str(nemo1_checkpoint_path),
        return_only_hidden_states=True,  # This is what we did in nemo1 for inference
    )


def test_nemo1_nemo2_weight_shapes_match(geneformer_config, seed: int = 42):
    data_dir = Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_dir}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_dir}")

    with (
        tarfile.open(nemo1_checkpoint_path, "r") as old_ckpt,
        torch.no_grad(),
        megatron_parallel_state_utils.distributed_model_parallel_state(seed),
    ):
        ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
        old_weights = torch.load(ckpt_file)
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": _}:
                pass
            case _:
                assert False
        new_model = geneformer_config.configure_model(tokenizer)
        new_state_dict = new_model.state_dict_for_save_checkpoint()
        # Set the new_model_prefix to "" since we are looking at the base megatron model and not the lightning module which stores a copy of
        #  this model into self.module
        old_keys = {nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=USE_TE) for k in old_weights}
        assert len(old_keys) == len(old_weights), "Mapping unexpectedly discarded some keys."
        new_keys = set(new_state_dict)
        for k, v in old_weights.items():
            # Make sure the shapes of the weights match.
            assert (
                new_state_dict[nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=USE_TE)].shape
                == v.shape
            )
        extra_keys = new_keys - old_keys
        extra_non_null_keys = {
            # TE adds non-null ._extra_state objects to layers so skip those.
            k
            for k in extra_keys
            if new_state_dict[k] is not None and not k.endswith("_extra_state")
        }
        assert not extra_non_null_keys, "There are new keys that have state that is missing from the old checkpoint."
        missing_old_keys = old_keys - new_keys
        assert not missing_old_keys, "There are keys in the old checkpoint that are missing from the new model."


def _apply_tokenizer(tokenizer, sequences: List[List[str]], device) -> List[torch.Tensor]:
    # parent pulls the tokenizer from the loaded model.
    try:
        token_ids = [
            torch.tensor(
                [tokenizer.class_id] + [tokenizer.token_to_id(gene_symbol) for gene_symbol in gene_symbols],
                device=device,
                dtype=torch.long,
            )
            for gene_symbols in sequences
        ]
    except TypeError as e:
        invalid_tokens = {gene_symbol for gene_symbols in sequences for gene_symbol in gene_symbols} - set(
            tokenizer.vocab.keys()
        )
        raise ValueError(
            f"Unknown token in gene symbols. Please filter genes for those present in self.tokenizer:\n{invalid_tokens}"
        ) from e
    return token_ids


def _batched_tokenizer(
    tokenizer, sequences: List[List[str]], device, seq_length: int = 2048, dynamic_padding: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize sequences.
    Returns:
        token_ids (torch.Tensor, long): token ids
        mask (torch.Tensor, long, float): boolean mask for padded sections
    """
    token_ids = _apply_tokenizer(tokenizer=tokenizer, sequences=sequences, device=device)

    # Validate input sequences length
    if any(len(t) > seq_length for t in token_ids):
        raise ValueError(f"One or more sequence exceeds max length({seq_length}).")

    # Set fixed padding when dynamic padding is disabled
    if not dynamic_padding:
        padding_length = seq_length
    else:
        padding_length = None
    # Pad token ids (1/True = Active, 0/False = Inactive)
    token_ids, mask = pad_token_ids(
        token_ids,
        padding_value=tokenizer.pad_id,
        padding_len=padding_length,
        device=device,
    )

    return token_ids, mask


class _DummyDataSet(torch.utils.data.Dataset):
    def __init__(self, cells: List[List[str]], tokenizer):
        input_ids, mask = _batched_tokenizer(tokenizer, cells, device=torch.device("cuda"))
        self.input_ids = input_ids
        self.mask = mask
        assert len(self.input_ids) == len(self.mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"text": self.input_ids[idx], "attention_mask": self.mask[idx]}


def test_geneformer_nemo1_v_nemo2_inference_golden_values(
    geneformer_config: GeneformerConfig, cells: List[List[str]], seed: int = 42
):
    """NOTE: this test is against old nemo1 inference golden values. It may be deprecated in the future as we move away from nemo1.
      This test documents _how_ different the two models are at the moment.
    Original model summary:
    BertModel(
    (language_model): TransformerLanguageModel(
        (embedding): Embedding(
            (word_embeddings): VocabParallelEmbedding()
            (position_embeddings): Embedding(2048, 256)
            (embedding_dropout): Dropout(p=0.02, inplace=False)
        )
        (encoder): ParallelTransformer(
        (layers): ModuleList(
            (0-5): 6 x ParallelTransformerLayer(
            (input_layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
            (self_attention): ParallelAttention(
                (query_key_value): ColumnParallelLinear()
                (core_attention): CoreAttention(
                (scale_mask_softmax): MatchedScaleMaskSoftmax()
                (attention_dropout): Dropout(p=0.1, inplace=False)
                )
                (dense): RowParallelLinear()
            )
            (post_attention_layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
            (mlp): ParallelMLP(
                (dense_h_to_4h): ColumnParallelLinear()
                (dense_4h_to_h): RowParallelLinear()
            )
            )
        )
        (final_layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
        )
    )
    (lm_head): BertLMHead(
        (dense): Linear(in_features=256, out_features=256, bias=True)
        (layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
    )
    )

    New model summary:
    MegatronBioBertModel(
    (embedding): LanguageModelEmbedding(
        (word_embeddings): VocabParallelEmbedding()
        (position_embeddings): Embedding(2048, 256)
        (embedding_dropout): Dropout(p=0.02, inplace=False)
    )
    (encoder): TransformerBlock(
        (layers): ModuleList(
        (0-5): 6 x TransformerLayer(
            (input_layernorm): FusedLayerNorm()
            (self_attention): SelfAttention(
            (core_attention): DotProductAttention(
                (scale_mask_softmax): FusedScaleMaskSoftmax()
                (attention_dropout): Dropout(p=0.1, inplace=False)
            )
            (linear_proj): RowParallelLinear()
            (linear_qkv): ColumnParallelLinear()
            (q_layernorm): IdentityOp()
            (k_layernorm): IdentityOp()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): FusedLayerNorm()
            (mlp): MLP(
            (linear_fc1): ColumnParallelLinear()
            (linear_fc2): RowParallelLinear()
            )
        )
        )
        (final_layernorm): LayerNorm()
    )
    (lm_head): BertLMHead(
        (dense): Linear(in_features=256, out_features=256, bias=True)
        (layer_norm): FusedLayerNorm()
    )
    (output_layer): ColumnParallelLinear()
    )


    """

    assert nemo_1_expected_values_path.exists(), f"Could not find expected values at {nemo_1_expected_values_path}."

    data_dir = Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_dir}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_dir}")

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": _}:
            pass
        case _:
            assert False

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        data_sampler=nl.MegatronDataSampler(
            micro_batch_size=3,
            global_batch_size=3,
            seq_len=16,
            output_log=False,  # this is needed for predict step to work
        ),
    )
    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION),
    )
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=1e-4,
            optimizer="adam",
            use_distributed_optimizer=True,
            fp16=geneformer_config.fp16,
            bf16=geneformer_config.bf16,
        )
    )
    module = biobert_lightning_module(config=geneformer_config, tokenizer=tokenizer, optimizer=optimizer)

    dataloader = torch.utils.data.DataLoader(_DummyDataSet(cells, tokenizer), batch_size=3, num_workers=0)
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        result = torch.cat(trainer.predict(module, dataloaders=dataloader), dim=1).transpose(1, 0).contiguous()
    assert len(result) == 3
    expected_vals = {k: v.to(result.device) for k, v in torch.load(nemo_1_expected_values_path).items()}
    assert_matrix_mape_below_value(
        result,
        expected_vals["expected_hidden_state"],
        mask=expected_vals["expected_pad_masks"],
        eps=0.1,
        max_mape=2.5,  # 2.5% average difference in final values with a magnitude over 0.1
    )
    assert_matrix_correlation_above_value(
        result,
        expected_vals["expected_hidden_state"],
        mask=expected_vals["expected_pad_masks"],
        min_correlation=0.9999,
    )


@pytest.mark.skipif(USE_TE, reason="This per-layer test does not yet support TE mapping.")
def test_geneformer_inference_nemo1_v_nemo2_golden_values_by_layer(
    geneformer_config: GeneformerConfig, cells: List[List[str]], seed: int = 42
):
    """NOTE: this test is against old nemo1 inference golden values. It may be deprecated in the future as we move away from nemo1.
    This test documents _how_ different the two models are at the moment at each layer, and highlights which layers are the most
    different. This test is useful for debugging and understanding the differences between the two models.
    """
    assert nemo_1_per_layer_outputs_path.exists(), (
        f"Could not find per-layer expected values at {nemo_1_per_layer_outputs_path}."
    )
    data_dir = Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_dir}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_dir}")

    with (
        tarfile.open(nemo1_checkpoint_path, "r") as old_ckpt,
        torch.inference_mode(),
        megatron_parallel_state_utils.distributed_model_parallel_state(seed),
    ):
        ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
        old_weights = torch.load(ckpt_file)
        new_state_dict_from_old = {}
        for k, v in old_weights.items():
            new_key = nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=USE_TE)
            new_v = v
            new_state_dict_from_old[new_key] = new_v
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": _}:
                pass
            case _:
                assert False
        new_model = geneformer_config.configure_model(tokenizer).eval().cuda()
        # TE adds non-null ._extra_state objects to layers, which store some kind of buffer bits
        #  so we need to allow those to pass through.
        new_model.load_state_dict(new_state_dict_from_old, strict=not USE_TE)
        for k, v in new_model.state_dict().items():
            # Make sure the weights were properly loaded
            if v is not None and not k.endswith("_extra_state"):
                torch.testing.assert_close(new_state_dict_from_old[k], v, check_dtype=False, check_device=False)
            else:
                assert k.endswith("_extra_state")

        input_ids, mask = _batched_tokenizer(tokenizer, cells, device=torch.device("cuda"))

        # with torch.autocast(device_type="cuda", dtype=get_autocast_dtype("bf16-mixed")):
        # new_model = new_model.bfloat16()  # if we move to the lightning way of calling forward we can drop this
        new_model.post_process = False  # so we get hidden states rather than logits
        new_model.encoder.post_process = True
        new_model.encoder.post_layer_norm = True

        # normally handled by ptl precision plugin
        new_model = (
            Float16Module(new_model.config, new_model)
            if new_model.config.autocast_dtype in {torch.float16, torch.bfloat16}
            else new_model
        )

        new_outputs = {}
        from functools import partial

        def register_hooks(model, hook_fn):
            for name, module in model.named_modules():
                module.register_forward_hook(partial(hook_fn, name))

        def hook_fn(name, module, input, output):
            new_outputs[name] = (str(type(module)), input, output)

        register_hooks(new_model, hook_fn)
        # Fill up the new_outputs
        # with torch.autocast(enabled=geneformer_config.enable_autocast, dtype=geneformer_config.autocast_dtype, device_type="cuda"):
        _ = new_model(input_ids, mask)
        ori_outputs = torch.load(nemo_1_per_layer_outputs_path)

        # Test settings for MAPE https://en.wikipedia.org/wiki/Mean_absolute_percentage_error thresholds
        softmax_mape_threshold = 9.88
        mape_tolerances = {
            "module.encoder.layers.0.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "module.encoder.layers.0.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "module.encoder.layers.1.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "module.encoder.layers.1.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "module.encoder.layers.2.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "module.encoder.layers.2.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "module.encoder.layers.3.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "module.encoder.layers.3.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "module.encoder.layers.4.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "module.encoder.layers.4.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "module.encoder.layers.5.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "module.encoder.layers.5.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "module.encoder.layers.4.pre_mlp_layernorm": 3.6,
            "module.encoder.layers.5.input_layernorm": 3.6,
            "module.encoder.layers.5.pre_mlp_layernorm": 4.1,
        }
        default_mape_tolerance = 3.3  # 3.3% difference in larger magnitude values with values over a magnitude of 0.1

        # Test settings for correlation https://en.wikipedia.org/wiki/Pearson_correlation_coefficient thresholds
        correlation_tolerances = {
            "module.encoder.layers.0.self_attention.core_attention.scale_mask_softmax": 0.985,
            "module.encoder.layers.0.self_attention.core_attention.attention_dropout": 0.985,
            "module.encoder.layers.1.self_attention.core_attention.scale_mask_softmax": 0.975,
            "module.encoder.layers.1.self_attention.core_attention.attention_dropout": 0.975,
            "module.encoder.layers.2.self_attention.core_attention.scale_mask_softmax": 0.975,
            "module.encoder.layers.2.self_attention.core_attention.attention_dropout": 0.975,
            "module.encoder.layers.3.self_attention.core_attention.scale_mask_softmax": 0.975,
            "module.encoder.layers.3.self_attention.core_attention.attention_dropout": 0.975,
            "module.encoder.layers.4.self_attention.core_attention.scale_mask_softmax": 0.96,
            "module.encoder.layers.4.self_attention.core_attention.attention_dropout": 0.96,
            "module.encoder.layers.5.self_attention.core_attention.scale_mask_softmax": 0.925,
            "module.encoder.layers.5.self_attention.core_attention.attention_dropout": 0.925,
        }
        default_correlation_tolerance = (
            0.9998 if new_model.config.autocast_dtype == torch.float32 else 0.99
        )  # 0.9999 correlation for final layer

        mask_t = mask.transpose(1, 0).contiguous()
        mask = mask[..., None]
        mask_t = mask_t[..., None]
        for module_name, (ori_cls_name, _, ori_output) in ori_outputs.items():
            new_module_name = nemo1_to_nemo2_biobert_key_mapping(
                module_name, new_model_prefix="module", te_mapping=USE_TE
            )
            if new_module_name == "module.language_model":
                new_module_name = "module.encoder"
            if new_module_name == "model":
                new_module_name = ""
            new_cls_name, _, new_output = new_outputs[new_module_name]
            if new_module_name == "" and module_name == "":
                new_output = new_output.transpose(0, 1).contiguous()
            if isinstance(ori_output, (tuple, list)) or isinstance(new_output, (tuple, list)):
                if isinstance(ori_output, (tuple, list)):
                    ori_output = [o for o in ori_output if o is not None]
                else:
                    ori_output = [ori_output]
                if isinstance(new_output, (tuple, list)):
                    new_output = [o for o in new_output if o is not None]
                else:
                    new_output = [new_output]
                assert type(ori_output) is type(new_output)
                assert len(ori_output) == len(new_output)
                for ori, new in zip(ori_output, new_output):
                    if ori is None and new is None:
                        continue
                    if ori is None or new is None:
                        assert False, f"One of the outputs is None, but the other is not. {ori}, {new}"
                    assert ori.shape == new.shape
                    if ori.shape[0:2] == (16, 3):
                        _mask = mask_t
                    elif ori.shape[0:2] == (3, 16):
                        _mask = mask
                    else:
                        _mask = None
                    assert_matrix_mape_below_value(
                        new,
                        ori,
                        mask=_mask,
                        max_mape=mape_tolerances.get(new_module_name, default_mape_tolerance),
                        eps=1e-1,
                        msg=f"Module: {new_module_name}",
                    )
                    assert_matrix_correlation_above_value(
                        new,
                        ori,
                        mask=_mask,
                        min_correlation=correlation_tolerances.get(new_module_name, default_correlation_tolerance),
                        msg=f"Module: {new_module_name}",
                    )
            else:
                if new_output.shape[0:2] == (16, 3):
                    _mask = mask_t
                elif new_output.shape[0:2] == (3, 16):
                    _mask = mask
                else:
                    _mask = None
                assert_matrix_mape_below_value(
                    new_output,
                    ori_output,
                    mask=_mask,
                    eps=1e-1,
                    max_mape=mape_tolerances.get(new_module_name, default_mape_tolerance),
                    msg=f"Module: {new_module_name}",
                )
                assert_matrix_correlation_above_value(
                    new_output,
                    ori_output,
                    mask=_mask,
                    min_correlation=correlation_tolerances.get(new_module_name, default_correlation_tolerance),
                    msg=f"Module: {new_module_name}",
                )


def _get_loss_from_model(model_config: GeneformerConfig, seed: int) -> float:
    """Shared test utility that we can use for a positive and negative control on the loss from our loaded checkpoint."""
    data_dir = Path(data_path)
    train_data_path = data_dir / "train"

    with (
        torch.inference_mode(),
        megatron_parallel_state_utils.distributed_model_parallel_state(seed),
        random_numpy_context(seed),
    ):
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": median_dict}:
                pass
            case _:
                assert False
        new_model = model_config.configure_model(tokenizer).eval().cuda()
        # normally handled by ptl precision plugin
        new_model = (
            Float16Module(new_model.config, new_model)
            if new_model.config.autocast_dtype in {torch.float16, torch.bfloat16}
            else new_model
        )

        # NOTE: a small change to randomization in the single-cell dataset could throw our test below off by a small amount
        #  maybe 0.02 or so, if the value is above that range then disable the 200 batch limit and check the global number
        #  going back to `n += 1` and `loss += F.cross_entropy(logits[loss_mask], target[loss_mask], reduction="mean")`
        #  for consistency with the old results. Then if those look good, redefine the target with our seeds and the
        #  updated dataset.
        from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule

        micro_batch_size, devices = 8, 1

        data_module = SingleCellDataModule(
            train_dataset_path=data_path / "train",
            val_dataset_path=data_path / "val",
            test_dataset_path=data_path / "test",
            tokenizer=tokenizer,
            seq_length=2048,
            random_token_prob=0.1,
            median_dict=median_dict,
            micro_batch_size=micro_batch_size,
            global_batch_size=micro_batch_size * devices,
            mask_prob=0.15,
            mask_token_prob=0.8,
            seed=42,
        )

        limit_batches = 200
        data_module.trainer = mock.Mock()
        data_module.trainer.max_epochs = 1
        data_module.trainer.max_steps = 10
        data_module.trainer.val_check_interval = 2
        data_module.trainer.limit_val_batches = limit_batches
        data_module.trainer.limit_test_batches = limit_batches

        data_module.setup()
        dl = data_module.test_dataloader()

        loss = 0
        n = 0
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            # with torch.autocast(enabled=model_config.enable_autocast, dtype=model_config.autocast_dtype, device_type="cuda"):
            result = new_model(
                input_ids=batch["text"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            )
            loss_mask = batch["loss_mask"].cuda()
            # token_logits is s,b and for simplicity here let's transpose to b,s. In general this reduces performance.
            logits = result["token_logits"].transpose(0, 1).contiguous()
            target = batch["labels"].cuda()

            loss += F.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
            n += loss_mask.sum()

            if limit_batches is not None and i + 1 >= limit_batches:
                break

        mean_loss: float = (loss / n).cpu().numpy().item()
    return mean_loss


def test_inference_loss_10m_released_checkpoint(geneformer_config: GeneformerConfig, seed: int = 42):
    """Test that we get a good loss when loading a bionemo1 checkpoint with a properly initialized config"""
    geneformer_config_logit = deepcopy(geneformer_config)
    # Set up the model to return logits and switch to the released 10M checkpoint
    geneformer_config_logit.set_hparam("return_only_hidden_states", False)  # return logits
    geneformer_config_logit.set_hparam(
        "nemo1_ckpt_path", nemo1_release_checkpoint_path
    )  # release checkpoint is important

    mean_loss = _get_loss_from_model(geneformer_config_logit, seed)

    # NOTE: the values in the table were from the average of averages of 8 sized batches
    # Experiment 1) loaded the 10M model and did the mean of mean loss with 8 sized batches
    #  this gives: 2.3558831214904785 vs 2.357126723703872, so we actually do better!
    # For NVIDIA employees see work here:
    #   https://docs.google.com/document/d/1CofamqHbQlp5U8SjmW7NR7PbTbF72Lj9L9xz1W5t3ZI/edit
    # Experiment 2)
    #  With a proper loss (sum/n) and limiting to 200 _random_ batches of size 8 for speed
    #  we get a similar range number of 2.368649959564209.
    #  a small change that has lower impact than the change between models is probably acceptable.
    #  the target is defined as described above for the 10M checkpoint based on our first pass
    #  of the megatron implementation. Since we manually passed experiment 1 this experiment
    #  will define our initial "golden value" test target.
    assert mean_loss < TARGET_MEAN_LOSS or mean_loss == pytest.approx(TARGET_MEAN_LOSS, abs=1e-2, rel=None)


def test_inference_loss_10m_nemo2_released_checkpoint(geneformer_config: GeneformerConfig, seed: int = 42):
    """Test that we get a good loss when loading a bionemo1 checkpoint with a properly initialized config"""
    geneformer_config_logit = deepcopy(geneformer_config)
    # Set up the model to return logits and switch to the released 10M checkpoint
    geneformer_config_logit.set_hparam("return_only_hidden_states", False)  # return logits
    geneformer_config_logit.set_hparam(
        "initial_ckpt_path", nemo2_release_checkpoint_path
    )  # release checkpoint is important

    mean_loss = _get_loss_from_model(geneformer_config_logit, seed)

    # NOTE: the values in the table were from the average of averages of 8 sized batches
    # Experiment 1) loaded the 10M model and did the mean of mean loss with 8 sized batches
    #  this gives: 2.3558831214904785 vs 2.357126723703872, so we actually do better!
    # For NVIDIA employees see work here:
    #   https://docs.google.com/document/d/1CofamqHbQlp5U8SjmW7NR7PbTbF72Lj9L9xz1W5t3ZI/edit
    # Experiment 2)
    #  With a proper loss (sum/n) and limiting to 200 _random_ batches of size 8 for speed
    #  we get a similar range number of 2.368649959564209.
    #  a small change that has lower impact than the change between models is probably acceptable.
    #  the target is defined as described above for the 10M checkpoint based on our first pass
    #  of the megatron implementation. Since we manually passed experiment 1 this experiment
    #  will define our initial "golden value" test target.
    assert mean_loss < TARGET_MEAN_LOSS or mean_loss == pytest.approx(TARGET_MEAN_LOSS, abs=1e-2, rel=None)


def test_inference_loss_10m_released_checkpoint_wrong_activation(geneformer_config: GeneformerConfig, seed: int = 42):
    """Test that when we use the wrong activation we get worse loss out of the same function we test for a positive
    signal. This acts as the negative control.
    """
    geneformer_config_logit = deepcopy(geneformer_config)
    # Set up the model to return logits and switch to the released 10M checkpoint
    geneformer_config_logit.set_hparam("return_only_hidden_states", False)  # return logits
    geneformer_config_logit.set_hparam(
        "nemo1_ckpt_path", nemo1_release_checkpoint_path
    )  # release checkpoint is important

    # introduce a breaking change with a future xfail as a negative control for our test
    geneformer_config_logit.set_hparam("activation_func", torch.nn.functional.relu)  # the model should be gelu
    geneformer_config_logit.set_hparam("bias_activation_fusion", False)  # this needs to be off for ReLu support

    mean_loss = _get_loss_from_model(geneformer_config_logit, seed)
    # In one run, this gave a mean_loss of 7.9! Very much broke the model.
    #  note that the model can be trained to work with relu and produces similar loss curves
    #  but the weights trained one way are not compatible with the other.
    # Our HF model was at 3, so 5 is pretty clearly out of expected range. This shows how
    #  sensitive the checkpoint is to a real change in the underlying function.
    #  Perhaps a future model is more robust, so if this value needs to come down we can
    #  do that.
    assert mean_loss > 5


def _train_model_get_ckpt(
    name: str,
    root_dir: Path,
    config: GeneformerConfig,
    n_steps_train: int,
    batch_size: int,
    peft: PEFT | None = None,
    lr: float = 5e-4,
) -> Tuple[Path, MetricTracker, nl.Trainer]:
    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    data_dir = Path(data_path)
    train_data_path = data_dir / "train"
    val_data_path = data_dir / "val"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_error_str}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": median_dict}:
            pass
        case _:
            assert False

    data_module = SingleCellDataModule(
        tokenizer=tokenizer,
        median_dict=median_dict,
        train_dataset_path=str(val_data_path),  # Val has 3 samples, so this will result in good overfitting.
        val_dataset_path=str(val_data_path),
        test_dataset_path=str(val_data_path),
        random_token_prob=0.1,
        micro_batch_size=batch_size,
        global_batch_size=batch_size,
        num_workers=0,
        persistent_workers=False,
    )

    checkpoint_callback = nl_callbacks.ModelCheckpoint(
        save_last=True,
        save_on_train_epoch_end=True,
        monitor="val_loss",
        always_save_context=True,  # Enables the .nemo file-like checkpointing where all IOMixins are under SerDe
    )
    save_dir = root_dir / name
    tb_logger = TensorBoardLogger(save_dir=save_dir, name=name)
    # Setup the logger and train the model
    nemo_logger = NeMoLogger(
        log_dir=str(root_dir),
        name=name,
        tensorboard=tb_logger,
        ckpt=checkpoint_callback,
    )
    # Needed so that the trainer can find an output directory for the profiler
    # ckpt_path needs to be a string for SerDe
    optimizer = MegatronOptimizerModule(
        config=OptimizerConfig(
            lr=lr,
            optimizer="adam",
            use_distributed_optimizer=True,
            fp16=config.fp16,
            bf16=config.bf16,
        ),
        lr_scheduler=WarmupPolicyScheduler(
            max_steps=n_steps_train,
            # minimum learning rate is 1/100th of the initial learning rate, so eg lr=1e-3 -> min_lr=1e-5
            min_lr=lr / 100,
            warmup_steps=math.ceil(n_steps_train * 0.05),
            interval="step",
            frequency=1,
        ),
    )
    module = biobert_lightning_module(config=config, tokenizer=tokenizer, optimizer=optimizer, model_transform=peft)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        always_save_context=True,
    )
    metric_tracker = MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"])
    callbacks = [metric_tracker]
    if peft is not None:
        callbacks.append(ModelTransform())
    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        limit_val_batches=2,  # need to upsample to an integer limit
        limit_test_batches=2,  # need to upsample to an integer limit
        val_check_interval=n_steps_train // 2,
        max_steps=n_steps_train,
        num_nodes=1,
        log_every_n_steps=n_steps_train // 2,
        callbacks=callbacks,
        plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION),
    )
    nllm.train(
        model=module,
        data=data_module,
        trainer=trainer,
        log=nemo_logger,
        resume=resume.AutoResume(
            resume_if_exists=True,  # Looks for the -last checkpoint to continue training.
            resume_ignore_no_checkpoint=True,  # When false this will throw an error with no existing checkpoint.
        ),
    )
    ckpt_dirpath = Path(checkpoint_callback.last_model_path.replace(".ckpt", ""))
    return ckpt_dirpath, metric_tracker, trainer


@pytest.mark.needs_gpu
def test_continue_from_checkpoint_geneformer(
    tmpdir, geneformer_config: GeneformerConfig, n_layers_test: int = 3, n_steps_train: int = 100, batch_size: int = 16
):
    base_geneformer_config = io.reinit(geneformer_config)  # generate a new copy by calling the cached init.

    # Modify both the variable and associated saved init hyper-param by calling config.mutate(...)
    base_geneformer_config.set_hparam("return_only_hidden_states", False)
    base_geneformer_config.set_hparam("nemo1_ckpt_path", None)
    base_geneformer_config.set_hparam("num_layers", n_layers_test)  # set to 3 layers
    base_geneformer_config.set_hparam("hidden_size", 128)
    base_geneformer_config.set_hparam("ffn_hidden_size", 256)
    # Turn off dropout for this quick test
    base_geneformer_config.set_hparam("attention_dropout", 0.0)
    base_geneformer_config.set_hparam("hidden_dropout", 0.0)
    # Re-initialize after manually updating hidden_size/ffn_hidden_size since so many other parameters
    #  are based off of these parameters and modified in post_init of the transformer config.
    base_geneformer_config = io.reinit(base_geneformer_config)
    assert base_geneformer_config.num_layers == n_layers_test
    assert base_geneformer_config.nemo1_ckpt_path is None
    assert not base_geneformer_config.return_only_hidden_states
    with megatron_parallel_state_utils.distributed_model_parallel_state(32):
        ckpt_path, initial_metrics, initial_trainer = _train_model_get_ckpt(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            config=base_geneformer_config,
            n_steps_train=n_steps_train,
            batch_size=batch_size,
            lr=1e-4,  # smaller LR so smooth initial dip
        )
        weights_ckpt = ckpt_path / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert initial_trainer.model.config.num_layers == n_layers_test
        # Make sure the loss dropped initially
        assert sum(initial_metrics.collection_train["loss"][:5]) > sum(initial_metrics.collection_train["loss"][-5:])
    with megatron_parallel_state_utils.distributed_model_parallel_state(43):
        # NOTE all other hparams will be pulled from this checkpoint.
        update_base_geneformer_config = GeneformerConfig(
            initial_ckpt_path=str(ckpt_path),
        )
        continue_checkpoint, continue_metrics, continue_trainer = _train_model_get_ckpt(
            name="test_experiment_continue",
            root_dir=tmpdir / "continue_training",  # new checkpoint will land in a subdir of this
            config=update_base_geneformer_config,  # same config as before since we are just continuing training
            n_steps_train=n_steps_train,
            batch_size=batch_size,
            lr=5e-4,  # Larger LR so loss dips faster after the first stage
        )
        weights_ckpt = ckpt_path / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert continue_trainer.model.config.num_layers == n_layers_test
        # Make sure that loss is dropping at continue
        assert sum(continue_metrics.collection_train["loss"][:10]) > sum(
            continue_metrics.collection_train["loss"][-10:]
        )
        # Make sure the loss at the beginning of continue is a bit better than the end of initial
        assert sum(continue_metrics.collection_train["loss"][:10]) < sum(
            initial_metrics.collection_train["loss"][-10:]
        )


@pytest.mark.needs_gpu
def test_finetune_geneformer(
    tmpdir, geneformer_config: GeneformerConfig, n_layers_test: int = 3, n_steps_train: int = 100, batch_size: int = 16
):
    base_geneformer_config = io.reinit(geneformer_config)  # generate a new copy by calling the cached init.

    # Modify both the variable and associated saved init hyper-param by calling config.mutate(...)
    base_geneformer_config.set_hparam("return_only_hidden_states", False)
    base_geneformer_config.set_hparam("nemo1_ckpt_path", None)
    base_geneformer_config.set_hparam("num_layers", n_layers_test)  # set to 3 layers
    base_geneformer_config.set_hparam("hidden_size", 128)
    base_geneformer_config.set_hparam("ffn_hidden_size", 256)
    # Turn off dropout for this quick test
    base_geneformer_config.set_hparam("attention_dropout", 0.0)
    base_geneformer_config.set_hparam("hidden_dropout", 0.0)
    # Re-initialize after manually updating hidden_size/ffn_hidden_size since so many other parameters
    #  are based off of these parameters and modified in post_init of the transformer config.
    base_geneformer_config = io.reinit(base_geneformer_config)
    assert base_geneformer_config.num_layers == n_layers_test
    assert base_geneformer_config.nemo1_ckpt_path is None
    assert not base_geneformer_config.return_only_hidden_states
    with megatron_parallel_state_utils.distributed_model_parallel_state(32):
        ckpt_path, initial_metrics, initial_trainer = _train_model_get_ckpt(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            config=base_geneformer_config,
            n_steps_train=n_steps_train,
            batch_size=batch_size,
            lr=5e-4,
        )
        weights_ckpt = ckpt_path / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert initial_trainer.model.config.num_layers == n_layers_test
        # Make sure we're training
        assert sum(initial_metrics.collection_train["loss"][:10]) > sum(initial_metrics.collection_train["loss"][-10:])
    with megatron_parallel_state_utils.distributed_model_parallel_state(43):
        ft_geneformer_config = FineTuneSeqLenBioBertConfig(
            # All other hparams will be pulled from this checkpoint, aside from those in `override_parent_fields``
            initial_ckpt_path=str(ckpt_path),
        )
        simple_ft_checkpoint, simple_ft_metrics, ft_trainer = _train_model_get_ckpt(
            name="finetune_new_head",
            root_dir=tmpdir / "finetune_new_head",  # new checkpoint will land in a subdir of this
            config=ft_geneformer_config,  # same config as before since we are just continuing training
            n_steps_train=n_steps_train,
            batch_size=batch_size,
            lr=5e-3,
        )
        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert ft_trainer.model.config.num_layers == n_layers_test
        assert sum(simple_ft_metrics.collection_train["loss"][:10]) > sum(
            simple_ft_metrics.collection_train["loss"][-10:]
        )


@pytest.mark.needs_gpu
@pytest.mark.skip(reason="PEFT currently broken with fusions activated.")
def test_finetune_geneformer_with_peft(
    tmpdir, geneformer_config: GeneformerConfig, n_layers_test: int = 3, n_steps_train: int = 100, batch_size: int = 16
):
    base_geneformer_config = io.reinit(geneformer_config)  # generate a new copy by calling the cached init.

    # Modify both the variable and associated saved init hyper-param by calling config.mutate(...)
    base_geneformer_config.set_hparam("return_only_hidden_states", False)
    base_geneformer_config.set_hparam("nemo1_ckpt_path", None)
    base_geneformer_config.set_hparam("num_layers", n_layers_test)  # set to 3 layers
    base_geneformer_config.set_hparam("hidden_size", 128)
    base_geneformer_config.set_hparam("ffn_hidden_size", 256)
    # Turn off dropout for this quick test
    base_geneformer_config.set_hparam("attention_dropout", 0.0)
    base_geneformer_config.set_hparam("hidden_dropout", 0.0)
    # Re-initialize after manually updating hidden_size/ffn_hidden_size since so many other parameters
    #  are based off of these parameters and modified in post_init of the transformer config.
    base_geneformer_config = io.reinit(base_geneformer_config)
    assert base_geneformer_config.num_layers == n_layers_test
    assert base_geneformer_config.nemo1_ckpt_path is None
    assert not base_geneformer_config.return_only_hidden_states
    with megatron_parallel_state_utils.distributed_model_parallel_state(32):
        ckpt_path, initial_metrics, initial_trainer = _train_model_get_ckpt(
            name="test_experiment",
            root_dir=tmpdir / "pretrain",
            config=base_geneformer_config,
            n_steps_train=n_steps_train,
            batch_size=batch_size,
            lr=5e-4,
        )
        weights_ckpt = ckpt_path / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert initial_trainer.model.config.num_layers == n_layers_test
        assert sum(initial_metrics.collection_train["loss"][:10]) > sum(initial_metrics.collection_train["loss"][-10:])
    with megatron_parallel_state_utils.distributed_model_parallel_state(43):
        ft_geneformer_config = FineTuneSeqLenBioBertConfig(
            # All other hparams will be pulled from this checkpoint, aside from those in `override_parent_fields``
            initial_ckpt_path=str(ckpt_path),
        )
        peft = LoRAForGeneFormerTokenRegressor()
        simple_ft_checkpoint, simple_ft_metrics, ft_trainer = _train_model_get_ckpt(
            name="finetune_new_head",
            root_dir=tmpdir / "finetune_new_head",  # new checkpoint will land in a subdir of this
            config=ft_geneformer_config,  # same config as before since we are just continuing training
            n_steps_train=n_steps_train,
            batch_size=batch_size,
            peft=peft,
            lr=5e-3,
        )
        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert ft_trainer.model.config.num_layers == n_layers_test
        assert sum(simple_ft_metrics.collection_train["loss"][:10]) > sum(
            simple_ft_metrics.collection_train["loss"][-10:]
        )

        model = ft_trainer.model[0].module.module.module
        assert all(not p.requires_grad for p in model.embedding.parameters())
        assert all(not p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" not in name)
        assert all(p.requires_grad for name, p in model.encoder.named_parameters() if "adapter" in name)
        assert all(p.requires_grad for p in model.regression_head.parameters())
