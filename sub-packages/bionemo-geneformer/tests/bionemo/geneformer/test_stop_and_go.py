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
"""
How to adapt these tests:

1) Need to hook up our own pretraining workflow. In the code below, we do this via subproc and CLI. Is this still best practice?
    a) use the structure in sub-packages/bionemo-geneformer/tests/bionemo/geneformer/test_model.py:test_geneformer_nemo1_v_nemo2_inference_golden_values
    b) might need to look at utilities for setup/teardown to make sure the distributed stuff is handled correctly.
2) Need to inject the callbacks either via CLI or by manually inserting them here.
3) How do we want this to work for other modules? Lots of code could be duplicated here which makes it a challenge.
4) is this the right set of code to do this on?

"""

import math
import pathlib
from typing import Literal

import lightning.pytorch as pl
import torch
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from nemo import lightning as nl
from nemo.lightning.pytorch.optim.lr_scheduler import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from typing_extensions import override

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.geneformer.api import GeneformerConfig
from bionemo.geneformer.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import testing_callbacks
from bionemo.testing.harnesses import stop_and_go
from bionemo.testing.harnesses.mode import Mode


DATA_PATH: pathlib.Path = load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl"

MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"
SEQ_LEN: int = 1024


def geneformer_config():
    """Setups the default geneformer config taken from pretrain.py. Update as needed."""
    autocast_dtype = get_autocast_dtype(MODEL_PRECISION)
    return GeneformerConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        # FIXME: for now the test doesn't work unless dropout is inactivated because the megatron rng state is not saved
        attention_dropout=0,
        hidden_dropout=0,
        seq_length=SEQ_LEN,
        fp16=autocast_dtype == torch.float16,
        bf16=autocast_dtype == torch.bfloat16,
    )


def geneformer_datamodule(tokenizer, seq_length, median_dict, data_path=DATA_PATH):
    from bionemo.geneformer.data.singlecell.datamodule import SingleCellDataModule

    num_dataset_workers = 0
    data = SingleCellDataModule(
        seq_length=seq_length,
        tokenizer=tokenizer,
        train_dataset_path=data_path / "train",
        val_dataset_path=data_path / "val",
        test_dataset_path=data_path / "test",
        random_token_prob=0.1,  # this is the incorrect setting we originally used.
        median_dict=median_dict,
        micro_batch_size=2,
        global_batch_size=2 * 1,  # micro batch size times divices
        # persistent workers is supported when num_dataset_workers > 0
        persistent_workers=num_dataset_workers > 0,
        pin_memory=False,
        num_workers=num_dataset_workers,
    )
    return data


class TestGeneformerStopAndGo(stop_and_go.StopAndGoHarness):
    num_steps: int = 10
    val_check_interval: int = 4
    limit_val_batches: int = 2
    lr: float = 1e-4
    precision: Literal["16-mixed", "bf16-mixed", "32"] = MODEL_PRECISION
    output_tensor_atol: float = 3e-2

    @override
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # setup data
        train_data_path = DATA_PATH / "train"
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": median_dict}:
                cls.tokenizer, cls.median_dict = tokenizer, median_dict
            case _:
                raise ValueError("Preprocessing must have failed.")

        # add your custom callbacks here
        # cls.stop_callbacks["YourCustomCallback"] = YourCustomCallback(mode=Mode.STOP)
        # cls.go_callbacks["YourCustomCallback"] = YourCustomCallback(mode=Mode.RESUME)

        # run stop and go
        cls.run_stop_and_go()

    @override
    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        optim = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=cls.lr,
                optimizer="adam",
                use_distributed_optimizer=True,
            ),
            lr_scheduler=CosineAnnealingScheduler(
                max_steps=cls.num_steps,
                min_lr=cls.lr / 100,
                warmup_steps=math.ceil(cls.num_steps * 0.1),
                interval="step",
                monitor="reduced_train_loss",
                constant_steps=math.ceil(cls.num_steps * 0.1),
            ),
        )
        module = biobert_lightning_module(config=geneformer_config(), tokenizer=cls.tokenizer, optimizer=optim)

        data = geneformer_datamodule(
            tokenizer=cls.tokenizer,
            seq_length=SEQ_LEN,
            median_dict=cls.median_dict,
        )
        return module, data, optim

    def test_train_val_init_consumed_samples(self):
        """Tests the initial consumed samples in stop-and-go scenario."""
        train_consumed_stop, val_consumed_stop = stop_and_go.get_callback(
            self.callbacks, Mode.STOP, testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
        ).data
        train_consumed_go, val_consumed_go = stop_and_go.get_callback(
            self.callbacks, Mode.RESUME, testing_callbacks.TrainValInitConsumedSamplesStopAndGoCallback
        ).data

        assert val_consumed_stop == 0
        assert val_consumed_go == 0
        assert train_consumed_stop == 0
        assert train_consumed_go > 0
