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


import signal
from pathlib import Path
from typing import Literal

import lightning.pytorch as pl
import pytest
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from typing_extensions import override

from bionemo.core.data.load import load
from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.esm2.api import ESM2Config
from bionemo.esm2.data.datamodule import ESMDataModule
from bionemo.esm2.data.dataset import RandomMaskStrategy
from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer, get_tokenizer
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.llm.model.lr_scheduler import WarmupAnnealDecayHoldScheduler
from bionemo.testing import testing_callbacks
from bionemo.testing.harnesses import stop_and_go
from bionemo.testing.harnesses.mode import Mode
from bionemo.testing.torch import recursive_assert_approx_equal


MODEL_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"


class TestESM2StopAndGo(stop_and_go.StopAndGoHarness):
    num_steps: int = 10
    val_check_interval: int = 4
    limit_val_batches: int = 2
    lr: float = 1e-4
    precision: Literal["16-mixed", "bf16-mixed", "32"] = get_autocast_dtype(MODEL_PRECISION)

    @override
    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.data_dir = Path(cls.tempdir.name) / "data"
        cls.data_dir.mkdir(parents=True, exist_ok=True)

        # setup data
        data_dir = load("esm2/testdata_esm2_pretrain:2.0") / "2024_03_sanity"

        cls.train_cluster_path = data_dir / "train_clusters_sanity.parquet"
        cls.train_database_path = data_dir / "train_sanity.db"
        cls.valid_cluster_path = data_dir / "valid_clusters.parquet"
        cls.valid_database_path = data_dir / "validation.db"
        cls.tokenizer: BioNeMoESMTokenizer = get_tokenizer()

        # run stop and go
        cls.run_stop_and_go()

    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        # build data module
        data = ESMDataModule(
            train_cluster_path=cls.train_cluster_path,
            train_database_path=cls.train_database_path,
            valid_cluster_path=cls.valid_cluster_path,
            valid_database_path=cls.valid_database_path,
            global_batch_size=2,
            micro_batch_size=2,
            min_seq_length=None,
            max_seq_length=1024,
            num_workers=1,
            persistent_workers=False,
            random_mask_strategy=RandomMaskStrategy.ALL_TOKENS,
        )

        # build optimizer
        optimizer = MegatronOptimizerModule(
            config=OptimizerConfig(
                lr=cls.lr,
                optimizer="adam",  # fused_adam not supported
                use_distributed_optimizer=True,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.98,
            ),
            lr_scheduler=WarmupAnnealDecayHoldScheduler(
                warmup_steps=50, max_steps=cls.num_steps, max_lr=cls.lr, min_lr=0.0, anneal_percentage=0.10
            ),
        )

        # light ESM2 config
        config = ESM2Config(
            num_layers=3,
            hidden_size=128,
            params_dtype=cls.precision,
            pipeline_dtype=cls.precision,
            autocast_dtype=cls.precision,
        )
        # Build lightning module
        module = biobert_lightning_module(config=config, tokenizer=cls.tokenizer, optimizer=optimizer)

        return module, data, optimizer


class TestESM2StopAndGoCheckpointNotAtValidation(TestESM2StopAndGo):
    @override
    @classmethod
    def get_default_callbacks(cls):
        callbacks = super().get_default_callbacks()
        callbacks[Mode.STOP][nl_callbacks.PreemptionCallback] = nl_callbacks.PreemptionCallback(sig=signal.SIGUSR2)
        callbacks[Mode.STOP][testing_callbacks.SignalAfterGivenStepCallback] = (
            testing_callbacks.SignalAfterGivenStepCallback(stop_step=2, signal_=signal.SIGUSR2)
        )

        return callbacks

    @override
    @classmethod
    def stop(cls) -> None:
        # The PreemptionCallback exits the process with sys.exit(0) after the checkpoint is saved. We obviously don't
        # want that here, so we catch the SystemExit exception and make sure it was called appropriately.
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            super().stop()

        assert pytest_wrapped_e.type is SystemExit
        assert pytest_wrapped_e.value.code == 0

    @pytest.mark.parametrize(
        "callback_type",
        [
            testing_callbacks.LearningRateCallback,
            testing_callbacks.GlobalStepStateCallback,
            testing_callbacks.ConsumedSamplesCallback,
            testing_callbacks.OptimizerStateCallback,
            testing_callbacks.TrainInputCallback,
            testing_callbacks.TrainOutputCallback,
            testing_callbacks.TrainLossCallback,
            testing_callbacks.ValidInputCallback,
            testing_callbacks.ValidOutputCallback,
            testing_callbacks.ValidLossCallback,
        ],
    )
    def test_stop_and_go_consistency(self, callback_type):
        if callback_type in [
            testing_callbacks.ValidInputCallback,
            testing_callbacks.ValidLossCallback,
            testing_callbacks.ValidOutputCallback,
        ]:
            # On resumption from a checkpoint that wasn't created at the end of validation, the validation interval is
            # shifted in the subsequent training jobs. See this slack thread for more details:
            # https://nvidia.slack.com/archives/C074Z808N05/p1733171223813409
            pytest.xfail(
                reason="Currently seeing issues in validation timing with PreemptionCallback. "
                "See https://nvbugspro.nvidia.com/bug/4994415F."
            )
        super().test_stop_and_go_consistency(callback_type)

    @pytest.mark.skip(reason="We don't expect the STOP variant to hit on_valid_epoch_end before stopping.")
    def test_train_val_init_consumed_samples(self):
        pass

    def test_all_valid_batch_inputs_are_identical(self):
        """A watered-down version of test_stop_and_go_consistency's ValidInputCallback that only checks whether the
        first batches are the same, not the over length."""

        valid_inputs_interrupted = stop_and_go.get_callback(
            self.callbacks, Mode.RESUME, testing_callbacks.ValidInputCallback
        ).data
        valid_inputs_continuous = stop_and_go.get_callback(
            self.callbacks, Mode.CONTINUOUS, testing_callbacks.ValidInputCallback
        ).data

        min_len = min(len(valid_inputs_interrupted), len(valid_inputs_continuous))
        assert min_len
        recursive_assert_approx_equal(valid_inputs_interrupted[:min_len], valid_inputs_continuous[:min_len])
