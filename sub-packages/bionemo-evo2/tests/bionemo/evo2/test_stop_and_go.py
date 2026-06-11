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


from typing import Literal

import lightning.pytorch as pl
import pytest
import torch
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections.llm import HyenaModel
from nemo.collections.llm.gpt.data import MockDataModule
from nemo.collections.llm.gpt.model.hyena import HyenaNVTestConfig
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.lightning.pytorch.strategies import MegatronStrategy
from typing_extensions import override

from bionemo.evo2.models.mamba import HybridMambaConfig8BEvo2Loss, MambaModel
from bionemo.testing import testing_callbacks
from bionemo.testing.harnesses import stop_and_go
from bionemo.testing.harnesses.mode import Mode


class TestEvo2StopAndGo(stop_and_go.StopAndGoHarness):
    """Most of these parameters are copied from test_evo2.py which runs train.py."""

    num_steps: int = 3
    val_check_interval: int = 1
    limit_val_batches: int = 1
    lr: float = 3e-4
    wd: float = 0.01
    clip_grad: float = 1.0
    micro_batch_size: int = 1
    global_batch_size: int = 1
    num_layers: int = 4
    precision: Literal["16-mixed", "bf16-mixed", "32"] = "bf16-mixed"
    workers: int = 8
    seq_length: int = 8
    hybrid_override_pattern: str = "SDH*"
    use_megatron_comm_overlap_llama3_8k: bool = False
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    no_renormalize_loss: bool = False
    sequence_parallel: bool = False
    cross_entropy_loss_fusion: bool = False
    no_fp32_residual_connection: bool = False
    add_bias_output: bool = True

    @classmethod
    def setup_trainer(
        cls,
        mode: Mode,
    ) -> nl.Trainer:
        """Setup trainer by passing stop, resume, or continuous callbacks according to mode."""
        ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=False,
            align_param_gather=False,
            average_in_collective=True,
        )
        strategy = MegatronStrategy(
            ddp=ddp,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=cls.sequence_parallel,
            pipeline_dtype=torch.bfloat16,
            ckpt_async_save=False,
            ckpt_load_optimizer=True,
            ckpt_save_optimizer=True,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        )

        trainer = nl.Trainer(
            devices=1,
            max_steps=cls.num_steps,
            num_nodes=1,
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=cls.limit_val_batches,
            num_sanity_val_steps=0,
            val_check_interval=cls.val_check_interval,
            log_every_n_steps=cls.val_check_interval,
            enable_checkpointing=True,
            use_distributed_sampler=False,
            callbacks=list(cls.callbacks[mode].values()),
            plugins=nl.MegatronMixedPrecision(
                precision=cls.precision, params_dtype=torch.bfloat16, grad_reduce_in_fp32=False, fp8_wgrad=False
            ),
        )
        return trainer

    @override
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # setup data
        cls.tokenizer = get_nmt_tokenizer("byte-level")
        # run stop and go
        cls.run_stop_and_go()

    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        # build data module
        data = MockDataModule(
            seq_length=cls.seq_length,
            micro_batch_size=cls.micro_batch_size,
            global_batch_size=cls.global_batch_size,
            num_workers=cls.workers,
            tokenizer=cls.tokenizer,
        )

        data.init_global_step = 0
        # config
        config = HyenaNVTestConfig(
            **{
                "tp_comm_overlap": cls.use_megatron_comm_overlap_llama3_8k,
                "seq_length": cls.seq_length,
                "use_te": True,
                "params_dtype": torch.bfloat16,
                "bf16": True,
                "recompute_granularity": None,
                "recompute_method": None,
                "recompute_num_layers": None,
                "num_layers": cls.num_layers,
                "hidden_size": 1920,
                "hybrid_override_pattern": cls.hybrid_override_pattern,
                "num_attention_heads": 15,
                "num_query_groups": 15,
                "ffn_hidden_size": 5120,
                "hidden_dropout": cls.hidden_dropout,
                "num_groups_hyena": 1920,
                "num_groups_hyena_medium": 128,
                "num_groups_hyena_short": 128,
                "attention_dropout": cls.attention_dropout,
                "to_upper": "weighted" if cls.no_renormalize_loss else "normalized_weighted",
                "distribute_saved_activations": False if cls.sequence_parallel else True,
                "cross_entropy_loss_fusion": cls.cross_entropy_loss_fusion,
                "fp32_residual_connection": not cls.no_fp32_residual_connection,
                "add_bias_output": cls.add_bias_output,
            }
        )

        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=cls.lr,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=cls.wd,
            clip_grad=cls.clip_grad,
            params_dtype=torch.float32,
            use_distributed_optimizer=True,
            bf16=True,
        )
        # build optimizer
        optimizer = MegatronOptimizerModule(
            config=optimizer_config,
            lr_scheduler=CosineAnnealingScheduler(warmup_steps=1, max_steps=cls.num_steps, min_lr=3e-5),
            no_weight_decay_cond=config.hyena_no_weight_decay_cond_fn,
        )

        # # Build model
        module = HyenaModel(config, tokenizer=data.tokenizer)
        optimizer.connect(module)
        return module, data, optimizer

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
            testing_callbacks.ValidLossCallback,
            testing_callbacks.ValidOutputCallback,
            testing_callbacks.TrainInputCallback,
            testing_callbacks.TrainOutputCallback,
            testing_callbacks.TrainLossCallback,
            testing_callbacks.OptimizerStateCallback,
        ]:
            pytest.xfail(reason="Tensors not close")
        super().test_stop_and_go_consistency(callback_type)

    @pytest.mark.skip(reason="TODO: assert train_consumed_go > 0 fails.")
    def test_train_val_init_consumed_samples(self):
        pass


class TestEvo2MambaStopAndGo(stop_and_go.StopAndGoHarness):
    """Most of these parameters are copied from test_evo2.py which runs train.py."""

    num_steps: int = 3
    val_check_interval: int = 1
    limit_val_batches: int = 1
    lr: float = 3e-4
    wd: float = 0.01
    clip_grad: float = 1.0
    micro_batch_size: int = 1
    global_batch_size: int = 1
    precision: Literal["16-mixed", "bf16-mixed", "32"] = "bf16-mixed"
    workers: int = 8
    seq_length: int = 8
    hybrid_override_pattern: str = "M-M-M-M*-"
    use_megatron_comm_overlap_llama3_8k: bool = False
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    no_renormalize_loss: bool = False
    sequence_parallel: bool = False
    cross_entropy_loss_fusion: bool = False
    no_fp32_residual_connection: bool = False
    use_targeted_variance_loss: bool = True
    embedding_init_method_std: float = 1.0
    num_layers: int = 9
    hidden_size: int = 128
    ffn_hidden_size: int = 512
    mamba_num_groups: int = 2
    mamba_head_dim: int = 32
    num_query_groups: int = 2
    num_attention_heads: int = 2
    mamba_state_dim: int = 32

    @classmethod
    def setup_trainer(
        cls,
        mode: Mode,
    ) -> nl.Trainer:
        """Setup trainer by passing stop, resume, or continuous callbacks according to mode."""
        ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            overlap_grad_reduce=False,
            overlap_param_gather=False,
            grad_reduce_in_fp32=False,
            align_param_gather=False,
            average_in_collective=True,
        )
        strategy = MegatronStrategy(
            ddp=ddp,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            sequence_parallel=cls.sequence_parallel,
            pipeline_dtype=torch.bfloat16,
            ckpt_async_save=False,
            ckpt_load_optimizer=True,
            ckpt_save_optimizer=True,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        )

        trainer = nl.Trainer(
            devices=1,
            max_steps=cls.num_steps,
            num_nodes=1,
            accelerator="gpu",
            strategy=strategy,
            limit_val_batches=cls.limit_val_batches,
            num_sanity_val_steps=0,
            val_check_interval=cls.val_check_interval,
            log_every_n_steps=cls.val_check_interval,
            enable_checkpointing=True,
            use_distributed_sampler=False,
            callbacks=list(cls.callbacks[mode].values()),
            plugins=nl.MegatronMixedPrecision(
                precision=cls.precision, params_dtype=torch.bfloat16, grad_reduce_in_fp32=False, fp8_wgrad=False
            ),
        )
        return trainer

    @override
    @classmethod
    def setup_class(cls):
        super().setup_class()

        # setup data
        cls.tokenizer = get_nmt_tokenizer("byte-level")
        # run stop and go
        cls.run_stop_and_go()

    @classmethod
    def setup_model(cls, mode: Mode) -> tuple[pl.LightningModule, pl.LightningDataModule, nl.MegatronOptimizerModule]:
        # build data module
        data = MockDataModule(
            seq_length=cls.seq_length,
            micro_batch_size=cls.micro_batch_size,
            global_batch_size=cls.global_batch_size,
            num_workers=cls.workers,
            tokenizer=cls.tokenizer,
        )

        data.init_global_step = 0
        # config
        config = HybridMambaConfig8BEvo2Loss(
            **{
                "tp_comm_overlap": cls.use_megatron_comm_overlap_llama3_8k,
                "seq_length": cls.seq_length,
                "params_dtype": torch.bfloat16,
                "bf16": True,
                "recompute_granularity": None,
                "recompute_method": None,
                "recompute_num_layers": None,
                "hidden_size": cls.hidden_size,
                "ffn_hidden_size": cls.ffn_hidden_size,
                "num_layers": cls.num_layers,
                "hybrid_override_pattern": cls.hybrid_override_pattern,
                "hidden_dropout": cls.hidden_dropout,
                "attention_dropout": cls.attention_dropout,
                "to_upper": "weighted" if cls.no_renormalize_loss else "normalized_weighted",
                "distribute_saved_activations": False if cls.sequence_parallel else True,
                "cross_entropy_loss_fusion": cls.cross_entropy_loss_fusion,
                "fp32_residual_connection": not cls.no_fp32_residual_connection,
                "use_targeted_variance_loss": cls.use_targeted_variance_loss,
                "embedding_init_method_std": cls.embedding_init_method_std,
                "mamba_num_groups": cls.mamba_num_groups,
                "mamba_head_dim": cls.mamba_head_dim,
                "num_query_groups": cls.num_query_groups,
                "num_attention_heads": cls.num_attention_heads,
                "mamba_state_dim": cls.mamba_state_dim,
            }
        )

        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=cls.lr,
            adam_beta1=0.9,
            adam_beta2=0.95,
            weight_decay=cls.wd,
            clip_grad=cls.clip_grad,
            params_dtype=torch.float32,
            use_distributed_optimizer=True,
            bf16=True,
        )
        # build optimizer
        optimizer = MegatronOptimizerModule(
            config=optimizer_config,
            lr_scheduler=CosineAnnealingScheduler(warmup_steps=1, max_steps=cls.num_steps, min_lr=3e-5),
            no_weight_decay_cond=config.hyena_no_weight_decay_cond_fn,
        )

        # # Build model
        module = MambaModel(config, tokenizer=data.tokenizer)
        optimizer.connect(module)
        return module, data, optimizer

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
            testing_callbacks.ValidLossCallback,
            testing_callbacks.ValidOutputCallback,
            testing_callbacks.TrainInputCallback,
            testing_callbacks.TrainOutputCallback,
            testing_callbacks.TrainLossCallback,
            testing_callbacks.OptimizerStateCallback,
        ]:
            pytest.xfail(reason="Tensors not close")
        super().test_stop_and_go_consistency(callback_type)

    @pytest.mark.skip(reason="TODO: assert train_consumed_go > 0 fails.")
    def test_train_val_init_consumed_samples(self):
        pass
