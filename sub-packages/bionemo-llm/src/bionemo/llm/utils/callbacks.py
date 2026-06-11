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


import os
from typing import Any, Literal, Sequence


try:  # Python 3.12+
    from typing import override
except ImportError:  # Python < 3.12
    from typing_extensions import override

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import BasePredictionWriter
from megatron.core import parallel_state
from nemo.utils import logging as logger

from bionemo.llm.lightning import batch_collator


IntervalT = Literal["epoch", "batch"]


class PredictionWriter(BasePredictionWriter, pl.Callback):
    """A callback that writes predictions to disk at specified intervals during training.

    Logits, Embeddings, Hiddens, Input IDs, and Labels may all be saved to the disk depending on trainer configuration.
    Batch Idxs are provided for each prediction in the same dictionary. These must be used to maintain order between
    multi device predictions and single device predictions.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike,
        write_interval: IntervalT,
        batch_dim_key_defaults: dict[str, int] | None = None,
        seq_dim_key_defaults: dict[str, int] | None = None,
        save_all_model_parallel_ranks: bool = False,
        files_per_subdir: int | None = None,
    ):
        """Initializes the callback.

        Args:
            output_dir: The directory where predictions will be written.
            write_interval: The interval at which predictions will be written (batch, epoch). Epoch may not be used with
                multi-device trainers.
            batch_dim_key_defaults: The default batch dimension for each key, if different from the standard 0.
            seq_dim_key_defaults: The default sequence dimension for each key, if different from the standard 1.
            save_all_model_parallel_ranks: Whether to save predictions for all model parallel ranks. Generally these
                will be redundant.
            files_per_subdir: Number of files to write to each subdirectory. If provided, subdirectories with N files
                each will be created. Ignored unless write_interval is 'batch'.
        """
        super().__init__(write_interval)
        self.write_interval = write_interval
        self.output_dir = str(output_dir)
        self.base_dir = self.output_dir  # start out like this, but output_dir will be updated if files_per_subdir>0
        self.batch_dim_key_defaults = batch_dim_key_defaults
        self.seq_dim_key_defaults = seq_dim_key_defaults
        self.save_all_model_parallel_ranks = save_all_model_parallel_ranks
        self.files_per_subdir = files_per_subdir
        # Initialize to infinity if files_per_subdir is provided so that we create a new subdirectory before writing
        #   any files.
        self.num_files_written = float("inf") if files_per_subdir else 0
        self.num_subdirs_written = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs) -> None:  # noqa: D417
        """Invoked with Trainer.fit, validate, test, and predict are called. Will immediately fail when 'write_interval' is 'epoch' and 'trainer.num_devices' > 1.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance.
        """
        if trainer.num_devices > 1 and self.write_interval == "epoch":
            logger.warning(
                "Multi-GPU predictions could result in shuffled inputs. Verify that the original indices are included "
                "in the model's predictions as outputs are not ordered and batch indices do not track input order."
            )

    @staticmethod
    def _assert_initialized():
        """Asserts that the environment is initialized."""
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized() and parallel_state.is_initialized()
        ):
            raise RuntimeError("This function is only defined within an initialized megatron parallel environment.")

    @property
    def data_parallel_world_size(self) -> int:
        """Returns the data parallel world size."""
        self._assert_initialized()
        return torch.distributed.get_world_size(parallel_state.get_data_parallel_group(with_context_parallel=False))

    @property
    def data_parallel_rank(self) -> int:
        """Returns the data parallel rank."""
        self._assert_initialized()
        return torch.distributed.get_rank(parallel_state.get_data_parallel_group(with_context_parallel=False))

    @property
    def should_write_predictions(self) -> bool:
        """Ensures that predictions are only written on TP/CP rank 0 and that it is the last stage of the pipeline."""
        self._assert_initialized()
        if not parallel_state.is_pipeline_last_stage():
            return False
        if self.save_all_model_parallel_ranks:
            return True
        # TODO: handle expert parallelism and other kinds of parallelism
        return parallel_state.get_tensor_model_parallel_rank() == 0 and parallel_state.get_context_parallel_rank() == 0

    @override
    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Sequence[int] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """Writes predictions to disk at the end of each batch.

        Predictions files follow the naming pattern, where rank is the active GPU in which the predictions were made.
        predictions__rank_{rank}__batch_{batch_idx}.pt

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance.
            prediction: The prediction made by the model.
            batch_indices: The indices of the batch.
            batch: The batch data.
            batch_idx: The index of the batch.
            dataloader_idx: The index of the dataloader.
        """
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        if self.should_write_predictions:
            if (
                self.files_per_subdir is not None
                and (self.num_files_written * self.data_parallel_world_size) >= self.files_per_subdir
            ):
                self.num_subdirs_written += 1
                self.output_dir = os.path.join(self.base_dir, f"subdir_{self.num_subdirs_written}")
                os.makedirs(self.output_dir, exist_ok=True)
                self.num_files_written = 0
            result_path = os.path.join(
                self.output_dir,
                f"predictions__rank_{trainer.global_rank}__dp_rank_{self.data_parallel_rank}__batch_{batch_idx}.pt",
            )

            # batch_indices is not captured due to a lightning bug when return_predictions = False
            # we use input IDs in the prediction to map the result to input.

            # NOTE store the batch_idx so we do not need to rely on filenames for reconstruction of inputs. This is wrapped
            # in a tensor and list container to ensure compatibility with batch_collator.
            prediction["batch_idx"] = torch.tensor([batch_idx], dtype=torch.int64)

            torch.save(prediction, result_path)
            logger.info(f"Inference predictions are stored in {result_path}\n{prediction.keys()}")
            self.num_files_written += 1

    @override
    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Any,
        batch_indices: Sequence[int],
    ) -> None:
        """Writes predictions to disk at the end of each epoch.

        Writing all predictions on epoch end is memory intensive. It is recommended to use the batch writer instead for
        large predictions.

        Multi-device predictions will likely yield predictions in an order that is inconsistent with single device predictions and the input data.

        Args:
            trainer: The Trainer instance.
            pl_module: The LightningModule instance.
            predictions: The predictions made by the model.
            batch_indices: The indices of the batch.

        Raises:
            Multi-GPU predictions are output in an inconsistent order with multiple devices.
        """
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        if self.should_write_predictions:
            result_path = os.path.join(
                self.output_dir,
                f"predictions__rank_{trainer.global_rank}__dp_rank_{self.data_parallel_rank}.pt",
            )

            # collate multiple batches / ignore empty ones
            collate_kwargs = {}
            if self.batch_dim_key_defaults is not None:
                collate_kwargs["batch_dim_key_defaults"] = self.batch_dim_key_defaults
            if self.seq_dim_key_defaults is not None:
                collate_kwargs["seq_dim_key_defaults"] = self.seq_dim_key_defaults

            prediction = batch_collator([item for item in predictions if item is not None], **collate_kwargs)

            # batch_indices is not captured due to a lightning bug when return_predictions = False
            # we use input IDs in the prediction to map the result to input
            if isinstance(prediction, dict):
                keys = prediction.keys()
            else:
                keys = "tensor"
            torch.save(prediction, result_path)
            logger.info(f"Inference predictions are stored in {result_path}\n{keys}")
