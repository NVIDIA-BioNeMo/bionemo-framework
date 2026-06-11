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


import os
from dataclasses import asdict, is_dataclass

import numpy as np
import torch
from lightning.pytorch.callbacks import BasePredictionWriter


class PredWriter(BasePredictionWriter):
    """Write predictions to disk and optionally merge across distributed ranks.

    This callback caches predictions every `caching_interval` batches and writes
    them to ``.npy`` files named by key, rank, and batch index. Optionally, it
    merges per-rank shards into a single array per key at predict-epoch end.

    Note:
        In DDP, if the final batch is not divisible by the number of GPU
        workers, duplicate samples may occur near epoch boundaries.

    """

    def __init__(
        self,
        output_dir: str,
        write_interval: str,
        caching_interval: int = 1,
        merge_on_epoch_end: bool = False,
        delete_after_merge: bool = False,
    ):
        """Initialize the prediction writer.

        Args:
            output_dir: Directory where prediction files are saved.
            write_interval: When to write predictions ("batch" or "epoch").
            caching_interval: Number of batches to cache before writing.
            merge_on_epoch_end: Whether to merge per-rank files after predict.
            delete_after_merge: Whether to delete shard files after merging.
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.caching_interval = caching_interval
        self.merge_on_epoch_end = merge_on_epoch_end
        self.delete_after_merge = delete_after_merge
        self.predictions_buffer = []
        self._last_batch_idx = -1
        os.makedirs(self.output_dir, exist_ok=True)

    def _convert_predictions(self, prediction):
        """Convert supported prediction types to a NumPy-friendly dict.

        Accepts a dataclass or a dict mapping string keys to tensors/arrays and
        returns a new dict with values converted to NumPy arrays.

        Args:
            prediction: A dataclass instance or dict[str, Any] of predictions.

        Returns:
            Dict[str, np.ndarray]: A mapping of keys to NumPy arrays.

        Raises:
            TypeError: If `prediction` is neither a dataclass nor a dict.
        """
        if is_dataclass(prediction):
            prediction = asdict(prediction)
        elif not isinstance(prediction, dict):
            raise TypeError("Prediction must be a dataclass or a dictionary.")
        return {
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            for key, value in prediction.items()
            if value is not None
        }

    def _save_predictions(self, trainer, batch_idx):
        """Flush the cached predictions to per-key NumPy files.

        Filenames include the trainer's global rank and the adjusted batch
        index, allowing later reconstruction and/or merging.

        Args:
            trainer: The running `pl.Trainer` instance.
            batch_idx: The adjusted batch index for naming consistency.
        """
        flattened_predictions = {
            key: np.concatenate([buffer[key] for buffer in self.predictions_buffer], axis=0)
            for key in self.predictions_buffer[0]
        }
        for key, value in flattened_predictions.items():
            file_path = os.path.join(self.output_dir, f"{key}_rank_{trainer.global_rank}_batch_{batch_idx}.npy")
            np.save(file_path, value)
        self.predictions_buffer.clear()

    def _merge_predictions(self):
        """Merge per-rank shard files into a single array per key.

        The method reconstructs the original sample ordering based on the DDP
        sampling pattern by interleaving ranks and batches in ascending order.

        Side Effects:
            Writes merged files named ``{key}_merged.npy`` and optionally
            deletes the original shard files when `delete_after_merge` is True.
        """
        all_files = [f for f in os.listdir(self.output_dir) if f.endswith(".npy")]
        merged_data = {}
        sorted_files = sorted(
            all_files, key=lambda file: (file.split("_rank_")[0], int(file.split("_batch_")[-1].split(".npy")[0]))
        )

        for file in sorted_files:
            key = file.split("_rank_")[0]
            rank = int(file.split("_rank_")[1].split("_batch_")[0])
            batch_idx = int(file.split("_batch_")[-1].split(".npy")[0])
            file_path = os.path.join(self.output_dir, file)
            data = np.load(file_path)

            if key not in merged_data:
                merged_data[key] = {}

            if batch_idx not in merged_data[key]:
                merged_data[key][batch_idx] = {}

            merged_data[key][batch_idx][rank] = data

        for key, batches in merged_data.items():
            all_data = []
            max_rank = max(max(rank_data.keys()) for rank_data in batches.values())

            for batch_idx, rank_data in sorted(batches.items()):
                batch_data = [None] * (max_rank + 1)

                for rank, data in rank_data.items():
                    batch_data[rank] = data

                # - reconstruct the original order based on the sampling logic
                total_size = sum(
                    len(batch_data[rank]) for rank in range(len(batch_data)) if batch_data[rank] is not None
                )
                num_replicas = max_rank + 1
                sample_data = next(data for data in batch_data if data is not None)
                data_shape = (total_size,) + sample_data.shape[1:]  # noqa: RUF005
                dtype = batch_data[0].dtype
                reconstructed_data = np.empty(data_shape, dtype=dtype if batch_data[0] is not None else float)

                for rank in range(len(batch_data)):
                    if batch_data[rank] is not None:
                        indices = list(range(rank, total_size, num_replicas))
                        for i, idx in enumerate(indices):
                            if i < len(batch_data[rank]):
                                reconstructed_data[idx] = batch_data[rank][i]

                all_data.append(reconstructed_data)

            merged_array = np.concatenate(all_data, axis=0)
            merged_file_path = os.path.join(self.output_dir, f"{key}_merged.npy")
            np.save(merged_file_path, merged_array)

        if self.delete_after_merge:
            for file in all_files:
                os.remove(os.path.join(self.output_dir, file))

    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """Cache batch predictions and write when cache reaches the threshold.

        Args:
            trainer: The running `pl.Trainer` instance.
            pl_module: The Lightning module producing predictions.
            prediction: The model prediction for the current batch.
            batch_indices: Indices for the current batch (unused).
            batch: The current batch (unused).
            batch_idx: The batch index within the epoch.
            dataloader_idx: Index of the dataloader (unused).
        """
        prediction = self._convert_predictions(prediction)
        self.predictions_buffer.append(prediction)
        self._last_batch_idx = batch_idx
        if len(self.predictions_buffer) >= self.caching_interval:
            if trainer.datamodule.global_batch_size is not None:
                adjusted_batch_idx = (
                    trainer.datamodule.init_consumed_samples // trainer.datamodule.global_batch_size + batch_idx
                )
            else:
                adjusted_batch_idx = batch_idx
            self._save_predictions(trainer, adjusted_batch_idx)

    def on_predict_epoch_end(self, trainer, pl_module):
        """Flush remaining predictions and optionally merge shards.

        Ensures any residual buffered predictions are saved with a consistent
        adjusted batch index derived from consumed samples, then performs a
        cross-rank merge on global rank zero if enabled.

        Args:
            trainer: The running `pl.Trainer` instance.
            pl_module: The Lightning module (unused).
        """
        if len(self.predictions_buffer) > 0:
            if trainer.datamodule.global_batch_size is not None:
                consumed_samples = trainer.datamodule.calc_consumed_samples()
                batch_idx = consumed_samples // trainer.datamodule.global_batch_size
                adjusted_batch_idx = batch_idx + int((consumed_samples % trainer.datamodule.global_batch_size) > 0)
            else:
                adjusted_batch_idx = self._last_batch_idx + 1
            self._save_predictions(trainer, adjusted_batch_idx)

        trainer.strategy.barrier()
        if self.merge_on_epoch_end and trainer.is_global_zero:
            self._merge_predictions()
