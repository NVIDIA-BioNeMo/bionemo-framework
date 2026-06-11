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


from collections import defaultdict
from typing import Dict, List

import torch
from lightning.pytorch import Callback


class MetricTracker(Callback):  # noqa: D101
    def __init__(self, metrics_to_track_val: List[str], metrics_to_track_train: List[str]):  # noqa: D107
        self.metrics_to_track_val = metrics_to_track_val
        self.metrics_to_track_train = metrics_to_track_train
        self._collection_val = defaultdict(list)
        self._collection_train = defaultdict(list)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: D102
        if isinstance(outputs, torch.Tensor):
            self._collection_val["unnamed"].append(outputs)
        else:
            for metric in self.metrics_to_track_val:
                self._collection_val[metric].append(outputs[metric])

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  # noqa: D102
        if isinstance(outputs, torch.Tensor):
            self._collection_train["unnamed"].append(outputs)
        else:
            for metric in self.metrics_to_track_train:
                self._collection_train[metric].append(outputs[metric])

    def on_validation_epoch_end(self, trainer, pl_module):  # noqa: D102
        elogs = trainer.logged_metrics  # access it here
        self._collection_val["logged_metrics"].extend(elogs)

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D102
        elogs = trainer.logged_metrics  # access it here
        self._collection_train["logged_metrics"].extend(elogs)

    @property
    def collection_val(self) -> Dict[str, torch.Tensor | List[str]]:  # noqa: D102
        res = {k: torch.tensor(v) for k, v in self._collection_val.items() if k != "logged_metrics"}
        res["logged_metrics"] = self._collection_val["logged_metrics"]
        return res

    @property
    def collection_train(self) -> Dict[str, torch.Tensor | str]:  # noqa: D102
        res = {k: torch.tensor(v) for k, v in self._collection_train.items() if k != "logged_metrics"}
        res["logged_metrics"] = self._collection_train["logged_metrics"]
        return res
