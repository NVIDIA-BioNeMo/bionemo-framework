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


import lightning.pytorch as pl
import torch.nn.functional as F


def compute_biobert_loss_singlegpu(trainer: pl.Trainer, pl_module: pl.LightningModule):
    """Computes the loss for BioBert models on a single GPU.

    This will not function in multi-gpu settings nor with models that do not conform to BioBert.

    Args:
        trainer (pl.Trainer): The Lightning Trainer object.
        pl_module (pl.LightningModule): The LightningModule being trained.

    Returns:
        float: The mean loss.

    See Also:
    - :class: BioBertModel
    """
    model = pl_module
    dl = trainer.datamodule.val_dataloader()

    n, loss = -1, 0.0
    model.eval()
    # batch = next(iter(dl))
    batch = model.data_step(iter(dl))
    result = model(
        input_ids=batch["text"].cuda(),  # 'tokens' also a valid input for MockGPTDataModule
        attention_mask=batch["attention_mask"].cuda(),
    )
    loss_mask = batch["loss_mask"].cuda()
    # Not guaranteed i guess?
    logits = result["token_logits"]
    target = batch["labels"].cuda()
    loss += F.cross_entropy(logits[loss_mask].float(), target[loss_mask], reduction="sum")
    n += loss_mask.sum()

    mean_loss: float = (loss / n).detach().cpu().numpy().item()
    model.train()
    return mean_loss
