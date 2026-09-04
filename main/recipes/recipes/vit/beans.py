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
#
#    MIT License
#
#    Copyright (c) 2020 AIR Lab Makerere University (AI-Lab-Makerere/ibean)
#
#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:
#
#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.
#
#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

import logging

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


logger = logging.getLogger(__name__)


def infinite_dataloader(dataloader, sampler):
    """Create an infinite iterator that automatically restarts at the end of each epoch."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # Update epoch for proper shuffling
        for batch in dataloader:
            yield batch
        epoch += 1  # Increment epoch counter after completing one full pass


class BeansDataset(Dataset):
    """
    Simple wrapper Dataset for AI-Lab-Makerere/beans that converts PIL images to Tensors.
    """

    def __init__(self, image_size: tuple[int, int], split: str = "train"):
        """
        Args:
            image_size (tuple[int, int]): Resize 2-D image data to this size.
            split (str): Dataset split to load. Options: ["train", "validation", "test"]
        """
        self.resize_dimensions = image_size
        # Download Beans Dataset.
        self.dataset = load_dataset("AI-Lab-Makerere/beans", split=split)
        self.class_list = self.dataset.features["labels"].names
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            logger.info(
                f"[AI-Lab-Makerere/beans (Split={split})]\nDataset Size: {len(self.dataset)}\nClasses (Count={len(self.class_list)}): {self.class_list}"
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Preprocess sample.
        sample = self.dataset[idx]
        image_tensor = to_tensor(sample["image"].resize(self.resize_dimensions).convert("RGB"))
        label_idx = sample["labels"]
        return image_tensor, label_idx
