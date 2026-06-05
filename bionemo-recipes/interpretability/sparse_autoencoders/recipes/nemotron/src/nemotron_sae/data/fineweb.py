# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""FineWeb dataset loading utilities.

FineWeb is a large-scale, high-quality web text dataset from HuggingFace.
Uses streaming to avoid downloading the full 15T+ token dataset.
"""

from typing import List, Optional


try:
    from datasets import load_dataset

    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False


def load_fineweb(
    split: str = "train",
    max_samples: Optional[int] = None,
    min_length: int = 50,
    subset: str = "sample-10BT",
) -> List[str]:
    """Load FineWeb dataset from HuggingFace via streaming.

    Args:
        split: Dataset split ('train')
        max_samples: Maximum number of samples to return (None for unlimited)
        min_length: Minimum character length for samples
        subset: FineWeb subset name (e.g. 'sample-10BT', 'sample-100BT')

    Returns:
        List of text strings
    """
    if not HAS_DATASETS:
        raise ImportError("Install datasets: pip install datasets")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb",
        name=subset,
        split=split,
        streaming=True,
    )

    texts = []
    for item in dataset:
        text = item["text"].strip()
        if len(text) >= min_length:
            texts.append(text)
            if max_samples and len(texts) >= max_samples:
                break

    print(f"Loaded {len(texts)} texts from FineWeb ({subset}) {split}")
    return texts
