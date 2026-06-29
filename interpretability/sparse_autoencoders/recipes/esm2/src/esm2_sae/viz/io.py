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

"""Parquet I/O utilities for visualization data."""

from pathlib import Path
from typing import List, Union

import pyarrow as pa
import pyarrow.parquet as pq

from .feature_activations import FeatureExample, FeatureStats
from .feature_umap import FeatureGeometry


def save_geometry(geometry: FeatureGeometry, path: Union[str, Path]) -> None:
    """Save feature geometry to parquet."""
    data = {
        "feature_id": geometry.feature_ids,
        "umap_x": geometry.umap_x,
        "umap_y": geometry.umap_y,
    }
    if geometry.cluster_ids is not None:
        data["cluster_id"] = geometry.cluster_ids

    table = pa.table(data)
    pq.write_table(table, str(path))


def save_stats(stats: List[FeatureStats], path: Union[str, Path]) -> None:
    """Save feature statistics to parquet."""
    data = {
        "feature_id": [s.feature_id for s in stats],
        "activation_frequency": [s.activation_frequency for s in stats],
        "mean_activation": [s.mean_activation for s in stats],
        "max_activation": [s.max_activation for s in stats],
        "n_proteins_active": [s.n_proteins_active for s in stats],
    }

    table = pa.table(data)
    pq.write_table(table, str(path))


def save_examples(examples: List[FeatureExample], path: Union[str, Path]) -> None:
    """Save feature examples to parquet."""
    data = {
        "feature_id": [e.feature_id for e in examples],
        "protein_id": [e.protein_id for e in examples],
        "residue_idx": [e.residue_idx for e in examples],
        "activation_value": [e.activation_value for e in examples],
        "sequence_window": [e.sequence_window for e in examples],
        "window_start": [e.window_start for e in examples],
        "highlight_values": [e.highlight_values for e in examples],
    }

    table = pa.table(data)
    pq.write_table(table, str(path))
