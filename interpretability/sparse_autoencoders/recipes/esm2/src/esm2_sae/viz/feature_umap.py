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

"""Feature UMAP computation from SAE decoder weights."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class FeatureGeometry:
    """UMAP coordinates and optional clusters for features."""

    feature_ids: np.ndarray  # (n_features,)
    umap_x: np.ndarray  # (n_features,)
    umap_y: np.ndarray  # (n_features,)
    cluster_ids: Optional[np.ndarray] = None  # (n_features,)


def compute_feature_umap(
    sae: torch.nn.Module,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    compute_clusters: bool = True,
    hdbscan_min_cluster_size: int = 10,
) -> FeatureGeometry:
    """Compute UMAP coordinates for SAE features from decoder weights.

    Args:
        sae: Trained SAE with decoder.weight attribute
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric (should be 'cosine')
        random_state: Random seed for reproducibility
        compute_clusters: Whether to compute HDBSCAN clusters
        hdbscan_min_cluster_size: Minimum cluster size for HDBSCAN

    Returns:
        FeatureGeometry with coordinates and optional clusters
    """
    from umap import UMAP

    # Extract decoder columns (each column = one feature's direction)
    # decoder.weight shape: (input_dim, hidden_dim) for nn.Linear(hidden_dim, input_dim)
    # We want one vector per feature, so transpose
    W_dec = sae.decoder.weight.detach().cpu().numpy()  # (input_dim, hidden_dim)
    feature_vectors = W_dec.T  # (hidden_dim, input_dim) = (n_features, feature_dim)

    # L2 normalize each feature vector
    norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    feature_vectors_normed = feature_vectors / norms

    n_features = feature_vectors_normed.shape[0]

    # Compute 2D UMAP
    umap_2d = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords_2d = umap_2d.fit_transform(feature_vectors_normed)

    # Optional: compute clusters via HDBSCAN on higher-dim UMAP
    cluster_ids = None
    if compute_clusters:
        try:
            import hdbscan

            # Higher-dim UMAP for clustering (more structure preserved)
            umap_cluster = UMAP(
                n_components=10,
                n_neighbors=n_neighbors,
                min_dist=0.0,
                metric=metric,
                random_state=random_state,
            )
            coords_high = umap_cluster.fit_transform(feature_vectors_normed)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=hdbscan_min_cluster_size,
                metric="euclidean",
            )
            cluster_ids = clusterer.fit_predict(coords_high)
        except ImportError:
            print("Warning: hdbscan not installed, skipping clustering")

    return FeatureGeometry(
        feature_ids=np.arange(n_features),
        umap_x=coords_2d[:, 0],
        umap_y=coords_2d[:, 1],
        cluster_ids=cluster_ids,
    )
