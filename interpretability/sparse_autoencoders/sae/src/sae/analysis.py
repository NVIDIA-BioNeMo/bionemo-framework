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

"""Feature analysis tools for trained SAEs.

Computes domain-agnostic artifacts:
- Per-feature activation statistics
- Top-firing examples per feature
- UMAP coordinates from decoder weights
- Cluster centroids and labels for UMAP dashboard

Results are saved as Parquet tables for easy querying with DuckDB or pandas.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch


try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


@dataclass
class FeatureStats:
    """Per-feature activation statistics."""

    feature_id: int
    activation_freq: float  # Fraction of inputs where feature fires (> 0)
    mean_activation: float  # Mean activation when active
    max_activation: float  # Maximum activation observed
    std_activation: float  # Std dev of activations when active
    total_activations: int  # Number of times feature fired


@dataclass
class TopExample:
    """A top-firing example for a feature."""

    feature_id: int
    example_idx: int  # Index in the dataset
    activation_value: float  # Activation value


@dataclass
class FeatureGeometry:
    """UMAP coordinates and optional clusters for features."""

    feature_ids: np.ndarray  # (n_features,)
    umap_x: np.ndarray  # (n_features,)
    umap_y: np.ndarray  # (n_features,)
    cluster_ids: Optional[np.ndarray] = None  # (n_features,)


@dataclass
class FeatureLogits:
    """Top positive/negative logits for a feature."""

    feature_id: int
    top_positive: List[tuple]  # [(token_str, logit_value), ...]
    top_negative: List[tuple]  # [(token_str, logit_value), ...]


@dataclass
class ClusterInfo:
    """Information about a feature cluster from UMAP/HDBSCAN."""

    cluster_id: int
    x: float  # centroid x
    y: float  # centroid y
    feature_ids: List[int]
    size: int


class _StatsAccumulator:
    """Accumulates per-feature activation statistics from code batches.

    Shared by compute_feature_stats() and TokenActivationCollector to avoid
    duplicating the vectorized stats logic.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.count_active = torch.zeros(n_features, dtype=torch.long)
        self.sum_act = torch.zeros(n_features, dtype=torch.float64)
        self.sum_sq = torch.zeros(n_features, dtype=torch.float64)
        self.max_act = torch.zeros(n_features, dtype=torch.float32)
        self.total_samples = 0

    def update(self, codes: torch.Tensor) -> None:
        """Update accumulators from a codes tensor [batch, n_features]."""
        self.total_samples += codes.shape[0]
        active_mask = codes > 0
        self.count_active += active_mask.sum(dim=0).long()
        masked = codes * active_mask.float()
        self.sum_act += masked.sum(dim=0).double()
        self.sum_sq += (masked**2).sum(dim=0).double()
        batch_max = codes.max(dim=0).values
        self.max_act = torch.maximum(self.max_act, batch_max)

    def build_stats(self) -> List[FeatureStats]:
        """Convert accumulators into a list of FeatureStats."""
        stats = []
        for i in range(self.n_features):
            count = self.count_active[i].item()
            if count > 0:
                mean = self.sum_act[i].item() / count
                variance = (self.sum_sq[i].item() / count) - (mean**2)
                std = max(0.0, variance) ** 0.5
            else:
                mean = 0.0
                std = 0.0

            stats.append(
                FeatureStats(
                    feature_id=i,
                    activation_freq=count / self.total_samples if self.total_samples > 0 else 0.0,
                    mean_activation=float(mean),
                    max_activation=float(self.max_act[i].item()),
                    std_activation=float(std),
                    total_activations=int(count),
                )
            )
        return stats


def compute_cluster_centroids(geometry: "FeatureGeometry") -> List[ClusterInfo]:
    """Group features by cluster and compute centroids.

    Args:
        geometry: FeatureGeometry with cluster_ids from compute_feature_umap()

    Returns:
        List of ClusterInfo sorted by size descending. Noise cluster (-1)
        is excluded.
    """
    if geometry.cluster_ids is None:
        return []

    clusters: Dict[int, List[int]] = {}
    for i, cid in enumerate(geometry.cluster_ids):
        cid = int(cid)
        if cid == -1:
            continue
        clusters.setdefault(cid, []).append(i)

    result = []
    for cid, indices in clusters.items():
        xs = geometry.umap_x[indices]
        ys = geometry.umap_y[indices]
        result.append(
            ClusterInfo(
                cluster_id=cid,
                x=float(np.mean(xs)),
                y=float(np.mean(ys)),
                feature_ids=[int(geometry.feature_ids[i]) for i in indices],
                size=len(indices),
            )
        )

    result.sort(key=lambda c: -c.size)
    return result


def build_cluster_label_prompt(
    cluster: ClusterInfo,
    descriptions: Dict[int, str],
    stats: List["FeatureStats"],
    max_features: int = 15,
) -> str:
    """Build an LLM prompt to generate a short label for a cluster.

    Selects the top features by activation frequency, lists their
    descriptions, and asks the LLM for a 2-5 word label.

    Args:
        cluster: ClusterInfo for the cluster
        descriptions: Dict mapping feature_id -> description string
        stats: List of FeatureStats (indexed by feature_id)
        max_features: Maximum number of features to include in the prompt

    Returns:
        Prompt string for the LLM
    """
    # Build stats lookup
    stats_by_id = {s.feature_id: s for s in stats}

    # Sort cluster features by activation frequency (descending)
    sorted_features = sorted(
        cluster.feature_ids,
        key=lambda fid: stats_by_id.get(fid, FeatureStats(fid, 0, 0, 0, 0, 0)).activation_freq,
        reverse=True,
    )[:max_features]

    lines = []
    for fid in sorted_features:
        desc = descriptions.get(fid, f"Feature {fid}")
        freq = stats_by_id.get(fid, FeatureStats(fid, 0, 0, 0, 0, 0)).activation_freq
        lines.append(f"  - Feature {fid} (freq={freq:.4f}): {desc}")

    feature_list = "\n".join(lines)

    return f"""Below are the top features in a cluster of sparse autoencoder features, along with their descriptions.

Cluster {cluster.cluster_id} ({cluster.size} features):
{feature_list}

Based on these feature descriptions, provide a short label (2-5 words) that captures the common theme of this cluster. Reply with ONLY the label, nothing else.

Label:"""


def save_cluster_labels(
    clusters: List[ClusterInfo],
    labels: List[str],
    output_path: Union[str, Path],
) -> None:
    """Save cluster labels as JSON for the dashboard.

    Writes a JSON array of label objects matching the embedding-atlas
    Label interface: [{x, y, text, priority}].

    Args:
        clusters: List of ClusterInfo from compute_cluster_centroids()
        labels: List of label strings, one per cluster (same order)
        output_path: Path to write the JSON file
    """
    if len(clusters) != len(labels):
        raise ValueError(f"clusters has {len(clusters)} items, labels has {len(labels)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for cluster, label in zip(clusters, labels):
        data.append(
            {
                "x": cluster.x,
                "y": cluster.y,
                "text": label,
                "priority": cluster.size,
            }
        )

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(data)} cluster labels to {output_path}")


def compute_feature_stats(
    sae: torch.nn.Module,
    activations,
    device: str = "cpu",
    top_k: int = 20,
    batch_size: int = 4096,
    output_dir: Optional[Union[str, Path]] = None,
    show_progress: bool = True,
) -> tuple:
    """Compute per-feature statistics and top examples.

    Encodes activations through the SAE in batches, accumulates per-feature
    statistics with vectorized ops, and tracks top-k examples per feature
    using bounded min-heaps (memory-safe).

    Args:
        sae: Trained SAE model
        activations: Tensor of shape [n_samples, input_dim], or a DataLoader
            yielding activation batches
        device: Device for SAE encoding
        top_k: Number of top examples per feature
        batch_size: Batch size for encoding (only used when activations is a tensor)
        output_dir: If provided, saves feature_stats.parquet and top_examples.parquet
        show_progress: Whether to show a progress bar

    Returns:
        Tuple of (stats: List[FeatureStats], top_examples: List[TopExample])

    Example:
        >>> stats, top_examples = compute_feature_stats(sae, activations_flat, device="cuda")
        >>> # Or with a DataLoader:
        >>> stats, top_examples = compute_feature_stats(sae, dataloader, device="cuda")
    """
    import heapq

    sae = sae.eval().to(device)
    n_features = sae.hidden_dim
    acc = _StatsAccumulator(n_features)

    # Bounded min-heaps: (activation, example_idx)
    heaps: List[list] = [[] for _ in range(n_features)]
    global_idx = 0

    # Build batch iterator from tensor or dataloader
    if isinstance(activations, torch.Tensor):
        n_samples = activations.shape[0]
        batches = (activations[i : i + batch_size] for i in range(0, n_samples, batch_size))
        n_batches = (n_samples + batch_size - 1) // batch_size
    else:
        batches = activations
        n_batches = len(activations) if hasattr(activations, "__len__") else None

    if show_progress:
        try:
            from tqdm.auto import tqdm

            batches = tqdm(batches, total=n_batches, desc="Computing feature statistics")
        except ImportError:
            pass

    for batch in batches:
        batch = batch.to(device)
        with torch.no_grad():
            codes = sae.encode(batch).cpu()

        acc.update(codes)

        # Top-k per feature using bounded heaps
        for feat_idx in range(n_features):
            feat_codes = codes[:, feat_idx]
            active = feat_codes > 0
            n_active = active.sum().item()
            if n_active == 0:
                continue
            k = min(top_k, n_active)
            vals, idxs = feat_codes.topk(k)
            heap = heaps[feat_idx]
            for v, idx in zip(vals.tolist(), idxs.tolist()):
                entry = (v, global_idx + idx)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif v > heap[0][0]:
                    heapq.heapreplace(heap, entry)

        global_idx += codes.shape[0]

    stats = acc.build_stats()

    # Build TopExample list (sorted descending per feature)
    top_examples = []
    for feat_idx in range(n_features):
        for act_val, example_idx in sorted(heaps[feat_idx], key=lambda x: -x[0]):
            top_examples.append(
                TopExample(
                    feature_id=feat_idx,
                    example_idx=example_idx,
                    activation_value=act_val,
                )
            )

    # Optionally save to Parquet
    if output_dir is not None:
        if not HAS_PARQUET:
            raise ImportError("pyarrow required for saving. Install with: pip install pyarrow")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stats_data = {
            "feature_id": [s.feature_id for s in stats],
            "activation_freq": [s.activation_freq for s in stats],
            "mean_activation": [s.mean_activation for s in stats],
            "max_activation": [s.max_activation for s in stats],
            "std_activation": [s.std_activation for s in stats],
            "total_activations": [s.total_activations for s in stats],
        }
        pq.write_table(pa.table(stats_data), str(output_dir / "feature_stats.parquet"))
        print(f"Saved {len(stats)} feature stats to {output_dir / 'feature_stats.parquet'}")

        ex_data = {
            "feature_id": [e.feature_id for e in top_examples],
            "example_idx": [e.example_idx for e in top_examples],
            "activation_value": [e.activation_value for e in top_examples],
        }
        pq.write_table(pa.table(ex_data), str(output_dir / "top_examples.parquet"))
        print(f"Saved {len(top_examples)} top examples to {output_dir / 'top_examples.parquet'}")

        n_active = sum(1 for s in stats if s.total_activations > 0)
        n_dead = n_features - n_active
        print("\nSummary:")
        print(f"  Total features: {n_features}")
        print(f"  Active features: {n_active}")
        print(f"  Dead features: {n_dead} ({100 * n_dead / n_features:.1f}%)")

    return stats, top_examples


def compute_feature_logits(
    sae: torch.nn.Module,
    unembedding: torch.Tensor,
    vocab: List[str],
    top_k: int = 10,
) -> List[FeatureLogits]:
    """Compute top positive/negative logits for each feature.

    Analyzes what tokens each feature promotes or suppresses by projecting
    the decoder weights through the model's unembedding matrix.

    Args:
        sae: Trained SAE with decoder.weight attribute
        unembedding: Model's unembedding matrix, shape (vocab_size, hidden_dim)
        vocab: List of token strings, length vocab_size
        top_k: Number of top/bottom tokens to return per feature

    Returns:
        List of FeatureLogits, one per feature

    Example:
        >>> # GPT-2
        >>> unembedding = gpt2.model.lm_head.weight.detach()
        >>> vocab = [gpt2.tokenizer.decode([i]) for i in range(len(gpt2.tokenizer))]
        >>> logits = compute_feature_logits(sae, unembedding, vocab)
        >>> print(logits[42].top_positive)  # tokens feature 42 promotes
    """
    # Get decoder weights: shape (input_dim, n_features)
    W_dec = sae.decoder.weight.detach()

    # Ensure unembedding is on same device
    if unembedding.device != W_dec.device:
        unembedding = unembedding.to(W_dec.device)

    # Compute logit effects: (vocab_size, input_dim) @ (input_dim, n_features) = (vocab_size, n_features)
    logit_effects = unembedding @ W_dec  # (V, N)

    n_features = logit_effects.shape[1]
    results = []

    for feat_idx in range(n_features):
        effects = logit_effects[:, feat_idx]

        # Top positive (tokens this feature promotes)
        pos_indices = torch.topk(effects, top_k).indices.tolist()
        pos_values = effects[pos_indices].tolist()
        top_positive = [(vocab[i], v) for i, v in zip(pos_indices, pos_values)]

        # Top negative (tokens this feature suppresses)
        neg_indices = torch.topk(-effects, top_k).indices.tolist()
        neg_values = effects[neg_indices].tolist()
        top_negative = [(vocab[i], v) for i, v in zip(neg_indices, neg_values)]

        results.append(
            FeatureLogits(
                feature_id=feat_idx,
                top_positive=top_positive,
                top_negative=top_negative,
            )
        )

    return results


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
    try:
        from umap import UMAP
    except ImportError:
        raise ImportError("umap-learn required. Install with: pip install umap-learn")

    # Extract decoder columns (each column = one feature's direction)
    # decoder.weight shape: (input_dim, hidden_dim) for nn.Linear(hidden_dim, input_dim)
    W_dec = sae.decoder.weight.detach().cpu().numpy()  # (input_dim, hidden_dim)
    feature_vectors = W_dec.T  # (hidden_dim, input_dim) = (n_features, feature_dim)

    # L2 normalize each feature vector
    norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    feature_vectors_normed = feature_vectors / norms

    n_features = feature_vectors_normed.shape[0]

    print(f"Computing UMAP for {n_features} features...")

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
            print(f"  Found {len(set(cluster_ids)) - (1 if -1 in cluster_ids else 0)} clusters")
        except ImportError:
            print("  Warning: hdbscan not installed, skipping clustering")

    return FeatureGeometry(
        feature_ids=np.arange(n_features),
        umap_x=coords_2d[:, 0],
        umap_y=coords_2d[:, 1],
        cluster_ids=cluster_ids,
    )


def save_feature_atlas(
    stats: List[FeatureStats],
    geometry: FeatureGeometry,
    output_path: Union[str, Path],
    top_examples: Optional[List[TopExample]] = None,
    extra_columns: Optional[dict] = None,
    labels: Optional[List[str]] = None,
) -> None:
    """Save combined feature atlas for visualization dashboard.

    Combines feature statistics and UMAP geometry into a single Parquet file
    compatible with the mosaic-linked-viz dashboard.

    Args:
        stats: List of FeatureStats from compute_feature_stats or TokenActivationCollector
        geometry: FeatureGeometry from compute_feature_umap
        output_path: Path to save the atlas Parquet file
        top_examples: Optional list of TopExample to include top example indices
        extra_columns: Optional dict of additional columns to include
        labels: Optional list of custom labels for each feature. If provided,
            uses these instead of default "Feature {id}" labels. Use this for
            domain-specific labels (e.g., from F1 scores, auto-interp, etc.)

    Example:
        >>> # Default labels ("Feature 0", "Feature 1", ...)
        >>> save_feature_atlas(stats, geometry, "atlas.parquet")

        >>> # With custom labels
        >>> save_feature_atlas(stats, geometry, "atlas.parquet",
        ...     labels=["kinase detector", "membrane helix", ...])
    """
    if not HAS_PARQUET:
        raise ImportError("pyarrow required. Install with: pip install pyarrow")

    n_features = len(stats)

    # Generate labels
    if labels is not None:
        if len(labels) != n_features:
            raise ValueError(f"labels has {len(labels)} items, expected {n_features}")
        feature_labels = list(labels)
    else:
        feature_labels = [f"Feature {s.feature_id}" for s in stats]

    # Build base data from stats
    data = {
        "feature_id": [s.feature_id for s in stats],
        "label": feature_labels,
        "activation_freq": [s.activation_freq for s in stats],
        "mean_activation": [s.mean_activation for s in stats],
        "max_activation": [s.max_activation for s in stats],
        "std_activation": [s.std_activation for s in stats],
        "total_activations": [s.total_activations for s in stats],
    }

    # Add computed columns
    data["log_frequency"] = [np.log10(s.activation_freq) if s.activation_freq > 0 else -10.0 for s in stats]

    # Add geometry
    data["x"] = geometry.umap_x.tolist()
    data["y"] = geometry.umap_y.tolist()
    if geometry.cluster_ids is not None:
        # HDBSCAN assigns -1 to noise points; DuckDB's UTINYINT (used by
        # embedding-atlas for the category column) can't represent negatives.
        # Replace -1 with None so they become NULL in the parquet file.
        data["cluster_id"] = [int(c) if c >= 0 else None for c in geometry.cluster_ids.tolist()]
    else:
        data["cluster_id"] = [0] * n_features

    # Add top example info if provided
    if top_examples:
        # Group by feature_id, take first (highest activation)
        top_by_feature = {}
        for ex in top_examples:
            if ex.feature_id not in top_by_feature:
                top_by_feature[ex.feature_id] = ex

        data["top_example_idx"] = [
            top_by_feature[i].example_idx if i in top_by_feature else None for i in range(n_features)
        ]
        data["top_example_activation"] = [
            top_by_feature[i].activation_value if i in top_by_feature else None for i in range(n_features)
        ]

    # Add any extra columns (domain-specific data like F1 scores, annotations, etc.)
    if extra_columns:
        for col_name, col_data in extra_columns.items():
            if len(col_data) != n_features:
                raise ValueError(f"Extra column '{col_name}' has {len(col_data)} rows, expected {n_features}")
            data[col_name] = list(col_data)

    table = pa.table(data)
    pq.write_table(table, str(output_path), compression="snappy")
    print(f"Saved feature atlas ({n_features} features) to {output_path}")


def export_text_features_parquet(
    collector_result,
    output_dir: Union[str, Path],
    feature_logits=None,
    descriptions: Optional[Dict[int, str]] = None,
    n_examples: int = 5,
) -> None:
    """Export feature data as Parquet files for the text UMAP dashboard.

    Produces two files that the dashboard loads via DuckDB-WASM:
    - feature_metadata.parquet: one row per feature with stats and logits
    - feature_examples.parquet: one row per (feature, example text), sorted by
      feature_id for efficient row-group pushdown queries.

    Args:
        collector_result: Result from TokenActivationCollector.collect()
        output_dir: Directory to save parquet files
        feature_logits: Optional list of FeatureLogits objects
        descriptions: Optional dict mapping feature_id -> description string
        n_examples: Number of top texts per feature (default: 5)
    """
    import json as _json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_features = len(collector_result.feature_stats)
    logits_map = {}
    if feature_logits:
        logits_map = {fl.feature_id: fl for fl in feature_logits}

    # Build metadata rows
    meta_feature_ids = []
    meta_descriptions = []
    meta_freqs = []
    meta_max_acts = []
    meta_pos_logits = []
    meta_neg_logits = []

    # Build example rows
    ex_feature_ids = []
    ex_ranks = []
    ex_text_idxs = []
    ex_max_acts = []
    ex_tokens_jsons = []

    # Sort features by frequency (descending) for export ordering
    feature_order = sorted(range(n_features), key=lambda i: -collector_result.feature_stats[i].activation_freq)

    for feature_idx in feature_order:
        stats = collector_result.feature_stats[feature_idx]

        # Metadata
        meta_feature_ids.append(feature_idx)
        meta_descriptions.append(
            descriptions.get(feature_idx, f"Feature {feature_idx}") if descriptions else f"Feature {feature_idx}"
        )
        meta_freqs.append(stats.activation_freq)
        meta_max_acts.append(stats.max_activation)

        fl = logits_map.get(feature_idx)
        meta_pos_logits.append(_json.dumps([[t, round(v, 3)] for t, v in fl.top_positive]) if fl else None)
        meta_neg_logits.append(_json.dumps([[t, round(v, 3)] for t, v in fl.top_negative]) if fl else None)

        # Examples: get top texts by text_codes
        text_acts = collector_result.text_codes[:, feature_idx]
        top_text_indices = torch.argsort(text_acts, descending=True)

        rank = 0
        for text_idx in top_text_indices[:n_examples]:
            text_idx_val = text_idx.item()
            max_act = text_acts[text_idx_val].item()
            if max_act <= 0:
                break

            text_labels = collector_result.get_text_labels(text_idx_val)
            text_codes = collector_result.get_text_codes(text_idx_val)

            if text_codes is not None:
                acts = text_codes[:, feature_idx]
                token_data = [
                    {"token": tok, "activation": round(act.item(), 4)} for tok, act in zip(text_labels, acts)
                ]
            else:
                token_data = [{"token": tok, "activation": 0.0} for tok in text_labels]

            ex_feature_ids.append(feature_idx)
            ex_ranks.append(rank)
            ex_text_idxs.append(text_idx_val)
            ex_max_acts.append(max_act)
            ex_tokens_jsons.append(_json.dumps(token_data))
            rank += 1

    # Write feature_metadata.parquet
    meta_table = pa.table(
        {
            "feature_id": pa.array(meta_feature_ids, type=pa.int32()),
            "description": pa.array(meta_descriptions, type=pa.utf8()),
            "activation_freq": pa.array(meta_freqs, type=pa.float32()),
            "max_activation": pa.array(meta_max_acts, type=pa.float32()),
            "top_positive_logits_json": pa.array(meta_pos_logits, type=pa.utf8()),
            "top_negative_logits_json": pa.array(meta_neg_logits, type=pa.utf8()),
        }
    )
    meta_path = output_dir / "feature_metadata.parquet"
    pq.write_table(meta_table, str(meta_path))
    print(f"Saved {n_features} feature metadata to {meta_path}")

    # Write feature_examples.parquet (sorted by feature_id)
    examples_table = pa.table(
        {
            "feature_id": pa.array(ex_feature_ids, type=pa.int32()),
            "example_rank": pa.array(ex_ranks, type=pa.int8()),
            "text_idx": pa.array(ex_text_idxs, type=pa.int32()),
            "max_activation": pa.array(ex_max_acts, type=pa.float32()),
            "tokens_json": pa.array(ex_tokens_jsons, type=pa.utf8()),
        }
    )
    examples_table = examples_table.sort_by("feature_id")
    examples_path = output_dir / "feature_examples.parquet"
    pq.write_table(examples_table, str(examples_path), row_group_size=n_examples * 100)
    print(f"Saved {len(ex_feature_ids)} feature examples to {examples_path}")


def launch_dashboard(
    data_path: Union[str, Path],
    features_dir: Union[str, Path, None] = None,
    cluster_labels_path: Union[str, Path, None] = None,
    viz_dir: Union[str, Path, None] = None,
    port: int = 5173,
):
    """Launch the feature explorer dashboard.

    Args:
        data_path: Path to the features_atlas.parquet file to visualize
        features_dir: Directory containing feature_metadata.parquet and
            feature_examples.parquet. If None, looks in same directory as data_path.
        cluster_labels_path: Optional path to cluster_labels.json.
        viz_dir: Path to the visualization directory. If None, uses the
            bundled dashboard from sae package.
        port: Port for the dev server (default: 5173)

    Returns:
        subprocess.Popen process (call .terminate() to stop)

    Example:
        >>> proc = launch_dashboard("./outputs/features_atlas.parquet")
        >>> proc.terminate()  # when done
    """
    import shutil
    import subprocess
    import time
    import webbrowser

    data_path = Path(data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if features_dir is None:
        features_dir = data_path.parent
    else:
        features_dir = Path(features_dir).resolve()

    # Find viz directory
    if viz_dir is None:
        viz_dir = Path(__file__).parent / "dashboard"
        if not viz_dir.exists():
            raise FileNotFoundError(
                f"Could not find bundled dashboard at {viz_dir}. Please specify viz_dir explicitly."
            )
    else:
        viz_dir = Path(viz_dir)

    if not (viz_dir / "package.json").exists():
        raise FileNotFoundError(f"No package.json found in {viz_dir}. Is this the right directory?")

    # Copy data files to public directory so Vite can serve them
    public_dir = viz_dir / "public"
    public_dir.mkdir(exist_ok=True)

    # Copy atlas parquet
    dest_parquet = public_dir / "features_atlas.parquet"
    if data_path != dest_parquet:
        shutil.copy2(data_path, dest_parquet)
        print(f"Copied {data_path} -> {dest_parquet}")

    # Copy feature parquet files
    for fname in ["feature_metadata.parquet", "feature_examples.parquet"]:
        src = features_dir / fname
        if src.exists():
            dest = public_dir / fname
            if src != dest:
                shutil.copy2(src, dest)
                print(f"Copied {src} -> {dest}")
        else:
            print(f"Warning: {fname} not found at {src}")

    # Find and copy cluster_labels.json
    if cluster_labels_path is None:
        cluster_labels_path = data_path.parent / "cluster_labels.json"
    else:
        cluster_labels_path = Path(cluster_labels_path).resolve()

    if cluster_labels_path.exists():
        dest_labels = public_dir / "cluster_labels.json"
        if cluster_labels_path != dest_labels:
            shutil.copy2(cluster_labels_path, dest_labels)
            print(f"Copied {cluster_labels_path} -> {dest_labels}")

    print(f"Starting dashboard server in {viz_dir}...")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=viz_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)

    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"Dashboard running at {url}")
    print(f"Visualizing: {data_path}")
    print("Call proc.terminate() or restart kernel to stop")

    return proc
