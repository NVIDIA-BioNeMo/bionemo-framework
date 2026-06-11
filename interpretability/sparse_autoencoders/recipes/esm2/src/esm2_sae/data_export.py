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

"""Data export utilities for saving activations and feature analysis to Parquet/DuckDB.

This module provides simple functions to save:
1. Raw activations (codes) from the SAE
2. Feature metadata (statistics, examples, annotations)

Both can be saved to Parquet (for portable storage) or DuckDB (for SQL queries).
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch


try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

try:
    import duckdb

    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class ActivationsExporter:
    """Export SAE activations to Parquet or DuckDB.

    Example:
        >>> exporter = ActivationsExporter()
        >>> codes = sae.encode(embeddings)  # [batch, hidden_dim]
        >>> protein_ids = ["protein1", "protein2", ...]
        >>>
        >>> # Save to Parquet
        >>> exporter.save_to_parquet(
        ...     codes=codes,
        ...     protein_ids=protein_ids,
        ...     output_path="activations.parquet"
        ... )
        >>>
        >>> # Or save to DuckDB
        >>> exporter.save_to_duckdb(
        ...     codes=codes,
        ...     protein_ids=protein_ids,
        ...     db_path="data.duckdb",
        ...     table_name="activations"
        ... )
    """

    def __init__(self):
        """Initialize the exporter and check for optional dependencies."""
        if not HAS_PARQUET:
            print("Warning: pyarrow not installed. Install with: pip install pyarrow")
        if not HAS_DUCKDB:
            print("Warning: duckdb not installed. Install with: pip install duckdb")

    def save_to_parquet(
        self,
        codes: Union[torch.Tensor, np.ndarray],
        protein_ids: List[str],
        output_path: Union[str, Path],
        residue_indices: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        compress: bool = True,
    ) -> None:
        """Save activations to Parquet format.

        Args:
            codes: Activation codes [batch, hidden_dim]
            protein_ids: List of protein identifiers
            output_path: Path to save Parquet file
            residue_indices: Optional residue positions for each activation
            metadata: Optional metadata dict to include
            compress: Whether to compress with Snappy
        """
        if not HAS_PARQUET:
            raise ImportError("pyarrow required. Install with: pip install pyarrow")

        # Convert to numpy
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        # Build data dict
        data = {
            "protein_id": protein_ids,
        }

        if residue_indices is not None:
            data["residue_idx"] = residue_indices

        # Add activation columns (sparse format: only non-zero features)
        # This is more efficient than storing the full dense matrix
        batch_size, hidden_dim = codes.shape

        # Find non-zero activations
        nonzero_rows, nonzero_cols = np.nonzero(codes)
        nonzero_values = codes[nonzero_rows, nonzero_cols]

        # Create sparse representation
        sparse_data = {
            "protein_id": [protein_ids[i] for i in nonzero_rows],
            "feature_id": nonzero_cols.tolist(),
            "activation": nonzero_values.tolist(),
        }

        if residue_indices is not None:
            sparse_data["residue_idx"] = [residue_indices[i] for i in nonzero_rows]

        # Create table
        table = pa.table(sparse_data)

        # Add metadata if provided
        if metadata:
            existing_metadata = table.schema.metadata or {}
            existing_metadata.update({k.encode(): str(v).encode() for k, v in metadata.items()})
            table = table.replace_schema_metadata(existing_metadata)

        # Write to file
        compression = "snappy" if compress else None
        pq.write_table(table, str(output_path), compression=compression)

        print(f"✓ Saved {len(nonzero_rows)} non-zero activations to {output_path}")
        print(f"  Sparsity: {100 * (1 - len(nonzero_rows) / (batch_size * hidden_dim)):.1f}%")

    def save_to_duckdb(
        self,
        codes: Union[torch.Tensor, np.ndarray],
        protein_ids: List[str],
        db_path: Union[str, Path],
        table_name: str = "activations",
        residue_indices: Optional[List[int]] = None,
        if_exists: str = "replace",
    ) -> None:
        """Save activations to DuckDB database.

        Args:
            codes: Activation codes [batch, hidden_dim]
            protein_ids: List of protein identifiers
            db_path: Path to DuckDB database file
            table_name: Name of table to create
            residue_indices: Optional residue positions
            if_exists: What to do if table exists ('replace', 'append', 'fail')
        """
        if not HAS_DUCKDB:
            raise ImportError("duckdb required. Install with: pip install duckdb")

        # Convert to numpy
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()

        batch_size, hidden_dim = codes.shape

        # Find non-zero activations (sparse format)
        nonzero_rows, nonzero_cols = np.nonzero(codes)
        nonzero_values = codes[nonzero_rows, nonzero_cols]

        # Create sparse representation
        data = {
            "protein_id": [protein_ids[i] for i in nonzero_rows],
            "feature_id": nonzero_cols.tolist(),
            "activation": nonzero_values.tolist(),
        }

        if residue_indices is not None:
            data["residue_idx"] = [residue_indices[i] for i in nonzero_rows]

        # Convert to records
        import pandas as pd

        activations_df = pd.DataFrame(data)  # noqa: F841 - referenced by DuckDB SQL

        # Write to DuckDB
        con = duckdb.connect(str(db_path))

        if if_exists == "replace":
            con.execute(f"DROP TABLE IF EXISTS {table_name}")

        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM activations_df")

        if if_exists == "append":
            con.execute(f"INSERT INTO {table_name} SELECT * FROM activations_df")

        con.close()

        print(f"✓ Saved {len(nonzero_rows)} non-zero activations to {db_path}::{table_name}")
        print(f"  Sparsity: {100 * (1 - len(nonzero_rows) / (batch_size * hidden_dim)):.1f}%")


class FeatureDataExporter:
    """Export feature metadata (stats, examples, annotations) to Parquet or DuckDB.

    Example:
        >>> exporter = FeatureDataExporter()
        >>>
        >>> # Compute feature statistics
        >>> stats = compute_feature_statistics(codes, protein_ids)
        >>> examples = get_top_examples(codes, sequences)
        >>> annotations = get_feature_annotations(codes, concept_labels)
        >>>
        >>> # Save all feature data
        >>> exporter.save_feature_data(
        ...     stats=stats,
        ...     examples=examples,
        ...     annotations=annotations,
        ...     output_dir="feature_data/"
        ... )
    """

    def __init__(self):
        """Initialize the exporter and check for optional dependencies."""
        if not HAS_PARQUET:
            print("Warning: pyarrow not installed. Install with: pip install pyarrow")
        if not HAS_DUCKDB:
            print("Warning: duckdb not installed. Install with: pip install duckdb")

    def save_feature_data(
        self,
        output_dir: Union[str, Path],
        stats: Optional[List] = None,
        examples: Optional[List] = None,
        annotations: Optional[List] = None,
        geometry: Optional[Any] = None,
        format: str = "parquet",
    ) -> None:
        """Save all feature data to a directory.

        Args:
            output_dir: Directory to save files
            stats: List of FeatureStats objects
            examples: List of FeatureExample objects
            annotations: List of FeatureAnnotation objects
            geometry: FeatureGeometry object (UMAP coordinates)
            format: Output format ('parquet' or 'duckdb')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "parquet":
            self._save_to_parquet(output_dir, stats, examples, annotations, geometry)
        elif format == "duckdb":
            self._save_to_duckdb(output_dir, stats, examples, annotations, geometry)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"\n✓ Saved feature data to {output_dir}/")

    def _save_to_parquet(self, output_dir, stats, examples, annotations, geometry):
        """Save to separate Parquet files."""
        from .viz.io import save_examples, save_geometry, save_stats

        if stats is not None:
            save_stats(stats, output_dir / "stats.parquet")
            print(f"  - stats.parquet ({len(stats)} features)")

        if examples is not None:
            save_examples(examples, output_dir / "examples.parquet")
            print(f"  - examples.parquet ({len(examples)} examples)")

        if geometry is not None:
            save_geometry(geometry, output_dir / "geometry.parquet")
            print(f"  - geometry.parquet ({len(geometry.feature_ids)} features)")

    def _save_to_duckdb(self, output_dir, stats, examples, annotations, geometry):
        """Save to single DuckDB file."""
        if not HAS_DUCKDB:
            raise ImportError("duckdb required. Install with: pip install duckdb")

        import pandas as pd

        db_path = output_dir / "features.duckdb"
        con = duckdb.connect(str(db_path))

        if stats is not None:
            stats_df = pd.DataFrame(  # noqa: F841 - referenced by DuckDB SQL
                [
                    {
                        "feature_id": s.feature_id,
                        "activation_frequency": s.activation_frequency,
                        "mean_activation": s.mean_activation,
                        "max_activation": s.max_activation,
                        "n_proteins_active": s.n_proteins_active,
                    }
                    for s in stats
                ]
            )
            con.execute("CREATE TABLE IF NOT EXISTS stats AS SELECT * FROM stats_df")
            print(f"  - stats table ({len(stats)} features)")

        if examples is not None:
            examples_df = pd.DataFrame(  # noqa: F841 - referenced by DuckDB SQL
                [
                    {
                        "feature_id": e.feature_id,
                        "protein_id": e.protein_id,
                        "residue_idx": e.residue_idx,
                        "activation_value": e.activation_value,
                        "sequence_window": e.sequence_window,
                        "window_start": e.window_start,
                    }
                    for e in examples
                ]
            )
            con.execute("CREATE TABLE IF NOT EXISTS examples AS SELECT * FROM examples_df")
            print(f"  - examples table ({len(examples)} examples)")

        if annotations is not None:
            annotations_df = pd.DataFrame(  # noqa: F841 - referenced by DuckDB SQL
                [
                    {
                        "feature_id": a.feature_id,
                        "best_annotation": a.best_annotation,
                        "best_f1": a.best_f1,
                        "best_precision": a.best_precision,
                        "best_recall": a.best_recall,
                        "best_threshold": a.best_threshold,
                    }
                    for a in annotations
                ]
            )
            con.execute("CREATE TABLE IF NOT EXISTS annotations AS SELECT * FROM annotations_df")
            print(f"  - annotations table ({len(annotations)} features)")

        if geometry is not None:
            data = {
                "feature_id": geometry.feature_ids,
                "umap_x": geometry.umap_x,
                "umap_y": geometry.umap_y,
            }
            if geometry.cluster_ids is not None:
                data["cluster_id"] = geometry.cluster_ids
            geometry_df = pd.DataFrame(data)  # noqa: F841 - referenced by DuckDB SQL
            con.execute("CREATE TABLE IF NOT EXISTS geometry AS SELECT * FROM geometry_df")
            print(f"  - geometry table ({len(geometry.feature_ids)} features)")

        con.close()
        print(f"\n  DuckDB database: {db_path}")


# Convenience functions
def save_activations_parquet(
    codes: Union[torch.Tensor, np.ndarray], protein_ids: List[str], output_path: Union[str, Path], **kwargs
) -> None:
    """Quick function to save activations to Parquet.

    Example:
        >>> codes = sae.encode(embeddings)
        >>> save_activations_parquet(codes, protein_ids, "activations.parquet")
    """
    exporter = ActivationsExporter()
    exporter.save_to_parquet(codes, protein_ids, output_path, **kwargs)


def save_activations_duckdb(
    codes: Union[torch.Tensor, np.ndarray],
    protein_ids: List[str],
    db_path: Union[str, Path],
    table_name: str = "activations",
    **kwargs,
) -> None:
    """Quick function to save activations to DuckDB.

    Example:
        >>> codes = sae.encode(embeddings)
        >>> save_activations_duckdb(codes, protein_ids, "data.duckdb")
    """
    exporter = ActivationsExporter()
    exporter.save_to_duckdb(codes, protein_ids, db_path, table_name, **kwargs)


def save_feature_data(output_dir: Union[str, Path], format: str = "parquet", **data) -> None:
    """Quick function to save feature data.

    Example:
        >>> save_feature_data(
        ...     "feature_data/",
        ...     stats=feature_stats,
        ...     examples=top_examples,
        ...     annotations=feature_annotations,
        ...     format="parquet"  # or "duckdb"
        ... )
    """
    exporter = FeatureDataExporter()
    exporter.save_feature_data(output_dir, format=format, **data)


def build_dashboard_data(
    sae: torch.nn.Module,
    activations_flat: torch.Tensor,
    activations: torch.Tensor,
    sequences: List[str],
    protein_ids: List[str],
    output_dir: Union[str, Path],
    masks: Optional[torch.Tensor] = None,
    n_examples: int = 6,
    device: str = "cpu",
) -> Tuple[Path, Path]:
    """Build all data files needed by the protein dashboard.

    Runs four steps:
    1. Compute per-feature activation statistics
    2. Compute UMAP from decoder weights
    3. Save feature atlas parquet (stats + UMAP)
    4. Export protein examples parquet (per-residue activations for top proteins)

    Args:
        sae: Trained SAE model
        activations_flat: Flattened residue activations (n_residues, hidden_dim)
        activations: Padded 3D activations (n_proteins, seq_len, hidden_dim)
        sequences: List of amino acid sequences
        protein_ids: List of protein accessions
        output_dir: Root directory for all output files
        masks: Optional validity masks (n_proteins, seq_len)
        n_examples: Number of top proteins per feature (default: 6)
        device: Compute device

    Returns:
        Tuple of (atlas_path, features_dir) for use with launch_protein_dashboard
    """
    from sae.analysis import compute_feature_stats, compute_feature_umap, save_feature_atlas

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas_path = output_dir / "features_atlas.parquet"
    features_dir = output_dir
    t_total = time.time()

    n_features = sae.hidden_dim
    n_residues = activations_flat.shape[0]
    n_proteins = len(sequences)
    print(f"\n{'=' * 60}")
    print("Building dashboard data")
    print(f"  {n_features:,} features | {n_residues:,} residues | {n_proteins:,} proteins")
    print(f"{'=' * 60}\n")

    # Step 1: Feature statistics
    print("[1/4] Computing feature statistics...")
    t0 = time.time()
    stats, _ = compute_feature_stats(sae, activations_flat, device=device)
    print(f"       Done in {time.time() - t0:.1f}s — {len(stats)} features\n")

    # Step 2: UMAP from decoder weights
    print("[2/4] Computing UMAP from decoder weights...")
    t0 = time.time()
    geometry = compute_feature_umap(sae, random_state=42)
    print(f"       Done in {time.time() - t0:.1f}s\n")

    # Step 3: Save feature atlas
    print("[3/4] Saving feature atlas...")
    t0 = time.time()
    save_feature_atlas(stats, geometry, atlas_path)
    print(f"       Saved to {atlas_path} in {time.time() - t0:.1f}s\n")

    # Step 4: Export protein examples
    print("[4/4] Exporting protein examples...")
    t0 = time.time()
    export_protein_features_parquet(
        sae=sae,
        activations=activations,
        sequences=sequences,
        protein_ids=protein_ids,
        output_dir=features_dir,
        masks=masks,
        n_examples=n_examples,
        device=device,
    )
    print(f"       Done in {time.time() - t0:.1f}s\n")

    print(f"{'=' * 60}")
    print(f"Dashboard data ready in {time.time() - t_total:.1f}s")
    print(f"  Atlas:    {atlas_path}")
    print(f"  Examples: {features_dir}")
    print(f"{'=' * 60}\n")

    return atlas_path, features_dir


def export_protein_features_parquet(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    sequences: List[str],
    protein_ids: List[str],
    output_dir: Union[str, Path],
    masks: Optional[torch.Tensor] = None,
    n_examples: int = 6,
    feature_stats: Optional[Dict[int, Dict]] = None,
    device: str = "cpu",
) -> None:
    """Export feature data as Parquet files for the protein UMAP dashboard.

    Produces two files that the dashboard loads via DuckDB-WASM:
    - feature_metadata.parquet: one row per feature (id, description, freq, etc.)
    - feature_examples.parquet: one row per (feature, example protein), sorted by
      feature_id for efficient row-group pushdown queries.

    Args:
        sae: Trained SAE model
        activations: Shape (n_proteins, seq_len, hidden_dim) - raw embeddings
        sequences: List of amino acid sequences
        protein_ids: List of protein accessions
        output_dir: Directory to save parquet files
        masks: Optional validity masks (n_proteins, seq_len)
        n_examples: Number of top proteins per feature (default: 6)
        feature_stats: Optional dict with activation_freq, best_f1, best_annotation per feature
        device: Compute device
    """
    if not HAS_PARQUET:
        raise ImportError("pyarrow required. Install with: pip install pyarrow")

    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sae = sae.eval().to(device)
    n_proteins, seq_len, _hidden_dim = activations.shape
    n_features = sae.hidden_dim

    # Precompute valid lengths once
    valid_lens = []
    for i in range(n_proteins):
        if masks is not None:
            valid_lens.append(masks[i].sum().int().item())
        else:
            valid_lens.append(seq_len)

    # Pass 1: Compute max activation per (protein, feature) pair.
    # Only stores the scalar max per feature — NOT the full per-residue arrays.
    # Memory: n_proteins * n_features * 4 bytes (e.g. 2000 * 20480 = ~160 MB)
    print("Pass 1: Computing per-protein max activations...")
    max_acts = np.zeros((n_proteins, n_features), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(n_proteins), desc="SAE encode (pass 1)"):
            emb = activations[i].to(device)
            acts = sae.encode(emb)  # (seq_len, n_features) — on device
            max_acts[i] = acts[: valid_lens[i]].max(dim=0).values.cpu().numpy()

    # For each feature, find top-N proteins by max activation
    print(f"Finding top {n_examples} proteins per feature...")
    feature_top = {}  # feat_idx -> [(prot_idx, max_act), ...]
    for feat_idx in range(n_features):
        col = max_acts[:, feat_idx]
        top_indices = np.argsort(col)[::-1][:n_examples]
        feature_top[feat_idx] = [(int(idx), float(col[idx])) for idx in top_indices if col[idx] > 0]

    # Build reverse index: which features need each protein's per-residue data
    protein_to_features: Dict[int, set] = {}
    for feat_idx, tops in feature_top.items():
        for prot_idx, _ in tops:
            protein_to_features.setdefault(prot_idx, set()).add(feat_idx)

    # Pass 2: Re-encode only needed proteins, extract per-residue activations
    # for only the relevant features. One protein at a time — constant memory.
    print(f"Pass 2: Re-encoding {len(protein_to_features)} proteins for per-residue activations...")
    example_acts: Dict[tuple, np.ndarray] = {}  # (prot_idx, feat_idx) -> 1-D numpy array
    with torch.no_grad():
        for prot_idx in tqdm(sorted(protein_to_features), desc="SAE encode (pass 2)"):
            emb = activations[prot_idx].to(device)
            acts = sae.encode(emb).cpu().numpy()  # (seq_len, n_features)
            vl = valid_lens[prot_idx]
            for feat_idx in protein_to_features[prot_idx]:
                example_acts[(prot_idx, feat_idx)] = acts[:vl, feat_idx].copy()

    # Build metadata and example rows
    print(f"Building parquet data ({n_features} features, {n_examples} examples each)...")
    metadata_rows = []
    example_rows = []

    for feat_idx in range(n_features):
        global_max = float(max_acts[:, feat_idx].max())

        stats = feature_stats.get(feat_idx, {}) if feature_stats else {}
        metadata_rows.append(
            {
                "feature_id": feat_idx,
                "description": stats.get("best_annotation") or f"Feature {feat_idx}",
                "activation_freq": stats.get("activation_freq", 0.0),
                "max_activation": global_max if global_max > 0 else 0.0,
                "best_f1": stats.get("best_f1"),
                "best_annotation": stats.get("best_annotation"),
            }
        )

        for rank, (prot_idx, max_act) in enumerate(feature_top[feat_idx]):
            acts_arr = example_acts[(prot_idx, feat_idx)]
            seq = sequences[prot_idx][: valid_lens[prot_idx]]
            pid = protein_ids[prot_idx]
            accession = pid.split("|")[1] if "|" in pid else pid

            example_rows.append(
                {
                    "feature_id": feat_idx,
                    "example_rank": rank,
                    "protein_id": pid,
                    "alphafold_id": f"AF-{accession}-F1",
                    "sequence": seq,
                    "activations": acts_arr.tolist(),
                    "max_activation": float(max_act),
                }
            )

    del max_acts, example_acts  # free memory

    # Write feature_metadata.parquet
    meta_table = pa.table(
        {
            "feature_id": pa.array([r["feature_id"] for r in metadata_rows], type=pa.int32()),
            "description": pa.array([r["description"] for r in metadata_rows], type=pa.utf8()),
            "activation_freq": pa.array([r["activation_freq"] for r in metadata_rows], type=pa.float32()),
            "max_activation": pa.array([r["max_activation"] for r in metadata_rows], type=pa.float32()),
            "best_f1": pa.array([r["best_f1"] for r in metadata_rows], type=pa.float32()),
            "best_annotation": pa.array([r["best_annotation"] for r in metadata_rows], type=pa.utf8()),
        }
    )
    meta_path = output_dir / "feature_metadata.parquet"
    pq.write_table(meta_table, str(meta_path))
    print(f"  Saved {len(metadata_rows)} features to {meta_path}")

    # Write feature_examples.parquet (sorted by feature_id for row-group pushdown)
    activations_type = pa.list_(pa.float32())
    examples_table = pa.table(
        {
            "feature_id": pa.array([r["feature_id"] for r in example_rows], type=pa.int32()),
            "example_rank": pa.array([r["example_rank"] for r in example_rows], type=pa.int8()),
            "protein_id": pa.array([r["protein_id"] for r in example_rows], type=pa.utf8()),
            "alphafold_id": pa.array([r["alphafold_id"] for r in example_rows], type=pa.utf8()),
            "sequence": pa.array([r["sequence"] for r in example_rows], type=pa.utf8()),
            "activations": pa.array([r["activations"] for r in example_rows], type=activations_type),
            "max_activation": pa.array([r["max_activation"] for r in example_rows], type=pa.float32()),
        }
    )
    examples_table = examples_table.sort_by("feature_id")
    examples_path = output_dir / "feature_examples.parquet"
    pq.write_table(examples_table, str(examples_path), row_group_size=n_examples * 100)
    print(f"  Saved {len(example_rows)} examples to {examples_path}")


def export_protein_features_json(
    sae: torch.nn.Module,
    activations: torch.Tensor,
    sequences: List[str],
    protein_ids: List[str],
    output_path: Union[str, Path],
    masks: Optional[torch.Tensor] = None,
    n_examples: int = 6,
    feature_stats: Optional[Dict[int, Dict]] = None,
    model_name: str = "esm2-8m",
    layer: int = 6,
    device: str = "cpu",
) -> None:
    """Deprecated: use export_protein_features_parquet() instead.

    Export features.json for the protein UMAP dashboard.
    For each feature, finds the top N proteins by max activation and includes
    full sequences with per-residue activations.

    Args:
        sae: Trained SAE model
        activations: Shape (n_proteins, seq_len, hidden_dim) - raw embeddings
        sequences: List of amino acid sequences
        protein_ids: List of protein accessions
        output_path: Path to save features.json
        masks: Optional validity masks (n_proteins, seq_len)
        n_examples: Number of top proteins per feature (default: 6)
        feature_stats: Optional dict with activation_freq, best_f1, best_annotation per feature
        model_name: Model name for metadata
        layer: Layer number for metadata
        device: Compute device
    """
    import json

    from tqdm import tqdm

    output_path = Path(output_path)

    sae = sae.eval().to(device)
    n_proteins, seq_len, _hidden_dim = activations.shape
    n_features = sae.hidden_dim

    # Precompute valid lengths
    valid_lens = []
    for i in range(n_proteins):
        if masks is not None:
            valid_lens.append(masks[i].sum().int().item())
        else:
            valid_lens.append(seq_len)

    # Pass 1: Compute max activation per (protein, feature) — streaming, constant memory
    print("Pass 1: Computing per-protein max activations...")
    max_acts = np.zeros((n_proteins, n_features), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm(range(n_proteins), desc="SAE encode (pass 1)"):
            emb = activations[i].to(device)
            acts = sae.encode(emb)
            max_acts[i] = acts[: valid_lens[i]].max(dim=0).values.cpu().numpy()

    # For each feature, find top-N proteins
    feature_top = {}
    for feat_idx in range(n_features):
        col = max_acts[:, feat_idx]
        top_indices = np.argsort(col)[::-1][:n_examples]
        feature_top[feat_idx] = [(int(idx), float(col[idx])) for idx in top_indices if col[idx] > 0]

    # Build reverse index: protein -> features needing per-residue data
    protein_to_features: Dict[int, set] = {}
    for feat_idx, tops in feature_top.items():
        for prot_idx, _ in tops:
            protein_to_features.setdefault(prot_idx, set()).add(feat_idx)

    # Pass 2: Re-encode needed proteins, extract per-residue activations
    print(f"Pass 2: Re-encoding {len(protein_to_features)} proteins for per-residue activations...")
    example_acts: Dict[tuple, np.ndarray] = {}
    with torch.no_grad():
        for prot_idx in tqdm(sorted(protein_to_features), desc="SAE encode (pass 2)"):
            emb = activations[prot_idx].to(device)
            acts = sae.encode(emb).cpu().numpy()
            vl = valid_lens[prot_idx]
            for feat_idx in protein_to_features[prot_idx]:
                example_acts[(prot_idx, feat_idx)] = acts[:vl, feat_idx].copy()

    # Build features list
    print(f"Building features.json ({n_features} features, {n_examples} examples each)...")
    features_list = []

    for feat_idx in tqdm(range(n_features), desc="Processing features"):
        global_max = float(max_acts[:, feat_idx].max())

        examples = []
        for prot_idx, max_act in feature_top[feat_idx]:
            acts_arr = example_acts[(prot_idx, feat_idx)]
            seq = sequences[prot_idx][: valid_lens[prot_idx]]
            pid = protein_ids[prot_idx]
            accession = pid.split("|")[1] if "|" in pid else pid

            examples.append(
                {
                    "protein_id": pid,
                    "alphafold_id": f"AF-{accession}-F1",
                    "sequence": seq,
                    "activations": acts_arr.tolist(),
                    "max_activation": float(max_act),
                }
            )

        stats = feature_stats.get(feat_idx, {}) if feature_stats else {}

        features_list.append(
            {
                "feature_id": feat_idx,
                "description": stats.get("best_annotation") or f"Feature {feat_idx}",
                "activation_freq": stats.get("activation_freq", 0.0),
                "max_activation": global_max if global_max > 0 else 0.0,
                "best_f1": stats.get("best_f1"),
                "best_annotation": stats.get("best_annotation"),
                "examples": examples,
            }
        )

    del max_acts, example_acts

    # Write features.json
    output_data = {
        "model": model_name,
        "layer": layer,
        "sae_hidden_dim": n_features,
        "features": features_list,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f)

    print(f"Exported {n_features} features ({n_examples} examples each) to {output_path}")


def get_protein_dashboard_dir() -> Path:
    """Get the path to the protein dashboard directory."""
    return Path(__file__).parent.parent.parent / "protein_dashboard"


def launch_protein_dashboard(
    data_path: Union[str, Path],
    features_dir: Optional[Union[str, Path]] = None,
    cluster_labels_path: Optional[Union[str, Path]] = None,
    port: int = 5176,
    clean_public: bool = True,
) -> "subprocess.Popen":
    """Launch the protein UMAP dashboard with Mol* structure viewers.

    Args:
        data_path: Path to features_atlas.parquet
        features_dir: Directory containing feature_metadata.parquet and
            feature_examples.parquet. If None, looks in same directory as data_path.
        cluster_labels_path: Optional path to cluster_labels.json.
        port: Port for the dev server (default: 5176)
        clean_public: Remove stale data files from public/ before copying (default: True)

    Returns:
        subprocess.Popen process (call .terminate() to stop)

    Example:
        >>> proc = launch_protein_dashboard("./outputs/features_atlas.parquet")
        >>> proc.terminate()  # when done
    """
    import shutil
    import time
    import webbrowser

    data_path = Path(data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if features_dir is None:
        features_dir = data_path.parent
    else:
        features_dir = Path(features_dir).resolve()

    viz_dir = get_protein_dashboard_dir()
    if not (viz_dir / "package.json").exists():
        raise FileNotFoundError(
            f"Could not find protein dashboard at {viz_dir}. Is the recipes/esm2/protein_dashboard directory present?"
        )

    public_dir = viz_dir / "public"
    public_dir.mkdir(exist_ok=True)

    if clean_public:
        for stale in [
            "features_atlas.parquet",
            "feature_metadata.parquet",
            "feature_examples.parquet",
            "features.json",
            "cluster_labels.json",
        ]:
            p = public_dir / stale
            if p.exists():
                p.unlink()

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

    # Copy cluster labels if available
    if cluster_labels_path is None:
        cluster_labels_path = data_path.parent / "cluster_labels.json"
    else:
        cluster_labels_path = Path(cluster_labels_path).resolve()

    if cluster_labels_path.exists():
        dest_labels = public_dir / "cluster_labels.json"
        if cluster_labels_path != dest_labels:
            shutil.copy2(cluster_labels_path, dest_labels)
            print(f"Copied {cluster_labels_path} -> {dest_labels}")

    # Install dependencies if needed
    if not (viz_dir / "node_modules").exists():
        print("Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=viz_dir, check=True)

    print(f"Starting protein dashboard at http://localhost:{port}")
    proc = subprocess.Popen(
        ["npm", "run", "dev", "--", "--port", str(port)],
        cwd=viz_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    time.sleep(2)
    url = f"http://localhost:{port}"
    webbrowser.open(url)
    print(f"Protein dashboard running at {url}")
    print("Call proc.terminate() or restart kernel to stop")

    return proc
