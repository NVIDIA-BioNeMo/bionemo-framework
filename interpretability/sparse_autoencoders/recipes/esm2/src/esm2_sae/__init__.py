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

"""ESM2 SAE Recipe: Sparse Autoencoders for ESM2 Protein Language Models.

This package provides ESM2-specific implementations for training and evaluating
Sparse Autoencoders on protein embeddings.

Main Components:
    - data: Protein datasets (FASTA, SwissProt, annotations)
    - eval: Biology-specific evaluation (F1 scores, comprehensive evaluation)
    - analysis: Protein ranking and interpretability
    - data_export: Data export and visualization pipeline
"""

from .data import (
    download_annotated_proteins,
    download_swissprot,
    download_uniref50,
    load_annotations_tsv,
    proteins_to_concept_labels,
    read_fasta,
)
from .data_export import (
    build_dashboard_data,
    export_protein_features_json,
    export_protein_features_parquet,
    launch_protein_dashboard,
    save_activations_duckdb,
    save_activations_parquet,
    save_feature_data,
)
from .eval import compute_f1_scores


__version__ = "0.1.0"

__all__ = [
    "build_dashboard_data",
    "compute_f1_scores",
    "download_annotated_proteins",
    "download_swissprot",
    "download_uniref50",
    "export_protein_features_json",
    "export_protein_features_parquet",
    "launch_protein_dashboard",
    "load_annotations_tsv",
    "proteins_to_concept_labels",
    "read_fasta",
    "save_activations_duckdb",
    "save_activations_parquet",
    "save_feature_data",
]
