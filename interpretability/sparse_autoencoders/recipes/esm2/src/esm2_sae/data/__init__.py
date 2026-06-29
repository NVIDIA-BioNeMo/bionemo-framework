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

"""Protein data loading and processing."""

from .annotations import (
    download_annotated_proteins,
    load_annotations_tsv,
    proteins_to_concept_labels,
)
from .dataset import ProteinDataset
from .fasta import read_fasta
from .types import ProteinRecord
from .uniprot import download_swissprot, download_uniref50


__all__ = [
    "ProteinDataset",
    "ProteinRecord",
    "download_annotated_proteins",
    "download_swissprot",
    "download_uniref50",
    "load_annotations_tsv",
    "proteins_to_concept_labels",
    "read_fasta",
]
