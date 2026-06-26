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

"""Tests for ``bionemo.evo2_phage_gen.arc_pipeline``."""

from bionemo.evo2_phage_gen.arc_pipeline import (
    ARC_LEGACY_CHECKV_ENV,
    ARC_LEGACY_PRODIGAL_CMD,
    ARC_PIPELINE_FILES,
    PATCHED_CHECKV_ENV,
    PATCHED_PRODIGAL_CMD,
    prepare_arc_pipeline_workdir,
)
from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA


def test_prepare_arc_pipeline_workdir_patches_legacy_reference_path(tmp_path):
    """The prepared Arc workdir should not depend on Arc's legacy absolute path."""
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    for filename in ARC_PIPELINE_FILES:
        content = "print('ok')\n"
        if filename == "genetic_architecture.py":
            content = f'fasta_file = "{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"\n'
        if filename == "genome_design_filtering_pipeline.py":
            content = f"{ARC_LEGACY_PRODIGAL_CMD}\n{ARC_LEGACY_CHECKV_ENV}\n"
        (source_dir / filename).write_text(content)
    phix174_fasta = tmp_path / "NC_001422_1.fna"
    phix174_fasta.write_text(">NC_001422.1\nACGT\n")

    written_paths = prepare_arc_pipeline_workdir(
        source_dir,
        tmp_path / "patched",
        phix174_fasta=phix174_fasta,
    )

    assert [path.name for path in written_paths] == list(ARC_PIPELINE_FILES)
    patched_text = (tmp_path / "patched" / "genetic_architecture.py").read_text()
    assert ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA not in patched_text
    assert str(phix174_fasta) in patched_text

    pipeline_text = (tmp_path / "patched" / "genome_design_filtering_pipeline.py").read_text()
    assert ARC_LEGACY_PRODIGAL_CMD not in pipeline_text
    assert ARC_LEGACY_CHECKV_ENV not in pipeline_text
    assert PATCHED_PRODIGAL_CMD in pipeline_text
    assert PATCHED_CHECKV_ENV in pipeline_text
