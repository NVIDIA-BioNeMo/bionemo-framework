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

"""Prepare a runnable local copy of Arc's phage filtering pipeline."""

import argparse
import shutil
from pathlib import Path

from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARC_PIPELINE_SOURCE_DIR = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "pipelines"
DEFAULT_ARC_PIPELINE_WORKDIR = RECIPE_ROOT / "data" / "arc_pipeline_patched"
DEFAULT_PHIX174_FASTA = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "data" / "NC_001422_1.fna"
ARC_LEGACY_PRODIGAL_CMD = (
    "cmd = f'/home/samuelking/prodigal/prodigal -i {input_sequences} "
    "-d {output_orf_file} -a {output_protein_file} -p meta'"
)
PATCHED_PRODIGAL_CMD = "cmd = f'prodigal -i {input_sequences} -d {output_orf_file} -a {output_protein_file} -p meta'"
ARC_LEGACY_CHECKV_ENV = (
    "env = {**os.environ, 'CHECKVDB': \"/large_experiments/hielab/brianhie/dna-gen/checkv/checkv-db-v1.5\"}"
)
PATCHED_CHECKV_ENV = "env = os.environ.copy()"
ARC_PIPELINE_FILES = (
    "genome_design_filtering_pipeline.py",
    "genetic_architecture.py",
    "genetic_architecture_visualization.py",
)


def prepare_arc_pipeline_workdir(
    source_dir: Path = DEFAULT_ARC_PIPELINE_SOURCE_DIR,
    output_dir: Path = DEFAULT_ARC_PIPELINE_WORKDIR,
    *,
    phix174_fasta: Path = DEFAULT_PHIX174_FASTA,
    overwrite: bool = False,
) -> list[Path]:
    """Copy Arc pipeline files and patch the import-time PhiX174 FASTA path."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    phix174_fasta = Path(phix174_fasta)
    if not source_dir.exists():
        raise FileNotFoundError(f"Arc pipeline source directory not found: {source_dir}")
    if not phix174_fasta.exists():
        raise FileNotFoundError(f"PhiX174 FASTA not found: {phix174_fasta}")
    if output_dir.exists() and not overwrite:
        raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for filename in ARC_PIPELINE_FILES:
        src = source_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Required Arc pipeline file not found: {src}")
        dst = output_dir / filename
        shutil.copy2(src, dst)
        written_paths.append(dst)

    genetic_architecture_path = output_dir / "genetic_architecture.py"
    text = genetic_architecture_path.read_text()
    patched_text = text.replace(ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA, str(phix174_fasta))
    if patched_text == text:
        raise ValueError(
            f"Did not find expected legacy PhiX174 path in {genetic_architecture_path}: "
            f"{ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA}"
        )
    genetic_architecture_path.write_text(patched_text)

    filtering_pipeline_path = output_dir / "genome_design_filtering_pipeline.py"
    text = filtering_pipeline_path.read_text()
    patched_text = text.replace(ARC_LEGACY_PRODIGAL_CMD, PATCHED_PRODIGAL_CMD).replace(
        ARC_LEGACY_CHECKV_ENV, PATCHED_CHECKV_ENV
    )
    missing_patches = []
    if ARC_LEGACY_PRODIGAL_CMD in text and PATCHED_PRODIGAL_CMD not in patched_text:
        missing_patches.append("Prodigal command")
    if ARC_LEGACY_CHECKV_ENV in text and PATCHED_CHECKV_ENV not in patched_text:
        missing_patches.append("CheckV environment")
    if missing_patches:
        raise ValueError(f"Failed to patch {', '.join(missing_patches)} in {filtering_pipeline_path}")
    filtering_pipeline_path.write_text(patched_text)
    return written_paths


def main() -> None:
    """CLI entry point for preparing Arc's local pipeline workdir."""
    parser = argparse.ArgumentParser(description="Prepare a patched local copy of Arc's phage filtering pipeline")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_ARC_PIPELINE_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARC_PIPELINE_WORKDIR)
    parser.add_argument("--phix174-fasta", type=Path, default=DEFAULT_PHIX174_FASTA)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in prepare_arc_pipeline_workdir(
        args.source_dir,
        args.output_dir,
        phix174_fasta=args.phix174_fasta,
        overwrite=args.overwrite,
    ):
        print(path)


if __name__ == "__main__":
    main()
