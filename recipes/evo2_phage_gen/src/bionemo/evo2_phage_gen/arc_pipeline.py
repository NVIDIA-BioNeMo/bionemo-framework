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
import subprocess
from pathlib import Path

from bionemo.evo2_phage_gen.external_qc import ARC_GENETIC_ARCHITECTURE_IMPORT_FASTA


RECIPE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARC_PIPELINE_SOURCE_DIR = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "pipelines"
DEFAULT_ARC_PIPELINE_WORKDIR = RECIPE_ROOT / "data" / "arc_pipeline_patched"
DEFAULT_PHIX174_FASTA = RECIPE_ROOT / "data" / "external" / "arc_evo2" / "phage_gen" / "data" / "NC_001422_1.fna"
DEFAULT_ARC_PIPELINE_PATCH = RECIPE_ROOT / "patches" / "arc-evo2-genome-design-filtering.patch"
ARC_EVO2_GIT_URL = "https://github.com/ArcInstitute/evo2.git"
ARC_EVO2_REV = "53f195997257c56c00e5ef8d33a54f5baad143a6"
ARC_LEGACY_PRODIGAL_CMD = (
    "cmd = f'/home/samuelking/prodigal/prodigal -i {input_sequences} "
    "-d {output_orf_file} -a {output_protein_file} -p meta'"
)
PATCHED_PRODIGAL_CMD = "cmd = f'prodigal -i {input_sequences} -d {output_orf_file} -a {output_protein_file} -p meta'"
ARC_LEGACY_CHECKV_ENV = (
    "env = {**os.environ, 'CHECKVDB': \"/large_experiments/hielab/brianhie/dna-gen/checkv/checkv-db-v1.5\"}"
)
PATCHED_CHECKV_ENV = "env = os.environ.copy()"
ARC_LEGACY_LOVIS4U_CONDA_WRAPPER = '''def run_lovis4u_in_conda_env(env_name: str, command: str) -> None:
    """
    Activate a Conda environment and run a command within it.

    Args:
        env_name (str): The name of the Conda environment to activate.
        command (str): The command to run inside the activated environment.
    """
    try:
        # Full command to initialize Conda and activate environment before running the given command
        full_command = f"""
        eval "$(conda shell.bash hook)"
        conda activate {env_name}
        {command}
        """
        subprocess.run(full_command, shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running command in Conda environment {env_name}: {e}")
'''
PATCHED_LOVIS4U_CONDA_WRAPPER = '''def run_lovis4u_in_conda_env(env_name: str, command: str) -> None:
    """Run LoVis4u command in the active environment.

    The original Arc script activates a separate conda environment here. The
    recipe installs LoVis4u into its uv-managed venv, so the active PATH is the
    reproducible environment boundary.
    """
    try:
        subprocess.run(command, shell=True, executable="/bin/bash", check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running LoVis4u command with active environment instead of {env_name}: {e}")
        raise
'''
ARC_LEGACY_MMSEQS_EMPTY_GUARD_ANCHOR = """    # Drop the sequences column in mmseqs_df (if any) since we don't want to redundantly add it to sequences_df
    if 'sequence' in mmseqs_df.columns:
        mmseqs_df = mmseqs_df.drop(columns=['sequence'])
"""
PATCHED_MMSEQS_EMPTY_GUARD = """    if mmseqs_df.empty:
        sequences_df[f'valid_{descriptive_prefix}_pident'] = False
        sequences_df[f'{descriptive_prefix}_mmseqs_percent_identity'] = pd.NA
        return sequences_df[sequences_df[f'valid_{descriptive_prefix}_pident']]

    # Drop the sequences column in mmseqs_df (if any) since we don't want to redundantly add it to sequences_df
    if 'sequence' in mmseqs_df.columns:
        mmseqs_df = mmseqs_df.drop(columns=['sequence'])
"""
ARC_LEGACY_EMPTY_DIVERSIFICATION_ANCHOR = """            else:
                raise ValueError("Unsupported file format. Please provide a .fna or .fasta file.")
        filtered_df = seq_df.copy()
"""
PATCHED_EMPTY_DIVERSIFICATION_GUARD = """            else:
                raise ValueError("Unsupported file format. Please provide a .fna or .fasta file.")
        filtered_df = seq_df.copy()
        if len(filtered_df) == 0:
            print("No sequences available for diversification filtering; writing empty diversification outputs.")
            config["mmseqs_clustering_filter"] = False
            config["mmseqs_reference_genome_sequence_identity_remove_filter"] = False
            config["genetic_architecture_remove_filter"] = False
"""
ARC_LEGACY_EMPTY_ORF_ANCHOR = """        ### Initialize counts ###
        filter_counts_df['count_initial_before_orf_metrics'] = len(seq_df)
        print(f"Initializing ORF filtering. Sequences to filter: {filter_counts_df['count_initial_before_orf_metrics'].values[0]}.")

        ### Run Prodigal to call ORFs ###
"""
PATCHED_EMPTY_ORF_GUARD = """        ### Initialize counts ###
        filter_counts_df['count_initial_before_orf_metrics'] = len(seq_df)
        print(f"Initializing ORF filtering. Sequences to filter: {filter_counts_df['count_initial_before_orf_metrics'].values[0]}.")
        filtered_df = seq_df.copy()
        if len(filtered_df) == 0:
            print("No sequences available for ORF filtering; writing empty ORF outputs.")
            config["prodigal_based_filters"] = False

        ### Run Prodigal to call ORFs ###
"""
ARC_LEGACY_EMPTY_HOMOLOGY_ANCHOR = """        ### Run orfipy to call ORFs ###
        # Pseudo-circularize ORFs
        print(f"Pseudo-circularizing {len(seq_df)} genomes...")
        append_upstream_of_last_frame_stop(seq_fasta, f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}')
        # Call ORFs by orfipy
        print(f"Running orfipy on {len(seq_df)} genomes...")
        run_orfipy(f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}',
                    config["orfipy_threads"],
                    config["orfipy_start_codons"],
                    config["orfipy_stop_codons"],
                    config["orfipy_strand"],
                    config["orfipy_min_max_orf_lengths"][0],
                    config["orfipy_min_max_orf_lengths"][1],
                    config["results_save_dir"],
                    config["orfipy_orfs_file_save_location"],
                    config["orfipy_tmp_proteins_file_save_location"],
                    config["orfipy_proteins_file_save_location"])
"""
PATCHED_EMPTY_HOMOLOGY_GUARD = """        filtered_df = seq_df.copy()
        if len(filtered_df) == 0:
            print("No sequences available for homology filtering; writing empty homology outputs.")
            config["protein_database_hit_count_filter"] = False
            config["training_data_sequence_identity_filter"] = False
            config["checkv_filter"] = False
            config["reference_genome_sequence_identity_filter"] = False
            config["genetic_architecture_filter"] = False
            config["tropism_protein_sequence_identity_filter"] = False

        ### Run orfipy to call ORFs ###
        # Pseudo-circularize ORFs
        if len(filtered_df) > 0:
            print(f"Pseudo-circularizing {len(seq_df)} genomes...")
            append_upstream_of_last_frame_stop(seq_fasta, f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}')
            # Call ORFs by orfipy
            print(f"Running orfipy on {len(seq_df)} genomes...")
            run_orfipy(f'{config["results_save_dir"]}/{config["homology_filter_seqs_circular_fasta_file_save_location"]}',
                        config["orfipy_threads"],
                        config["orfipy_start_codons"],
                        config["orfipy_stop_codons"],
                        config["orfipy_strand"],
                        config["orfipy_min_max_orf_lengths"][0],
                        config["orfipy_min_max_orf_lengths"][1],
                        config["results_save_dir"],
                        config["orfipy_orfs_file_save_location"],
                        config["orfipy_tmp_proteins_file_save_location"],
                        config["orfipy_proteins_file_save_location"])
"""
ARC_LEGACY_EMPTY_SYNTENY_ANCHOR = """    ### Annotate & visualize genomes ###
    if config["genetic_architecture_visualization_and_synteny_filtering"] == True:
"""
PATCHED_EMPTY_SYNTENY_GUARD = """    ### Annotate & visualize genomes ###
    if config["genetic_architecture_visualization_and_synteny_filtering"] == True:
        if config["diversification_filtering"] == True:
            synteny_input_csv = f'{config["results_save_dir"]}/{config["diversification_filter_seqs_csv_file_save_location"]}'
            synteny_counts_csv = f'{config["results_save_dir"]}/{config["diversification_filter_counts_file_save_location"]}'
        else:
            synteny_input_csv = f'{config["results_save_dir"]}/{config["homology_filter_seqs_csv_file_save_location"]}'
            synteny_counts_csv = f'{config["results_save_dir"]}/{config["homology_filter_counts_file_save_location"]}'
        if os.path.exists(synteny_input_csv):
            synteny_preview_df = pd.read_csv(synteny_input_csv)
            if len(synteny_preview_df) == 0:
                print("No sequences available for genome visualization and synteny filtering; writing empty synteny outputs.")
                filter_counts = pd.read_csv(synteny_counts_csv)
                filter_counts.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_counts_file_save_location"]}', index=False)
                synteny_preview_df.to_csv(f'{config["results_save_dir"]}/{config["synteny_filter_seqs_csv_file_save_location"]}', index=False)
                save_df_as_fasta(synteny_preview_df, f'{config["results_save_dir"]}/{config["synteny_filter_seqs_fasta_file_save_location"]}')
                filtered_df = synteny_preview_df
                config["genetic_architecture_visualization_and_synteny_filtering"] = False

    if config["genetic_architecture_visualization_and_synteny_filtering"] == True:
"""
ARC_PIPELINE_FILES = (
    "genome_design_filtering_pipeline.py",
    "genetic_architecture.py",
    "genetic_architecture_visualization.py",
)


def _git_head(path: Path) -> str | None:
    """Return the Git HEAD for ``path`` when it is inside a Git checkout."""
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _assert_arc_source_revision(source_dir: Path, expected_revision: str) -> None:
    """Fail if the Arc source directory is not the pinned revision for the maintained patch."""
    actual_revision = _git_head(source_dir)
    if actual_revision is None:
        raise RuntimeError(
            f"Arc pipeline source {source_dir} is not in a Git checkout; "
            f"the maintained patch expects {ARC_EVO2_GIT_URL}@{expected_revision}."
        )
    if actual_revision != expected_revision:
        raise RuntimeError(
            f"Arc pipeline source revision mismatch for {source_dir}: expected "
            f"{ARC_EVO2_GIT_URL}@{expected_revision}, found {actual_revision}."
        )


def _apply_arc_pipeline_patch(output_dir: Path, patch_path: Path) -> None:
    """Apply the maintained Arc pipeline patch to a freshly copied workdir."""
    result = subprocess.run(
        ["patch", "--batch", "--forward", "--ignore-whitespace", "-p0", "-i", str(patch_path)],
        cwd=output_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to apply Arc pipeline patch {patch_path}:\n{result.stdout}")


def _apply_legacy_string_patches(filtering_pipeline_path: Path) -> None:
    """Apply small string patches used by synthetic tests that omit the maintained patch."""
    text = filtering_pipeline_path.read_text()
    patched_text = (
        text.replace(ARC_LEGACY_PRODIGAL_CMD, PATCHED_PRODIGAL_CMD)
        .replace(ARC_LEGACY_CHECKV_ENV, PATCHED_CHECKV_ENV)
        .replace(ARC_LEGACY_LOVIS4U_CONDA_WRAPPER, PATCHED_LOVIS4U_CONDA_WRAPPER)
        .replace(ARC_LEGACY_MMSEQS_EMPTY_GUARD_ANCHOR, PATCHED_MMSEQS_EMPTY_GUARD)
        .replace(ARC_LEGACY_EMPTY_ORF_ANCHOR, PATCHED_EMPTY_ORF_GUARD)
        .replace(ARC_LEGACY_EMPTY_HOMOLOGY_ANCHOR, PATCHED_EMPTY_HOMOLOGY_GUARD)
        .replace(ARC_LEGACY_EMPTY_DIVERSIFICATION_ANCHOR, PATCHED_EMPTY_DIVERSIFICATION_GUARD)
        .replace(ARC_LEGACY_EMPTY_SYNTENY_ANCHOR, PATCHED_EMPTY_SYNTENY_GUARD)
    )
    missing_patches = []
    if ARC_LEGACY_PRODIGAL_CMD in text and PATCHED_PRODIGAL_CMD not in patched_text:
        missing_patches.append("Prodigal command")
    if ARC_LEGACY_CHECKV_ENV in text and PATCHED_CHECKV_ENV not in patched_text:
        missing_patches.append("CheckV environment")
    if ARC_LEGACY_LOVIS4U_CONDA_WRAPPER in text and PATCHED_LOVIS4U_CONDA_WRAPPER not in patched_text:
        missing_patches.append("LoVis4u environment")
    if ARC_LEGACY_MMSEQS_EMPTY_GUARD_ANCHOR in text and PATCHED_MMSEQS_EMPTY_GUARD not in patched_text:
        missing_patches.append("empty MMseqs hit guard")
    if ARC_LEGACY_EMPTY_ORF_ANCHOR in text and PATCHED_EMPTY_ORF_GUARD not in patched_text:
        missing_patches.append("empty ORF guard")
    if ARC_LEGACY_EMPTY_HOMOLOGY_ANCHOR in text and PATCHED_EMPTY_HOMOLOGY_GUARD not in patched_text:
        missing_patches.append("empty homology guard")
    if ARC_LEGACY_EMPTY_DIVERSIFICATION_ANCHOR in text and PATCHED_EMPTY_DIVERSIFICATION_GUARD not in patched_text:
        missing_patches.append("empty diversification guard")
    if ARC_LEGACY_EMPTY_SYNTENY_ANCHOR in text and PATCHED_EMPTY_SYNTENY_GUARD not in patched_text:
        missing_patches.append("empty synteny guard")
    if missing_patches:
        raise ValueError(f"Failed to patch {', '.join(missing_patches)} in {filtering_pipeline_path}")
    filtering_pipeline_path.write_text(patched_text)


def prepare_arc_pipeline_workdir(
    source_dir: Path = DEFAULT_ARC_PIPELINE_SOURCE_DIR,
    output_dir: Path = DEFAULT_ARC_PIPELINE_WORKDIR,
    *,
    phix174_fasta: Path = DEFAULT_PHIX174_FASTA,
    pipeline_patch: Path | None = DEFAULT_ARC_PIPELINE_PATCH,
    arc_revision: str | None = ARC_EVO2_REV,
    overwrite: bool = False,
) -> list[Path]:
    """Copy Arc pipeline files and patch the import-time PhiX174 FASTA path."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    phix174_fasta = Path(phix174_fasta)
    pipeline_patch = Path(pipeline_patch) if pipeline_patch is not None else None
    if not source_dir.exists():
        raise FileNotFoundError(f"Arc pipeline source directory not found: {source_dir}")
    if not phix174_fasta.exists():
        raise FileNotFoundError(f"PhiX174 FASTA not found: {phix174_fasta}")
    if pipeline_patch is not None and not pipeline_patch.exists():
        raise FileNotFoundError(f"Arc pipeline patch not found: {pipeline_patch}")
    if pipeline_patch is not None and arc_revision:
        _assert_arc_source_revision(source_dir, arc_revision)
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
    if pipeline_patch is not None:
        _apply_arc_pipeline_patch(output_dir, pipeline_patch)
    else:
        _apply_legacy_string_patches(filtering_pipeline_path)
    return written_paths


def main() -> None:
    """CLI entry point for preparing Arc's local pipeline workdir."""
    parser = argparse.ArgumentParser(description="Prepare a patched local copy of Arc's phage filtering pipeline")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_ARC_PIPELINE_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ARC_PIPELINE_WORKDIR)
    parser.add_argument("--phix174-fasta", type=Path, default=DEFAULT_PHIX174_FASTA)
    parser.add_argument("--patch", type=Path, default=DEFAULT_ARC_PIPELINE_PATCH)
    parser.add_argument("--arc-revision", type=str, default=ARC_EVO2_REV)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in prepare_arc_pipeline_workdir(
        args.source_dir,
        args.output_dir,
        phix174_fasta=args.phix174_fasta,
        pipeline_patch=args.patch,
        arc_revision=args.arc_revision,
        overwrite=args.overwrite,
    ):
        print(path)


if __name__ == "__main__":
    main()
