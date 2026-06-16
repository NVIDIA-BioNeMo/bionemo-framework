#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Script to re-copy files from a given source filepath to destination filepaths, used as a pre-commit hook to ensure
# that copied files between recipe folders stay up-to-date.
#
# Destination files that support comments (e.g. .py) get a banner inserted after the license block indicating they are
# copies and linking back to the source file.

import argparse
import functools
import logging
import operator
import shutil
from pathlib import Path


logger = logging.getLogger(__name__)

BANNER_START_MARKER = "--- BEGIN COPIED FILE NOTICE ---"
BANNER_END_MARKER = "--- END COPIED FILE NOTICE ---"

# File extensions that support single-line comments, mapped to their comment prefix.
COMMENT_PREFIXES: dict[str, str] = {
    ".py": "#",
}


def get_comment_prefix(filepath: Path) -> str | None:
    """Return the single-line comment prefix for a file, or None if unsupported."""
    return COMMENT_PREFIXES.get(filepath.suffix)


def make_banner_lines(source_path: str, comment_prefix: str) -> list[str]:
    """Build the banner lines (without surrounding blank lines)."""
    return [
        f"{comment_prefix} {BANNER_START_MARKER}",
        f"{comment_prefix} This file is copied from: {source_path}",
        f"{comment_prefix} Do not modify this file directly. Instead, modify the source and run:",
        f"{comment_prefix}     python ci/scripts/check_copied_files.py --fix",
        f"{comment_prefix} {BANNER_END_MARKER}",
    ]


def _find_license_block_end(lines: list[str]) -> int:
    """Return the index of the first non-comment line after the leading comment block.

    Skips an optional shebang (``#!``) and the blank line that follows it.
    """
    start = 0
    if lines and lines[0].startswith("#!"):
        start = 2  # shebang + blank line
    i = start
    while i < len(lines) and lines[i].startswith("#"):
        i += 1
    return i


def add_banner_to_content(content: str, source_path: str, comment_prefix: str) -> str:
    """Insert a copied-file banner after the license block.

    A single blank line is added before the banner.  Any existing blank lines between the license
    block and the code are preserved after the banner so that
    ``strip_banner_from_content(add_banner_to_content(c, ...)) == c``.
    """
    lines = content.splitlines()
    license_end = _find_license_block_end(lines)

    banner = make_banner_lines(source_path, comment_prefix)
    # Insert: one blank line + banner, then keep whatever was already there (blank lines, code, …).
    new_lines = lines[:license_end] + [""] + banner + lines[license_end:]
    result = "\n".join(new_lines)
    if content.endswith("\n"):
        result += "\n"
    return result


def strip_banner_from_content(content: str) -> str:
    """Remove a copied-file banner from *content*.

    Only the blank line *before* the banner is consumed (the one ``add_banner_to_content``
    inserted).  Blank lines after the banner are left intact so that the original content is
    faithfully restored.
    """
    lines = content.splitlines()
    start = end = None
    for i, line in enumerate(lines):
        if BANNER_START_MARKER in line:
            start = i
        if BANNER_END_MARKER in line:
            end = i
            break
    if start is None or end is None:
        return content

    # Consume the single blank line that add_banner_to_content inserted before the banner.
    if start > 0 and lines[start - 1].strip() == "":
        start -= 1

    result = "\n".join(lines[:start] + lines[end + 1 :])
    if content.endswith("\n"):
        result += "\n"
    return result


def _add_banner_to_file(filepath: Path, source_path: str) -> None:
    """Add a banner to *filepath* if it supports comments."""
    prefix = get_comment_prefix(filepath)
    if prefix is None:
        return
    content = filepath.read_text()
    filepath.write_text(add_banner_to_content(content, source_path, prefix))


def _compare_file_contents(source_file: Path, dest_file: Path, source_display: str) -> None:
    """Compare *source_file* and *dest_file*, raising on mismatch.

    For files that support banners, the banner is stripped from the destination before comparing.
    """
    prefix = get_comment_prefix(dest_file)
    if prefix is not None:
        source_content = source_file.read_text()
        dest_content = dest_file.read_text()
        dest_stripped = strip_banner_from_content(dest_content)
        if source_content != dest_stripped:
            raise ValueError(
                f"Files {source_file} and {dest_file} do not match (ignoring banner). Run "
                f"{Path(__file__).relative_to(Path.cwd())} --fix to fix."
            )
    else:
        with open(source_file, "rb") as f1, open(dest_file, "rb") as f2:
            if f1.read() != f2.read():
                raise ValueError(
                    f"Files {source_file} and {dest_file} do not match. Run "
                    f"{Path(__file__).relative_to(Path.cwd())} --fix to fix."
                )


SOURCE_TO_DESTINATION_MAP: dict[str, list[str]] = {
    "bionemo-recipes/models/esm2/modeling_esm_te.py": [
        "bionemo-recipes/recipes/esm2_native_te/modeling_esm_te.py",
        "bionemo-recipes/recipes/esm2_peft_te/example_8m_checkpoint/esm_nv.py",
        "bionemo-recipes/recipes/esm2_accelerate_te/example_8m_checkpoint/esm_nv.py",
        "bionemo-recipes/recipes/vllm_inference/esm2/modeling_esm_te.py",
    ],
    "bionemo-recipes/models/esm2/collator.py": [
        "bionemo-recipes/models/llama3/collator.py",
        "bionemo-recipes/models/mixtral/collator.py",
        "bionemo-recipes/models/qwen/collator.py",
        "bionemo-recipes/recipes/esm2_native_te/collator.py",
        "bionemo-recipes/recipes/llama3_native_te/collator.py",
        "bionemo-recipes/recipes/opengenome2_llama_native_te/collator.py",
        "bionemo-recipes/recipes/mixtral_native_te/collator.py",
        "bionemo-recipes/recipes/opengenome2_mixtral_native_te/collator.py",
        "bionemo-recipes/recipes/esm2_peft_te/collator.py",
    ],
    "bionemo-recipes/models/esm2/state.py": [
        "bionemo-recipes/models/amplify/src/amplify/state.py",
        "bionemo-recipes/models/llama3/state.py",
        "bionemo-recipes/models/mixtral/state.py",
        "bionemo-recipes/models/qwen/state.py",
        "bionemo-recipes/recipes/vllm_inference/esm2/state.py",
    ],
    "bionemo-recipes/models/llama3/modeling_llama_te.py": [
        "bionemo-recipes/recipes/llama3_native_te/modeling_llama_te.py",
        "bionemo-recipes/recipes/opengenome2_llama_native_te/modeling_llama_te.py",
    ],
    "bionemo-recipes/models/llama3/nucleotide_fast_tokenizer": [
        "bionemo-recipes/recipes/llama3_native_te/tokenizers/nucleotide_fast_tokenizer",
    ],
    "bionemo-recipes/models/esm2/convert.py": [
        "bionemo-recipes/recipes/vllm_inference/esm2/convert.py",
    ],
    "bionemo-recipes/models/esm2/export.py": [
        "bionemo-recipes/recipes/vllm_inference/esm2/export.py",
    ],
    "bionemo-recipes/models/esm2/esm_fast_tokenizer": [
        "bionemo-recipes/recipes/vllm_inference/esm2/esm_fast_tokenizer",
    ],
    "bionemo-recipes/models/esm2/model_readme.template": [
        "bionemo-recipes/recipes/vllm_inference/esm2/model_readme.template",
    ],
    "bionemo-recipes/models/esm2/LICENSE": [
        "bionemo-recipes/recipes/vllm_inference/esm2/LICENSE",
    ],
    "bionemo-recipes/models/mixtral/modeling_mixtral_te.py": [
        "bionemo-recipes/recipes/mixtral_native_te/modeling_mixtral_te.py",
        "bionemo-recipes/recipes/opengenome2_mixtral_native_te/modeling_mixtral_te.py",
    ],
    # CodonFM model -> recipe sync
    "bionemo-recipes/models/codonfm/modeling_codonfm_te.py": [
        "bionemo-recipes/recipes/codonfm_native_te/modeling_codonfm_te.py",
    ],
    # Common test library - synced between models
    "bionemo-recipes/models/esm2/tests/common": [
        "bionemo-recipes/models/llama3/tests/common",
        "bionemo-recipes/models/mixtral/tests/common",
        "bionemo-recipes/models/qwen/tests/common",
        "bionemo-recipes/models/codonfm/tests/common",
    ],
}


def main():
    """Copy files from the source to the destinations."""
    parser = argparse.ArgumentParser(description="Ensure copied files are synchronized across recipe folders")
    parser.add_argument("files", nargs="*", help="Files to process", default=[])
    parser.add_argument("--fix", action="store_true", help="Copy the files from source to destinations")

    args = parser.parse_args()

    # Check if the script needs to run.
    all_files = set(SOURCE_TO_DESTINATION_MAP.keys()) | set(
        functools.reduce(operator.iadd, SOURCE_TO_DESTINATION_MAP.values(), [])
    )
    relevant_files = [f for f in args.files if f in all_files]
    # If pre-commit passed a list of files and none are relevant, skip.
    if args.files and not relevant_files:
        return

    for source, destinations in SOURCE_TO_DESTINATION_MAP.items():
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(
                f"Source file {source} does not exist -- if this file was removed, please update the "
                f"source-to-destination map in {Path(__file__).relative_to(Path.cwd())}"
            )

        for destination in destinations:
            destination_path = Path(destination)
            if not destination_path.exists():
                raise ValueError(
                    f"Destination file {destination} does not exist -- if this file was removed, please update the "
                    f"source-to-destination map in {Path(__file__).relative_to(Path.cwd())}"
                )

            if args.fix:
                if source_path.is_dir():
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                    for file in source_path.glob("*"):
                        if file.is_dir():
                            continue
                        source_rel = str(Path(source) / file.name)
                        _add_banner_to_file(destination_path / file.name, source_rel)
                else:
                    shutil.copy(source, destination)
                    _add_banner_to_file(destination_path, source)
                logger.info(f"Copied {source} to {destination}")

            else:
                if source_path.is_dir():
                    for file in source_path.glob("*"):
                        # Skip directories when checking - they're copied recursively with copytree
                        if file.is_dir():
                            continue
                        _compare_file_contents(file, destination_path / file.name, source)
                else:
                    _compare_file_contents(source_path, destination_path, source)


if __name__ == "__main__":
    main()
