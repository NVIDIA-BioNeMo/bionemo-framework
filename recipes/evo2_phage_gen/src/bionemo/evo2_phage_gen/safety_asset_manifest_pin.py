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

"""Pin a validated safety-asset manifest and its recipe into a run-owned directory."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from bionemo.evo2_phage_gen.sequence_safety_cli import load_safety_asset_manifest


PINNED_MANIFEST_NAME = "asset_manifest.yaml"
PINNED_RECIPE_NAME = "phage_safety_assets.yaml"
PINNING_RECEIPT_NAME = "PINNING.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_yaml_atomic(path: Path, value: Mapping[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            yaml.safe_dump(dict(value), output, sort_keys=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def pin_safety_asset_manifest(manifest_path: Path, output_dir: Path) -> dict[str, object]:
    """Copy the validated recipe, rebind its manifest, and validate the pinned result."""
    source_path = Path(manifest_path).absolute()
    destination = Path(output_dir).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"destination already exists: {destination}")

    source = load_safety_asset_manifest(source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    pinned_recipe = destination / PINNED_RECIPE_NAME
    pinned_manifest = destination / PINNED_MANIFEST_NAME
    receipt_path = destination / PINNING_RECEIPT_NAME
    try:
        recipe_bytes = source.recipe_path.read_bytes()
        with pinned_recipe.open("xb") as output:
            output.write(recipe_bytes)
            output.flush()
            os.fsync(output.fileno())
        pinned_recipe_sha256 = _sha256(pinned_recipe)
        if pinned_recipe_sha256 != source.recipe_sha256:
            raise ValueError("copied safety-asset recipe digest does not match the validated source")

        pinned_payload = copy.deepcopy(dict(source.manifest))
        pinned_payload["recipe"] = {
            "path": str(pinned_recipe),
            "sha256": pinned_recipe_sha256,
        }
        _write_yaml_atomic(pinned_manifest, pinned_payload)
        validated = load_safety_asset_manifest(pinned_manifest)
        if validated.recipe_path != pinned_recipe or validated.recipe_sha256 != pinned_recipe_sha256:
            raise ValueError("pinned safety-asset manifest did not retain its recipe binding")

        receipt: dict[str, object] = {
            "schema_version": 1,
            "pin_type": "sequence_safety_asset_manifest",
            "source_manifest": {
                "path": str(source.manifest_path),
                "sha256": source.manifest_sha256,
            },
            "source_recipe": {
                "path": str(source.recipe_path),
                "sha256": source.recipe_sha256,
            },
            "pinned_manifest": {
                "path": str(pinned_manifest),
                "sha256": validated.manifest_sha256,
            },
            "pinned_recipe": {
                "path": str(pinned_recipe),
                "sha256": pinned_recipe_sha256,
            },
        }
        _write_json_atomic(receipt_path, receipt)
        return receipt
    except BaseException:
        shutil.rmtree(destination)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Build the safety-asset pinning command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the safety-asset pinning command."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        receipt = pin_safety_asset_manifest(args.manifest, args.output_dir)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return 0
    except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
        parser._print_message(f"{parser.prog}: error: {error}\n", sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
