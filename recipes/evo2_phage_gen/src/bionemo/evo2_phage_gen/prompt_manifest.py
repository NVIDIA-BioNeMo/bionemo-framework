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

"""Write secure reproducibility manifests for prompt JSONL files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DNA_ALPHABET = frozenset("ACGTacgt")
TARGET_METADATA_KEYS = (
    "id",
    "target_id",
    "target_name",
    "source_id",
    "prefix_id",
    "prefix_nt_length",
    "prompt_nt_length",
    "task_name",
)


def sha256_file(path: Path) -> str:
    """Hash a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Hash normalized prompt text without storing the text itself."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prompt_nucleotides(prompt: str) -> str:
    return "".join(char for char in prompt if char in DNA_ALPHABET)


def _prompt_from_record(record: dict[str, Any]) -> str:
    if "prompt" in record:
        return str(record["prompt"])
    messages = record.get("messages")
    if isinstance(messages, list):
        return "".join(
            str(message.get("content", ""))
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
    return ""


def prompt_file_manifest(path: Path) -> dict[str, Any]:
    """Build a secure manifest for one prompt JSONL."""
    records = []
    with Path(path).open() as handle:
        for line_index, raw_line in enumerate(handle):
            stripped = raw_line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            prompt = _prompt_from_record(record)
            prompt_metadata = {
                key: record[key]
                for key in TARGET_METADATA_KEYS
                if key in record and isinstance(record[key], str | int | float | bool)
            }
            prompt_metadata.setdefault("line_index", line_index)
            prompt_metadata["prompt_sha256"] = sha256_text(prompt)
            prompt_metadata["prompt_nt_length"] = len(_prompt_nucleotides(prompt))
            records.append(prompt_metadata)

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "num_records": len(records),
        "records": records,
    }


def build_prompt_manifest(
    prompt_paths: list[Path],
    *,
    deterministic_generation_procedure: str,
    generation_seed: int | None = None,
    generation_call_index: int | None = None,
    training_seed: int | None = None,
    validation_seed: int | None = None,
    arc_revision: str | None = None,
    external_asset_paths: list[Path] | None = None,
    wandb_run_id: str | None = None,
) -> dict[str, Any]:
    """Build a manifest that preserves prompt reproducibility without embedding prompt sequences."""
    assets = []
    for raw_asset_path in external_asset_paths or []:
        asset_path = Path(raw_asset_path)
        assets.append(
            {
                "path": str(asset_path),
                "sha256": sha256_file(asset_path) if asset_path.is_file() else None,
                "exists": asset_path.exists(),
            }
        )

    return {
        "schema_version": 1,
        "deterministic_generation_procedure": deterministic_generation_procedure,
        "generation_seed": generation_seed,
        "generation_call_index": generation_call_index,
        "training_seed": training_seed,
        "validation_seed": validation_seed,
        "arc_revision": arc_revision,
        "wandb_run_id": wandb_run_id,
        "external_assets": assets,
        "prompt_files": [prompt_file_manifest(Path(path)) for path in prompt_paths],
    }


def main() -> None:
    """CLI entry point for writing prompt manifests."""
    parser = argparse.ArgumentParser(description="Write a secure manifest for Evo2 phage prompt JSONLs")
    parser.add_argument("prompt_jsonl", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--procedure", required=True, help="Deterministic prompt generation procedure")
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--generation-call-index", type=int)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--arc-revision")
    parser.add_argument("--external-asset", action="append", default=[], type=Path)
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()

    manifest = build_prompt_manifest(
        args.prompt_jsonl,
        deterministic_generation_procedure=args.procedure,
        generation_seed=args.generation_seed,
        generation_call_index=args.generation_call_index,
        training_seed=args.training_seed,
        validation_seed=args.validation_seed,
        arc_revision=args.arc_revision,
        external_asset_paths=args.external_asset,
        wandb_run_id=args.wandb_run_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
