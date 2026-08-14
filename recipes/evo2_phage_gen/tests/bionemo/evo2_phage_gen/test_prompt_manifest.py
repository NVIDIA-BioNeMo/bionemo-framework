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

"""Tests for secure prompt reproducibility manifests."""

import json

from bionemo.evo2_phage_gen.prompt_manifest import build_prompt_manifest, sha256_text


def test_build_prompt_manifest_hashes_prompts_without_storing_sequences(tmp_path):
    prompt_jsonl = tmp_path / "prompts.jsonl"
    prompt_jsonl.write_text(
        json.dumps({"id": "p1", "prompt": "+~ACGT", "target_id": "phiX174", "prefix_nt_length": 4}) + "\n"
    )
    asset = tmp_path / "arc_revision.txt"
    asset.write_text("arc-revision")
    missing_asset = tmp_path / "missing-database.dmnd"

    manifest = build_prompt_manifest(
        [prompt_jsonl],
        deterministic_generation_procedure="select PhiX174 prefixes by fixed length list",
        generation_seed=11,
        generation_call_index=3,
        training_seed=101,
        validation_seed=202,
        arc_revision="abc123",
        external_asset_paths=[asset, missing_asset],
        wandb_run_id="run-1",
    )

    assert manifest["schema_version"] == 1
    assert manifest["generation_seed"] == 11
    assert manifest["generation_call_index"] == 3
    assert manifest["training_seed"] == 101
    assert manifest["validation_seed"] == 202
    assert manifest["arc_revision"] == "abc123"
    assert manifest["wandb_run_id"] == "run-1"
    assert manifest["external_assets"][0]["exists"] is True
    assert manifest["external_assets"][1]["exists"] is False
    assert manifest["external_assets"][1]["sha256"] is None
    prompt_file = manifest["prompt_files"][0]
    assert prompt_file["num_records"] == 1
    record = prompt_file["records"][0]
    assert record["id"] == "p1"
    assert record["target_id"] == "phiX174"
    assert record["prompt_nt_length"] == 4
    assert record["prompt_sha256"] == sha256_text("+~ACGT")
    assert "+~ACGT" not in json.dumps(manifest)
