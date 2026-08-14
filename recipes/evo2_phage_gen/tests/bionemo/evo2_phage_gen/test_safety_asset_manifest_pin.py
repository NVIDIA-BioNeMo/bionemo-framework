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

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from bionemo.evo2_phage_gen import safety_asset_manifest_pin


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pin_safety_asset_manifest_copies_recipe_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = tmp_path / "source-recipe.yaml"
    recipe.write_text("schema_version: 3\nvalue: pinned\n", encoding="utf-8")
    source_manifest = tmp_path / "source-manifest.yaml"
    source_payload = {
        "schema_version": 3,
        "recipe": {"path": str(recipe), "sha256": _sha256(recipe)},
        "assets": {"identity": "unchanged"},
    }
    source_manifest.write_text(yaml.safe_dump(source_payload, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "pinned"
    validated_paths: list[Path] = []

    def load(path: Path):
        selected = Path(path).absolute()
        validated_paths.append(selected)
        payload = yaml.safe_load(selected.read_text(encoding="utf-8"))
        selected_recipe = Path(payload["recipe"]["path"])
        assert selected_recipe.is_file()
        assert payload["recipe"]["sha256"] == _sha256(selected_recipe)
        return SimpleNamespace(
            manifest=payload,
            manifest_path=selected,
            manifest_sha256=_sha256(selected),
            recipe_path=selected_recipe,
            recipe_sha256=_sha256(selected_recipe),
        )

    monkeypatch.setattr(safety_asset_manifest_pin, "load_safety_asset_manifest", load)

    receipt = safety_asset_manifest_pin.pin_safety_asset_manifest(source_manifest, output_dir)

    pinned_recipe = output_dir / "phage_safety_assets.yaml"
    pinned_manifest = output_dir / "asset_manifest.yaml"
    assert pinned_recipe.read_bytes() == recipe.read_bytes()
    payload = yaml.safe_load(pinned_manifest.read_text(encoding="utf-8"))
    assert payload["recipe"] == {
        "path": str(pinned_recipe.absolute()),
        "sha256": _sha256(pinned_recipe),
    }
    assert payload["assets"] == source_payload["assets"]
    assert validated_paths == [source_manifest.absolute(), pinned_manifest.absolute()]
    assert receipt["pinned_manifest"]["sha256"] == _sha256(pinned_manifest)
    assert json.loads((output_dir / "PINNING.json").read_text(encoding="utf-8")) == receipt


def test_pin_safety_asset_manifest_refuses_existing_destination(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    with pytest.raises(FileExistsError, match="destination already exists"):
        safety_asset_manifest_pin.pin_safety_asset_manifest(tmp_path / "manifest.yaml", output_dir)
