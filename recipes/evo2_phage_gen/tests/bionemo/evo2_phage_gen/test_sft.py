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

"""Tests for ``bionemo.evo2_phage_gen.sft``."""

from __future__ import annotations

from pathlib import Path

import yaml

from bionemo.evo2_phage_gen import sft


RECIPE_ROOT = Path(__file__).parents[3]
PREPROCESS_CONFIG = RECIPE_ROOT / "configs" / "sft_microviridae_preprocess.yaml"
DATASET_CONFIG = RECIPE_ROOT / "configs" / "sft_microviridae_dataset.yaml"


def test_preprocess_config_points_at_processed_zenodo_fasta_and_512_tokenizer() -> None:
    """The committed preprocessing config should be portable from the recipe root."""
    config = yaml.safe_load(PREPROCESS_CONFIG.read_text())

    assert config[0]["datapaths"] == ["data/external/zenodo/microviridae_sft_training_data_processed.fna"]
    assert config[0]["hf_tokenizer_model_path"] == "tokenizers/nucleotide_fast_tokenizer_512"
    assert config[0]["force_uppercase"] is False


def test_dataset_config_uses_expected_preprocessed_prefixes() -> None:
    """Dataset prefixes should match preprocess_evo2's tokenizer-suffixed output names."""
    config = yaml.safe_load(DATASET_CONFIG.read_text())

    assert [entry["dataset_split"] for entry in config] == ["train", "validation", "test"]
    assert config[0]["dataset_prefix"].endswith("microviridae_sft_processed_nucleotide_fast_tokenizer_512_train")
    assert config[1]["dataset_prefix"].endswith("microviridae_sft_processed_nucleotide_fast_tokenizer_512_val")
    assert config[2]["dataset_prefix"].endswith("microviridae_sft_processed_nucleotide_fast_tokenizer_512_test")


def test_prepare_sft_data_downloads_processed_file_by_default(monkeypatch) -> None:
    """The SFT helper should download only the processed training FASTA unless raw is requested."""
    calls = []

    def fake_download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
        calls.append((url, output_path, overwrite))
        return output_path

    monkeypatch.setattr(sft, "_download", fake_download)

    paths = sft.prepare_sft_data(overwrite=True)

    assert paths == [sft.DEFAULT_SFT_PROCESSED]
    assert calls == [(sft.DEFAULT_SFT_PROCESSED_URL, sft.DEFAULT_SFT_PROCESSED, True)]


def test_prepare_sft_data_can_include_raw_fasta(monkeypatch) -> None:
    """The raw FASTA is optional because training uses the processed soft-prompt file."""
    calls = []

    def fake_download(url: str, output_path: Path, *, overwrite: bool = False) -> Path:
        calls.append((url, output_path, overwrite))
        return output_path

    monkeypatch.setattr(sft, "_download", fake_download)

    paths = sft.prepare_sft_data(include_raw=True)

    assert paths == [sft.DEFAULT_SFT_PROCESSED, sft.DEFAULT_SFT_RAW]
    assert calls == [
        (sft.DEFAULT_SFT_PROCESSED_URL, sft.DEFAULT_SFT_PROCESSED, False),
        (sft.DEFAULT_SFT_RAW_URL, sft.DEFAULT_SFT_RAW, False),
    ]
