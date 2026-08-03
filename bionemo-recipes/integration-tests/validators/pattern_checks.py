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

"""Pattern-based validation of generated code for TE conversion patterns."""

import re
from pathlib import Path


def _read(filepath: Path) -> str:
    """Read file contents."""
    assert filepath.exists(), f"File does not exist: {filepath}"
    return filepath.read_text()


def has_te_imports(filepath: Path) -> None:
    """Assert that the file imports TransformerEngine modules."""
    code = _read(filepath)
    assert "transformer_engine" in code, f"{filepath.name} does not import transformer_engine"
    # Should import at least TransformerLayer or the pytorch module
    assert any(
        pattern in code
        for pattern in [
            "transformer_engine.pytorch.TransformerLayer",
            "transformer_engine.pytorch",
            "from transformer_engine",
            "import transformer_engine",
        ]
    ), f"{filepath.name} does not import TE TransformerLayer or pytorch module"


def has_state_dict_mapping(filepath: Path) -> None:
    """Assert that the file contains a state dict mapping dictionary."""
    code = _read(filepath)
    # Should have a mapping dict with wildcard patterns
    assert re.search(r"mapping\s*=\s*\{", code) or re.search(r'["\'].*\*.*["\']\s*:', code), (
        f"{filepath.name} does not contain a state dict mapping with wildcards"
    )


def has_bidirectional_conversion(filepath: Path) -> None:
    """Assert that the file has both HF->TE and TE->HF conversion functions."""
    code = _read(filepath)
    has_hf_to_te = bool(re.search(r"def\s+convert_\w+_hf_to_te", code))
    has_te_to_hf = bool(re.search(r"def\s+convert_\w+_te_to_hf", code))
    assert has_hf_to_te, f"{filepath.name} missing convert_*_hf_to_te function"
    assert has_te_to_hf, f"{filepath.name} missing convert_*_te_to_hf function"


def has_layer_precision_config(filepath: Path) -> None:
    """Assert that a config or model file has layer_precision support."""
    code = _read(filepath)
    assert "layer_precision" in code, f"{filepath.name} does not reference layer_precision"


def has_fp8_autocast(filepath: Path) -> None:
    """Assert that a model file uses TE autocast for FP8."""
    code = _read(filepath)
    assert any(
        pattern in code
        for pattern in [
            "transformer_engine.pytorch.autocast",
            "te.autocast",
            "autocast(enabled=",
        ]
    ), f"{filepath.name} does not use TE autocast"


def has_vocab_padding(filepath: Path) -> None:
    """Assert that the model/config handles vocabulary padding for FP8."""
    code = _read(filepath)
    assert "padded_vocab_size" in code or "pad" in code.lower(), f"{filepath.name} does not handle vocabulary padding"
