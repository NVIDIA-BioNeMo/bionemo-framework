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

"""Evo2 + SAE inference engine — reused by the live server, the batch CLI, and the viz backend."""

from typing import TYPE_CHECKING


if TYPE_CHECKING:  # for type checkers / ruff F822 — runtime access goes through __getattr__ below
    from .core import DEFAULT_ORGANISM_TAGS, Evo2SAE, clean_dna


__all__ = ["DEFAULT_ORGANISM_TAGS", "Evo2SAE", "clean_dna"]


def __getattr__(name: str):
    """Lazily pull the heavy engine symbols from ``.core`` (importing ``.core`` loads torch).

    Keeps ``import evo2_sae`` (and lightweight submodules like ``evo2_sae.fasta``) cheap so
    stdlib-only callers don't drag in torch, while ``from evo2_sae import Evo2SAE`` still works.
    """
    if name in __all__:
        from . import core

        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
