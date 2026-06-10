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


# import json
from typing import Dict, List


CODON_TO_AA = {
    "AAA": "K",
    "AAC": "N",
    "AAG": "K",
    "AAU": "N",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACU": "T",
    "AGA": "R",
    "AGC": "S",
    "AGG": "R",
    "AGU": "S",
    "AUA": "I",
    "AUC": "I",
    "AUG": "M",
    "AUU": "I",
    "CAA": "Q",
    "CAC": "H",
    "CAG": "Q",
    "CAU": "H",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCU": "P",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGU": "R",
    "CUA": "L",
    "CUC": "L",
    "CUG": "L",
    "CUU": "L",
    "GAA": "E",
    "GAC": "D",
    "GAG": "E",
    "GAU": "D",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCU": "A",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGU": "G",
    "GUA": "V",
    "GUC": "V",
    "GUG": "V",
    "GUU": "V",
    "UAA": "*",
    "UAC": "Y",
    "UAG": "*",
    "UAU": "Y",
    "UCA": "S",
    "UCC": "S",
    "UCG": "S",
    "UCU": "S",
    "UGA": "*",
    "UGC": "C",
    "UGG": "W",
    "UGU": "C",
    "UUA": "L",
    "UUC": "F",
    "UUG": "L",
    "UUU": "F",
}


class _CODON_TABLE:  # noqa: N801
    RNA: Dict[str, str] = CODON_TO_AA
    DNA: Dict[str, str] = {k.replace("U", "T"): v for k, v in CODON_TO_AA.items()}  # noqa: RUF012

    def __getitem__(self, type_: str) -> Dict[str, str]:
        assert type_.lower() in ["rna", "dna"], f"Invalid type: {type_}"
        return self.RNA if type_.lower() == "rna" else self.DNA


class _AA_TABLE:  # noqa: N801
    RNA: Dict[str, List[str]] = {  # noqa: RUF012
        aa: [k for k, v in CODON_TO_AA.items() if v == aa] for aa in set(CODON_TO_AA.values())
    }
    DNA: Dict[str, List[str]] = {  # noqa: RUF012
        aa: [k.replace("U", "T") for k, v in CODON_TO_AA.items() if v == aa] for aa in set(CODON_TO_AA.values())
    }

    def __getitem__(self, type_: str) -> Dict[str, List[str]]:
        assert type_.lower() in ["rna", "dna"], f"Invalid type: {type_}"
        return self.RNA if type_.lower() == "rna" else self.DNA


CODON_TABLE = _CODON_TABLE()
AA_TABLE = _AA_TABLE()
