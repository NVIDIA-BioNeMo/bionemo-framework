# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from typing import Iterable, Optional

from rdkit.Chem import Mol

from bionemo.geometric.base_featurizer import BaseBondFeaturizer


ALL_BOND_FEATURIZERS = ["RingFeaturizer"]


class RingFeaturizer(BaseBondFeaturizer):
    """Class for featurizing bond its ring membership."""

    def __init__(self, n_ring_sizes=7) -> None:
        """Initializes RingFeaturizer class."""
        self.n_ring_sizes = n_ring_sizes  # ring size 3 - 8 and UNK

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.n_ring_sizes

    def get_bond_features(self, mol: Mol, bond_indices: Optional[Iterable]) -> list[tuple[int]]:
        """Computes ring sizes a bonds of the molecule are present in.

        Args:
            mol: An RDkit Chem.Mol object
            bond_indices: Indices of bonds for feature computation. By default, features for all bonds is computed.

        Returns:
            An list of tuples indicating the size of ring(s) the bonds are present in.
        """
        _bond_indices = bond_indices if bond_indices else range(mol.GetNumBonds())

        ri = mol.GetRingInfo()
        return [ri.BondRingSizes(bidx) for bidx in _bond_indices]
