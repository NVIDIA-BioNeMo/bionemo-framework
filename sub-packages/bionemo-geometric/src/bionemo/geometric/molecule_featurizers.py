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


import torch
from rdkit.Chem import Descriptors, Mol

from bionemo.geometric.base_featurizer import (
    BaseMoleculeFeaturizer,
)


class RDkit2DDescriptorFeaturizer(BaseMoleculeFeaturizer):
    """Class for featurizing molecule by computed RDkit descriptors.

    Typical usage example:
    rdf = RDkit2DDescriptorFeaturizer()
    rdf(Chem.MolFromSmiles("CCO"))
    """

    def __init__(self) -> None:
        """Initializes RDkit2DDescriptorFeaturizer class."""
        self.n_rdkit_descriptors = len(Descriptors.descList)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.n_rdkit_descriptors

    def get_molecule_features(self, mol: Mol) -> torch.Tensor:
        """Returns features of the molecule.

        Args:
            mol: An RDkit Chem.Mol object

        Returns:
        A torch.tensor representing RDkit-computed 2D descriptors of the molecule.
        """
        return torch.Tensor([f(mol) for desc_name, f in Descriptors.descList])
