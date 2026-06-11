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


from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Mol, rdMolDescriptors
from rdkit.Chem.rdchem import ChiralType, HybridizationType
from rdkit.Chem.Scaffolds import MurckoScaffold

from bionemo.geometric.base_featurizer import (
    BaseAtomFeaturizer,
)


ALL_ATOM_FEATURIZERS = [
    "PeriodicTableFeaturizer",
    "ElectronicPropertyFeaturizer",
    "ScaffoldFeaturizer",
    "SmartsFeaturizer",
    "CrippenFeaturizer",
]


class AtomicNumberFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its atomic number."""

    def __init__(self, dim_atomic_num: Optional[int] = None) -> None:
        """Initializes AtomicNumberFeaturizer class."""
        DIM_ATOMIC_NUM = 118
        self.dim_atomic_num = dim_atomic_num if dim_atomic_num else DIM_ATOMIC_NUM

    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.dim_atomic_num

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor of integers representing atomic numbers of atoms.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([mol.GetAtomWithIdx(a).GetAtomicNum() for a in _atom_indices], dtype=torch.int)


class DegreeFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its degree (excluding hydrogens) of connectivity."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 6

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor of integers representing degree of connectivity of atoms.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([mol.GetAtomWithIdx(a).GetDegree() for a in _atom_indices], dtype=torch.int)


class TotalDegreeFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its total degree (including hydrogens) of connectivity."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 6

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor of integers representing total connectivity (including hydrogens) of atoms.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([mol.GetAtomWithIdx(a).GetTotalDegree() for a in _atom_indices], dtype=torch.int)


class ChiralTypeFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its chirality type."""

    def __init__(self) -> None:
        """Initializes ChiralTypeFeaturizer class."""
        self.dim_chiral_types = len(ChiralType.values)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.dim_chiral_types

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor representing chirality type of atoms as integers.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([int(mol.GetAtomWithIdx(a).GetChiralTag()) for a in _atom_indices], dtype=torch.int)


class TotalNumHFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by total number of hydrogens."""

    def __init__(self) -> None:
        """Initializes TotalNumHFeaturizer class."""
        self.dim_total_num_hydrogen = 5  # 4 + 1 (no hydrogens)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.dim_total_num_hydrogen

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor of integers representing total number of hydrogens on atoms.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([mol.GetAtomWithIdx(a).GetTotalNumHs() for a in _atom_indices], dtype=torch.int)


class HybridizationFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its hybridization type."""

    def __init__(self) -> None:
        """Initializes HybridizationFeaturizer class."""
        self.dim_hybridization_types = len(HybridizationType.values)

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return self.dim_hybridization_types

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor representing hybridization type of atoms as integers.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([int(mol.GetAtomWithIdx(a).GetHybridization()) for a in _atom_indices], dtype=torch.int)


class AromaticityFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom based on its aromaticity."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes features of atoms of all of select atoms.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor of representing if atoms are aromatic as integers.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor([int(mol.GetAtomWithIdx(a).GetIsAromatic()) for a in _atom_indices], dtype=torch.int)


class PeriodicTableFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its position (period and group) in the periodic table."""

    def __init__(self) -> None:
        """Initializes PeriodicTableFeaturizer class."""
        self.pt = Chem.GetPeriodicTable()
        # The number of elements per period in the periodic table
        self.period_limits = [2, 10, 18, 36, 54, 86, 118]

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 25

    def get_period(self, atom: Chem.Atom) -> int:
        """Returns periodic table period of atom."""
        atomic_number = atom.GetAtomicNum()

        # Determine the period based on atomic number.
        for period, limit in enumerate(self.period_limits, start=1):
            if atomic_number <= limit:
                return period
        return None

    def get_group(self, atom: Chem.Atom) -> int:
        """Returns periodic table group of atom."""
        group = self.pt.GetNOuterElecs(atom.GetAtomicNum())
        return group

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes periodic table position of atoms of all or select atoms specific in `atom_indices`.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor of representing positions of atoms in periodic table. First index represents period and second index represents group.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return torch.tensor(
            [(self.get_period(mol.GetAtomWithIdx(a)), self.get_group(mol.GetAtomWithIdx(a))) for a in _atom_indices],
            dtype=torch.int,
        )


class AtomicRadiusFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its bond, covalent, and vdW radii."""

    def __init__(self) -> None:
        """Initializes AtomicRadiusFeaturizer class."""
        self.pt = Chem.GetPeriodicTable()
        self._min_val = torch.Tensor(
            [
                0.0,  # Bond radius
                0.28,  # Covalent radius
                1.2,  # van der Waals radius
            ]
        )

        self._max_val = torch.Tensor(
            [
                2.4,  # Bond radius
                2.6,  # Covalent radius
                3.0,  # van der Waals radius
            ]
        )

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 3

    @property
    def min_val(self) -> torch.tensor:
        """Returns minimum values for features: bond, covalent, and vdW radius."""
        return self._min_val

    @property
    def max_val(self) -> torch.tensor:
        """Returns maximum values for features: bond, covalent, and vdW radius."""
        return self._max_val

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.Tensor:
        """Computes bond radius, covalent radius, and van der Waals radius without normalization.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.Tensor of different atomic radii. Each atom is featurizer by bond radius, covalent radius, and van der Waals radius.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())

        feats = []
        for aidx in _atom_indices:
            atomic_num = mol.GetAtomWithIdx(aidx).GetAtomicNum()
            feats.append([self.pt.GetRb0(atomic_num), self.pt.GetRcovalent(atomic_num), self.pt.GetRvdw(atomic_num)])

        return torch.Tensor(feats)


class ElectronicPropertyFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by its electronic properties.

    This class computes electronic properties like electronegativity, ionization energy, and electron affinity.
    """

    def __init__(self, data_file=None) -> None:
        """Initializes PeriodicTableFeaturizer class.

        Args:
            data_file: Path to the data file.
        """
        if data_file is None:
            # Use default
            root_path = Path(__file__).resolve().parent
            data_file = root_path / "data" / "electronic_data.csv"
        self.data_df = pd.read_csv(data_file).set_index("AtomicNumber")

        self.pauling_en_dict = self.data_df["Electronegativity"].to_dict()
        self.ie_dict = self.data_df["IonizationEnergy"].to_dict()
        self.ea_dict = self.data_df["ElectronAffinity"].to_dict()

        self._min_val = torch.Tensor(
            [
                self.data_df["Electronegativity"].min(),
                self.data_df["IonizationEnergy"].min(),
                self.data_df["ElectronAffinity"].min(),
            ]
        )

        self._max_val = torch.Tensor(
            [
                self.data_df["Electronegativity"].max(),
                self.data_df["IonizationEnergy"].max(),
                self.data_df["ElectronAffinity"].max(),
            ]
        )

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 3

    @property
    def min_val(self) -> torch.Tensor:
        """Returns minimum values for features: electronegativity, ionization energy, electron affinity."""
        return self._min_val

    @property
    def max_val(self) -> torch.Tensor:
        """Returns maximum values for features: electronegativity, ionization energy, electron affinity."""
        return self._max_val

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.Tensor:
        """Returns electronic features of the atom.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.Tensor consisting of Pauling scale electronegativity, ionization energy, and electron affinity for each atom.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())

        feats = []
        for aidx in _atom_indices:
            atomic_num = mol.GetAtomWithIdx(aidx).GetAtomicNum()
            feats.append([self.pauling_en_dict[atomic_num], self.ie_dict[atomic_num], self.ea_dict[atomic_num]])
        return torch.Tensor(feats)


class ScaffoldFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom based on whether it is present in Bemis-Murcko scaffold."""

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 1

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Returns position of the atoms with respect to Bemis-Murcko scaffold.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.tensor indicating if atoms are present in the Bemis-Murcko scaffold of the molecule.
        """
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())

        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_atom_idx = set(mol.GetSubstructMatch(scaffold))

        feats = [int(aidx in scaffold_atom_idx) for aidx in _atom_indices]
        return torch.tensor(feats, dtype=torch.int)


class SmartsFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by hydrogen donor/acceptor and acidity/basicity."""

    def __init__(self):
        """Initializes SmartsFeaturizer class."""
        self.hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
        self.hydrogen_acceptor = Chem.MolFromSmarts(
            "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),"
            "n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]"
        )
        self.acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
        self.basic = Chem.MolFromSmarts(
            "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);"
            "!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]"
        )

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 4

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.tensor:
        """Computes matches by prefixed SMARTS patterns.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            An torch.tensor indicating if atoms are hydrogen bond donors, hydrogen bond acceptors, acidic, or basic.
        """
        hydrogen_donor_match = sum(mol.GetSubstructMatches(self.hydrogen_donor), ())
        hydrogen_acceptor_match = sum(mol.GetSubstructMatches(self.hydrogen_acceptor), ())
        acidic_match = sum(mol.GetSubstructMatches(self.acidic), ())
        basic_match = sum(mol.GetSubstructMatches(self.basic), ())

        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        feats = [
            [
                aidx in hydrogen_donor_match,
                aidx in hydrogen_acceptor_match,
                aidx in acidic_match,
                aidx in basic_match,
            ]
            for aidx in _atom_indices
        ]

        return torch.tensor(feats, dtype=torch.int)


class CrippenFeaturizer(BaseAtomFeaturizer):
    """Class for featurizing atom by Crippen logP and molar refractivity."""

    def __init__(self):
        """Initializes CrippenFeaturizer class."""
        self._min_val = torch.Tensor(
            [
                -2.996,  # logP
                0.0,  # MR
            ]
        )

        self._max_val = torch.Tensor(
            [
                0.8857,  # logP
                6.0,  # MR
            ]
        )

    @property
    def n_dim(self) -> int:
        """Returns dimensionality of the computed features."""
        return 2

    @property
    def min_val(self) -> torch.tensor:
        """Returns minimum values for features: logP and molar refractivity."""
        return self._min_val

    @property
    def max_val(self) -> torch.tensor:
        """Returns maximum values for features: logP and molar refractivity."""
        return self._max_val

    def get_atom_features(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> torch.Tensor:
        """Compute atomic contributions to Crippen logP and molar refractivity.

        Args:
            mol: An RDkit Chem.Mol object
            atom_indices: Indices of atoms for feature computation. By default, features for all atoms is computed.

        Returns:
            A torch.Tensor featurizing atoms by its atomic contribution to logP and molar refractivity.
        """
        logp_mr_list = torch.Tensor(rdMolDescriptors._CalcCrippenContribs(mol))
        logp_mr_list = torch.clamp(logp_mr_list, min=self.min_val, max=self.max_val)
        _atom_indices = atom_indices if atom_indices else range(mol.GetNumAtoms())
        return logp_mr_list[_atom_indices, :]
