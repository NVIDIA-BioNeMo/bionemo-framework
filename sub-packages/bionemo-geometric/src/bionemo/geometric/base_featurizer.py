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


from abc import ABC, abstractmethod, abstractproperty
from typing import Iterable, Optional

import torch
from rdkit.Chem import Atom, Mol


class BaseAtomFeaturizer(ABC):
    """Abstract base featurizer class for all atom featurization classes."""

    @abstractproperty
    def n_dim(self) -> int:
        """Number of dimensions of computed feature."""
        ...

    @abstractmethod
    def get_atom_features(self):
        """Computes atom features."""
        ...

    def __call__(self, mol: Mol, atom_indices: Optional[Iterable] = None) -> list[int]:
        """Returns computed atom features."""
        return self.get_atom_features(mol, atom_indices)


class BaseBondFeaturizer(ABC):
    """Abstract base featurizer class for all bond featurization classes."""

    @abstractproperty
    def n_dim(self) -> int:
        """Number of dimensions of computed feature."""
        ...

    @abstractmethod
    def get_bond_features(self):
        """Computes bond features."""
        ...

    def __call__(self, mol: Mol, bond_indices: Optional[Iterable] = None) -> list[int]:
        """Returns computed bond features."""
        return self.get_bond_features(mol, bond_indices)


class BaseMoleculeFeaturizer(ABC):
    """Abstract base featurizer class for molecule featurization classes."""

    @abstractproperty
    def n_dim(self) -> int:
        """Number of dimensions of computed feature."""
        ...

    @abstractmethod
    def get_molecule_features(self, mol: Mol) -> torch.Tensor:
        """Computes molecule features."""
        ...

    def __call__(self, mol: Mol) -> torch.Tensor:
        """Returns computed molecule features."""
        return self.get_molecule_features(mol)


def one_hot_enc(val: int, num_class: int) -> list[bool]:
    """Performs one-hot encoding on an integer value.

    This function creates a one-hot encoded representation of the input value
    as a list of boolean values. The resulting list has a length equal to
    `num_class`, where only the element at index `val` is set to True.

    Args:
        val (int): An integer representing the value to be one-hot encoded.
            Must be in the range [0, num_class - 1].
        num_class (int): An integer representing the total number of classes or
            possible classes.

    Returns:
        One-hot encoding of `val`.
    """
    one_hot = [False] * num_class
    one_hot[val] = True
    return one_hot


def get_boolean_atomic_prop(atom: Atom, prop_list=None) -> list[bool]:
    """Retrieves boolean atomic properties for a given atom.

    This function fetches boolean properties of an atom. If a specific list of
    properties is provided, it retrieves those properties. Otherwise, it fetches
    all available boolean properties for the atom.

    Args:
        atom: The atom object to retrieve properties from.
        prop_list (list, optional): A list of specific property names to retrieve.
            If None, all available properties will be fetched. Defaults to None.

    Returns:
        list: A list of boolean values corresponding to the requested properties.
    """
    _prop_list = prop_list if prop_list else atom.GetPropNames()

    return [atom.GetBoolProp(prop) for prop in _prop_list]


def get_double_atomic_prop(atom, prop_list=None) -> list[float]:
    """Retrieves double atomic properties for a given atom.

    This function fetches double properties of an atom. If a specific list of
    properties is provided, it retrieves those properties. Otherwise, it fetches
    all available double properties for the atom.

    Args:
        atom: The atom object to retrieve properties from.
        prop_list (list, optional): A list of specific property names to retrieve.
            If None, all available properties will be fetched. Defaults to None.

    Returns:
        list: A list of float values corresponding to the requested properties.
    """
    if prop_list is not None:
        _prop_list = prop_list
    else:
        _prop_list = atom.GetPropNames()

    return [atom.GetDoubleProp(prop) for prop in _prop_list]
