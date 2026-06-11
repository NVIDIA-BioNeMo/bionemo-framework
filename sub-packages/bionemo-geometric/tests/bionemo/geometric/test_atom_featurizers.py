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


import pytest
import torch
from rdkit import Chem

from bionemo.geometric.atom_featurizers import (
    AromaticityFeaturizer,
    AtomicNumberFeaturizer,
    AtomicRadiusFeaturizer,
    ChiralTypeFeaturizer,
    CrippenFeaturizer,
    DegreeFeaturizer,
    ElectronicPropertyFeaturizer,
    HybridizationFeaturizer,
    PeriodicTableFeaturizer,
    ScaffoldFeaturizer,
    SmartsFeaturizer,
    TotalDegreeFeaturizer,
    TotalNumHFeaturizer,
)


@pytest.fixture(scope="module")
def test_mol():
    return Chem.MolFromSmiles("NC(=O)c1cn(-c2ccc(S(N)(=O)=O)cc2)nc1-c1ccc(Cl)cc1")  # CHEMBL3126825


@pytest.fixture(scope="module")
def acetic_acid():
    return Chem.MolFromSmiles("CC(=O)O")


@pytest.fixture(scope="module")
def methylamine():
    return Chem.MolFromSmiles("CN")


@pytest.fixture(scope="module")
def chiral_mol():
    return Chem.MolFromSmiles("Cn1cc(C(=O)N2CC[C@@](O)(c3ccccc3)[C@H]3CCCC[C@@H]32)ccc1=O")


def test_atomic_num_featurizer(test_mol):
    anf = AtomicNumberFeaturizer()
    anf_feats = anf(test_mol)

    # Indicates the atomic number of the atom
    anf_feats_ref = torch.tensor(
        [7, 6, 8, 6, 6, 7, 6, 6, 6, 6, 16, 7, 8, 8, 6, 6, 7, 6, 6, 6, 6, 6, 17, 6, 6], dtype=torch.int
    )
    assert torch.allclose(anf_feats, anf_feats_ref)


def test_degree_featurizer(test_mol):
    df = DegreeFeaturizer()
    df_feats = df(test_mol)

    # Indicates the total degree of connectivty (excluding hydrogens) of the atom
    df_feats_ref = torch.tensor(
        [1, 3, 1, 3, 2, 3, 3, 2, 2, 3, 4, 1, 1, 1, 2, 2, 2, 3, 3, 2, 2, 3, 1, 2, 2], dtype=torch.int
    )
    assert torch.allclose(df_feats, df_feats_ref)


def test_total_degree_featurizer(test_mol):
    tdf = TotalDegreeFeaturizer()

    tdf_feats = tdf(test_mol)

    # Indicates the total degree of connectivity (including hydrogens) of the atom
    tdf_feats_ref = torch.tensor(
        [3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 4, 3, 1, 1, 3, 3, 2, 3, 3, 3, 3, 3, 1, 3, 3], dtype=torch.int
    )
    assert torch.allclose(tdf_feats, tdf_feats_ref)


def test_chiral_type_featurizer(chiral_mol):
    cf = ChiralTypeFeaturizer()

    cf_feats = cf(chiral_mol)

    # Indicates the type of atomic chirality as an integer
    cf_feats_ref = torch.tensor(
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0], dtype=torch.int
    )
    assert torch.allclose(cf_feats, cf_feats_ref)


def test_total_numh_featurizer(test_mol):
    num_hf = TotalNumHFeaturizer()

    h2_feats = num_hf(test_mol)

    # Indicates the total number of hydrogens on the atom
    h2_feats_ref = torch.tensor(
        [2, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.int
    )
    assert torch.allclose(h2_feats, h2_feats_ref)


def test_hybridization_featurizer(test_mol, chiral_mol):
    hf = HybridizationFeaturizer()

    hf_feats = hf(test_mol)

    # Indicated the hybridization of the atom as an integer
    hf_feats_ref = torch.tensor(
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3], dtype=torch.int
    )
    assert torch.allclose(hf_feats, hf_feats_ref)


def test_aromaticity_featurizer(test_mol):
    af = AromaticityFeaturizer()
    af_feats = af(test_mol)

    # Indices if the atom is aromatic or not
    af_feats_ref = torch.tensor(
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=torch.int
    )
    assert torch.allclose(af_feats, af_feats_ref)


def test_periodic_table_featurizer(test_mol):
    pt = PeriodicTableFeaturizer()

    pt_feats = pt(test_mol)

    # The reference is a tensor of dimension 2
    # 1st dim: Atoms in the molecule
    # 2nd dim: [period, group] of the atom's position in the periodic table respectively.
    # Example: pt_feats_ref[1, 0] indicates the period of the 2nd atom in the molecule.
    pt_feats_ref = torch.tensor(
        [
            (2, 5),
            (2, 4),
            (2, 6),
            (2, 4),
            (2, 4),
            (2, 5),
            (2, 4),
            (2, 4),
            (2, 4),
            (2, 4),
            (3, 6),
            (2, 5),
            (2, 6),
            (2, 6),
            (2, 4),
            (2, 4),
            (2, 5),
            (2, 4),
            (2, 4),
            (2, 4),
            (2, 4),
            (2, 4),
            (3, 7),
            (2, 4),
            (2, 4),
        ],
        dtype=torch.int,
    )

    assert torch.allclose(pt_feats, pt_feats_ref)


def test_electronic_property_featurizer(test_mol):
    ep = ElectronicPropertyFeaturizer()

    ep_feats = ep(test_mol)

    # Reference is a tensor of dimension 2
    # 1st dim: Atoms in the molecule
    # 2nd dim: [electronegativity, ionization energy, electron affinity]
    ep_feats_ref = torch.Tensor(
        [
            [3.04, 14.534, 1.0721403509],
            [2.55, 11.26, 1.263],
            [3.44, 13.618, 1.461],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [3.04, 14.534, 1.0721403509],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [2.58, 10.36, 2.077],
            [3.04, 14.534, 1.0721403509],
            [3.44, 13.618, 1.461],
            [3.44, 13.618, 1.461],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [3.04, 14.534, 1.0721403509],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
            [3.16, 12.968, 3.617],
            [2.55, 11.26, 1.263],
            [2.55, 11.26, 1.263],
        ]
    )

    assert torch.allclose(ep_feats, ep_feats_ref)


def test_scaffold_featurizer(test_mol):
    sf = ScaffoldFeaturizer()
    sf_feats = sf(test_mol)

    # Indices if atom is present in the Bemis-Murcko scaffold of the molecule
    sf_feats_ref = torch.tensor(
        [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            True,
            False,
            True,
            True,
        ],
        dtype=torch.int,
    )
    assert torch.allclose(sf_feats, sf_feats_ref)


def test_smarts_featurizer(test_mol, acetic_acid, methylamine):
    sf = SmartsFeaturizer()
    sf_feats = sf(test_mol)

    # Reference is a tensor of dim 2
    # 1st dim: Atoms in the molecules
    # 2nd dim: [hydrogen bond donor, hydrogen bond acceptor, acidic, basic]

    sf_feats_ref = torch.tensor(
        [
            [True, False, False, False],
            [False, False, False, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [True, False, False, False],
            [False, True, False, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ],
        dtype=torch.int,
    )
    assert torch.allclose(sf_feats, sf_feats_ref)

    sf_feats = sf(acetic_acid)
    sf_feats_ref = torch.tensor(
        [
            [False, False, False, False],
            [False, False, True, False],
            [False, True, False, False],
            [True, False, False, False],
        ],
        dtype=torch.int,
    )
    assert torch.allclose(sf_feats, sf_feats_ref)

    sf_feats = sf(methylamine)
    sf_feats_ref = torch.tensor([[False, False, False, False], [True, True, False, True]], dtype=torch.int)
    assert torch.allclose(sf_feats, sf_feats_ref)


def test_crippen_featurizer(test_mol):
    cf = CrippenFeaturizer()

    cf_feats = cf(test_mol)

    # Reference is of dimension 2
    # 1st dimension: Atoms in the molecule
    # 2nd dimension: [logP, molar refractivity]
    cf_feats_ref = torch.Tensor(
        [
            [-1.019e00, 2.262e00],
            [-2.783e-01, 5.007e00],
            [1.129e-01, 2.215e-01],
            [1.360e-01, 3.509e00],
            [1.581e-01, 3.350e00],
            [-3.239e-01, 2.202e00],
            [2.713e-01, 3.904e00],
            [1.581e-01, 3.350e00],
            [1.581e-01, 3.350e00],
            [1.893e-01, 2.673e00],
            [-2.400e-03, 6.000e00],
            [-1.019e00, 2.262e00],
            [-3.339e-01, 7.774e-01],
            [-3.339e-01, 7.774e-01],
            [1.581e-01, 3.350e00],
            [1.581e-01, 3.350e00],
            [-3.239e-01, 2.202e00],
            [2.713e-01, 3.904e00],
            [2.713e-01, 3.904e00],
            [1.581e-01, 3.350e00],
            [1.581e-01, 3.350e00],
            [2.450e-01, 3.564e00],
            [6.895e-01, 5.853e00],
            [1.581e-01, 3.350e00],
            [1.581e-01, 3.350e00],
        ]
    )

    assert torch.allclose(cf_feats, cf_feats_ref)


def test_atomic_radius_featurizer(test_mol):
    arf = AtomicRadiusFeaturizer()
    arf_feats = arf(test_mol)

    # Reference is a tensor of dimension 2
    # 1st dim: Atoms in the molecule
    # 2nd dim: [bond radius, covalent radius, vdW radius]
    arf_feats_ref = torch.Tensor(
        [
            [0.7, 0.71, 1.6],
            [0.77, 0.76, 1.7],
            [0.66, 0.66, 1.55],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.7, 0.71, 1.6],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [1.04, 1.05, 1.8],
            [0.7, 0.71, 1.6],
            [0.66, 0.66, 1.55],
            [0.66, 0.66, 1.55],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.7, 0.71, 1.6],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
            [0.997, 1.02, 1.8],
            [0.77, 0.76, 1.7],
            [0.77, 0.76, 1.7],
        ]
    )

    assert torch.allclose(arf_feats, arf_feats_ref)
