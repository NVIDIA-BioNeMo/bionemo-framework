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

from dataclasses import dataclass, field

import pytest
from nemo.lightning import io

from bionemo.llm.utils import iomixin_utils as iom


@dataclass
class BaseDataClass(iom.WillHaveGetSetHparam):
    a: int = field(default_factory=lambda: 0)
    b: int = 3

    def lazy_update(self):
        self.set_hparam("b", self.b + 2)
        # Will update the value of b set later on making use of a future subclass IOMixin


@dataclass
class OverrideModelDataClass1(BaseDataClass, iom.IOMixinWithGettersSetters):
    a: int = field(default_factory=lambda: 4)
    c: int = 3


@dataclass
class OverrideModelDataClass2(BaseDataClass, iom.IOMixinWithGettersSetters):
    a: int = field(default_factory=lambda: 5)  # override default of a
    # do not define/override b
    c: int = 4  # new variable


class TestIOMixin:
    """TestCase on IOMixin.

    Notes:
        IOMixin only captures non-default __init__ arguments into self.__io__ to ensure no compatibility in loading older mcore config in newer versions.
    """

    def test_dataclasses_two_versions(self):
        _ = OverrideModelDataClass1(b=2)
        v1 = OverrideModelDataClass2(b=4)
        v1.lazy_update()  # the mutate method allows a variable that matches the init arg to be changed and tracked.
        coppied_v1 = io.reinit(v1)  # Simulate loading from a checkpoint
        v2 = OverrideModelDataClass2(a=3, b=1, c=5)
        coppied_v2 = io.reinit(v2)  # Simulate loading from a checkpoint
        assert v1.a != v2.a
        assert v1.a == coppied_v1.a
        assert v2.a == coppied_v2.a
        assert v1.b != v2.b
        assert v1.a == 5
        assert v1.b == 6
        assert v1.c == 4
        assert v2.a == 3
        assert v2.b == 1
        assert v2.c == 5
        assert v1.a == coppied_v1.a
        assert v1.b == coppied_v1.b
        assert v1.c == coppied_v1.c
        assert v2.a == coppied_v2.a
        assert v2.b == coppied_v2.b
        assert v2.c == coppied_v2.c

    def test_dataclass_out_of_sync(self):
        v1 = OverrideModelDataClass1()
        v1.set_hparam("b", 7, also_change_value=False)
        assert v1.b == 3, "Also change value False should not update the object in self."
        v1_copy = io.reinit(v1)
        assert v1_copy.b == 7, "V1 should re-initialize with the updated hyper-parameter."

        # Make sure looking up a non-existant hyper-parameter raises an error
        with pytest.raises(KeyError):
            v1.get_hparam("q")

        # Make sure we can get all hyper-parameters that are **non-default** non-defaultfactory objects
        assert v1.get_hparams() == {"b": 7}

        # Make sure by default we can change both the hyper-parameter and the attribute.
        v1_copy.set_hparam("b", 8)
        assert v1_copy.b == 8
        assert v1_copy.get_hparam("b") == 8

    def test_dataclass_hparam_modify_parent_default(self):
        v1 = OverrideModelDataClass1()
        v1.set_hparam("a", 7)
        assert v1.a == 7
        # Make sure we can get all **non-default** **non-defaultfactory** hyper-parameters
        assert v1.get_hparams() == {"a": 7}

        v1_copy = io.reinit(v1)
        assert v1_copy.a == 7, "V1 should re-initialize with the updated hyper-parameter."
        assert v1_copy.get_hparam("a") == 7
