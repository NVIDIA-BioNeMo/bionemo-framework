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


import sys

import fiddle as fdl

from src.utils.nemorun_utils import config_to_dict


class DummyClass:
    def __init__(self, a, b):
        self.a = a
        self.b = b


# Make DummyClass available through the module name that will be used in tests
sys.modules[__name__].DummyClass = DummyClass


class TestNemorunUtils:
    def test_config_to_dict(self):
        cfg = fdl.Config(DummyClass, a=1, b=fdl.Partial(DummyClass, a=2, b=3))

        d = config_to_dict(cfg)

        expected_dict = {"a": 1, "b": {"a": 2, "b": 3}}
        assert d == expected_dict

    def test_config_to_dict_with_dict(self):
        cfg = {"key1": fdl.Config(DummyClass, a=1, b=2), "key2": "value"}
        d = config_to_dict(cfg)
        expected_dict = {"key1": {"a": 1, "b": 2}, "key2": "value"}
        assert d == expected_dict
