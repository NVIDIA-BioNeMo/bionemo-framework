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


def test_import_bionemo_core():
    from bionemo import core as subpackage

    assert subpackage is not None
    del subpackage


def test_import_bionemo_llm():
    from bionemo import core as subpackage

    assert subpackage is not None
    del subpackage


def test_import_bionemo_geneformer():
    from bionemo import geneformer as subpackage

    assert subpackage is not None
    del subpackage


def test_import_bionemo_esm2():
    from bionemo import esm2 as subpackage

    assert subpackage is not None
    del subpackage
