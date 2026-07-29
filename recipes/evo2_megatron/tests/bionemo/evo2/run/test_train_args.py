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

from bionemo.evo2.run.train import parse_args


def test_no_save_optim_flag_defaults_to_false():
    args = parse_args(["--mock-data"])

    assert args.no_save_optim is False


def test_no_save_optim_flag_can_be_enabled():
    args = parse_args(["--mock-data", "--no-save-optim"])

    assert args.no_save_optim is True
