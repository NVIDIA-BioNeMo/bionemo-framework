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

import sys
from types import SimpleNamespace

from bionemo.common.data.load import default_ngc_client


def test_default_ngc_client_guest_configuration_omits_ace(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, *args):
            self.args = args
            self.configure_calls = []
            clients.append(self)

        def configure(self, **kwargs):
            self.configure_calls.append(kwargs)
            if len(clients) == 1 and not kwargs:
                raise ValueError("invalid API key")

    monkeypatch.setitem(sys.modules, "ngcsdk", SimpleNamespace(Client=FakeClient))

    client = default_ngc_client()

    assert client is clients[1]
    assert client.args == ("no-apikey",)
    assert client.configure_calls == [
        {
            "api_key": "no-apikey",
            "org_name": "no-org",
            "team_name": "no-team",
        }
    ]
