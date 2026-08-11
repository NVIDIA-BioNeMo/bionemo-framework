# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from unittest.mock import sentinel

from pytest import MonkeyPatch

from . import utils


def test_initialize_distributed_process_group_keeps_kernel_assigned_port_reserved(monkeypatch):
    class FakeStore:
        port = 41000

    store = FakeStore()
    tcp_store_calls = []
    init_process_group_calls = []

    def fake_tcp_store(**kwargs):
        tcp_store_calls.append(kwargs)
        return store

    def fake_init_process_group(**kwargs):
        init_process_group_calls.append(kwargs)

    monkeypatch.setattr(utils.torch.distributed, "TCPStore", fake_tcp_store)
    monkeypatch.setattr(utils.torch.distributed, "init_process_group", fake_init_process_group)
    monkeypatch.setenv("MASTER_PORT", "original")

    with MonkeyPatch.context() as environment:
        utils._initialize_distributed_process_group(environment, sentinel.backend, rank=0, world_size=1)

        assert os.environ["MASTER_PORT"] == "41000"

    assert os.environ["MASTER_PORT"] == "original"
    assert tcp_store_calls == [
        {
            "host_name": utils.DEFAULT_MASTER_ADDR,
            "port": 0,
            "world_size": 1,
            "is_master": True,
            "wait_for_workers": False,
        }
    ]
    assert init_process_group_calls == [
        {
            "backend": sentinel.backend,
            "store": store,
            "rank": 0,
            "world_size": 1,
        }
    ]


def test_initialize_distributed_process_group_shares_rank_zero_endpoint(monkeypatch):
    tcp_store_calls = []
    init_process_group_calls = []

    class FakeStore:
        def __init__(self, port):
            self.port = 41000 if port == 0 else port

    def fake_tcp_store(**kwargs):
        tcp_store_calls.append(kwargs)
        return FakeStore(kwargs["port"])

    def fake_init_process_group(**kwargs):
        init_process_group_calls.append(kwargs)

    monkeypatch.setattr(utils.torch.distributed, "TCPStore", fake_tcp_store)
    monkeypatch.setattr(utils.torch.distributed, "init_process_group", fake_init_process_group)
    rendezvous_store = utils.torch.distributed.HashStore()

    with MonkeyPatch.context() as environment:
        utils._initialize_distributed_process_group(
            environment,
            sentinel.backend,
            rank=0,
            world_size=2,
            rendezvous_store=rendezvous_store,
        )
        utils._initialize_distributed_process_group(
            environment,
            sentinel.backend,
            rank=1,
            world_size=2,
            rendezvous_store=rendezvous_store,
        )

    assert [call["port"] for call in tcp_store_calls] == [0, 41000]
    assert [call["is_master"] for call in tcp_store_calls] == [True, False]
    assert [call["store"].port for call in init_process_group_calls] == [41000, 41000]
