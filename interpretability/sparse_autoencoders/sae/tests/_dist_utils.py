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

"""Multi-process test harness for tensor-parallel logic (CPU / gloo).

`run_distributed(fn, world_size)` launches `world_size` processes, each running
`fn(rank, world_size, *args)` inside an initialized gloo process group, and returns
`{rank: return_value}`. Exceptions in a worker are re-raised in the parent with the
full traceback.

We use the *spawn* start method: the pytest session also runs GPU/autograd tests,
and fork-after-autograd/CUDA is unsafe ("Unable to handle autograd's threading in
combination with fork-based multiprocessing"). Spawn starts clean interpreters, so
we extend PYTHONPATH for the children to import both the `sae` package and this
`tests` package (which holds the worker functions).
"""

import os
import socket
import traceback

import torch.distributed as dist
import torch.multiprocessing as mp


_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))  # .../sae/tests
_SAE_DIR = os.path.dirname(_TESTS_DIR)  # .../sae      (root of the `tests` package)
_SRC_DIR = os.path.join(_SAE_DIR, "src")  # .../sae/src (root of the `sae` package)


def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _worker(rank, world_size, backend, fn, args, ret):
    try:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        ret[rank] = fn(rank, world_size, *args)
    except Exception:
        ret[f"error_{rank}"] = traceback.format_exc()
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_distributed(fn, world_size, backend="gloo", args=()):
    """Spawn `world_size` gloo workers running fn; return {rank: result}."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_free_port())
    # Ensure spawned children can import `sae` and the `tests` package.
    os.environ["PYTHONPATH"] = os.pathsep.join([_SRC_DIR, _SAE_DIR, os.environ.get("PYTHONPATH", "")])

    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    procs = [
        ctx.Process(target=_worker, args=(rank, world_size, backend, fn, args, ret)) for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    errors = {k: v for k, v in ret.items() if isinstance(k, str) and k.startswith("error_")}
    if errors:
        raise AssertionError("distributed worker failed:\n" + "\n".join(errors.values()))
    for p in procs:
        if p.exitcode != 0:
            raise RuntimeError(f"worker exited with code {p.exitcode}")
    return {k: v for k, v in ret.items() if isinstance(k, int)}
