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

"""Run a real Evo2 endpoint while recording flat segmented-kernel execution per rank."""

import json
import os
import sys
from collections import Counter
from pathlib import Path

from bionemo.evo2.models.megatron.hyena.hyena_mixer import HyenaMixer
from bionemo.evo2.run import infer as infer_module
from bionemo.evo2.run import predict as predict_module


_ORIGINAL_SEGMENTED_PREFILL = HyenaMixer._mix_flat_segmented_prefill
_OPERATOR_COUNTS: Counter[str] = Counter()
_PROBE_STATE = {"max_segments": 0}


def _recording_segmented_prefill(self, *args, **kwargs):
    """Record only calls that completed the real segmented implementation."""
    result = _ORIGINAL_SEGMENTED_PREFILL(self, *args, **kwargs)
    _OPERATOR_COUNTS[self.operator_type] += 1
    cu_seqlens = args[1] if len(args) > 1 else kwargs["cu_seqlens"]
    _PROBE_STATE["max_segments"] = max(_PROBE_STATE["max_segments"], int(cu_seqlens.numel()) - 1)
    probe_dir = Path(os.environ["EVO2_PACKED_PROBE_DIR"])
    probe_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ["RANK"])
    record = {
        "rank": rank,
        "calls": sum(_OPERATOR_COUNTS.values()),
        "max_segments": _PROBE_STATE["max_segments"],
        "operator_counts": dict(sorted(_OPERATOR_COUNTS.items())),
    }
    temporary_path = probe_dir / f"rank-{rank}.json.tmp"
    temporary_path.write_text(json.dumps(record, sort_keys=True))
    temporary_path.replace(probe_dir / f"rank-{rank}.json")
    return result


def main() -> None:
    """Patch the concrete kernel entrypoint and dispatch the requested CLI."""
    if len(sys.argv) < 2 or sys.argv[1] not in {"infer", "predict"}:
        raise SystemExit("usage: packed_parallel_probe.py {infer,predict} [endpoint arguments]")
    endpoint = sys.argv.pop(1)
    HyenaMixer._mix_flat_segmented_prefill = _recording_segmented_prefill
    if endpoint == "infer":
        infer_module.main()
    else:
        predict_module.main()


if __name__ == "__main__":
    main()
