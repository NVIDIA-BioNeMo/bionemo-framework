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

import json
import os
import subprocess
from pathlib import Path


RECIPE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = RECIPE_ROOT / "scripts/run_paper_hpo_generation.sh"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def test_hpo_generation_resume_overwrite_dry_run_and_length_guard(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "evo2_phage_generation",
        """#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--prompt-lengths", nargs="+", required=True)
parser.add_argument("--num-prompts", type=int, required=True)
parser.add_argument("--id-prefix", required=True)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
for length in args.prompt_lengths:
    path = args.output_dir / f"{args.id_prefix}_prompt{length}_{args.num_prompts}.jsonl"
    with path.open("w") as stream:
        for index in range(args.num_prompts):
            stream.write(json.dumps({"id": f"p{index}", "prompt": "+~" + "A" * int(length)}) + "\\n")
""",
    )
    _write_executable(
        fake_bin / "torchrun",
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
output = Path(arguments[arguments.index("--output-file") + 1])
prompts = Path(arguments[arguments.index("--prompt-file") + 1])
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w") as stream:
    for line in prompts.read_text().splitlines():
        record = json.loads(line)
        stream.write(json.dumps({**record, "completion": "ACGT"}) + "\\n")
with Path(os.environ["FAKE_TORCHRUN_CALLS"]).open("a") as stream:
    stream.write(json.dumps(arguments) + "\\n")
""",
    )

    run_root = tmp_path / "run"
    output = run_root / "jsonl/phix174_prompt4_temp0.7.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text('{"id":"existing-1"}\n{"id":"existing-2"}\n')
    calls = tmp_path / "torchrun-calls.jsonl"
    env = {
        **os.environ,
        "SOURCE_ENV": "0",
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "RUN_ROOT": str(run_root),
        "PROMPT_LENGTHS": "4",
        "TEMPERATURES": "0.7",
        "NUM_PROMPTS": "2",
        "TARGET_LENGTH": "10",
        "FAKE_TORCHRUN_CALLS": str(calls),
    }

    subprocess.run(["bash", str(SCRIPT)], check=True, env=env, cwd=RECIPE_ROOT, timeout=30)
    manifest = (run_root / "hpo_generation_manifest.tsv").read_text()
    assert "\tSKIP\t2\t" in manifest
    assert not calls.exists()

    subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        env={**env, "OVERWRITE": "1"},
        cwd=RECIPE_ROOT,
        timeout=30,
    )
    arguments = json.loads(calls.read_text().splitlines()[-1])
    assert arguments[arguments.index("--max-new-tokens") + 1] == "6"
    assert len(output.read_text().splitlines()) == 2

    dry_root = tmp_path / "dry"
    subprocess.run(
        ["bash", str(SCRIPT)],
        check=True,
        env={**env, "RUN_ROOT": str(dry_root), "DRY_RUN": "1"},
        cwd=RECIPE_ROOT,
        timeout=30,
    )
    assert "DRY_RUN" in (dry_root / "logs/phix174_prompt4_temp0.7.infer.log").read_text()
    assert len(calls.read_text().splitlines()) == 1

    guarded = subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env={**env, "RUN_ROOT": str(tmp_path / "guarded"), "TARGET_LENGTH": "4", "DRY_RUN": "1"},
        cwd=RECIPE_ROOT,
        timeout=30,
    )
    assert guarded.returncode == 2
    assert "must exceed prompt_len=4" in guarded.stderr
    assert len(calls.read_text().splitlines()) == 1
