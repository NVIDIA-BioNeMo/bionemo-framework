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


import copy
import os
import shlex
import subprocess
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ..utils import find_free_network_port, is_a6000_gpu


_REPO_BASE_DIR = Path(__file__).resolve().parents[4]
DEFAULT_HF_TOKENIZER_MODEL_PATH = str(_REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_256")


# Do this at collection time before we run any tests.
PRETEST_ENV = copy.deepcopy(os.environ)


# =============================================================================
# Eden (Llama) training tests
# =============================================================================


@pytest.mark.timeout(300)
@pytest.mark.slow
def test_eden_fine_tuning(
    tmp_path: Path,
    precision_recipe: str = "bf16_mixed",
):
    """Test that Eden (Llama 3.1 variant) models can train and fine-tune via the mbridge recipe.

    This verifies the Eden training pipeline works end-to-end with mock data.
    """
    world_size = 1
    mbs = 32
    gbs = mbs
    run_dir = tmp_path / "eden_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    master_port = find_free_network_port()
    cmd1 = f"""torchrun --nproc-per-node {world_size} --no-python --master_port {master_port} \
    train_eden \
        --hf-tokenizer-model-path {DEFAULT_HF_TOKENIZER_MODEL_PATH} \
        --model-size eden_7b --num-layers 2 \
        --max-steps 5 --eval-interval 5 \
        --eval-iters 1 --mock-data --result-dir {run_dir} \
        --micro-batch-size {mbs} --global-batch-size {gbs} --seq-length 64 \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 1 \
        --context-parallel-size 1 \
        --mixed-precision-recipe {precision_recipe} \
        --no-activation-checkpointing \
        --decay-steps 1000 --warmup-steps 10 \
        --log-interval 1 \
        --seed 41 --dataset-seed 33 \
    """

    cmd_parts = shlex.split(cmd1)
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        env["NCCL_P2P_DISABLE"] = "1"
    result = subprocess.run(cmd_parts, check=False, capture_output=True, text=True, cwd=run_dir, env=env)

    print(f"Return code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    print(f"STDERR:\n{result.stderr}")

    assert result.returncode == 0, (
        f"Eden training failed with return code {result.returncode}\nSTDERR:\n{result.stderr}"
    )
    result_dir = run_dir / "eden"
    ckpt_dir = result_dir / "checkpoints"
    tb_log_dir = result_dir / "tb_logs"
    assert ckpt_dir.exists() and ckpt_dir.is_dir(), "Checkpoints directory not found"
    assert tb_log_dir.exists() and tb_log_dir.is_dir(), "TensorBoard logs directory not found"
    iter_5_dir = ckpt_dir / "iter_0000005"
    assert iter_5_dir.exists() and iter_5_dir.is_dir(), f"No iteration-5 checkpoint found in {ckpt_dir}"

    event_acc = EventAccumulator(str(tb_log_dir))
    event_acc.Reload()
    lm_loss_events = event_acc.Scalars("lm loss")
    assert len(lm_loss_events) > 0, "No 'lm loss' events found"
    assert lm_loss_events[-1].step == 5, f"Expected training to end at step 5, got {lm_loss_events[-1].step}"

    # Fine-tune from the checkpoint into a new result dir
    ft_run_dir = tmp_path / "eden_ft_run"
    ft_run_dir.mkdir(parents=True, exist_ok=True)
    cmd2 = cmd1.rstrip().replace(f"--result-dir {run_dir}", f"--result-dir {ft_run_dir}")
    cmd2 += f" --finetune-ckpt-dir {ckpt_dir} "
    cmd_parts_2 = shlex.split(cmd2)

    result_2 = subprocess.run(cmd_parts_2, check=False, capture_output=True, text=True, cwd=ft_run_dir, env=env)
    print(f"Run 2 Return code: {result_2.returncode}")
    if result_2.returncode != 0:
        print(f"Run 2 STDERR:\n{result_2.stderr}")
    assert result_2.returncode == 0, f"Eden fine-tuning failed with return code {result_2.returncode}"

    ft_result_dir = ft_run_dir / "eden"
    ft_tb_log_dir = ft_result_dir / "tb_logs"
    assert ft_tb_log_dir.exists(), "TensorBoard logs directory not found after fine-tuning"

    event_acc_2 = EventAccumulator(str(ft_tb_log_dir))
    event_acc_2.Reload()
    lm_loss_events_2 = event_acc_2.Scalars("lm loss")
    assert len(lm_loss_events_2) > 0, "No 'lm loss' events found in fine-tuning run"
