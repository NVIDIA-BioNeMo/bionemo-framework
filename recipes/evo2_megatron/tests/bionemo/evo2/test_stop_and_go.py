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
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH

from .utils import find_free_network_port, is_fp4_supported, is_fp8_supported, is_mxfp8_supported


# Do this at collection time before we run any tests.
PRETEST_ENV = copy.deepcopy(os.environ)


@pytest.mark.parametrize(
    "tp_size,cp_size,dp_size,dp_rank_check,precision_recipe",
    [
        pytest.param(
            1, 1, 1, False, "bf16_mixed", marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
        ),
        pytest.param(1, 1, 1, False, "bf16_with_fp8_current_scaling_mixed"),
        pytest.param(
            1,
            1,
            1,
            False,
            "bf16_with_fp8_delayed_scaling_mixed",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI"),
        ),  # XFAIL
        pytest.param(
            1,
            1,
            1,
            False,
            "bf16_with_fp8_subchannel_scaling_mixed",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI"),
        ),
        pytest.param(
            1,
            1,
            1,
            False,
            "bf16_with_nvfp4_mixed",
            marks=(
                pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI"),
                pytest.mark.xfail(reason="NVFP4: TE checkpoint/backward issues; known on non-Blackwell", strict=False),
            ),
        ),
        pytest.param(
            1,
            1,
            1,
            False,
            "bf16_with_mxfp8_mixed",
            marks=(
                pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI"),
                pytest.mark.xfail(reason="MXFP8: TE checkpoint/backward issues; known on non-Blackwell", strict=False),
            ),
        ),
        pytest.param(
            1, 1, 2, True, "bf16_mixed", marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
        ),
        pytest.param(
            1, 1, 2, False, "bf16_mixed", marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
        ),
        pytest.param(
            1, 2, 1, True, "bf16_mixed", marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
        ),
        pytest.param(
            2, 1, 1, False, "bf16_mixed", marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
        ),
    ],
)
@pytest.mark.slow
def test_stop_and_go(
    tmp_path: Path,
    tp_size: int,
    cp_size: int,
    dp_size: int,
    dp_rank_check: bool,
    precision_recipe: str,
    pp_size: int = 1,
):
    """Test stop and go functionality."""
    world_size = tp_size * pp_size * cp_size * dp_size
    mbs = 32
    gbs = mbs * dp_size
    num_gpus = torch.cuda.device_count()
    if world_size > num_gpus:
        pytest.skip(f"World size {world_size} is greater than the number of GPUs {num_gpus}")
    if "nvfp4" in precision_recipe and not is_fp4_supported():
        pytest.skip("NVFP4 is not supported on this device")
    if "mxfp8" in precision_recipe and not is_mxfp8_supported():
        pytest.skip("MXFP8 is not supported on this device")
    if "fp8" in precision_recipe and not is_fp8_supported():
        pytest.skip("FP8 is not supported on this device")
    if "bf16_with_fp8_delayed_scaling_mixed" == precision_recipe and is_fp8_supported():
        pytest.skip(reason="FP8 delayed scaling is not currently working with Evo2, use another FP8 recipe.")
    if "bf16_with_fp8_subchannel_scaling_mixed" == precision_recipe and is_fp8_supported():
        pytest.skip(reason="FP8 subchannel scaling is not currently working with Evo2 on some GPUs.")
    run_dir = tmp_path / f"run_tp{tp_size}_pp{pp_size}_cp{cp_size}_dp{dp_size}_rc{dp_rank_check}_pr{precision_recipe}"
    run_dir.mkdir(parents=True, exist_ok=True)
    master_port = find_free_network_port()
    dp_rank_check_str = "--debug-ddp-parity-freq 5" if dp_rank_check else ""
    cmd1 = f"""torchrun --nproc-per-node {world_size} --no-python --master_port {master_port} \
    train_evo2 \
        --hf-tokenizer-model-path {DEFAULT_HF_TOKENIZER_MODEL_PATH} \
        --model-size striped_hyena_1b_nv_parallel --num-layers 4 --hybrid-override-pattern SDH* \
        --max-steps 5 --eval-interval 5 \
        --eval-iters 3 --mock-data --result-dir {run_dir} \
        --micro-batch-size {mbs} --global-batch-size {gbs} --seq-length 512 \
        --tensor-model-parallel {tp_size} \
        --pipeline-model-parallel {pp_size} \
        --context-parallel {cp_size} \
        --mixed-precision-recipe {precision_recipe} \
        --overlap-param-gather \
        --overlap-grad-reduce \
        {dp_rank_check_str} \
        --use-precision-aware-optimizer --dataset-seed 33 \
        --seed 41 --spike-no-more-embedding-init \
        --no-weight-decay-embeddings --cross-entropy-loss-fusion \
        --grad-reduce-in-fp32 \
        --decay-steps 1000 --warmup-steps 10 \
        --eod-pad-in-loss-mask \
        --log-interval 1
    """

    # Split the command and run it
    cmd_parts = shlex.split(cmd1)
    env = copy.deepcopy(PRETEST_ENV)
    env["NCCL_P2P_DISABLE"] = "1"
    result = subprocess.run(cmd_parts, check=False, capture_output=True, text=True, cwd=run_dir, env=env)

    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    # For debugging, print the output
    print(f"Return code: {returncode}")
    print(f"STDOUT:\n{stdout}")
    print(f"STDERR:\n{stderr}")

    # Assert the command succeeded
    assert returncode == 0, f"Command failed with return code {returncode}\nSTDERR:\n{stderr}"
    result_dir = run_dir / "evo2"
    ckpt_dir = result_dir / "checkpoints"
    tb_log_dir = result_dir / "tb_logs"
    assert ckpt_dir.exists() and ckpt_dir.is_dir(), "Checkpoints directory not found"
    assert tb_log_dir.exists() and tb_log_dir.is_dir(), "TensorBoard logs directory not found"
    iter_5_dir = ckpt_dir / "iter_0000005"
    assert iter_5_dir.exists() and iter_5_dir.is_dir(), f"No iterations 5 checkpoint found in {ckpt_dir}"
    assert len(list(ckpt_dir.glob("iter_*"))) == 1, f"Expected 1 iterations, found {list(ckpt_dir.glob('iter_*'))}"
    # Load tensorboard logs to verify they were written correctly

    # Find the events file(s) in tb_log_dir
    event_files = list(tb_log_dir.rglob("events.out.*"))
    assert len(event_files) > 0, f"No tensorboard event files found in {tb_log_dir}"

    # Load events from the event files
    event_acc = EventAccumulator(str(tb_log_dir))
    event_acc.Reload()

    # 1. collect the last loss, as well as the average of the last step validation losses, as well as the last step
    # Note: EventAccumulator.Scalars returns a list of ScalarEvent(wall_time, step, value)
    lm_loss_events = event_acc.Scalars("lm loss")

    assert len(lm_loss_events) > 0, "No 'lm loss' events found in run 1"
    last_lm_loss_step = lm_loss_events[-1].step

    assert last_lm_loss_step == 5, f"Expected run 1 to end at step 5, but got {last_lm_loss_step}"

    # 2. run the above training command a second time, this time set max_steps to 10. Verify that the run resumes from the last step.
    #   Do this by moving the tb_logs to a different directory from the first part so the second run makes fresh logs.
    tb_log_dir_run1 = result_dir / "tb_logs_run1"
    if tb_log_dir.exists():
        shutil.move(str(tb_log_dir), str(tb_log_dir_run1))

    # Modify the command to increase max steps to 10
    # We reuse the same result_dir so it picks up the checkpoint
    cmd2 = cmd1.replace("--max-steps 5", "--max-steps 10")
    cmd_parts_2 = shlex.split(cmd2)

    print("Starting Run 2 (resuming to step 10)...")
    result_2 = subprocess.run(cmd_parts_2, check=False, capture_output=True, text=True, cwd=run_dir, env=env)

    print(f"Run 2 Return code: {result_2.returncode}")
    if result_2.returncode != 0:
        print(f"Run 2 STDERR:\n{result_2.stderr}")

    assert result_2.returncode == 0, f"Run 2 failed with return code {result_2.returncode}"

    # 3. Load the new tb logs as before, and sanity check my recommendations as well as any others that make sense.
    assert tb_log_dir.exists(), "TensorBoard logs directory not found after Run 2"

    event_acc_2 = EventAccumulator(str(tb_log_dir))
    event_acc_2.Reload()

    lm_loss_events_2 = event_acc_2.Scalars("lm loss")
    assert len(lm_loss_events_2) > 0, "No 'lm loss' events found in run 2"

    first_step_run2 = lm_loss_events_2[0].step
    last_step_run2 = lm_loss_events_2[-1].step

    # Sanity checks:
    # 1. Resumption: Should start after step 5 (e.g., step 6)
    assert first_step_run2 > 5, f"Run 2 should resume after step 5, but started at {first_step_run2}"

    # 2. Completion: Should reach step 10
    assert last_step_run2 == 10, f"Run 2 should reach step 10, but ended at {last_step_run2}"

    # 3. Loss Continuity check (basic): The first loss of run 2 should be reasonably close to the last loss of run 1,
    #    or at least not exploding, though optimization steps might cause fluctuations.
    first_loss_run2 = lm_loss_events_2[0].value
    last_loss_run1 = lm_loss_events[-1].value
    print(f"Run 1 Last Loss: {last_loss_run1}, Run 2 First Loss: {first_loss_run2}")
    assert first_loss_run2 - last_loss_run1 < 0.1, (
        f"Run 2 first loss {first_loss_run2} is not better than run 1 last loss {last_loss_run1} by no worse than 0.1"
    )
