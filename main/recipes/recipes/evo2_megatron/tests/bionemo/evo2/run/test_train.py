# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import load

from bionemo.core.data.load import load as bionemo_load
from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH, DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

from ..utils import find_free_network_port, is_a6000_gpu, is_fp4_supported, is_fp8_supported, is_mxfp8_supported


TensorLike = Union[torch.Tensor, Iterable[torch.Tensor]]


def _as_iter(x: TensorLike):
    return x if (isinstance(x, Iterable) and not isinstance(x, torch.Tensor)) else [x]


def _fro_norm(x: TensorLike) -> torch.Tensor:
    """Frobenius norm; supports sharded tensors (sum of shard ||·||_F^2)."""
    it = list(_as_iter(x))  # Convert to list to avoid iterator consumption issues
    if not it:
        return torch.tensor(0.0, device="cpu")
    s = torch.tensor(0.0, device=it[0].device)
    for t in it:
        s = s + t.float().pow(2).sum()
    return torch.sqrt(s)


def machine_epsilon_for_dtype(dtype: torch.dtype) -> float:
    """Return machine epsilon for dtype. For FP8, use BF16 epsilon per paper."""
    # Standard types
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return float(torch.finfo(dtype).eps)
    # FP8 recipes: accum/store typically BF16/FP32; use BF16 epsilon
    if hasattr(torch, "float8_e4m3fn") and dtype in (
        torch.float8_e4m3fn,
        getattr(torch, "float8_e5m2fn", None),
    ):
        return float(torch.finfo(torch.bfloat16).eps)
    # Fallback
    return float(torch.finfo(torch.float32).eps)


def relative_grad_diff(g_hat: TensorLike, g_ref: TensorLike, eps_den: float = 1e-30) -> float:
    """Relative difference ||g_hat - g_ref||_F / ||g_ref||_F.

    Accepts a single tensor or an iterable of shards for each argument.
    """
    # Convert to lists to avoid iterator consumption issues
    gh_list = list(_as_iter(g_hat))
    gr_list = list(_as_iter(g_ref))

    if len(gh_list) != len(gr_list):
        raise ValueError(f"Shard count mismatch: {len(gh_list)} vs {len(gr_list)}")

    if not gh_list:
        return 0.0

    num_sq = torch.tensor(0.0, device=gh_list[0].device)
    for a, b in zip(gh_list, gr_list):
        num_sq = num_sq + (a.float() - b.float()).pow(2).sum()
    num = torch.sqrt(num_sq)
    den = _fro_norm(g_ref)
    return float(num / (den + eps_den))


def expected_rel_bound(
    l: int,  # noqa: E741
    *,
    L: int = 32,  # noqa: N803
    C: float = 1.03,  # noqa: N803
    dtype: Optional[torch.dtype] = torch.bfloat16,
    k: float = 4.0,
) -> float:
    """Bound ~ k * (C ** (L + 1 - l)) * eps_mch, with 1-based layer index l.

    - L is hard-coded default to 32 per your request.
    - C is 'close to 1'; 1.01-1.05 are reasonable defaults.
    - k absorbs the hidden constant in big-O; 2-8 are common choices.
    - dtype controls eps_mch; for FP8 use BF16 epsilon (see https://www.arxiv.org/pdf/2506.09280 theorem 5.3).
    """
    eps_mch = machine_epsilon_for_dtype(dtype or torch.bfloat16)
    depth = L + 1 - l  # 1-based depth from the top (as in the theorem)
    depth = max(depth, 0)
    return float(k * (C**depth) * eps_mch)


def check_gradient(
    g_hat: TensorLike,
    g_ref: TensorLike,
    l: int,  # noqa: E741
    *,
    L: int = 32,  # noqa: N803
    C: float = 1.03,  # noqa: N803
    dtype: Optional[torch.dtype] = None,
    k: float = 4.0,
) -> Tuple[float, float, bool]:
    """Compute (rel_error, bound, ok) for layer l.

    - If dtype is None, infer from g_ref (or g_hat if needed).
    # See https://www.arxiv.org/pdf/2506.09280 theorem 5.3
    """
    # Infer dtype if not provided
    if dtype is None:
        gr_list = list(_as_iter(g_ref))
        if gr_list:
            dtype = gr_list[0].dtype
        else:
            dtype = torch.bfloat16  # fallback
    rel = relative_grad_diff(g_hat, g_ref)
    bnd = expected_rel_bound(l, L=L, C=C, dtype=dtype, k=k)
    return rel, bnd, (rel <= bnd)


def _filter_optimizer_tensors(plain_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return only optimizer-related tensors from a flat checkpoint tensor dict."""
    return {k: v for k, v in plain_tensors.items() if k.startswith("optimizer.") and ".exp_avg." in k}


def assert_grads_close(left: torch.Tensor, right: torch.Tensor):
    """Assert that two gradient tensors are close using theorem 5.3 of https://www.arxiv.org/pdf/2506.09280."""
    # Implement theorem 5.3 of https://www.arxiv.org/pdf/2506.09280

    # This is the real test:
    # k=5.0 provides margin for small numerical differences in sequence parallel gradient sync
    rel, bnd, ok = check_gradient(
        left, right, l=0, dtype=torch.bfloat16, k=5.0
    )  # hard code to layer 0 since that's the most permissive

    # If the real test above fails, run an assert close for the useful diagnostics and raise either way.
    if not ok:
        rel_shuff, _, ok_shuff = check_gradient(
            left, torch.roll(right, shifts=-1, dims=-1), l=0, dtype=torch.bfloat16, k=5.0
        )

        try:
            torch.testing.assert_close(left, right)
            msg = (
                "AssertionError on relative norm magnitude "
                f"(rel={rel}, bnd={bnd}, ok={ok}, rel_shuff={rel_shuff}, ok_shuff={ok_shuff}) "
                "but torch.testing.assert_close(left, right) passes. \n"
                f"Left: {left.shape}/{left.dtype} {left}\n"
                f"Right: {right.shape}/{right.dtype} {right}"
            )
        except AssertionError as e:
            msg = (
                "AssertionError on relative norm magnitude "
                f"(rel={rel}, bnd={bnd}, ok={ok}, rel_shuff={rel_shuff}, ok_shuff={ok_shuff}): {e}\n"
                f"Left: {left.shape}/{left.dtype} {left}\n"
                f"Right: {right.shape}/{right.dtype} {right}"
            )
        raise AssertionError(msg)


def _assert_optimizer_tensors_equal(
    left: Dict[str, torch.Tensor],
    right: Dict[str, torch.Tensor],
    eps=1e-4,
):
    left_keys = set(left.keys())
    right_keys = set(right.keys())

    only_left = sorted(left_keys - right_keys)
    only_right = sorted(right_keys - left_keys)
    assert not only_left and not only_right, (
        f"Optimizer tensor keys mismatch.\nOnly in left: {only_left}\nOnly in right: {only_right}"
    )
    some_non_zero = False
    assertions = []
    for key in sorted(left_keys):
        lt, rt = left[key], right[key]
        assert lt.shape == rt.shape and lt.dtype == rt.dtype, (
            f"Tensor meta mismatch for {key}: {lt.shape}/{lt.dtype} vs {rt.shape}/{rt.dtype}"
        )
        # Reduce the rate of 0 vs near 0 rtol failures by adding a small epsilon
        left_scale = torch.max(torch.abs(lt))
        right_scale = torch.max(torch.abs(rt))
        if left_scale <= eps and right_scale <= eps:
            print(
                f"WARNING: zero-ish scale tensors ({left_scale=} vs {right_scale=}) "
                f"so they will trivially pass comparing {key=}"
            )
        else:
            some_non_zero = True
        try:
            assert_grads_close(lt, rt)
            print(f"Optimizer tensors match for {key}")
        except AssertionError as e:
            assertions.append(AssertionError(f"AssertionError for {key}: {e}"))
    assert not assertions, f"Assertion Errors found comparing keys: {assertions}"
    assert some_non_zero, "No non-zero tensors found in this comparison"


def load_dist_checkpoint_pt(
    ckpt_dir,
    metadata_ckpt_dir=None,
    pattern=r"optimizer",
    device="cpu",
    return_full_empty: bool = False,
):
    """Return {full_key: tensor} for every tensor whose key matches *pattern*."""
    meta_ckpt_dir = Path(metadata_ckpt_dir or ckpt_dir)
    meta_reader = FileSystemReader(str(meta_ckpt_dir))

    # --- fast metadata pass (no tensor data yet) -----------------------------
    meta = meta_reader.read_metadata()  # tiny JSON read
    tmeta = meta.state_dict_metadata  # key ➜ TensorMetadata
    if return_full_empty:
        wanted = [k for k in tmeta if hasattr(tmeta[k], "size")]
    else:
        wanted = [k for k in tmeta if re.search(pattern, k) and hasattr(tmeta[k], "size")]
    if not wanted:
        raise ValueError(f"No keys matching /{pattern}/ in {ckpt_dir}")

    # --- build "empty" placeholders -----------------------------------------
    placeholders = {
        k: torch.empty(tuple(tmeta[k].size), dtype=tmeta[k].properties.dtype, device=device) for k in wanted
    }
    if return_full_empty:
        return placeholders
    # --- stream just those tensors (no process-group needed) -----------------
    data_reader = FileSystemReader(str(ckpt_dir))

    load(
        state_dict=placeholders,
        storage_reader=data_reader,
        no_dist=False,  # switches off all collectives
    )
    return placeholders  # dict[str, Tensor]


def assert_optimizer_states_match(checkpoint_dirs):
    """Compare optimizer state across provided torch_dist checkpoints.

    - Keys: ensure the set of optimizer tensor keys match across checkpoints
    - Values: ensure corresponding tensors are equal (allclose)
    - Structure (non-tensor common state): ensure common optimizer structures match
    """
    assert len(checkpoint_dirs) > 1, "This test requires 2 or more checkpoints <dir1> [<dir2> ...]."

    base_dir = checkpoint_dirs[0]

    # Compare optimizer tensors
    base_plain = load_dist_checkpoint_pt(base_dir)
    base_opt_tensors = _filter_optimizer_tensors(base_plain)
    assert base_opt_tensors, f"No optimizer tensors found in checkpoint: {base_dir}"
    assertions = []
    for other_dir in checkpoint_dirs[1:]:
        try:
            other_plain = load_dist_checkpoint_pt(other_dir)
            other_opt_tensors = _filter_optimizer_tensors(other_plain)
            assert other_opt_tensors, f"No optimizer tensors found in checkpoint: {other_dir}"
            _assert_optimizer_tensors_equal(base_opt_tensors, other_opt_tensors)
            print(f"Optimizer tensors match for {base_dir} and {other_dir}")
            del other_plain
            del other_opt_tensors
        except AssertionError as e:  # noqa: PERF203
            msg = f"AssertionError comparing {base_dir} to {other_dir}:\n{e}"
            print(f"Optimizer tensors mismatch for {base_dir} and {other_dir}:\n{msg}")
            assertions.append(AssertionError(msg))
    assert not assertions, f"AssertionErrors comparing {checkpoint_dirs}:\n{assertions}"


# Do this at collection time before we run any tests.
PRETEST_ENV = copy.deepcopy(os.environ)


def _run_train_command(cmd: str, run_dir: Path) -> str:
    env = copy.deepcopy(PRETEST_ENV)
    env["MASTER_PORT"] = str(find_free_network_port())
    result = subprocess.run(
        shlex.split(cmd),
        check=False,
        capture_output=True,
        text=True,
        cwd=run_dir,
        env=env,
    )
    if result.returncode != 0:
        print(f"Return code: {result.returncode}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"Command failed with return code {result.returncode}\nSTDERR:\n{result.stderr}"
    return result.stdout


def _distributed_training_cmd(
    *,
    path: Path,
    max_steps: int,
    val_check: int,
    num_devices: int,
    dp: int,
    tp: int,
    cp: int,
    pp: int,
    finetune_ckpt_dir: Path,
    additional_args: str = "",
) -> str:
    micro_batch_size = 1 if dp == 2 else 2
    return (
        f"torchrun --nproc-per-node {num_devices} --no-python train_evo2 "
        f"--mock-data --result-dir {path} "
        f"--hf-tokenizer-model-path {DEFAULT_HF_TOKENIZER_MODEL_PATH} "
        "--model-size evo2_7b --num-layers 4 --hybrid-override-pattern SDH* "
        "--no-activation-checkpointing --optim-full-reshardable "
        f"--finetune-ckpt-dir {finetune_ckpt_dir} "
        f"--max-steps {max_steps} --eval-interval {val_check} --eval-iters 1 "
        f"--seq-length 64 --hidden-dropout 0.0 --attention-dropout 0.0 "
        f"--micro-batch-size {micro_batch_size} --global-batch-size 2 "
        f"--tensor-model-parallel-size {tp} --pipeline-model-parallel-size {pp} --context-parallel-size {cp} "
        "--adam-beta1 0 --adam-beta2 0 --ckpt-format torch_dist --log-interval 1  --decay-steps 1000 --warmup-steps 10 "
        f"--seed 42 --dataset-seed 33 {additional_args}"
    )


@pytest.mark.timeout(300)
@pytest.mark.slow
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, id="tp_1_pretrain"),
        pytest.param(
            2,
            id="tp_2_pretrain",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="TP=2 requires at least 2 GPUs for pretraining."
            ),
        ),
    ],
)
def test_fine_tuning(
    tmp_path: Path,
    tp_size: int,
    cp_size: int = 1,
    dp_size: int = 1,
    final_tp: int = 1,
    dp_rank_check: bool = True,
    precision_recipe: str = "bf16_mixed",
    pp_size: int = 1,
):
    """Test fine-tuning functionality, which should mirror stop/go but reset optimizer, data, and training state."""
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
        pytest.xfail(reason="FP8 delayed scaling is not currently working with Evo2, use another FP8 recipe.")
    if "bf16_with_fp8_subchannel_scaling_mixed" == precision_recipe and is_fp8_supported():
        pytest.xfail(reason="FP8 subchannel scaling is not currently working with Evo2 on some GPUs.")
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
        --log-interval 1 \
    """

    # Split the command and run it
    cmd_parts = shlex.split(cmd1)
    env = copy.deepcopy(PRETEST_ENV)
    if is_a6000_gpu():
        # Fix hanging issue on A6000 GPUs with multi-gpu tests
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
    ft_run_dir = (
        tmp_path / f"ft_run_tp{tp_size}_pp{pp_size}_cp{cp_size}_dp{dp_size}_rc{dp_rank_check}_pr{precision_recipe}"
    )
    ft_run_dir.mkdir(parents=True, exist_ok=True)
    ft_world_size = final_tp * pp_size * cp_size * dp_size
    cmd2 = (
        cmd1.rstrip()
        .replace(f"--nproc-per-node {world_size}", f"--nproc-per-node {ft_world_size}")
        .replace(f"--result-dir {run_dir}", f"--result-dir {ft_run_dir}")
        .replace(f"--tensor-model-parallel {tp_size}", f"--tensor-model-parallel {final_tp}")
    )
    cmd2 += f" --finetune-ckpt-dir {ckpt_dir} "
    cmd_parts_2 = shlex.split(cmd2)

    print("Starting Run 2 (resuming to step 10)...")
    result_2 = subprocess.run(cmd_parts_2, check=False, capture_output=True, text=True, cwd=run_dir, env=env)

    print(f"Run 2 Return code: {result_2.returncode}")
    if result_2.returncode != 0:
        print(f"Run 2 STDERR:\n{result_2.stderr}")

    assert result_2.returncode == 0, f"Run 2 failed with return code {result_2.returncode}"

    # 3. Load the new tb logs as before, and sanity check my recommendations as well as any others that make sense.
    ft_result_dir = ft_run_dir / "evo2"
    ft_tb_log_dir = ft_result_dir / "tb_logs"
    assert ft_tb_log_dir.exists(), "TensorBoard logs directory not found after Run 2"

    event_acc_2 = EventAccumulator(str(ft_tb_log_dir))
    event_acc_2.Reload()

    lm_loss_events_2 = event_acc_2.Scalars("lm loss")
    assert len(lm_loss_events_2) > 0, "No 'lm loss' events found in run 2"

    first_step_run2 = lm_loss_events_2[0].step
    first_step_run1 = lm_loss_events[0].step
    last_step_run2 = lm_loss_events_2[-1].step

    # Sanity checks:
    # 1. Resumption: Should start after step 5 (e.g., step 6)
    assert first_step_run2 == first_step_run1, (
        f"Run 2 FT steps should match run 1, but started at {first_step_run2} vs {first_step_run1}"
    )

    # 2. Completion: Should reach step 5 like run 1
    assert last_step_run2 == 5, f"Run 2 should reach step 5, but ended at {last_step_run2}"

    # 3. Loss Continuity check (basic): The first loss of run 2 should be reasonably close to the last loss of run 1,
    #    or at least not exploding, though optimization steps might cause fluctuations.
    first_loss_run1 = lm_loss_events[0].value
    first_loss_run2 = lm_loss_events_2[0].value
    last_loss_run1 = lm_loss_events[-1].value
    assert first_loss_run1 > last_loss_run1, (
        f"Run 1 first loss {first_loss_run1} is less than run 1 last loss {last_loss_run1}"
    )
    assert first_loss_run2 < first_loss_run1, (
        f"Run 2 first loss {first_loss_run2} is greater than run 1 first loss {first_loss_run1}"
    )
    assert abs(first_loss_run2 - first_loss_run1) > abs(last_loss_run1 - first_loss_run2), (
        f"Run 2 beginning {first_loss_run2} should be closer to end of run 1 {last_loss_run1} than beginning {first_loss_run1}."
    )
    assert first_loss_run2 - last_loss_run1 < 0.1, (
        f"Run 2 first loss {first_loss_run2} is not better than run 1 last loss {last_loss_run1} by no worse than 0.1"
    )


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_1m(tmp_path_factory) -> Path:
    """Module-scoped MBridge checkpoint for the 1b-8k-bf16 model.

    This fixture converts the NeMo2 checkpoint to MBridge format and exists for the duration of tests in this file.

    Returns:
        Path to the MBridge checkpoint iteration directory (e.g., .../iter_0000001)
    """
    try:
        nemo2_ckpt_path = bionemo_load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    output_dir = tmp_path_factory.mktemp("mbridge_checkpoint_7b_1m_module")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_ckpt_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=output_dir / "evo2_7b_1m_mbridge",
        model_size="evo2_7b",
        seq_length=1_048_576,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    # Return the parent directory (containing latest_train_state.pt), not the iter_0000001 subdirectory
    # The checkpoint loading code looks for tracker files in the parent directory
    return mbridge_ckpt_dir


@pytest.fixture(scope="module")
def base_checkpoint(tmp_path_factory: pytest.TempPathFactory, mbridge_checkpoint_7b_1m: Path) -> Path:
    """Create a base checkpoint by training one step with no parallelism."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Test requires at least 1 GPU")
    num_steps = 1
    tmp_path = tmp_path_factory.mktemp("base_checkpoint_module")
    base_path = tmp_path / "base_training"
    base_path.mkdir(parents=True, exist_ok=True)

    cmd = _distributed_training_cmd(
        path=base_path,
        max_steps=num_steps,
        val_check=num_steps,
        num_devices=1,
        dp=1,
        tp=1,
        cp=1,
        pp=1,
        finetune_ckpt_dir=mbridge_checkpoint_7b_1m,
    )
    _run_train_command(cmd, base_path)

    ckpt_dir = base_path / "evo2" / "checkpoints" / "iter_0000001"
    assert ckpt_dir.exists() and ckpt_dir.is_dir(), f"Checkpoint dir not found: {ckpt_dir}"
    return ckpt_dir


@pytest.mark.parametrize(
    "dp,cp,tp,pp",
    [
        pytest.param(2, 1, 1, 1, id="data_parallel"),
        pytest.param(1, 2, 1, 1, id="context_parallel"),
        pytest.param(1, 1, 2, 1, id="tensor_parallel"),
        pytest.param(1, 1, 1, 2, id="pipeline_parallel"),
    ],
)
@pytest.mark.timeout(900)
@pytest.mark.slow
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Test requires at least 2 GPUs")
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space limitations")
def test_distributed_training_gradient_equivalence(
    tmp_path: Path, base_checkpoint: Path, mbridge_checkpoint_7b_1m: Path, dp, cp, tp, pp
):
    """Test that optimizer states match across different distributed training strategies."""
    num_steps = 1
    num_devices = dp * cp * tp * pp
    assert num_devices == 2, (
        f"Test is designed for 2 GPUs but got {num_devices} for dp={dp}, cp={cp}, tp={tp}, pp={pp}"
    )

    parallel_path = tmp_path / f"parallel_dp{dp}_cp{cp}_tp{tp}_pp{pp}"
    parallel_path.mkdir(parents=True, exist_ok=True)
    cmd = _distributed_training_cmd(
        path=parallel_path,
        max_steps=num_steps,
        val_check=num_steps,
        num_devices=num_devices,
        dp=dp,
        tp=tp,
        cp=cp,
        pp=pp,
        finetune_ckpt_dir=mbridge_checkpoint_7b_1m,  # must use the same checkpoint since PP/TP will have different RNG
        additional_args=" --sequence-parallel " if tp > 1 else "",
    )
    _run_train_command(cmd, parallel_path)

    parallel_checkpoint = parallel_path / "evo2" / "checkpoints" / "iter_0000001"
    assert parallel_checkpoint.exists() and parallel_checkpoint.is_dir(), (
        f"Checkpoint dir not found: {parallel_checkpoint}"
    )

    checkpoint_dirs = [str(base_checkpoint), str(parallel_checkpoint)]
    assert_optimizer_states_match(checkpoint_dirs)
