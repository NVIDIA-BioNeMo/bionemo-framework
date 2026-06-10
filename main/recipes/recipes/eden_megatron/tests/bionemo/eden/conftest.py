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


# conftest.py
import copy
import gc
import os
import shlex
import socket
import subprocess
from pathlib import Path

import pytest
import torch


_REPO_BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_HF_TOKENIZER_MODEL_PATH = str(_REPO_BASE_DIR / "tokenizers" / "nucleotide_fast_tokenizer_256")


def get_device_and_memory_allocated() -> str:
    """Get the current device index, name, and memory usage."""
    current_device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(current_device_index)
    message = f"""
        current device index: {current_device_index}
        current device uuid: {props.uuid}
        current device name: {props.name}
        memory, total on device: {torch.cuda.mem_get_info()[1] / 1024**3:.3f} GB
        memory, available on device: {torch.cuda.mem_get_info()[0] / 1024**3:.3f} GB
        memory allocated for tensors etc: {torch.cuda.memory_allocated() / 1024**3:.3f} GB
        max memory reserved for tensors etc: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB
        """
    return message


def pytest_sessionstart(session):
    """Called at the start of the test session."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(
            f"""
            bionemo-recipes/recipes/eden_megatron/tests/bionemo/eden: Starting test session
            {get_device_and_memory_allocated()}
            """
        )


def pytest_sessionfinish(session, exitstatus):
    """Called at the end of the test session."""
    if torch.cuda.is_available():
        print(
            f"""
            bionemo-recipes/recipes/eden_megatron/tests/bionemo/eden: Test session complete
            {get_device_and_memory_allocated()}
            """
        )


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up GPU memory, reset state, and restore env vars after each test.

    Megatron Core's LanguageModule._set_attention_backend() mutates os.environ
    (e.g. NVTE_FUSED_ATTN, NVTE_FLASH_ATTN) when models are constructed in-process.
    Capturing and restoring the full environment prevents cross-test pollution.
    """
    saved_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(saved_environ)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


def pytest_addoption(parser: pytest.Parser):
    """Pytest configuration for bionemo.eden.run tests. Adds custom command line options for dataset paths."""
    parser.addoption("--dataset-dir", action="store", default=None, help="Path to preprocessed dataset directory")
    parser.addoption("--training-config", action="store", default=None, help="Path to training data config YAML file")


# =============================================================================
# Session-scoped checkpoint fixtures for Eden models
# =============================================================================


@pytest.fixture(scope="session")
def mbridge_eden_checkpoint(tmp_path_factory) -> Path:
    """Session-scoped MBridge checkpoint for a tiny Eden (Llama) model.

    Creates an Eden mbridge checkpoint by training a tiny model (2 layers, seq_length=64)
    for 2 steps with mock data. Shared across test files that need an Eden checkpoint.

    Returns:
        Path to the MBridge checkpoint directory (parent of iter_0000002).
    """

    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return s.getsockname()[1]

    tmp_dir = tmp_path_factory.mktemp("eden_ckpt_session")
    run_dir = tmp_dir / "eden_train"
    run_dir.mkdir(parents=True, exist_ok=True)
    port = _find_free_port()

    cmd = (
        f"torchrun --nproc-per-node 1 --no-python --master_port {port} "
        f"train_eden "
        f"--hf-tokenizer-model-path {DEFAULT_HF_TOKENIZER_MODEL_PATH} "
        "--model-size eden_7b --num-layers 2 "
        "--max-steps 2 --eval-interval 2 --eval-iters 1 "
        f"--mock-data --result-dir {run_dir} "
        "--micro-batch-size 4 --global-batch-size 4 --seq-length 64 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--mixed-precision-recipe bf16_mixed "
        "--no-activation-checkpointing "
        "--decay-steps 1000 --warmup-steps 10 "
        "--log-interval 1 --seed 41 --dataset-seed 33"
    )

    env = copy.deepcopy(os.environ)
    result = subprocess.run(shlex.split(cmd), check=False, capture_output=True, text=True, cwd=run_dir, env=env)
    if result.returncode != 0:
        print(f"Eden checkpoint creation STDOUT:\n{result.stdout}")
        print(f"Eden checkpoint creation STDERR:\n{result.stderr}")
    assert result.returncode == 0, f"Eden checkpoint creation failed: {result.stderr[-2000:]}"

    ckpt_dir = run_dir / "eden" / "checkpoints"
    iter_dir = ckpt_dir / "iter_0000002"
    assert iter_dir.exists(), f"Eden checkpoint not found at {iter_dir}"
    return iter_dir


@pytest.fixture(scope="module")
def mbridge_checkpoint_path(mbridge_eden_checkpoint) -> Path:
    """Module-scoped alias for the session-scoped Eden checkpoint.

    Provides backward compatibility for tests that use the name 'mbridge_checkpoint_path'.

    Returns:
        Path to the MBridge checkpoint iteration directory
    """
    return mbridge_eden_checkpoint
