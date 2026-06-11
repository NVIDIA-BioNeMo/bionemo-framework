# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import train


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Test requires a GPU",
)

requires_multi_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Test requires at least 2 GPUs",
)


def extract_final_train_loss(output_text: str) -> float:
    """Parse the training output to extract the final train_loss value.

    Args:
        output_text: Combined stdout and stderr from training process

    Returns:
        Final train_loss value as float

    Raises:
        ValueError: If no train_loss found or parsing fails
    """
    # Look for dictionary-like patterns containing train_loss
    # Pattern matches: {'key': value, 'train_loss': value, ...}
    pattern = r'\{[^{}]*[\'"]train_loss[\'"]:\s*([0-9.]+)[^{}]*\}'

    matches = re.findall(pattern, output_text)

    if not matches:
        # Fallback: try to find train_loss in any context
        simple_pattern = r'[\'"]train_loss[\'"]:\s*([0-9.]+)'
        matches = re.findall(simple_pattern, output_text)

    if not matches:
        raise ValueError("No train_loss found in training output")

    # Return the last (final) train_loss value found
    final_train_loss = float(matches[-1])
    return final_train_loss


def launch_accelerate(
    accelerate_config: str,
    tmp_path: Path,
    num_processes: int,
    hydra_config_name: str = "L0_sanity",
    *overrides: str,
) -> float:
    """Test that accelerate launch runs successfully and returns the final train_loss."""
    # Find the recipe directory and train.py
    recipe_dir = Path(train.__file__).parent
    train_py = Path(train.__file__)
    accelerate_config_path = recipe_dir / "accelerate_config" / accelerate_config

    assert train_py.exists(), f"train.py not found at {train_py}"
    assert accelerate_config_path.exists(), f"deepspeed_config.yaml not found at {accelerate_config_path}"

    # Run 'accelerate launch train.py' as a subprocess
    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(accelerate_config_path),
        "--num_processes",
        f"{num_processes}",
        "--main_process_port",
        f"{random.randint(20000, 40000)}",
        str(train_py),
        "--config-name",
        str(hydra_config_name),
        f"trainer.output_dir={tmp_path}",
        f"hydra.run.dir={tmp_path}/outputs",
        "trainer.do_eval=False",
        *overrides,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=240,
    )

    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        pytest.fail(f"Command:\n{' '.join(cmd)}\nfailed with exit code {result.returncode}")

    # Parse the training output to check final train_loss
    combined_output = result.stdout + result.stderr

    return extract_final_train_loss(combined_output)
