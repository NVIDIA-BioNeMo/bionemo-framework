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

import os
import shlex
import subprocess
from pathlib import Path

import pytest
from lightning.fabric.plugins.environments.lightning import find_free_network_port

from bionemo.core.data.load import load


@pytest.fixture
def data_path() -> Path:
    """Gets the path to the directory with with cellx small dataset in Single Cell Memmap format.
    Returns:
        A Path object that is the directory with the specified test data.
    """
    return load("single_cell/testdata-20241203") / "cellxgene_2023-12-15_small_processed_scdl"


def test_bionemo2_rootdir(data_path):
    data_error_str = (
        "Please download test data with:\n"
        "`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    )
    assert data_path.exists(), f"Could not find test data directory.\n{data_error_str}"
    assert data_path.is_dir(), f"Test data directory is supposed to be a directory.\n{data_error_str}"


# TODO: These tests currently take an inordinate amount of time. See https://jirasw.nvidia.com/browse/BIONEMO-553
@pytest.mark.slow
def test_pretrain_cli_from_ckpt(tmpdir, data_path):
    # Same as test_pretrain, but includes a checkpoint to initialize from.
    result_dir = Path(tmpdir.mkdir("results"))

    open_port = find_free_network_port()
    # NOTE: if this test is ever failing, you may want to put the config somewhere easily accessible.
    config = f"{result_dir}/test_config.yaml"
    # Invoke with blocking, continue when finished (and the yaml config is generated)
    checkpoint_path: Path = load("geneformer/10M_240530:2.0")
    cmd_str = f"""bionemo-geneformer-recipe --dest {config} --recipe geneformer_tiny_test_recipe --data-path {data_path} --result-dir {result_dir} --initial-ckpt-path {checkpoint_path}""".strip()
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain recipe failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    cmd_str = f"""bionemo-geneformer-train --conf {config}""".strip()
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        # More helpful failure
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    # Must match the experiment directory configured.
    assert (result_dir / "test-experiment").exists(), "Could not find test experiment directory."


# TODO: These tests currently take an inordinate amount of time. See https://jirasw.nvidia.com/browse/BIONEMO-553
@pytest.mark.slow
def test_pretrain_cli(tmpdir, data_path):
    """trains from scratch"""
    # data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"
    result_dir = Path(tmpdir.mkdir("results"))

    open_port = find_free_network_port()
    config = f"{result_dir}/test_config.yaml"
    # Invoke with blocking
    cmd_str = f"""bionemo-geneformer-recipe --dest {config} --recipe geneformer_tiny_test_recipe --data-path {data_path} --result-dir {result_dir}""".strip()
    # continue when finished
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    # Now do pretrain
    if result.returncode != 0:
        raise Exception(f"Pretrain recipe failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    cmd_str = f"""bionemo-geneformer-train --conf {config}""".strip()
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")
    # NOTE this looks a lot like a magic value. But we also could do yaml.loads(config)['experiment_config']['experiment_name']
    assert (result_dir / "test-experiment").exists(), "Could not find test experiment directory."


# TODO: These tests currently take an inordinate amount of time. See https://jirasw.nvidia.com/browse/BIONEMO-553
@pytest.mark.slow
def test_finetune_cli(tmpdir, data_path):
    """Uses CLI to invoke the entrypoint"""
    # data_path: Path = load("single_cell/testdata-20240506") / "cellxgene_2023-12-15_small" / "processed_data"
    result_dir = Path(tmpdir.mkdir("results"))
    checkpoint_path: Path = load("geneformer/10M_240530:2.0")

    open_port = find_free_network_port()

    config = f"{result_dir}/test_config.yaml"

    # TODO add initial path
    cmd_str = f"""bionemo-geneformer-recipe --dest {config} --recipe finetune_test_recipe --data-path {data_path} --result-dir {result_dir} --initial-ckpt-path {checkpoint_path}""".strip()
    # continue when finished
    env = dict(**os.environ)  # a local copy of the environment
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    import sys

    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        capture_output=True,
    )
    # Now do pretrain
    if result.returncode != 0:
        raise Exception(f"Pretrain recipe failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")

    cmd_str = f"bionemo-geneformer-train --conf {config} --model-config-cls ExposedFineTuneSeqLenBioBertConfig"
    env = dict(**os.environ)  # a local copy of the environment
    open_port = find_free_network_port()
    env["MASTER_PORT"] = str(open_port)
    cmd = shlex.split(cmd_str)
    result = subprocess.run(
        cmd,
        cwd=tmpdir,
        env=env,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        raise Exception(f"Pretrain script failed:\n{cmd_str=}\n{result.stdout=}\n{result.stderr=}")
    assert (result_dir / "test-experiment").exists(), "Could not find test experiment directory."
