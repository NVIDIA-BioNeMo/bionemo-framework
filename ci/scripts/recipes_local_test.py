#!/usr/bin/env python3

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

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from platformdirs import user_cache_dir


PIP_CACHE_DIR = user_cache_dir(appname="bionemo-pip-cache", appauthor="nvidia")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

DOCKER_RUN_ARGS = [
    "--rm",
    "--gpus",
    "all",
    "--ipc=host",
    "--ulimit",
    "memlock=-1",
    "--ulimit",
    "stack=67108864",
    "-v",
    f"{PIP_CACHE_DIR}:/workspace/.cache/pip",
]

CUSTOM_CONTAINERS = {
    "models/amplify": "svcbionemo023/bionemo-framework:amplify-model-devcontainer-082025",
}

# DEFAULT_CONTAINER = "nvcr.io/nvidia/pytorch:26.02-py3"

# This is a squashed version of the pytorch:26.02-py3 image, generated with
# docker-squash nvcr.io/nvidia/pytorch:26.02-py3 -t svcbionemo023/bionemo-framework:pytorch26.02-py3-squashed
# --output type=registry,compression=zstd,force-compression=true,oci-mediatypes=true,compression-level=15
# and pushed to the dockerhub registry. Our github actions are able to cache image pulls from dockerhub but not nvcr, so
# hopefully this cuts down slightly on CI time at the expense of having a slightly in-directed image location.
DEFAULT_CONTAINER = "svcbionemo023/bionemo-framework:pytorch26.02-py3-squashed"


def get_git_root() -> str:
    """Get the git repository root directory."""
    cmd = ["git", "rev-parse", "--show-toplevel"]
    logger.debug(f"Running command: {' '.join(cmd)}")
    git_root = subprocess.check_output(cmd, text=True).strip()
    (Path(git_root) / ".cache" / "pip").mkdir(parents=True, exist_ok=True)
    return git_root


def get_test_directories(input_dirs: Optional[List[str]] = None) -> List[str]:
    """Get directories to test.

    Returns list of (directory_path, docker_image) tuples.
    If input_dirs is None, scans all subdirectories under models/ and recipes/.
    """
    git_root = get_git_root()

    # Scan models/ and recipes/ directories
    directories = []
    for base_dir in ["models", "recipes"]:
        base_path = Path(git_root) / base_dir
        if base_path.exists():
            for subdir in base_path.iterdir():
                if subdir.is_dir():
                    directories.append(str(subdir))

    if input_dirs:
        absolute_input_dirs = [os.path.abspath(d) for d in input_dirs]
        directories = list(set(directories) & set(absolute_input_dirs))
        assert set(absolute_input_dirs).issubset(set(directories)), (
            f"Input directory {set(input_dirs) - set(directories)} not found"
        )

    return directories


def run_tests_in_docker(work_dir: str) -> bool:
    """Run dependency installation and tests in a single docker run command."""
    git_root = get_git_root()

    # Create bash script that installs dependencies and runs tests
    install_and_test_script = textwrap.dedent("""
        set -e  # Exit on any error

        # Ensure image-embedded constraints do not leak into local recipe installs
        unset PIP_CONSTRAINT || true

        echo "Checking for dependency files..."
        # Install dependencies based on available files
        if [ -f pyproject.toml ] || [ -f setup.py ]; then
            echo "Installing package in editable mode..."
            PIP_CACHE_DIR=/workspace/.cache/pip pip install -e .
            echo "Installed package as editable package"
        elif [ -f requirements.txt ]; then
            echo "Installing from requirements.txt..."
            PIP_CACHE_DIR=/workspace/.cache/pip pip install -r requirements.txt
            echo "Installed from requirements.txt"
        else
            echo "No pyproject.toml, setup.py, or requirements.txt found"
            exit 1
        fi

        echo "Running tests..."
        python -m pytest -v .
        """)

    relative_path = Path(work_dir).relative_to(git_root).as_posix()
    if relative_path in CUSTOM_CONTAINERS:
        image = CUSTOM_CONTAINERS[relative_path]
    else:
        image = DEFAULT_CONTAINER

    logger.info(f"Running tests in {work_dir} with image {image}")

    # Build docker run command
    docker_cmd = (
        [
            "docker",
            "run",
        ]
        + DOCKER_RUN_ARGS
        + [
            "-v",
            f"{git_root}:/workspace",
            "-w",
            f"/workspace/{Path(work_dir).relative_to(git_root)}",
            image,
            "bash",
            "-c",
            install_and_test_script,
        ]
    )

    logger.debug(f"Running command: {' '.join(docker_cmd)}")
    result = subprocess.run(docker_cmd, text=True)

    success = result.returncode == 0
    if success:
        logger.info("Tests passed!")
    else:
        logger.info("Tests failed!")

    return success


def main():
    """Main function to run tests for all specified directories."""
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Run docker-based unit tests for models/ and recipes/")
    parser.add_argument("directories", nargs="*", help="Directories to test (default: all under models/ and recipes/)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Caching pip installations to: {PIP_CACHE_DIR}")

    # Get directories to test
    test_dirs = get_test_directories(args.directories)

    if not test_dirs:
        logger.info("No directories found to test")
        return

    logger.info(f"Found {len(test_dirs)} directories to test:")

    failed_tests = []

    # Run tests for each directory
    for dir_path in test_dirs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing: {dir_path}")
        logger.info(f"{'=' * 60}")

        # Run tests in single docker command
        success = run_tests_in_docker(dir_path)

        if not success:
            failed_tests.append(dir_path)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total directories tested: {len(test_dirs)}")
    logger.info(f"Passed: {len(test_dirs) - len(failed_tests)}")
    logger.info(f"Failed: {len(failed_tests)}")

    if failed_tests:
        logger.info("\nFailed tests:")
        for failed_dir in failed_tests:
            logger.info(f"  {failed_dir}")
        sys.exit(1)
    else:
        logger.info("\nAll tests passed!")


if __name__ == "__main__":
    main()
