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
import subprocess
import sys
import time

from lightning.fabric.plugins.environments.lightning import find_free_network_port


def run_command_with_timeout(command, path, env, timeout=3600):
    """Run command with timeout and incremental output processing to prevent hanging."""
    # Start process without capturing output in the main process
    process = subprocess.Popen(
        command,
        shell=True,
        cwd=path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # Line buffered
    )

    stdout_data = []
    stderr_data = []
    start_time = time.time()

    try:
        # Use select to handle output in a non-blocking way
        import select

        # Get file descriptors for stdout and stderr
        stdout_fd = process.stdout.fileno()
        stderr_fd = process.stderr.fileno()

        # Set up select lists
        read_fds = [stdout_fd, stderr_fd]

        # Process output incrementally
        while read_fds and process.poll() is None:
            # Check for timeout
            if timeout and time.time() - start_time > timeout:
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None:
                    process.kill()
                raise subprocess.TimeoutExpired(command, timeout)

            # Wait for output with a short timeout to allow checking process status
            ready_fds, _, _ = select.select(read_fds, [], [], 1.0)

            for fd in ready_fds:
                if fd == stdout_fd:
                    line = process.stdout.readline()
                    if not line:
                        read_fds.remove(stdout_fd)
                        continue
                    stdout_data.append(line)
                    # Optionally process/print output incrementally
                    # print(f"STDOUT: {line.strip()}")

                if fd == stderr_fd:
                    line = process.stderr.readline()
                    if not line:
                        read_fds.remove(stderr_fd)
                        continue
                    stderr_data.append(line)
                    # Optionally process/print error output incrementally
                    # print(f"STDERR: {line.strip()}")

        # Get any remaining output
        remaining_stdout, remaining_stderr = process.communicate()
        if remaining_stdout:
            stdout_data.append(remaining_stdout)
        if remaining_stderr:
            stderr_data.append(remaining_stderr)

        # Create result object similar to subprocess.run
        result = subprocess.CompletedProcess(
            args=command, returncode=process.returncode, stdout="".join(stdout_data), stderr="".join(stderr_data)
        )
        return result

    except Exception as e:
        # Make sure we don't leave zombie processes
        if process.poll() is None:
            process.terminate()
            time.sleep(0.5)
            if process.poll() is None:
                process.kill()
        raise e


def run_command_in_subprocess(command: str, path: str, timeout: int = 3600) -> str:
    """Run a command in a subprocess and return the output."""
    open_port = find_free_network_port()
    # a local copy of the environment
    env = dict(**os.environ)
    env["MASTER_PORT"] = str(open_port)

    result = run_command_with_timeout(
        command=command,
        path=path,
        env=env,
        timeout=timeout,  # Set an appropriate timeout in seconds
    )

    # For debugging purposes, print the output if the test fails.
    if result.returncode != 0:
        sys.stderr.write("STDOUT:\n" + result.stdout + "\n")
        sys.stderr.write("STDERR:\n" + result.stderr + "\n")

    assert result.returncode == 0, f"Command failed: {command}"

    return result.stdout
