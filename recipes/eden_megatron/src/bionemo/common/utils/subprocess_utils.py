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

# --- BEGIN COPIED FILE NOTICE ---
# This file is copied from: recipes/evo2_megatron/src/bionemo/common/utils/subprocess_utils.py
# Do not modify this file directly. Instead, modify the source and run:
#     python ci/scripts/check_copied_files.py --fix
# --- END COPIED FILE NOTICE ---


import logging
import shlex
import subprocess
from typing import Any, Dict


logger = logging.getLogger(__name__)


def run_subprocess_safely(command: str, timeout: int = 2000) -> Dict[str, Any]:
    """Run a subprocess and raise an error if it fails.

    Args:
        command: The command to run.
        timeout: The timeout for the command.

    Returns:
        The result of the subprocess.
    """
    try:
        # Use Popen to enable real-time output while still capturing it
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        stdout_lines = []
        stderr_lines = []

        # Read output in real-time
        import select
        import sys

        while True:
            # Use select to check for available output (Unix/Linux/Mac only)
            if hasattr(select, "select"):
                ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

                if process.stdout in ready:
                    line = process.stdout.readline()
                    if line:
                        stdout_lines.append(line)
                        print(line.rstrip(), file=sys.stdout, flush=True)

                if process.stderr in ready:
                    line = process.stderr.readline()
                    if line:
                        stderr_lines.append(line)
                        print(line.rstrip(), file=sys.stderr, flush=True)
            else:
                # Fallback for Windows - read with timeout
                try:
                    stdout_data, stderr_data = process.communicate(timeout=0.1)
                    if stdout_data:
                        stdout_lines.extend(stdout_data.splitlines(keepends=True))
                        print(stdout_data.rstrip(), file=sys.stdout, flush=True)
                    if stderr_data:
                        stderr_lines.extend(stderr_data.splitlines(keepends=True))
                        print(stderr_data.rstrip(), file=sys.stderr, flush=True)
                    break
                except subprocess.TimeoutExpired:
                    pass

            # Check if process has finished
            if process.poll() is not None:
                # Read any remaining output
                remaining_stdout, remaining_stderr = process.communicate()
                if remaining_stdout:
                    stdout_lines.extend(remaining_stdout.splitlines(keepends=True))
                    print(remaining_stdout.rstrip(), file=sys.stdout, flush=True)
                if remaining_stderr:
                    stderr_lines.extend(remaining_stderr.splitlines(keepends=True))
                    print(remaining_stderr.rstrip(), file=sys.stderr, flush=True)
                break

        # Check for timeout
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            raise

        # Check return code
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, command, output="".join(stdout_lines), stderr="".join(stderr_lines)
            )

        # Create result object similar to subprocess.run
        class Result:
            def __init__(self, stdout, stderr, returncode):
                self.stdout = stdout
                self.stderr = stderr
                self.returncode = returncode

        result = Result("".join(stdout_lines), "".join(stderr_lines), process.returncode)
        return {"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode}
    except subprocess.TimeoutExpired as e:
        logger.error(f"Command timed out. Command: {command}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}")
        return {"error": "timeout", "stdout": e.stdout, "stderr": e.stderr, "returncode": None}

    except subprocess.CalledProcessError as e:
        logger.error(
            f"Command failed. Command: {command}\nreturncode: {e.returncode}\nstdout:\n{e.stdout}\nstderr:\n{e.stderr}"
        )
        return {"error": "non-zero exit", "stdout": e.stdout, "stderr": e.stderr, "returncode": e.returncode}

    except FileNotFoundError as e:
        logger.error(f"Command not found. Command: {command}\nstderr:\n{e!s}")
        return {"error": "not found", "stdout": "", "stderr": str(e), "returncode": None}

    except Exception as e:
        # catch-all for other unexpected errors
        return {"error": "other", "message": str(e), "stdout": "", "stderr": "", "returncode": None}
