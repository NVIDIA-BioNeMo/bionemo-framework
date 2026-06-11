#!/bin/bash
#
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


# Enable strict mode with better error handling
set -euox pipefail

# Function to display usage information
usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Options:
    --skip-docs         Skip running tests in the docs directory
    --no-nbval          Skip jupyter notebook validation tests
    --skip-slow         Skip tests marked as slow (@pytest.mark.slow)
    --only-slow         Only run tests marked as slow (@pytest.mark.slow)
    --allow-no-tests    Allow sub-packages with no found tests (for example no slow tests if --only-slow is set)
    --ignore-files      Skip files from tests using glob patterns (comma-separated, no spaces).
                            Example: --ignore-files docs/*.ipynb,src/specific_test.py

Note: Documentation tests (docs/) are only run when notebook validation
      is enabled (--no-nbval not set) and docs are not skipped
      (--skip-docs not set)
    -h, --help     Display this help message
EOF
    exit "${1:-0}"
}

# Set default environment variables
: "${BIONEMO_DATA_SOURCE:=pbss}"
: "${PYTHONDONTWRITEBYTECODE:=1}"
: "${PYTORCH_CUDA_ALLOC_CONF:=expandable_segments:True}"

# Export necessary environment variables
export BIONEMO_DATA_SOURCE PYTHONDONTWRITEBYTECODE PYTORCH_CUDA_ALLOC_CONF

# Initialize variables
declare -a coverage_files
SKIP_DOCS=false
NO_NBVAL=false
SKIP_SLOW=false
ONLY_SLOW=false
ALLOW_NO_TESTS=false
# TODO(@cspades): Ignore this Evo2 notebook test, which has a tendency to leave a 32GB orphaned process in GPU.
declare -a IGNORE_FILES=()
error=false

# Parse command line arguments
while (( $# > 0 )); do
    case "$1" in
        --skip-docs) SKIP_DOCS=true ;;
        --no-nbval) NO_NBVAL=true ;;
        --skip-slow) SKIP_SLOW=true ;;
        --only-slow) ONLY_SLOW=true ;;
        --allow-no-tests) ALLOW_NO_TESTS=true ;;
        --ignore-files)
            shift
            IFS=',' read -ra IGNORE_FILES <<< "$1"
            ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage 1 ;;
    esac
    shift
done

# Echo some useful information
lscpu
nvidia-smi
uname -a

# Set up pytest options
PYTEST_OPTIONS=(
    -v
    --cov=bionemo
    --cov-append
    --cov-report=xml:coverage.xml
)
# Add multiple file ignores if specified
for ignore_file in "${IGNORE_FILES[@]}"; do
    PYTEST_OPTIONS+=(--ignore-glob="$ignore_file")
done
[[ "$NO_NBVAL" != true ]] && PYTEST_OPTIONS+=(--nbval-lax)
[[ "$SKIP_SLOW" == true ]] && PYTEST_OPTIONS+=(-m "not slow")
[[ "$ONLY_SLOW" == true ]] && PYTEST_OPTIONS+=(-m "slow")

# Define test directories
TEST_DIRS=(./sub-packages/bionemo-*/)
if [[ "$NO_NBVAL" != true && "$SKIP_DOCS" != true ]]; then
    TEST_DIRS+=(docs/)
fi

echo "Test directories: ${TEST_DIRS[*]}"

clean_pycache() {
    # Use the provided base directory or default to current directory
    local base_dir="${1:-.}"
    echo "Cleaning Python cache files in $base_dir..."
    find "$base_dir" -regex '^.*\(__pycache__\|\.py[co]\)$' -delete
}

# Run tests with coverage
for dir in "${TEST_DIRS[@]}"; do
    echo "Running pytest in $dir"
    # Run pytest but don't exit on failure - we'll handle the exit code separately. This is needed because our script is
    #  running in pipefail mode and pytest will exit with a non-zero exit code if it finds no tests.
    { pytest "${PYTEST_OPTIONS[@]}" --junitxml=$(basename $dir).junit.xml -o junit_family=legacy "$dir"; exit_code=$?; } || true

    if [[ $exit_code -ne 0 ]]; then
        if [[ "$ALLOW_NO_TESTS" == true && $exit_code -eq 5 ]]; then
            # Exit code 5 means no tests found, which is allowed if --allow-no-tests is set
            echo "No tests found in $dir (exit code $exit_code) - continuing as --allow-no-tests is set"
        else
            echo "Error: pytest failed with exit code $exit_code"
            error=true
        fi
    fi

    # Avoid duplicated pytest cache filenames.
    clean_pycache "$dir"
done

# Exit with appropriate status
$error && exit 1
exit 0
