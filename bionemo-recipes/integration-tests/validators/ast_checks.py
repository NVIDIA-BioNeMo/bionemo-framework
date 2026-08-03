# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""AST-based validation of generated Python files."""

import ast
from pathlib import Path


def validate_python_syntax(code: str) -> bool:
    """Check that a string of Python code parses without syntax errors."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def validate_python_file(filepath: Path) -> None:
    """Assert that a Python file has valid syntax."""
    assert filepath.exists(), f"File does not exist: {filepath}"
    code = filepath.read_text()
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise AssertionError(f"Syntax error in {filepath}: {e}") from e


def has_class(filepath: Path, class_name: str) -> bool:
    """Check if a Python file defines a class with the given name."""
    code = filepath.read_text()
    tree = ast.parse(code)
    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))


def has_function(filepath: Path, func_name: str) -> bool:
    """Check if a Python file defines a function with the given name."""
    code = filepath.read_text()
    tree = ast.parse(code)
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name for node in ast.walk(tree)
    )


def count_classes(filepath: Path) -> int:
    """Count the number of class definitions in a Python file."""
    code = filepath.read_text()
    tree = ast.parse(code)
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
