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


import pytest
import toml

from bionemo.fw.dependency_graph import (
    build_dependency_graph,
    find_bionemo_subpackages,
    parse_dependencies,
    parse_tach_toml,
    resolve_dependencies,
    visualize_dependency_graph,
)


@pytest.fixture
def temp_project_structure(tmp_path):
    """Creates a temporary directory structure with pyproject.toml files."""
    subpackage1 = tmp_path / "bionemo-subpackage1"
    subpackage2 = tmp_path / "bionemo-subpackage2"
    subpackage1.mkdir()
    subpackage2.mkdir()

    pyproject_data1 = {"project": {"name": "bionemo-subpackage1", "dependencies": ["bionemo-core", "bionemo-utils"]}}
    with open(subpackage1 / "pyproject.toml", "w") as f:
        toml.dump(pyproject_data1, f)

    pyproject_data2 = {"project": {"name": "bionemo-subpackage2", "dependencies": ["bionemo-subpackage1"]}}
    with open(subpackage2 / "pyproject.toml", "w") as f:
        toml.dump(pyproject_data2, f)

    yield tmp_path  # Provide the base directory path


def test_parse_dependencies_list_format(tmp_path):
    """Test parsing dependencies when dependencies are in a list format."""
    pyproject_data = {"project": {"name": "bionemo-example", "dependencies": ["bionemo-core", "bionemo-utils"]}}
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text(toml.dumps(pyproject_data))
    package_name, dependencies = parse_dependencies(pyproject_toml)

    assert package_name == "bionemo-example"
    assert dependencies == {"bionemo-core": "unpinned", "bionemo-utils": "unpinned"}


def test_parse_dependencies_dict_format(tmp_path):
    """Test parsing dependencies when dependencies are in a dictionary format."""
    pyproject_data = {
        "project": {"name": "bionemo-example", "dependencies": {"bionemo-core": "1.0.0", "bionemo-utils": "2.0.0"}}
    }
    pyproject_toml = tmp_path / "pyproject.toml"

    pyproject_toml.write_text(toml.dumps(pyproject_data))
    package_name, dependencies = parse_dependencies(pyproject_toml)

    assert package_name == "bionemo-example"
    assert dependencies == {"bionemo-core": "1.0.0", "bionemo-utils": "2.0.0"}


def test_parse_dependencies_missing_sections(tmp_path):
    """Test handling missing project name and dependencies."""
    pyproject_data = {}
    pyproject_toml = tmp_path / "pyproject.toml"

    pyproject_toml.write_text(toml.dumps(pyproject_data))

    package_name, dependencies = parse_dependencies(pyproject_toml)

    assert package_name is None
    assert dependencies == {}


def test_build_dependency_graph(temp_project_structure):
    """Test building the dependency graph."""
    directories = ["bionemo-subpackage1", "bionemo-subpackage2"]
    dependency_graph = build_dependency_graph(temp_project_structure, directories)

    # Debugging: Print the graph if it's empty
    if not dependency_graph:
        print("DEBUG: Dependency graph is empty. Check pyproject.toml paths.")
    assert "bionemo-subpackage1" in dependency_graph
    assert "bionemo-subpackage2" in dependency_graph
    assert dependency_graph["bionemo-subpackage1"] == {"bionemo-core": "unpinned", "bionemo-utils": "unpinned"}
    assert dependency_graph["bionemo-subpackage2"] == {"bionemo-subpackage1": "unpinned"}


def test_resolve_dependencies():
    """Test resolving transitive dependencies."""
    toml_imports = {
        "bionemo-subpackage1": ["bionemo-core", "bionemo-utils"],
        "bionemo-subpackage2": ["bionemo-subpackage1"],
    }

    resolved = resolve_dependencies("bionemo-subpackage2", toml_imports)
    assert resolved == {"bionemo-subpackage1", "bionemo-core", "bionemo-utils"}


def test_parse_tach_toml(tmp_path):
    """Test parsing a tach.toml file."""
    pyproject_data = {
        "modules": [
            {"path": "bionemo.core", "depends_on": ["bionemo.utils"]},
            {"path": "bionemo.utils", "depends_on": []},
        ]
    }
    pyproject_toml = tmp_path / "pyproject.toml"
    pyproject_toml.write_text(toml.dumps(pyproject_data))

    tach_toml_dependencies = parse_tach_toml(pyproject_toml)

    assert tach_toml_dependencies == {
        "bionemo-core": ["bionemo-utils"],
        "bionemo-utils": [],
    }


def test_visualize_dependency_graph(tmp_path):
    """Test visualization of the dependency graph (ensuring no exceptions)."""
    dependency_graph = {
        "bionemo-subpackage1": {"bionemo-core": "unpinned", "bionemo-utils": "unpinned"},
        "bionemo-subpackage2": {"bionemo-subpackage1": "unpinned"},
    }
    visualize_dependency_graph(dependency_graph, tmp_path / "output.png")
    assert (tmp_path / "output.png").exists()


def test_find_bionemo_subpackages(temp_project_structure):
    """Test finding bionemo subpackages in Python files."""
    subpackage_src = temp_project_structure / "bionemo-subpackage1" / "src"
    subpackage_src.mkdir(parents=True, exist_ok=True)

    # Create a Python file with some bionemo imports
    python_file = subpackage_src / "example.py"
    with open(python_file, "w") as f:
        f.write("import bionemo.core\nfrom bionemo.utils import some_function\nimport bionemo.experiment\n")

    directories = ["bionemo-subpackage1"]
    found_imports = find_bionemo_subpackages(temp_project_structure, directories)

    assert "bionemo-subpackage1" in found_imports
    assert found_imports["bionemo-subpackage1"] == {"bionemo-core", "bionemo-utils", "bionemo-experiment"}


def test_find_bionemo_subpackages_multiple_files(temp_project_structure):
    """Test `find_bionemo_subpackages` when scanning multiple files."""
    subpackage_src = temp_project_structure / "bionemo-subpackage1" / "src"
    subpackage_src.mkdir(parents=True, exist_ok=True)

    # Create multiple Python files
    python_file1 = subpackage_src / "file1.py"
    python_file2 = subpackage_src / "file2.py"

    with open(python_file1, "w") as f:
        f.write("import bionemo.data\n")

    with open(python_file2, "w") as f:
        f.write("from bionemo.visualization import plot_graph\n")

    directories = ["bionemo-subpackage1"]
    found_imports = find_bionemo_subpackages(temp_project_structure, directories)

    assert "bionemo-subpackage1" in found_imports
    assert found_imports["bionemo-subpackage1"] == {"bionemo-data", "bionemo-visualization"}


def test_find_bionemo_subpackages_no_imports(temp_project_structure):
    """Test `find_bionemo_subpackages` when there are no bionemo imports."""
    subpackage_src = temp_project_structure / "bionemo-subpackage1" / "src"
    subpackage_src.mkdir(parents=True, exist_ok=True)

    # Create a Python file with NO bionemo imports
    python_file = subpackage_src / "empty.py"
    with open(python_file, "w") as f:
        f.write("print('Hello, world!')\n")

    directories = ["bionemo-subpackage1"]
    found_imports = find_bionemo_subpackages(temp_project_structure, directories)

    assert "bionemo-subpackage1" in found_imports
    assert found_imports["bionemo-subpackage1"] == set()  # No bionemo imports found


def test_find_bionemo_subpackages_nested_imports(temp_project_structure):
    """Test `find_bionemo_subpackages` when imports are in nested directories."""
    subpackage_src = temp_project_structure / "bionemo-subpackage1" / "src" / "nested"
    subpackage_src.mkdir(parents=True, exist_ok=True)

    # Create a Python file inside a nested directory
    python_file = subpackage_src / "nested_example.py"
    with open(python_file, "w") as f:
        f.write("import bionemo.models\n")

    directories = ["bionemo-subpackage1"]
    found_imports = find_bionemo_subpackages(temp_project_structure, directories)

    assert "bionemo-subpackage1" in found_imports
    assert found_imports["bionemo-subpackage1"] == {"bionemo-models"}


def test_find_bionemo_subpackages_syntax_error(temp_project_structure):
    """Test `find_bionemo_subpackages` when encountering a syntax error in a file."""
    subpackage_src = temp_project_structure / "bionemo-subpackage1" / "src"
    subpackage_src.mkdir(parents=True, exist_ok=True)

    # Create a Python file with a syntax error
    python_file = subpackage_src / "syntax_error.py"
    with open(python_file, "w") as f:
        f.write("import bionemo.analysis\nSyntax Error Here!!\n")

    directories = ["bionemo-subpackage1"]
    found_imports = find_bionemo_subpackages(temp_project_structure, directories)

    assert "bionemo-subpackage1" in found_imports
    assert found_imports["bionemo-subpackage1"] == {"bionemo-analysis"}  # Syntax error shouldn't break parsing
