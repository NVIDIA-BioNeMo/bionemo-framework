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


import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import toml


def parse_dependencies(pyproject_path):
    """Parse dependencies from a pyproject.toml file."""
    with open(pyproject_path, "r") as f:
        pyproject_data = toml.load(f)
    dependencies = {}
    package_name = None

    # Extract package name
    try:
        package_name = pyproject_data["project"]["name"]
    except KeyError:
        print(f"Warning: Could not find package name in {pyproject_path}")

    # Extract dependencies
    try:
        deps = pyproject_data["project"]["dependencies"]
        if isinstance(deps, dict):  # If dependencies are a dictionary
            for dep, version in deps.items():
                if dep.startswith("bionemo-"):
                    dependencies[dep] = version  # Keep dependency with its version

        elif isinstance(deps, list):  # If dependencies are a list
            for dep in deps:
                if dep.startswith("bionemo-"):
                    dependencies[dep] = "unpinned"
    except KeyError:
        print(f"Warning: Could not find dependencies in {pyproject_path}")

    if "tool" in pyproject_data and "maturin" in pyproject_data["tool"]:
        dep = pyproject_data["tool"]["maturin"]["module-name"]
        if dep.startswith("bionemo."):
            dependencies[dep.replace(".", "-")] = "unpinned"

    return package_name, dependencies


def build_dependency_graph(base_dir, directories):
    """Build a dependency graph for all sub-packages."""
    pyproject_files = []
    for directory in directories:
        pyproject_files.append(base_dir / directory / "pyproject.toml")
    dependency_graph = defaultdict(dict)

    for pyproject_file in pyproject_files:
        package_name, dependencies = parse_dependencies(pyproject_file)
        if package_name:
            dependency_graph[package_name] = dependencies

    return dependency_graph


def visualize_dependency_graph(dependency_graph, filename):
    """Visualize the dependency graph using NetworkX."""
    G = nx.DiGraph()
    edge_labels = {}

    # Track all packages explicitly
    all_packages = set(dependency_graph.keys())

    for package, dependencies in dependency_graph.items():
        if isinstance(dependencies, dict):
            for dep, version in dependencies.items():
                G.add_edge(dep, package)  # Add edge from package to dependency
                edge_labels[(dep, package)] = version  # Label the edge with the version
                all_packages.add(dep)
        else:
            for dep in dependencies:
                G.add_edge(dep, package)  # Add edge from package to dependency
                all_packages.add(dep)

    # Ensure isolated nodes (without edges) are included in the graph
    for package in all_packages:
        if package not in G:
            G.add_node(package)
    # Use a circular layout, ensuring packages are evenly distributed
    pos = nx.circular_layout(G)

    plt.figure(figsize=(14, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        font_size=10,
        font_weight="bold",
        arrowsize=20,
        edge_color="gray",
    )

    # Draw edge labels for the dependency versions
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color="red")
    plt.title("Dependency Graph", fontsize=16)
    plt.savefig(filename)


def find_bionemo_subpackages(base_dir, directories):
    """Find all unique `bionemo.<name>` imports in Python files within a directory."""
    bionemo_import_pattern = re.compile(
        r"^\s*(?:from|import)\s+bionemo\.([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+|\.|$)", re.MULTILINE
    )
    found_imports = {}
    for dir_name in directories:
        directory = base_dir / dir_name / "src"
        subpackages = set()

        for file_path in Path(directory).rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    matches = bionemo_import_pattern.findall(content)
                    subpackages.update(matches)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
        full_subpackage_names = {f"bionemo-{subpackage}" for subpackage in subpackages}
        if dir_name in full_subpackage_names:
            full_subpackage_names.remove(dir_name)
        found_imports[dir_name] = full_subpackage_names
    return found_imports


def parse_tach_toml(toml_path):
    """Parse dependencies from a tach.toml file."""
    tach_toml_dependencies = {}
    with open(toml_path, "r") as f:
        toml_data = toml.load(f)
        for module in toml_data["modules"]:
            tach_toml_dependencies[(module["path"].replace(".", "-"))] = [
                item.replace(".", "-") for item in module["depends_on"]
            ]
    return tach_toml_dependencies


def resolve_dependencies(subpackage, toml_imports, resolved=None, seen=None):
    """Recursively resolve all dependencies, including transitive ones."""
    if resolved is None:
        resolved = set()
    if seen is None:
        seen = set()

    if subpackage in seen:
        return resolved  # Avoid circular dependencies
    seen.add(subpackage)

    for dep in toml_imports.get(subpackage, []):
        resolved.add(dep)
        if dep in toml_imports:  # Resolve further if it's a subpackage
            resolve_dependencies(dep, toml_imports, resolved, seen)

    return resolved


if __name__ == "__main__":
    script_path = Path(__file__).resolve()
    logger = logging.getLogger(__name__)

    # Get the parent directory
    parent_directory = script_path.parents[5]
    base_dir = parent_directory / "sub-packages"
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(base_dir / d)]
    pyproject_dependency_graph = build_dependency_graph(base_dir, directories)

    tach_toml_dependency_graph = parse_tach_toml(parent_directory / "tach.toml")
    file_path_imports = find_bionemo_subpackages(base_dir, directories)
    console_handler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    pyproject_not_toml = set(pyproject_dependency_graph.keys()) - set(tach_toml_dependency_graph.keys())
    toml_not_pyproject = set(tach_toml_dependency_graph.keys()) - set(pyproject_dependency_graph.keys())

    if len(pyproject_not_toml) > 0:
        logger.warning(f"\npyproject.toml - tach.toml: {', '.join(pyproject_not_toml)}")
    if len(toml_not_pyproject) > 0:
        logger.warning(f"\npyproject.toml - tach.toml: {', '.join(toml_not_pyproject)}")

    for name, dependency_graph in zip(
        ["pyproject.toml", "tach.toml"], [pyproject_dependency_graph, tach_toml_dependency_graph]
    ):
        logger.warning(f"\nDependencies not resolved in {name}:")
        for directory in file_path_imports:
            resolved_dependencies = resolve_dependencies(directory, dependency_graph)
            if not (file_path_imports[directory] <= resolved_dependencies):
                logger.warning(f"{directory} : {file_path_imports[directory] - resolved_dependencies}")

    logger.warning("\nDifferences in pyproject.toml and tach.toml per-package: ")
    for d in pyproject_dependency_graph:
        if d in tach_toml_dependency_graph:
            pyproject_minus_tach = set(pyproject_dependency_graph[d].keys()) - set(tach_toml_dependency_graph[d])
            tach_minus_pyproject = set(tach_toml_dependency_graph[d]) - set(pyproject_dependency_graph[d].keys())
            if len(pyproject_minus_tach) > 0:
                logger.warning(f"{d} project.toml - tach.toml: {' ,'.join(pyproject_minus_tach)}")
            if len(tach_minus_pyproject) > 0:
                logger.warning(f"{d} tach.toml - project.toml: {', '.join(tach_minus_pyproject)}")

    visualize_dependency_graph(pyproject_dependency_graph, "dependency_graph_pyproject.png")
    visualize_dependency_graph(tach_toml_dependency_graph, "dependency_graph_tach.png")
    visualize_dependency_graph(file_path_imports, "dependency_file_imports.png")
