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
"""Generate reference pages and copy docs from framework packages and recipes."""

import itertools
import json
import logging
import os
import re
from pathlib import Path

import mkdocs_gen_files


class _IgnoreNotebookAltWarnings(logging.Filter):
    """Filter out nbconvert warnings for auto-filled image alt text."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False for the specific warning we intentionally suppress."""
        return "Alternative text is missing on" not in record.getMessage()


# log stuff
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("traitlets").addFilter(_IgnoreNotebookAltWarnings())

SUPPORT_FILE_SUFFIXES = {
    ".cfg",
    ".conf",
    ".csv",
    ".fasta",
    ".gif",
    ".ini",
    ".jpeg",
    ".jpg",
    ".json",
    ".png",
    ".py",
    ".sh",
    ".sql",
    ".svg",
    ".toml",
    ".txt",
    ".tsv",
    ".webp",
    ".yaml",
    ".yml",
}
SUPPORT_FILE_NAMES = {"Dockerfile", "LICENSE", "Makefile", "requirements.txt"}
SKIP_SUPPORT_DIRS = {"assets", "examples", "notebooks", ".venv", "__pycache__", ".pytest_cache"}
GITHUB_BLOB_BASE = "https://github.com/NVIDIA-BioNeMo/bionemo-framework/blob/main"


def _rewrite_relative_links(
    source: Path, dest: Path, root: Path, text: str, rendered_markdown_links: bool = False
) -> str:
    """Rewrite relative links so they resolve correctly from the docs-tree destination.

    Handles path families that commonly appear in imported READMEs:

    * Paths resolving under ``docs/docs/`` (shared assets, images) are rewritten
      relative to the docs-tree root so they reach the correct asset.
    * Paths resolving under ```` are rewritten relative to the
      ``main/recipes/`` subtree so cross-recipe links keep working.
    * Paths from sub-package READMEs to examples, notebooks, and assets are
      rewritten to the generated docs locations where those files are copied.

    All other relative links (e.g. to ``ci/scripts/``) are left untouched.

    Args:
        source: Absolute source file path in the repository.
        dest: Destination path in the generated docs tree (relative to docs root).
        root: Repository root directory (absolute).
        text: Markdown content to process.
        rendered_markdown_links: Whether Markdown links are already in rendered
            HTML context, as with Markdown cells inside notebooks.

    Returns:
        Markdown text with rewritten relative links.
    """
    source_dir = source.parent
    # Mkdocs rewrites relative paths in markdown syntax (![](path), [](path))
    # when converting foo.md -> foo/index.html (use_directory_urls).  For those,
    # compute paths relative to the markdown file's parent directory.
    dest_dir_md = str(dest.parent)

    # Mkdocs does NOT rewrite paths inside raw HTML tags (<img src>, <a href>).
    # Those resolve from the final HTML location, which is one level deeper for
    # non-index files (foo.md -> foo/index.html).
    if dest.name in ("index.md", "README.md"):
        dest_dir_html = dest_dir_md
    else:
        dest_dir_html = str(dest.parent / dest.stem)

    docs_docs_dir = root / "docs" / "docs"
    recipe_roots = {
        root / "models": "models",
        root / "recipes": "recipes",
        root / "interpretability": "interpretability",
    }

    def _make_rewriter(ref_dir: str, rendered_links: bool = False):
        """Return a rewrite function that computes paths relative to *ref_dir*."""

        def _final_link(target: str, append_trailing_slash: bool, suffix: str) -> str:
            if rendered_links:
                target_path = Path(target)
                is_directory_url = False
                if target_path.name in ("index.md", "README.md"):
                    target = target_path.parent.as_posix()
                    is_directory_url = True
                elif target_path.suffix in {".md", ".ipynb"}:
                    target = target_path.with_suffix("").as_posix()
                    is_directory_url = True

                new_rel = Path(os.path.relpath(target, ref_dir)).as_posix()
                if is_directory_url and not new_rel.endswith("/"):
                    new_rel += "/"
                return new_rel + suffix

            new_rel = Path(os.path.relpath(target, ref_dir)).as_posix()
            return new_rel + ("/" if append_trailing_slash else "") + suffix

        def _rewrite(rel_path: str) -> str:
            if not rel_path or rel_path.startswith(
                ("#", "http://", "https://", "mailto:", "data:", "tel:", "javascript:")
            ):
                return rel_path

            clean = rel_path
            suffix = ""
            if "#" in clean:
                clean, frag = clean.split("#", 1)
                suffix = "#" + frag
            if "?" in clean:
                clean, qs = clean.split("?", 1)
                suffix = "?" + qs + suffix

            if not clean:
                return rel_path

            trailing_slash = clean.endswith("/")

            try:
                resolved = (source_dir / clean).resolve()
            except (ValueError, OSError):
                return rel_path

            if not resolved.exists() and resolved.suffix == "":
                markdown_file = resolved.with_suffix(".md")
                if markdown_file.exists():
                    resolved = markdown_file

            # docs/docs/... -> docs tree root
            try:
                rel_to_docs = resolved.relative_to(docs_docs_dir)
                target = rel_to_docs.as_posix()
                return _final_link(target, trailing_slash, suffix)
            except ValueError:
                pass

            for recipe_root, section in recipe_roots.items():
                try:
                    rel_to_recipe_root = resolved.relative_to(recipe_root)
                except ValueError:
                    continue

                if resolved.suffix == ".md" and "src" in rel_to_recipe_root.parts:
                    repo_rel = resolved.relative_to(root).as_posix()
                    return f"{GITHUB_BLOB_BASE}/{repo_rel}" + suffix

                target = f"main/recipes/{section}/" + rel_to_recipe_root.as_posix()
                if target.endswith("/README.md"):
                    target = target[:-10] + "/index.md"
                return _final_link(target, trailing_slash, suffix)

            return rel_path

        return _rewrite

    rewrite_md = _make_rewriter(dest_dir_html if rendered_markdown_links else dest_dir_md, rendered_markdown_links)
    rewrite_html = _make_rewriter(dest_dir_html)

    # Markdown images and links: ![alt](path) and [text](path)
    text = re.sub(
        r'(!?\[[^\]]*\]\()([^)\s"]+)((?:\s+"[^"]*")?\))',
        lambda m: m.group(1) + rewrite_md(m.group(2)) + m.group(3),
        text,
    )
    # HTML <img src="..."> — needs the deeper reference directory
    text = re.sub(
        r'(<img\s[^>]*?src=")([^"]+)(")',
        lambda m: m.group(1) + rewrite_html(m.group(2)) + m.group(3),
        text,
    )
    # HTML <a href="..."> — same
    text = re.sub(
        r'(<a\s[^>]*?href=")([^"]+)(")',
        lambda m: m.group(1) + rewrite_html(m.group(2)) + m.group(3),
        text,
    )

    return text


def _sanitize_imported_text(
    source: Path, dest: Path, root: Path, text: str, rendered_markdown_links: bool = False
) -> str:
    """Apply docs-specific cleanups and link rewriting to imported Markdown files.

    Args:
        source: Absolute source file path in the repository.
        dest: Destination path in the generated docs tree.
        root: Repository root directory.
        text: Source file contents.
        rendered_markdown_links: Whether Markdown links are rendered directly
            from this text instead of being rewritten by MkDocs.

    Returns:
        Sanitized Markdown content with correct relative links.
    """
    source_str = source.as_posix()
    if source_str.endswith("recipes/geneformer_native_te_mfsdp_fp8/README.md"):
        heading_idx = text.find("# Geneformer Pretraining")
        if heading_idx != -1:
            text = text[heading_idx:]

    # The macros plugin parses `{#id}` as a Jinja comment start before
    # attr_list can turn it into a heading id. Preserve the anchor explicitly.
    text = re.sub(
        r"(?m)^(#{1,6}\s+.+?)\s+\{#([A-Za-z0-9_.:-]+)\}\s*$",
        r'<a id="\2"></a>' + "\n" + r"\1",
        text,
    )

    recipe_roots = (root / "models", root / "recipes", root / "interpretability")
    for recipe_root in recipe_roots:
        try:
            source.resolve().relative_to(recipe_root.resolve())
            text = _rewrite_relative_links(source, dest, root, text, rendered_markdown_links)
            break
        except ValueError:
            pass

    return text


def copy_text_file(source: Path, dest: Path, root: Path, log_message: str) -> None:
    """Copy a text file and set up edit path.

    Args:
        source (Path): Source file path.
        dest (Path): Destination file path.
        root (Path): Root directory for relative path calculation.
        log_message (str): Message to log after copying.
    """
    with mkdocs_gen_files.open(dest, "w") as fd:
        fd.write(_sanitize_imported_text(source, dest, root, source.read_text()))
    logger.info(log_message)
    mkdocs_gen_files.set_edit_path(dest, source.relative_to(root))


def copy_binary_file(source: Path, dest: Path, log_message: str) -> None:
    """Copy a binary file.

    Args:
        source (Path): Source file path.
        dest (Path): Destination file path.
        log_message (str): Message to log after copying.
    """
    with mkdocs_gen_files.open(dest, "wb") as fd:
        fd.write(source.read_bytes())
    logger.info(log_message)


def copy_notebook_file(source: Path, dest: Path, root: Path, log_message: str) -> None:
    """Copy a notebook while normalizing minor schema issues for docs builds.

    Some notebooks contain legacy `stream` outputs without a `name` field,
    which `mkdocs-jupyter` rejects during site generation. Default those
    outputs to `stdout` when copying into the generated docs tree.

    Args:
        source (Path): Source notebook path.
        dest (Path): Destination notebook path.
        root (Path): Repository root for edit-path calculation.
        log_message (str): Message to log after copying.
    """
    notebook = json.loads(source.read_text())
    notebook.setdefault("metadata", {})

    for cell in notebook.get("cells", []):
        cell.setdefault("metadata", {})
        if cell.get("cell_type") == "markdown":
            source_text = cell.get("source", "")
            if isinstance(source_text, list):
                text = "".join(source_text)
                cell["source"] = _sanitize_imported_text(
                    source, dest, root, text, rendered_markdown_links=True
                ).splitlines(keepends=True)
            elif isinstance(source_text, str):
                cell["source"] = _sanitize_imported_text(source, dest, root, source_text, rendered_markdown_links=True)

        if cell.get("cell_type") == "code":
            cell.setdefault("outputs", [])
            cell.setdefault("execution_count", None)

        for output in cell.get("outputs", []):
            output_type = output.get("output_type")

            if output_type == "stream":
                output.setdefault("name", "stdout")
            elif output_type in {"display_data", "execute_result"}:
                output.setdefault("metadata", {})

                if output_type == "execute_result":
                    output.setdefault("execution_count", None)

    with mkdocs_gen_files.open(dest, "w") as fd:
        json.dump(notebook, fd)

    logger.info(log_message)
    mkdocs_gen_files.set_edit_path(dest, source.relative_to(root))


def write_directory_index(dest_dir: Path, title: str, entries: list[Path]) -> None:
    """Write a simple index page for a generated documentation directory."""
    if not entries:
        return

    index_file = dest_dir / "index.md"
    with mkdocs_gen_files.open(index_file, "w") as fd:
        fd.write(f"# {title}\n\n")
        for entry in sorted(entries):
            label = entry.stem.replace("_", " ").replace("-", " ").title()
            fd.write(f"- [{label}]({entry.relative_to(dest_dir).as_posix()})\n")

    logger.info("Added generated index: %s", index_file)


def copy_docs_from_dir(source_dir: Path, dest_dir: Path, root: Path, log_prefix: str) -> list[Path]:
    """Copy Markdown and notebook files from a directory tree.

    Args:
        source_dir (Path): Directory containing documentation files.
        dest_dir (Path): Destination directory in the generated docs tree.
        root (Path): Repository root for edit-path calculation.
        log_prefix (str): Prefix used in log messages.

    Returns:
        Paths copied into the generated docs tree.
    """
    copied_docs: list[Path] = []
    has_directory_index = False

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file() or path.suffix not in {".md", ".ipynb"}:
            continue

        dest_file = dest_dir / path.relative_to(source_dir)
        if dest_file.parent == dest_dir and dest_file.name in {"index.md", "README.md"}:
            has_directory_index = True

        if path.suffix == ".ipynb":
            copy_notebook_file(path, dest_file, root, f"{log_prefix}: {dest_file}")
        else:
            copy_text_file(path, dest_file, root, f"{log_prefix}: {dest_file}")
        copied_docs.append(dest_file)

    if copied_docs and not has_directory_index:
        title = source_dir.name.replace("_", " ").replace("-", " ").title()
        write_directory_index(dest_dir, title, copied_docs)

    return copied_docs


def copy_assets_dir(source_dir: Path, dest_dir: Path, log_prefix: str) -> None:
    """Copy static assets into the generated docs tree.

    Args:
        source_dir (Path): Directory containing assets.
        dest_dir (Path): Destination assets directory in the generated docs tree.
        log_prefix (str): Prefix used in log messages.
    """
    if not source_dir.exists():
        return

    for asset_path in source_dir.rglob("*"):
        if asset_path.is_file():
            relative_path = asset_path.relative_to(source_dir)
            dest_asset = dest_dir / relative_path
            copy_binary_file(asset_path, dest_asset, f"{log_prefix}: {dest_asset}")


def _should_copy_support_file(path: Path) -> bool:
    """Return whether a file should be mirrored as a support asset.

    Args:
        path (Path): Relative path within an imported directory.

    Returns:
        bool: True if the file should be copied into the generated docs tree.
    """
    if any(part in SKIP_SUPPORT_DIRS for part in path.parts[:-1]):
        return False

    if path.name == "README.md":
        return len(path.parts) > 1

    if path.suffix in {".ipynb", ".md"}:
        return False

    if path.name.startswith("."):
        return False

    return (
        path.suffix in SUPPORT_FILE_SUFFIXES or path.name in SUPPORT_FILE_NAMES or path.name.startswith("Dockerfile")
    )


def _should_write_support_index(relative_dir: Path) -> bool:
    """Return whether a copied support directory should get a generated index."""
    parts = relative_dir.parts
    return parts in {
        (),
        ("config",),
        ("hydra_config",),
        ("hydra_config", "model"),
        ("esm2",),
    }


def write_support_directory_indexes(dest_dir: Path, copied_files: list[Path], explicit_index_dirs: set[Path]) -> None:
    """Write index pages for support-only directories copied into docs."""
    directories: set[Path] = set(explicit_index_dirs)
    for copied_file in copied_files:
        relative_dir = copied_file.parent.relative_to(dest_dir)
        if _should_write_support_index(relative_dir):
            directories.add(copied_file.parent)

    for directory in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        if directory in explicit_index_dirs:
            continue

        entries: list[Path] = []
        for copied_file in copied_files:
            if copied_file.parent == directory and copied_file.name != "index.md":
                entries.append(copied_file)

        child_dirs = {child_dir for child_dir in directories if child_dir.parent == directory}
        entries.extend(child_dir / "index.md" for child_dir in sorted(child_dirs))
        write_directory_index(directory, directory.name.replace("_", " ").replace("-", " ").title(), entries)


def copy_support_files(source_dir: Path, dest_dir: Path, root: Path, log_prefix: str) -> None:
    """Mirror linked support files into the generated docs tree.

    This keeps repo-relative links to scripts, configs, and tests working after
    imported READMEs are rendered inside the documentation site.

    Args:
        source_dir (Path): Directory to mirror from.
        dest_dir (Path): Destination directory in the generated docs tree.
        root (Path): Repository root for edit-path calculation.
        log_prefix (str): Prefix used in log messages.
    """
    copied_files: list[Path] = []
    explicit_index_dirs: set[Path] = set()

    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(source_dir)
        if not _should_copy_support_file(relative_path):
            continue

        if relative_path.name == "README.md":
            dest_file = dest_dir / relative_path.parent / "index.md"
            explicit_index_dirs.add(dest_file.parent)
            copy_text_file(path, dest_file, root, f"{log_prefix}: {dest_file}")
        else:
            dest_file = dest_dir / relative_path
            copy_binary_file(path, dest_file, f"{log_prefix}: {dest_file}")
        copied_files.append(dest_file)

    write_support_directory_indexes(dest_dir, copied_files, explicit_index_dirs)


def generate_api_reference() -> None:
    """Generate API reference documentation for recipe and model source directories."""
    root = Path(__file__).parent.parent.parent
    source_roots = itertools.chain((root / "models").rglob("src"), (root / "recipes").rglob("src"))

    for src in source_roots:
        # Process Python files
        for path in sorted(src.rglob("*.py")):
            module_path = path.relative_to(src).with_suffix("")
            doc_path = path.relative_to(src).with_suffix(".md")
            full_doc_path = Path("main/references/API_reference") / doc_path
            parts = tuple(module_path.parts)

            if parts[-1] in ("__init__", "__main__"):
                continue

            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                identifier = ".".join(parts)
                print("::: " + identifier, file=fd)

            mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

        # Process Markdown files
        for path in sorted(src.rglob("*.md")):
            doc_path = path.relative_to(src)
            full_doc_path = Path("main/references/API_reference") / doc_path
            copy_text_file(path, full_doc_path, root, f"Added Markdown file: {full_doc_path}")


def get_recipes_readmes(recipes_dir: Path, root: Path) -> None:
    """Copy README files from root recipe directories to the docs recipes directory.

    Args:
        recipes_dir (Path): The repository root.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    # Main README
    main_readme = recipes_dir / "README.md"
    if main_readme.exists():
        dest_file = Path("main/recipes/index.md")
        copy_text_file(main_readme, dest_file, root, f"Added recipes README: {dest_file}")
        alias_file = Path("main/recipes/README.md")
        copy_text_file(main_readme, alias_file, root, f"Added recipes README alias: {alias_file}")

    # Process both models and recipes subdirectories
    for subdir in ["models", "recipes"]:
        subdir_path = recipes_dir / subdir
        if not subdir_path.exists():
            continue

        # Copy subdirectory README if it exists
        subdir_readme = subdir_path / "README.md"
        if subdir_readme.exists():
            dest_file = Path("main/recipes") / subdir / "index.md"
            copy_text_file(subdir_readme, dest_file, root, f"Added recipes {subdir} README: {dest_file}")
            alias_file = Path("main/recipes") / subdir / "README.md"
            copy_text_file(subdir_readme, alias_file, root, f"Added recipes {subdir} README alias: {alias_file}")

        # Copy collection-level markdown docs such as context guides.
        for extra_doc in sorted(subdir_path.glob("*.md")):
            if extra_doc.name == "README.md":
                continue

            dest_file = Path("main/recipes") / subdir / extra_doc.name
            copy_text_file(extra_doc, dest_file, root, f"Added recipes {subdir} doc: {dest_file}")

        # Copy individual model/recipe READMEs
        for item in subdir_path.iterdir():
            if not item.is_dir():
                continue

            readme_file = item / "README.md"
            if readme_file.exists():
                dest_dir = Path("main/recipes") / subdir / item.name
                dest_file = dest_dir / "index.md"
                copy_text_file(readme_file, dest_file, root, f"Added {subdir} README: {dest_file}")


def get_recipe_docs(recipe_item: Path, section: str, root: Path) -> None:
    """Copy supplementary docs from a recipe or model into the recipes tree.

    Args:
        recipe_item (Path): Path to an individual recipe or model directory.
        section (str): Either "models" or "recipes".
        root (Path): Repository root for edit-path calculation.
    """
    dest_dir = Path("main/recipes") / section / recipe_item.name

    for extra_doc in sorted(recipe_item.glob("*.md")):
        if (
            extra_doc.name == "README.md"
            or extra_doc.name.startswith("AGENT_")
            or extra_doc.name == "AI_DOCUMENTATION.md"
        ):
            continue

        dest_file = dest_dir / extra_doc.name
        copy_text_file(extra_doc, dest_file, root, f"Added recipe doc: {dest_file}")

    for docs_dir_name in ("examples", "notebooks"):
        docs_dir = recipe_item / docs_dir_name
        if docs_dir.exists():
            copy_docs_from_dir(docs_dir, dest_dir / docs_dir_name, root, "Added recipe example")


def get_recipe_examples(recipe_item: Path, root: Path) -> None:
    """Copy recipe examples and notebooks into the tutorials tree.

    Args:
        recipe_item (Path): Path to an individual recipe or model directory.
        root (Path): Repository root for edit-path calculation.
    """
    dest_name = "bionemo-evo2" if recipe_item.name == "evo2_megatron" else recipe_item.name
    dest_root = Path("main/examples") / dest_name
    copy_assets_dir(recipe_item / "assets", dest_root / "assets", "Added recipe tutorial asset")

    example_sections: list[Path] = []
    for docs_dir_name in ("examples", "notebooks"):
        docs_dir = recipe_item / docs_dir_name
        if docs_dir.exists():
            dest_dir = dest_root / docs_dir_name
            copied_docs = copy_docs_from_dir(docs_dir, dest_dir, root, "Added recipe tutorial")
            if copied_docs:
                example_sections.append(dest_dir / "index.md")

    write_directory_index(dest_root, f"{dest_name} Examples", example_sections)


def get_recipes_assets(recipes_dir: Path, root: Path) -> None:
    """Copy assets from root recipe directories to the docs recipes directory.

    Args:
        recipes_dir (Path): The repository root.
        root (Path): The root directory of the project.

    Returns:
        None
    """
    if not recipes_dir.exists():
        return

    # Handle root-level assets directory
    copy_assets_dir(recipes_dir / "assets", Path("main/recipes/assets"), "Added root recipe asset")

    # Process both models and recipes subdirectories
    for subdir in ["models", "recipes"]:
        subdir_path = recipes_dir / subdir
        if not subdir_path.exists():
            continue

        for item in subdir_path.iterdir():
            if not item.is_dir():
                continue

            dest_dir = Path("main/recipes") / subdir / item.name
            copy_assets_dir(item / "assets", dest_dir / "assets", "Added recipe asset")
            copy_support_files(item, dest_dir, root, "Added recipe support file")


def generate_pages() -> None:
    """Generate pages for documentation.

    This function orchestrates API references, notebooks, and README files for recipes.

    Returns:
        None
    """
    root = Path(__file__).parent.parent.parent
    recipes_dir = root

    # Provide a stub versions.json so mike's version-selector JS doesn't
    # flood the console with 404 retries during local `mkdocs serve`.
    with mkdocs_gen_files.open("versions.json", "w") as f:
        json.dump([{"version": "main", "title": "main", "aliases": ["latest"]}], f)

    # Generate API docs for recipe and model packages.
    generate_api_reference()

    # Process recipes
    get_recipes_assets(recipes_dir, root)
    get_recipes_readmes(recipes_dir, root)

    for section in ("models", "recipes"):
        section_dir = recipes_dir / section
        if not section_dir.exists():
            continue

        for recipe_item in sorted(section_dir.iterdir()):
            if not recipe_item.is_dir():
                continue

            get_recipe_docs(recipe_item, section, root)
            get_recipe_examples(recipe_item, root)


if __name__ in {"__main__", "<run_path>"}:
    generate_pages()
