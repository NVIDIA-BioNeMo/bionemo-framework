#!/usr/bin/env python3

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

r"""Verify that repo paths cited in skill documentation still exist, and that SRC-4 holds.

Two checks per line:

1. **Stale path** — a ``$BIONEMO_RECIPES/<path>``-style citation resolves against the repo root;
   if the path no longer exists, the commit is blocked. This is the original drift check.

2. **SRC-4 violation** — a bare ``models/...`` or ``recipes/...`` citation with no
   ``$BIONEMO_RECIPES/`` prefix is a policy violation. Skills live in ``skills/`` and are vendored
   into the BioNeMo Agent Toolkit by ``rsync --delete``; any path that escapes the skill subtree
   arrives broken in the catalog. The sanctioned form is ``$BIONEMO_RECIPES/<path>``, per SRC-4:
   "Runtime refs to a repo the user clones are fine."

Background: ``recipes/geneformer_native_te_mfsdp_fp8/AGENT_DOCUMENTATION.md`` is this repo's
worked example of documentation rot: it cited a ``tokenizer/`` dir, a ``gitingest.txt``, and two
``.sqsh`` files that no longer exist. It also cited no paths outside its own dir, so it passed no
cross-dir reference check -- but our new skills do reference the repo, making SRC-4 essential.

Usage:
    python ci/scripts/check_skill_references.py [files...]
"""

import argparse
import re
import sys
from pathlib import Path


# Canonical skill source root (SRC-2/SRC-3).
SKILLS_CANONICAL_ROOT = Path("skills")

# Local harness discovery root — may contain symlinks into SKILLS_CANONICAL_ROOT.
SKILLS_HARNESS_ROOT = Path(".claude/skills")

# Repo-relative path pattern: anchored to roots a skill might reference.
# Matches both bare and $BIONEMO_RECIPES/-prefixed citations.
_REPO_ROOTS = r"(?:models|recipes|ci|docs|interpretability)"
BARE_PATH_PAT = re.compile(
    r"(?<![/$\w.-])(" + _REPO_ROOTS + r"/[\w./\-]*[\w/])",
)
PREFIXED_PATH_PAT = re.compile(
    r"\$BIONEMO_RECIPES/(" + _REPO_ROOTS + r"/[\w./\-]*[\w/])",
)

SYMBOL_SUFFIX = "::"
STRIP_TRAILING = ".,;:)]}`\"'"


def find_repo_root(start: Path) -> Path:
    """Walk upward to the repository root.

    Args:
        start: Directory to start from.

    Returns:
        The first ancestor containing a ``.git`` entry, or ``start`` if none is found.
    """
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _strip(path_str: str) -> str:
    """Strip symbol suffix and trailing prose punctuation from a path string.

    Args:
        path_str: Raw path string extracted from the document.

    Returns:
        Cleaned path string.
    """
    return path_str.split(SYMBOL_SUFFIX, 1)[0].rstrip(STRIP_TRAILING)


def check_file(doc: Path, repo_root: Path) -> list[str]:
    """Run both checks on a single documentation file.

    Args:
        doc: The skill documentation file.
        repo_root: Repository root that ``$BIONEMO_RECIPES``-prefixed paths are resolved against.

    Returns:
        Human-readable error strings; empty when the document is clean.
    """
    errors = []
    text = doc.read_text(encoding="utf-8", errors="replace")
    for line_number, line in enumerate(text.splitlines(), start=1):
        # Check 1: $BIONEMO_RECIPES/-prefixed citations must still exist in this repo.
        for raw in PREFIXED_PATH_PAT.findall(line):
            candidate = _strip(raw)
            if candidate and not (repo_root / candidate).exists():
                errors.append(f"{doc}:{line_number}: cited path does not exist: {candidate}")

        # Check 2: bare citations are SRC-4 violations in a skill file.
        for raw in BARE_PATH_PAT.findall(line):
            candidate = _strip(raw)
            if not candidate:
                continue
            # Allow prose that only says "models/" or "recipes/" as a directory name (no sub-path).
            if "/" not in candidate.lstrip("/"):
                continue
            # Allow if it is immediately preceded by $BIONEMO_RECIPES on the same line — the regex
            # is anchored by the lookbehind, but a double-hit on the same token is possible if the
            # prefix check already matched it; guard explicitly.
            if f"$BIONEMO_RECIPES/{candidate}" in line:
                continue
            errors.append(
                f"{doc}:{line_number}: SRC-4 violation — bare repo path '{candidate}' escapes the "
                "skill subtree. Use '$BIONEMO_RECIPES/{candidate}' instead."
            )

    return errors


def collect_docs(files: list[str], repo_root: Path) -> list[Path]:
    """Resolve the set of skill documents to check, deduplicating symlink targets.

    Scans both ``skills/`` (canonical) and ``.claude/skills/`` (harness symlinks), but reports
    each physical file only once so symlinked dirs are not double-checked.

    Args:
        files: Explicit files passed by pre-commit, or empty to scan all skill markdown.
        repo_root: The repository root.

    Returns:
        Deduplicated list of markdown documents to check.
    """
    seen_inodes: set[int] = set()
    docs: list[Path] = []

    def _add(path: Path) -> None:
        try:
            stat = path.stat()
        except OSError:
            return
        if stat.st_ino not in seen_inodes:
            seen_inodes.add(stat.st_ino)
            docs.append(path)

    canonical = repo_root / SKILLS_CANONICAL_ROOT
    harness = repo_root / SKILLS_HARNESS_ROOT

    if files:
        all_roots = {canonical.resolve(), harness.resolve()}
        for f in files:
            path = Path(f).resolve()
            if path.suffix != ".md" or not path.is_file():
                continue
            # Accept files under either root.
            if any(str(path).startswith(str(r)) for r in all_roots):
                _add(path)
        return sorted(docs, key=lambda p: str(p))

    for root in (canonical, harness):
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.md"), key=lambda p: str(p)):
            if path.is_file():
                _add(path)

    return docs


def main() -> int:
    """Check skill documentation for stale paths and SRC-4 violations.

    Returns:
        0 when every check passes, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="Files to check (default: all skill markdown)")
    args = parser.parse_args()

    repo_root = find_repo_root(Path.cwd())
    docs = collect_docs(args.files, repo_root)
    if not docs:
        return 0

    errors: list[str] = []
    for doc in docs:
        errors.extend(check_file(doc, repo_root))

    if errors:
        print("Skill reference errors:\n", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        print(
            "\nFor stale paths: update the citation to the file's new location.\n"
            "For SRC-4 violations: prefix bare repo paths with $BIONEMO_RECIPES/ so the skill\n"
            "remains self-contained when vendored into the BioNeMo Agent Toolkit catalog.",
            file=sys.stderr,
        )
        return 1

    print(f"check_skill_references: {len(docs)} document(s) OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
