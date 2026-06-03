#!/usr/bin/env python

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

"""Check generated MkDocs HTML for broken internal links."""

from __future__ import annotations

import argparse
import fnmatch
import posixpath
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse


IGNORED_SCHEMES = {"data", "http", "https", "javascript", "mailto", "tel"}


@dataclass(frozen=True)
class Reference:
    """A URL-bearing HTML attribute discovered in a rendered page."""

    source: Path
    attr: str
    url: str


class LinkParser(HTMLParser):
    """Extract URL-bearing attributes from rendered HTML."""

    def __init__(self, source: Path) -> None:
        """Initialize the parser for a rendered HTML source file."""
        super().__init__(convert_charrefs=True)
        self.source = source
        self.references: list[Reference] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect references and anchors from an opening tag."""
        self._handle_attrs(attrs)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect references and anchors from a self-closing tag."""
        self._handle_attrs(attrs)

    def _handle_attrs(self, attrs: list[tuple[str, str | None]]) -> None:
        for attr, value in attrs:
            if value is None:
                continue

            if attr in {"id", "name"} and value:
                self.anchors.add(value)
            elif attr in {"href", "src", "data"}:
                self.references.append(Reference(self.source, attr, value))
            elif attr == "srcset":
                for srcset_url in _parse_srcset(value):
                    self.references.append(Reference(self.source, attr, srcset_url))


def _parse_srcset(value: str) -> Iterable[str]:
    for candidate in value.split(","):
        candidate = candidate.strip()
        if candidate:
            yield candidate.split()[0]


def _site_base_path(config_path: Path) -> str:
    if not config_path.exists():
        return "/"

    match = re.search(r"(?m)^site_url:\s*(?P<site_url>\S+)\s*$", config_path.read_text())
    site_url = match.group("site_url") if match else ""
    path = urlparse(site_url).path or "/"
    if not path.startswith("/"):
        path = "/" + path
    if not path.endswith("/"):
        path += "/"
    return path


def _page_path(site_dir: Path, html_file: Path) -> str:
    rel = html_file.relative_to(site_dir).as_posix()
    if rel == "index.html":
        return "/"
    if rel.endswith("/index.html"):
        return "/" + rel[: -len("index.html")]
    return "/" + rel


def _target_path(site_dir: Path, source: Path, url: str, base_path: str) -> tuple[Path, str] | None:
    parsed = urlparse(url)
    if parsed.scheme in IGNORED_SCHEMES or parsed.netloc:
        return None

    raw_path = unquote(parsed.path)
    if not raw_path:
        return source, unquote(parsed.fragment)

    if raw_path.startswith("/"):
        if base_path != "/" and raw_path.startswith(base_path):
            raw_path = "/" + raw_path[len(base_path) :]
        elif base_path != "/":
            return site_dir / raw_path.lstrip("/"), unquote(parsed.fragment)

        site_rel = raw_path.lstrip("/")
    else:
        base_url_path = _page_path(site_dir, source)
        if not base_url_path.endswith("/"):
            base_url_path = posixpath.dirname(base_url_path) + "/"
        site_rel = posixpath.normpath(posixpath.join(base_url_path, raw_path)).lstrip("/")

    target = site_dir / site_rel
    if raw_path.endswith("/"):
        return target / "index.html", unquote(parsed.fragment)

    if target.is_dir():
        return target / "index.html", unquote(parsed.fragment)

    if target.exists():
        return target, unquote(parsed.fragment)

    if target.suffix in {".ipynb", ".md"}:
        return target.with_suffix("") / "index.html", unquote(parsed.fragment)

    if target.suffix == "":
        return target / "index.html", unquote(parsed.fragment)

    return target, unquote(parsed.fragment)


def _load_pages(site_dir: Path) -> tuple[dict[Path, list[Reference]], dict[Path, set[str]]]:
    references_by_page: dict[Path, list[Reference]] = {}
    anchors_by_page: dict[Path, set[str]] = {}

    for html_file in sorted(site_dir.rglob("*.html")):
        rel_parts = html_file.relative_to(site_dir).parts
        if html_file.name == "404.html" or "SUMMARY" in rel_parts:
            continue

        parser = LinkParser(html_file)
        parser.feed(html_file.read_text(errors="ignore"))
        references_by_page[html_file] = parser.references
        anchors_by_page[html_file] = parser.anchors

    return references_by_page, anchors_by_page


def _load_ignore_patterns(ignore_file: Path | None) -> list[str]:
    if ignore_file is None or not ignore_file.exists():
        return []

    return [
        line.strip()
        for line in ignore_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _check_links(site_dir: Path, base_path: str, ignore_patterns: list[str]) -> list[str]:
    references_by_page, anchors_by_page = _load_pages(site_dir)
    errors: list[str] = []

    def _add_error(error: str) -> None:
        if any(fnmatch.fnmatchcase(error, pattern) for pattern in ignore_patterns):
            return
        errors.append(error)

    for source, references in references_by_page.items():
        for reference in references:
            target = _target_path(site_dir, source, reference.url, base_path)
            if target is None:
                continue

            target_file, fragment = target
            if not target_file.exists():
                _add_error(
                    f"{source.relative_to(site_dir)}: {reference.attr}={reference.url!r} "
                    f"-> missing {target_file.relative_to(site_dir)}"
                )
                continue

            if fragment and target_file.suffix == ".html" and fragment not in anchors_by_page.get(target_file, set()):
                _add_error(
                    f"{source.relative_to(site_dir)}: {reference.attr}={reference.url!r} -> missing anchor #{fragment}"
                )

    return errors


def main() -> int:
    """Run the internal link checker."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("site_dir", nargs="?", default="site", type=Path)
    parser.add_argument("--mkdocs-config", default="mkdocs.yml", type=Path)
    parser.add_argument("--ignore-file", default="link-check-ignore.txt", type=Path)
    args = parser.parse_args()

    site_dir = args.site_dir.resolve()
    if not site_dir.exists():
        parser.error(f"site directory does not exist: {site_dir}")

    errors = _check_links(
        site_dir,
        _site_base_path(args.mkdocs_config),
        _load_ignore_patterns(args.ignore_file),
    )
    if errors:
        print("Broken internal links found:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("No broken internal links found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
