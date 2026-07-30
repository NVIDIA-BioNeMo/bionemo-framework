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

"""Synchronize, validate, and extract the small literature asset bundle.

This tool deliberately uses only the Python standard library so a human can
run it from a fresh checkout.  Run ``--help`` for the public interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.cookiejar
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from html import escape as html_escape
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, ClassVar, Iterator, Sequence


CONVERTER_VERSION = "1"
KING_WORKBOOK_SHA256 = "3cd26d4cca8bc1273a863c4b2304e755635fe0c7bed46308f54029b88f063fc9"
KING_WORKBOOK_SIZE = 1_094_903
ALLOWED_HOSTS = frozenset({"biorxiv.org", "www.biorxiv.org"})
USER_AGENT = "Mozilla/5.0 (compatible; phage-design-literature-sync/1.0; +https://agentskills.io/)"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024


class LiteratureAssetError(RuntimeError):
    """Base class for actionable, user-facing failures."""


class SourceValidationError(LiteratureAssetError):
    """A source or downloaded payload violates the pinned contract."""


class WorkbookValidationError(LiteratureAssetError):
    """The workbook or requested extraction violates its schema contract."""


class SourceDriftError(LiteratureAssetError):
    """Pinned input or converter identity changed without explicit update."""


class OfflineCacheMiss(LiteratureAssetError):
    """Offline mode was requested but a required object is not cached."""


class HttpFetchError(LiteratureAssetError):
    """Report a non-retryable HTTP response."""

    def __init__(self, status: int, message: str):
        """Initialize the error from an HTTP status and message."""
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class RetryableFetchError(LiteratureAssetError):
    """Report a fetch failure that may succeed on retry."""

    def __init__(self, message: str, *, retry_after: float | None = None):
        """Initialize the error and optional server-requested retry delay."""
        super().__init__(message)
        self.retry_after = retry_after


@dataclass(frozen=True)
class PaperSpec:
    """Define pinned source and asset expectations for one paper."""

    slug: str
    title: str
    doi: str
    version: str
    license: str
    article_url: str
    supplement_url: str | None
    expected_figures: int
    expected_figure_files: int
    expected_equations: int
    has_supplement: bool
    workbook_url: str | None = None


PAPERS: dict[str, PaperSpec] = {
    "king-2025-generative-phage-design": PaperSpec(
        slug="king-2025-generative-phage-design",
        title="Generative design of novel bacteriophages with genome language models",
        doi="10.1101/2025.09.12.675911",
        version="v1",
        license="CC-BY-4.0",
        article_url="https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.full",
        supplement_url="https://www.biorxiv.org/content/10.1101/2025.09.12.675911v1.supplementary-material",
        expected_figures=27,
        expected_figure_files=28,
        expected_equations=0,
        has_supplement=True,
        workbook_url=(
            "https://www.biorxiv.org/content/biorxiv/early/2025/09/17/"
            "2025.09.12.675911/DC1/embed/media-1.xlsx?download=true"
        ),
    ),
    "black-2026-design-efficiency": PaperSpec(
        slug="black-2026-design-efficiency",
        title="Quantifying evolutionary novelty and design efficiency in generative genome design",
        doi="10.64898/2026.06.12.731871",
        version="v1",
        license="CC-BY-4.0",
        article_url="https://www.biorxiv.org/content/10.64898/2026.06.12.731871v1.full",
        supplement_url=None,
        expected_figures=3,
        expected_figure_files=3,
        expected_equations=13,
        has_supplement=False,
    ),
}


@dataclass
class Node:
    """Represent one parsed HTML node."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)

    @property
    def classes(self) -> set[str]:
        """Return the node's CSS classes."""
        return set(self.attrs.get("class", "").split())

    def text(self) -> str:
        """Return concatenated descendant text."""
        return "".join(child if isinstance(child, str) else child.text() for child in self.children)

    def walk(self) -> Iterator[Node]:
        """Yield this node and its descendants depth first."""
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.walk()


class _DomParser(HTMLParser):
    _VOID: ClassVar[set[str]] = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("document")
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = Node(tag.lower(), {key.lower(): value or "" for key, value in attrs})
        self.stack[-1].children.append(node)
        if tag.lower() not in self._VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if self.stack[-1].tag == tag.lower() and tag.lower() not in self._VOID:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data: str) -> None:
        self.stack[-1].children.append(data)


@dataclass(frozen=True)
class MediaItem:
    """Describe one media asset referenced by an article."""

    source_url: str
    output_path: str
    kind: str
    label: str


@dataclass(frozen=True)
class ConvertedArticle:
    """Hold converted article text and referenced media."""

    paper_markdown: str
    supplement_markdown: str | None
    media: tuple[MediaItem, ...]
    source_title: str
    authors: tuple[str, ...]


def _parse_html(source: str) -> Node:
    parser = _DomParser()
    parser.feed(source)
    parser.close()
    return parser.root


def _find_first(root: Node, predicate: Callable[[Node], bool]) -> Node | None:
    return next((node for node in root.walk() if predicate(node)), None)


def _meta_values(root: Node, name: str) -> list[str]:
    return [
        node.attrs.get("content", "")
        for node in root.walk()
        if node.tag == "meta" and node.attrs.get("name", "").lower() == name.lower()
    ]


def _contains_heading(node: Node, wanted: str) -> bool:
    wanted = wanted.casefold()
    return any(
        child.tag in {"h1", "h2", "h3"} and " ".join(child.text().split()).casefold() == wanted
        for child in node.walk()
    )


def _safe_anchor(raw: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", raw.casefold()).strip("-")


def _absolute_url(base: str, url: str) -> str:
    return urllib.parse.urljoin(base, unescape(url))


def _canonical_media_url(url: str) -> str:
    static_prefix = "https://www.biorxiv.org/sites/default/files/highwire/"
    if url.startswith(static_prefix):
        return "https://www.biorxiv.org/content/" + url[len(static_prefix) :]
    return url


def _validate_article_identity(root: Node, spec: PaperSpec) -> tuple[str, tuple[str, ...]]:
    dois = _meta_values(root, "citation_doi")
    if dois != [spec.doi]:
        raise SourceValidationError(f"DOI mismatch for {spec.slug}: expected {spec.doi!r}, found {dois!r}")
    full_urls = _meta_values(root, "citation_fulltext_html_url") + _meta_values(root, "citation_full_html_url")
    if not full_urls or not any(f"{spec.doi}{spec.version}" in url for url in full_urls):
        raise SourceValidationError(f"version mismatch for {spec.slug}: expected {spec.version}")
    licenses = [
        node.attrs.get("href", "")
        for node in root.walk()
        if node.tag == "link" and "license" in node.attrs.get("rel", "").split()
    ]
    licenses.extend(_meta_values(root, "DC.Rights"))
    if not any("creativecommons.org/licenses/by/4.0" in value.casefold() for value in licenses):
        raise SourceValidationError(f"license mismatch for {spec.slug}: expected CC-BY-4.0")
    titles = _meta_values(root, "citation_title")
    title = titles[0] if titles else spec.title
    authors = tuple(value for value in _meta_values(root, "citation_author") if value)
    return title, authors


def _figure_urls(node: Node, base_url: str) -> list[str]:
    links = [
        _absolute_url(base_url, item.attrs["href"])
        for item in node.walk()
        if item.tag == "a"
        and "highwire-figure-link-newtab" in item.classes
        and ".large.jpg" in item.attrs.get("href", "").casefold()
    ]
    if not links:
        links = [
            _absolute_url(base_url, item.attrs["href"])
            for item in node.walk()
            if item.tag == "a" and ".large.jpg" in item.attrs.get("href", "").casefold()
        ]
    return list(dict.fromkeys(links))


def _figure_label(node: Node, fallback: str) -> str:
    label = _find_first(node, lambda item: "fig-label" in item.classes)
    return " ".join((label.text() if label else fallback).split())


def _caption_text(node: Node) -> str:
    caption = _find_first(node, lambda item: "fig-caption" in item.classes or "table-caption" in item.classes)
    if caption is None:
        return ""
    return " ".join(caption.text().split())


class _MarkdownRenderer:
    _BLOCKS: ClassVar[set[str]] = {"div", "section", "p", "ul", "ol", "li", "table", "tr", "blockquote"}

    def __init__(
        self,
        *,
        base_url: str,
        current_file: str,
        main_ids: set[str],
        supplement_ids: set[str],
        figures: dict[int, tuple[str, tuple[MediaItem, ...]]],
        equations: dict[int, MediaItem],
    ) -> None:
        self.base_url = base_url
        self.current_file = current_file
        self.main_ids = {_safe_anchor(value) for value in main_ids}
        self.supplement_ids = {_safe_anchor(value) for value in supplement_ids}
        self.figures = figures
        self.equations = equations

    def nodes(self, nodes: Sequence[Node | str]) -> str:
        rendered = "".join(self.node(item) for item in nodes)
        rendered = re.sub(r"[ \t]+\n", "\n", rendered)
        rendered = re.sub(r"\n{3,}", "\n\n", rendered)
        return rendered.strip() + "\n"

    def children(self, node: Node) -> str:
        return "".join(self.node(child) for child in node.children)

    def node(self, value: Node | str) -> str:
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value)
        node = value
        classes = node.classes
        if classes & {"highwire-figure-links", "sb-div", "cit-extra"} or node.tag in {"script", "style", "noscript"}:
            return ""
        if id(node) in self.figures:
            label, items = self.figures[id(node)]
            anchor = _safe_anchor(node.attrs.get("id", label))
            images = "\n".join(
                f"![{label}{f', part {index}' if len(items) > 1 else ''}]({item.output_path})"
                for index, item in enumerate(items, 1)
            )
            caption = _caption_text(node)
            return f'\n<a id="{anchor}"></a>\n\n{images}\n\n**{caption or label}**\n\n'
        if node.tag == "img":
            equation = self.equations.get(id(node))
            return f"![{equation.label}]({equation.output_path})" if equation else ""
        if node.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(node.tag[1])
            text = " ".join(self.children(node).split())
            anchor = _safe_anchor(node.attrs.get("id", ""))
            prefix = f'<a id="{anchor}"></a>\n\n' if anchor else ""
            return f"\n{prefix}{'#' * level} {text}\n\n"
        if node.tag == "p":
            return f"\n{self.children(node).strip()}\n\n"
        if node.tag in {"strong", "b"}:
            return f"**{self.children(node).strip()}**"
        if node.tag in {"em", "i"}:
            return f"*{self.children(node).strip()}*"
        if node.tag == "code":
            return f"`{self.children(node).strip()}`"
        if node.tag == "sup":
            return f"<sup>{html_escape(self.children(node).strip())}</sup>"
        if node.tag == "sub":
            return f"<sub>{html_escape(self.children(node).strip())}</sub>"
        if node.tag == "br":
            return "  \n"
        if node.tag == "a":
            label = self.children(node).strip() or node.attrs.get("title", "link")
            href = unescape(node.attrs.get("href", ""))
            if not href:
                return label
            if href.startswith("#"):
                anchor = _safe_anchor(href[1:])
                if anchor in self.supplement_ids and self.current_file == "paper.md":
                    href = f"supplement.md#{anchor}"
                elif anchor in self.main_ids and self.current_file == "supplement.md":
                    href = f"paper.md#{anchor}"
                else:
                    href = f"#{anchor}"
            elif not urllib.parse.urlsplit(href).scheme:
                href = _absolute_url(self.base_url, href)
            return f"[{label}]({href})"
        if node.tag == "li":
            return f"\n- {self.children(node).strip()}"
        if node.tag in {"ul", "ol"}:
            return f"\n{self.children(node).strip()}\n\n"
        if node.tag == "table":
            rows: list[list[str]] = []
            for row in node.walk():
                if row.tag != "tr":
                    continue
                cells = [
                    " ".join(cell.text().split())
                    for cell in row.children
                    if isinstance(cell, Node) and cell.tag in {"th", "td"}
                ]
                if cells:
                    rows.append(cells)
            if not rows:
                return f"\n{self.children(node)}\n"
            width = max(map(len, rows))
            rows = [row + [""] * (width - len(row)) for row in rows]
            lines = ["| " + " | ".join(row) + " |" for row in rows]
            lines.insert(1, "| " + " | ".join(["---"] * width) + " |")
            return "\n" + "\n".join(lines) + "\n\n"
        anchor = _safe_anchor(node.attrs.get("id", ""))
        prefix = f'<a id="{anchor}"></a>\n\n' if anchor and node.tag in {"div", "section"} else ""
        content = self.children(node)
        return f"{prefix}{content}"


def convert_article(source: str, spec: PaperSpec) -> ConvertedArticle:
    """Validate and convert one complete HighWire article HTML document."""
    root = _parse_html(source)
    title, authors = _validate_article_identity(root, spec)
    article = _find_first(root, lambda node: node.tag == "div" and {"article", "fulltext-view"}.issubset(node.classes))
    if article is None:
        raise SourceValidationError("complete article container div.article.fulltext-view was not found")
    supplement_index = next(
        (
            index
            for index, child in enumerate(article.children)
            if isinstance(child, Node) and _contains_heading(child, "Supplementary material")
        ),
        None,
    )
    if spec.has_supplement and supplement_index is None:
        raise SourceValidationError("expected embedded Supplementary material section was not found")
    if not spec.has_supplement and supplement_index is not None:
        raise SourceValidationError("unexpected Supplementary material section was found")
    if supplement_index is None:
        main_nodes = list(article.children)
        supplement_nodes: list[Node | str] = []
    else:
        acknowledgement_index = next(
            (
                index
                for index, child in enumerate(article.children[supplement_index + 1 :], supplement_index + 1)
                if isinstance(child, Node)
                and (
                    "ack" in child.classes
                    or _contains_heading(child, "Acknowledgements")
                    or _contains_heading(child, "Acknowledgments")
                )
            ),
            None,
        )
        if acknowledgement_index is None:
            raise SourceValidationError("Supplementary material boundary has no following Acknowledgements section")
        supplement_nodes = list(article.children[supplement_index:acknowledgement_index])
        main_nodes = list(article.children[:supplement_index]) + list(article.children[acknowledgement_index:])

    figures: dict[int, tuple[str, tuple[MediaItem, ...]]] = {}
    media: list[MediaItem] = []
    figure_nodes = [node for node in article.walk() if "type-figure" in node.classes]
    for figure_number, node in enumerate(figure_nodes, 1):
        urls = _figure_urls(node, spec.article_url)
        if not urls:
            raise SourceValidationError(f"figure {node.attrs.get('id', figure_number)} has no full-size JPEG source")
        label = _figure_label(node, f"Figure {figure_number}")
        items: list[MediaItem] = []
        for part, url in enumerate(urls, 1):
            suffix = f"-part-{part}" if len(urls) > 1 else ""
            item = MediaItem(url, f"figures/figure-{figure_number:02d}{suffix}.jpg", "figure", label)
            media.append(item)
            items.append(item)
        figures[id(node)] = (label, tuple(items))

    figure_descendants = {id(item) for figure in figure_nodes for item in figure.walk()}
    equation_nodes = [
        node
        for node in article.walk()
        if node.tag == "img"
        and id(node) not in figure_descendants
        and re.search(r"/graphic-\d+\.gif(?:\?|$)", node.attrs.get("src", ""), re.IGNORECASE)
    ]
    equations: dict[int, MediaItem] = {}
    for number, node in enumerate(equation_nodes, 1):
        item = MediaItem(
            _canonical_media_url(_absolute_url(spec.article_url, node.attrs["src"])),
            f"equations/equation-{number:02d}.gif",
            "equation",
            f"Equation {number}",
        )
        media.append(item)
        equations[id(node)] = item

    main_ids = {
        node.attrs["id"]
        for value in main_nodes
        if isinstance(value, Node)
        for node in value.walk()
        if node.attrs.get("id")
    }
    supplement_ids = {
        node.attrs["id"]
        for value in supplement_nodes
        if isinstance(value, Node)
        for node in value.walk()
        if node.attrs.get("id")
    }
    common = {
        "base_url": spec.article_url,
        "main_ids": main_ids,
        "supplement_ids": supplement_ids,
        "figures": figures,
        "equations": equations,
    }
    main_body = _MarkdownRenderer(current_file="paper.md", **common).nodes(main_nodes)
    byline = ", ".join(authors) if authors else "Authors as listed by the publisher"
    header = (
        f"# {title}\n\n"
        f"{byline}\n\n"
        f"Version: {spec.version}  \nDOI: [{spec.doi}](https://doi.org/{spec.doi})  \n"
        f"License: [{spec.license}](https://creativecommons.org/licenses/by/4.0/)\n\n"
    )
    supplement_markdown = None
    if supplement_nodes:
        supplement_markdown = _MarkdownRenderer(current_file="supplement.md", **common).nodes(supplement_nodes)
    return ConvertedArticle(header + main_body, supplement_markdown, tuple(media), title, authors)


def is_allowed_url(url: str) -> bool:
    """Return whether a URL satisfies the HTTPS source allowlist."""
    parts = urllib.parse.urlsplit(url)
    return (
        parts.scheme == "https"
        and (parts.hostname or "").casefold() in ALLOWED_HOSTS
        and not parts.username
        and not parts.password
    )


def validate_redirect_chain(initial_url: str, final_url: str) -> None:
    """Validate that source and final URLs are allowlisted."""
    if not is_allowed_url(initial_url) or not is_allowed_url(final_url):
        raise SourceValidationError(
            f"URL or redirect target is outside the HTTPS host allowlist: {initial_url!r} -> {final_url!r}"
        )


def validate_payload(data: bytes, content_type: str, kind: str, *, min_size: int, max_size: int) -> None:
    """Validate payload size, MIME type, and file signature."""
    if len(data) < min_size or len(data) > max_size:
        raise SourceValidationError(f"{kind} payload size {len(data)} is outside [{min_size}, {max_size}]")
    mime = content_type.split(";", 1)[0].strip().casefold()
    allowed_mime = {
        "html": {"text/html", "application/xhtml+xml"},
        "jpeg": {"image/jpeg", "image/jpg"},
        "gif": {"image/gif"},
        "xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/octet-stream"},
    }
    magic = {
        "html": data.lstrip().lower().startswith((b"<!doctype html", b"<html")),
        "jpeg": data.startswith(b"\xff\xd8\xff"),
        "gif": data.startswith((b"GIF87a", b"GIF89a")),
        "xlsx": data.startswith(b"PK\x03\x04"),
    }
    if mime not in allowed_mime[kind] or not magic[kind]:
        raise SourceValidationError(f"{kind} MIME/magic validation failed: {content_type!r}")


def with_retries(
    operation: Callable[[], bytes],
    *,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 3,
) -> bytes:
    """Run an operation with bounded retry delays."""
    for attempt in range(attempts):
        try:
            return operation()
        except RetryableFetchError as error:
            if attempt + 1 == attempts:
                raise
            delay = error.retry_after if error.retry_after is not None else min(5 * (2**attempt), 30)
            sleep(max(0.0, min(delay, 30.0)))
    raise AssertionError("unreachable")


def with_retryable_fallback(primary: Callable[[], bytes], secondary: Callable[[], bytes]) -> bytes:
    """Use a secondary transport only when the primary exhausts retryable failures."""
    try:
        return primary()
    except RetryableFetchError:
        return secondary()


def pace_request(
    previous_mark: float,
    *,
    now: float,
    minimum_interval: float,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Wait for the unsatisfied portion of a serial request interval."""
    target = previous_mark + max(0.0, minimum_interval)
    if now < target:
        sleep(target - now)
        return target
    return now


@dataclass(frozen=True)
class Downloaded:
    """Hold a downloaded payload and response metadata."""

    data: bytes
    content_type: str
    final_url: str


class DownloadCache:
    """Store downloaded payloads and metadata by source URL."""

    def __init__(self, root: Path):
        """Initialize a cache rooted at the given directory."""
        self.root = root

    def _paths(self, url: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.root / f"{digest}.bin", self.root / f"{digest}.json"

    def read(self, url: str) -> Downloaded:
        """Read a cached download for a source URL."""
        payload, metadata = self._paths(url)
        if not payload.is_file() or not metadata.is_file():
            raise OfflineCacheMiss(f"offline cache miss for {url}")
        record = json.loads(metadata.read_text(encoding="utf-8"))
        return Downloaded(payload.read_bytes(), record["content_type"], record["final_url"])

    def write(self, url: str, value: Downloaded) -> None:
        """Persist a downloaded response for a source URL."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload, metadata = self._paths(url)
        payload.write_bytes(value.data)
        metadata.write_text(
            json.dumps({"content_type": value.content_type, "final_url": value.final_url}, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class Fetcher:
    """One cookie-aware serial HTTP session with deterministic local caching."""

    def __init__(self, cache: DownloadCache, *, offline: bool, refresh: bool):
        """Initialize the fetcher with cache and network policy."""
        self.cache = cache
        self.offline = offline
        self.refresh = refresh
        self.minimum_interval = 1.0
        self._last_request_mark = 0.0
        jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def get(self, url: str) -> Downloaded:
        """Fetch a URL from cache or the publisher."""
        if not is_allowed_url(url):
            raise SourceValidationError(f"source URL is outside the HTTPS allowlist: {url}")
        if self.offline:
            return self.cache.read(url)
        if not self.refresh:
            try:
                return self.cache.read(url)
            except OfflineCacheMiss:
                pass

        def attempt() -> bytes:
            self._last_request_mark = pace_request(
                self._last_request_mark,
                now=time.monotonic(),
                minimum_interval=self.minimum_interval,
            )
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            try:
                with self.opener.open(request, timeout=45) as response:
                    final_url = response.geturl()
                    validate_redirect_chain(url, final_url)
                    data = response.read(MAX_DOWNLOAD_BYTES + 1)
                    if len(data) > MAX_DOWNLOAD_BYTES:
                        raise SourceValidationError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes: {url}")
                    value = Downloaded(data, response.headers.get_content_type(), final_url)
                    self.cache.write(url, value)
                    return data
            except urllib.error.HTTPError as error:
                retry_after = error.headers.get("Retry-After") if error.headers else None
                if error.code in {403, 429} or error.code >= 500:
                    try:
                        delay = float(retry_after) if retry_after else None
                    except ValueError:
                        delay = None
                    raise RetryableFetchError(f"HTTP {error.code} for {url}", retry_after=delay) from error
                raise HttpFetchError(error.code, url) from error
            except (TimeoutError, urllib.error.URLError) as error:
                raise RetryableFetchError(f"network timeout/error for {url}: {error}") from error

        with_retryable_fallback(lambda: with_retries(attempt), lambda: self._curl_download(url))
        return self.cache.read(url)

    def _curl_download(self, url: str) -> bytes:
        """Use curl's HTTP/2 stack when a publisher repeatedly rejects urllib."""
        executable = shutil.which("curl")
        if executable is None:
            raise RetryableFetchError("publisher retries exhausted and curl is unavailable")
        self.cache.root.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix="curl-", suffix=".download", dir=self.cache.root)
        os.close(descriptor)
        temporary = Path(temporary_name)
        cookie_jar = self.cache.root / "curl-cookies.txt"
        try:
            command = [
                executable,
                "--location",
                "--silent",
                "--show-error",
                "--max-redirs",
                "5",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--connect-timeout",
                "15",
                "--max-time",
                "120",
                "--user-agent",
                USER_AGENT,
                "--cookie",
                str(cookie_jar),
                "--cookie-jar",
                str(cookie_jar),
                "--output",
                str(temporary),
                "--write-out",
                "%{http_code}\t%{url_effective}\t%{content_type}",
                url,
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=135)
            if completed.returncode != 0:
                raise RetryableFetchError(f"curl transport failed for {url}: {completed.stderr.strip()}")
            fields = completed.stdout.rsplit("\t", 2)
            if len(fields) != 3 or not fields[0].isdigit():
                raise RetryableFetchError(f"curl returned invalid response metadata for {url}")
            status, final_url, content_type = int(fields[0]), fields[1], fields[2]
            validate_redirect_chain(url, final_url)
            if status in {403, 429} or status >= 500:
                raise RetryableFetchError(f"HTTP {status} for {url}")
            if status >= 400:
                raise HttpFetchError(status, url)
            data = temporary.read_bytes()
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise SourceValidationError(f"download exceeds {MAX_DOWNLOAD_BYTES} bytes: {url}")
            self.cache.write(url, Downloaded(data, content_type, final_url))
            return data
        finally:
            temporary.unlink(missing_ok=True)


def may_use_workbook_fallback(error: BaseException) -> bool:
    """Return whether an error permits the local workbook fallback."""
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, RetryableFetchError):
        return True
    return isinstance(error, HttpFetchError) and (error.status in {403, 429} or error.status >= 500)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix()
    )


def write_manifest(root: Path, metadata: dict[str, object]) -> str:
    """Write and return a deterministic asset manifest."""
    records = []
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        if relative == "MANIFEST.json":
            continue
        data = path.read_bytes()
        records.append({"path": relative, "sha256": _sha256(data), "size": len(data)})
    payload = {"schema_version": 1, "asset": metadata, "files": records}
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    manifest = root / "MANIFEST.json"
    if not manifest.is_file() or manifest.read_text(encoding="utf-8") != rendered:
        manifest.write_text(rendered, encoding="utf-8")
    return rendered


_MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def check_asset_tree(root: Path) -> list[str]:
    """Return integrity and local-link errors for an asset tree."""
    errors: list[str] = []
    manifest_path = root / "MANIFEST.json"
    if not manifest_path.is_file():
        return ["missing MANIFEST.json"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return [f"invalid MANIFEST.json: {error}"]
    expected = {item["path"]: item for item in manifest.get("files", [])}
    actual = {path.relative_to(root).as_posix(): path for path in _files(root) if path != manifest_path}
    for missing in sorted(expected.keys() - actual.keys()):
        errors.append(f"missing file: {missing}")
    for extra in sorted(actual.keys() - expected.keys()):
        errors.append(f"extra file: {extra}")
    for relative in sorted(expected.keys() & actual.keys()):
        data = actual[relative].read_bytes()
        record = expected[relative]
        if len(data) != record["size"] or _sha256(data) != record["sha256"]:
            errors.append(f"tampered file: {relative}")
    for relative, path in actual.items():
        if path.suffix.casefold() != ".md":
            continue
        source = path.read_text(encoding="utf-8")
        for match in _MARKDOWN_LINK.finditer(source):
            target = match.group(1).strip().split("#", 1)[0]
            if not target or target.startswith(("https://", "http://", "mailto:")):
                continue
            resolved = (path.parent / urllib.parse.unquote(target)).resolve()
            try:
                resolved.relative_to(root.resolve())
            except ValueError:
                errors.append(f"local link escapes asset root: {relative}: {target}")
                continue
            if not resolved.is_file():
                errors.append(f"broken local link: {relative}: {target}")
    return errors


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _files(root):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_install(staged: Path, destination: Path, *, validate: Callable[[Path], None]) -> bool:
    """Validate and atomically install a staged asset tree."""
    validate(staged)
    if destination.is_dir() and _tree_digest(staged) == _tree_digest(destination):
        shutil.rmtree(staged)
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup-{os.getpid()}")
    if backup.exists():
        shutil.rmtree(backup)
    had_destination = destination.exists()
    try:
        if had_destination:
            os.replace(destination, backup)
        os.replace(staged, destination)
    except BaseException:
        if destination.exists() and not had_destination:
            shutil.rmtree(destination)
        if backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)
    return True


def enforce_update_policy(root: Path, proposed: dict[str, object], *, update: bool) -> None:
    """Reject unapproved drift from an existing asset manifest."""
    manifest = root / "MANIFEST.json"
    if not manifest.is_file():
        return
    existing = json.loads(manifest.read_text(encoding="utf-8")).get("asset", {})
    drift = {
        key: (existing.get(key), proposed.get(key))
        for key in ("converter_version", "converter_sha256", "source_sha256")
        if existing.get(key) != proposed.get(key)
    }
    if drift and not update:
        raise SourceDriftError(f"source/converter drift requires --update: {drift}")


@dataclass(frozen=True)
class XlsxSheet:
    """Hold a worksheet name and normalized rows."""

    name: str
    rows: tuple[tuple[str, ...], ...]


_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _column_number(reference: str) -> int:
    letters = re.match(r"[A-Za-z]+", reference)
    if not letters:
        raise WorkbookValidationError(f"invalid cell reference: {reference!r}")
    result = 0
    for letter in letters.group(0).upper():
        result = result * 26 + ord(letter) - 64
    return result


def read_xlsx_sheet(workbook: Path, sheet_name: str) -> XlsxSheet:
    """Read and validate a named worksheet from an XLSX workbook."""
    try:
        archive = zipfile.ZipFile(workbook)
    except (OSError, zipfile.BadZipFile) as error:
        raise WorkbookValidationError(f"invalid XLSX: {workbook}: {error}") from error
    with archive:
        try:
            workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
            rel_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        except (KeyError, ET.ParseError) as error:
            raise WorkbookValidationError(f"invalid XLSX workbook metadata: {error}") from error
        sheets = workbook_root.findall(f".//{{{_SHEET_NS}}}sheet")
        if len(sheets) < 2 or sheets[1].attrib.get("name") != "Sheet 1":
            raise WorkbookValidationError("the workbook's second sheet must be named 'Sheet 1'")
        selected = next((item for item in sheets if item.attrib.get("name") == sheet_name), None)
        if selected is None:
            raise WorkbookValidationError(f"sheet not found: {sheet_name!r}")
        relationship_id = selected.attrib.get(f"{{{_REL_NS}}}id")
        targets = {
            item.attrib.get("Id"): item.attrib.get("Target")
            for item in rel_root.findall(f"{{{_PKG_REL_NS}}}Relationship")
        }
        target = targets.get(relationship_id)
        if not target:
            raise WorkbookValidationError(f"worksheet relationship is missing for {sheet_name!r}")
        sheet_path = "xl/" + target.lstrip("/")
        try:
            sheet_root = ET.fromstring(archive.read(sheet_path))
        except (KeyError, ET.ParseError) as error:
            raise WorkbookValidationError(f"invalid worksheet XML: {error}") from error
        shared: list[str] = []
        try:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        except KeyError:
            pass
        else:
            for item in shared_root.findall(f"{{{_SHEET_NS}}}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{_SHEET_NS}}}t")))

        parsed: list[dict[int, str]] = []
        max_column = 0
        for row in sheet_root.findall(f".//{{{_SHEET_NS}}}row"):
            values: dict[int, str] = {}
            for cell in row.findall(f"{{{_SHEET_NS}}}c"):
                column = _column_number(cell.attrib.get("r", ""))
                max_column = max(max_column, column)
                formula = cell.find(f"{{{_SHEET_NS}}}f")
                value = cell.find(f"{{{_SHEET_NS}}}v")
                if formula is not None and (value is None or value.text is None or value.text == ""):
                    raise WorkbookValidationError(f"formula cell {cell.attrib.get('r')} has no cached value")
                cell_type = cell.attrib.get("t")
                if cell_type == "inlineStr":
                    text = "".join(item.text or "" for item in cell.iter(f"{{{_SHEET_NS}}}t"))
                elif value is None or value.text is None:
                    text = ""
                elif cell_type == "s":
                    try:
                        text = shared[int(value.text)]
                    except (ValueError, IndexError) as error:
                        raise WorkbookValidationError(f"invalid shared-string cell {cell.attrib.get('r')}") from error
                elif cell_type == "b":
                    text = "TRUE" if value.text == "1" else "FALSE"
                else:
                    text = value.text
                values[column] = text.replace("\r\n", "\n").replace("\r", "\n")
            parsed.append(values)
        rows = tuple(tuple(row.get(column, "") for column in range(1, max_column + 1)) for row in parsed)
        return XlsxSheet(sheet_name, rows)


def _sheet1_tsv(workbook: Path, *, expected_rows: int, expected_columns: int) -> tuple[bytes, dict[str, object]]:
    sheet = read_xlsx_sheet(workbook, "Sheet 1")
    if len(sheet.rows) != expected_rows + 1:
        raise WorkbookValidationError(
            f"Sheet 1 expected {expected_rows} data rows, found {max(0, len(sheet.rows) - 1)}"
        )
    if not sheet.rows or any(len(row) != expected_columns for row in sheet.rows):
        widths = sorted({len(row) for row in sheet.rows})
        raise WorkbookValidationError(f"Sheet 1 expected {expected_columns} columns, found widths {widths}")
    identifiers = [row[0] for row in sheet.rows[1:]]
    if not all(identifiers) or len(set(identifiers)) != expected_rows:
        raise WorkbookValidationError("Sheet 1 first-column design identifiers are missing or not unique")
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerows(sheet.rows)
    data = buffer.getvalue().encode("utf-8")
    return data, {
        "path": "sheet-1.tsv",
        "rows": expected_rows,
        "columns": expected_columns,
        "unique_ids": len(set(identifiers)),
        "sha256": _sha256(data),
    }


def extract_sheet1(
    workbook: Path,
    output_dir: Path,
    *,
    check: bool,
    expected_rows: int = 302,
    expected_columns: int = 33,
) -> dict[str, object]:
    """Extract or verify the pinned workbook sheet as TSV."""
    data, report = _sheet1_tsv(workbook, expected_rows=expected_rows, expected_columns=expected_columns)
    output = output_dir / "sheet-1.tsv"
    if check:
        if not output.is_file() or output.read_bytes() != data:
            raise WorkbookValidationError(f"extracted TSV is absent or stale: {output}")
        return report
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tsv.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, output)
    return report


def _attribution(spec: PaperSpec, source_url: str) -> str:
    return (
        f"# Attribution\n\n"
        f"- Work: *{spec.title}*\n"
        f"- DOI: [{spec.doi}](https://doi.org/{spec.doi})\n"
        f"- Source version: {spec.version}\n"
        f"- Source: [{source_url}]({source_url})\n"
        f"- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)\n"
        f"- Transformation: publisher HTML converted to Markdown; full-size publisher figure/equation files retained without resizing.\n"
    )


def _load_source_map(value: str | None) -> dict[str, dict[str, str]]:
    if not value:
        return {}
    candidate = Path(value)
    text = candidate.read_text(encoding="utf-8") if candidate.is_file() else value
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise SourceValidationError("--source-map must be a JSON object")
    return payload


def _kind_for_media(item: MediaItem) -> str:
    return "jpeg" if item.kind == "figure" else "gif"


def _validate_staged(root: Path, spec: PaperSpec) -> None:
    errors = check_asset_tree(root)
    if errors:
        raise SourceValidationError("asset tree validation failed: " + "; ".join(errors))
    figures = list((root / "figures").glob("*.jpg")) if (root / "figures").is_dir() else []
    equations = list((root / "equations").glob("*.gif")) if (root / "equations").is_dir() else []
    if len(figures) != spec.expected_figure_files:
        raise SourceValidationError(
            f"{spec.slug}: expected {spec.expected_figure_files} figure files, found {len(figures)}"
        )
    if len(equations) != spec.expected_equations:
        raise SourceValidationError(
            f"{spec.slug}: expected {spec.expected_equations} equation files, found {len(equations)}"
        )
    if spec.has_supplement != (root / "supplement.md").is_file():
        raise SourceValidationError(f"{spec.slug}: supplement presence does not match pinned specification")
    if not spec.has_supplement and not (root / "NO_SUPPLEMENT.md").is_file():
        raise SourceValidationError(f"{spec.slug}: missing NO_SUPPLEMENT.md")
    if spec.workbook_url:
        workbook = root / "supplementary" / "media-1.xlsx"
        data = workbook.read_bytes() if workbook.is_file() else b""
        if len(data) != KING_WORKBOOK_SIZE or _sha256(data) != KING_WORKBOOK_SHA256:
            raise SourceValidationError("official media-1.xlsx hash/size does not match the pinned v1 workbook")
        sheet = read_xlsx_sheet(workbook, "Sheet 1")
        if len(sheet.rows) != 303 or any(len(row) != 33 for row in sheet.rows):
            raise SourceValidationError("official media-1.xlsx Sheet 1 must contain 302 data rows and 33 columns")


def sync_paper(
    spec: PaperSpec,
    *,
    output_root: Path,
    fetcher: Fetcher,
    source_overrides: dict[str, str],
    fallback_media_1: Path | None,
    update: bool,
) -> dict[str, object]:
    """Synchronize one paper's validated literature asset tree."""
    article_url = source_overrides.get("article_url", spec.article_url)
    article = fetcher.get(article_url)
    validate_payload(article.data, article.content_type, "html", min_size=1_000, max_size=10_000_000)
    converted = convert_article(article.data.decode("utf-8"), spec)
    if len([item for item in converted.media if item.kind == "figure"]) != spec.expected_figure_files:
        raise SourceValidationError(f"{spec.slug}: converted figure-file count differs from pinned source")
    if len([item for item in converted.media if item.kind == "equation"]) != spec.expected_equations:
        raise SourceValidationError(f"{spec.slug}: converted equation count differs from pinned source")
    source_sha = _sha256(article.data)
    proposed = {
        "converter_version": CONVERTER_VERSION,
        "converter_sha256": _sha256(Path(__file__).read_bytes()),
        "source_sha256": source_sha,
    }
    destination = output_root / spec.slug
    enforce_update_policy(destination, proposed, update=update)
    output_root.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{spec.slug}.stage-", dir=output_root))
    source_records: list[dict[str, object]] = [
        {"role": "article-html", "url": article.final_url, "sha256": source_sha, "size": len(article.data)}
    ]
    try:
        (staged / "paper.md").write_text(converted.paper_markdown, encoding="utf-8", newline="\n")
        if converted.supplement_markdown is not None:
            (staged / "supplement.md").write_text(converted.supplement_markdown, encoding="utf-8", newline="\n")
        else:
            (staged / "NO_SUPPLEMENT.md").write_text(
                "# No supplementary material\n\nThe publisher's pinned v1 record lists no supplementary files or embedded supplementary section.\n",
                encoding="utf-8",
                newline="\n",
            )
        (staged / "ATTRIBUTION.md").write_text(_attribution(spec, article.final_url), encoding="utf-8", newline="\n")
        for item in converted.media:
            downloaded = fetcher.get(item.source_url)
            kind = _kind_for_media(item)
            validate_payload(downloaded.data, downloaded.content_type, kind, min_size=20, max_size=MAX_DOWNLOAD_BYTES)
            path = staged / item.output_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(downloaded.data)
            source_records.append(
                {
                    "role": item.kind,
                    "url": downloaded.final_url,
                    "output": item.output_path,
                    "sha256": _sha256(downloaded.data),
                    "size": len(downloaded.data),
                }
            )
        if spec.workbook_url:
            workbook_url = source_overrides.get("workbook_url", spec.workbook_url)
            try:
                workbook = fetcher.get(workbook_url)
                validate_payload(workbook.data, workbook.content_type, "xlsx", min_size=100_000, max_size=20_000_000)
                workbook_data = workbook.data
                workbook_source = workbook.final_url
            except BaseException as error:
                if fallback_media_1 is None or not may_use_workbook_fallback(error):
                    raise
                workbook_data = fallback_media_1.read_bytes()
                workbook_source = f"explicit-local-fallback:{fallback_media_1.name}"
            if len(workbook_data) != KING_WORKBOOK_SIZE or _sha256(workbook_data) != KING_WORKBOOK_SHA256:
                raise SourceValidationError("media-1.xlsx differs from the official pinned v1 hash/size")
            workbook_path = staged / "supplementary" / "media-1.xlsx"
            workbook_path.parent.mkdir(parents=True, exist_ok=True)
            workbook_path.write_bytes(workbook_data)
            source_records.append(
                {
                    "role": "supplementary-workbook",
                    "url": workbook_source,
                    "output": "supplementary/media-1.xlsx",
                    "sha256": KING_WORKBOOK_SHA256,
                    "size": KING_WORKBOOK_SIZE,
                }
            )
        metadata: dict[str, object] = {
            "slug": spec.slug,
            "title": spec.title,
            "doi": spec.doi,
            "version": spec.version,
            "license": spec.license,
            **proposed,
            "sources": source_records,
        }
        write_manifest(staged, metadata)
        changed = atomic_install(staged, destination, validate=lambda root: _validate_staged(root, spec))
        return {"paper": spec.slug, "changed": changed, "path": str(destination), "source_sha256": source_sha}
    except BaseException:
        if staged.exists():
            shutil.rmtree(staged)
        raise


def _select_papers(values: Sequence[str] | None) -> list[PaperSpec]:
    names = list(values or PAPERS.keys())
    unknown = sorted(set(names) - PAPERS.keys())
    if unknown:
        raise SourceValidationError(f"unknown paper slug(s): {', '.join(unknown)}")
    return [PAPERS[name] for name in names]


def _default_output_root() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "literature"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser(
        "sync", help="download, convert, validate, and atomically install literature assets"
    )
    sync_parser.add_argument(
        "--paper", action="append", choices=sorted(PAPERS), help="paper slug; repeat; default: all"
    )
    sync_parser.add_argument("--output-root", type=Path, default=_default_output_root())
    sync_parser.add_argument("--fallback-media-1", type=Path, help="explicit verified local workbook fallback")
    sync_parser.add_argument("--source-map", help="JSON string or JSON file with per-paper official URL overrides")
    sync_parser.add_argument("--offline", action="store_true", help="require every source to be present in the cache")
    sync_parser.add_argument("--update", action="store_true", help="accept pinned source or converter drift")
    sync_parser.add_argument("--json", action="store_true")

    check_parser = subparsers.add_parser("check", help="verify manifests, hashes, counts, workbook, and local links")
    check_parser.add_argument(
        "--paper", action="append", choices=sorted(PAPERS), help="paper slug; repeat; default: all"
    )
    check_parser.add_argument("--output-root", type=Path, default=_default_output_root())
    check_parser.add_argument("--json", action="store_true")

    extract = subparsers.add_parser("extract-sheet1", help="extract the official workbook's Sheet 1 as LF-only TSV")
    extract.add_argument("--workbook", required=True, type=Path)
    extract.add_argument("--output-dir", required=True, type=Path)
    extract.add_argument("--check", action="store_true", help="verify an existing extraction without writing")
    extract.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the literature asset command-line interface."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "sync":
            source_map = _load_source_map(args.source_map)
            cache_root = Path(os.environ.get("XDG_CACHE_HOME", tempfile.gettempdir())) / "phage-design-literature-v1"
            fetcher = Fetcher(DownloadCache(cache_root), offline=args.offline, refresh=args.update)
            reports = [
                sync_paper(
                    spec,
                    output_root=args.output_root,
                    fetcher=fetcher,
                    source_overrides=source_map.get(spec.slug, {}),
                    fallback_media_1=args.fallback_media_1,
                    update=args.update,
                )
                for spec in _select_papers(args.paper)
            ]
            output: object = {"ok": True, "papers": reports}
        elif args.command == "check":
            reports = []
            for spec in _select_papers(args.paper):
                root = args.output_root / spec.slug
                errors = check_asset_tree(root)
                if not errors:
                    try:
                        _validate_staged(root, spec)
                    except LiteratureAssetError as error:
                        errors.append(str(error))
                reports.append({"paper": spec.slug, "ok": not errors, "errors": errors})
            output = {"ok": all(item["ok"] for item in reports), "papers": reports}
            if not output["ok"]:
                raise SourceValidationError(json.dumps(output, sort_keys=True))
        else:
            output = {"ok": True, **extract_sheet1(args.workbook, args.output_dir, check=args.check)}
        if getattr(args, "json", False):
            print(json.dumps(output, indent=2, sort_keys=True))
        else:
            print("OK")
            if isinstance(output, dict):
                for item in output.get("papers", []):
                    print(f"- {item.get('paper')}: {'ok' if item.get('ok', True) else 'failed'}")
        return 0
    except (LiteratureAssetError, OSError, json.JSONDecodeError) as error:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(error)}, indent=2, sort_keys=True))
        else:
            print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
