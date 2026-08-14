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

"""Tests for the portable literature-asset synchronizer.

The suite uses only synthetic HTML/XLSX fixtures.  Network synchronization is
an explicit integration step, not part of the unit test run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "sync_literature_assets.py"
SPEC = importlib.util.spec_from_file_location("sync_literature_assets", SCRIPT)
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync
SPEC.loader.exec_module(sync)


def _article_html(
    *, doi: str, version: str = "v1", license_url: str = "https://creativecommons.org/licenses/by/4.0/"
) -> str:
    """Return a compact HighWire-like article with split and media edge cases."""
    return f"""<!doctype html>
<html><head>
<meta name="citation_doi" content="{doi}">
<meta name="citation_title" content="Synthetic title">
<meta name="citation_author" content="Ada Example">
<meta name="citation_publication_date" content="2026/06/19">
<meta name="citation_fulltext_html_url" content="https://www.biorxiv.org/content/{doi}{version}.full">
<link rel="license" href="{license_url}">
</head><body>
<div class="article fulltext-view">
 <span class="highwire-journal-article-marker-start"></span>
 <div class="section abstract" id="abstract-1"><h2>Abstract</h2><p id="p-1">Alpha <em>beta</em>.</p></div>
 <div class="section" id="sec-1"><h2>Results</h2>
  <p>See <a href="#F1">Figure 1</a> and <a href="#F2">Figure S1</a>.</p>
  <div id="F1" class="fig type-figure"><div class="highwire-figure">
   <a class="highwire-figure-link highwire-figure-link-newtab" href="https://www.biorxiv.org/media/F1.large.jpg">Open</a>
  </div><div class="fig-caption"><span class="fig-label">Figure 1</span><span class="caption-title">Main caption.</span></div></div>
  <p>Inline equation <img class="inline-graphic" src="https://www.biorxiv.org/media/graphic-4.gif" alt="equation"> follows.</p>
 </div>
 <div class="section" id="sec-supp"><h2>Supplementary material</h2>
  <div id="F2" class="fig type-figure"><div class="highwire-figure">
   <a class="highwire-figure-link highwire-figure-link-newtab" href="https://www.biorxiv.org/media/F2/graphic-2.large.jpg">one</a>
   <a class="highwire-figure-link highwire-figure-link-newtab" href="https://www.biorxiv.org/media/F2/graphic-3.large.jpg">two</a>
  </div><div class="fig-caption"><span class="fig-label">Figure S1</span><span class="caption-title">Two parts in order.</span></div></div>
  <div class="table" id="T1"><div class="table-caption"><span class="table-label">Table S1</span><span class="caption-title">Candidate table; source is File S1.</span></div></div>
 </div>
 <div class="section" id="sec-files"><h2><strong>D.</strong> Supplementary files</h2><p>File S1 workbook.</p></div>
 <div class="section ack" id="ack-1"><h2>Acknowledgements</h2><p>Thanks.</p></div>
 <div class="section ref-list" id="ref-list-1"><h2>References</h2><ol><li id="ref-1">A reference.</li></ol></div>
 <span class="highwire-journal-article-marker-end"></span>
</div></body></html>"""


def _make_xlsx(
    path: Path,
    *,
    rows: int = 3,
    columns: int = 33,
    formula_values: bool = True,
    sheet1_second: bool = True,
) -> None:
    """Build a minimal XLSX with a configurable Sheet 1 position."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    sheets = (
        '<sheet name="Contents" sheetId="1" r:id="rId1"/><sheet name="Sheet 1" sheetId="2" r:id="rId2"/>'
        if sheet1_second
        else ('<sheet name="Sheet 1" sheetId="1" r:id="rId2"/><sheet name="Contents" sheetId="2" r:id="rId1"/>')
    )
    workbook = f"""<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>
{sheets}
</sheets></workbook>"""
    rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>"""
    contents = f"""<?xml version="1.0"?><worksheet xmlns="{main}"><sheetData/></worksheet>"""

    def col_name(number: int) -> str:
        out = ""
        while number:
            number, rem = divmod(number - 1, 26)
            out = chr(65 + rem) + out
        return out

    xml_rows: list[str] = []
    for row in range(1, rows + 2):
        cells: list[str] = []
        for column in range(1, columns + 1):
            ref = f"{col_name(column)}{row}"
            value = (
                f"column-{column}" if row == 1 else (f"design-{row - 1}" if column == 1 else f"r{row - 1}c{column}")
            )
            if column == 31 and row > 1:
                cached = f"{0.1 * (row - 1):.1f}" if formula_values else ""
                cells.append(f'<c r="{ref}"><f>1/10</f><v>{cached}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>')
        xml_rows.append(f'<row r="{row}">{"".join(cells)}</row>')
    sheet = f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="{main}"><dimension ref="A1:AG{rows + 1}"/><sheetData>{"".join(xml_rows)}</sheetData></worksheet>"""
    content_types = """<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", contents)
        archive.writestr("xl/worksheets/sheet2.xml", sheet)


def test_help_lists_all_subcommands() -> None:
    completed = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, text=True, capture_output=True)
    assert "sync" in completed.stdout
    assert "check" in completed.stdout
    assert "extract-sheet1" in completed.stdout


def _article_spec() -> sync.PaperSpec:
    return sync.PaperSpec(
        slug="synthetic",
        title="Synthetic title",
        doi="10.1234/synthetic",
        version="v1",
        license="CC-BY-4.0",
        article_url="https://www.biorxiv.org/content/10.1234/syntheticv1.full",
        supplement_url=None,
        expected_figures=2,
        expected_figure_files=3,
        expected_equations=1,
        has_supplement=True,
    )


def test_supplement_boundary_keeps_acknowledgements_and_references_in_main() -> None:
    converted = sync.convert_article(_article_html(doi=_article_spec().doi), _article_spec())
    assert "# Supplementary material" in converted.supplement_markdown
    assert "Acknowledgements" not in converted.supplement_markdown
    assert "Supplementary files" in converted.supplement_markdown
    assert "Supplementary files" not in converted.paper_markdown
    assert "Acknowledgements" in converted.paper_markdown
    assert "References" in converted.paper_markdown


def test_multi_image_order_and_cross_file_links_are_preserved() -> None:
    converted = sync.convert_article(_article_html(doi=_article_spec().doi), _article_spec())
    assert [item.output_path for item in converted.media if item.kind == "figure"] == [
        "figures/figure-01.jpg",
        "figures/figure-02-part-1.jpg",
        "figures/figure-02-part-2.jpg",
    ]
    assert "[Figure S1](supplement.md#f2)" in converted.paper_markdown
    assert "![Figure S1, part 1](figures/figure-02-part-1.jpg)" in converted.supplement_markdown
    assert converted.supplement_markdown.index("part 1") < converted.supplement_markdown.index("part 2")


def test_equation_image_survives_conversion() -> None:
    converted = sync.convert_article(_article_html(doi=_article_spec().doi), _article_spec())
    assert "![Equation 1](equations/equation-01.gif)" in converted.paper_markdown
    equations = [item for item in converted.media if item.kind == "equation"]
    assert 1 == len(equations)


def test_lazy_equation_source_is_deduplicated_and_rendered() -> None:
    html = _article_html(doi=_article_spec().doi).replace(
        '<img class="inline-graphic" src="https://www.biorxiv.org/media/graphic-4.gif" alt="equation">',
        '<img class="lazyload" src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yw=" '
        'data-src="https://www.biorxiv.org/media/graphic-4.gif" alt="equation">'
        '<noscript><img src="https://www.biorxiv.org/media/graphic-4.gif" alt="equation"></noscript>',
    )

    converted = sync.convert_article(html, _article_spec())

    assert "![Equation 1](equations/equation-01.gif)" in converted.paper_markdown
    assert 1 == len([item for item in converted.media if item.kind == "equation"])


def test_configured_source_equation_is_rendered_as_tex() -> None:
    spec = sync.PaperSpec(**{**_article_spec().__dict__, "equation_tex": (r"p_{\mathrm{eff}} = 1 - 0.053^{1/k}",)})

    converted = sync.convert_article(_article_html(doi=spec.doi), spec)

    assert "$$\np_{\\mathrm{eff}} = 1 - 0.053^{1/k}\n$$" in converted.paper_markdown


def test_reference_anchors_and_split_external_urls_survive_conversion() -> None:
    html = _article_html(doi=_article_spec().doi)
    html = html.replace(
        "<p>See ",
        '<p>Citation <a href="#ref-1">one</a>. Tools '
        '<a href="http://plan">http://plan</a> '
        '<a href="https://notate.barricklab.org">notate.barricklab.org</a>/. '
        '<a href="https://gene">https://gene</a> mark.example.org/tool.cgi and '
        '<a href="https://example.org/l">https://example.org/l</a> ogo.cgi. '
        "See ",
    )
    html = html.replace('<li id="ref-1">', '<li><a id="ref-1"></a>')

    converted = sync.convert_article(html, _article_spec())

    assert "[one](#ref-1)" in converted.paper_markdown
    assert '<a id="ref-1"></a>' in converted.paper_markdown
    assert "[http://plannotate.barricklab.org/](http://plannotate.barricklab.org/)" in converted.paper_markdown
    assert "[http://plan](http://plan)" not in converted.paper_markdown
    assert "[https://genemark.example.org/tool.cgi](https://genemark.example.org/tool.cgi)" in converted.paper_markdown
    assert "[https://example.org/logo.cgi](https://example.org/logo.cgi)" in converted.paper_markdown


def test_biorxiv_static_equation_url_uses_official_content_route() -> None:
    html = _article_html(doi=_article_spec().doi).replace(
        "https://www.biorxiv.org/media/graphic-4.gif",
        "https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2026/06/19/example/embed/graphic-4.gif",
    )
    converted = sync.convert_article(html, _article_spec())
    equation = next(item for item in converted.media if item.kind == "equation")
    assert (
        "https://www.biorxiv.org/content/biorxiv/early/2026/06/19/example/embed/graphic-4.gif" == equation.source_url
    )


def test_table_s1_caption_and_workbook_source_survive() -> None:
    converted = sync.convert_article(_article_html(doi=_article_spec().doi), _article_spec())
    assert "Table S1" in converted.supplement_markdown
    assert "File S1" in converted.supplement_markdown


def test_doi_version_and_license_mismatches_are_rejected() -> None:
    cases = [
        _article_html(doi="10.1234/wrong"),
        _article_html(doi=_article_spec().doi, version="v2"),
        _article_html(doi=_article_spec().doi, license_url="https://example.org/proprietary"),
    ]
    for html in cases:
        with pytest.raises(sync.SourceValidationError):
            sync.convert_article(html, _article_spec())


def test_publisher_citation_full_html_url_proves_version() -> None:
    html = _article_html(doi=_article_spec().doi).replace("citation_fulltext_html_url", "citation_full_html_url")
    converted = sync.convert_article(html, _article_spec())
    assert "Synthetic title" in converted.paper_markdown


def test_publisher_dc_rights_proves_cc_by_license() -> None:
    html = _article_html(doi=_article_spec().doi).replace(
        '<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">',
        '<meta name="DC.Rights" content="Creative Commons License (Attribution 4.0 International), CC BY 4.0, http://creativecommons.org/licenses/by/4.0/">',
    )
    converted = sync.convert_article(html, _article_spec())
    assert "CC-BY-4.0" in converted.paper_markdown


def test_user_agent_is_publisher_compatible_and_identifies_tool() -> None:
    assert sync.USER_AGENT.startswith("Mozilla/5.0")
    assert "phage-design-literature-sync" in sync.USER_AGENT


def test_allowlist_checks_initial_and_redirect_urls() -> None:
    assert sync.is_allowed_url("https://www.biorxiv.org/content/a")
    assert not sync.is_allowed_url("http://www.biorxiv.org/content/a")
    assert not sync.is_allowed_url("https://evil.example/content/a")
    with pytest.raises(sync.SourceValidationError):
        sync.validate_redirect_chain(
            "https://www.biorxiv.org/content/a",
            "https://evil.example/payload",
        )


def test_redirect_handler_rejects_disallowed_hop_before_request() -> None:
    request = sync.urllib.request.Request("https://www.biorxiv.org/content/a")

    with pytest.raises(sync.SourceValidationError):
        sync._AllowlistRedirectHandler().redirect_request(
            request,
            None,
            302,
            "Found",
            {},
            "https://evil.example/intermediate",
        )


def test_curl_fallback_rejects_disallowed_redirect_before_following() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fetcher = sync.Fetcher(sync.DownloadCache(Path(tmp)), offline=False, refresh=True)
        calls: list[str] = []

        def run(command, **_kwargs):
            calls.append(command[-1])
            header_path = Path(command[command.index("--dump-header") + 1])
            output_path = Path(command[command.index("--output") + 1])
            header_path.write_text(
                "HTTP/2 302 Found\r\nlocation: https://evil.example/intermediate\r\n\r\n",
                encoding="utf-8",
            )
            output_path.write_bytes(b"redirect")
            return sync.subprocess.CompletedProcess(
                command,
                0,
                "302\thttps://www.biorxiv.org/content/a\ttext/html",
                "",
            )

        with (
            mock.patch.object(sync.shutil, "which", return_value="/usr/bin/curl"),
            mock.patch.object(sync.subprocess, "run", side_effect=run),
            pytest.raises(sync.SourceValidationError),
        ):
            fetcher._curl_download("https://www.biorxiv.org/content/a")

        assert ["https://www.biorxiv.org/content/a"] == calls


def test_curl_timeout_is_reported_as_retryable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fetcher = sync.Fetcher(sync.DownloadCache(Path(tmp)), offline=False, refresh=True)
        timeout = subprocess.TimeoutExpired(["curl"], 135)
        with (
            mock.patch.object(sync.shutil, "which", return_value="/usr/bin/curl"),
            mock.patch.object(sync.subprocess, "run", side_effect=timeout),
            pytest.raises(sync.RetryableFetchError, match="curl timed out"),
        ):
            fetcher._curl_download("https://www.biorxiv.org/content/a")


def test_curl_caps_download_size_and_reports_curl_limit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        fetcher = sync.Fetcher(sync.DownloadCache(Path(tmp)), offline=False, refresh=True)

        def run(command, **_kwargs):
            assert str(sync.MAX_DOWNLOAD_BYTES) == command[command.index("--max-filesize") + 1]
            return sync.subprocess.CompletedProcess(command, 63, "", "maximum file size exceeded")

        with (
            mock.patch.object(sync.shutil, "which", return_value="/usr/bin/curl"),
            mock.patch.object(sync.subprocess, "run", side_effect=run),
            pytest.raises(sync.SourceValidationError, match="exceeds"),
        ):
            fetcher._curl_download("https://www.biorxiv.org/content/a")


def test_mime_magic_and_size_validation() -> None:
    sync.validate_payload(b"\xff\xd8\xffjpeg", "image/jpeg", "jpeg", min_size=4, max_size=100)
    sync.validate_payload(b"GIF89a-data", "image/gif", "gif", min_size=4, max_size=100)
    sync.validate_payload(
        b"PK\x03\x04xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xlsx",
        min_size=4,
        max_size=100,
    )
    for data, mime, kind in [
        (b"<html>blocked</html>", "image/jpeg", "jpeg"),
        (b"\xff\xd8\xffjpeg", "text/html", "jpeg"),
        (b"GIF89a-data", "image/gif", "jpeg"),
    ]:
        with pytest.raises(sync.SourceValidationError):
            sync.validate_payload(data, mime, kind, min_size=4, max_size=100)
    with pytest.raises(sync.SourceValidationError):
        sync.validate_payload(b"GIF89a", "image/gif", "gif", min_size=100, max_size=200)


def test_retry_honors_retry_after_and_is_bounded() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def operation() -> bytes:
        attempts.append(1)
        if len(attempts) < 3:
            raise sync.RetryableFetchError("busy", retry_after=0.25)
        return b"ok"

    assert b"ok" == sync.with_retries(operation, sleep=sleeps.append, attempts=3)
    assert 3 == len(attempts)
    assert [0.25, 0.25] == sleeps


def test_retryable_primary_failure_uses_bounded_secondary_transport() -> None:
    calls: list[str] = []

    def primary() -> bytes:
        calls.append("primary")
        raise sync.RetryableFetchError("publisher rejected HTTP/1.1")

    def secondary() -> bytes:
        calls.append("secondary")
        return b"official payload"

    result = sync.with_retryable_fallback(primary, secondary)
    assert b"official payload" == result
    assert ["primary", "secondary"] == calls
    with pytest.raises(sync.SourceValidationError):
        sync.with_retryable_fallback(lambda: (_ for _ in ()).throw(sync.SourceValidationError("corrupt")), secondary)


def test_request_pacing_waits_only_for_remaining_interval() -> None:
    sleeps: list[float] = []
    next_mark = sync.pace_request(10.0, now=10.25, minimum_interval=1.0, sleep=sleeps.append)
    assert [0.75] == sleeps
    assert 11.0 == next_mark
    sleeps.clear()
    assert 12.0 == sync.pace_request(next_mark, now=12.0, minimum_interval=1.0, sleep=sleeps.append)
    assert [] == sleeps


def test_offline_cache_miss_fails_without_network() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = sync.DownloadCache(Path(tmp))
        with pytest.raises(sync.OfflineCacheMiss):
            cache.read("https://www.biorxiv.org/missing")


def test_cache_rejects_partial_or_digest_mismatched_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cache = sync.DownloadCache(Path(tmp))
        url = "https://www.biorxiv.org/content/a"
        value = sync.Downloaded(b"payload", "text/plain", url)
        cache.write(url, value)
        assert value == cache.read(url)

        payload, metadata = cache._paths(url)
        payload.write_bytes(b"tampered")
        with pytest.raises(sync.OfflineCacheMiss):
            cache.read(url)

        payload.unlink()
        with pytest.raises(sync.OfflineCacheMiss):
            cache.read(url)


def test_fallback_is_allowed_only_for_retryable_workbook_failures() -> None:
    for status in (403, 429, 500, 503):
        assert sync.may_use_workbook_fallback(sync.HttpFetchError(status, "failed"))
    for status in (400, 404, 410):
        assert not sync.may_use_workbook_fallback(sync.HttpFetchError(status, "failed"))
    assert sync.may_use_workbook_fallback(TimeoutError())
    assert not sync.may_use_workbook_fallback(sync.SourceValidationError("corrupt"))


def test_manifest_records_supplied_fallback_name_without_host_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fallback = root / "user-cache" / "verified-workbook.xlsx"
        fallback.parent.mkdir()
        workbook = b"PK\x03\x04verified"
        fallback.write_bytes(workbook)
        article = b"<!doctype html><html>" + (b"x" * 1_100) + b"</html>"
        article_url = "https://example.invalid/article"
        workbook_url = "https://example.invalid/workbook"
        spec = sync.PaperSpec(
            slug="portable-fallback",
            title="Portable fallback",
            doi="10.0000/example",
            version="v1",
            license="CC-BY-4.0",
            article_url=article_url,
            supplement_url=None,
            expected_figures=0,
            expected_figure_files=0,
            expected_equations=0,
            has_supplement=False,
            workbook_url=workbook_url,
            workbook_sha256=hashlib.sha256(workbook).hexdigest(),
            workbook_size=len(workbook),
            workbook_expected_rows=302,
            workbook_expected_columns=33,
        )

        class Fetcher:
            def get(self, url: str) -> sync.Downloaded:
                if url == article_url:
                    return sync.Downloaded(article, "text/html", article_url)
                raise TimeoutError("retryable workbook failure")

        converted = sync.ConvertedArticle(
            paper_markdown="# Paper\n",
            supplement_markdown=None,
            media=(),
            source_title="Portable fallback",
            authors=("Ada Example",),
        )
        with (
            mock.patch.object(sync, "convert_article", return_value=converted),
            mock.patch.object(sync, "_validate_staged", return_value=None),
        ):
            sync.sync_paper(
                spec,
                output_root=root / "output",
                fetcher=Fetcher(),
                source_overrides={},
                fallback_media_1=fallback,
                update=False,
            )

        manifest = json.loads((root / "output" / spec.slug / "MANIFEST.json").read_text(encoding="utf-8"))
        source = next(row for row in manifest["asset"]["sources"] if row["role"] == "supplementary-workbook")
        assert "explicit-local-fallback:verified-workbook.xlsx" == source["url"]
        assert "dist/media-1.xlsx" not in source["url"]
        assert str(root) not in source["url"]


def test_invalid_article_utf8_has_a_source_validation_error() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        article_url = "https://example.invalid/article"
        spec = sync.PaperSpec(
            slug="invalid-utf8",
            title="Invalid UTF-8",
            doi="10.0000/invalid",
            version="v1",
            license="CC-BY-4.0",
            article_url=article_url,
            supplement_url=None,
            expected_figures=0,
            expected_figure_files=0,
            expected_equations=0,
            has_supplement=False,
        )

        class Fetcher:
            def get(self, _url: str) -> sync.Downloaded:
                return sync.Downloaded(b"\xff" * 1_100, "text/html", article_url)

        with (
            mock.patch.object(sync, "validate_payload", return_value=None),
            pytest.raises(sync.SourceValidationError, match="not valid UTF-8"),
        ):
            sync.sync_paper(
                spec,
                output_root=Path(tmp) / "output",
                fetcher=Fetcher(),
                source_overrides={},
                fallback_media_1=None,
                update=False,
            )


def test_second_sheet_name_dimensions_ids_and_cached_formulas() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workbook = Path(tmp) / "media-1.xlsx"
        _make_xlsx(workbook)
        sheet = sync.read_xlsx_sheet(workbook, "Sheet 1")
        assert 4 == len(sheet.rows)
        assert all(len(row) == 33 for row in sheet.rows)
        assert 3 == len({row[0] for row in sheet.rows[1:]})
        assert ["0.1", "0.2", "0.3"] == [row[30] for row in sheet.rows[1:]]


def test_named_sheet_reader_is_generic_but_sheet1_contract_checks_position() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workbook = Path(tmp) / "media-1.xlsx"
        _make_xlsx(workbook, sheet1_second=False)

        assert "Sheet 1" == sync.read_xlsx_sheet(workbook, "Sheet 1").name
        with pytest.raises(sync.WorkbookValidationError, match="second sheet"):
            sync._sheet1_tsv(workbook, expected_rows=3, expected_columns=33)


def test_missing_formula_cache_is_rejected() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workbook = Path(tmp) / "media-1.xlsx"
        _make_xlsx(workbook, formula_values=False)
        with pytest.raises(sync.WorkbookValidationError):
            sync.read_xlsx_sheet(workbook, "Sheet 1")


def test_extract_writes_lf_tsv_and_check_detects_drift() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        workbook = root / "media-1.xlsx"
        output = root / "result"
        _make_xlsx(workbook)
        report = sync.extract_sheet1(workbook, output, check=False, expected_rows=3, expected_columns=33)
        tsv = output / "sheet-1.tsv"
        assert "sheet-1.tsv" == report["path"]
        data = tsv.read_bytes()
        assert b"\r" not in data
        assert 4 == len(data.splitlines())
        sync.extract_sheet1(workbook, output, check=True, expected_rows=3, expected_columns=33)
        tsv.write_text("tampered\n", encoding="utf-8")
        with pytest.raises(sync.WorkbookValidationError):
            sync.extract_sheet1(workbook, output, check=True, expected_rows=3, expected_columns=33)


def _write_asset_tree(root: Path) -> None:
    (root / "figures").mkdir(parents=True)
    (root / "paper.md").write_text("# Paper\n\n![Figure](figures/figure-01.jpg)\n", encoding="utf-8")
    (root / "figures" / "figure-01.jpg").write_bytes(b"\xff\xd8\xffsynthetic")
    (root / "ATTRIBUTION.md").write_text("CC-BY-4.0\n", encoding="utf-8")


def test_manifest_is_deterministic_and_excludes_itself_and_absolute_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_asset_tree(root)
        metadata = {"slug": "synthetic", "source_url": "https://www.biorxiv.org/content/example"}
        first = sync.write_manifest(root, metadata)
        second = sync.write_manifest(root, metadata)
        assert first == second
        payload = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        assert "MANIFEST.json" not in {item["path"] for item in payload["files"]}
        assert str(root) not in json.dumps(payload)
        assert "generated_at" not in payload


def test_check_detects_missing_extra_tamper_and_broken_local_links() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_asset_tree(root)
        sync.write_manifest(root, {"slug": "synthetic"})
        assert [] == sync.check_asset_tree(root)
        (root / "paper.md").write_text("![missing](figures/nope.jpg)\n", encoding="utf-8")
        assert any("tampered" in item or "local link" in item for item in sync.check_asset_tree(root))
        (root / "extra.txt").write_text("extra", encoding="utf-8")
        assert any("extra" in item for item in sync.check_asset_tree(root))
        (root / "figures" / "figure-01.jpg").unlink()
        assert any("missing" in item for item in sync.check_asset_tree(root))


def test_check_detects_missing_local_fragments() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "paper.md").write_text("[citation](#ref-1)\n", encoding="utf-8")
        sync.write_manifest(root, {"slug": "synthetic"})

        errors = sync.check_asset_tree(root)

        assert any("broken local fragment" in item for item in errors)

        (root / "paper.md").write_text(
            '<a id="ref-1"></a>\n\n[citation](#ref-1)\n',
            encoding="utf-8",
        )
        sync.write_manifest(root, {"slug": "synthetic"})
        assert [] == sync.check_asset_tree(root)


def test_atomic_install_is_idempotent_and_rolls_back_on_validation_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        destination = base / "assets"
        staged = base / "staged"
        destination.mkdir()
        (destination / "old.txt").write_text("old", encoding="utf-8")
        staged.mkdir()
        (staged / "new.txt").write_text("new", encoding="utf-8")
        with pytest.raises(sync.SourceValidationError):
            sync.atomic_install(
                staged, destination, validate=lambda _: (_ for _ in ()).throw(sync.SourceValidationError("bad"))
            )
        assert "old" == (destination / "old.txt").read_text(encoding="utf-8")

        staged = base / "staged-ok"
        staged.mkdir()
        (staged / "new.txt").write_text("new", encoding="utf-8")
        sync.atomic_install(staged, destination, validate=lambda _: None)
        mtime = (destination / "new.txt").stat().st_mtime_ns
        staged = base / "staged-same"
        staged.mkdir()
        (staged / "new.txt").write_text("new", encoding="utf-8")
        changed = sync.atomic_install(staged, destination, validate=lambda _: None)
        assert not changed
        assert mtime == (destination / "new.txt").stat().st_mtime_ns


def test_source_drift_requires_update() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_asset_tree(root)
        sync.write_manifest(root, {"slug": "synthetic", "converter_version": "1", "source_sha256": "a"})
        with pytest.raises(sync.SourceDriftError):
            sync.enforce_update_policy(root, {"converter_version": "2", "source_sha256": "b"}, update=False)
        sync.enforce_update_policy(root, {"converter_version": "2", "source_sha256": "b"}, update=True)


def test_converter_byte_drift_requires_update() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_asset_tree(root)
        sync.write_manifest(
            root,
            {"slug": "synthetic", "converter_version": "1", "converter_sha256": "old", "source_sha256": "a"},
        )
        with pytest.raises(sync.SourceDriftError):
            sync.enforce_update_policy(
                root,
                {"converter_version": "1", "converter_sha256": "new", "source_sha256": "a"},
                update=False,
            )
