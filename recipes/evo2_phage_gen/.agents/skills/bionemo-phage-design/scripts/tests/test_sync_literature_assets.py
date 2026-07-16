"""Tests for the portable literature-asset synchronizer.

The suite uses only synthetic HTML/XLSX fixtures.  Network synchronization is
an explicit integration step, not part of the unit test run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "sync_literature_assets.py"
SPEC = importlib.util.spec_from_file_location("sync_literature_assets", SCRIPT)
assert SPEC and SPEC.loader
sync = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sync
SPEC.loader.exec_module(sync)


def _article_html(*, doi: str, version: str = "v1", license_url: str = "https://creativecommons.org/licenses/by/4.0/") -> str:
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


def _make_xlsx(path: Path, *, rows: int = 3, columns: int = 33, formula_values: bool = True) -> None:
    """Build a minimal XLSX whose second sheet is named ``Sheet 1``."""
    main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    workbook = f'''<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="{main}" xmlns:r="{rel}"><sheets>
<sheet name="Contents" sheetId="1" r:id="rId1"/><sheet name="Sheet 1" sheetId="2" r:id="rId2"/>
</sheets></workbook>'''
    rels = '''<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet2.xml"/>
</Relationships>'''
    contents = f'''<?xml version="1.0"?><worksheet xmlns="{main}"><sheetData/></worksheet>'''

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
            value = f"column-{column}" if row == 1 else (f"design-{row - 1}" if column == 1 else f"r{row - 1}c{column}")
            if column == 31 and row > 1:
                cached = f"{0.1 * (row - 1):.1f}" if formula_values else ""
                cells.append(f'<c r="{ref}"><f>1/10</f><v>{cached}</v></c>')
            else:
                cells.append(f'<c r="{ref}" t="inlineStr"><is><t>{value}</t></is></c>')
        xml_rows.append(f'<row r="{row}">{"".join(cells)}</row>')
    sheet = f'''<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="{main}"><dimension ref="A1:AG{rows + 1}"/><sheetData>{"".join(xml_rows)}</sheetData></worksheet>'''
    content_types = '''<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="xml" ContentType="application/xml"/></Types>'''
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", rels)
        archive.writestr("xl/worksheets/sheet1.xml", contents)
        archive.writestr("xl/worksheets/sheet2.xml", sheet)


class CliTests(unittest.TestCase):
    def test_help_lists_all_subcommands(self) -> None:
        completed = subprocess.run([sys.executable, str(SCRIPT), "--help"], check=True, text=True, capture_output=True)
        self.assertIn("sync", completed.stdout)
        self.assertIn("check", completed.stdout)
        self.assertIn("extract-sheet1", completed.stdout)


class ArticleConversionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = sync.PaperSpec(
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

    def test_supplement_boundary_keeps_acknowledgements_and_references_in_main(self) -> None:
        converted = sync.convert_article(_article_html(doi=self.spec.doi), self.spec)
        self.assertIn("# Supplementary material", converted.supplement_markdown)
        self.assertNotIn("Acknowledgements", converted.supplement_markdown)
        self.assertIn("Supplementary files", converted.supplement_markdown)
        self.assertNotIn("Supplementary files", converted.paper_markdown)
        self.assertIn("Acknowledgements", converted.paper_markdown)
        self.assertIn("References", converted.paper_markdown)

    def test_multi_image_order_and_cross_file_links_are_preserved(self) -> None:
        converted = sync.convert_article(_article_html(doi=self.spec.doi), self.spec)
        self.assertEqual(
            [item.output_path for item in converted.media if item.kind == "figure"],
            ["figures/figure-01.jpg", "figures/figure-02-part-1.jpg", "figures/figure-02-part-2.jpg"],
        )
        self.assertIn("[Figure S1](supplement.md#f2)", converted.paper_markdown)
        self.assertIn("![Figure S1, part 1](figures/figure-02-part-1.jpg)", converted.supplement_markdown)
        self.assertLess(converted.supplement_markdown.index("part 1"), converted.supplement_markdown.index("part 2"))

    def test_equation_image_survives_conversion(self) -> None:
        converted = sync.convert_article(_article_html(doi=self.spec.doi), self.spec)
        self.assertIn("![Equation 1](equations/equation-01.gif)", converted.paper_markdown)
        equations = [item for item in converted.media if item.kind == "equation"]
        self.assertEqual(1, len(equations))

    def test_biorxiv_static_equation_url_uses_official_content_route(self) -> None:
        html = _article_html(doi=self.spec.doi).replace(
            "https://www.biorxiv.org/media/graphic-4.gif",
            "https://www.biorxiv.org/sites/default/files/highwire/biorxiv/early/2026/06/19/example/embed/graphic-4.gif",
        )
        converted = sync.convert_article(html, self.spec)
        equation = next(item for item in converted.media if item.kind == "equation")
        self.assertEqual(
            "https://www.biorxiv.org/content/biorxiv/early/2026/06/19/example/embed/graphic-4.gif",
            equation.source_url,
        )

    def test_table_s1_caption_and_workbook_source_survive(self) -> None:
        converted = sync.convert_article(_article_html(doi=self.spec.doi), self.spec)
        self.assertIn("Table S1", converted.supplement_markdown)
        self.assertIn("File S1", converted.supplement_markdown)

    def test_doi_version_and_license_mismatches_fail_closed(self) -> None:
        cases = [
            _article_html(doi="10.1234/wrong"),
            _article_html(doi=self.spec.doi, version="v2"),
            _article_html(doi=self.spec.doi, license_url="https://example.org/proprietary"),
        ]
        for html in cases:
            with self.subTest(html=html[:100]), self.assertRaises(sync.SourceValidationError):
                sync.convert_article(html, self.spec)

    def test_publisher_citation_full_html_url_proves_version(self) -> None:
        html = _article_html(doi=self.spec.doi).replace("citation_fulltext_html_url", "citation_full_html_url")
        converted = sync.convert_article(html, self.spec)
        self.assertIn("Synthetic title", converted.paper_markdown)

    def test_publisher_dc_rights_proves_cc_by_license(self) -> None:
        html = _article_html(doi=self.spec.doi).replace(
            '<link rel="license" href="https://creativecommons.org/licenses/by/4.0/">',
            '<meta name="DC.Rights" content="Creative Commons License (Attribution 4.0 International), CC BY 4.0, http://creativecommons.org/licenses/by/4.0/">',
        )
        converted = sync.convert_article(html, self.spec)
        self.assertIn("CC-BY-4.0", converted.paper_markdown)


class FetchValidationTests(unittest.TestCase):
    def test_user_agent_is_publisher_compatible_and_identifies_tool(self) -> None:
        self.assertTrue(sync.USER_AGENT.startswith("Mozilla/5.0"))
        self.assertIn("phage-design-literature-sync", sync.USER_AGENT)

    def test_allowlist_checks_initial_and_redirect_urls(self) -> None:
        self.assertTrue(sync.is_allowed_url("https://www.biorxiv.org/content/a"))
        self.assertFalse(sync.is_allowed_url("http://www.biorxiv.org/content/a"))
        self.assertFalse(sync.is_allowed_url("https://evil.example/content/a"))
        with self.assertRaises(sync.SourceValidationError):
            sync.validate_redirect_chain(
                "https://www.biorxiv.org/content/a",
                "https://evil.example/payload",
            )

    def test_mime_magic_and_size_validation(self) -> None:
        sync.validate_payload(b"\xff\xd8\xffjpeg", "image/jpeg", "jpeg", min_size=4, max_size=100)
        sync.validate_payload(b"GIF89a-data", "image/gif", "gif", min_size=4, max_size=100)
        sync.validate_payload(b"PK\x03\x04xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx", min_size=4, max_size=100)
        for data, mime, kind in [
            (b"<html>blocked</html>", "image/jpeg", "jpeg"),
            (b"\xff\xd8\xffjpeg", "text/html", "jpeg"),
            (b"GIF89a-data", "image/gif", "jpeg"),
        ]:
            with self.subTest(kind=kind), self.assertRaises(sync.SourceValidationError):
                sync.validate_payload(data, mime, kind, min_size=4, max_size=100)
        with self.assertRaises(sync.SourceValidationError):
            sync.validate_payload(b"GIF89a", "image/gif", "gif", min_size=100, max_size=200)

    def test_retry_honors_retry_after_and_is_bounded(self) -> None:
        attempts: list[int] = []
        sleeps: list[float] = []

        def operation() -> bytes:
            attempts.append(1)
            if len(attempts) < 3:
                raise sync.RetryableFetchError("busy", retry_after=0.25)
            return b"ok"

        self.assertEqual(b"ok", sync.with_retries(operation, sleep=sleeps.append, attempts=3))
        self.assertEqual(3, len(attempts))
        self.assertEqual([0.25, 0.25], sleeps)

    def test_retryable_primary_failure_uses_bounded_secondary_transport(self) -> None:
        calls: list[str] = []

        def primary() -> bytes:
            calls.append("primary")
            raise sync.RetryableFetchError("publisher rejected HTTP/1.1")

        def secondary() -> bytes:
            calls.append("secondary")
            return b"official payload"

        result = sync.with_retryable_fallback(primary, secondary)
        self.assertEqual(b"official payload", result)
        self.assertEqual(["primary", "secondary"], calls)
        with self.assertRaises(sync.SourceValidationError):
            sync.with_retryable_fallback(lambda: (_ for _ in ()).throw(sync.SourceValidationError("corrupt")), secondary)

    def test_request_pacing_waits_only_for_remaining_interval(self) -> None:
        sleeps: list[float] = []
        next_mark = sync.pace_request(10.0, now=10.25, minimum_interval=1.0, sleep=sleeps.append)
        self.assertEqual([0.75], sleeps)
        self.assertEqual(11.0, next_mark)
        sleeps.clear()
        self.assertEqual(12.0, sync.pace_request(next_mark, now=12.0, minimum_interval=1.0, sleep=sleeps.append))
        self.assertEqual([], sleeps)

    def test_offline_cache_miss_fails_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = sync.DownloadCache(Path(tmp))
            with self.assertRaises(sync.OfflineCacheMiss):
                cache.read("https://www.biorxiv.org/missing")

    def test_fallback_is_allowed_only_for_retryable_workbook_failures(self) -> None:
        for status in (403, 429, 500, 503):
            self.assertTrue(sync.may_use_workbook_fallback(sync.HttpFetchError(status, "failed")))
        for status in (400, 404, 410):
            self.assertFalse(sync.may_use_workbook_fallback(sync.HttpFetchError(status, "failed")))
        self.assertTrue(sync.may_use_workbook_fallback(TimeoutError()))
        self.assertFalse(sync.may_use_workbook_fallback(sync.SourceValidationError("corrupt")))


class WorkbookTests(unittest.TestCase):
    def test_second_sheet_name_dimensions_ids_and_cached_formulas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workbook = Path(tmp) / "media-1.xlsx"
            _make_xlsx(workbook)
            sheet = sync.read_xlsx_sheet(workbook, "Sheet 1")
            self.assertEqual(4, len(sheet.rows))
            self.assertTrue(all(len(row) == 33 for row in sheet.rows))
            self.assertEqual(3, len({row[0] for row in sheet.rows[1:]}))
            self.assertEqual(["0.1", "0.2", "0.3"], [row[30] for row in sheet.rows[1:]])

    def test_missing_formula_cache_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workbook = Path(tmp) / "media-1.xlsx"
            _make_xlsx(workbook, formula_values=False)
            with self.assertRaises(sync.WorkbookValidationError):
                sync.read_xlsx_sheet(workbook, "Sheet 1")

    def test_extract_writes_lf_tsv_and_check_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "media-1.xlsx"
            output = root / "result"
            _make_xlsx(workbook)
            report = sync.extract_sheet1(workbook, output, check=False, expected_rows=3, expected_columns=33)
            tsv = output / "sheet-1.tsv"
            self.assertEqual("sheet-1.tsv", report["path"])
            data = tsv.read_bytes()
            self.assertNotIn(b"\r", data)
            self.assertEqual(4, len(data.splitlines()))
            sync.extract_sheet1(workbook, output, check=True, expected_rows=3, expected_columns=33)
            tsv.write_text("tampered\n", encoding="utf-8")
            with self.assertRaises(sync.WorkbookValidationError):
                sync.extract_sheet1(workbook, output, check=True, expected_rows=3, expected_columns=33)


class ManifestAndTreeTests(unittest.TestCase):
    def _tree(self, root: Path) -> None:
        (root / "figures").mkdir(parents=True)
        (root / "paper.md").write_text("# Paper\n\n![Figure](figures/figure-01.jpg)\n", encoding="utf-8")
        (root / "figures" / "figure-01.jpg").write_bytes(b"\xff\xd8\xffsynthetic")
        (root / "ATTRIBUTION.md").write_text("CC-BY-4.0\n", encoding="utf-8")

    def test_manifest_is_deterministic_and_excludes_itself_and_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            metadata = {"slug": "synthetic", "source_url": "https://www.biorxiv.org/content/example"}
            first = sync.write_manifest(root, metadata)
            second = sync.write_manifest(root, metadata)
            self.assertEqual(first, second)
            payload = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
            self.assertNotIn("MANIFEST.json", {item["path"] for item in payload["files"]})
            self.assertNotIn(str(root), json.dumps(payload))
            self.assertNotIn("generated_at", payload)

    def test_check_detects_missing_extra_tamper_and_broken_local_links(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            sync.write_manifest(root, {"slug": "synthetic"})
            self.assertEqual([], sync.check_asset_tree(root))
            (root / "paper.md").write_text("![missing](figures/nope.jpg)\n", encoding="utf-8")
            self.assertTrue(any("tampered" in item or "local link" in item for item in sync.check_asset_tree(root)))
            (root / "extra.txt").write_text("extra", encoding="utf-8")
            self.assertTrue(any("extra" in item for item in sync.check_asset_tree(root)))
            (root / "figures" / "figure-01.jpg").unlink()
            self.assertTrue(any("missing" in item for item in sync.check_asset_tree(root)))

    def test_atomic_install_is_idempotent_and_rolls_back_on_validation_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            destination = base / "assets"
            staged = base / "staged"
            destination.mkdir()
            (destination / "old.txt").write_text("old", encoding="utf-8")
            staged.mkdir()
            (staged / "new.txt").write_text("new", encoding="utf-8")
            with self.assertRaises(sync.SourceValidationError):
                sync.atomic_install(staged, destination, validate=lambda _: (_ for _ in ()).throw(sync.SourceValidationError("bad")))
            self.assertEqual("old", (destination / "old.txt").read_text(encoding="utf-8"))

            staged = base / "staged-ok"
            staged.mkdir()
            (staged / "new.txt").write_text("new", encoding="utf-8")
            sync.atomic_install(staged, destination, validate=lambda _: None)
            mtime = (destination / "new.txt").stat().st_mtime_ns
            staged = base / "staged-same"
            staged.mkdir()
            (staged / "new.txt").write_text("new", encoding="utf-8")
            changed = sync.atomic_install(staged, destination, validate=lambda _: None)
            self.assertFalse(changed)
            self.assertEqual(mtime, (destination / "new.txt").stat().st_mtime_ns)

    def test_source_drift_requires_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            sync.write_manifest(root, {"slug": "synthetic", "converter_version": "1", "source_sha256": "a"})
            with self.assertRaises(sync.SourceDriftError):
                sync.enforce_update_policy(root, {"converter_version": "2", "source_sha256": "b"}, update=False)
            sync.enforce_update_policy(root, {"converter_version": "2", "source_sha256": "b"}, update=True)

    def test_converter_byte_drift_requires_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._tree(root)
            sync.write_manifest(
                root,
                {"slug": "synthetic", "converter_version": "1", "converter_sha256": "old", "source_sha256": "a"},
            )
            with self.assertRaises(sync.SourceDriftError):
                sync.enforce_update_policy(
                    root,
                    {"converter_version": "1", "converter_sha256": "new", "source_sha256": "a"},
                    update=False,
                )


if __name__ == "__main__":
    unittest.main()
