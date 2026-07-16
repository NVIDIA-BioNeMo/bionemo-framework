"""Tests for the portable Agent Skills eval validator and Codex runner."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_skill_evals.py"
ASSERTIONS = [
    "The response identifies the required input before proposing mutation.",
    "The response records unresolved assumptions rather than inventing values.",
]


def _case(case_id: str = "alpha-001") -> dict[str, object]:
    return {
        "id": case_id,
        "prompt": "Plan a compact alpha workflow.",
        "expected_output": "A concise plan grounded in the alpha skill.",
        "assertions": ASSERTIONS,
        "expected_skill": "alpha",
        "expected_script": None,
    }


def _write_suite(root: Path, skill: str = "alpha", cases: list[dict[str, object]] | None = None) -> Path:
    skill_dir = root / skill
    path = skill_dir / "evals" / "evals.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"skill_name": skill, "evals": cases or [_case()]}, indent=2) + "\n",
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {skill}\ndescription: Test skill.\n---\n\n# {skill.title()}\n\n"
        "Read [contract.md](references/contract.md).\n",
        encoding="utf-8",
    )
    references = skill_dir / "references"
    references.mkdir()
    (references / "contract.md").write_text("# Contract\n\nStay traceable.\n", encoding="utf-8")
    return path


def _write_fake_codex(root: Path, mode: str) -> Path:
    fake = root / f"fake_codex_{mode}.py"
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            from pathlib import Path
            import sys

            MODE = {mode!r}
            ASSERTIONS = {ASSERTIONS!r}
            args = sys.argv[1:]
            if args == ["--version"]:
                print("codex-cli 9.9-test")
                raise SystemExit(0)
            prompt = sys.stdin.read()
            if MODE == "transport" and "--output-schema" not in args:
                print("error: network connection unavailable while contacting api.openai.com", file=sys.stderr)
                raise SystemExit(1)
            if MODE == "stdout-marker" and "--output-schema" not in args:
                print("the answer discussed a rate limit")
                print("ordinary generation failure", file=sys.stderr)
                raise SystemExit(1)
            output = Path(args[args.index("-o") + 1])
            if "--output-schema" not in args:
                output.write_text("# Alpha answer\\n", encoding="utf-8")
                print(json.dumps({{"type": "generation", "prompt_bytes": len(prompt)}}))
                raise SystemExit(0)
            assertion_rows = [
                {{"assertion": assertion, "passed": True, "evidence": "answer"}}
                for assertion in ASSERTIONS
            ]
            outcome = "pass"
            passed = True
            summary = "All assertions passed."
            if MODE == "scientific-fail":
                assertion_rows[0]["passed"] = False
                assertion_rows[0]["evidence"] = "No qualifying scientific source was found."
                outcome = "fail"
                passed = False
                summary = "No qualifying scientific source was found."
            elif MODE == "grader-skip":
                for row in assertion_rows:
                    row["passed"] = False
                outcome = "skip"
                passed = False
                summary = "No qualifying scientific source was found, so skip."
            elif MODE == "mismatch":
                assertion_rows[0]["assertion"] = "A different assertion"
            elif MODE == "empty-evidence":
                assertion_rows[0]["evidence"] = ""
            output.write_text(json.dumps({{
                "case_id": "alpha-001",
                "outcome": outcome,
                "passed": passed,
                "assertions": assertion_rows,
                "summary": summary,
            }}) + "\\n", encoding="utf-8")
            print(json.dumps({{"type": "grader", "prompt_bytes": len(prompt)}}))
            """
        ),
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _init_git_repo(root: Path) -> str:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Eval Test",
            "-c",
            "user.email=eval@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        check=True,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


class EvalRunnerTests(unittest.TestCase):
    def _run(self, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            check=check,
            text=True,
            capture_output=True,
        )

    def test_validate_accepts_bionemo_compatible_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_root = Path(tmp) / "skills"
            _write_suite(skill_root)
            completed = self._run("--skill-root", str(skill_root), "--validate", check=True)
            self.assertIn("1 eval file", completed.stdout)
            self.assertIn("1 case", completed.stdout)

    def test_validate_rejects_duplicate_ids_across_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_root = Path(tmp) / "skills"
            _write_suite(skill_root, "alpha", [_case("shared-001")])
            _write_suite(skill_root, "beta", [{**_case("shared-001"), "expected_skill": "beta"}])
            completed = self._run("--skill-root", str(skill_root), "--validate")
            self.assertEqual(2, completed.returncode)
            self.assertIn("duplicate eval id", completed.stderr.lower())

    def test_validate_rejects_missing_compatible_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_root = Path(tmp) / "skills"
            invalid = _case()
            del invalid["expected_output"]
            _write_suite(skill_root, cases=[invalid])
            completed = self._run("--skill-root", str(skill_root), "--validate")
            self.assertEqual(2, completed.returncode)
            self.assertIn("expected_output", completed.stderr)

    def test_list_json_reports_owning_file_and_skill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            skill_root = Path(tmp) / "skills"
            suite = _write_suite(skill_root)
            completed = self._run("--skill-root", str(skill_root), "--list", "--format", "json", check=True)
            payload = json.loads(completed.stdout)
            self.assertEqual("alpha-001", payload[0]["id"])
            self.assertEqual("alpha", payload[0]["skill_name"])
            self.assertEqual(str(suite), payload[0]["source"])

    def test_dry_run_writes_reproducible_plan_without_launching(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            results = root / "results"
            completed = self._run(
                "--skill-root", str(skill_root), "--repo-root", str(root), "--case", "alpha-001",
                "--results-dir", str(results), "--dry-run", check=True,
            )
            self.assertIn("alpha-001", completed.stdout)
            plan = json.loads((results / "alpha-001" / "run-plan.json").read_text(encoding="utf-8"))
            generation = plan["generation_command"]
            grading = plan["grading_command"]
            self.assertIn("--ephemeral", generation)
            self.assertIn("--json", generation)
            self.assertIn("read-only", generation)
            self.assertIn("--output-schema", grading)
            self.assertEqual("dry-run-not-probed", plan["provenance"]["codex"]["version"])
            self.assertIn("instruction_files", plan["provenance"])
            self.assertFalse((results / "alpha-001" / "generation.trace.jsonl").exists())

    def test_repeated_dry_run_fails_cleanly_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            results = root / "results"
            args = (
                "--skill-root", str(skill_root), "--repo-root", str(root), "--case", "alpha-001",
                "--results-dir", str(results), "--dry-run",
            )
            self.assertEqual(0, self._run(*args).returncode)
            original = (results / "alpha-001" / "run-plan.json").read_bytes()
            repeated = self._run(*args)
            self.assertEqual(2, repeated.returncode)
            self.assertIn("occupied result destination", repeated.stderr)
            self.assertNotIn("Traceback", repeated.stderr)
            self.assertEqual(original, (results / "alpha-001" / "run-plan.json").read_bytes())

    def test_partial_multi_case_destination_is_preflighted_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root, cases=[_case("alpha-001"), _case("alpha-002")])
            results = root / "results"
            occupied = results / "alpha-001"
            occupied.mkdir(parents=True)
            (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
            completed = self._run(
                "--skill-root", str(skill_root), "--repo-root", str(root), "--all",
                "--results-dir", str(results), "--dry-run",
            )
            self.assertEqual(2, completed.returncode)
            self.assertIn("alpha-001", completed.stderr)
            self.assertFalse((results / "alpha-002").exists())
            self.assertEqual("keep\n", (occupied / "keep.txt").read_text(encoding="utf-8"))

    def test_live_run_requires_explicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            fake = _write_fake_codex(root, "pass")
            completed = self._run(
                "--skill-root", str(skill_root), "--repo-root", str(root), "--codex", str(fake),
                "--case", "alpha-001", "--results-dir", str(root / "results"), "--run",
            )
            self.assertEqual(2, completed.returncode)
            self.assertIn("--model is required", completed.stderr)

    def test_run_preserves_artifacts_and_reproducibility_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            fake = _write_fake_codex(root, "pass")
            revision = _init_git_repo(root)
            results = root / "results"
            completed = self._run(
                "--skill-root", str(skill_root), "--repo-root", str(root), "--codex", str(fake),
                "--model", "test-model", "--case", "alpha-001", "--results-dir", str(results), "--run",
                check=True,
            )
            case_dir = results / "alpha-001"
            self.assertIn("PASS", completed.stdout)
            self.assertEqual("# Alpha answer\n", (case_dir / "answer.md").read_text(encoding="utf-8"))
            self.assertIn('"type": "generation"', (case_dir / "generation.trace.jsonl").read_text())
            self.assertIn('"type": "grader"', (case_dir / "grading.trace.jsonl").read_text())
            grade = json.loads((case_dir / "grade.json").read_text(encoding="utf-8"))
            self.assertTrue(grade["passed"])
            plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
            self.assertEqual("test-model", plan["provenance"]["codex"]["model"])
            self.assertEqual("codex-cli 9.9-test", plan["provenance"]["codex"]["version"])
            self.assertEqual(revision, plan["provenance"]["repository"]["revision"])
            instruction_paths = {row["path"] for row in plan["provenance"]["instruction_files"]}
            self.assertIn("alpha/SKILL.md", instruction_paths)
            self.assertIn("alpha/references/contract.md", instruction_paths)

    def test_grader_authored_skip_is_rejected(self) -> None:
        completed, _ = self._run_fake("grader-skip")
        self.assertEqual(2, completed.returncode)
        self.assertIn("outcome must be pass or fail", completed.stderr)
        self.assertNotIn("SKIP", completed.stdout)

    def test_scientific_no_result_is_a_failed_eval(self) -> None:
        completed, case_dir = self._run_fake("scientific-fail")
        self.assertEqual(1, completed.returncode)
        self.assertIn("FAIL", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("failed", status["status"])

    def test_assertion_text_mismatch_is_rejected(self) -> None:
        completed, _ = self._run_fake("mismatch")
        self.assertEqual(2, completed.returncode)
        self.assertIn("assertion text/order", completed.stderr)

    def test_empty_assertion_evidence_is_rejected(self) -> None:
        completed, _ = self._run_fake("empty-evidence")
        self.assertEqual(2, completed.returncode)
        self.assertIn("non-empty evidence", completed.stderr)

    def test_stdout_rate_limit_words_do_not_turn_unrelated_error_into_skip(self) -> None:
        completed, case_dir = self._run_fake("stdout-marker")
        self.assertEqual(2, completed.returncode)
        self.assertNotIn("SKIP", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("generation-error", status["status"])

    def test_stderr_transport_failure_is_runner_classified_skip(self) -> None:
        completed, case_dir = self._run_fake("transport")
        self.assertEqual(0, completed.returncode)
        self.assertIn("SKIP", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("infrastructure-skip", status["status"])
        self.assertEqual("network", status["reason_category"])
        self.assertEqual("generation", status["phase"])

    def _run_fake(self, mode: str) -> tuple[subprocess.CompletedProcess[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        skill_root = root / "skills"
        _write_suite(skill_root)
        fake = _write_fake_codex(root, mode)
        results = root / "results"
        completed = self._run(
            "--skill-root", str(skill_root), "--repo-root", str(root), "--codex", str(fake),
            "--model", "test-model", "--case", "alpha-001", "--results-dir", str(results), "--run",
        )
        return completed, results / "alpha-001"


if __name__ == "__main__":
    unittest.main()
