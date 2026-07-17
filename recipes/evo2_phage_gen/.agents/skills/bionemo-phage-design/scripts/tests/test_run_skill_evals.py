"""Tests for the portable Agent Skills eval validator and harness adapters."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_skill_evals.py"
RECIPE_SKILL_ROOT = SCRIPT.parents[2]
RECIPE_ROOT = SCRIPT.parents[4]
EXPECTED_RECIPE_SKILLS = {
    "bionemo-phage-design",
    "bionemo-phage-design-adapt-execution",
    "bionemo-phage-design-collect-genomes",
    "bionemo-phage-design-generate-and-screen",
    "bionemo-phage-design-implement-rl-objectives",
    "bionemo-phage-design-operate-mbridge-sft",
    "bionemo-phage-design-operate-nemo-rl",
    "bionemo-phage-design-plan-rl-objectives",
    "bionemo-phage-design-prepare-sft",
    "bionemo-phage-design-research-evidence",
}
RETIRED_LEAF_SKILL_NAMES = {
    "adapt-phage-execution",
    "collect-phage-genomes",
    "design-phage-rl-objectives",
    "generate-and-screen-phages",
    "implement-phage-rl-objectives",
    "operate-mbridge-phage-sft",
    "operate-nemo-rl-phage",
    "prepare-phage-sft",
    "research-phage-evidence",
}
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


def _write_fake_claude(root: Path, mode: str = "pass") -> Path:
    fake = root / f"fake_claude_{mode}.py"
    fake.write_text(
        textwrap.dedent(
            f"""\
            #!/usr/bin/env python3
            import json
            import os
            from pathlib import Path
            import sys

            MODE = {mode!r}
            ASSERTIONS = {ASSERTIONS!r}
            ORIGINAL_ROOT = {str(root)!r}
            args = sys.argv[1:]
            if args == ["--version"]:
                print("2.1.211 (Claude Code test)")
                raise SystemExit(0)
            prompt = sys.stdin.read()
            if MODE == "isolation" and (
                os.environ.get("CLAUDE_CODE_DISABLE_CLAUDE_MDS") != "1"
                or os.environ.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY") != "1"
            ):
                print("Claude memory isolation environment is missing", file=sys.stderr)
                raise SystemExit(7)
            if MODE == "answer-key-isolation" and "--json-schema" not in args:
                cwd = Path.cwd()
                failures = []
                if cwd.resolve() == Path(ORIGINAL_ROOT).resolve():
                    failures.append("generation used the source repository")
                if list(cwd.glob("skills/*/evals/evals.json")):
                    failures.append("eval answer key is visible")
                if list(cwd.glob("skills/*/assets/VALIDATION.md")):
                    failures.append("prior validation audit is visible")
                if list(cwd.glob("skills/*/scripts/tests/*")):
                    failures.append("eval audit tests are visible")
                if (cwd / "tmp_RUNLOG.md").exists():
                    failures.append("ignored run history is visible")
                if (cwd / "tmp_TRACKED.md").exists():
                    failures.append("tracked temporary history is visible")
                if list(cwd.rglob("*.egg-info")):
                    failures.append("generated package metadata is visible")
                if (cwd / "external-runtime-link").is_symlink():
                    failures.append("external symlink escaped the staged workspace")
                if not (cwd / "internal-runtime-link").is_symlink():
                    failures.append("safe tracked symlink is missing")
                if not (cwd / "skills" / "alpha" / "SKILL.md").is_file():
                    failures.append("selected skill is missing")
                elif "dirty tracked marker" not in (
                    cwd / "skills" / "alpha" / "SKILL.md"
                ).read_text(encoding="utf-8"):
                    failures.append("tracked working-tree edit is missing")
                if "EVALUATION RESPONSE CONTRACT" not in prompt:
                    failures.append("concise response contract is missing")
                if failures:
                    print("; ".join(failures), file=sys.stderr)
                    raise SystemExit(8)
            if "--json-schema" not in args:
                if not prompt.startswith("/evo2-phage-gen:alpha\\n\\n"):
                    print("missing explicit plugin skill invocation", file=sys.stderr)
                    raise SystemExit(4)
                print(json.dumps({{
                    "type": "system",
                    "subtype": "init",
                    "cwd": os.getcwd(),
                    "model": "claude-default-test",
                    "tools": args[args.index("--tools") + 1],
                }}))
                if MODE == "transport":
                    print(json.dumps({{
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "api_error_status": None,
                        "errors": ["network connection unavailable"],
                        "result": "",
                        "total_cost_usd": 0.015,
                        "modelUsage": {{"claude-default-test": {{"inputTokens": 5}}}},
                    }}))
                    raise SystemExit(1)
                if MODE == "rate-limit-status":
                    print(json.dumps({{
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "api_error_status": 429,
                        "errors": [],
                        "result": "",
                    }}))
                    raise SystemExit(1)
                if MODE == "policy-refusal":
                    print(json.dumps({{
                        "type": "system",
                        "subtype": "model_refusal_no_fallback",
                        "api_refusal_category": "bio",
                    }}))
                    print(json.dumps({{
                        "type": "result",
                        "subtype": "success",
                        "is_error": True,
                        "stop_reason": "refusal",
                        "result": "API policy refusal",
                    }}))
                    raise SystemExit(1)
                if MODE == "budget-exhausted":
                    print(json.dumps({{
                        "type": "result",
                        "subtype": "error_max_budget_usd",
                        "is_error": True,
                        "errors": ["Reached maximum budget ($0.25)"],
                        "result": "",
                        "total_cost_usd": 0.35,
                    }}))
                    raise SystemExit(1)
                if MODE == "empty-result":
                    print(json.dumps({{
                        "type": "result",
                        "subtype": "success",
                        "is_error": False,
                        "result": "",
                    }}))
                    raise SystemExit(0)
                print(json.dumps({{
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": "# Alpha answer\\n",
                    "total_cost_usd": 0.01,
                }}))
                raise SystemExit(0)
            grade = {{
                "case_id": "alpha-001",
                "outcome": "pass",
                "passed": True,
                "assertions": [
                    {{"assertion": assertion, "passed": True, "evidence": "answer"}}
                    for assertion in ASSERTIONS
                ],
                "summary": "All assertions passed.",
            }}
            envelope = {{
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "",
                "total_cost_usd": 0.02,
                "modelUsage": {{"claude-default-test": {{"inputTokens": 10}}}},
                "structured_output": grade,
            }}
            if MODE == "missing-structured-grade":
                del envelope["structured_output"]
            print(json.dumps(envelope))
            """
        ),
        encoding="utf-8",
    )
    fake.chmod(0o755)
    return fake


def _write_claude_plugin_manifest(skill_root: Path) -> Path:
    manifest = skill_root.parent / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "name": "evo2-phage-gen",
                "version": "0.1.0",
                "description": "Test bridge for recipe-local Agent Skills.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


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


def _prepare_live_claude_repo(root: Path) -> str:
    (root / ".gitignore").write_text(
        "results/\ntmp_*.md\n*.egg-info/\n",
        encoding="utf-8",
    )
    (root / "tracked-runtime.txt").write_text("tracked runtime\n", encoding="utf-8")
    (root / "internal-runtime-link").symlink_to("tracked-runtime.txt")
    (root / "external-runtime-link").symlink_to("/etc/hosts")
    tracked_tmp = root / "tmp_TRACKED.md"
    tracked_tmp.write_text("tracked prior history\n", encoding="utf-8")
    tracked_egg = root / "src" / "tracked.egg-info" / "PKG-INFO"
    tracked_egg.parent.mkdir(parents=True)
    tracked_egg.write_text("tracked generated metadata\n", encoding="utf-8")
    audit_test = root / "skills" / "alpha" / "scripts" / "tests" / "test_eval_audit.py"
    audit_test.parent.mkdir(parents=True)
    audit_test.write_text("answer-adjacent audit\n", encoding="utf-8")
    nested_tmp_answer = root / "nested" / "tmp_CASE" / "answer.md"
    nested_tmp_answer.parent.mkdir(parents=True)
    nested_tmp_answer.write_text("tracked generated answer\n", encoding="utf-8")
    nested_result_grade = root / "nested" / "results" / "grade.json"
    nested_result_grade.parent.mkdir(parents=True)
    nested_result_grade.write_text("{}\n", encoding="utf-8")
    _init_git_repo(root)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "add",
            "-f",
            tracked_tmp.relative_to(root).as_posix(),
            tracked_egg.relative_to(root).as_posix(),
        ],
        check=True,
    )
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
            "tracked generated fixtures",
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

    def test_recipe_skills_use_the_bionemo_phage_design_namespace(self) -> None:
        actual = {
            path.name
            for path in RECIPE_SKILL_ROOT.iterdir()
            if path.is_dir() and (path / "SKILL.md").is_file()
        }
        self.assertEqual(EXPECTED_RECIPE_SKILLS, actual)
        for skill_name in sorted(actual):
            skill_dir = RECIPE_SKILL_ROOT / skill_name
            frontmatter_name = next(
                line.removeprefix("name:").strip()
                for line in (skill_dir / "SKILL.md").read_text(encoding="utf-8").splitlines()
                if line.startswith("name:")
            )
            self.assertEqual(skill_name, frontmatter_name)
            for eval_path in sorted((skill_dir / "evals").glob("*.json")):
                payload = json.loads(eval_path.read_text(encoding="utf-8"))
                self.assertEqual(skill_name, payload["skill_name"])
                for case in payload["evals"]:
                    if case["expected_skill"] is not None:
                        self.assertEqual(skill_name, case["expected_skill"])
                    self.assertTrue(case["id"].startswith(f"{skill_name}-"), case["id"])
        text_paths = [
            RECIPE_ROOT / "README.md",
            *(
                path
                for path in RECIPE_SKILL_ROOT.rglob("*")
                if path.is_file()
                and path.resolve() != Path(__file__).resolve()
                and path.suffix in {".json", ".md", ".py"}
            ),
        ]
        for path in text_paths:
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("bionemo-bionemo-phage-design", text, str(path))
            for retired in RETIRED_LEAF_SKILL_NAMES:
                self.assertNotIn(retired, text, str(path))

    def test_recipe_codex_plugin_manifest_exposes_the_skill_bundle(self) -> None:
        manifest_path = RECIPE_SKILL_ROOT.parent / ".codex-plugin" / "plugin.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual("bionemo-phage-design", manifest["name"])
        self.assertEqual("./skills/", manifest["skills"])
        self.assertRegex(manifest["version"], r"^\d+\.\d+\.\d+$")
        self.assertTrue(manifest["description"].strip())
        self.assertTrue(manifest["author"]["name"].strip())

        interface = manifest["interface"]
        for field in (
            "displayName",
            "shortDescription",
            "longDescription",
            "developerName",
            "category",
        ):
            self.assertTrue(interface[field].strip(), field)
        self.assertTrue(interface["capabilities"])
        self.assertGreaterEqual(len(interface["defaultPrompt"]), 1)
        self.assertLessEqual(len(interface["defaultPrompt"]), 3)
        self.assertTrue(all(len(prompt) <= 128 for prompt in interface["defaultPrompt"]))

    def test_active_skills_defer_biological_safety_policy_to_the_harness(self) -> None:
        active_instruction_paths = [
            *RECIPE_SKILL_ROOT.glob("*/SKILL.md"),
            *RECIPE_SKILL_ROOT.glob("*/evals/*.json"),
            *RECIPE_SKILL_ROOT.glob("*/references/*.md"),
        ]
        prohibited_markers = (
            "## safety boundary",
            "## biological safety policy",
            "custom biological policy",
            "prokaryotic-only policy",
        )
        for path in active_instruction_paths:
            text = path.read_text(encoding="utf-8").lower()
            self.assertNotIn("385", text, str(path))
            for marker in prohibited_markers:
                self.assertNotIn(marker, text, str(path))

    def test_research_skill_keeps_its_artifact_contract_in_read_only_responses(self) -> None:
        skill_path = RECIPE_SKILL_ROOT / "bionemo-phage-design-research-evidence" / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8")
        self.assertIn("read-only", text.lower())
        for artifact in (
            "artifacts/EVIDENCE.md",
            "artifacts/SOURCES.yaml",
            "artifacts/SEARCH_LOG.jsonl",
            "artifacts/DATASET_CANDIDATES.yaml",
            "OUTPUTS.yaml",
            "SUMMARY.md",
            "RUNLOG.md",
        ):
            self.assertIn(artifact, text)

    def test_execution_and_collection_evals_are_compatible_with_read_only_generation(self) -> None:
        adapt_path = (
            RECIPE_SKILL_ROOT
            / "bionemo-phage-design-adapt-execution"
            / "evals"
            / "evals.json"
        )
        adapt = json.loads(adapt_path.read_text(encoding="utf-8"))["evals"]
        local_preflight = next(case for case in adapt if case["id"].endswith("local-preflight"))
        lepton = next(case for case in adapt if case["id"].endswith("lepton-contract"))
        self.assertIn("read-only", local_preflight["assertions"][-1].lower())
        self.assertIn("when available", lepton["assertions"][0].lower())
        self.assertIn("orders", lepton["assertions"][1].lower())

        collection_path = (
            RECIPE_SKILL_ROOT
            / "bionemo-phage-design-collect-genomes"
            / "evals"
            / "evals.json"
        )
        collection = json.loads(collection_path.read_text(encoding="utf-8"))["evals"]
        jumbo = next(case for case in collection if case["id"].endswith("current-jumbo-corpus"))
        self.assertIn("leaves the payload unverified", jumbo["assertions"][2].lower())

    def test_execution_skill_preserves_read_only_handoff_and_lepton_egress(self) -> None:
        skill_path = RECIPE_SKILL_ROOT / "bionemo-phage-design-adapt-execution" / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("read-only", text)
        lepton = text.split("- **lepton:**", maxsplit=1)[1].split("\n", maxsplit=1)[0]
        self.assertIn("egress", lepton)

    def test_collection_skill_fails_closed_when_prefix_tools_are_unavailable(self) -> None:
        skill_path = RECIPE_SKILL_ROOT / "bionemo-phage-design-collect-genomes" / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("bounded validation command", text)
        self.assertIn("unverified", text)

    def test_collection_read_only_response_keeps_query_and_access_provenance(self) -> None:
        skill_path = RECIPE_SKILL_ROOT / "bionemo-phage-design-collect-genomes" / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("every response", text)
        self.assertIn("exact retrieval query", text)
        self.assertIn("access date", text)

    def test_sft_due_monitoring_reports_health_without_forcing_not_due_polling(self) -> None:
        skill_path = RECIPE_SKILL_ROOT / "bionemo-phage-design-operate-mbridge-sft" / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("every substantive or due monitoring decision", text)
        for metric in (
            "train and validation loss",
            "learning rate",
            "gradient norm",
            "throughput",
            "gpu utilization and memory",
            "checkpoint integrity",
            "free space",
        ):
            self.assertIn(metric, text)
        self.assertIn("not-due", text)
        self.assertIn("last-observed timestamp and staleness", text)
        self.assertIn("without querying", text)

    def test_rl_relaunch_requires_independent_prior_job_terminal_check(self) -> None:
        skill_path = RECIPE_SKILL_ROOT / "bionemo-phage-design-operate-nemo-rl" / "SKILL.md"
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("before any resume or relaunch", text)
        self.assertIn("prior process/job is absent or terminal", text)
        self.assertIn("never duplicate a live submission", text)

    def test_rl_objective_audit_states_telemetry_contract_before_code(self) -> None:
        skill_path = (
            RECIPE_SKILL_ROOT
            / "bionemo-phage-design-implement-rl-objectives"
            / "SKILL.md"
        )
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("even when the audit stops before code", text)
        self.assertIn("raw numerator/denominator/support", text)
        self.assertIn("compares online scoring with final qc", text)
        self.assertIn("recipe-local src/ and tests/", text)
        self.assertIn(
            "installed-runtime name/order/dtype/device/shape/reduction checks",
            text,
        )
        self.assertIn("tiny deterministic rl smoke", text)
        self.assertIn(
            "reward calculation, logging, checkpoint writing, and restart metadata",
            text,
        )

    def test_research_packet_requires_complete_objective_portfolio_rows(self) -> None:
        skill_path = (
            RECIPE_SKILL_ROOT
            / "bionemo-phage-design-research-evidence"
            / "SKILL.md"
        )
        text = skill_path.read_text(encoding="utf-8").lower()
        self.assertIn("portfolio coverage checklist", text)
        self.assertIn("topology/packaging", text)
        self.assertIn("desired and undesired directional changes", text)
        self.assertIn(
            "each axis a decision-table row or mark it unresolved/not applicable",
            text,
        )
        self.assertIn("material interactions across the portfolio", text)

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

    def test_nested_repo_root_provenance_uses_git_toplevel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            nested = root / "nested" / "recipe"
            nested.mkdir(parents=True)
            _init_git_repo(root)
            results = nested / "results"
            self._run(
                "--skill-root", str(skill_root), "--repo-root", str(nested),
                "--case", "alpha-001", "--results-dir", str(results),
                "--dry-run", check=True,
            )
            provenance = json.loads(
                (results / "run-provenance.json").read_text(encoding="utf-8")
            )
            repository = provenance["repository"]
            self.assertEqual(str(root.resolve()), repository["worktree"])
            self.assertEqual([], repository["status_entries"])

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

    def test_live_run_may_use_cli_default_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            fake = _write_fake_codex(root, "pass")
            completed = self._run(
                "--skill-root", str(skill_root), "--repo-root", str(root), "--codex", str(fake),
                "--case", "alpha-001", "--results-dir", str(root / "results"), "--run",
                check=True,
            )
            self.assertIn("PASS", completed.stdout)
            plan = json.loads(
                (root / "results" / "alpha-001" / "run-plan.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsNone(plan["provenance"]["harness"]["model"])
            self.assertNotIn("-m", plan["generation_command"])

    def test_claude_live_run_requires_explicit_external_upload_consent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            _write_claude_plugin_manifest(skill_root)
            fake = _write_fake_claude(root)
            completed = self._run(
                "--harness", "claude", "--skill-root", str(skill_root),
                "--repo-root", str(root), "--claude", str(fake),
                "--case", "alpha-001",
                "--results-dir", str(root / "results"), "--run",
            )
            self.assertEqual(2, completed.returncode)
            self.assertIn("--allow-external-skill-upload", completed.stderr)
            self.assertIn("staged recipe files it reads", completed.stderr)

    def test_claude_dry_run_uses_local_plugin_and_read_only_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            manifest = _write_claude_plugin_manifest(skill_root)
            results = root / "results"
            completed = self._run(
                "--harness", "claude", "--skill-root", str(skill_root),
                "--repo-root", str(root),
                "--case", "alpha-001", "--results-dir", str(results),
                "--dry-run", check=True,
            )
            self.assertIn("alpha-001", completed.stdout)
            plan = json.loads(
                (results / "alpha-001" / "run-plan.json").read_text(encoding="utf-8")
            )
            generation = plan["generation_command"]
            grading = plan["grading_command"]
            self.assertEqual("claude", generation[0])
            self.assertIn("--plugin-dir", generation)
            self.assertIn(str(skill_root.parent), generation)
            self.assertIn("--no-session-persistence", generation)
            self.assertIn("--permission-mode", generation)
            self.assertIn("--disallowedTools", generation)
            self.assertNotIn("Edit", generation[generation.index("--tools") + 1])
            self.assertNotIn("Bash", generation[generation.index("--tools") + 1])
            self.assertNotIn(
                "Bash", generation[generation.index("--allowedTools") + 1]
            )
            self.assertEqual(
                "", generation[generation.index("--setting-sources") + 1]
            )
            self.assertNotIn("--plugin-dir", grading)
            self.assertIn("--json-schema", grading)
            effective_schema = json.loads(grading[grading.index("--json-schema") + 1])
            self.assertNotIn("$schema", effective_schema)
            provenance = plan["provenance"]
            self.assertEqual("claude", provenance["harness"]["name"])
            self.assertEqual([], provenance["harness"]["setting_sources"])
            self.assertEqual(
                manifest.read_bytes(),
                (skill_root.parent / ".claude-plugin" / "plugin.json").read_bytes(),
            )

    def test_claude_run_extracts_stream_result_and_structured_grade(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            _write_claude_plugin_manifest(skill_root)
            fake = _write_fake_claude(root)
            _prepare_live_claude_repo(root)
            results = root / "results"
            completed = self._run(
                "--harness", "claude", "--skill-root", str(skill_root),
                "--repo-root", str(root), "--claude", str(fake),
                "--case", "alpha-001",
                "--results-dir", str(results), "--allow-external-skill-upload",
                "--run", check=True,
            )
            case_dir = results / "alpha-001"
            self.assertIn("PASS", completed.stdout)
            self.assertEqual(
                "# Alpha answer\n",
                (case_dir / "answer.md").read_text(encoding="utf-8"),
            )
            grade = json.loads((case_dir / "grade.json").read_text(encoding="utf-8"))
            self.assertTrue(grade["passed"])
            trace = (case_dir / "generation.trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"type": "result"', trace)
            plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
            self.assertIn(plan["working_directory"], trace)
            self.assertNotEqual(str(root), plan["working_directory"])
            self.assertEqual(
                "2.1.211 (Claude Code test)",
                plan["provenance"]["harness"]["version"],
            )
            self.assertIsNone(plan["provenance"]["harness"]["model"])
            self.assertEqual(str(root), plan["repo_root"])
            status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(["claude-default-test"], status["generation_models_observed"])
            self.assertEqual(["claude-default-test"], status["grading_models_observed"])
            run_status = json.loads(
                (results / "run-status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                ["claude-default-test"],
                run_status["observations"]["generation_models"],
            )
            self.assertEqual(
                ["claude-default-test"],
                run_status["observations"]["grading_models"],
            )
            self.assertAlmostEqual(0.03, run_status["observations"]["total_cost_usd"])

    def test_claude_disables_claude_md_and_auto_memory_for_both_processes(self) -> None:
        completed, case_dir = self._run_fake_claude("isolation")
        self.assertEqual(0, completed.returncode, completed.stderr)
        plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
        self.assertEqual(
            {
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
            },
            plan["provenance"]["harness"]["environment_overrides"],
        )

    def test_claude_generation_hides_eval_answer_keys_in_a_sanitized_workspace(self) -> None:
        completed, case_dir = self._run_fake_claude("answer-key-isolation")
        self.assertEqual(0, completed.returncode, completed.stderr)
        plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
        source_root = case_dir.parents[1]
        self.assertEqual(str(source_root), plan["repo_root"])
        self.assertNotEqual(str(source_root), plan["working_directory"])
        isolation = plan["provenance"]["evaluation_workspace"]
        self.assertTrue(isolation["enabled"])
        self.assertTrue(isolation["answer_keys_excluded"])
        self.assertEqual(str(source_root), isolation["source_root"])
        self.assertEqual("git-tracked-working-tree-allowlist", isolation["method"])
        manifest = {entry["path"]: entry for entry in isolation["content_manifest"]}
        self.assertIn("skills/alpha/SKILL.md", manifest)
        self.assertIn("internal-runtime-link", manifest)
        self.assertNotIn("external-runtime-link", manifest)
        self.assertNotIn("tmp_RUNLOG.md", manifest)
        self.assertNotIn("tmp_TRACKED.md", manifest)
        self.assertNotIn("skills/alpha/assets/VALIDATION.md", manifest)
        self.assertNotIn(
            "skills/alpha/scripts/tests/test_eval_audit.py",
            manifest,
        )
        self.assertNotIn("nested/tmp_CASE/answer.md", manifest)
        self.assertNotIn("nested/results/grade.json", manifest)
        self.assertFalse(any(".egg-info" in path for path in manifest))
        self.assertTrue(isolation["untracked_paths_excluded"])
        self.assertIn("**/tmp_*/**", isolation["generated_path_patterns_excluded"])
        self.assertIn("**/results/**", isolation["generated_path_patterns_excluded"])
        self.assertEqual(
            [
                ".claude-plugin/plugin.json",
                "skills/alpha/SKILL.md",
            ],
            isolation["required_paths"],
        )
        self.assertEqual(
            ["external-runtime-link"],
            [entry["path"] for entry in isolation["excluded_symlinks"]],
        )

    def test_claude_live_run_requires_a_git_tracked_source_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            _write_claude_plugin_manifest(skill_root)
            fake = _write_fake_claude(root)
            completed = self._run(
                "--harness", "claude", "--skill-root", str(skill_root),
                "--repo-root", str(root), "--claude", str(fake),
                "--case", "alpha-001", "--results-dir", str(root / "results"),
                "--allow-external-skill-upload", "--run",
            )
            self.assertEqual(2, completed.returncode)
            self.assertIn("Git-tracked", completed.stderr)

    def test_claude_structured_transport_failure_is_runner_classified_skip(self) -> None:
        completed, case_dir = self._run_fake_claude("transport")
        self.assertEqual(0, completed.returncode)
        self.assertIn("SKIP", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("infrastructure-skip", status["status"])
        self.assertEqual("network", status["reason_category"])
        self.assertEqual("trace:result", status["evidence_source"])
        self.assertEqual(["claude-default-test"], status["generation_models_observed"])
        self.assertEqual(0.015, status["generation_cost_usd"])
        self.assertTrue(status["generation_cost_reported"])

    def test_claude_api_status_is_runner_classified_skip(self) -> None:
        completed, case_dir = self._run_fake_claude("rate-limit-status")
        self.assertEqual(0, completed.returncode)
        self.assertIn("SKIP", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("infrastructure-skip", status["status"])
        self.assertEqual("rate-limit", status["reason_category"])
        self.assertEqual("trace:result.api_error_status", status["evidence_source"])

    def test_claude_policy_refusal_is_runner_classified_skip(self) -> None:
        completed, case_dir = self._run_fake_claude("policy-refusal")
        self.assertEqual(0, completed.returncode)
        self.assertIn("SKIP", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("infrastructure-skip", status["status"])
        self.assertEqual("model-policy-refusal", status["reason_category"])
        self.assertEqual(
            "trace:system.model_refusal_no_fallback",
            status["evidence_source"],
        )

    def test_claude_budget_exhaustion_is_runner_classified_skip(self) -> None:
        completed, case_dir = self._run_fake_claude("budget-exhausted")
        self.assertEqual(0, completed.returncode)
        self.assertIn("SKIP", completed.stdout)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("infrastructure-skip", status["status"])
        self.assertEqual("budget-exhausted", status["reason_category"])
        self.assertEqual("trace:result.subtype", status["evidence_source"])
        self.assertEqual(0.35, status["generation_cost_usd"])

    def test_claude_empty_generation_result_is_harness_error(self) -> None:
        completed, case_dir = self._run_fake_claude("empty-result")
        self.assertEqual(2, completed.returncode)
        self.assertIn("empty result", completed.stderr)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("generation-output-error", status["status"])

    def test_claude_missing_structured_grade_is_harness_error(self) -> None:
        completed, case_dir = self._run_fake_claude("missing-structured-grade")
        self.assertEqual(2, completed.returncode)
        self.assertIn("structured_output", completed.stderr)
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual("grading-output-error", status["status"])

    def test_legacy_codex_grade_schema_path_remains_supported(self) -> None:
        legacy_schema = SCRIPT.parent / "codex_grade.schema.json"
        self.assertTrue(legacy_schema.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_root = root / "skills"
            _write_suite(skill_root)
            completed = self._run(
                "--skill-root", str(skill_root), "--repo-root", str(root),
                "--case", "alpha-001", "--grade-schema", str(legacy_schema),
                "--results-dir", str(root / "results"), "--dry-run",
            )
            self.assertEqual(0, completed.returncode, completed.stderr)

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
            status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
            self.assertIsNone(status["generation_cost_usd"])
            self.assertFalse(status["generation_cost_reported"])
            run_status = json.loads(
                (results / "run-status.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(run_status["observations"]["total_cost_usd"])

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

    def _run_fake_claude(
        self, mode: str
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        skill_root = root / "skills"
        _write_suite(skill_root)
        _write_claude_plugin_manifest(skill_root)
        validation = skill_root / "alpha" / "assets" / "VALIDATION.md"
        validation.parent.mkdir()
        validation.write_text("prior evaluation outcome\n", encoding="utf-8")
        fake = _write_fake_claude(root, mode)
        _prepare_live_claude_repo(root)
        if mode == "answer-key-isolation":
            (root / "tmp_RUNLOG.md").write_text("prior result\n", encoding="utf-8")
            egg_info = root / "src" / "alpha.egg-info"
            egg_info.mkdir(parents=True)
            (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
            skill = skill_root / "alpha" / "SKILL.md"
            skill.write_text(
                skill.read_text(encoding="utf-8") + "\ndirty tracked marker\n",
                encoding="utf-8",
            )
        results = root / "results"
        completed = self._run(
            "--harness", "claude", "--skill-root", str(skill_root),
            "--repo-root", str(root), "--claude", str(fake),
            "--case", "alpha-001", "--results-dir", str(results),
            "--allow-external-skill-upload", "--run",
        )
        return completed, results / "alpha-001"


if __name__ == "__main__":
    unittest.main()
