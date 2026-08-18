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

"""Tests for the portable Agent Skills eval validator and harness adapters."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_skill_evals.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_skill_evals", SCRIPT)
assert RUNNER_SPEC and RUNNER_SPEC.loader
runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = runner
RUNNER_SPEC.loader.exec_module(runner)
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
            import os
            from pathlib import Path
            import sys

            MODE = {mode!r}
            ASSERTIONS = {ASSERTIONS!r}
            ORIGINAL_ROOT = {str(root)!r}
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
            if MODE == "answer-key-isolation" and "--output-schema" not in args:
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
                if (cwd / "tmp_RUNLOG.md").exists() or (cwd / "tmp_TRACKED.md").exists():
                    failures.append("run history is visible")
                if list(cwd.rglob("*.egg-info")):
                    failures.append("generated package metadata is visible")
                if (cwd / "external-runtime-link").is_symlink():
                    failures.append("external symlink escaped the staged workspace")
                if not (cwd / "internal-runtime-link").is_symlink():
                    failures.append("safe tracked symlink is missing")
                skill = cwd / "skills" / "alpha" / "SKILL.md"
                if not skill.is_file():
                    failures.append("selected skill is missing")
                elif "dirty tracked marker" not in skill.read_text(encoding="utf-8"):
                    failures.append("tracked working-tree edit is missing")
                if failures:
                    print("; ".join(failures), file=sys.stderr)
                    raise SystemExit(8)
            output = Path(args[args.index("-o") + 1])
            if "--output-schema" not in args:
                output.write_text("# Alpha answer\\n", encoding="utf-8")
                print(json.dumps({{
                    "type": "generation",
                    "prompt_bytes": len(prompt),
                    "cwd": os.getcwd(),
                }}))
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
                "description": "Test bridge for portable Agent Skills.",
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


def _run(
    *args: str,
    check: bool = False,
    use_default_recipe: bool = False,
) -> subprocess.CompletedProcess[str]:
    command_args = list(args)
    if (
        not use_default_recipe
        and "--recipe-root" not in command_args
        and ("--run" in command_args or "--dry-run" in command_args)
    ):
        command_args.extend(["--recipe-root", "."])
    return subprocess.run(
        [sys.executable, str(SCRIPT), *command_args],
        check=check,
        text=True,
        capture_output=True,
    )


def test_validate_accepts_bionemo_compatible_suite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill_root = Path(tmp) / "skills"
        _write_suite(skill_root)
        completed = _run("--skill-root", str(skill_root), "--validate", check=True)
        assert "1 eval file" in completed.stdout
        assert "1 case" in completed.stdout


def test_validate_rejects_duplicate_ids_across_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill_root = Path(tmp) / "skills"
        _write_suite(skill_root, "alpha", [_case("shared-001")])
        _write_suite(skill_root, "beta", [{**_case("shared-001"), "expected_skill": "beta"}])
        completed = _run("--skill-root", str(skill_root), "--validate")
        assert 2 == completed.returncode
        assert "duplicate eval id" in completed.stderr.lower()


def test_validate_rejects_missing_compatible_field() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill_root = Path(tmp) / "skills"
        invalid = _case()
        del invalid["expected_output"]
        _write_suite(skill_root, cases=[invalid])
        completed = _run("--skill-root", str(skill_root), "--validate")
        assert 2 == completed.returncode
        assert "expected_output" in completed.stderr


def test_list_json_reports_owning_file_and_skill() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        skill_root = Path(tmp) / "skills"
        suite = _write_suite(skill_root)
        completed = _run("--skill-root", str(skill_root), "--list", "--format", "json", check=True)
        payload = json.loads(completed.stdout)
        assert "alpha-001" == payload[0]["id"]
        assert "alpha" == payload[0]["skill_name"]
        assert str(suite) == payload[0]["source"]


def test_dry_run_writes_reproducible_plan_without_launching() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        results = root / "results"
        completed = _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--dry-run",
            check=True,
        )
        assert "alpha-001" in completed.stdout
        plan = json.loads((results / "alpha-001" / "run-plan.json").read_text(encoding="utf-8"))
        generation = plan["generation_command"]
        grading = plan["grading_command"]
        assert "--ephemeral" in generation
        assert "--skip-git-repo-check" in generation
        assert "--json" in generation
        assert "read-only" in generation
        assert "--output-schema" in grading
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        assert "dry-run-not-probed" == provenance["codex"]["version"]
        assert "instruction_files" in provenance
        assert "../run-provenance.json" == plan["provenance_path"]
        assert not (results / "alpha-001" / "generation.trace.jsonl").exists()


def test_default_recipe_root_is_relative_to_repository() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo_root = root / "checkout"
        recipe_root = repo_root / "recipes" / "evo2_phage_gen"
        recipe_root.mkdir(parents=True)
        skill_root = root / "installed-plugin" / "skills"
        _write_suite(skill_root)
        results = recipe_root / "results" / "default-recipe"
        _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(repo_root),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--dry-run",
            use_default_recipe=True,
            check=True,
        )
        plan = json.loads((results / "alpha-001" / "run-plan.json").read_text(encoding="utf-8"))
        assert str(repo_root.resolve()) == plan["repo_root"]
        assert str(recipe_root.resolve()) == plan["recipe_root"]
        assert str(recipe_root.resolve()) == plan["working_directory"]
        generation = plan["generation_command"]
        assert str(recipe_root.resolve()) == generation[generation.index("-C") + 1]


def test_live_codex_runs_in_selected_recipe() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        recipe_root = repo_root / "recipes" / "custom_phage"
        recipe_root.mkdir(parents=True)
        (recipe_root / "README.md").write_text("# Custom recipe\n", encoding="utf-8")
        skill_root = repo_root / "installed" / "skills"
        _write_suite(skill_root)
        fake = _write_fake_codex(repo_root, "pass")
        _init_git_repo(repo_root)
        results = recipe_root / "results" / "live-cwd"
        completed = _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(repo_root),
            "--recipe-root",
            "recipes/custom_phage",
            "--codex",
            str(fake),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--run",
            check=True,
        )
        assert "PASS" in completed.stdout
        trace = (results / "alpha-001" / "generation.trace.jsonl").read_text(encoding="utf-8")
        event = json.loads(trace.splitlines()[0])
        assert str(recipe_root.resolve()) != event["cwd"]
        assert event["cwd"].endswith("/recipes/custom_phage")
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        isolation = provenance["evaluation_workspace"]
        assert isolation["enabled"]
        assert isolation["answer_keys_excluded"]
        assert "git-tracked-working-tree" == isolation["method"]
        assert "content_manifest" not in isolation
        assert "content_manifest_sha256" not in isolation


def test_trace_summary_accepts_installation_independent_skill_path() -> None:
    trace = json.dumps(
        {
            "type": "item.completed",
            "path": "/opt/nvidia/skills/alpha/SKILL.md",
        }
    )
    assert runner._trace_summary(trace, "alpha")["expected_skill_path_observed"]


def test_nested_repo_root_provenance_uses_git_toplevel() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        nested = root / "nested" / "recipe"
        nested.mkdir(parents=True)
        _init_git_repo(root)
        results = nested / "results"
        _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(nested),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--dry-run",
            check=True,
        )
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        repository = provenance["repository"]
        assert str(root.resolve()) == repository["worktree"]
        assert not repository["dirty"]


def test_repeated_dry_run_fails_cleanly_before_writing() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        results = root / "results"
        args = (
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--dry-run",
        )
        assert 0 == _run(*args).returncode
        original = (results / "alpha-001" / "run-plan.json").read_bytes()
        repeated = _run(*args)
        assert 2 == repeated.returncode
        assert "occupied result destination" in repeated.stderr
        assert "Traceback" not in repeated.stderr
        assert original == (results / "alpha-001" / "run-plan.json").read_bytes()


def test_partial_multi_case_destination_is_preflighted_atomically() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root, cases=[_case("alpha-001"), _case("alpha-002")])
        results = root / "results"
        occupied = results / "alpha-001"
        occupied.mkdir(parents=True)
        (occupied / "keep.txt").write_text("keep\n", encoding="utf-8")
        completed = _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--all",
            "--results-dir",
            str(results),
            "--dry-run",
        )
        assert 2 == completed.returncode
        assert "alpha-001" in completed.stderr
        assert not (results / "alpha-002").exists()
        assert "keep\n" == (occupied / "keep.txt").read_text(encoding="utf-8")


def test_live_run_may_use_cli_default_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        fake = _write_fake_codex(root, "pass")
        _init_git_repo(root)
        completed = _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--codex",
            str(fake),
            "--case",
            "alpha-001",
            "--results-dir",
            str(root / "results"),
            "--run",
            check=True,
        )
        assert "PASS" in completed.stdout
        plan = json.loads((root / "results" / "alpha-001" / "run-plan.json").read_text(encoding="utf-8"))
        provenance = json.loads((root / "results" / "run-provenance.json").read_text(encoding="utf-8"))
        assert provenance["harness"]["model"] is None
        assert "-m" not in plan["generation_command"]


def test_claude_live_run_requires_explicit_external_upload_consent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        _write_claude_plugin_manifest(skill_root)
        fake = _write_fake_claude(root)
        completed = _run(
            "--harness",
            "claude",
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--claude",
            str(fake),
            "--case",
            "alpha-001",
            "--results-dir",
            str(root / "results"),
            "--run",
        )
        assert 2 == completed.returncode
        assert "--allow-external-skill-upload" in completed.stderr
        assert "staged recipe files it reads" in completed.stderr


def test_claude_dry_run_uses_local_plugin_and_read_only_tools() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        manifest = _write_claude_plugin_manifest(skill_root)
        results = root / "results"
        completed = _run(
            "--harness",
            "claude",
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--dry-run",
            check=True,
        )
        assert "alpha-001" in completed.stdout
        plan = json.loads((results / "alpha-001" / "run-plan.json").read_text(encoding="utf-8"))
        generation = plan["generation_command"]
        grading = plan["grading_command"]
        assert "claude" == generation[0]
        assert "--plugin-dir" in generation
        assert str(skill_root.parent) in generation
        assert "--no-session-persistence" in generation
        assert "--permission-mode" in generation
        assert "--disallowedTools" in generation
        assert "Edit" not in generation[generation.index("--tools") + 1]
        assert "Bash" not in generation[generation.index("--tools") + 1]
        assert "Bash" not in generation[generation.index("--allowedTools") + 1]
        assert "" == generation[generation.index("--setting-sources") + 1]
        assert "--plugin-dir" not in grading
        assert "--json-schema" in grading
        effective_schema = json.loads(grading[grading.index("--json-schema") + 1])
        assert "$schema" not in effective_schema
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        assert "claude" == provenance["harness"]["name"]
        assert [] == provenance["harness"]["setting_sources"]
        assert manifest.read_bytes() == (skill_root.parent / ".claude-plugin" / "plugin.json").read_bytes()


def test_claude_run_extracts_stream_result_and_structured_grade() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        _write_claude_plugin_manifest(skill_root)
        fake = _write_fake_claude(root)
        _prepare_live_claude_repo(root)
        results = root / "results"
        completed = _run(
            "--harness",
            "claude",
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--claude",
            str(fake),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--allow-external-skill-upload",
            "--run",
            check=True,
        )
        case_dir = results / "alpha-001"
        assert "PASS" in completed.stdout
        assert "# Alpha answer\n" == (case_dir / "answer.md").read_text(encoding="utf-8")
        grade = json.loads((case_dir / "grade.json").read_text(encoding="utf-8"))
        assert grade["passed"]
        trace = (case_dir / "generation.trace.jsonl").read_text(encoding="utf-8")
        assert '"type": "result"' in trace
        plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
        assert plan["working_directory"] in trace
        assert str(root) != plan["working_directory"]
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        assert "2.1.211 (Claude Code test)" == provenance["harness"]["version"]
        assert provenance["harness"]["model"] is None
        assert str(root) == plan["repo_root"]
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        assert ["claude-default-test"] == status["generation_models_observed"]
        assert ["claude-default-test"] == status["grading_models_observed"]
        run_status = json.loads((results / "run-status.json").read_text(encoding="utf-8"))
        assert ["claude-default-test"] == run_status["observations"]["generation_models"]
        assert ["claude-default-test"] == run_status["observations"]["grading_models"]
        assert round(abs(0.03 - run_status["observations"]["total_cost_usd"]), 7) == 0


def test_claude_preserves_recipe_cwd_with_external_tracked_plugin() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        repo_root = root / "checkout"
        recipe_root = repo_root / "recipes" / "evo2_phage_gen"
        recipe_root.mkdir(parents=True)
        (recipe_root / "README.md").write_text("# Recipe\n", encoding="utf-8")
        _init_git_repo(repo_root)

        plugin_root = root / "installed-plugin"
        skill_root = plugin_root / "skills"
        _write_suite(skill_root)
        _write_claude_plugin_manifest(skill_root)
        _init_git_repo(plugin_root)

        fake = _write_fake_claude(root)
        results = recipe_root / "results" / "external-plugin"
        completed = _run(
            "--harness",
            "claude",
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(repo_root),
            "--recipe-root",
            "recipes/evo2_phage_gen",
            "--claude",
            str(fake),
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--allow-external-skill-upload",
            "--run",
            check=True,
        )
        assert "PASS" in completed.stdout
        case_dir = results / "alpha-001"
        plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
        assert plan["working_directory"].endswith("/recipes/evo2_phage_gen")
        trace = (case_dir / "generation.trace.jsonl").read_text(encoding="utf-8")
        assert plan["working_directory"] in trace
        generation = plan["generation_command"]
        staged_plugin = generation[generation.index("--plugin-dir") + 1]
        assert "/.external-skill-plugin" in staged_plugin
        assert str(plugin_root.resolve()) != staged_plugin
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        assert str(recipe_root.resolve()) == provenance["harness"]["source_working_directory"]
        isolation = provenance["evaluation_workspace"]
        assert "git-tracked-working-tree-plus-sanitized-plugin" == isolation["method"]
        external = isolation["external_plugin_workspace"]
        assert external["answer_keys_excluded"]
        assert [
            ".claude-plugin/plugin.json",
            "skills/alpha/SKILL.md",
        ] == external["required_paths"]


def test_claude_disables_claude_md_and_auto_memory_for_both_processes(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("isolation", tmp_path)
    assert 0 == completed.returncode, completed.stderr
    assert {
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
        "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
    } == json.loads((case_dir.parent / "run-provenance.json").read_text(encoding="utf-8"))["harness"][
        "environment_overrides"
    ]


def test_claude_generation_hides_eval_answer_keys_in_a_sanitized_workspace(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("answer-key-isolation", tmp_path)
    assert 0 == completed.returncode, completed.stderr
    plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
    source_root = case_dir.parents[1]
    assert str(source_root) == plan["repo_root"]
    assert str(source_root) != plan["working_directory"]
    provenance = json.loads((case_dir.parent / "run-provenance.json").read_text(encoding="utf-8"))
    isolation = provenance["evaluation_workspace"]
    assert isolation["enabled"]
    assert isolation["answer_keys_excluded"]
    assert str(source_root) == isolation["source_root"]
    assert "git-tracked-working-tree" == isolation["method"]
    assert "content_manifest" not in isolation
    assert "content_manifest_sha256" not in isolation
    assert isolation["untracked_paths_excluded"]
    assert "**/tmp_*/**" in isolation["generated_path_patterns_excluded"]
    assert "**/results/**" in isolation["generated_path_patterns_excluded"]
    assert [
        ".claude-plugin/plugin.json",
        "skills/alpha/SKILL.md",
    ] == isolation["required_paths"]
    assert ["external-runtime-link"] == [entry["path"] for entry in isolation["excluded_symlinks"]]


def test_codex_generation_hides_eval_answer_keys_in_a_sanitized_workspace(tmp_path: Path) -> None:
    completed, case_dir = _run_fake("answer-key-isolation", tmp_path)
    assert 0 == completed.returncode, completed.stderr
    plan = json.loads((case_dir / "run-plan.json").read_text(encoding="utf-8"))
    source_root = case_dir.parents[1]
    assert str(source_root) == plan["repo_root"]
    assert str(source_root) != plan["working_directory"]
    provenance = json.loads((case_dir.parent / "run-provenance.json").read_text(encoding="utf-8"))
    isolation = provenance["evaluation_workspace"]
    assert isolation["enabled"]
    assert isolation["answer_keys_excluded"]
    assert "git-tracked-working-tree" == isolation["method"]
    assert "content_manifest" not in isolation
    assert "content_manifest_sha256" not in isolation


def test_claude_live_run_requires_a_git_tracked_source_workspace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        _write_claude_plugin_manifest(skill_root)
        fake = _write_fake_claude(root)
        completed = _run(
            "--harness",
            "claude",
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--claude",
            str(fake),
            "--case",
            "alpha-001",
            "--results-dir",
            str(root / "results"),
            "--allow-external-skill-upload",
            "--run",
        )
        assert 2 == completed.returncode
        assert "Git-tracked" in completed.stderr


def test_claude_structured_transport_failure_is_runner_classified_skip(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("transport", tmp_path)
    assert 0 == completed.returncode
    assert "SKIP" in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "infrastructure-skip" == status["status"]
    assert "network" == status["reason_category"]
    assert "trace:result" == status["evidence_source"]
    assert ["claude-default-test"] == status["generation_models_observed"]
    assert 0.015 == status["generation_cost_usd"]
    assert status["generation_cost_reported"]


def test_claude_api_status_is_runner_classified_skip(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("rate-limit-status", tmp_path)
    assert 0 == completed.returncode
    assert "SKIP" in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "infrastructure-skip" == status["status"]
    assert "rate-limit" == status["reason_category"]
    assert "trace:result.api_error_status" == status["evidence_source"]


def test_claude_policy_refusal_is_runner_classified_skip(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("policy-refusal", tmp_path)
    assert 0 == completed.returncode
    assert "SKIP" in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "infrastructure-skip" == status["status"]
    assert "model-policy-refusal" == status["reason_category"]
    assert "trace:system.model_refusal_no_fallback" == status["evidence_source"]


def test_claude_budget_exhaustion_is_runner_classified_skip(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("budget-exhausted", tmp_path)
    assert 0 == completed.returncode
    assert "SKIP" in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "infrastructure-skip" == status["status"]
    assert "budget-exhausted" == status["reason_category"]
    assert "trace:result.subtype" == status["evidence_source"]
    assert 0.35 == status["generation_cost_usd"]


def test_claude_empty_generation_result_is_harness_error(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("empty-result", tmp_path)
    assert 2 == completed.returncode
    assert "empty result" in completed.stderr
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "generation-output-error" == status["status"]


def test_claude_missing_structured_grade_is_harness_error(tmp_path: Path) -> None:
    completed, case_dir = _run_fake_claude("missing-structured-grade", tmp_path)
    assert 2 == completed.returncode
    assert "structured_output" in completed.stderr
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "grading-output-error" == status["status"]


def test_legacy_codex_grade_schema_path_remains_supported() -> None:
    legacy_schema = SCRIPT.parent / "codex_grade.schema.json"
    assert legacy_schema.is_file()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        completed = _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--case",
            "alpha-001",
            "--grade-schema",
            str(legacy_schema),
            "--results-dir",
            str(root / "results"),
            "--dry-run",
        )
        assert 0 == completed.returncode, completed.stderr


def test_run_preserves_artifacts_and_reproducibility_provenance() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_root = root / "skills"
        _write_suite(skill_root)
        fake = _write_fake_codex(root, "pass")
        revision = _init_git_repo(root)
        results = root / "results"
        completed = _run(
            "--skill-root",
            str(skill_root),
            "--repo-root",
            str(root),
            "--codex",
            str(fake),
            "--model",
            "test-model",
            "--case",
            "alpha-001",
            "--results-dir",
            str(results),
            "--run",
            check=True,
        )
        case_dir = results / "alpha-001"
        assert "PASS" in completed.stdout
        assert "# Alpha answer\n" == (case_dir / "answer.md").read_text(encoding="utf-8")
        assert '"type": "generation"' in (case_dir / "generation.trace.jsonl").read_text()
        assert '"type": "grader"' in (case_dir / "grading.trace.jsonl").read_text()
        grade = json.loads((case_dir / "grade.json").read_text(encoding="utf-8"))
        assert grade["passed"]
        provenance = json.loads((results / "run-provenance.json").read_text(encoding="utf-8"))
        assert "test-model" == provenance["codex"]["model"]
        assert "codex-cli 9.9-test" == provenance["codex"]["version"]
        assert revision == provenance["repository"]["revision"]
        instruction_paths = {row["path"] for row in provenance["instruction_files"]}
        assert "alpha/SKILL.md" in instruction_paths
        assert "alpha/references/contract.md" in instruction_paths
        status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
        assert status["generation_cost_usd"] is None
        assert not status["generation_cost_reported"]
        run_status = json.loads((results / "run-status.json").read_text(encoding="utf-8"))
        assert run_status["observations"]["total_cost_usd"] is None


def test_grader_authored_skip_is_rejected(tmp_path: Path) -> None:
    completed, _ = _run_fake("grader-skip", tmp_path)
    assert 2 == completed.returncode
    assert "outcome must be pass or fail" in completed.stderr
    assert "SKIP" not in completed.stdout


def test_scientific_no_result_is_a_failed_eval(tmp_path: Path) -> None:
    completed, case_dir = _run_fake("scientific-fail", tmp_path)
    assert 1 == completed.returncode
    assert "FAIL" in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "failed" == status["status"]


def test_assertion_text_mismatch_rejected(tmp_path: Path) -> None:
    completed, _ = _run_fake("mismatch", tmp_path)
    assert 2 == completed.returncode
    assert "assertion text/order" in completed.stderr


def test_empty_assertion_evidence_is_rejected(tmp_path: Path) -> None:
    completed, _ = _run_fake("empty-evidence", tmp_path)
    assert 2 == completed.returncode
    assert "non-empty evidence" in completed.stderr


def test_stdout_rate_limit_words_do_not_turn_unrelated_error_into_skip(tmp_path: Path) -> None:
    completed, case_dir = _run_fake("stdout-marker", tmp_path)
    assert 2 == completed.returncode
    assert "SKIP" not in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "generation-error" == status["status"]


def test_stderr_transport_failure_is_runner_classified_skip(tmp_path: Path) -> None:
    completed, case_dir = _run_fake("transport", tmp_path)
    assert 0 == completed.returncode
    assert "SKIP" in completed.stdout
    status = json.loads((case_dir / "status.json").read_text(encoding="utf-8"))
    assert "infrastructure-skip" == status["status"]
    assert "network" == status["reason_category"]
    assert "generation" == status["phase"]


def _run_fake(mode: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path
    skill_root = root / "skills"
    _write_suite(skill_root)
    validation = skill_root / "alpha" / "assets" / "VALIDATION.md"
    validation.parent.mkdir()
    validation.write_text("prior evaluation outcome\n", encoding="utf-8")
    fake = _write_fake_codex(root, mode)
    _prepare_live_claude_repo(root)
    if mode == "answer-key-isolation":
        (root / "tmp_RUNLOG.md").write_text("prior result\n", encoding="utf-8")
        egg_info = root / "src" / "alpha.egg-info"
        egg_info.mkdir(parents=True)
        (egg_info / "PKG-INFO").write_text("generated\n", encoding="utf-8")
        skill = skill_root / "alpha" / "SKILL.md"
        skill.write_text(skill.read_text(encoding="utf-8") + "\ndirty tracked marker\n", encoding="utf-8")
    results = root / "results"
    completed = _run(
        "--skill-root",
        str(skill_root),
        "--repo-root",
        str(root),
        "--codex",
        str(fake),
        "--model",
        "test-model",
        "--case",
        "alpha-001",
        "--results-dir",
        str(results),
        "--run",
    )
    return completed, results / "alpha-001"


def _run_fake_claude(mode: str, tmp_path: Path) -> tuple[subprocess.CompletedProcess[str], Path]:
    root = tmp_path
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
    completed = _run(
        "--harness",
        "claude",
        "--skill-root",
        str(skill_root),
        "--repo-root",
        str(root),
        "--claude",
        str(fake),
        "--case",
        "alpha-001",
        "--results-dir",
        str(results),
        "--allow-external-skill-upload",
        "--run",
    )
    return completed, results / "alpha-001"


def test_trace_summary_deduplicates_web_calls() -> None:
    trace = "\n".join(
        (
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"id": "web-1", "type": "web_search", "query": "published threshold"},
                }
            ),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "web-1", "type": "web_search", "query": "published threshold"},
                }
            ),
        )
    )

    summary = runner._trace_summary(trace, "bionemo-phage-design-research-evidence")

    assert summary["web_tool_calls_observed"]
    assert summary["web_tool_call_count"] == 1
    assert summary["web_tool_call_types"] == ["web_search"]


def test_codex_grade_schema_uses_api_supported_flat_contract() -> None:
    schema = json.loads((SCRIPT.parent / "codex_grade.schema.json").read_text(encoding="utf-8"))

    assert "allOf" not in schema
    assert "anyOf" not in schema
    assert "oneOf" not in schema
    assert schema["required"] == ["case_id", "outcome", "passed", "assertions", "summary"]


def test_validate_grade_rejects_inconsistent_top_level_verdict(tmp_path: Path) -> None:
    case = runner.EvalCase(skill_name="alpha", source=tmp_path / "evals.json", payload=_case())
    grade_path = tmp_path / "grade.json"
    grade_path.write_text(
        json.dumps(
            {
                "case_id": case.id,
                "outcome": "pass",
                "passed": False,
                "assertions": [
                    {"assertion": assertion, "passed": True, "evidence": "observed"}
                    for assertion in case.payload["assertions"]
                ],
                "summary": "Deliberately inconsistent fixture.",
            }
        ),
        encoding="utf-8",
    )

    try:
        runner._validate_grade(case, grade_path)
    except runner.EvalError as exc:
        assert "inconsistent with assertion verdicts" in str(exc)
    else:
        raise AssertionError("inconsistent top-level verdict was accepted")
