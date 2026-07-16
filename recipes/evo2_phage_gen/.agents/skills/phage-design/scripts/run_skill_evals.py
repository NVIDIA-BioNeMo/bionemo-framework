#!/usr/bin/env python3
"""Validate portable Agent Skills eval JSON and run selected cases with Codex.

The eval files retain the BioNeMo Agent Toolkit envelope. Codex-specific
execution, provenance capture, and grading live here so the cases remain
usable by other harnesses.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Sequence


REQUIRED_CASE_FIELDS = (
    "id",
    "prompt",
    "expected_output",
    "assertions",
    "expected_skill",
    "expected_script",
)
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
INFRASTRUCTURE_PATTERNS = (
    (
        "network",
        re.compile(
            r"(?:network connection (?:is )?unavailable|connection (?:refused|reset)|"
            r"temporary failure in name resolution|could not resolve host|"
            r"dns (?:lookup|resolution) failed)",
            re.IGNORECASE,
        ),
    ),
    (
        "authentication",
        re.compile(
            r"(?:failed to refresh (?:auth )?token|not logged in|"
            r"authentication (?:failed|required)|unauthorized(?: request)?|"
            r"\bhttp\s*401\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "rate-limit",
        re.compile(
            r"(?:\b(?:http\s*)?429\b|too many requests|rate limit(?:ed| exceeded))",
            re.IGNORECASE,
        ),
    ),
    (
        "service",
        re.compile(
            r"(?:service unavailable|upstream service error|\bhttp\s*50[234]\b)",
            re.IGNORECASE,
        ),
    ),
)
STRUCTURED_ERROR_TYPES = {"error", "turn.failed", "turn_failed", "response.failed"}


class EvalError(RuntimeError):
    """Raised for invalid suites or failed harness operations."""


@dataclass(frozen=True)
class EvalCase:
    skill_name: str
    source: Path
    payload: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload["id"])


@dataclass(frozen=True)
class PreparedCase:
    case: EvalCase
    case_dir: Path
    generation: list[str]
    grading: list[str]


def _plural(count: int, singular: str) -> str:
    return f"{count} {singular}" if count == 1 else f"{count} {singular}s"


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_case(case: Any, *, source: Path, index: int, skill_name: str) -> list[str]:
    where = f"{source}: evals[{index}]"
    if not isinstance(case, dict):
        return [f"{where} must be an object"]
    errors: list[str] = []
    for field in REQUIRED_CASE_FIELDS:
        if field not in case:
            errors.append(f"{where} is missing required field {field!r}")
    if errors:
        return errors
    for field in ("id", "prompt", "expected_output"):
        if not isinstance(case[field], str) or not case[field].strip():
            errors.append(f"{where}.{field} must be a non-empty string")
    if isinstance(case["id"], str) and not SAFE_ID.fullmatch(case["id"]):
        errors.append(f"{where}.id must match {SAFE_ID.pattern}")
    assertions = case["assertions"]
    if not isinstance(assertions, list) or not assertions:
        errors.append(f"{where}.assertions must be a non-empty string array")
    elif any(not isinstance(item, str) or not item.strip() for item in assertions):
        errors.append(f"{where}.assertions contains an empty or non-string assertion")
    expected_skill = case["expected_skill"]
    if expected_skill is not None and (
        not isinstance(expected_skill, str) or not expected_skill.strip()
    ):
        errors.append(f"{where}.expected_skill must be a non-empty string or null")
    elif isinstance(expected_skill, str) and expected_skill != skill_name:
        errors.append(
            f"{where}.expected_skill {expected_skill!r} does not match owning skill {skill_name!r}"
        )
    expected_script = case["expected_script"]
    if expected_script is not None and (
        not isinstance(expected_script, str) or not expected_script.strip()
    ):
        errors.append(f"{where}.expected_script must be a non-empty string or null")
    return errors


def load_cases(skill_root: Path) -> tuple[list[EvalCase], list[Path]]:
    skill_root = skill_root.resolve()
    files = sorted(
        path for path in skill_root.glob("*/evals/*.json") if not path.name.endswith(".schema.json")
    )
    if not files:
        raise EvalError(f"no eval JSON found under {skill_root}/*/evals/")
    cases: list[EvalCase] = []
    errors: list[str] = []
    seen: dict[str, Path] = {}
    for source in files:
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{source}: cannot parse JSON: {exc}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"{source}: top level must be an object")
            continue
        skill_name = payload.get("skill_name")
        if not isinstance(skill_name, str) or not skill_name.strip():
            errors.append(f"{source}: skill_name must be a non-empty string")
            continue
        owner = source.parent.parent.name
        if skill_name != owner:
            errors.append(f"{source}: skill_name {skill_name!r} does not match directory {owner!r}")
        raw_cases = payload.get("evals")
        if not isinstance(raw_cases, list) or not raw_cases:
            errors.append(f"{source}: evals must be a non-empty array")
            continue
        for index, raw_case in enumerate(raw_cases):
            errors.extend(_validate_case(raw_case, source=source, index=index, skill_name=skill_name))
            if not isinstance(raw_case, dict) or not isinstance(raw_case.get("id"), str):
                continue
            case_id = raw_case["id"]
            if case_id in seen:
                errors.append(
                    f"{source}: duplicate eval id {case_id!r}; first seen in {seen[case_id]}"
                )
            else:
                seen[case_id] = source
            cases.append(EvalCase(skill_name=skill_name, source=source, payload=raw_case))
    if errors:
        raise EvalError("\n".join(errors))
    return cases, files


def select_cases(cases: Sequence[EvalCase], selected_ids: Sequence[str] | None) -> list[EvalCase]:
    if not selected_ids:
        return list(cases)
    by_id = {case.id: case for case in cases}
    missing = sorted(set(selected_ids) - set(by_id))
    if missing:
        raise EvalError(f"unknown eval case(s): {', '.join(missing)}")
    wanted = set(selected_ids)
    return [case for case in cases if case.id in wanted]


def build_commands(
    *,
    codex: str,
    repo_root: Path,
    case_dir: Path,
    grade_schema: Path,
    sandbox: str,
    model: str | None,
) -> tuple[list[str], list[str]]:
    base = [codex, "exec", "--ephemeral", "--json", "-s", sandbox, "-C", str(repo_root)]
    if model:
        base.extend(["-m", model])
    generation = [*base, "-o", str(case_dir / "answer.md")]
    grading = [
        *base,
        "--output-schema",
        str(grade_schema),
        "-o",
        str(case_dir / "grade.json"),
    ]
    return generation, grading


def _trace_summary(trace: str, expected_skill: str | None) -> dict[str, Any]:
    types: Counter[str] = Counter()
    parsed = 0
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed += 1
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            types[event["type"]] += 1
    skill_marker = f".agents/skills/{expected_skill}/SKILL.md" if expected_skill else None
    return {
        "bytes": len(trace.encode("utf-8")),
        "json_events": parsed,
        "event_types": dict(sorted(types.items())),
        "command_execution_markers": trace.count("command_execution"),
        "expected_skill_path_observed": bool(skill_marker and skill_marker in trace),
    }


def _grader_prompt(case: EvalCase, answer: str, trace_summary: dict[str, Any]) -> str:
    contract = {
        "case_id": case.id,
        "prompt": case.payload["prompt"],
        "expected_output": case.payload["expected_output"],
        "assertions": case.payload["assertions"],
        "expected_skill": case.payload["expected_skill"],
        "expected_script": case.payload["expected_script"],
        "trace_summary": trace_summary,
        "answer": answer,
    }
    return (
        "Grade one Agent Skill eval. Judge the answer against every assertion using only "
        "observable evidence in the answer and trace summary. Do not require the answer to name "
        "the skill explicitly. Fail invented evidence, missing required outputs, unsafe action, or "
        "materially unjustified assumptions. Return outcome=pass or outcome=fail; infrastructure "
        "classification belongs only to the runner. The top-level passed value must equal the "
        "conjunction of the per-assertion verdicts. Return one result for every supplied assertion "
        "in the same order, repeat its exact text, and provide non-empty evidence and summary.\n\n"
        "EVAL CONTRACT:\n"
        + json.dumps(contract, indent=2, sort_keys=True)
    )


def _run_process(
    command: Sequence[str], *, prompt: str, timeout: int
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise EvalError(f"cannot execute {command[0]!r}: {exc}") from exc
    except subprocess.TimeoutExpired as exc:
        raise EvalError(
            f"command timed out after {timeout} seconds: {' '.join(command[:3])}"
        ) from exc


def _structured_error_messages(trace: str) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") not in STRUCTURED_ERROR_TYPES:
            continue
        candidates: list[Any] = [event.get("message")]
        error = event.get("error")
        if isinstance(error, dict):
            candidates.extend([error.get("message"), error.get("detail")])
        else:
            candidates.append(error)
        data = event.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("message"), data.get("error")])
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                messages.append((f"trace:{event['type']}", candidate))
    return messages


def _classify_infrastructure_failure(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, str] | None:
    if completed.returncode == 0:
        return None
    sources = [("stderr", completed.stderr), *_structured_error_messages(completed.stdout)]
    for evidence_source, message in sources:
        for category, pattern in INFRASTRUCTURE_PATTERNS:
            match = pattern.search(message)
            if match:
                return {
                    "reason_category": category,
                    "matched_marker": match.group(0).lower(),
                    "evidence_source": evidence_source,
                }
    return None


def _write_infrastructure_skip(
    case_dir: Path,
    *,
    phase: str,
    completed: subprocess.CompletedProcess[str],
    classification: dict[str, str],
) -> None:
    _write_json(
        case_dir / "status.json",
        {
            "status": "infrastructure-skip",
            "phase": phase,
            "exit_code": completed.returncode,
            **classification,
        },
    )


def _validate_grade(case: EvalCase, grade_path: Path) -> dict[str, Any]:
    try:
        grade = json.loads(grade_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"{case.id}: invalid structured grade: {exc}") from exc
    if not isinstance(grade, dict):
        raise EvalError(f"{case.id}: structured grade must be an object")
    results = grade.get("assertions")
    if grade.get("case_id") != case.id:
        raise EvalError(f"{case.id}: grader returned mismatched case_id")
    expected_assertions = case.payload["assertions"]
    if not isinstance(results, list) or len(results) != len(expected_assertions):
        raise EvalError(f"{case.id}: grader did not return one result per assertion")
    verdicts: list[bool] = []
    for index, (item, expected) in enumerate(zip(results, expected_assertions, strict=True)):
        if not isinstance(item, dict) or item.get("assertion") != expected:
            raise EvalError(f"{case.id}: grader assertion text/order mismatch at index {index}")
        verdict = item.get("passed")
        if not isinstance(verdict, bool):
            raise EvalError(f"{case.id}: grader assertion verdict at index {index} is malformed")
        if not isinstance(item.get("evidence"), str) or not item["evidence"].strip():
            raise EvalError(f"{case.id}: grader assertion {index} requires non-empty evidence")
        verdicts.append(verdict)
    outcome = grade.get("outcome")
    if outcome not in {"pass", "fail"}:
        raise EvalError(f"{case.id}: grader outcome must be pass or fail")
    if not isinstance(grade.get("summary"), str) or not grade["summary"].strip():
        raise EvalError(f"{case.id}: grader summary must be non-empty")
    expected_pass = all(verdicts)
    if not isinstance(grade.get("passed"), bool):
        raise EvalError(f"{case.id}: top-level passed value is malformed")
    if grade["passed"] != expected_pass or (outcome == "pass") != expected_pass:
        raise EvalError(f"{case.id}: top-level grade is inconsistent with assertion verdicts")
    return grade


def _run_capture(command: Sequence[str], *, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return subprocess.CompletedProcess(list(command), 127, "", str(exc))


def _repository_provenance(repo_root: Path) -> dict[str, Any]:
    discovered = _run_capture(["git", "-C", str(repo_root), "rev-parse", "--absolute-git-dir"])
    if discovered.returncode != 0:
        return {
            "worktree": str(repo_root),
            "available": False,
            "revision": None,
            "branch": None,
            "dirty": None,
            "status_entries": [],
            "status_sha256": None,
            "tracked_diff_sha256": None,
            "reason": (discovered.stderr or "not a Git worktree").strip().splitlines()[0],
        }
    git_dir = Path(discovered.stdout.strip()).resolve()
    base = [f"--git-dir={git_dir}", f"--work-tree={repo_root}"]
    revision = _run_capture(["git", *base, "rev-parse", "HEAD"])
    branch = _run_capture(["git", *base, "branch", "--show-current"])
    status = _run_capture(["git", *base, "status", "--short", "--untracked-files=all"])
    diff = _run_capture(["git", *base, "diff", "--binary", "HEAD"])
    status_text = status.stdout if status.returncode == 0 else ""
    entries = [line for line in status_text.splitlines() if line]
    return {
        "worktree": str(repo_root),
        "git_dir": str(git_dir),
        "available": revision.returncode == 0,
        "revision": revision.stdout.strip() or None,
        "branch": branch.stdout.strip() or None,
        "dirty": bool(entries) if status.returncode == 0 else None,
        "status_entries": entries,
        "status_sha256": _sha256_text(status_text) if status.returncode == 0 else None,
        "tracked_diff_sha256": _sha256_text(diff.stdout) if diff.returncode == 0 else None,
    }


def _instruction_manifest(skill_root: Path) -> tuple[list[dict[str, Any]], str]:
    candidates: set[Path] = set(skill_root.glob("*/SKILL.md"))
    for skill_dir in skill_root.iterdir():
        if not skill_dir.is_dir():
            continue
        references = skill_dir / "references"
        if references.is_dir():
            candidates.update(path for path in references.rglob("*") if path.is_file())
    rows = [
        {
            "path": path.relative_to(skill_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(candidates)
    ]
    bundle = _sha256_text(
        "\n".join(f"{row['path']}\t{row['sha256']}\t{row['bytes']}" for row in rows)
    )
    return rows, bundle


def _codex_provenance(
    codex: str, *, live: bool, model: str | None, sandbox: str
) -> dict[str, Any]:
    resolved = shutil.which(codex)
    if resolved is None and Path(codex).is_file():
        resolved = str(Path(codex).resolve())
    version = "dry-run-not-probed"
    if live:
        completed = _run_process([codex, "--version"], prompt="", timeout=30)
        if completed.returncode != 0:
            raise EvalError(
                f"cannot record Codex version: {completed.stderr.strip() or completed.returncode}"
            )
        version = (completed.stdout or completed.stderr).strip().splitlines()[0]
        if not version:
            raise EvalError("cannot record Codex version: command returned no version")
    config_root = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    config = config_root / "config.toml"
    try:
        config_hash = _sha256_file(config) if config.is_file() else None
    except OSError:
        config_hash = None
    return {
        "requested_executable": codex,
        "resolved_executable": resolved,
        "executable_sha256": (
            _sha256_file(Path(resolved)) if resolved and Path(resolved).is_file() else None
        ),
        "version": version,
        "model": model,
        "model_source": "--model" if model else None,
        "sandbox": sandbox,
        "ephemeral": True,
        "json_trace": True,
        "user_config_present": config.is_file(),
        "user_config_sha256": config_hash,
    }


def build_provenance(
    *,
    selected: Sequence[EvalCase],
    skill_root: Path,
    repo_root: Path,
    codex: str,
    model: str | None,
    sandbox: str,
    grade_schema: Path,
    live: bool,
    argv: Sequence[str],
) -> dict[str, Any]:
    instruction_files, instruction_bundle = _instruction_manifest(skill_root)
    sources = sorted({case.source for case in selected})
    case_rows = [
        {
            "id": case.id,
            "source": str(case.source),
            "canonical_sha256": _sha256_text(
                json.dumps(case.payload, sort_keys=True, separators=(",", ":"))
            ),
        }
        for case in selected
    ]
    runner = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "path": str(runner),
            "sha256": _sha256_file(runner),
            "python": sys.version,
            "platform": sys.platform,
            "argv": list(argv),
        },
        "codex": _codex_provenance(codex, live=live, model=model, sandbox=sandbox),
        "repository": _repository_provenance(repo_root),
        "grade_schema": {"path": str(grade_schema), "sha256": _sha256_file(grade_schema)},
        "eval_sources": [
            {"path": str(source), "sha256": _sha256_file(source)} for source in sources
        ],
        "cases": case_rows,
        "instruction_files": instruction_files,
        "instruction_bundle_sha256": instruction_bundle,
    }


def prepare_case(
    case: EvalCase,
    *,
    results_dir: Path,
    repo_root: Path,
    codex: str,
    grade_schema: Path,
    sandbox: str,
    model: str | None,
    provenance: dict[str, Any],
    provenance_sha256: str,
) -> PreparedCase:
    case_dir = results_dir / case.id
    case_dir.mkdir(exist_ok=False)
    captured = {"skill_name": case.skill_name, "source": str(case.source), **case.payload}
    _write_json(case_dir / "case.json", captured)
    generation, grading = build_commands(
        codex=codex,
        repo_root=repo_root,
        case_dir=case_dir,
        grade_schema=grade_schema,
        sandbox=sandbox,
        model=model,
    )
    _write_json(
        case_dir / "run-plan.json",
        {
            "case_id": case.id,
            "repo_root": str(repo_root),
            "prompt_sha256": _sha256_text(case.payload["prompt"]),
            "generation_command": generation,
            "grading_command": grading,
            "grade_schema_sha256": _sha256_file(grade_schema),
            "provenance_path": "../run-provenance.json",
            "provenance_sha256": provenance_sha256,
            "provenance": provenance,
        },
    )
    return PreparedCase(
        case=case,
        case_dir=case_dir,
        generation=generation,
        grading=grading,
    )


def run_prepared_case(prepared: PreparedCase, *, timeout: int) -> str:
    case = prepared.case
    case_dir = prepared.case_dir
    started = datetime.now(timezone.utc).isoformat()
    generated = _run_process(prepared.generation, prompt=case.payload["prompt"], timeout=timeout)
    (case_dir / "generation.trace.jsonl").write_text(generated.stdout, encoding="utf-8")
    (case_dir / "generation.stderr.log").write_text(generated.stderr, encoding="utf-8")
    if generated.returncode != 0:
        classification = _classify_infrastructure_failure(generated)
        if classification:
            _write_infrastructure_skip(
                case_dir,
                phase="generation",
                completed=generated,
                classification=classification,
            )
            return "skip"
        _write_json(
            case_dir / "status.json",
            {"status": "generation-error", "exit_code": generated.returncode},
        )
        raise EvalError(f"{case.id}: Codex generation exited {generated.returncode}")
    answer_path = case_dir / "answer.md"
    if not answer_path.is_file():
        raise EvalError(f"{case.id}: Codex did not write answer.md")
    answer = answer_path.read_text(encoding="utf-8")
    trace_summary = _trace_summary(generated.stdout, case.payload["expected_skill"])
    _write_json(case_dir / "trace-summary.json", trace_summary)
    graded = _run_process(
        prepared.grading,
        prompt=_grader_prompt(case, answer, trace_summary),
        timeout=timeout,
    )
    (case_dir / "grading.trace.jsonl").write_text(graded.stdout, encoding="utf-8")
    (case_dir / "grading.stderr.log").write_text(graded.stderr, encoding="utf-8")
    if graded.returncode != 0:
        classification = _classify_infrastructure_failure(graded)
        if classification:
            _write_infrastructure_skip(
                case_dir,
                phase="grading",
                completed=graded,
                classification=classification,
            )
            return "skip"
        _write_json(
            case_dir / "status.json",
            {"status": "grading-error", "exit_code": graded.returncode},
        )
        raise EvalError(f"{case.id}: Codex grader exited {graded.returncode}")
    grade = _validate_grade(case, case_dir / "grade.json")
    _write_json(
        case_dir / "status.json",
        {
            "status": {"pass": "passed", "fail": "failed"}[grade["outcome"]],
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "answer_sha256": _sha256_file(answer_path),
            "grade_sha256": _sha256_file(case_dir / "grade.json"),
        },
    )
    return str(grade["outcome"])


def _reserve_results_dir(results_dir: Path, selected: Sequence[EvalCase]) -> None:
    try:
        results_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        occupied = [case.id for case in selected if (results_dir / case.id).exists()]
        detail = f"; existing selected cases: {', '.join(occupied)}" if occupied else ""
        raise EvalError(f"occupied result destination: {results_dir}{detail}") from exc
    except OSError as exc:
        raise EvalError(f"cannot reserve result destination {results_dir}: {exc}") from exc


def _parser() -> argparse.ArgumentParser:
    default_skill_root = Path(__file__).resolve().parents[2]
    default_schema = Path(__file__).resolve().with_name("codex_grade.schema.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", type=Path, default=default_skill_root)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--case", action="append", dest="case_ids", help="case ID; repeat to select several")
    parser.add_argument("--all", action="store_true", help="allow --run to execute every discovered case")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--codex", default="codex", help="Codex executable")
    parser.add_argument("--model", help="required explicit Codex model for live --run")
    parser.add_argument("--sandbox", choices=("read-only", "workspace-write"), default="read-only")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--grade-schema", type=Path, default=default_schema)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--validate", action="store_true")
    action.add_argument("--list", action="store_true")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parsed_argv = list(argv) if argv is not None else sys.argv[1:]
    args = _parser().parse_args(parsed_argv)
    results_dir: Path | None = None
    try:
        cases, files = load_cases(args.skill_root)
        selected = select_cases(cases, args.case_ids)
        if args.validate:
            print(f"Validated {_plural(len(files), 'eval file')} with {_plural(len(cases), 'case')}.")
            return 0
        if args.list:
            rows = [
                {"id": case.id, "skill_name": case.skill_name, "source": str(case.source)}
                for case in selected
            ]
            if args.format == "json":
                print(json.dumps(rows, indent=2))
            else:
                for row in rows:
                    print(f"{row['id']}\t{row['skill_name']}\t{row['source']}")
            return 0
        if not args.results_dir:
            raise EvalError("--results-dir is required with --run or --dry-run")
        if args.run and not args.case_ids and not args.all:
            raise EvalError("refusing to run every case without --all; select --case or pass --all")
        if args.run and not args.model:
            raise EvalError("--model is required for live --run reproducibility")
        repo_root = args.repo_root.resolve()
        grade_schema = args.grade_schema.resolve()
        if not grade_schema.is_file():
            raise EvalError(f"grade schema not found: {grade_schema}")
        results_dir = args.results_dir.resolve()
        _reserve_results_dir(results_dir, selected)
        try:
            provenance = build_provenance(
                selected=selected,
                skill_root=args.skill_root.resolve(),
                repo_root=repo_root,
                codex=args.codex,
                model=args.model,
                sandbox=args.sandbox,
                grade_schema=grade_schema,
                live=args.run,
                argv=parsed_argv,
            )
            _write_json(results_dir / "run-provenance.json", provenance)
            provenance_sha256 = _sha256_file(results_dir / "run-provenance.json")
            prepared = [
                prepare_case(
                    case,
                    results_dir=results_dir,
                    repo_root=repo_root,
                    codex=args.codex,
                    grade_schema=grade_schema,
                    sandbox=args.sandbox,
                    model=args.model,
                    provenance=provenance,
                    provenance_sha256=provenance_sha256,
                )
                for case in selected
            ]
        except (EvalError, OSError) as exc:
            _write_json(
                results_dir / "run-status.json",
                {"status": "preflight-failed", "error": str(exc)},
            )
            if isinstance(exc, EvalError):
                raise
            raise EvalError(f"eval preflight failed: {exc}") from exc
        if args.dry_run:
            for prepared_case in prepared:
                print(f"{prepared_case.case.id}: PLANNED")
            _write_json(
                results_dir / "run-status.json",
                {"status": "planned", "case_count": len(prepared)},
            )
            return 0
        counts: Counter[str] = Counter()
        _write_json(
            results_dir / "run-status.json",
            {"status": "running", "case_count": len(prepared)},
        )
        try:
            for prepared_case in prepared:
                verdict = run_prepared_case(prepared_case, timeout=args.timeout_seconds)
                counts[verdict] += 1
                print(f"{prepared_case.case.id}: {verdict.upper()}")
        except EvalError as exc:
            _write_json(
                results_dir / "run-status.json",
                {"status": "harness-error", "error": str(exc), "counts": dict(counts)},
            )
            raise
        _write_json(
            results_dir / "run-status.json",
            {
                "status": "complete-with-failures" if counts["fail"] else "complete",
                "counts": dict(counts),
            },
        )
        return 1 if counts["fail"] else 0
    except EvalError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
