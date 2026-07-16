#!/usr/bin/env python3
"""Validate portable Agent Skills eval JSON and run selected cases.

The eval files retain the BioNeMo Agent Toolkit envelope. Harness-specific
execution, provenance capture, and grading live here so the cases remain
portable.
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
import tempfile
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
CLAUDE_API_STATUS_CATEGORIES = {
    401: "authentication",
    408: "network",
    429: "rate-limit",
    500: "service",
    502: "service",
    503: "service",
    504: "service",
    529: "service",
}
CLAUDE_PLUGIN_NAME = "evo2-phage-gen"
CLAUDE_TOOLS = "Read,Glob,Grep,WebSearch,WebFetch,Skill"
CLAUDE_ALLOWED_TOOLS = CLAUDE_TOOLS
CLAUDE_ENVIRONMENT_OVERRIDES = {
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}
EVALUATION_RESPONSE_CONTRACT = """EVALUATION RESPONSE CONTRACT
- Answer the request directly and self-containedly; do not mutate files or launch jobs.
- Use the selected skill and only the references or repository files needed for this case.
- Work only inside the provided working directory. Do not inspect eval definitions, grading files, or paths outside it.
- Use web tools only when the request requires current or source-backed research.
- Use no more than 1,800 words. Prefer a much shorter answer when it can satisfy the request.
"""
EVALUATION_WORKSPACE_EXCLUDED_TOP_LEVEL = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "data",
    "results",
    "tmp",
    "tmp_inference_artifacts",
}
EVALUATION_WORKSPACE_EXCLUDED_NAMES = {
    "__pycache__",
    "evals",
    "evals.json",
}


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
    harness: str
    working_directory: Path
    plugin_name: str | None
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


def _generation_prompt(case: EvalCase, *, harness: str, plugin_name: str | None) -> str:
    prompt = f"{case.payload['prompt']}\n\n{EVALUATION_RESPONSE_CONTRACT}"
    if harness != "claude":
        return prompt
    if not plugin_name:
        raise EvalError(f"{case.id}: Claude case has no plugin namespace")
    return f"/{plugin_name}:{case.skill_name}\n\n{prompt}"


def _evaluation_workspace_ignore(source_root: Path):
    def ignore(current: str, names: list[str]) -> set[str]:
        current_path = Path(current)
        relative = current_path.relative_to(source_root)
        excluded = set(EVALUATION_WORKSPACE_EXCLUDED_NAMES.intersection(names))
        if relative == Path("."):
            excluded.update(EVALUATION_WORKSPACE_EXCLUDED_TOP_LEVEL.intersection(names))
        return excluded

    return ignore


def _stage_evaluation_workspace(source_root: Path, working_directory: Path) -> dict[str, Any]:
    shutil.copytree(
        source_root,
        working_directory,
        symlinks=True,
        ignore=_evaluation_workspace_ignore(source_root),
    )
    leaked_answer_keys = sorted(
        path.relative_to(working_directory).as_posix()
        for path in working_directory.rglob("evals.json")
    )
    if leaked_answer_keys:
        raise EvalError(
            "sanitized evaluation workspace retained answer keys: "
            + ", ".join(leaked_answer_keys)
        )
    manifest_rows: list[str] = []
    regular_files = 0
    symlinks = 0
    for path in sorted(working_directory.rglob("*")):
        relative = path.relative_to(working_directory).as_posix()
        if path.is_symlink():
            symlinks += 1
            manifest_rows.append(f"L\t{relative}\t{os.readlink(path)}")
        elif path.is_file():
            regular_files += 1
            manifest_rows.append(f"F\t{relative}\t{_sha256_file(path)}")
    return {
        "enabled": True,
        "method": "temporary-copy-with-eval-and-generated-path-exclusions",
        "source_root": str(source_root),
        "working_directory": str(working_directory),
        "excluded_top_level": sorted(EVALUATION_WORKSPACE_EXCLUDED_TOP_LEVEL),
        "excluded_names": sorted(EVALUATION_WORKSPACE_EXCLUDED_NAMES),
        "answer_keys_excluded": True,
        "regular_file_count": regular_files,
        "symlink_count": symlinks,
        "content_manifest_sha256": _sha256_text("\n".join(manifest_rows)),
        "ephemeral": True,
    }


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
    harness: str,
    codex: str,
    claude: str,
    repo_root: Path,
    case_dir: Path,
    grade_schema: Path,
    sandbox: str,
    model: str | None,
    grader_model: str | None,
    plugin_root: Path | None,
    max_budget_usd: float | None,
) -> tuple[list[str], list[str]]:
    if harness == "codex":
        generation_base = [
            codex,
            "exec",
            "--ephemeral",
            "--json",
            "-s",
            sandbox,
            "-C",
            str(repo_root),
        ]
        if model:
            generation_base.extend(["-m", model])
        grading_base = list(generation_base)
        if grader_model and grader_model != model:
            if "-m" in grading_base:
                model_index = grading_base.index("-m") + 1
                grading_base[model_index] = grader_model
            else:
                grading_base.extend(["-m", grader_model])
        return (
            [*generation_base, "-o", str(case_dir / "answer.md")],
            [
                *grading_base,
                "--output-schema",
                str(grade_schema),
                "-o",
                str(case_dir / "grade.json"),
            ],
        )

    if plugin_root is None:
        raise EvalError("Claude execution requires a local plugin root")
    common = [
        claude,
        "-p",
        "--no-session-persistence",
        "--no-chrome",
        "--strict-mcp-config",
        "--permission-mode",
        "dontAsk",
        "--setting-sources",
        "",
    ]
    generation = [
        *common,
        "--output-format",
        "stream-json",
        "--verbose",
        "--plugin-dir",
        str(plugin_root),
        "--tools",
        CLAUDE_TOOLS,
        "--allowedTools",
        CLAUDE_ALLOWED_TOOLS,
        "--disallowedTools",
        "Edit,Write,NotebookEdit",
    ]
    grading = [
        *common,
        "--output-format",
        "json",
        "--disable-slash-commands",
        "--tools",
        "",
        "--json-schema",
        json.dumps(json.loads(grade_schema.read_text(encoding="utf-8")), separators=(",", ":")),
    ]
    if model:
        generation.extend(["--model", model])
    if grader_model:
        grading.extend(["--model", grader_model])
    if max_budget_usd is not None:
        budget = str(max_budget_usd)
        generation.extend(["--max-budget-usd", budget])
        grading.extend(["--max-budget-usd", budget])
    return generation, grading


def _collect_observed_models(value: Any, models: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"model", "model_id", "model_name"} and isinstance(item, str):
                if item.strip():
                    models.add(item.strip())
            elif key in {"modelUsage", "model_usage"} and isinstance(item, dict):
                models.update(str(name) for name in item if str(name).strip())
            _collect_observed_models(item, models)
    elif isinstance(value, list):
        for item in value:
            _collect_observed_models(item, models)


def _trace_summary(
    trace: str, expected_skill: str | None, plugin_name: str | None = None
) -> dict[str, Any]:
    types: Counter[str] = Counter()
    parsed = 0
    observed_models: set[str] = set()
    total_cost_usd: float | None = None
    cost_reported = False
    model_usage: dict[str, Any] = {}
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        parsed += 1
        if isinstance(event, dict) and isinstance(event.get("type"), str):
            types[event["type"]] += 1
        if isinstance(event, dict):
            cost = event.get("total_cost_usd")
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                total_cost_usd = (total_cost_usd or 0.0) + float(cost)
                cost_reported = True
            usage = event.get("modelUsage") or event.get("model_usage")
            if isinstance(usage, dict):
                model_usage.update(usage)
        _collect_observed_models(event, observed_models)
    skill_marker = f".agents/skills/{expected_skill}/SKILL.md" if expected_skill else None
    plugin_marker = (
        f"{plugin_name}:{expected_skill}" if plugin_name and expected_skill else None
    )
    return {
        "bytes": len(trace.encode("utf-8")),
        "json_events": parsed,
        "event_types": dict(sorted(types.items())),
        "command_execution_markers": trace.count("command_execution"),
        "expected_skill_path_observed": bool(skill_marker and skill_marker in trace),
        "expected_plugin_skill_observed": bool(plugin_marker and plugin_marker in trace),
        "observed_models": sorted(observed_models),
        "model_usage": model_usage,
        "total_cost_usd": total_cost_usd,
        "cost_reported": cost_reported,
    }


def _phase_observations(
    generation: dict[str, Any], grading: dict[str, Any] | None = None
) -> dict[str, Any]:
    observations = {
        "generation_models_observed": generation["observed_models"],
        "generation_cost_usd": generation["total_cost_usd"],
        "generation_cost_reported": generation["cost_reported"],
    }
    if grading is not None:
        observations.update(
            {
                "grading_models_observed": grading["observed_models"],
                "grading_cost_usd": grading["total_cost_usd"],
                "grading_cost_reported": grading["cost_reported"],
            }
        )
    return observations


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
    command: Sequence[str],
    *,
    prompt: str,
    timeout: int,
    cwd: Path | None = None,
    environment_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = None
    if environment_overrides:
        environment = os.environ.copy()
        environment.update(environment_overrides)
    try:
        return subprocess.run(
            list(command),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            env=environment,
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
        if not isinstance(event, dict):
            continue
        claude_error = event.get("type") == "result" and event.get("is_error") is True
        if event.get("type") not in STRUCTURED_ERROR_TYPES and not claude_error:
            continue
        candidates: list[Any] = [event.get("message"), event.get("result")]
        error = event.get("error")
        if isinstance(error, dict):
            candidates.extend([error.get("message"), error.get("detail")])
        else:
            candidates.append(error)
        data = event.get("data")
        if isinstance(data, dict):
            candidates.extend([data.get("message"), data.get("error")])
        errors = event.get("errors")
        if isinstance(errors, list):
            candidates.extend(errors)
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                messages.append((f"trace:{event['type']}", candidate))
    return messages


def _structured_api_failure(trace: str) -> dict[str, str] | None:
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("is_error") is not True:
            continue
        raw_status = event.get("api_error_status")
        try:
            status = int(raw_status)
        except (TypeError, ValueError):
            continue
        category = CLAUDE_API_STATUS_CATEGORIES.get(status)
        if category:
            return {
                "reason_category": category,
                "matched_marker": f"api_error_status={status}",
                "evidence_source": "trace:result.api_error_status",
            }
    return None


def _structured_harness_skip(trace: str) -> dict[str, str] | None:
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        subtype = event.get("subtype")
        if event_type == "system" and subtype == "model_refusal_no_fallback":
            category = event.get("api_refusal_category")
            marker = f"api_refusal_category={category}" if category else subtype
            return {
                "reason_category": "model-policy-refusal",
                "matched_marker": marker,
                "evidence_source": "trace:system.model_refusal_no_fallback",
            }
        if event_type == "result" and subtype == "error_max_budget_usd":
            return {
                "reason_category": "budget-exhausted",
                "matched_marker": subtype,
                "evidence_source": "trace:result.subtype",
            }
        if event_type == "result" and event.get("stop_reason") == "refusal":
            return {
                "reason_category": "model-policy-refusal",
                "matched_marker": "stop_reason=refusal",
                "evidence_source": "trace:result.stop_reason",
            }
    return None


def _claude_result(trace: str, *, case_id: str, phase: str) -> dict[str, Any]:
    result: dict[str, Any] | None = None
    for line in trace.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            result = event
    if result is None:
        try:
            candidate = json.loads(trace)
        except json.JSONDecodeError:
            candidate = None
        if isinstance(candidate, dict):
            result = candidate
    if result is None:
        raise EvalError(f"{case_id}: Claude {phase} returned no result event")
    if result.get("is_error") is True:
        detail = result.get("result") or result.get("subtype") or "unknown error"
        raise EvalError(f"{case_id}: Claude {phase} result was an error: {detail}")
    return result


def _write_claude_answer(case: EvalCase, case_dir: Path, trace: str) -> Path:
    result = _claude_result(trace, case_id=case.id, phase="generation")
    answer = result.get("result")
    if not isinstance(answer, str) or not answer.strip():
        raise EvalError(f"{case.id}: Claude generation returned an empty result")
    path = case_dir / "answer.md"
    path.write_text(answer, encoding="utf-8")
    return path


def _write_claude_grade(case: EvalCase, case_dir: Path, trace: str) -> Path:
    result = _claude_result(trace, case_id=case.id, phase="grading")
    grade = result.get("structured_output")
    if not isinstance(grade, dict):
        raise EvalError(f"{case.id}: Claude grader returned no structured_output object")
    path = case_dir / "grade.json"
    _write_json(path, grade)
    return path


def _classify_infrastructure_failure(
    completed: subprocess.CompletedProcess[str],
) -> dict[str, str] | None:
    if completed.returncode == 0:
        return None
    api_failure = _structured_api_failure(completed.stdout)
    if api_failure:
        return api_failure
    harness_skip = _structured_harness_skip(completed.stdout)
    if harness_skip:
        return harness_skip
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
    observations: dict[str, Any],
) -> None:
    _write_json(
        case_dir / "status.json",
        {
            "status": "infrastructure-skip",
            "phase": phase,
            "exit_code": completed.returncode,
            **classification,
            **observations,
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
    top = _run_capture(["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"])
    if top.returncode != 0:
        return {
            "worktree": str(repo_root),
            "evaluation_working_directory": str(repo_root),
            "available": False,
            "revision": None,
            "branch": None,
            "dirty": None,
            "status_entries": [],
            "status_sha256": None,
            "tracked_diff_sha256": None,
            "reason": (top.stderr or "not a Git worktree").strip().splitlines()[0],
        }
    worktree = Path(top.stdout.strip()).resolve()
    base = ["git", "-C", str(worktree)]
    git_dir_result = _run_capture([*base, "rev-parse", "--absolute-git-dir"])
    revision = _run_capture([*base, "rev-parse", "HEAD"])
    branch = _run_capture([*base, "branch", "--show-current"])
    status = _run_capture([*base, "status", "--short", "--untracked-files=all"])
    diff = _run_capture([*base, "diff", "--binary", "HEAD"])
    status_text = status.stdout if status.returncode == 0 else ""
    entries = [line for line in status_text.splitlines() if line]
    return {
        "worktree": str(worktree),
        "evaluation_working_directory": str(repo_root),
        "git_dir": (
            str(Path(git_dir_result.stdout.strip()).resolve())
            if git_dir_result.returncode == 0
            else None
        ),
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


def _claude_plugin(skill_root: Path) -> tuple[Path, str, Path]:
    plugin_root = skill_root.parent.resolve()
    manifest = plugin_root / ".claude-plugin" / "plugin.json"
    if not manifest.is_file():
        raise EvalError(
            f"Claude local-plugin manifest not found: {manifest}; "
            "the skill root must be <plugin>/skills"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot parse Claude plugin manifest {manifest}: {exc}") from exc
    name = payload.get("name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise EvalError(f"Claude plugin manifest has no non-empty name: {manifest}")
    if name != CLAUDE_PLUGIN_NAME:
        raise EvalError(
            f"Claude plugin name must remain {CLAUDE_PLUGIN_NAME!r} for stable invocations; got {name!r}"
        )
    return plugin_root, name, manifest


def _claude_provenance(
    claude: str,
    *,
    live: bool,
    model: str | None,
    grader_model: str | None,
    plugin_root: Path,
    plugin_manifest: Path,
    repo_root: Path,
    working_directory: Path,
    max_budget_usd: float | None,
    external_skill_upload_allowed: bool,
) -> dict[str, Any]:
    resolved = shutil.which(claude)
    if resolved is None and Path(claude).is_file():
        resolved = str(Path(claude).resolve())
    version = "dry-run-not-probed"
    if live:
        completed = _run_process(
            [claude, "--version"], prompt="", timeout=30, cwd=repo_root
        )
        if completed.returncode != 0:
            raise EvalError(
                f"cannot record Claude version: {completed.stderr.strip() or completed.returncode}"
            )
        version = (completed.stdout or completed.stderr).strip().splitlines()[0]
        if not version:
            raise EvalError("cannot record Claude version: command returned no version")
    config = Path.home() / ".claude" / "settings.json"
    return {
        "name": "claude",
        "requested_executable": claude,
        "resolved_executable": resolved,
        "executable_sha256": (
            _sha256_file(Path(resolved)) if resolved and Path(resolved).is_file() else None
        ),
        "version": version,
        "model": model,
        "grader_model": grader_model,
        "model_source": "--model" if model else None,
        "source_working_directory": str(repo_root),
        "working_directory": str(working_directory),
        "no_session_persistence": True,
        "permission_mode": "dontAsk",
        "setting_sources": [],
        "environment_overrides": CLAUDE_ENVIRONMENT_OVERRIDES,
        "tools": CLAUDE_TOOLS,
        "allowed_tools": CLAUDE_ALLOWED_TOOLS,
        "disallowed_tools": "Edit,Write,NotebookEdit",
        "plugin_root": str(plugin_root),
        "plugin_manifest": str(plugin_manifest),
        "plugin_manifest_sha256": _sha256_file(plugin_manifest),
        "max_budget_usd_per_process": max_budget_usd,
        "external_skill_upload_allowed": external_skill_upload_allowed,
        "user_config_present": config.is_file(),
        "user_config_loaded": False,
    }


def build_provenance(
    *,
    selected: Sequence[EvalCase],
    skill_root: Path,
    repo_root: Path,
    working_directory: Path,
    harness: str,
    codex: str,
    claude: str,
    model: str | None,
    grader_model: str | None,
    sandbox: str,
    grade_schema: Path,
    plugin_root: Path | None,
    plugin_manifest: Path | None,
    max_budget_usd: float | None,
    external_skill_upload_allowed: bool,
    evaluation_workspace: dict[str, Any],
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
    if harness == "codex":
        harness_provenance = {
            "name": "codex",
            **_codex_provenance(codex, live=live, model=model, sandbox=sandbox),
            "grader_model": grader_model,
        }
    else:
        if plugin_root is None or plugin_manifest is None:
            raise EvalError("Claude provenance requires a validated local plugin")
        harness_provenance = _claude_provenance(
            claude,
            live=live,
            model=model,
            grader_model=grader_model,
            plugin_root=plugin_root,
            plugin_manifest=plugin_manifest,
            repo_root=repo_root,
            working_directory=working_directory,
            max_budget_usd=max_budget_usd,
            external_skill_upload_allowed=external_skill_upload_allowed,
        )
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "path": str(runner),
            "sha256": _sha256_file(runner),
            "python": sys.version,
            "platform": sys.platform,
            "argv": list(argv),
        },
        "harness": harness_provenance,
        "repository": _repository_provenance(repo_root),
        "evaluation_workspace": evaluation_workspace,
        "grade_schema": {"path": str(grade_schema), "sha256": _sha256_file(grade_schema)},
        "eval_sources": [
            {"path": str(source), "sha256": _sha256_file(source)} for source in sources
        ],
        "cases": case_rows,
        "instruction_files": instruction_files,
        "instruction_bundle_sha256": instruction_bundle,
    }
    payload[harness] = harness_provenance
    return payload


def prepare_case(
    case: EvalCase,
    *,
    results_dir: Path,
    repo_root: Path,
    working_directory: Path,
    harness: str,
    codex: str,
    claude: str,
    grade_schema: Path,
    sandbox: str,
    model: str | None,
    grader_model: str | None,
    plugin_root: Path | None,
    plugin_name: str | None,
    max_budget_usd: float | None,
    provenance: dict[str, Any],
    provenance_sha256: str,
) -> PreparedCase:
    case_dir = results_dir / case.id
    case_dir.mkdir(exist_ok=False)
    captured = {"skill_name": case.skill_name, "source": str(case.source), **case.payload}
    _write_json(case_dir / "case.json", captured)
    generation, grading = build_commands(
        harness=harness,
        codex=codex,
        claude=claude,
        repo_root=working_directory,
        case_dir=case_dir,
        grade_schema=grade_schema,
        sandbox=sandbox,
        model=model,
        grader_model=grader_model,
        plugin_root=plugin_root,
        max_budget_usd=max_budget_usd,
    )
    _write_json(
        case_dir / "run-plan.json",
        {
            "case_id": case.id,
            "harness": harness,
            "repo_root": str(repo_root),
            "working_directory": str(working_directory),
            "case_prompt_sha256": _sha256_text(case.payload["prompt"]),
            "prompt_sha256": _sha256_text(
                _generation_prompt(case, harness=harness, plugin_name=plugin_name)
            ),
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
        harness=harness,
        working_directory=working_directory,
        plugin_name=plugin_name,
        generation=generation,
        grading=grading,
    )


def run_prepared_case(prepared: PreparedCase, *, timeout: int) -> str:
    case = prepared.case
    case_dir = prepared.case_dir
    started = datetime.now(timezone.utc).isoformat()
    generation_prompt = _generation_prompt(
        case,
        harness=prepared.harness,
        plugin_name=prepared.plugin_name,
    )
    generated = _run_process(
        prepared.generation,
        prompt=generation_prompt,
        timeout=timeout,
        cwd=prepared.working_directory,
        environment_overrides=(
            CLAUDE_ENVIRONMENT_OVERRIDES if prepared.harness == "claude" else None
        ),
    )
    (case_dir / "generation.trace.jsonl").write_text(generated.stdout, encoding="utf-8")
    (case_dir / "generation.stderr.log").write_text(generated.stderr, encoding="utf-8")
    trace_summary = _trace_summary(
        generated.stdout,
        case.payload["expected_skill"],
        plugin_name=prepared.plugin_name,
    )
    _write_json(case_dir / "trace-summary.json", trace_summary)
    generation_observations = _phase_observations(trace_summary)
    if generated.returncode != 0:
        classification = _classify_infrastructure_failure(generated)
        if classification:
            _write_infrastructure_skip(
                case_dir,
                phase="generation",
                completed=generated,
                classification=classification,
                observations=generation_observations,
            )
            return "skip"
        _write_json(
            case_dir / "status.json",
            {
                "status": "generation-error",
                "exit_code": generated.returncode,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **generation_observations,
            },
        )
        raise EvalError(
            f"{case.id}: {prepared.harness} generation exited {generated.returncode}"
        )
    try:
        answer_path = (
            _write_claude_answer(case, case_dir, generated.stdout)
            if prepared.harness == "claude"
            else case_dir / "answer.md"
        )
        if not answer_path.is_file():
            raise EvalError(f"{case.id}: {prepared.harness} did not write answer.md")
    except EvalError as exc:
        _write_json(
            case_dir / "status.json",
            {
                "status": "generation-output-error",
                "error": str(exc),
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **generation_observations,
            },
        )
        raise
    answer = answer_path.read_text(encoding="utf-8")
    graded = _run_process(
        prepared.grading,
        prompt=_grader_prompt(case, answer, trace_summary),
        timeout=timeout,
        cwd=prepared.working_directory,
        environment_overrides=(
            CLAUDE_ENVIRONMENT_OVERRIDES if prepared.harness == "claude" else None
        ),
    )
    (case_dir / "grading.trace.jsonl").write_text(graded.stdout, encoding="utf-8")
    (case_dir / "grading.stderr.log").write_text(graded.stderr, encoding="utf-8")
    grading_trace_summary = _trace_summary(graded.stdout, None)
    _write_json(case_dir / "grading-trace-summary.json", grading_trace_summary)
    all_observations = _phase_observations(trace_summary, grading_trace_summary)
    if graded.returncode != 0:
        classification = _classify_infrastructure_failure(graded)
        if classification:
            _write_infrastructure_skip(
                case_dir,
                phase="grading",
                completed=graded,
                classification=classification,
                observations=all_observations,
            )
            return "skip"
        _write_json(
            case_dir / "status.json",
            {
                "status": "grading-error",
                "exit_code": graded.returncode,
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **all_observations,
            },
        )
        raise EvalError(f"{case.id}: {prepared.harness} grader exited {graded.returncode}")
    try:
        grade_path = (
            _write_claude_grade(case, case_dir, graded.stdout)
            if prepared.harness == "claude"
            else case_dir / "grade.json"
        )
        grade = _validate_grade(case, grade_path)
    except EvalError as exc:
        _write_json(
            case_dir / "status.json",
            {
                "status": "grading-output-error",
                "error": str(exc),
                "started_at": started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                **all_observations,
            },
        )
        raise
    _write_json(
        case_dir / "status.json",
        {
            "status": {"pass": "passed", "fail": "failed"}[grade["outcome"]],
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "answer_sha256": _sha256_file(answer_path),
            "grade_sha256": _sha256_file(grade_path),
            **all_observations,
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


def _run_observations(prepared: Sequence[PreparedCase]) -> dict[str, Any]:
    generation_models: set[str] = set()
    grading_models: set[str] = set()
    total_cost_usd: float | None = None
    cost_reported_processes = 0
    observed_cases = 0
    for item in prepared:
        status_path = item.case_dir / "status.json"
        if not status_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status, dict):
            continue
        observed_cases += 1
        generation_models.update(status.get("generation_models_observed") or [])
        grading_models.update(status.get("grading_models_observed") or [])
        for phase in ("generation", "grading"):
            if status.get(f"{phase}_cost_reported") is not True:
                continue
            value = status.get(f"{phase}_cost_usd")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                total_cost_usd = (total_cost_usd or 0.0) + float(value)
                cost_reported_processes += 1
    return {
        "observed_case_count": observed_cases,
        "generation_models": sorted(generation_models),
        "grading_models": sorted(grading_models),
        "total_cost_usd": total_cost_usd,
        "cost_reported_processes": cost_reported_processes,
    }


def _parser() -> argparse.ArgumentParser:
    default_skill_root = Path(__file__).resolve().parents[2]
    default_schema = Path(__file__).resolve().with_name("codex_grade.schema.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-root", type=Path, default=default_skill_root)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--case", action="append", dest="case_ids", help="case ID; repeat to select several")
    parser.add_argument("--all", action="store_true", help="allow --run to execute every discovered case")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument(
        "--harness", choices=("codex", "claude"), default="codex", help="CLI adapter"
    )
    parser.add_argument("--codex", default="codex", help="Codex executable")
    parser.add_argument("--claude", default="claude", help="Claude Code executable")
    parser.add_argument(
        "--model",
        help="optional generation-model override; omit to let the isolated CLI process resolve its default",
    )
    parser.add_argument(
        "--grader-model",
        help="optional independent-grader model override; defaults to --model or the isolated CLI-resolved default",
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        help="optional Claude cost ceiling per generation or grading process",
    )
    parser.add_argument(
        "--allow-external-skill-upload",
        action="store_true",
        help="acknowledge that live Claude evals transmit recipe skill text and prompts to Anthropic",
    )
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
    evaluation_workspace_handle: tempfile.TemporaryDirectory[str] | None = None
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
        repo_root = args.repo_root.resolve()
        skill_root = args.skill_root.resolve()
        working_directory = repo_root
        evaluation_workspace: dict[str, Any] = {
            "enabled": False,
            "source_root": str(repo_root),
            "working_directory": str(repo_root),
            "answer_keys_excluded": False,
            "reason": "dry-run or non-Claude harness",
        }
        grader_model = args.grader_model or args.model
        plugin_root: Path | None = None
        plugin_name: str | None = None
        plugin_manifest: Path | None = None
        if args.harness == "claude":
            plugin_root, plugin_name, plugin_manifest = _claude_plugin(skill_root)
            if args.run and not args.allow_external_skill_upload:
                raise EvalError(
                    "live Claude evals send recipe-local skill text and eval prompts to Anthropic; "
                    "rerun with --allow-external-skill-upload after confirming that transfer is allowed"
                )
        if args.max_budget_usd is not None and args.max_budget_usd <= 0:
            raise EvalError("--max-budget-usd must be positive")
        grade_schema = args.grade_schema.resolve()
        if not grade_schema.is_file():
            raise EvalError(f"grade schema not found: {grade_schema}")
        results_dir = args.results_dir.resolve()
        _reserve_results_dir(results_dir, selected)
        try:
            execution_plugin_root = plugin_root
            if args.run and args.harness == "claude":
                if plugin_root is None:
                    raise EvalError("Claude execution requires a validated local plugin")
                try:
                    plugin_relative = plugin_root.relative_to(repo_root)
                except ValueError as exc:
                    raise EvalError("Claude plugin root must be inside --repo-root") from exc
                evaluation_workspace_handle = tempfile.TemporaryDirectory(
                    prefix="bionemo-skill-eval-"
                )
                working_directory = (
                    Path(evaluation_workspace_handle.name) / repo_root.name
                )
                evaluation_workspace = _stage_evaluation_workspace(
                    repo_root, working_directory
                )
                execution_plugin_root = working_directory / plugin_relative
                staged_manifest = execution_plugin_root / ".claude-plugin" / "plugin.json"
                if not staged_manifest.is_file():
                    raise EvalError(
                        f"sanitized Claude plugin manifest not found: {staged_manifest}"
                    )
            provenance = build_provenance(
                selected=selected,
                skill_root=skill_root,
                repo_root=repo_root,
                working_directory=working_directory,
                harness=args.harness,
                codex=args.codex,
                claude=args.claude,
                model=args.model,
                grader_model=grader_model,
                sandbox=args.sandbox,
                grade_schema=grade_schema,
                plugin_root=plugin_root,
                plugin_manifest=plugin_manifest,
                max_budget_usd=args.max_budget_usd,
                external_skill_upload_allowed=args.allow_external_skill_upload,
                evaluation_workspace=evaluation_workspace,
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
                    working_directory=working_directory,
                    harness=args.harness,
                    codex=args.codex,
                    claude=args.claude,
                    grade_schema=grade_schema,
                    sandbox=args.sandbox,
                    model=args.model,
                    grader_model=grader_model,
                    plugin_root=execution_plugin_root,
                    plugin_name=plugin_name,
                    max_budget_usd=args.max_budget_usd,
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
                {
                    "status": "harness-error",
                    "error": str(exc),
                    "counts": dict(counts),
                    "observations": _run_observations(prepared),
                },
            )
            raise
        _write_json(
            results_dir / "run-status.json",
            {
                "status": "complete-with-failures" if counts["fail"] else "complete",
                "counts": dict(counts),
                "observations": _run_observations(prepared),
            },
        )
        return 1 if counts["fail"] else 0
    except EvalError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if evaluation_workspace_handle is not None:
            evaluation_workspace_handle.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
