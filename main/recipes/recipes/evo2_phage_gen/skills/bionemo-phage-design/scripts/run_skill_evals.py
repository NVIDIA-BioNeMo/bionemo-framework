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

"""Validate portable Agent Skills eval JSON and run selected cases.

The eval files retain the BioNeMo Agent Toolkit envelope. Harness-specific
execution, provenance capture, and grading live here so the cases remain
portable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
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
- Ground claims in relevant checked-in primary sources. Web use may check current status or newer versions, but must not replace the local source.
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
    "VALIDATION.md",
    "evals",
    "evals.json",
}
EVALUATION_WORKSPACE_GENERATED_PATTERNS = (
    "*.egg-info",
    "*.dist-info",
    "**/results/**",
    "**/tmp/**",
    "**/tmp_*",
    "**/tmp_*/**",
    "*/scripts/tests/*",
)


class EvalError(RuntimeError):
    """Raised for invalid suites or failed harness operations."""


@dataclass(frozen=True)
class EvalCase:
    """Represent one validated evaluation case."""

    skill_name: str
    source: Path
    payload: dict[str, Any]

    @property
    def id(self) -> str:
        """Return the stable evaluation case identifier."""
        return str(self.payload["id"])


@dataclass(frozen=True)
class PreparedCase:
    """Hold one evaluation case and its executable run plan."""

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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _generation_prompt(case: EvalCase, *, harness: str, plugin_name: str | None) -> str:
    prompt = f"{case.payload['prompt']}\n\n{EVALUATION_RESPONSE_CONTRACT}"
    if harness != "claude":
        return prompt
    if not plugin_name:
        raise EvalError(f"{case.id}: Claude case has no plugin namespace")
    return f"/{plugin_name}:{case.skill_name}\n\n{prompt}"


def _evaluation_workspace_exclusion(relative: Path) -> str | None:
    if relative.parts and relative.parts[0] in EVALUATION_WORKSPACE_EXCLUDED_TOP_LEVEL:
        return "excluded-top-level"
    if any(part in EVALUATION_WORKSPACE_EXCLUDED_NAMES for part in relative.parts):
        return "excluded-name"
    if "results" in relative.parts:
        return "excluded-results"
    if any(part == "tmp" or part.startswith("tmp_") for part in relative.parts):
        return "excluded-generated-name"
    if any(part.endswith((".egg-info", ".dist-info")) for part in relative.parts):
        return "excluded-generated-metadata"
    if any(
        left == "scripts" and right == "tests" for left, right in zip(relative.parts, relative.parts[1:], strict=False)
    ):
        return "excluded-audit-tests"
    return None


def _git_tracked_evaluation_paths(source_root: Path) -> tuple[Path, list[Path]]:
    top = _run_capture(["git", "-C", str(source_root), "rev-parse", "--show-toplevel"])
    if top.returncode != 0:
        reason = (top.stderr or "not a Git worktree").strip().splitlines()[0]
        raise EvalError("live evaluation requires a Git-tracked source workspace: " + reason)
    worktree = Path(top.stdout.strip()).resolve()
    listed = _run_capture(["git", "-C", str(source_root), "ls-files", "-z", "--cached", "--", "."])
    if listed.returncode != 0:
        reason = (listed.stderr or "git ls-files failed").strip().splitlines()[0]
        raise EvalError(f"cannot enumerate Git-tracked evaluation files: {reason}")
    relative_paths: list[Path] = []
    for raw in listed.stdout.split("\0"):
        if not raw:
            continue
        relative = Path(raw)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise EvalError(f"unsafe Git-tracked evaluation path: {raw!r}")
        relative_paths.append(relative)
    return worktree, sorted(set(relative_paths), key=lambda path: path.as_posix())


def _stage_evaluation_workspace(
    source_root: Path,
    working_directory: Path,
    *,
    required_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    source_root = source_root.resolve()
    worktree, tracked_paths = _git_tracked_evaluation_paths(source_root)
    working_directory.mkdir(parents=True)
    excluded_tracked_paths: list[dict[str, str]] = []
    excluded_symlinks: list[dict[str, str]] = []
    deleted_tracked_paths: list[str] = []
    unsupported_tracked_paths: list[str] = []
    tracked_symlinks: list[tuple[Path, Path]] = []

    for relative in tracked_paths:
        exclusion = _evaluation_workspace_exclusion(relative)
        if exclusion:
            excluded_tracked_paths.append({"path": relative.as_posix(), "reason": exclusion})
            continue
        source = source_root / relative
        if source.is_symlink():
            tracked_symlinks.append((relative, source))
            continue
        if source.is_file():
            destination = working_directory / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination, follow_symlinks=False)
            continue
        if not os.path.lexists(source):
            deleted_tracked_paths.append(relative.as_posix())
        else:
            unsupported_tracked_paths.append(relative.as_posix())

    staged_root = working_directory.resolve()
    for relative, source in tracked_symlinks:
        target = os.readlink(source)
        if Path(target).is_absolute():
            excluded_symlinks.append(
                {
                    "path": relative.as_posix(),
                    "target": target,
                    "reason": "absolute-target",
                }
            )
            continue
        resolved_source_target = (source.parent / target).resolve(strict=False)
        try:
            target_relative = resolved_source_target.relative_to(source_root)
        except ValueError:
            excluded_symlinks.append(
                {
                    "path": relative.as_posix(),
                    "target": target,
                    "reason": "target-outside-source-root",
                }
            )
            continue
        staged_target = working_directory / target_relative
        if not staged_target.exists():
            excluded_symlinks.append(
                {
                    "path": relative.as_posix(),
                    "target": target,
                    "reason": "target-not-staged",
                }
            )
            continue
        destination = working_directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(target, target_is_directory=resolved_source_target.is_dir())
        try:
            destination.resolve(strict=True).relative_to(staged_root)
        except (FileNotFoundError, ValueError) as exc:
            destination.unlink(missing_ok=True)
            raise EvalError(f"staged symlink escapes or is unresolved: {relative.as_posix()}") from exc

    normalized_required: list[str] = []
    for required in required_paths:
        relative = Path(required)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise EvalError(f"unsafe required evaluation path: {required}")
        normalized_required.append(relative.as_posix())
        if not (working_directory / relative).is_file():
            raise EvalError(
                "required evaluation runtime path is not present in the Git-tracked "
                f"workspace: {relative.as_posix()}; add it to the Git index first"
            )
    normalized_required = sorted(set(normalized_required))

    leaked_excluded_paths = sorted(
        path.relative_to(working_directory).as_posix()
        for path in working_directory.rglob("*")
        if _evaluation_workspace_exclusion(path.relative_to(working_directory)) is not None
    )
    if leaked_excluded_paths:
        raise EvalError("sanitized evaluation workspace retained excluded paths: " + ", ".join(leaked_excluded_paths))
    return {
        "enabled": True,
        "method": "git-tracked-working-tree",
        "source_root": str(source_root),
        "source_git_worktree": str(worktree),
        "working_directory": str(working_directory),
        "excluded_top_level": sorted(EVALUATION_WORKSPACE_EXCLUDED_TOP_LEVEL),
        "excluded_names": sorted(EVALUATION_WORKSPACE_EXCLUDED_NAMES),
        "answer_keys_excluded": True,
        "untracked_paths_excluded": True,
        "generated_path_patterns_excluded": list(EVALUATION_WORKSPACE_GENERATED_PATTERNS),
        "tracked_path_count": len(tracked_paths),
        "excluded_tracked_paths": excluded_tracked_paths,
        "excluded_symlinks": excluded_symlinks,
        "deleted_tracked_paths": deleted_tracked_paths,
        "unsupported_tracked_paths": unsupported_tracked_paths,
        "required_paths": normalized_required,
        "ephemeral": True,
    }


def _validate_case(case: Any, *, source: Path, index: int, skill_name: str) -> list[str]:
    where = f"{source}: evals[{index}]"
    if not isinstance(case, dict):
        return [f"{where} must be an object"]
    errors = [f"{where} is missing required field {field!r}" for field in REQUIRED_CASE_FIELDS if field not in case]
    if errors:
        return errors
    errors.extend(
        f"{where}.{field} must be a non-empty string"
        for field in ("id", "prompt", "expected_output")
        if not isinstance(case[field], str) or not case[field].strip()
    )
    if isinstance(case["id"], str) and not SAFE_ID.fullmatch(case["id"]):
        errors.append(f"{where}.id must match {SAFE_ID.pattern}")
    assertions = case["assertions"]
    if not isinstance(assertions, list) or not assertions:
        errors.append(f"{where}.assertions must be a non-empty string array")
    elif any(not isinstance(item, str) or not item.strip() for item in assertions):
        errors.append(f"{where}.assertions contains an empty or non-string assertion")
    expected_skill = case["expected_skill"]
    if expected_skill is not None and (not isinstance(expected_skill, str) or not expected_skill.strip()):
        errors.append(f"{where}.expected_skill must be a non-empty string or null")
    elif isinstance(expected_skill, str) and expected_skill != skill_name:
        errors.append(f"{where}.expected_skill {expected_skill!r} does not match owning skill {skill_name!r}")
    expected_script = case["expected_script"]
    if expected_script is not None and (not isinstance(expected_script, str) or not expected_script.strip()):
        errors.append(f"{where}.expected_script must be a non-empty string or null")
    return errors


def load_cases(skill_root: Path) -> tuple[list[EvalCase], list[Path]]:
    """Load and validate evaluation cases beneath a skill root."""
    skill_root = skill_root.resolve()
    files = sorted(path for path in skill_root.glob("*/evals/*.json") if not path.name.endswith(".schema.json"))
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
                errors.append(f"{source}: duplicate eval id {case_id!r}; first seen in {seen[case_id]}")
            else:
                seen[case_id] = source
            cases.append(EvalCase(skill_name=skill_name, source=source, payload=raw_case))
    if errors:
        raise EvalError("\n".join(errors))
    return cases, files


def select_cases(cases: Sequence[EvalCase], selected_ids: Sequence[str] | None) -> list[EvalCase]:
    """Return cases matching the requested identifiers."""
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
    working_directory: Path,
    case_dir: Path,
    grade_schema: Path,
    sandbox: str,
    model: str | None,
    grader_model: str | None,
    plugin_root: Path | None,
    max_budget_usd: float | None,
) -> tuple[list[str], list[str]]:
    """Build generation and grading commands for one evaluation case."""
    if harness == "codex":
        generation_base = [
            codex,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "-s",
            sandbox,
            "-C",
            str(working_directory),
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
    claude_schema = json.loads(grade_schema.read_text(encoding="utf-8"))
    if not isinstance(claude_schema, dict):
        raise EvalError("grade schema must be a JSON object")
    claude_schema.pop("$schema", None)
    grading = [
        *common,
        "--output-format",
        "json",
        "--disable-slash-commands",
        "--tools",
        "",
        "--json-schema",
        json.dumps(claude_schema, separators=(",", ":")),
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


_TOOL_CALL_TYPES = {"function_call", "tool_call", "tool_use"}


def _normalized_web_tool_type(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
    if compact.endswith("websearch"):
        return "web_search"
    if compact.endswith("webfetch"):
        return "web_fetch"
    if compact.endswith("webrun"):
        return "web_run"
    return None


def _trace_call_key(kind: str, value: dict[str, Any]) -> str:
    identifier = value.get("id") or value.get("tool_use_id") or value.get("call_id")
    if isinstance(identifier, str) and identifier:
        return f"{kind}:{identifier}"
    payload = {
        key: value.get(key) for key in ("name", "input", "query", "queries", "action") if value.get(key) is not None
    }
    return f"{kind}:{_sha256_text(json.dumps(payload, sort_keys=True, default=str))}"


def _collect_trace_web_evidence(
    value: object,
    *,
    web_calls: set[str],
    web_types: set[str],
) -> None:
    if isinstance(value, dict):
        event_type = value.get("type")
        direct_web_type = _normalized_web_tool_type(event_type)
        if direct_web_type is not None:
            web_types.add(direct_web_type)
            web_calls.add(_trace_call_key(direct_web_type, value))

        if event_type in _TOOL_CALL_TYPES:
            web_tool_type = _normalized_web_tool_type(value.get("name"))
            if web_tool_type is not None:
                web_types.add(web_tool_type)
                web_calls.add(_trace_call_key(web_tool_type, value))

        for item in value.values():
            _collect_trace_web_evidence(item, web_calls=web_calls, web_types=web_types)
    elif isinstance(value, list):
        for item in value:
            _collect_trace_web_evidence(item, web_calls=web_calls, web_types=web_types)


def _trace_summary(trace: str, expected_skill: str | None, plugin_name: str | None = None) -> dict[str, Any]:
    types: Counter[str] = Counter()
    parsed = 0
    observed_models: set[str] = set()
    total_cost_usd: float | None = None
    cost_reported = False
    model_usage: dict[str, Any] = {}
    web_calls: set[str] = set()
    web_types: set[str] = set()
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
        _collect_trace_web_evidence(event, web_calls=web_calls, web_types=web_types)
    normalized_trace = trace.replace("\\", "/")
    skill_marker = f"skills/{expected_skill}/SKILL.md" if expected_skill else None
    plugin_marker = f"{plugin_name}:{expected_skill}" if plugin_name and expected_skill else None
    return {
        "bytes": len(trace.encode("utf-8")),
        "json_events": parsed,
        "event_types": dict(sorted(types.items())),
        "command_execution_markers": trace.count("command_execution"),
        "expected_skill_path_observed": bool(skill_marker and skill_marker in normalized_trace),
        "expected_plugin_skill_observed": bool(plugin_marker and plugin_marker in trace),
        "observed_models": sorted(observed_models),
        "web_tool_calls_observed": bool(web_calls),
        "web_tool_call_count": len(web_calls),
        "web_tool_call_types": sorted(web_types),
        "model_usage": model_usage,
        "total_cost_usd": total_cost_usd,
        "cost_reported": cost_reported,
    }


def _phase_observations(generation: dict[str, Any], grading: dict[str, Any] | None = None) -> dict[str, Any]:
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
        "EVAL CONTRACT:\n" + json.dumps(contract, indent=2, sort_keys=True)
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
        raise EvalError(f"command timed out after {timeout} seconds: {' '.join(command[:3])}") from exc


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
        messages.extend(
            (f"trace:{event['type']}", candidate)
            for candidate in candidates
            if isinstance(candidate, str) and candidate.strip()
        )
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
            "reason": (top.stderr or "not a Git worktree").strip().splitlines()[0],
        }
    worktree = Path(top.stdout.strip()).resolve()
    base = ["git", "-C", str(worktree)]
    revision = _run_capture([*base, "rev-parse", "HEAD"])
    branch = _run_capture([*base, "branch", "--show-current"])
    status = _run_capture([*base, "status", "--short", "--untracked-files=all"])
    status_text = status.stdout if status.returncode == 0 else ""
    entries = [line for line in status_text.splitlines() if line]
    return {
        "worktree": str(worktree),
        "evaluation_working_directory": str(repo_root),
        "available": revision.returncode == 0,
        "revision": revision.stdout.strip() or None,
        "branch": branch.stdout.strip() or None,
        "dirty": bool(entries) if status.returncode == 0 else None,
    }


def _instruction_files(skill_root: Path) -> list[dict[str, str]]:
    candidates: set[Path] = set(skill_root.glob("*/SKILL.md"))
    for skill_dir in skill_root.iterdir():
        if not skill_dir.is_dir():
            continue
        references = skill_dir / "references"
        if references.is_dir():
            candidates.update(path for path in references.rglob("*") if path.is_file())
    return [{"path": path.relative_to(skill_root).as_posix()} for path in sorted(candidates)]


def _codex_provenance(codex: str, *, live: bool, model: str | None, sandbox: str) -> dict[str, Any]:
    resolved = shutil.which(codex)
    if resolved is None and Path(codex).is_file():
        resolved = str(Path(codex).resolve())
    version = "dry-run-not-probed"
    if live:
        completed = _run_process([codex, "--version"], prompt="", timeout=30)
        if completed.returncode != 0:
            raise EvalError(f"cannot record Codex version: {completed.stderr.strip() or completed.returncode}")
        version = (completed.stdout or completed.stderr).strip().splitlines()[0]
        if not version:
            raise EvalError("cannot record Codex version: command returned no version")
    return {
        "requested_executable": codex,
        "resolved_executable": resolved,
        "version": version,
        "model": model,
        "model_source": "--model" if model else None,
        "sandbox": sandbox,
        "ephemeral": True,
        "json_trace": True,
    }


def _claude_plugin(skill_root: Path) -> tuple[Path, str, Path]:
    plugin_root = skill_root.parent.resolve()
    manifest = plugin_root / ".claude-plugin" / "plugin.json"
    if not manifest.is_file():
        raise EvalError(f"Claude local-plugin manifest not found: {manifest}; the skill root must be <plugin>/skills")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot parse Claude plugin manifest {manifest}: {exc}") from exc
    name = payload.get("name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name.strip():
        raise EvalError(f"Claude plugin manifest has no non-empty name: {manifest}")
    if name != CLAUDE_PLUGIN_NAME:
        raise EvalError(f"Claude plugin name must remain {CLAUDE_PLUGIN_NAME!r} for stable invocations; got {name!r}")
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
    source_working_directory: Path,
    working_directory: Path,
    max_budget_usd: float | None,
    external_skill_upload_allowed: bool,
) -> dict[str, Any]:
    resolved = shutil.which(claude)
    if resolved is None and Path(claude).is_file():
        resolved = str(Path(claude).resolve())
    version = "dry-run-not-probed"
    if live:
        completed = _run_process([claude, "--version"], prompt="", timeout=30, cwd=repo_root)
        if completed.returncode != 0:
            raise EvalError(f"cannot record Claude version: {completed.stderr.strip() or completed.returncode}")
        version = (completed.stdout or completed.stderr).strip().splitlines()[0]
        if not version:
            raise EvalError("cannot record Claude version: command returned no version")
    config = Path.home() / ".claude" / "settings.json"
    return {
        "name": "claude",
        "requested_executable": claude,
        "resolved_executable": resolved,
        "version": version,
        "model": model,
        "grader_model": grader_model,
        "model_source": "--model" if model else None,
        "source_working_directory": str(source_working_directory),
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
    recipe_root: Path,
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
    """Build reproducibility metadata for an evaluation run."""
    instruction_files = _instruction_files(skill_root)
    sources = sorted({case.source for case in selected})
    case_rows = [{"id": case.id, "source": str(case.source)} for case in selected]
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
            source_working_directory=recipe_root,
            working_directory=working_directory,
            max_budget_usd=max_budget_usd,
            external_skill_upload_allowed=external_skill_upload_allowed,
        )
    payload = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runner": {
            "path": str(runner),
            "python": sys.version,
            "platform": sys.platform,
            "argv": list(argv),
        },
        "harness": harness_provenance,
        "repository": _repository_provenance(repo_root),
        "workspace": {
            "repository_root": str(repo_root),
            "recipe_root": str(recipe_root),
            "execution_working_directory": str(working_directory),
        },
        "evaluation_workspace": evaluation_workspace,
        "grade_schema": str(grade_schema),
        "eval_sources": [str(source) for source in sources],
        "cases": case_rows,
        "instruction_files": instruction_files,
    }
    payload[harness] = harness_provenance
    return payload


def prepare_case(
    case: EvalCase,
    *,
    results_dir: Path,
    repo_root: Path,
    recipe_root: Path,
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
) -> PreparedCase:
    """Materialize one case and its execution plan in the results directory."""
    case_dir = results_dir / case.id
    case_dir.mkdir(exist_ok=False)
    captured = {"skill_name": case.skill_name, "source": str(case.source), **case.payload}
    _write_json(case_dir / "case.json", captured)
    generation, grading = build_commands(
        harness=harness,
        codex=codex,
        claude=claude,
        working_directory=working_directory,
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
            "recipe_root": str(recipe_root),
            "working_directory": str(working_directory),
            "generation_command": generation,
            "grading_command": grading,
            "grade_schema": str(grade_schema),
            "provenance_path": "../run-provenance.json",
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
    """Execute and grade a prepared evaluation case."""
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
        environment_overrides=(CLAUDE_ENVIRONMENT_OVERRIDES if prepared.harness == "claude" else None),
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
        raise EvalError(f"{case.id}: {prepared.harness} generation exited {generated.returncode}")
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
        environment_overrides=(CLAUDE_ENVIRONMENT_OVERRIDES if prepared.harness == "claude" else None),
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


def _resolve_recipe_root(repo_root: Path, requested: Path) -> tuple[Path, Path]:
    repo_root = repo_root.resolve()
    if not repo_root.is_dir():
        raise EvalError(f"--repo-root is not a directory: {repo_root}")
    recipe_root = requested.resolve() if requested.is_absolute() else (repo_root / requested).resolve()
    try:
        relative = recipe_root.relative_to(repo_root)
    except ValueError as exc:
        raise EvalError(
            "--recipe-root must resolve inside --repo-root; point --repo-root at the selected checkout or worktree"
        ) from exc
    if not recipe_root.is_dir():
        raise EvalError(f"--recipe-root is not a directory: {recipe_root}")
    return recipe_root, relative


def _parser() -> argparse.ArgumentParser:
    default_skill_root = Path(__file__).resolve().parents[2]
    default_schema = Path(__file__).resolve().with_name("codex_grade.schema.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skill-root",
        type=Path,
        default=default_skill_root,
        help="installed skill directory; independent of the target checkout",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="target BioNeMo Recipes checkout or worktree; never inferred from --skill-root",
    )
    parser.add_argument(
        "--recipe-root",
        type=Path,
        default=Path("recipes/evo2_phage_gen"),
        help="selected recipe directory, absolute or relative to --repo-root",
    )
    parser.add_argument("--case", action="append", dest="case_ids", help="case ID; repeat to select several")
    parser.add_argument("--all", action="store_true", help="allow --run to execute every discovered case")
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--harness", choices=("codex", "claude"), default="codex", help="CLI adapter")
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
        help=(
            "acknowledge that live Claude evals may transmit installed skill text, "
            "prompts, and staged recipe files they read to Anthropic"
        ),
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
    """Run the evaluation command-line interface."""
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
            rows = [{"id": case.id, "skill_name": case.skill_name, "source": str(case.source)} for case in selected]
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
        recipe_root, recipe_relative = _resolve_recipe_root(repo_root, args.recipe_root)
        working_directory = recipe_root
        evaluation_workspace: dict[str, Any] = {
            "enabled": False,
            "source_root": str(repo_root),
            "source_recipe_root": str(recipe_root),
            "working_directory": str(recipe_root),
            "answer_keys_excluded": False,
            "reason": "dry-run",
        }
        grader_model = args.grader_model or args.model
        plugin_root: Path | None = None
        plugin_name: str | None = None
        plugin_manifest: Path | None = None
        if args.harness == "claude":
            plugin_root, plugin_name, plugin_manifest = _claude_plugin(skill_root)
            if args.run and not args.allow_external_skill_upload:
                raise EvalError(
                    "live Claude evals may send installed skill text, eval prompts, "
                    "and staged recipe files it reads to Anthropic; "
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
            if args.run:
                plugin_relative: Path | None = None
                if args.harness == "claude":
                    if plugin_root is None:
                        raise EvalError("Claude execution requires a validated local plugin")
                    try:
                        plugin_relative = plugin_root.relative_to(repo_root)
                    except ValueError:
                        pass
                evaluation_workspace_handle = tempfile.TemporaryDirectory(prefix="bionemo-skill-eval-")
                staged_repository_root = Path(evaluation_workspace_handle.name) / repo_root.name
                required_runtime_paths: set[Path] = set()
                if args.harness == "claude" and plugin_relative is not None:
                    required_runtime_paths = {
                        plugin_relative / ".claude-plugin" / "plugin.json",
                        *(plugin_relative / "skills" / case.skill_name / "SKILL.md" for case in selected),
                    }
                evaluation_workspace = _stage_evaluation_workspace(
                    repo_root,
                    staged_repository_root,
                    required_paths=sorted(
                        required_runtime_paths,
                        key=lambda path: path.as_posix(),
                    ),
                )
                evaluation_workspace["source_recipe_root"] = str(recipe_root)
                working_directory = staged_repository_root / recipe_relative
                evaluation_workspace["working_directory"] = str(working_directory)
                if not working_directory.is_dir():
                    raise EvalError(
                        "selected recipe is not present in the sanitized Git-tracked "
                        f"workspace: {recipe_relative.as_posix()}"
                    )
                if args.harness == "claude":
                    if plugin_relative is not None:
                        execution_plugin_root = staged_repository_root / plugin_relative
                    else:
                        assert plugin_root is not None
                        execution_plugin_root = staged_repository_root / ".external-skill-plugin"
                        external_required = {
                            Path(".claude-plugin/plugin.json"),
                            *(Path("skills") / case.skill_name / "SKILL.md" for case in selected),
                        }
                        try:
                            external_workspace = _stage_evaluation_workspace(
                                plugin_root,
                                execution_plugin_root,
                                required_paths=sorted(
                                    external_required,
                                    key=lambda path: path.as_posix(),
                                ),
                            )
                        except EvalError as exc:
                            raise EvalError(
                                "live Claude evaluation with a plugin outside --repo-root "
                                "requires that plugin root to be Git-tracked for sanitized "
                                f"staging: {exc}"
                            ) from exc
                        evaluation_workspace["method"] = "git-tracked-working-tree-plus-sanitized-plugin"
                        evaluation_workspace["external_plugin_workspace"] = external_workspace
                    staged_manifest = execution_plugin_root / ".claude-plugin" / "plugin.json"
                    if not staged_manifest.is_file():
                        raise EvalError(f"sanitized Claude plugin manifest not found: {staged_manifest}")
            provenance = build_provenance(
                selected=selected,
                skill_root=skill_root,
                repo_root=repo_root,
                recipe_root=recipe_root,
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
            prepared = [
                prepare_case(
                    case,
                    results_dir=results_dir,
                    repo_root=repo_root,
                    recipe_root=recipe_root,
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
