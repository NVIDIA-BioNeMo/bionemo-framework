from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Mapping, Sequence

import torch
import torch._dynamo as dynamo
import torch._logging as torch_logging


logger = logging.getLogger(__name__)


def add_torch_compile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use-torch-compile",
        "--use_torch_compile",
        action="store_true",
        dest="use_torch_compile",
        help="Wrap the model with torch.compile before running the benchmark/eval loop.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        "--torch_compile_mode",
        default="default",
        dest="torch_compile_mode",
        help="torch.compile mode to use when --use-torch-compile is enabled.",
    )
    parser.add_argument(
        "--diagnostics-output",
        "--diagnostics_output",
        type=Path,
        default=None,
        dest="diagnostics_output",
        help="Optional JSON output path for a torch._dynamo.explain pass on the first real batch.",
    )


def maybe_compile(model_or_fn: Any, *, enabled: bool, mode: str) -> Any:
    if not enabled:
        return model_or_fn
    return torch.compile(model_or_fn, mode=mode)


def maybe_capture_dynamo_diagnostics(
    fn: Callable[..., Any],
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    output_path: Path | None = None,
    label: str,
    rank: int = 0,
    world_size: int = 1,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if output_path is None:
        return None

    kwargs = dict(kwargs or {})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = _configure_rank_logging(output_path, rank)

    barrier()
    _reset_compile_state()
    try:
        explain = dynamo.explain(fn)(*args, **kwargs)
        payload = {
            "label": label,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "world_size": world_size,
            "graph_break_count": int(explain.graph_break_count),
            "graph_count": int(explain.graph_count),
            "op_count": int(explain.op_count),
            "break_reasons": [_serialize_break_reason(reason) for reason in explain.break_reasons],
            "ops_per_graph": [[str(op) for op in ops] for ops in explain.ops_per_graph],
            "compile_times": str(explain.compile_times),
            "log_path": str(log_path),
        }
    except Exception as exc:  # pragma: no cover - exercised by runtime harnesses
        logger.exception("torch._dynamo.explain failed for %s", label)
        payload = {
            "label": label,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "world_size": world_size,
            "graph_break_count": -1,
            "graph_count": 0,
            "op_count": 0,
            "break_reasons": [],
            "ops_per_graph": [],
            "compile_times": "",
            "log_path": str(log_path),
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    if extra_metadata:
        payload["metadata"] = dict(extra_metadata)

    if rank == 0:
        output_path.write_text(json.dumps(payload, indent=2))

    barrier()
    return payload


def barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _reset_compile_state() -> None:
    if hasattr(torch, "compiler") and hasattr(torch.compiler, "reset"):
        torch.compiler.reset()
    else:
        dynamo.reset()


def _configure_rank_logging(output_path: Path, rank: int) -> Path:
    log_path = _derive_rank_log_path(output_path, rank)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    existing_paths = {
        Path(handler.baseFilename)
        for handler in root_logger.handlers
        if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None)
    }
    if log_path not in existing_paths:
        file_handler = logging.FileHandler(log_path, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_logger.addHandler(file_handler)

    torch_logging.set_logs(graph_breaks=True, recompiles=True)
    return log_path


def _derive_rank_log_path(output_path: Path, rank: int) -> Path:
    if output_path.suffix:
        return output_path.with_suffix(f".rank{rank}.log")
    return output_path.parent / f"{output_path.name}.rank{rank}.log"


def _serialize_break_reason(reason: Any) -> dict[str, Any]:
    user_stack = getattr(reason, "user_stack", [])
    return {
        "reason": getattr(reason, "reason", str(reason)),
        "graph_break": getattr(reason, "graph_break", None),
        "user_stack": [_format_frame(frame) for frame in user_stack],
    }


def _format_frame(frame: FrameType | Any) -> str:
    filename = getattr(frame, "filename", None) or getattr(frame, "f_code", None)
    if filename is not None and not isinstance(filename, str):
        filename = getattr(filename, "co_filename", str(filename))
    lineno = getattr(frame, "lineno", None) or getattr(frame, "f_lineno", None)
    name = getattr(frame, "name", None) or getattr(getattr(frame, "f_code", None), "co_name", None)
    if filename is None and lineno is None and name is None:
        return str(frame)
    return f"{filename}:{lineno} in {name}"
