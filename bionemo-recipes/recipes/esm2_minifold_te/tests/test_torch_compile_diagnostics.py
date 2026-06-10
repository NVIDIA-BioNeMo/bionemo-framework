from pathlib import Path
import sys

import torch


RECIPE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(RECIPE_DIR))

from checkpoint import get_latest_checkpoint
from torch_compile_diagnostics import maybe_capture_dynamo_diagnostics


def test_get_latest_checkpoint_accepts_explicit_step_dir(tmp_path):
    step_dir = tmp_path / "step_15000"
    step_dir.mkdir()
    resolved_path, step = get_latest_checkpoint(step_dir)
    assert resolved_path == step_dir
    assert step == 15000


def test_maybe_capture_dynamo_diagnostics_writes_payload(tmp_path):
    output_path = tmp_path / "diagnostics.json"

    def fn(x):
        return x + 1

    payload = maybe_capture_dynamo_diagnostics(
        fn,
        args=(torch.randn(4),),
        output_path=output_path,
        label="unit_test",
        rank=0,
        world_size=1,
    )

    assert payload is not None
    assert payload["graph_break_count"] == 0
    assert payload["graph_count"] == 1
    assert output_path.exists()
    assert Path(payload["log_path"]).exists()


def test_maybe_capture_dynamo_diagnostics_serializes_break_reasons(tmp_path):
    output_path = tmp_path / "graph_breaks.json"

    def fn(x):
        scalar = x.sum().item()
        return x + scalar

    payload = maybe_capture_dynamo_diagnostics(
        fn,
        args=(torch.randn(4),),
        output_path=output_path,
        label="graph_break_unit_test",
        rank=0,
        world_size=1,
    )

    assert payload is not None
    assert payload["graph_break_count"] >= 1
    assert payload["break_reasons"]
    assert payload["break_reasons"][0]["reason"]


def test_maybe_capture_dynamo_diagnostics_serializes_exceptions(tmp_path):
    output_path = tmp_path / "diagnostics_error.json"

    def fn(x):
        raise RuntimeError("boom")

    payload = maybe_capture_dynamo_diagnostics(
        fn,
        args=(torch.randn(4),),
        output_path=output_path,
        label="error_unit_test",
        rank=0,
        world_size=1,
    )

    assert payload is not None
    assert payload["graph_break_count"] == -1
    assert payload["error"]["type"] == "RuntimeError"
    assert "boom" in payload["error"]["message"]
