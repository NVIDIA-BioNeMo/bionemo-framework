# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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

"""Tests for Evo2 text generation (inference) using MBridge.

infer.py drives generation through the NATIVE mcore dynamic-inference engine (paged-KV attention +
Hyena recurrent state packed into mcore's two Mamba slots), which is the only engine here.
The general generation tests below
(test_infer_runs, test_infer_temperature, test_infer_top_k, test_infer_phylogenetic_prompt,
test_identical_prompts_should_be_identical, test_subquadratic_ops_matches_baseline,
test_different_prompts_produce_different_outputs, test_different_results_with_without_peft,
the batch-padding prefix-invariance test, and the parallel-accuracy tests) all exercise this
engine; they assert "infer.py generates valid DNA" rather than any engine-specific internal.
The native dynamic tests add edge-case coverage (full-prompt multi-block prefill, opt-in
chunked prefill, single-token decode, longer generation, short-prompt right-aligned seed, TP=2
batch=1).

The core forward pass (predict.py) and HyenaInferenceContext are tested
in test_evo2.py which has working test_forward_manual and test_forward_ckpt_conversion.
"""

import copy
import csv
import json
import os
import subprocess
from pathlib import Path

import pytest
import torch

from bionemo.core.data.load import load as bionemo_load
from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.models.evo2_provider import HyenaInferenceContext
from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge
from bionemo.evo2.utils.checkpoint.savanna_to_mbridge import savanna_to_mbridge

from ..utils import find_free_network_port


# Capture environment at import time (consistent with test_predict.py)
PRETEST_ENV = copy.deepcopy(os.environ)

# Note: mbridge_checkpoint_path fixture is provided by conftest.py at session scope


def _xfail_if_unsupported_subquadratic_ops(result: subprocess.CompletedProcess, use_subquadratic_ops: bool) -> None:
    if use_subquadratic_ops and "failed a CUDA self-test" in result.stderr:
        pytest.xfail("subquadratic_ops_torch CUDA kernels are unsupported in this environment")


def _read_jsonl_results(output_file: Path) -> list[dict]:
    """Read JSONL output file and return parsed records."""
    records = []
    with open(output_file) as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def test_infer_runs(mbridge_checkpoint_path, tmp_path):
    """Test that infer.py runs without errors and produces JSONL output."""
    output_file = tmp_path / "output.jsonl"

    # Use a longer DNA prompt to meet FP8 dimension requirements (divisible by 8)
    # 64 characters should be safe
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "10",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",  # Non-zero temperature required by MCore
        "--top-k",
        "1",  # Top-k=1 for greedy decoding
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) == 1, f"Expected 1 result, got {len(records)}"
    record = records[0]
    assert record["id"] == "0"
    assert record["prompt"] == prompt
    assert len(record["completion"]) > 0, "Generated text is empty"
    assert record["finish_reason"] in ("length", "stop")
    assert "usage" in record
    assert record["usage"]["prompt_tokens"] > 0
    assert record["usage"]["completion_tokens"] > 0


@pytest.mark.parametrize("temperature", [0.5, 1.0])
def test_infer_temperature(mbridge_checkpoint_path, tmp_path, temperature):
    """Test that different temperatures produce output."""
    output_file = tmp_path / f"output_temp_{temperature}.jsonl"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "5",
        "--temperature",
        str(temperature),
        "--output-file",
        str(output_file),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_top_k(mbridge_checkpoint_path, tmp_path):
    """Test top-k sampling."""
    output_file = tmp_path / "output_topk.jsonl"
    # Use a longer prompt for FP8 compatibility
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "5",
        "--top-k",
        "4",  # Only sample from top 4 tokens (A, C, G, T)
        "--output-file",
        str(output_file),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"


def test_infer_phylogenetic_prompt(mbridge_checkpoint_path, tmp_path):
    """Test generation with a phylogenetic lineage prompt.

    Evo2 is trained with phylogenetic tags, so generation should work
    well when conditioned on these tags. Using a longer prompt for FP8.
    """
    output_file = tmp_path / "output_phylo.jsonl"

    # Phylogenetic prompt (padded to be longer for FP8 compatibility)
    prompt = (
        "|d__Bacteria;"
        "p__Pseudomonadota;"
        "c__Gammaproteobacteria;"
        "o__Enterobacterales;"
        "f__Enterobacteriaceae;"
        "g__Escherichia;"
        "s__Escherichia|"
    )
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        "20",
        "--temperature",
        "1.0",  # Non-zero temperature required by MCore
        "--top-k",
        "1",  # Top-k=1 for greedy decoding
        "--output-file",
        str(output_file),
    ]

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) == 1
    assert len(records[0]["completion"]) > 0, "Generated text is empty"


# DNA prompts for reproducibility tests (from test_prompt.py)
PROMPT_1 = "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGGGAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGG"
PROMPT_2 = "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTT"


def run_infer_subprocess(
    mbridge_checkpoint_path,
    prompt: str,
    output_file,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    use_subquadratic_ops: bool = False,
    max_seq_length: int | None = None,
    return_log_probs: bool = False,
    extra_args: list[str] | None = None,
):
    """Helper function to run inference as a subprocess.

    Generation runs through the native mcore dynamic-inference engine (the only engine: paged-KV
    attention + Hyena state in mcore Mamba slots).

    Args:
        mbridge_checkpoint_path: Path to the MBridge checkpoint
        prompt: Input prompt for the model
        output_file: Path to write output (JSONL)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter (1 for greedy)
        seed: Random seed for reproducibility
        use_subquadratic_ops: Pass --use-subquadratic-ops to the CLI.
        max_seq_length: If set, pass --max-seq-length (caps the per-context allocation).
        return_log_probs: Pass --return-log-probs (logprobs included in the JSONL record).
        extra_args: Additional CLI arguments appended to the infer command.

    Returns:
        The single JSONL result record (dict) for the prompt.
    """
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt",
        prompt,
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-file",
        str(output_file),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
    ]
    if use_subquadratic_ops:
        cmd.append("--use-subquadratic-ops")
    if max_seq_length is not None:
        cmd.extend(["--max-seq-length", str(max_seq_length)])
    if return_log_probs:
        cmd.append("--return-log-probs")
    if extra_args:
        cmd.extend(extra_args)

    env = copy.deepcopy(PRETEST_ENV)

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minutes
        env=env,
    )

    _xfail_if_unsupported_subquadratic_ops(result, use_subquadratic_ops)
    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    records = _read_jsonl_results(output_file)
    assert len(records) == 1, f"Expected 1 JSONL record, got {len(records)}"
    return records[0]


def mid_point_split(*, seq, num_tokens: int | None = None, fraction: float = 0.5):
    """Split a sequence at a midpoint for prompt/target evaluation."""
    mid_point = int(fraction * len(seq))
    prompt = seq[:mid_point]
    if num_tokens is not None:
        target = seq[mid_point : mid_point + num_tokens]
    else:
        target = seq[mid_point:]
    return prompt, target


def calculate_sequence_identity(seq1: str, seq2: str) -> float | None:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))
    return (matches / min_length) * 100


def _recipe_root() -> Path:
    """Return the recipe root directory (evo2_megatron/)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent


def _infer_script_path() -> Path:
    """Return the path to the source infer.py script.

    Uses the source version directly (rather than the installed module via ``-m``)
    so that local fixes to infer.py are picked up without reinstalling the package.
    """
    return _recipe_root() / "src" / "bionemo" / "evo2" / "run" / "infer.py"


def _write_prompts_jsonl(prompt_file: Path, prompts: list[tuple[str, str]]) -> None:
    """Write a list of (id, prompt) pairs into a JSONL file."""
    with open(prompt_file, "w") as f:
        f.writelines(json.dumps({"id": prompt_id, "prompt": prompt_text}) + "\n" for prompt_id, prompt_text in prompts)


@pytest.fixture(
    params=[False, True],
    ids=["causal-conv1d", "subquadratic-ops"],
)
def infer_use_subquadratic_ops(request):
    """Whether infer should use subquadratic Hyena kernels."""
    return request.param


def _run_infer_prompt_file(
    *,
    mbridge_checkpoint_path: Path,
    prompt_file: Path,
    output_file: Path,
    max_batch_size: int,
    use_subquadratic_ops: bool,
) -> dict[str, dict]:
    open_port = find_free_network_port()
    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        "-m",
        "bionemo.evo2.run.infer",
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        "1",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",
        "--top-k",
        "1",
        "--seed",
        "1234",
        "--max-batch-size",
        str(max_batch_size),
        "--max-seq-length",
        "512",
        "--return-log-probs",
    ]
    if use_subquadratic_ops:
        cmd.append("--use-subquadratic-ops")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=512,
        env=copy.deepcopy(PRETEST_ENV),
    )
    _xfail_if_unsupported_subquadratic_ops(result, use_subquadratic_ops)
    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    records = _read_jsonl_results(output_file)
    return {record["id"]: record for record in records}


def _completion_logprobs(record: dict) -> torch.Tensor:
    logprobs = record.get("logprobs", {}).get("completion_logprobs")
    assert logprobs is not None, f"Missing completion logprobs in record: {record}"
    tensor = torch.as_tensor(logprobs, dtype=torch.float32).flatten()
    assert tensor.numel() == 1
    return tensor


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_infer_evo2_short_prefill_is_prefix_invariant_across_batch_padding(
    mbridge_checkpoint_path,
    tmp_path,
    infer_use_subquadratic_ops: bool,
):
    """A short prefill should generate the same next token alone or in a padded batch.

    Routes through the native default engine. Native decodes each prompt as its own single-request
    context (no static batch padding), so the short prompt's completion + logprob must match whether
    it is submitted alone or alongside a longer prompt — the same "infer.py generates valid,
    batch-independent DNA" invariant, now exercised on the working path.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Inference prefill prefix-invariance test requires a GPU")

    short_prompt = "ACGTACGTAA"
    padding_prompt = ("GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTC" * 4)[:256]

    alone_prompt_file = tmp_path / "short_alone_prompts.jsonl"
    padded_prompt_file = tmp_path / "short_padded_prompts.jsonl"
    _write_prompts_jsonl(alone_prompt_file, [("short", short_prompt)])
    _write_prompts_jsonl(padded_prompt_file, [("padding", padding_prompt), ("short", short_prompt)])

    alone_records = _run_infer_prompt_file(
        mbridge_checkpoint_path=mbridge_checkpoint_path,
        prompt_file=alone_prompt_file,
        output_file=tmp_path / "alone_output.jsonl",
        max_batch_size=1,
        use_subquadratic_ops=infer_use_subquadratic_ops,
    )
    padded_records = _run_infer_prompt_file(
        mbridge_checkpoint_path=mbridge_checkpoint_path,
        prompt_file=padded_prompt_file,
        output_file=tmp_path / "padded_output.jsonl",
        max_batch_size=2,
        use_subquadratic_ops=infer_use_subquadratic_ops,
    )

    assert set(alone_records) == {"short"}
    assert set(padded_records) == {"padding", "short"}
    assert padded_records["short"]["prompt"] == short_prompt
    assert alone_records["short"]["completion"] == padded_records["short"]["completion"]

    torch.testing.assert_close(
        _completion_logprobs(alone_records["short"]),
        _completion_logprobs(padded_records["short"]),
        rtol=2e-2,
        atol=5e-2,
    )


def run_infer_subprocess_parallel(
    mbridge_checkpoint_path,
    prompt_file: Path,
    output_file: Path,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    top_k: int = 1,
    seed: int = 42,
    tensor_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
) -> list[dict]:
    """Run inference as a subprocess with model parallelism.

    Runs the source infer.py script directly (not the installed module) so that
    local fixes are picked up without reinstalling the package.  The caller is
    responsible for writing the JSONL prompt file beforehand.

    Args:
        mbridge_checkpoint_path: Path to the MBridge checkpoint.
        prompt_file: Path to an existing JSONL prompt file.
        output_file: Path to write JSONL output.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter (1 for greedy).
        seed: Random seed for reproducibility.
        tensor_parallel_size: Tensor parallelism degree.
        pipeline_model_parallel_size: Pipeline parallelism degree.
        context_parallel_size: Context parallelism degree.

    Returns:
        List of parsed JSONL result dicts.
    """
    nproc_per_node = tensor_parallel_size * pipeline_model_parallel_size * context_parallel_size
    open_port = find_free_network_port()

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(nproc_per_node),
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        str(_infer_script_path()),
        "--ckpt-dir",
        str(mbridge_checkpoint_path),
        "--prompt-file",
        str(prompt_file),
        "--max-new-tokens",
        str(max_new_tokens),
        "--output-file",
        str(output_file),
        "--temperature",
        str(temperature),
        "--top-k",
        str(top_k),
        "--seed",
        str(seed),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--pipeline-model-parallel-size",
        str(pipeline_model_parallel_size),
        "--context-parallel-size",
        str(context_parallel_size),
    ]

    env = copy.deepcopy(PRETEST_ENV)
    # Prepend the source src/ directory to PYTHONPATH so that local model code
    # (hyena_mixer.py, hyena_utils.py, etc.) is used instead of the installed package.
    src_dir = str(_recipe_root() / "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,  # 15 minutes for parallel configs
        env=env,
    )

    assert result.returncode == 0, f"infer command failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    assert output_file.exists(), "Output file was not created"

    return _read_jsonl_results(output_file)


def test_identical_prompts_should_be_identical(mbridge_checkpoint_path, tmp_path):
    """Test that identical prompts produce identical sequences.

    With greedy decoding (top_k=1) and the same seed, identical prompts
    should produce identical outputs.
    """
    output_file_1 = tmp_path / "output_prompt1_run1.jsonl"
    output_file_2 = tmp_path / "output_prompt1_run2.jsonl"

    # Run inference twice with the same prompt
    generated_1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_1,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding for determinism
        seed=42,
    )

    generated_2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding for determinism
        seed=42,
    )

    assert len(generated_1) > 0, "First generation produced empty output"
    assert len(generated_2) > 0, "Second generation produced empty output"
    assert generated_1 == generated_2, (
        f"Identical prompts with same seed and greedy decoding produced different outputs:\n"
        f"Run 1: {generated_1}\n"
        f"Run 2: {generated_2}"
    )


def test_subquadratic_ops_matches_baseline(mbridge_checkpoint_path, tmp_path):
    """Greedy generation with --use-subquadratic-ops must match the standard path.

    This is the end-to-end correctness check for the subq-ops inference path.
    The subq path uses guarded subquadratic kernels. If the local CUDA/GPU
    combination cannot run those kernels correctly, the guard raises before
    invalid outputs can propagate. With greedy decoding (top_k=1) and the same
    seed, supported subq kernels must produce identical output.
    """
    output_baseline = tmp_path / "output_baseline.jsonl"
    output_subq = tmp_path / "output_subq.jsonl"

    generated_baseline = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_baseline,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        use_subquadratic_ops=False,
    )

    generated_subq = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_subq,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        use_subquadratic_ops=True,
    )

    assert len(generated_baseline) > 0, "Baseline generation produced empty output"
    assert len(generated_subq) > 0, "Subq-ops generation produced empty output"
    assert generated_baseline == generated_subq, (
        f"Subq-ops path diverged from baseline:\nBaseline: {generated_baseline}\nSubq-ops: {generated_subq}"
    )


def test_different_prompts_produce_different_outputs(mbridge_checkpoint_path, tmp_path):
    """Test that different prompts produce different sequences.

    Different input prompts should produce different outputs, demonstrating
    that the model is actually responding to the prompt content.
    """
    output_file_1 = tmp_path / "output_prompt1.jsonl"
    output_file_2 = tmp_path / "output_prompt2.jsonl"

    # Run inference with two different prompts
    generated_1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=output_file_1,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
    )

    generated_2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_2,
        output_file=output_file_2,
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
    )

    assert len(generated_1) > 0, "First generation produced empty output"
    assert len(generated_2) > 0, "Second generation produced empty output"

    # The outputs should be different since the prompts are different
    # We check that the generated portions (after the prompt) are not identical
    assert generated_1 != generated_2, (
        f"Different prompts produced identical outputs:\n"
        f"Prompt 1 output: {generated_1}\n"
        f"Prompt 2 output: {generated_2}"
    )


@pytest.fixture
def dna_sequences():
    """Load DNA sequences from prompts.csv test data."""
    prompts_csv = Path(__file__).resolve().parent.parent / "data" / "prompts.csv"
    with prompts_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        return [row["Sequence"] for row in reader]


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp, cp",
    [
        # The 1b model only supports TP=1 through infer.py due to divisibility constraints
        # (15 attention heads and 128-width HyenaMixer). TP>1 requires the 7b model.
        pytest.param(1, 1, id="tp=1,cp=1"),
        pytest.param(
            1,
            2,
            id="tp=1,cp=2",
            marks=pytest.mark.xfail(reason="CP>1 is known broken for inference", strict=False),
        ),
    ],
)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy(mbridge_checkpoint_path, tmp_path, dna_sequences, tp, cp):
    """Test that parallel inference produces accurate generation results.

    Loads real DNA sequences, splits them in half, generates 500 tokens from the first half,
    and compares the generated tokens against the known second half using sequence identity.
    This mirrors the pattern in test_batch_generate_mbridge in test_evo2.py but exercises
    the subprocess-based infer.py CLI with parallelism.
    """
    num_gpus_required = tp * cp
    if torch.cuda.device_count() < num_gpus_required:
        pytest.skip(f"Not enough GPUs: need {num_gpus_required}, have {torch.cuda.device_count()}")

    num_tokens = 500
    # Expected sequence identity percentages for the 1b-8k-bf16 checkpoint (from test_evo2.py)
    expected_matchpercents = [96.8, 29.7, 76.6, 71.6]

    # Build a single JSONL prompt file with all sequences, keyed by id
    targets_by_id: dict[str, str] = {}
    expected_by_id: dict[str, float] = {}
    jsonl_entries = []
    for i, (seq, expected_mp) in enumerate(zip(dna_sequences, expected_matchpercents)):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)
        seq_id = f"seq_{i}"
        targets_by_id[seq_id] = target
        expected_by_id[seq_id] = expected_mp
        jsonl_entries.append((seq_id, prompt))

    prompt_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "outputs.jsonl"
    _write_prompts_jsonl(prompt_file, jsonl_entries)

    # Single inference call processes all prompts (batching handled internally)
    records = run_infer_subprocess_parallel(
        mbridge_checkpoint_path,
        prompt_file=prompt_file,
        output_file=output_file,
        max_new_tokens=num_tokens,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
        tensor_parallel_size=tp,
        context_parallel_size=cp,
    )

    assert len(records) == len(dna_sequences), f"Expected {len(dna_sequences)} results, got {len(records)}"

    # Match results by id (output order is not guaranteed with dynamic engines)
    results_by_id = {r["id"]: r for r in records}
    match_percents = {}
    for seq_id, target in targets_by_id.items():
        assert seq_id in results_by_id, f"Missing result for {seq_id}"
        identity = calculate_sequence_identity(target, results_by_id[seq_id]["completion"])
        match_percents[seq_id] = identity

    matchperc_print = {k: f"{v:.2f}%" for k, v in match_percents.items()}
    matchperc_print_expected = {k: f"{v:.2f}%" for k, v in expected_by_id.items()}

    assert all(match_percents[sid] >= 0.90 * expected_by_id[sid] for sid in targets_by_id), (
        f"Expected at least 90% of {matchperc_print_expected}, got {matchperc_print}"
    )


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_1m_path(tmp_path_factory) -> Path:
    """Create or load a MBridge checkpoint for 7b-1m model testing."""
    try:
        nemo2_checkpoint_path = bionemo_load("evo2/7b-1m:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            pytest.skip(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e

    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b")
    mbridge_ckpt_dir = run_nemo2_to_mbridge(
        nemo2_ckpt_dir=nemo2_checkpoint_path,
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="evo2_7b",
        seq_length=8192,
        mixed_precision_recipe="bf16_mixed",
        vortex_style_fp8=False,
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.mark.slow
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tp, pp, cp",
    [
        # The 7b model has 32 attention heads, supporting TP=1, 2, 4, 8
        # TP-only configs
        pytest.param(1, 1, 1, id="tp=1,pp=1,cp=1"),
        pytest.param(2, 1, 1, id="tp=2,pp=1,cp=1"),
        pytest.param(4, 1, 1, id="tp=4,pp=1,cp=1"),
        pytest.param(8, 1, 1, id="tp=8,pp=1,cp=1"),
        # PP-only configs
        pytest.param(1, 2, 1, id="tp=1,pp=2,cp=1"),
        pytest.param(1, 4, 1, id="tp=1,pp=4,cp=1"),
        pytest.param(1, 8, 1, id="tp=1,pp=8,cp=1"),
        # Combined TP+PP configs
        pytest.param(2, 2, 1, id="tp=2,pp=2,cp=1"),
        pytest.param(4, 2, 1, id="tp=4,pp=2,cp=1"),
        # CP>1 configs (known broken)
        pytest.param(
            1,
            1,
            2,
            id="tp=1,pp=1,cp=2",
            marks=pytest.mark.xfail(reason="CP>1 is known broken for inference", strict=False),
        ),
    ],
)
@pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI")
def test_parallel_inference_accuracy_7b(mbridge_checkpoint_7b_1m_path, tmp_path, dna_sequences, tp, pp, cp):
    """Test that parallel inference with the 7b model produces accurate generation results.

    Uses the 7b-1m checkpoint which supports TP>1 (32 attention heads) and PP>1,
    enabling proper tensor and pipeline parallel accuracy testing.
    """
    num_gpus_required = tp * pp * cp
    if torch.cuda.device_count() < num_gpus_required:
        pytest.skip(f"Not enough GPUs: need {num_gpus_required}, have {torch.cuda.device_count()}")

    num_tokens = 500
    # Expected sequence identity percentages for the 7b model (from test_evo2.py)
    expected_matchpercents = [97.60, 89.63, 80.03, 84.57]

    # Build a single JSONL prompt file with all sequences, keyed by id
    targets_by_id: dict[str, str] = {}
    expected_by_id: dict[str, float] = {}
    jsonl_entries = []
    for i, (seq, expected_mp) in enumerate(zip(dna_sequences, expected_matchpercents)):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)
        seq_id = f"seq_{i}"
        targets_by_id[seq_id] = target
        expected_by_id[seq_id] = expected_mp
        jsonl_entries.append((seq_id, prompt))

    prompt_file = tmp_path / "prompts.jsonl"
    output_file = tmp_path / "outputs.jsonl"
    _write_prompts_jsonl(prompt_file, jsonl_entries)

    # Single inference call processes all prompts (batching handled internally)
    records = run_infer_subprocess_parallel(
        mbridge_checkpoint_7b_1m_path,
        prompt_file=prompt_file,
        output_file=output_file,
        max_new_tokens=num_tokens,
        temperature=1.0,
        top_k=1,  # Greedy decoding
        seed=42,
        tensor_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
    )

    assert len(records) == len(dna_sequences), f"Expected {len(dna_sequences)} results, got {len(records)}"

    # Match results by id (output order is not guaranteed with dynamic engines)
    results_by_id = {r["id"]: r for r in records}
    match_percents = {}
    for seq_id, target in targets_by_id.items():
        assert seq_id in results_by_id, f"Missing result for {seq_id}"
        identity = calculate_sequence_identity(target, results_by_id[seq_id]["completion"])
        match_percents[seq_id] = identity

    matchperc_print = {k: f"{v:.2f}%" for k, v in match_percents.items()}
    matchperc_print_expected = {k: f"{v:.2f}%" for k, v in expected_by_id.items()}

    assert all(match_percents[sid] >= 0.90 * expected_by_id[sid] for sid in targets_by_id), (
        f"Expected at least 90% of {matchperc_print_expected}, got {matchperc_print}"
    )


SAVANNA_7B_REPO = "arcinstitute/savanna_evo2_7b"


@pytest.fixture(scope="module")
def mbridge_checkpoint_7b_from_savanna(tmp_path_factory) -> Path:
    """Convert the ARC Savanna 7B checkpoint to MBridge and return the iteration directory.

    Downloads the savanna checkpoint from HuggingFace, converts it via
    ``savanna_to_mbridge``, and returns the ``iter_0000001`` path ready for
    inference.
    """
    tmp_dir = tmp_path_factory.mktemp("mbridge_ckpt_7b_savanna")
    mbridge_ckpt_dir = savanna_to_mbridge(
        savanna_ckpt_path=SAVANNA_7B_REPO,
        mbridge_ckpt_dir=tmp_dir / "mbridge_checkpoint",
        model_size="evo2_7b",
        tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
        seq_length=8192,
        te_enabled=True,
        mixed_precision_recipe="bf16_mixed",
    )
    return mbridge_ckpt_dir / "iter_0000001"


@pytest.mark.slow
@pytest.mark.timeout(1800)
@pytest.mark.skipif(
    not os.environ.get("LONG_TESTS"),
    reason="Set LONG_TESTS=1 to run (downloads ~30GB savanna checkpoint)",
)
def test_savanna_to_mbridge_inference_accuracy_7b(mbridge_checkpoint_7b_from_savanna, tmp_path, dna_sequences):
    """Validate the Savanna-to-MBridge conversion by running inference at TP=2.

    Downloads the ARC 7B savanna checkpoint, converts it to MBridge, generates
    500 tokens for each test sequence, and checks that sequence identity matches
    expected baselines within 90%.
    """
    tp = 2
    if torch.cuda.device_count() < tp:
        pytest.skip(f"Not enough GPUs: need {tp}, have {torch.cuda.device_count()}")

    num_tokens = 500
    expected_matchpercents = [97.60, 89.63, 80.03, 84.57]

    match_percents = []
    for i, seq in enumerate(dna_sequences):
        prompt, target = mid_point_split(seq=seq, num_tokens=num_tokens, fraction=0.5)

        prompt_file = tmp_path / f"prompt_savanna7b_seq{i}.txt"
        output_file = tmp_path / f"output_savanna7b_seq{i}.txt"
        prompt_file.write_text(prompt)

        generated_text = run_infer_subprocess_parallel(
            mbridge_checkpoint_7b_from_savanna,
            prompt_file=prompt_file,
            output_file=output_file,
            max_new_tokens=num_tokens,
            temperature=1.0,
            top_k=1,
            seed=42,
            tensor_parallel_size=tp,
        )

        identity = calculate_sequence_identity(target, generated_text)
        match_percents.append(identity)

    matchperc_print = [f"{mp:.2f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.2f}%" for ep in expected_matchpercents]

    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )


@pytest.mark.timeout(512)
@pytest.mark.slow
def test_different_results_with_without_peft(tmp_path, mbridge_checkpoint_path, lora_finetune_checkpoint):
    """Greedy-generate from the base ckpt vs. the LoRA ckpt and assert the logprobs differ."""
    env = copy.deepcopy(PRETEST_ENV)
    # 64-char prompt for FP8 divisibility.
    prompt = "ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"

    def _run_infer(ckpt: Path, output_file: Path) -> dict:
        port = find_free_network_port()
        cmd = [
            "torchrun",
            "--nproc_per_node",
            "1",
            "--nnodes",
            "1",
            "--master_port",
            str(port),
            "-m",
            "bionemo.evo2.run.infer",
            "--ckpt-dir",
            str(ckpt),
            "--prompt",
            prompt,
            "--max-new-tokens",
            "10",
            "--temperature",
            "1.0",
            "--top-k",
            "1",
            "--seed",
            "0",
            "--return-log-probs",
            "--output-file",
            str(output_file),
        ]
        r = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=300, env=env)
        assert r.returncode == 0, f"infer_evo2 failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
        with open(output_file) as f:
            return json.loads(f.readline())

    base = _run_infer(mbridge_checkpoint_path, tmp_path / "out_base.jsonl")
    lora = _run_infer(lora_finetune_checkpoint, tmp_path / "out_lora.jsonl")

    base_lp = base["logprobs"]["completion_logprobs"]
    lora_lp = lora["logprobs"]["completion_logprobs"]
    assert len(base_lp) == len(lora_lp), f"Different completion lengths: {len(base_lp)} vs {len(lora_lp)}"
    assert base_lp != lora_lp, "LoRA adapter had no effect on completion logprobs"


def test_hyena_inference_context_initialization():
    """Test that HyenaInferenceContext can be initialized."""
    context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
    assert context is not None
    assert context.max_batch_size == 1
    assert context.max_sequence_length == 8192


def test_hyena_inference_context_reset():
    """Test that context reset works without error."""
    context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
    # Add some fake filter state (simulating what hyena layers do)
    context.filter_state_dict_layer_0 = {"key": torch.zeros(10)}
    context.filter_state_dict_layer_1 = {"key": torch.ones(10)}

    # Verify the state was added
    assert hasattr(context, "filter_state_dict_layer_0")
    assert hasattr(context, "filter_state_dict_layer_1")

    # Reset should remove all filter_state_dict attributes
    context.reset()

    assert not hasattr(context, "filter_state_dict_layer_0")
    assert not hasattr(context, "filter_state_dict_layer_1")


def test_hyena_inference_context_materialize_logits_setting():
    """Test that materialize_only_last_token_logits can be configured."""
    context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)

    # Default should be True for efficiency
    # We can set it to False if we need full sequence logits
    context.materialize_only_last_token_logits = False
    assert context.materialize_only_last_token_logits is False

    context.materialize_only_last_token_logits = True
    assert context.materialize_only_last_token_logits is True


def test_hyena_inference_context_multiple_batches():
    """Test context with different batch sizes."""
    for batch_size in [1, 2, 4]:
        context = HyenaInferenceContext(max_batch_size=batch_size, max_sequence_length=4096)
        assert context.max_batch_size == batch_size
        context.reset()  # Should not error


def test_hyena_inference_context_different_sequence_lengths():
    """Test context with different max sequence lengths."""
    for seq_len in [1024, 8192, 16384]:
        context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=seq_len)
        assert context.max_sequence_length == seq_len
        context.reset()


# =============================================================================
# Native dynamic-inference engine edge-case tests
# =============================================================================
# These exercise the NATIVE mcore dynamic-inference path (paged-KV attention + Hyena recurrent
# state packed into mcore's two Mamba slots). They run against the small 1b-8k-bf16 fixture
# checkpoint (real weights, validates the mechanism + correctness, not just shapes). Edge cases
# cover full-prompt multi-block prefill (prompt > block_size_tokens), opt-in chunked prefill,
# single-token decode, longer generation, TP-non-divisible batch (batch=1 on TP=2), and
# prompt-shorter-than-the-medium-FIR-ring behavior. Greedy decoding (top_k=1) keeps the
# assertions deterministic.

# A long DNA prompt (> block_size_tokens=256) that forces multi-block paged-KV prefill.
LONG_DNA_PROMPT = (
    "GAATAGGAACAGCTCCGGTCTACAGCTCCCAGCGTGAGCGACGCAGAAGACGGTGATTTCTGCATTTCCATCTGAGGTACCGGGTTCATCTCACTAGG"
    "GAGTGCCAGACAGTGGGCGCAGGCCAGTGTGTGTGCGCACCGTGCGCGAGCCGAAGCAGGGCGAGGCATTGCCTCACCTGGGAAGCGCAAGGGGTCAG"
    "GGAGTTCCCTTTCCGAGTCAAAGAAAGGGGTGACGGACGCACCTGGAAAATCGGGTCACTCCCACCCGAATATTGCGCTTTTCAGACCGGCTTAAGAA"
    "ACGGCGCACCACGAGACTATATCCCACAC"
)
assert len(LONG_DNA_PROMPT) > 256, "LONG_DNA_PROMPT must exceed block_size_tokens=256 to cover >1 KV block"

DNA_BASES = set("ACGTacgtNn")


def _is_dna_completion(text: str) -> bool:
    """True when every character of ``text`` is a DNA base (Evo2's byte vocab)."""
    return len(text) > 0 and all(c in DNA_BASES for c in text)


def test_native_dynamic_runs(mbridge_checkpoint_path, tmp_path):
    """A short prompt generates a non-empty DNA completion through the native engine."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt="ACGTACGTAACCGGTTACGTACGTAACCGGTT",
        output_file=tmp_path / "native_runs.jsonl",
        max_new_tokens=10,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    assert record["usage"]["prompt_tokens"] > 0
    assert record["usage"]["completion_tokens"] == 10
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_full_prefill_multi_block(mbridge_checkpoint_path, tmp_path):
    """A prompt longer than block_size_tokens (256) prefills as one multi-block request.

    Without --enable-chunked-prefill, the whole prompt is enqueued as one prefill chunk that
    uses multiple 256-token KV blocks. The first forward processes all prompt tokens, and
    last_token_logits selects the true final position before decode begins.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    n_prompt_tokens = len(LONG_DNA_PROMPT)
    assert n_prompt_tokens > 256
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "native_full_prefill_multi_block.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    # The whole prompt must have been prefilled (multi-block) and 20 tokens generated.
    assert record["usage"]["prompt_tokens"] == n_prompt_tokens, (
        f"prompt_tokens {record['usage']['prompt_tokens']} != {n_prompt_tokens}; multi-block "
        "prefill did not enqueue the full prompt"
    )
    assert record["usage"]["completion_tokens"] == 20
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_chunked_prefill_cli_multi_chunk(mbridge_checkpoint_path, tmp_path):
    """--enable-chunked-prefill allows prompts to exceed the per-step token budget."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    n_prompt_tokens = len(LONG_DNA_PROMPT)
    max_tokens = 256
    assert n_prompt_tokens > max_tokens
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=LONG_DNA_PROMPT,
        output_file=tmp_path / "native_chunked_prefill.jsonl",
        max_new_tokens=4,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
        extra_args=[
            "--enable-chunked-prefill",
            "--inference-dynamic-batching-max-tokens",
            str(max_tokens),
        ],
    )
    assert record["usage"]["prompt_tokens"] == n_prompt_tokens
    assert record["usage"]["completion_tokens"] == 4
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_single_token_decode(mbridge_checkpoint_path, tmp_path):
    """A single decode step (max_new_tokens=1) produces exactly one token after prefill."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt="ACGTACGTAACCGGTTACGTACGTAACCGGTT",
        output_file=tmp_path / "native_single.jsonl",
        max_new_tokens=1,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=256,
    )
    assert record["usage"]["completion_tokens"] == 1, "expected exactly one decoded token"
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_short_prompt_under_medium_ring(mbridge_checkpoint_path, tmp_path):
    """A prompt shorter than the medium-FIR ring (127) prefills via the right-aligned seed.

    The medium Hyena operator's recurrent FIR ring is 127 wide; a short prompt produces a seed
    shorter than the ring. The packed-slot path right-aligns that short seed into the fixed-width
    ring (numerically equivalent to the eager grow path for the flip-filter medium operator). This
    guards that fix: a ~16-token prompt must still generate a valid DNA completion.
    """
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt="ACGTACGTAACCGGTT",  # 16 tokens << 127 (medium ring width)
        output_file=tmp_path / "native_short.jsonl",
        max_new_tokens=10,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=256,
    )
    assert record["usage"]["completion_tokens"] == 10
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_long_generation(mbridge_checkpoint_path, tmp_path):
    """A longer generation (100 tokens) runs many decode steps without context overflow."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    record = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_long_gen.jsonl",
        max_new_tokens=100,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=1024,
    )
    assert record["usage"]["completion_tokens"] == 100, "long generation did not reach 100 tokens"
    assert _is_dna_completion(record["completion"]), f"non-DNA completion: {record['completion']!r}"


def test_native_dynamic_deterministic(mbridge_checkpoint_path, tmp_path):
    """Greedy decoding with the same prompt + seed is reproducible across runs."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    rec1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_det1.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    rec2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_det2.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    assert rec1["completion"] == rec2["completion"], (
        f"native greedy decode not deterministic:\n  run1: {rec1['completion']}\n  run2: {rec2['completion']}"
    )


def test_native_dynamic_different_prompts_differ(mbridge_checkpoint_path, tmp_path):
    """Different prompts produce different completions (the model responds to the prompt)."""
    if torch.cuda.device_count() < 1:
        pytest.skip("Native dynamic-engine test requires a GPU")
    rec1 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_1,
        output_file=tmp_path / "native_diff1.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    rec2 = run_infer_subprocess(
        mbridge_checkpoint_path,
        prompt=PROMPT_2,
        output_file=tmp_path / "native_diff2.jsonl",
        max_new_tokens=20,
        temperature=1.0,
        top_k=1,
        seed=42,
        max_seq_length=512,
    )
    assert rec1["completion"] != rec2["completion"], "different prompts produced identical completions"


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_native_dynamic_tp2_batch1(mbridge_checkpoint_7b_1m_path, tmp_path):
    """TP=2 with a single request (batch=1) runs through decode-only CUDA graphs.

    Evo2 keeps sequence parallelism disabled for standalone inference and sizes each context to
    the active request count, while mcore pads decode graph dimensions only as needed for TP
    alignment. Needs the 7b checkpoint (32 heads, TP-divisible) + 2 GPUs.
    """
    tp = 2
    if torch.cuda.device_count() < tp:
        pytest.skip(f"TP={tp} requires {tp} GPUs, have {torch.cuda.device_count()}")
    open_port = find_free_network_port()
    output_file = tmp_path / "native_tp2.jsonl"
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(tp),
        "--nnodes",
        "1",
        "--master_port",
        str(open_port),
        str(_infer_script_path()),
        "--ckpt-dir",
        str(mbridge_checkpoint_7b_1m_path),
        "--prompt",
        "ACGTACGTAACCGGTTACGTACGTAACCGGTT",
        "--max-new-tokens",
        "10",
        "--output-file",
        str(output_file),
        "--temperature",
        "1.0",
        "--top-k",
        "1",
        "--seed",
        "42",
        "--tensor-parallel-size",
        str(tp),
        "--max-seq-length",
        "256",
    ]
    env = copy.deepcopy(PRETEST_ENV)
    env["PYTHONPATH"] = str(_recipe_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=600, env=env)
    assert result.returncode == 0, f"native TP=2 infer failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    records = _read_jsonl_results(output_file)
    assert len(records) == 1
    assert records[0]["usage"]["completion_tokens"] == 10
    assert _is_dna_completion(records[0]["completion"]), f"non-DNA completion: {records[0]['completion']!r}"
