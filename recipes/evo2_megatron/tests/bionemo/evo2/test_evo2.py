# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved.
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

import inspect
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal, Set

import pandas as pd
import pytest
import torch
from megatron.bridge.training.checkpointing import (
    _load_model_weights_from_checkpoint,
)
from megatron.bridge.training.model_load_save import load_model_config, load_tokenizer
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module

from bionemo.common.data.load import load
from bionemo.evo2.data.dataset_tokenizer import DEFAULT_HF_TOKENIZER_MODEL_PATH, DEFAULT_HF_TOKENIZER_MODEL_PATH_512
from bionemo.evo2.models.evo2_provider import (
    Hyena1bModelProvider,
    Hyena7bARCLongContextModelProvider,
    Hyena7bModelProvider,
    HyenaInferenceContext,
)
from bionemo.evo2.utils.checkpoint.nemo2_to_mbridge import run_nemo2_to_mbridge

from .utils import check_fp8_support, distributed_model_parallel_state


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


#############################################################################################
# Core utility functions: Below are some utility functions that allow for loading a nemo2
#  trained model back into a newly initialized megatron core model. The key insight is that
#  the nemo2 lightning module owns a single `self.module = config.configure_model(...)`
#  object. This `config.configure_module(...)` object is the megatron model that we want
#  to load weights into. So we need to adjust the checkpoint keys since they will all
#  have the extra `module.` prefix on them, while the megatron model we just initialized
#  will not. These functions should make a wide variety of fine-tuning strategies doable.


def _munge_key_megatron_to_nemo2(k: str) -> str:
    return f"module.{k}"


def _munge_sharded_tensor_key_megatron_to_nemo2(v: ShardedTensor) -> ShardedTensor:
    # This works with PP=1, how do we handle PP>1?
    key = v.key
    v.key = _munge_key_megatron_to_nemo2(key)
    return v


def _key_in_filter(k: str, filter: Set[str]) -> bool:
    for prefix in filter:
        if k.startswith(prefix):
            return True
    return False


def determine_memory_requirement_and_skip_if_not_met(ckpt_name: str, test_name: str | None = None) -> int:
    """Determine the memory requirement for a given checkpoint and test_name.

    The memory requirement recorded is not discriminated for flash_decode True or False.  The memory requirement
    recorded depend on checkpoint name only through model size.

    Args:
        ckpt_name: str
            the name of the checkpoint to test
        test_name: str | None
            the name of the test that is to be run.

    Returns:
        The input sequence length cap, for the model sin the checkpoint, given certain memory requirements.
        If the memory requirement is not met, the test is skipped.
    """
    # memory_needed_by_test: max reserved rounded up + 1, for stand-alone test
    memory_needed_df = pd.DataFrame(
        [
            {
                "test_name": "test_forward",
                "model_size": "evo2_1b_base",
                "seq_len_cap": 6000,
                "memory_needed_by_test": 18,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward",
                "model_size": "evo2_7b_base",
                "seq_len_cap": 4000,
                "memory_needed_by_test": 33,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward_manual",
                "model_size": "evo2_1b_base",
                "seq_len_cap": 6000,
                "memory_needed_by_test": 18,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward_manual",
                "model_size": "evo2_7b_base",
                "seq_len_cap": 4000,
                "memory_needed_by_test": 21,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward_ckpt_conversion",
                "model_size": "evo2_1b_base",
                "seq_len_cap": 6000,
                "memory_needed_by_test": 18,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward_ckpt_conversion",
                "model_size": "evo2_7b_base",
                "seq_len_cap": 4000,
                "memory_needed_by_test": 21,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate_mbridge",
                "model_size": "evo2_1b_base",
                "seq_len_cap": -1,
                "memory_needed_by_test": 16,
            },  # checked both variants in isolation - needs ~21GB peak on L4
            {
                "test_name": "test_batch_generate_mbridge",
                "model_size": "evo2_7b_base",
                "seq_len_cap": -1,
                "memory_needed_by_test": 43,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate_coding_sequences",
                "model_size": "evo2_1b_base",
                "seq_len_cap": -1,
                "memory_needed_by_test": 12,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate_coding_sequences",
                "model_size": "evo2_7b_base",
                "seq_len_cap": -1,
                "memory_needed_by_test": 28,
            },  # checked both variants in isolation
        ],
        columns=["test_name", "model_size", "seq_len_cap", "memory_needed_by_test"],
    )
    memory_needed_df_wi_index = memory_needed_df.set_index(["test_name", "model_size"])

    if "1b" in ckpt_name:
        model_size = "evo2_1b_base"
    elif "7b" in ckpt_name:
        model_size = "evo2_7b_base"
    else:
        raise ValueError(f"{ckpt_name=} is not supported for testing")

    seq_len_cap = memory_needed_df_wi_index.loc[(test_name, model_size), "seq_len_cap"]
    memory_needed_by_test = memory_needed_df_wi_index.loc[(test_name, model_size), "memory_needed_by_test"]

    # skip_condition_flash = flash_decode is None or flash_decode
    gb_available = torch.cuda.mem_get_info()[0] / 1024**3
    skip_condition = gb_available < memory_needed_by_test
    if skip_condition:
        pytest.skip(
            ", ".join(
                [
                    f"Inference API requires at least {memory_needed_by_test}GB of available memory for {model_size} models",
                    f"{gb_available=}",
                ]
            )
        )
    return seq_len_cap


def load_weights_sharded_inplace_nemo2_to_mcore(
    model: Float16Module,
    distributed_checkpoint_dir: str | Path,
    skip_keys_with_these_prefixes: set[str],
    ckpt_format: Literal["zarr", "torch_dist"] = "torch_dist",
):
    """Load the weights of a nemo2 checkpoint into a megatron core model in place. Deprecate once ckpt is converted."""
    logger.info("Start setting up state dict")
    sharded_state_dict = {
        _munge_key_megatron_to_nemo2(k): _munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not _key_in_filter(
            k, skip_keys_with_these_prefixes
        )  # and "_extra_state" not in k  # extra state is needed for fp8 sharded states
    }
    # Load the checkpoint with strict=false to allow for missing keys (backward compatibility)
    # Error: megatron.core.dist_checkpointing.core.CheckpointingException:
    # Object shard ... module.decoder.final_norm._extra_state/shard_0_1.pt not found
    dist_checkpointing.load(sharded_state_dict, str(distributed_checkpoint_dir))


def _load_prompt_sequences() -> list[str]:
    """Return the DNA prompts used by generation accuracy tests."""
    with (Path(__file__).parent / "data" / "prompts.csv").open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


@pytest.fixture
def sequences():
    """Fixture that returns a list of sequences from the prompts.csv file."""
    return _load_prompt_sequences()


@pytest.fixture
def coding_sequences():
    """Fixture that returns coding sequences from the cds_prompts.csv file."""
    cds_file = Path(__file__).parent / "data" / "cds_prompts.csv"
    if not cds_file.exists():
        pytest.skip(f"CDS prompts file not found: {cds_file}")
    with cds_file.open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


def _calc_matchrate(*, tokenizer, in_seq, logits):
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    o = softmax_logprobs.argmax(dim=-1)[0]
    if hasattr(tokenizer, "tokenize"):
        i = torch.tensor(tokenizer.tokenize(in_seq[1:]), device=o.device)
    else:
        i = torch.tensor(tokenizer.text_to_ids(in_seq[1:]), device=o.device)
    return (i == o).sum().item() / (i.size()[0] - 1)


def _check_matchrate(*, ckpt_name, matchrate, assert_matchrate=True):
    logger.info(f"{ckpt_name} {matchrate = }")
    if "1b-" in ckpt_name:
        if assert_matchrate:
            assert matchrate > 0.70, (ckpt_name, matchrate)
        else:
            print(f"{ckpt_name} {matchrate = }")
    elif "7b-" in ckpt_name:
        if assert_matchrate:
            assert matchrate > 0.79, (ckpt_name, matchrate)
        else:
            print(f"{ckpt_name} {matchrate = }")
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents,flash_decode,subquadratic_ops",
    [
        # Try flash decode with one and not the other to verify that both paths work.
        pytest.param("evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30], True, False, id="1b-8k-bf16"),
        pytest.param(
            "evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30], False, True, id="1b-8k-bf16-subquadratic-ops"
        ),
        pytest.param(
            "evo2/1b-8k-bf16:1.0",
            [96.27, 67.93, 77.50, 80.30],
            True,
            True,
            id="1b-8k-bf16-subquadratic-ops-flash",
        ),
        pytest.param(
            "evo2/1b-8k:1.0",
            [96.27, 67.93, 77.50, 80.30],
            False,
            False,
            id="1b-8k",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-8k:1.0",
            [97.60, 89.63, 80.03, 84.57],
            False,
            False,
            id="7b-8k",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-1m:1.0",
            [97.60, 89.63, 80.03, 84.57],
            False,
            False,
            id="7b-1m",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
    ],
)
def test_forward_manual(
    sequences: list[str],
    ckpt_name: str,
    expected_matchpercents: list[float],
    flash_decode: bool,
    subquadratic_ops: bool,
):
    """Test the forward pass of the megatron model."""
    assert len(sequences) > 0
    seq_len_cap = determine_memory_requirement_and_skip_if_not_met(
        ckpt_name, test_name=inspect.currentframe().f_code.co_name
    )

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    vortex_style_fp8 = is_fp8_supported and ("bf16" not in ckpt_name or "7b" not in ckpt_name)
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    with distributed_model_parallel_state(), torch.no_grad():
        tokenizer = build_tokenizer(
            TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                hf_tokenizer_kwargs={"trust_remote_code": False},
                tokenizer_model=DEFAULT_HF_TOKENIZER_MODEL_PATH,
            )
        )
        flash_decode_kwargs: dict[str, Any] = {"flash_decode": flash_decode}
        if flash_decode:
            flash_decode_kwargs["attention_backend"] = AttnBackend.flash
        if "1b-8k" in ckpt_name:
            model_config = Hyena1bModelProvider(
                use_te=True,
                vocab_size=tokenizer.vocab_size,
                seq_length=8192,
                vortex_style_fp8=vortex_style_fp8,
                use_subquadratic_ops=subquadratic_ops,
                **flash_decode_kwargs,
            )
        elif "7b-8k" in ckpt_name:
            model_config = Hyena7bModelProvider(
                use_te=True,
                vocab_size=tokenizer.vocab_size,
                seq_length=8192,
                vortex_style_fp8=vortex_style_fp8,
                use_subquadratic_ops=subquadratic_ops,
                **flash_decode_kwargs,
            )
        elif "7b-1m" in ckpt_name:
            model_config = Hyena7bARCLongContextModelProvider(
                use_te=True,
                vocab_size=tokenizer.vocab_size,
                seq_length=8192,
                vortex_style_fp8=vortex_style_fp8,
                use_subquadratic_ops=subquadratic_ops,
                **flash_decode_kwargs,
            )
        else:
            raise NotImplementedError
        ckpt_weights: Path = load(ckpt_name) / "weights"
        model_config.finalize()  # important to call finalize before providing the model, this does post_init etc.
        raw_megatron_model = model_config.provide(pre_process=True, post_process=True).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, ckpt_weights, set(), "torch_dist")
        model = Float16Module(model_config, raw_megatron_model)
        if flash_decode:
            inference_context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
            # Ensure full-sequence logits are materialized for tests expecting [B, S, V]
            inference_context.materialize_only_last_token_logits = False
            forward_kwargs = {"runtime_gather_output": True, "inference_context": inference_context}
        else:
            forward_kwargs = {}
        matchrates = []
        for seq in sequences:
            # TODO: artificial limit, megatron uses more memory. Vortex can process full sequences
            partial_seq = seq[:seq_len_cap]
            with torch.no_grad():
                device = torch.cuda.current_device()
                input_ids = torch.tensor(tokenizer.tokenize(partial_seq)).int().unsqueeze(0).to(device)
                attention_mask = None
                # when labels is None, the model returns logits
                logits = model(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attention_mask,
                    labels=None,
                    **forward_kwargs,
                )
                if flash_decode:
                    forward_kwargs["inference_context"].reset()
                matchrate = _calc_matchrate(tokenizer=tokenizer, in_seq=partial_seq, logits=logits)
                matchrates.append(matchrate)
                _check_matchrate(ckpt_name=ckpt_name, matchrate=matchrate, assert_matchrate=False)
        assert len(matchrates) == len(expected_matchpercents)
        matchperc_print = [f"{m * 100.0:.1f}%" for m in matchrates]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
        assert all(m * 100.0 >= 0.95 * ep for m, ep in zip(matchrates, expected_matchpercents)), (
            f"Expected at least 95% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents,flash_decode",
    [
        # Try flash decode with one and not the other to verify that both paths work.
        pytest.param("evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30], True, id="1b-8k-bf16"),
        pytest.param(
            "evo2/1b-8k:1.0",
            [96.27, 67.93, 77.50, 80.30],
            False,
            id="1b-8k",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-8k:1.0",
            [97.60, 89.63, 80.03, 84.57],
            False,
            id="7b-8k",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-1m:1.0",
            [97.60, 89.63, 80.03, 84.57],
            False,
            id="7b-1m",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
    ],
)
def test_forward_ckpt_conversion(
    tmp_path: Path, sequences: list[str], ckpt_name: str, expected_matchpercents: list[float], flash_decode: bool
):
    """Test the forward pass of the megatron model."""
    assert len(sequences) > 0
    seq_len_cap = determine_memory_requirement_and_skip_if_not_met(
        ckpt_name, test_name=inspect.currentframe().f_code.co_name
    )

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported

    # vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    with distributed_model_parallel_state(), torch.no_grad():
        ckpt_path: Path = load(ckpt_name)

        mbridge_ckpt_dir = run_nemo2_to_mbridge(
            nemo2_ckpt_dir=ckpt_path,
            tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
            mbridge_ckpt_dir=tmp_path / "mbridge_checkpoint",
            model_size="evo2_1b_base" if "1b" in ckpt_name else "evo2_7b_base" if "7b-8k" in ckpt_name else "evo2_7b",
            seq_length=1048576 if "1m" in ckpt_name else 8192,
            mixed_precision_recipe="bf16_mixed" if not is_fp8_supported else "bf16_with_fp8_current_scaling_mixed",
            # The checkpoints from the original evo2 training that are "fp8 sensitive" require vortex_style_fp8=True
            #  to run correctly. If we set it in the config going into the conversion then at load time users will
            #  get this setting without having to think about it.
            vortex_style_fp8=is_fp8_supported and "evo2/1b-8k:" in ckpt_name,
        )

        mbridge_ckpt_path = mbridge_ckpt_dir / "iter_0000001"

        model_config, mtron_args = load_model_config(mbridge_ckpt_path)
        assert mtron_args is None, "mtron_args should be None since this is a Megatron Bridge checkpoint"
        tokenizer = load_tokenizer(mbridge_ckpt_path)
        model_config.finalize()  # important to call finalize before providing the model, this does post_init etc.
        raw_megatron_model = model_config.provide(pre_process=True, post_process=True).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        _load_model_weights_from_checkpoint(
            checkpoint_path=mbridge_ckpt_path, model=[raw_megatron_model], dist_ckpt_strictness="ignore_all"
        )
        model = Float16Module(model_config, raw_megatron_model)

        if flash_decode:
            inference_context = HyenaInferenceContext(max_batch_size=1, max_sequence_length=8192)
            # Ensure full-sequence logits are materialized for tests expecting [B, S, V]
            inference_context.materialize_only_last_token_logits = False
            forward_kwargs = {"runtime_gather_output": True, "inference_context": inference_context}
        else:
            forward_kwargs = {}
        matchrates = []
        for seq in sequences:
            # TODO: artificial limit, megatron uses more memory. Vortex can process full sequences
            partial_seq = seq[:seq_len_cap]
            with torch.no_grad():
                # tokens = torch.tensor([tokenizer.tokenize(seq)], device=device)
                input_ids = torch.tensor(tokenizer.tokenize(partial_seq)).int().unsqueeze(0).to(device)
                attention_mask = None
                # when labels is None, the model returns logits
                logits = model(
                    input_ids=input_ids,
                    position_ids=None,
                    attention_mask=attention_mask,
                    labels=None,
                    **forward_kwargs,
                )
                if flash_decode:
                    forward_kwargs["inference_context"].reset()
                matchrate = _calc_matchrate(tokenizer=tokenizer, in_seq=partial_seq, logits=logits)
                matchrates.append(matchrate)
                _check_matchrate(ckpt_name=ckpt_name, matchrate=matchrate, assert_matchrate=False)
        assert len(matchrates) == len(expected_matchpercents)
        matchperc_print = [f"{m * 100.0:.1f}%" for m in matchrates]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
        assert all(m * 100.0 >= 0.95 * ep for m, ep in zip(matchrates, expected_matchpercents)), (
            f"Expected at least 95% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


def mid_point_split(*, seq, num_tokens: int | None = None, fraction: float = 0.5):
    """Split a sequence at a midpoint for prompt/target evaluation."""
    mid_point = int(fraction * len(seq))
    prompt = seq[:mid_point]
    if num_tokens is not None:
        target = seq[mid_point : mid_point + num_tokens]  # Only compare to the section of sequence directly
    else:
        target = seq[mid_point:]
    return prompt, target


def calculate_sequence_identity(seq1: str, seq2: str) -> float | None:
    """Calculate sequence identity between two sequences through direct comparison."""
    if not seq1 or not seq2:
        return None

    # Direct comparison of sequences
    min_length = min(len(seq1), len(seq2))
    matches = sum(a == b for a, b in zip(seq1[:min_length], seq2[:min_length]))

    return (matches / min_length) * 100


@pytest.mark.timeout(900)
@pytest.mark.slow
@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents,fp8",
    [
        pytest.param(
            "evo2/1b-8k-bf16:1.0",
            [86.4, 78.8, 49.7],
            False,
            id="1b-bf16_bf16",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to slow speed"),
        ),
        pytest.param("evo2/1b-8k-bf16:1.0", [86.4, 78.8, 49.7], True, id="1b-bf16_fp8"),
        pytest.param(
            "evo2/1b-8k:1.0",
            [86.4, 78.8, 49.7],
            True,
            id="1b_fp8",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-8k:1.0",
            [88.8, 88.5, 82.2],
            False,
            id="7b-8k_bf16",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-1m:1.0",
            [88.8, 88.5, 82.2],
            False,
            id="7b-1m_bf16",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
    ],
)
def test_batch_generate_coding_sequences(
    coding_sequences: list[str],
    tmp_path: Path,
    ckpt_name: str,
    expected_matchpercents: list[float],
    fp8: bool,
):
    """Test generation on coding sequences through the Evo2 dynamic-inference endpoint.

    This test validates that the model can generate reasonable coding sequence
    continuations, checking for proper stop codon placement and sequence identity.
    """
    from bionemo.evo2.run.infer import generate, setup_inference_engine

    assert len(coding_sequences) > 0

    # Check memory availability
    try:
        _ = determine_memory_requirement_and_skip_if_not_met(
            ckpt_name, test_name="test_batch_generate_coding_sequences"
        )
    except KeyError:
        gb_available = torch.cuda.mem_get_info()[0] / 1024**3
        if gb_available < 16:
            pytest.skip(f"Insufficient GPU memory: {gb_available:.1f}GB available, need at least 16GB")

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    if fp8 and not is_fp8_supported:
        pytest.skip(f"Skipping {ckpt_name} - FP8 not supported on {device_info} ({compute_capability})")

    # Use bf16 checkpoint to avoid FP8 issues with single-token generation
    if "bf16" not in ckpt_name and not fp8:
        pytest.skip(f"Skipping {ckpt_name} - use bf16 checkpoint or enable FP8 for this test")

    # Prepare prompts and targets
    seq_prompts = [mid_point_split(seq=seq, num_tokens=None, fraction=0.3) for seq in coding_sequences]
    num_tokens = max(len(sq[1]) for sq in seq_prompts) + 15
    original_cds_lengths: list[int] = [len(seq) for seq in coding_sequences]

    vortex_style_fp8 = ckpt_name == "evo2/1b-8k:1.0" and fp8
    mixed_precision_recipe = "bf16_with_fp8_current_scaling_mixed" if fp8 and not vortex_style_fp8 else "bf16_mixed"

    with distributed_model_parallel_state(), torch.no_grad():
        # Convert checkpoint to MBridge format
        nemo2_ckpt_path = load(ckpt_name)
        mbridge_ckpt_dir = run_nemo2_to_mbridge(
            nemo2_ckpt_dir=nemo2_ckpt_path,
            tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
            mbridge_ckpt_dir=tmp_path / "mbridge_checkpoint",
            model_size="evo2_1b_base" if "1b" in ckpt_name else "evo2_7b" if "7b-1m" in ckpt_name else "evo2_7b_base",
            seq_length=8192,
            mixed_precision_recipe=mixed_precision_recipe,
            vortex_style_fp8=vortex_style_fp8,
        )
        mbridge_ckpt_path = mbridge_ckpt_dir / "iter_0000001"

        # Extract prompts for generation
        prompts = [split[0] for split in seq_prompts]

        # Setup the public Evo2 generation endpoint; generation is driven by the native dynamic path.
        batch_size = len(prompts) // 2
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_path,
            max_seq_length=8192,
            max_batch_size=batch_size,
            tensor_parallel_size=1,
            random_seed=42,
        )

        # Generate all sequences through the public endpoint.
        results = generate(
            components,
            prompts=prompts,
            max_new_tokens=num_tokens,
            temperature=1.0,
            top_k=1,  # Greedy for determinism
        )

        # Process results
        match_percents: list[float] = []
        cds_lengths: list[int | None] = []
        stop_codons = {"TAA", "TAG", "TGA"}

        for i, (result, (prompt, target)) in enumerate(zip(results, seq_prompts)):
            gen_seq = result.generated_text if result else ""
            logger.info(f"{ckpt_name} {gen_seq=}")
            logger.info(f"{ckpt_name} {target=}")

            full_seq = prompt + gen_seq
            assert full_seq[:3] == "ATG", f"Expected start codon ATG, got {full_seq[:3]}"

            # Find first stop codon
            cds_length = None
            for codon_start in range(0, len(full_seq), 3):
                codon = full_seq[codon_start : codon_start + 3]
                if codon in stop_codons:
                    cds_length = codon_start + 3
                    break
            if cds_length is None:
                logger.warning(f"{ckpt_name} {gen_seq=} no stop codon found")
                cds_length = len(full_seq)
            match_percent: float = calculate_sequence_identity(target, gen_seq) or 0.0
            logger.info(f"{ckpt_name} {match_percent=} expected: {expected_matchpercents[i]}")
            match_percents.append(match_percent)
            cds_lengths.append(cds_length)

        # Verify results
        assert len(match_percents) == len(expected_matchpercents)
        assert len(cds_lengths) == len(original_cds_lengths)
        matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]

        # By chance you expect to have a stop codon within the first 96 codons if everything were random
        # so verify that we are putting the first stop codon after this point, as well as it being at least 90% of the
        # original sequence length.
        assert all(
            pcl is None or ((pcl - len(pmpt) > 96 * 3 or len(tgt) < 96 * 3) and pcl >= 0.90 * ocl)
            for pcl, ocl, (pmpt, tgt) in zip(cds_lengths, original_cds_lengths, seq_prompts)
        ), f"Expected at least 90% of {original_cds_lengths=}, got {cds_lengths=}"

        assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
            f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


# =============================================================================
# MBridge-based generation tests using the public Evo2 dynamic-inference endpoint
# =============================================================================


@pytest.mark.timeout(900)
@pytest.mark.slow
@pytest.mark.parametrize(
    "ckpt_name,expected_matchpercents,fp8",
    [
        pytest.param(
            "evo2/1b-8k-bf16:1.0",
            [96.8, 29.7, 76.6, 71.6],
            False,
            id="1b-bf16_bf16",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to slow speed"),
        ),
        # Full fp8 (fp8 on ALL TE linears) is exercised AT INFERENCE on the bf16 checkpoint. It runs
        # via mcore's fp8 token-padding (prepare_model_for_fp8_inference in setup_inference_engine) but
        # quantizing every linear degrades the highly-conserved seq[0] (~96.6% in bf16 -> ~82.8%); the
        # other sequences are essentially unchanged. These golden values reflect full-fp8 inference.
        pytest.param("evo2/1b-8k-bf16:1.0", [82.8, 32.4, 73.0, 71.2], True, id="1b-bf16_fp8"),
        # The 1b-8k checkpoint is fp8-sensitive and only works as vortex-style fp8 (bf16 recipe + fp8 on
        # just the hyena dense_projection), which preserves accuracy at inference (~96.4% on seq[0]).
        pytest.param(
            "evo2/1b-8k:1.0",
            [96.8, 29.7, 76.6, 71.6],
            True,
            id="1b_fp8",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-8k:1.0",
            [97.60, 89.63, 80.03, 84.57],
            True,
            id="7b-8k_fp8",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
        pytest.param(
            "evo2/7b-1m:1.0",
            [97.60, 89.63, 80.03, 84.57],
            False,
            id="7b-1m_bf16",
            marks=pytest.mark.skipif(bool(os.environ.get("CI")), reason="Skip in CI due to disk space"),
        ),
    ],
)
def test_batch_generate_mbridge(
    sequences: list[str],
    tmp_path: Path,
    ckpt_name: str,
    expected_matchpercents: list[float],
    fp8: bool,
):
    """Test autoregressive generation through the Evo2 dynamic-inference endpoint.

    This test validates that the model can generate reasonable continuations
    of DNA sequences using the same setup_inference_engine/generate API exposed by the standalone `infer_evo2` CLI.

    Note: setup_inference_engine wires the model onto the native dynamic path,
    so this test exercises request creation, Hyena state binding, and decode
    through the same endpoint used by the standalone CLI.

    Uses the same expected values as the original NeMo test_batch_generate.
    """
    from bionemo.evo2.run.infer import generate, setup_inference_engine

    assert len(sequences) > 0

    # Check memory availability (use test_batch_generate requirements as proxy)
    try:
        _ = determine_memory_requirement_and_skip_if_not_met(ckpt_name, test_name="test_batch_generate_mbridge")
    except KeyError:
        # If no entry exists, check basic memory availability
        gb_available = torch.cuda.mem_get_info()[0] / 1024**3
        if gb_available < 16:
            pytest.skip(f"Insufficient GPU memory: {gb_available:.1f}GB available, need at least 16GB")

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    if fp8 and not is_fp8_supported:
        pytest.skip(f"Skipping {ckpt_name} - FP8 not supported on {device_info} ({compute_capability})")

    num_tokens_to_generate = 500  # Match original test
    # Precision modes covered by the (ckpt, fp8) params, run at inference (not just at conversion):
    #   * bf16            -> bf16_mixed,  no fp8                (id "1b-bf16_bf16")
    #   * full fp8        -> fp8 on ALL TE linears             (id "1b-bf16_fp8", bf16 checkpoint)
    #   * vortex-style fp8 -> bf16 recipe + fp8 only on the    (id "1b_fp8", the fp8-required 1b-8k ckpt)
    #                         hyena dense_projection
    # The 1b-8k checkpoint is fp8-sensitive and only works as vortex-style fp8 (a bf16 recipe with a
    # very small number of fp8 layers); the bf16 checkpoint tolerates full fp8.
    vortex_style_fp8 = ckpt_name == "evo2/1b-8k:1.0" and fp8
    mixed_precision_recipe = "bf16_with_fp8_current_scaling_mixed" if fp8 and not vortex_style_fp8 else "bf16_mixed"

    with distributed_model_parallel_state(), torch.no_grad():
        # Convert checkpoint to MBridge format
        nemo2_ckpt_path = load(ckpt_name)
        mbridge_ckpt_dir = run_nemo2_to_mbridge(
            nemo2_ckpt_dir=nemo2_ckpt_path,
            tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
            mbridge_ckpt_dir=tmp_path / "mbridge_checkpoint",
            model_size="evo2_1b_base" if "1b" in ckpt_name else "evo2_7b" if "7b-1m" in ckpt_name else "evo2_7b_base",
            seq_length=8192,
            mixed_precision_recipe=mixed_precision_recipe,
            vortex_style_fp8=vortex_style_fp8,
        )
        mbridge_ckpt_path = mbridge_ckpt_dir / "iter_0000001"

        # Split all sequences at midpoint to get prompts and targets
        seq_splits = [mid_point_split(seq=seq, num_tokens=num_tokens_to_generate, fraction=0.5) for seq in sequences]
        prompts = [split[0] for split in seq_splits]
        targets = [split[1] for split in seq_splits]

        # Setup the public Evo2 generation endpoint.
        # max_batch_size=1 keeps this memory-heavy test bounded.
        # Run inference at the SAME precision the checkpoint was converted with (the prior version
        # always inferred in bf16, so the fp8 ids never actually exercised fp8 at generation time).
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_path,
            max_seq_length=8192,
            max_batch_size=1,  # 1 because this test takes more memory.
            tensor_parallel_size=1,
            random_seed=42,
            mixed_precision_recipe=mixed_precision_recipe,
            vortex_style_fp8=vortex_style_fp8,
        )

        # Generate all sequences through the public endpoint.
        results = generate(
            components,
            prompts=prompts,
            max_new_tokens=num_tokens_to_generate,
            temperature=1.0,
            top_k=1,  # Greedy for determinism
        )

        # Calculate match percentages for each result
        match_percents: list[float] = []
        for i, (result, target) in enumerate(zip(results, targets)):
            generated_text = result.generated_text if result else ""
            match_percent = calculate_sequence_identity(target, generated_text)
            if match_percent is not None:
                match_percents.append(match_percent)
                logger.info(
                    f"{ckpt_name} seq[{i}] identity: {match_percent:.1f}% expected: {expected_matchpercents[i]:.1f}%"
                )

        # Use original assertion style - expect at least 90% of expected values
        assert len(match_percents) == len(expected_matchpercents)
        matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
        assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
            f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


_RUN_SUBQ_MBRIDGE_INPROCESS_ENV = "BIONEMO_EVO2_RUN_SUBQ_MBRIDGE_INPROCESS"


def _run_subq_mbridge_test_subprocess(tmp_path: Path) -> None:
    """Run the native-kernel subquadratic coverage in a child pytest process."""
    env = os.environ.copy()
    env[_RUN_SUBQ_MBRIDGE_INPROCESS_ENV] = "1"
    node_id = f"{Path(__file__).resolve()}::test_batch_generate_mbridge_subquadratic_ops"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-vv",
        node_id,
        "--basetemp",
        str(tmp_path / "subpytest"),
    ]
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=1200,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"subquadratic MBridge subprocess timed out after {exc.timeout}s")

    output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    if result.returncode == 0:
        if "XFAIL" in result.stdout or "xfailed" in result.stdout.lower():
            pytest.xfail("subquadratic MBridge subprocess xfailed")
        return

    if "failed a CUDA self-test" in output or "subquadratic_ops kernels unsupported" in output:
        pytest.xfail("subquadratic_ops_torch CUDA kernels are unsupported in this environment")

    pytest.fail(f"subquadratic MBridge subprocess failed with returncode={result.returncode}:\n{output}")


@pytest.mark.timeout(900)
@pytest.mark.slow
def test_batch_generate_mbridge_subquadratic_ops(tmp_path: Path):
    """Run subquadratic MBridge generation coverage in an isolated pytest subprocess."""
    if os.environ.get(_RUN_SUBQ_MBRIDGE_INPROCESS_ENV) != "1":
        _run_subq_mbridge_test_subprocess(tmp_path)
        return
    _run_batch_generate_mbridge_subquadratic_ops(_load_prompt_sequences(), tmp_path)


def _run_batch_generate_mbridge_subquadratic_ops(sequences: list[str], tmp_path: Path):
    """Second-half match accuracy through the dynamic engine with the fused subquadratic-ops kernels.

    Mirrors :func:`test_batch_generate_mbridge` (1b-bf16, greedy) but enables ``use_subquadratic_ops``
    so the b2b causal-conv1d prefill and fft/causal-conv1d FIR kernels are exercised end-to-end on the
    native dynamic decode path. Because subquadratic-ops kernels cannot be captured into a CUDA graph,
    ``setup_inference_engine`` forces ``cuda_graph_impl='none'`` (eager decode) when they are enabled.
    This gives accuracy coverage for the subquadratic-ops path through the dynamic engine, not just the
    default conv path. It xfails on hardware where the
    prebuilt subquadratic kernels fail their CUDA self-test (unsupported PTX/toolchain), matching the
    other subquadratic tests in this recipe.
    """
    from bionemo.evo2.models.megatron.hyena.subquadratic_safety import ensure_subquadratic_ops_supported
    from bionemo.evo2.run.infer import generate, setup_inference_engine

    ckpt_name = "evo2/1b-8k-bf16:1.0"
    expected_matchpercents = [96.8, 29.7, 76.6, 71.6]

    assert len(sequences) > 0
    try:
        _ = determine_memory_requirement_and_skip_if_not_met(
            ckpt_name, test_name="test_batch_generate_mbridge_subquadratic_ops"
        )
    except KeyError:
        gb_available = torch.cuda.mem_get_info()[0] / 1024**3
        if gb_available < 16:
            pytest.skip(f"Insufficient GPU memory: {gb_available:.1f}GB available, need at least 16GB")

    # Skip-as-xfail when the prebuilt subquadratic kernels do not pass their CUDA self-test here.
    try:
        ensure_subquadratic_ops_supported(torch.cuda.current_device())
    except Exception as exc:
        pytest.xfail(f"subquadratic_ops kernels unsupported on this device: {exc}")

    num_tokens_to_generate = 500
    with distributed_model_parallel_state(), torch.no_grad():
        nemo2_ckpt_path = load(ckpt_name)
        mbridge_ckpt_dir = run_nemo2_to_mbridge(
            nemo2_ckpt_dir=nemo2_ckpt_path,
            tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
            mbridge_ckpt_dir=tmp_path / "mbridge_checkpoint",
            model_size="evo2_1b_base",
            seq_length=8192,
            mixed_precision_recipe="bf16_mixed",
            vortex_style_fp8=False,
        )
        mbridge_ckpt_path = mbridge_ckpt_dir / "iter_0000001"

        seq_splits = [mid_point_split(seq=seq, num_tokens=num_tokens_to_generate, fraction=0.5) for seq in sequences]
        prompts = [split[0] for split in seq_splits]
        targets = [split[1] for split in seq_splits]

        # use_subquadratic_ops=True routes prefill/FIR through the fused subquadratic kernels.
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_path,
            max_seq_length=8192,
            max_batch_size=1,
            tensor_parallel_size=1,
            random_seed=42,
            use_subquadratic_ops=True,
        )
        results = generate(
            components,
            prompts=prompts,
            max_new_tokens=num_tokens_to_generate,
            temperature=1.0,
            top_k=1,  # Greedy for determinism
        )

        match_percents = [
            calculate_sequence_identity(target, (result.generated_text if result else "")) or 0.0
            for result, target in zip(results, targets)
        ]
        for i, mp in enumerate(match_percents):
            logger.info(
                f"{ckpt_name} subquadratic seq[{i}] identity: {mp:.1f}% expected: {expected_matchpercents[i]:.1f}%"
            )

        assert len(match_percents) == len(expected_matchpercents)
        matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
        assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
            f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


@pytest.mark.timeout(900)
@pytest.mark.slow
def test_native_dynamic_multi_batch_reuses_engine(tmp_path: Path):
    """The dynamic engine serves many generate() batches through ONE persistent CUDA-graphed context.

    ``setup_inference_engine`` builds a single dynamic context and captures the per-layer decode CUDA
    graphs once (warmup). This test drives several ``generate`` calls of differing prompt counts and
    lengths through that one engine and asserts (a) no CUDA-graph argument mismatch / crash across
    calls and (b) greedy determinism: re-running an earlier batch yields byte-identical output. Both
    only hold if the persistent context and its captured graph are correctly reused across calls --
    i.e. it guards the regression where a fresh per-prompt context broke graph replay, and the
    capture-on-first-real-prompt bug that corrupted the first generated sequence.
    """
    from bionemo.evo2.run.infer import generate, setup_inference_engine

    ckpt_name = "evo2/1b-8k-bf16:1.0"
    try:
        _ = determine_memory_requirement_and_skip_if_not_met(
            ckpt_name, test_name="test_native_dynamic_multi_batch_reuses_engine"
        )
    except KeyError:
        gb_available = torch.cuda.mem_get_info()[0] / 1024**3
        if gb_available < 16:
            pytest.skip(f"Insufficient GPU memory: {gb_available:.1f}GB available, need at least 16GB")

    # Batches of differing sizes/lengths; batch_a is repeated last to check cross-call determinism.
    batch_a = ["ACGTACGTACGTACGTACGT" * 4, "TTACGGGCATTACGGGCATT" * 3]
    batch_b = ["ACGTACGTACGTACGTACGT" * 30]  # a single, longer prompt routed through the same engine

    with distributed_model_parallel_state(), torch.no_grad():
        nemo2_ckpt_path = load(ckpt_name)
        mbridge_ckpt_dir = run_nemo2_to_mbridge(
            nemo2_ckpt_dir=nemo2_ckpt_path,
            tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
            mbridge_ckpt_dir=tmp_path / "mbridge_checkpoint",
            model_size="evo2_1b_base",
            seq_length=8192,
            mixed_precision_recipe="bf16_mixed",
            vortex_style_fp8=False,
        )
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_dir / "iter_0000001",
            max_seq_length=8192,
            max_batch_size=2,
            tensor_parallel_size=1,
            random_seed=42,
        )

        def _generate_texts(prompts: list[str]) -> list[str]:
            results = generate(components, prompts=prompts, max_new_tokens=30, temperature=1.0, top_k=1)
            return [result.generated_text for result in results]

        out_a1 = _generate_texts(batch_a)
        out_b = _generate_texts(batch_b)  # a different batch through the same persistent engine
        out_a2 = _generate_texts(batch_a)  # repeat of the first batch

        assert len(out_a1) == 2 and len(out_b) == 1 and len(out_a2) == 2
        assert all(len(text) > 0 for text in out_a1 + out_b + out_a2), "a generate() batch produced 0 tokens"
        assert out_a1 == out_a2, f"cross-call nondeterminism with the reused engine:\n{out_a1}\n{out_a2}"


@pytest.mark.timeout(900)
@pytest.mark.slow
def test_native_dynamic_auto_max_seq_length_and_grow(tmp_path: Path):
    """Prompt-based auto ``max_seq_length`` that GROWS on demand for a later, larger prompt.

    With no manual ``max_seq_length`` the engine sizes its persistent (CUDA-graph-pinned) context to
    the prompt it first sees — longest prompt + ``max_new_tokens`` + headroom — rather than a fixed
    8k / GPU-memory heuristic. A later, longer prompt does not fail: because mcore has no in-place
    resize, the engine rebuilds the context at a larger size and re-captures the CUDA graphs once,
    then keeps generating. A subsequent shorter prompt reuses the (now larger) context without
    shrinking and reproduces the earlier output, proving the rebuild + re-capture stays correct.
    This guards the auto-sizing and the grow-by-rebuild + graph-recapture path.
    """
    from bionemo.evo2.run.infer import generate, setup_inference_engine

    ckpt_name = "evo2/1b-8k-bf16:1.0"
    try:
        _ = determine_memory_requirement_and_skip_if_not_met(
            ckpt_name, test_name="test_native_dynamic_auto_max_seq_length_and_grow"
        )
    except KeyError:
        gb_available = torch.cuda.mem_get_info()[0] / 1024**3
        if gb_available < 16:
            pytest.skip(f"Insufficient GPU memory: {gb_available:.1f}GB available, need at least 16GB")

    short_prompt = "ACGT" * 10  # ~40 tokens
    long_prompt = "ACGT" * 200  # ~800 tokens -> forces a grow of the auto budget
    max_new_tokens = 20

    with distributed_model_parallel_state(), torch.no_grad():
        nemo2_ckpt_path = load(ckpt_name)
        mbridge_ckpt_dir = run_nemo2_to_mbridge(
            nemo2_ckpt_dir=nemo2_ckpt_path,
            tokenizer_path=DEFAULT_HF_TOKENIZER_MODEL_PATH_512,
            mbridge_ckpt_dir=tmp_path / "mbridge_checkpoint",
            model_size="evo2_1b_base",
            seq_length=8192,
            mixed_precision_recipe="bf16_mixed",
            vortex_style_fp8=False,
        )
        # No max_seq_length => auto-size from prompts (and grow on demand).
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_dir / "iter_0000001",
            max_batch_size=1,
            tensor_parallel_size=1,
            random_seed=42,
        )
        nd = components.native_dynamic
        assert nd.max_seq_length is None and nd.max_seq_length_is_auto

        r_short = generate(components, prompts=[short_prompt], max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)
        initial_msl = len(components.tokenizer.tokenize(short_prompt)) + max_new_tokens + 8  # headroom
        assert nd.max_seq_length == initial_msl, (nd.max_seq_length, initial_msl)
        assert r_short[0].generated_length > 0

        # A larger prompt GROWS the context (rebuild + CUDA-graph re-capture), no error, still generates.
        r_long = generate(components, prompts=[long_prompt], max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)
        grown_msl = nd.max_seq_length
        assert grown_msl > initial_msl
        assert grown_msl >= len(components.tokenizer.tokenize(long_prompt)) + max_new_tokens + 8
        assert r_long[0].generated_length > 0

        # A later shorter prompt reuses the grown context (no shrink) and reproduces the earlier output.
        r_short2 = generate(
            components, prompts=[short_prompt], max_new_tokens=max_new_tokens, temperature=1.0, top_k=1
        )
        assert nd.max_seq_length == grown_msl
        assert r_short2[0].generated_text == r_short[0].generated_text
