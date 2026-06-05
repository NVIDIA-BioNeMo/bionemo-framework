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

from bionemo.core.data.load import load
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


@pytest.fixture
def sequences():
    """Fixture that returns a list of sequences from the prompts.csv file."""
    with (Path(__file__).parent / "data" / "prompts.csv").open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


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
    """Test generation on coding sequences using MCore inference infrastructure.

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

        # Setup MCore inference engine with batch size matching number of prompts
        batch_size = len(prompts) // 2
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_path,
            max_seq_length=8192,
            max_batch_size=batch_size,
            tensor_parallel_size=1,
            random_seed=42,
        )

        # Generate all sequences - engine handles iteration internally
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
# MBridge-based generation tests using HyenaInferenceContext
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
        pytest.param("evo2/1b-8k-bf16:1.0", [96.8, 29.7, 76.6, 71.6], True, id="1b-bf16_fp8"),
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
    """Test autoregressive generation using MCore inference infrastructure.

    This test validates that the model can generate reasonable continuations
    of DNA sequences using the StaticInferenceEngine and TextGenerationController.

    Note: Hyena/Evo2 SSM state caching currently only supports batch size 1,
    so prompts are processed sequentially. The MCore inference engine handles
    this internally through legacy mode.

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

        # Setup MCore inference engine
        # Note: max_batch_size=1 due to Hyena SSM state constraints, but engine handles iteration
        components = setup_inference_engine(
            ckpt_dir=mbridge_ckpt_path,
            max_seq_length=8192,
            max_batch_size=1,  # 1 because this test takes more memory.
            tensor_parallel_size=1,
            random_seed=42,
        )

        # Generate all sequences - engine handles iteration internally with max_batch_size=1
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
