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
import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
import pytest
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.module import Float16Module
from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import HyenaInferenceContext
from nemo.collections.llm.inference import generate
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.io.pl import MegatronCheckpointIO

from bionemo.core.data.load import load
from bionemo.llm.utils.weight_utils import (
    MegatronModelType,
    _key_in_filter,
    _munge_key_megatron_to_nemo2,
    _munge_sharded_tensor_key_megatron_to_nemo2,
)
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state
from bionemo.testing.torch import check_fp8_support


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


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
                "model_size": "1b",
                "seq_len_cap": 6000,
                "memory_needed_by_test": 18,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward",
                "model_size": "7b",
                "seq_len_cap": 4000,
                "memory_needed_by_test": 33,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward_manual",
                "model_size": "1b",
                "seq_len_cap": 6000,
                "memory_needed_by_test": 18,
            },  # checked both variants in isolation
            {
                "test_name": "test_forward_manual",
                "model_size": "7b",
                "seq_len_cap": 4000,
                "memory_needed_by_test": 21,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate",
                "model_size": "1b",
                "seq_len_cap": -1,
                "memory_needed_by_test": 16,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate",
                "model_size": "7b",
                "seq_len_cap": -1,
                "memory_needed_by_test": 43,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate_coding_sequences",
                "model_size": "1b",
                "seq_len_cap": -1,
                "memory_needed_by_test": 6,
            },  # checked both variants in isolation
            {
                "test_name": "test_batch_generate_coding_sequences",
                "model_size": "7b",
                "seq_len_cap": -1,
                "memory_needed_by_test": 21,
            },  # checked both variants in isolation
            {
                "test_name": "test_generate_speed",
                "model_size": "1b",
                "seq_len_cap": -1,
                "memory_needed_by_test": -1,
            },  # skipped for now until Anton's changes
            {
                "test_name": "test_generate_speed",
                "model_size": "7b",
                "seq_len_cap": -1,
                "memory_needed_by_test": -1,
            },  # skipped for now until Anton's changes
        ],
        columns=["test_name", "model_size", "seq_len_cap", "memory_needed_by_test"],
    )
    memory_needed_df_wi_index = memory_needed_df.set_index(["test_name", "model_size"])

    if "1b" in ckpt_name:
        model_size = "1b"
    elif "7b" in ckpt_name:
        model_size = "7b"
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
    model: MegatronModelType,
    distributed_checkpoint_dir: str | Path,
    skip_keys_with_these_prefixes: set[str],
    ckpt_format: Literal["zarr", "torch_dist"] = "torch_dist",
):
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
    MegatronCheckpointIO(save_ckpt_format=ckpt_format).load_checkpoint(
        distributed_checkpoint_dir, sharded_state_dict=sharded_state_dict, strict=False
    )


@pytest.mark.parametrize("seq_len", [8_192, 16_384])
def test_golden_values_top_k_logits_and_cosine_similarity(seq_len: int):
    try:
        evo2_1b_checkpoint_weights: Path = load("evo2/1b-8k:1.0") / "weights"
        gold_standard_no_fp8 = load("evo2/1b-8k-nofp8-te-goldvalue-testdata-A6000:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    with distributed_model_parallel_state(), torch.no_grad():
        hyena_config = llm.Hyena1bConfig(use_te=True, seq_length=seq_len)
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        raw_megatron_model = hyena_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, evo2_1b_checkpoint_weights, {}, "torch_dist")
        model = Float16Module(hyena_config, raw_megatron_model)
        input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAAT"
        input_ids = torch.tensor(tokenizer.text_to_ids(input_seq)).int().unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_seq)).unsqueeze(0).to(device)
        attention_mask = None
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        gold_standard_no_fp8_tensor = torch.load(gold_standard_no_fp8).to(device=outputs.device, dtype=outputs.dtype)
        top_2_logits_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=True, largest=True, k=2)
        ambiguous_positions = (
            top_2_logits_golden.values[..., 0] - top_2_logits_golden.values[..., 1]
        ).abs() < 9.9e-3  # hand tunes for observed diffs from A100 and H100
        n_ambiguous = ambiguous_positions.sum()

        assert n_ambiguous <= 19

        our_char_indices = outputs.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        not_amb_positions = ~ambiguous_positions.flatten().cpu().numpy()
        # Generate our string, removing the ambiguous positions.
        our_generation_str = "".join([chr(idx) for idx in our_char_indices[not_amb_positions].tolist()])
        # Do the same to the golden values
        gold_std_char_indices = (
            gold_standard_no_fp8_tensor.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        )
        # Make the string
        gold_std_str = "".join([chr(idx) for idx in gold_std_char_indices[not_amb_positions].tolist()])
        array_eq = np.array(list(our_generation_str)) == np.array(list(gold_std_str))
        # Ensure the two strings are approximately equal.
        if array_eq.mean() < 0.95:
            array_eq = np.array(list(our_generation_str)) == np.array(list(gold_std_str))
            mismatch_positions = np.arange(outputs.shape[1])[not_amb_positions][~array_eq]
            err_str = f"Fraction of expected mismatch positions exceeds 5%: {(~array_eq).mean()}"
            err_str += f"Mismatch positions: {mismatch_positions}"
            err_str += f"Fraction of unexpected mismatch positions: {(~array_eq).mean()}"
            top_two_logits_at_mismatch = top_2_logits_golden.values[0, mismatch_positions]
            top_2_logits_pred = outputs.topk(dim=-1, sorted=True, largest=True, k=2)
            top_two_pred_logits_at_mismatch = top_2_logits_pred.values[0, mismatch_positions]
            err_str += f"Top two logits at mismatch positions: {top_two_logits_at_mismatch}"
            err_str += f"Top two pred logits at mismatch positions: {top_two_pred_logits_at_mismatch}"
            raise AssertionError(err_str)

        # Verify that the top-4 from the logit vectors are the same.
        # A: 65
        # C: 67
        # G: 71
        # T: 84
        # Find the corresponding ATGC and compare the two vectors with those four values.
        # Ensures that the top 4 ascii characters of the output are ACGT.
        top_4_inds = outputs.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        output_vector = outputs[0, -1, top_4_inds.indices]

        # Then its the top 4 indices of the gold standard tensor
        top_4_inds_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds_golden.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        gold_standard_no_fp8_vector = gold_standard_no_fp8_tensor[0, -1, top_4_inds_golden.indices]

        # Run cosine similarity between the two vectors.
        logit_similarity = torch.nn.functional.cosine_similarity(output_vector, gold_standard_no_fp8_vector, dim=-1)
        assert torch.mean(torch.abs(logit_similarity - torch.ones_like(logit_similarity))) < 0.03


@pytest.mark.skip(reason="test fails on main, not due to #1058")
@pytest.mark.slow
def test_golden_values_top_k_logits_and_cosine_similarity_7b(seq_len: int = 8_192):
    try:
        evo2_7b_checkpoint_weights: Path = load("evo2/7b-8k:1.0") / "weights"
        gold_standard_no_fp8 = load("evo2/7b-8k-nofp8-te-goldvalue-testdata:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    with distributed_model_parallel_state(), torch.no_grad():
        hyena_config = llm.Hyena7bConfig(use_te=True, seq_length=seq_len)
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        raw_megatron_model = hyena_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, evo2_7b_checkpoint_weights, {}, "torch_dist")
        model = Float16Module(hyena_config, raw_megatron_model)
        input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAAT"
        input_ids = torch.tensor(tokenizer.text_to_ids(input_seq)).int().unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_seq)).unsqueeze(0).to(device)
        attention_mask = None
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        gold_standard_no_fp8_tensor = torch.load(gold_standard_no_fp8).to(device=outputs.device, dtype=outputs.dtype)
        is_fp8_supported, compute_capability, device_info = check_fp8_support(device.index)

        if is_fp8_supported and compute_capability == "9.0":
            # Most rigurous assertion for output equivalence currently works on devices that are new enough to
            #  support FP8.
            logger.info(
                f"Device {device_info} ({compute_capability}) supports FP8 with 9.0 compute capability, the "
                "same configuration as the gold standard was generated with. Running most rigurous assertion."
            )
            torch.testing.assert_close(outputs, gold_standard_no_fp8_tensor)
        else:
            logger.info(
                f"Device {device_info} ({compute_capability}) does not support FP8. Running less rigurous assertions."
            )
        top_2_logits_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=True, largest=True, k=2)
        ambiguous_positions = (
            top_2_logits_golden.values[..., 0] - top_2_logits_golden.values[..., 1]
        ).abs() < 9.9e-3  # hand tunes for observed diffs from A100 and H100 with 7b model
        n_ambiguous = ambiguous_positions.sum()

        assert n_ambiguous <= 19

        our_char_indices = outputs.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        not_amb_positions = ~ambiguous_positions.flatten().cpu().numpy()
        # Generate our string, removing the ambiguous positions.
        our_generation_str = "".join([chr(idx) for idx in our_char_indices[not_amb_positions].tolist()])
        # Do the same to the golden values
        gold_std_char_indices = (
            gold_standard_no_fp8_tensor.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy()
        )
        # Make the string
        gold_std_str = "".join([chr(idx) for idx in gold_std_char_indices[not_amb_positions].tolist()])

        # Ensure the two strings are equal.
        assert all(np.array(list(our_generation_str)) == np.array(list(gold_std_str)))

        # Verify that the top-4 from the logit vectors are the same.
        # A: 65
        # C: 67
        # G: 71
        # T: 84
        # Find the corresponding ATGC and compare the two vectors with those four values.
        # Ensures that the top 4 ascii characters of the output are ACGT.
        top_4_inds = outputs.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        output_vector = outputs[0, -1, top_4_inds.indices]

        # Then its the top 4 indices of the gold standard tensor
        top_4_inds_golden = gold_standard_no_fp8_tensor.topk(dim=-1, sorted=False, largest=True, k=4)
        assert set(top_4_inds_golden.indices.flatten().cpu().numpy().tolist()).issubset((65, 67, 71, 84))
        gold_standard_no_fp8_vector = gold_standard_no_fp8_tensor[0, -1, top_4_inds_golden.indices]

        # Run cosine similarity between the two vectors.
        logit_similarity = torch.nn.functional.cosine_similarity(output_vector, gold_standard_no_fp8_vector, dim=-1)
        assert torch.mean(torch.abs(logit_similarity - torch.ones_like(logit_similarity))) < 9.9e-3


@pytest.fixture
def sequences():
    with (Path(__file__).parent / "data" / "prompts.csv").open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


@pytest.fixture
def coding_sequences():
    with (Path(__file__).parent / "data" / "cds_prompts.csv").open(newline="") as f:
        from csv import DictReader

        reader = DictReader(f)
        return [row["Sequence"] for row in reader]


def get_trainer(pipeline_parallel=1):
    import nemo.lightning as nl

    fp8 = True
    full_fp8 = False
    return nl.Trainer(
        accelerator="gpu",
        devices=pipeline_parallel,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_parallel,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )


# here: pass arg through to inference_batch_times_seqlen_threshold and inference_max_seq_length
def get_model_and_tokenizer_raw(ckpt_dir_or_name: Path | str, seq_len_max: int = 8192, **kwargs):
    """
    Load a model and tokenizer from a checkpoint directory or name. If you supply a Path argument then we assume that
    the path is already a checkpoint directory, otherwise we load the checkpoint from NGC or PBSS depending on
    the environment variable BIONEMO_DATA_SOURCE.
    """
    trainer = get_trainer()
    from bionemo.core.data.load import load

    if isinstance(ckpt_dir_or_name, Path):
        ckpt_dir: Path = ckpt_dir_or_name
    else:
        ckpt_dir: Path = load(ckpt_dir_or_name)
    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=trainer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=seq_len_max,
        inference_max_seq_length=seq_len_max,
        recompute_granularity=None,
        recompute_num_layers=None,
        recompute_method=None,
        **kwargs,
    )
    return inference_wrapped_model, mcore_tokenizer


def get_model_and_tokenizer(ckpt_name, vortex_style_fp8=False, seq_len_max: int = 8192, **kwargs):
    return get_model_and_tokenizer_raw(ckpt_name, vortex_style_fp8=vortex_style_fp8, seq_len_max=seq_len_max, **kwargs)


def get_model_and_tokenizer_ignore_vortex(ckpt_name, vortex_style_fp8=False, seq_len_max: int = 8192, **kwargs):
    # Capture and remove the vortex_style_fp8 argument for mamba models.
    return get_model_and_tokenizer_raw(ckpt_name, seq_len_max=seq_len_max, **kwargs)


def calc_matchrate(*, tokenizer, in_seq, logits):
    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1]
    o = softmax_logprobs.argmax(dim=-1)[0]
    if hasattr(tokenizer, "tokenize"):
        i = torch.tensor(tokenizer.tokenize(in_seq[1:]), device=o.device)
    else:
        i = torch.tensor(tokenizer.text_to_ids(in_seq[1:]), device=o.device)
    return (i == o).sum().item() / (i.size()[0] - 1)


def check_matchrate(*, ckpt_name, matchrate, assert_matchrate=True):
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
    "ckpt_name,expected_matchpercents",
    [
        ("evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30]),
        ("evo2/1b-8k:1.0", [96.27, 67.93, 77.50, 80.30]),
        ("evo2/7b-8k:1.0", [97.60, 89.63, 80.03, 84.57]),
        ("evo2/7b-1m:1.0", [97.60, 89.63, 80.03, 84.57]),
    ],
)
def test_forward(sequences: list[str], ckpt_name: str, expected_matchpercents: list[float]):
    assert len(sequences) > 0
    seq_len_cap = determine_memory_requirement_and_skip_if_not_met(
        ckpt_name, test_name=inspect.currentframe().f_code.co_name
    )

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer(
        ckpt_name, vortex_style_fp8=vortex_style_fp8, flash_decode=True, enable_flash_decode=True
    )
    matchrates = []
    for seq in sequences:
        seq = seq[:seq_len_cap]  # TODO: artificial limit, megatron uses more memory. Vortex can process full sequences
        with torch.no_grad():
            device = torch.cuda.current_device()
            tokens = torch.tensor([mcore_tokenizer.tokenize(seq)], device=device)
            forward_args = {
                "tokens": tokens,
                "position_ids": None,
                "attention_mask": None,
            }

            inference_wrapped_model.prep_model_for_inference(prompts_tokens=None)
            # Ensure full-sequence logits are materialized for tests expecting [B, S, V]
            inference_wrapped_model.inference_context.materialize_only_last_token_logits = False
            logits = inference_wrapped_model.run_one_forward_step(forward_args)
            inference_wrapped_model.inference_context.reset()

            from megatron.core.inference.communication_utils import broadcast_from_last_pipeline_stage

            batch_size, context_length, vocab_size = 1, len(seq), 512
            logits = broadcast_from_last_pipeline_stage(
                [batch_size, context_length, vocab_size],
                dtype=inference_wrapped_model.inference_wrapper_config.params_dtype,
                tensor=logits,
            )

            matchrate = calc_matchrate(tokenizer=mcore_tokenizer, in_seq=seq, logits=logits)
            matchrates.append(matchrate)
            check_matchrate(ckpt_name=ckpt_name, matchrate=matchrate, assert_matchrate=False)
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
        ("evo2/1b-8k-bf16:1.0", [96.27, 67.93, 77.50, 80.30], True),
        ("evo2/1b-8k:1.0", [96.27, 67.93, 77.50, 80.30], False),
        ("evo2/7b-8k:1.0", [97.60, 89.63, 80.03, 84.57], False),
        ("evo2/7b-1m:1.0", [97.60, 89.63, 80.03, 84.57], False),
    ],
)
def test_forward_manual(sequences: list[str], ckpt_name: str, expected_matchpercents: list[float], flash_decode: bool):
    assert len(sequences) > 0
    seq_len_cap = determine_memory_requirement_and_skip_if_not_met(
        ckpt_name, test_name=inspect.currentframe().f_code.co_name
    )

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported

    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    with distributed_model_parallel_state(), torch.no_grad():
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        flash_decode_kwargs: dict[str, Any] = {"flash_decode": flash_decode}
        if flash_decode:
            flash_decode_kwargs["attention_backend"] = AttnBackend.flash
        if "1b-8k" in ckpt_name:
            model_config = llm.Hyena1bConfig(
                use_te=True,
                seq_length=8192,
                vortex_style_fp8=vortex_style_fp8,
                **flash_decode_kwargs,
            )
        elif "7b-8k" in ckpt_name:
            model_config = llm.Hyena7bConfig(
                use_te=True,
                seq_length=8192,
                vortex_style_fp8=vortex_style_fp8,
                **flash_decode_kwargs,
            )
        elif "7b-1m" in ckpt_name:
            model_config = llm.Hyena7bARCLongContextConfig(
                use_te=True,
                seq_length=8192,
                vortex_style_fp8=vortex_style_fp8,
                **flash_decode_kwargs,
            )
        else:
            raise NotImplementedError
        ckpt_weights: Path = load(ckpt_name) / "weights"
        raw_megatron_model = model_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, ckpt_weights, {}, "torch_dist")
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
            seq = seq[
                :seq_len_cap
            ]  # TODO: artificial limit, megatron uses more memory. Vortex can process full sequences
            with torch.no_grad():
                device = torch.cuda.current_device()
                # tokens = torch.tensor([tokenizer.tokenize(seq)], device=device)
                input_ids = torch.tensor(tokenizer.text_to_ids(seq)).int().unsqueeze(0).to(device)
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
                matchrate = calc_matchrate(tokenizer=tokenizer, in_seq=seq, logits=logits)
                matchrates.append(matchrate)
                check_matchrate(ckpt_name=ckpt_name, matchrate=matchrate, assert_matchrate=False)
        assert len(matchrates) == len(expected_matchpercents)
        matchperc_print = [f"{m * 100.0:.1f}%" for m in matchrates]
        matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
        assert all(m * 100.0 >= 0.95 * ep for m, ep in zip(matchrates, expected_matchpercents)), (
            f"Expected at least 95% of {matchperc_print_expected=}, got {matchperc_print=}"
        )


def mid_point_split(*, seq, num_tokens: int | None = None, fraction: float = 0.5):
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


@pytest.mark.parametrize(
    "ckpt_name,model_tokenizer_provider,expected_matchpercents",
    [
        ("evo2/1b-8k-bf16:1.0", get_model_and_tokenizer, [96.8, 29.7, 76.6, 71.6]),
        ("evo2/1b-8k:1.0", get_model_and_tokenizer, [96.8, 29.7, 76.6, 71.6]),
        ("evo2_mamba/7b-8k:0.1", get_model_and_tokenizer_ignore_vortex, [99.2, 51.0, 73.0, 82.6]),
        ("evo2/7b-8k:1.0", get_model_and_tokenizer, [97.60, 89.63, 80.03, 84.57]),
        ("evo2/7b-1m:1.0", get_model_and_tokenizer, [97.60, 89.63, 80.03, 84.57]),
    ],
)
def test_batch_generate(
    sequences: list[str], ckpt_name: str, model_tokenizer_provider: Callable, expected_matchpercents: list[float]
):
    assert len(sequences) > 0
    _ = determine_memory_requirement_and_skip_if_not_met(ckpt_name, test_name=inspect.currentframe().f_code.co_name)

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    if "evo2_mamba" in ckpt_name and os.environ.get("BIONEMO_DATA_SOURCE") != "pbss":
        # TODO: add evo2_mamba/7b-8k to NGC and remove this skip
        pytest.skip(f"Skipping {ckpt_name} because it is not on NGC yet. Run with `BIONEMO_DATA_SOURCE=pbss`.")
    # only use vortex_style_fp8 for non-bf16 checkpoints with fp8 support
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name

    num_tokens = 500
    seq_prompts = [mid_point_split(seq=seq, num_tokens=num_tokens) for seq in sequences]
    seq_len_max = num_tokens + max([len(sq[0]) for sq in seq_prompts])
    inference_wrapped_model, mcore_tokenizer = model_tokenizer_provider(
        ckpt_name,
        vortex_style_fp8=vortex_style_fp8,
        seq_len_max=seq_len_max,
    )

    results = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=[sq[0] for sq in seq_prompts],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=num_tokens,
        ),
    )

    match_percents = []
    for i, (result, (prompt, target)) in enumerate(zip(results, seq_prompts)):
        gen_seq = result.generated_text
        logging.info(f"{ckpt_name} {torch.distributed.get_rank()=} {gen_seq=}")
        logging.info(f"{ckpt_name} {torch.distributed.get_rank()=} {target=}")
        match_percent = calculate_sequence_identity(target, gen_seq)
        logging.info(
            f"{ckpt_name} {torch.distributed.get_rank()=} {match_percent=} expected: {expected_matchpercents[i]}"
        )
        match_percents.append(match_percent)

    assert len(match_percents) == len(expected_matchpercents)
    matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )


@pytest.mark.parametrize(
    "ckpt_name,model_tokenizer_provider,expected_matchpercents",
    [
        ("evo2/1b-8k-bf16:1.0", get_model_and_tokenizer, [86.4, 78.8, 49.7]),
        ("evo2/1b-8k:1.0", get_model_and_tokenizer, [86.4, 78.8, 49.7]),
        ("evo2_mamba/7b-8k:0.1", get_model_and_tokenizer_ignore_vortex, [86.5, 88.4, 88.2]),
        ("evo2/7b-8k:1.0", get_model_and_tokenizer, [88.8, 88.5, 82.2]),
        ("evo2/7b-1m:1.0", get_model_and_tokenizer, [88.8, 88.5, 82.2]),
    ],
)
def test_batch_generate_coding_sequences(
    coding_sequences: list[str],
    ckpt_name: str,
    model_tokenizer_provider: Callable,
    expected_matchpercents: list[float],
):
    assert len(coding_sequences) > 0
    determine_memory_requirement_and_skip_if_not_met(ckpt_name, test_name=inspect.currentframe().f_code.co_name)

    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    if "evo2_mamba" in ckpt_name and os.environ.get("BIONEMO_DATA_SOURCE") != "pbss":
        # TODO: add evo2_mamba/7b-8k to NGC and remove this skip
        pytest.skip(f"Skipping {ckpt_name} because it is not on NGC yet. Run with `BIONEMO_DATA_SOURCE=pbss`.")
    # only use vortex_style_fp8 for non-bf16 checkpoints with fp8 support
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name

    match_percents: list[float] = []
    cds_lengths: list[int | None] = []
    original_cds_lengths: list[int] = [len(seq) for seq in coding_sequences]
    seq_prompts = [mid_point_split(seq=seq, num_tokens=None, fraction=0.3) for seq in coding_sequences]
    num_tokens = max(len(sq[1]) for sq in seq_prompts) + 15

    inference_wrapped_model, mcore_tokenizer = model_tokenizer_provider(
        ckpt_name, vortex_style_fp8=vortex_style_fp8, enable_flash_decode=True, flash_decode=True
    )

    _ = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=["AAACCC"],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=1,
        ),
    )
    results = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=[sq[0] for sq in seq_prompts],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=num_tokens,
        ),
    )

    for i, (result, (prompt, target)) in enumerate(zip(results, seq_prompts)):
        gen_seq = result.generated_text
        logging.info(f"{ckpt_name} {torch.distributed.get_rank()=} {gen_seq=}")
        logging.info(f"{ckpt_name} {torch.distributed.get_rank()=} {target=}")
        full_seq = prompt + gen_seq
        stop_codons = {"TAA", "TAG", "TGA"}
        assert full_seq[:3] == "ATG"  # start codon
        cds_length = None
        for codon_start in range(0, len(full_seq), 3):
            codon = full_seq[codon_start : codon_start + 3]
            if codon in stop_codons:
                cds_length = codon_start + 3
                break
        match_percent = calculate_sequence_identity(target, gen_seq)
        logging.info(
            f"{ckpt_name} {torch.distributed.get_rank()=} {match_percent=} expected: {expected_matchpercents[i]}"
        )
        match_percents.append(match_percent)
        cds_lengths.append(cds_length)
        # 99% of the time, you have a stop codon within the first 96 codons if everything were random.

    assert len(match_percents) == len(expected_matchpercents)
    assert len(cds_lengths) == len(original_cds_lengths)
    matchperc_print = [f"{mp:.1f}%" for mp in match_percents]
    matchperc_print_expected = [f"{ep:.1f}%" for ep in expected_matchpercents]
    # By chance you expect to have a stop codon within the first 96 codons if everything were random
    #  so verify that we are putting the first stop codon after this point, as well as it being at least 90% of the
    #  original sequence length.
    assert all(
        pcl is None or ((pcl - len(pmpt) > 96 * 3 or len(tgt) < 96 * 3) and pcl >= 0.9 * ocl)
        for pcl, ocl, (pmpt, tgt) in zip(cds_lengths, original_cds_lengths, seq_prompts)
    ), f"Expected at least 70% of {original_cds_lengths=}, got {cds_lengths=}"
    assert all(mp >= 0.90 * ep for mp, ep in zip(match_percents, expected_matchpercents)), (
        f"Expected at least 90% of {matchperc_print_expected=}, got {matchperc_print=}"
    )


@pytest.mark.skip(
    reason="skip the test for now, and decide what to do after getting Anton's changes sorted and merged."
)
@pytest.mark.slow
@pytest.mark.parametrize(
    "ckpt_name,model_tokenizer_provider,expected_tokens_sec",
    [
        ("evo2/1b-8k-bf16:1.0", get_model_and_tokenizer, 41.0),
        ("evo2/1b-8k:1.0", get_model_and_tokenizer, 41.0),
        ("evo2_mamba/7b-8k:0.1", get_model_and_tokenizer_ignore_vortex, 39.73),
        ("evo2/7b-8k:1.0", get_model_and_tokenizer, 32.0),
        ("evo2/7b-1m:1.0", get_model_and_tokenizer, 32.0),
    ],
)
def test_generate_speed(
    ckpt_name: str,
    model_tokenizer_provider: Callable,
    expected_tokens_sec: float,
):
    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    determine_memory_requirement_and_skip_if_not_met(ckpt_name, test_name=inspect.currentframe().f_code.co_name)

    skip = "evo2/1b-8k:" in ckpt_name and not is_fp8_supported
    if skip:
        # This checkpoint is sensitive to FP8, so we skip it if it is not supported on the current device.
        pytest.skip(f"Skipping {ckpt_name} because it is not supported on {device_info} ({compute_capability})")
    if "evo2_mamba" in ckpt_name and os.environ.get("BIONEMO_DATA_SOURCE") != "pbss":
        # TODO: add evo2_mamba/7b-8k to NGC and remove this skip
        pytest.skip(f"Skipping {ckpt_name} because it is not on NGC yet. Run with `BIONEMO_DATA_SOURCE=pbss`.")
    # only use vortex_style_fp8 for non-bf16 checkpoints with fp8 support
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    inference_wrapped_model, mcore_tokenizer = model_tokenizer_provider(
        ckpt_name,
        vortex_style_fp8=vortex_style_fp8,
        fp32_residual_connection=False,
        enable_flash_decode=True,
        flash_decode=True,
    )

    # warm up the model with a single call before timing. This should take care of compilation etc.
    _ = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=["AAACCC"],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=1,
        ),
    )
    t0 = time.perf_counter_ns()
    results = generate(
        model=inference_wrapped_model,
        max_batch_size=1,  # vortex only supports batch size 1
        tokenizer=mcore_tokenizer,
        prompts=["A"],
        random_seed=42,
        inference_params=CommonInferenceParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            return_log_probs=False,
            num_tokens_to_generate=300,
        ),
    )
    dt = (time.perf_counter_ns() - t0) / 1e9  # seconds
    tokens_per_sec = (len(results[0].generated_text) + 1) / dt  # +1 for the prompt
    assert tokens_per_sec > expected_tokens_sec * 0.85, (
        f"Expected at least {expected_tokens_sec} tokens/sec, got {tokens_per_sec}"
    )
