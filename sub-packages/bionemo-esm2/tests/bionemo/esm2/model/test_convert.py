# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import pytest
import torch
from nemo.lightning import io
from transformers import AutoModelForMaskedLM
from typer.testing import CliRunner

from bionemo.core.data.load import load
from bionemo.esm2.model.convert import (
    HFESM2Importer,  # noqa: F401
    app,
)
from bionemo.esm2.model.model import ESM2Config
from bionemo.esm2.testing.compare import assert_esm2_equivalence
from bionemo.llm.model.biobert.lightning import biobert_lightning_module
from bionemo.testing import megatron_parallel_state_utils


def test_nemo2_conversion_equivalent_8m(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag)


def test_nemo2_conversion_equivalent_8m_with_local_path(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    hf_model = AutoModelForMaskedLM.from_pretrained(model_tag)
    hf_model.save_pretrained(tmp_path / "hf_checkpoint")

    module = biobert_lightning_module(config=ESM2Config(), post_process=True)
    io.import_ckpt(module, f"hf://{tmp_path / 'hf_checkpoint'}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag)


def test_nemo2_export_8m_weights_equivalent(tmp_path):
    ckpt_path = load("esm2/8m:2.0")
    output_path = io.export_ckpt(ckpt_path, "hf", tmp_path / "hf_checkpoint")

    hf_model_from_nemo = AutoModelForMaskedLM.from_pretrained(output_path)
    hf_model_from_hf = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # We should not have any NaNs in the model, even in heads that we don't explicitly initialize.
    assert not {name: param for name, param in hf_model_from_nemo.named_parameters() if param.data.isnan().any()}

    del hf_model_from_nemo.esm.contact_head
    del hf_model_from_hf.esm.contact_head

    for key in hf_model_from_nemo.state_dict().keys():
        if key == "esm.embeddings.position_embeddings.weight":
            # Do not know why exactly only this key has a mismatch
            # Appears to be a HF issue?? https://github.com/huggingface/transformers/issues/39038
            continue
        torch.testing.assert_close(
            hf_model_from_nemo.state_dict()[key],
            hf_model_from_hf.state_dict()[key],
            atol=1e-4,
            rtol=1e-4,
            msg=lambda msg: f"{key}: {msg}",
        )


def test_nemo2_export_golden_values(tmp_path):
    ckpt_path = load("esm2/8m:2.0")
    output_path = io.export_ckpt(ckpt_path, "hf", tmp_path / "hf_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(ckpt_path, output_path, precision="bf16")


def test_nemo2_export_on_gpu(tmp_path):
    ckpt_path = load("esm2/8m:2.0")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        io.export_ckpt(ckpt_path, "hf", tmp_path / "hf_checkpoint")


def test_nemo2_conversion_equivalent_8m_bf16(tmp_path):
    model_tag = "facebook/esm2_t6_8M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag, precision="bf16")


@pytest.mark.slow
def test_nemo2_conversion_equivalent_650m(tmp_path):
    model_tag = "facebook/esm2_t33_650M_UR50D"
    module = biobert_lightning_module(config=ESM2Config())
    io.import_ckpt(module, f"hf://{model_tag}", tmp_path / "nemo_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(tmp_path / "nemo_checkpoint", model_tag, atol=1e-4, rtol=1e-4)


@pytest.mark.slow
def test_nemo2_export_equivalent_650m(tmp_path):
    ckpt_path = load("esm2/nv_650m:2.1")
    output_path = io.export_ckpt(ckpt_path, "hf", tmp_path / "hf_checkpoint")
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        assert_esm2_equivalence(ckpt_path, output_path, precision="bf16")


def test_cli_nemo2_conversion_equivalent_8m(tmp_path):
    """Test that the CLI conversion functions maintain model equivalence."""
    model_tag = "facebook/esm2_t6_8M_UR50D"
    runner = CliRunner()

    # First convert HF to NeMo
    nemo_path = tmp_path / "nemo_checkpoint"
    result = runner.invoke(app, ["convert-hf-to-nemo", model_tag, str(nemo_path)])
    assert result.exit_code == 0, f"CLI command failed: {result.output}"

    # Then convert back to HF
    hf_path = tmp_path / "hf_checkpoint"
    result = runner.invoke(app, ["convert-nemo-to-hf", str(nemo_path), str(hf_path)])
    assert result.exit_code == 0, f"CLI command failed: {result.output}"

    hf_model_from_nemo = AutoModelForMaskedLM.from_pretrained(model_tag)
    hf_model_from_hf = AutoModelForMaskedLM.from_pretrained(hf_path)

    # These aren't initialized, so they're going to be different.
    del hf_model_from_nemo.esm.contact_head
    del hf_model_from_hf.esm.contact_head

    for key in hf_model_from_nemo.state_dict().keys():
        if key == "esm.embeddings.position_embeddings.weight":
            # Do not know why exactly only this key has a mismatch
            # Appears to be a HF issue?? https://github.com/huggingface/transformers/issues/39038
            continue
        torch.testing.assert_close(
            hf_model_from_nemo.state_dict()[key],
            hf_model_from_hf.state_dict()[key],
            atol=1e-4,
            rtol=1e-4,
            msg=lambda msg: f"{key}: {msg}",
        )
