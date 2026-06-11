# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Golden-value tests for ESM2 vLLM compatibility.

Performs a fresh facebook -> TE export, then cross-validates embeddings across
vLLM, HuggingFace (exported checkpoint), and HuggingFace (nvidia Hub reference).

vLLM's pooling runner returns last-token, L2-normalized embeddings by default,  # gitleaks:allow
so the HuggingFace paths replicate that post-processing for comparison.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoTokenizer


try:
    from vllm import LLM

    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False

from export import export_hf_checkpoint


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

EXPORT_TAG = "esm2_t6_8M_UR50D"
REFERENCE_MODEL_ID = "nvidia/esm2_t6_8M_UR50D"
ESM2_MODEL_DIR = Path(__file__).resolve().parent.parent

SEQUENCES = [
    "LKGHAMCLGCLHMLMCGLLAGAMCGLMKLLKCCGKCLMHLMKAMLGLKCACHHHHLLLHACAAKKLCLGAKLAMGLKLLGAHGKGLKMACGHHMLHLHMH",
    "CLLCCMHMHAHHCHGHGHKCKCLMMGMALMCAGCCACGMKGGCHCCLLAHCAHAKAGKGKCKLMCKKKHGLHAGLHAMLLCHLGLGCGHHHKKCKKHKCA",
]


def _last_token_l2(hidden_state: torch.Tensor) -> np.ndarray:
    """Extract last-token hidden state and L2-normalise (matches vLLM pooling defaults)."""
    vec = hidden_state[0, -1, :].cpu().float().numpy()
    norm = np.linalg.norm(vec)
    if norm > 1e-9:
        vec = vec / norm
    return vec


def _hf_embed(model_id: str, sequences: list[str], dtype=torch.float32) -> np.ndarray:
    """Run HuggingFace inference and return last-token L2-normalised embeddings."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to("cuda", dtype=dtype).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    vecs = []
    with torch.no_grad():
        for seq in sequences:
            inputs = tokenizer(seq, return_tensors="pt")
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            out = model(**inputs)
            vecs.append(_last_token_l2(out.last_hidden_state))

    del model, tokenizer
    torch.cuda.empty_cache()
    return np.stack(vecs)


def _vllm_embed(model_id: str, sequences: list[str]) -> np.ndarray:
    """Run vLLM pooling inference and return embeddings."""
    engine = LLM(
        model=model_id,
        runner="pooling",
        trust_remote_code=True,
        dtype="float32",
        enforce_eager=True,
        max_num_batched_tokens=1026,
        seed=42,
    )
    outputs = engine.embed(sequences)

    vecs = []
    for output in outputs:
        emb = output.outputs.embedding
        if isinstance(emb, list):
            emb = np.array(emb)
        vecs.append(emb)

    del engine
    return np.stack(vecs)


# ---- Fixtures ----


@pytest.fixture(scope="session")
def exported_checkpoint(tmp_path_factory):
    """Fresh facebook -> TE export. Session-scoped so it runs once."""
    export_dir = tmp_path_factory.mktemp("vllm_export")
    export_hf_checkpoint(EXPORT_TAG, export_dir)
    return str(export_dir / EXPORT_TAG)


@pytest.fixture(scope="session")
def vllm_embeddings(exported_checkpoint):
    """Embeddings from vLLM pooling runner on the exported checkpoint."""
    if not _VLLM_AVAILABLE:
        pytest.skip("vllm not installed")
    return _vllm_embed(exported_checkpoint, SEQUENCES)


@pytest.fixture(scope="session")
def hf_exported_embeddings(exported_checkpoint):
    """Embeddings from HuggingFace on the exported checkpoint."""
    return _hf_embed(exported_checkpoint, SEQUENCES)


@pytest.fixture(scope="session")
def hf_reference_embeddings():
    """Embeddings from HuggingFace on the nvidia Hub model (ground truth)."""
    return _hf_embed(REFERENCE_MODEL_ID, SEQUENCES)


# ---- Tests ----


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vllm not installed")
def test_vllm_vs_hf_exported(vllm_embeddings, hf_exported_embeddings):
    """vLLM and native HuggingFace on the same exported checkpoint must match."""
    np.testing.assert_allclose(vllm_embeddings, hf_exported_embeddings, atol=2e-4)


@pytest.mark.skipif(not _VLLM_AVAILABLE, reason="vllm not installed")
def test_vllm_vs_hf_reference(vllm_embeddings, hf_reference_embeddings):
    """vLLM on exported checkpoint must match HuggingFace on nvidia Hub model."""
    np.testing.assert_allclose(vllm_embeddings, hf_reference_embeddings, atol=2e-4)


def test_hf_exported_vs_hf_reference(hf_exported_embeddings, hf_reference_embeddings):
    """Our exported checkpoint must produce identical results to the nvidia Hub model."""
    np.testing.assert_array_equal(hf_exported_embeddings, hf_reference_embeddings)
