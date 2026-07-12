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

"""Model adapters for BioNeMo models — Adapter Pattern.

Each model has a dedicated adapter class that knows how to:
  1. Download its pretrained checkpoint
  2. Create/load the model on CUDA
  3. Build synthetic calibration data
  4. Prepare forward() arguments
  5. Parse the model output

Supported models (3):
  - ESM-2:       Transformer encoder for protein sequences (NGC checkpoint)
  - Geneformer:  Transformer encoder for gene expression (NGC checkpoint)
  - Evo2:        Hyena/Mamba SSM for DNA sequences (NGC checkpoint)

Usage:
    from src.adapters import get_adapter, ADAPTER_REGISTRY

    adapter = get_adapter("esm2")
    ckpt = adapter.download_checkpoint()
    model = adapter.load_model(ckpt)
    data = adapter.build_calibration_data(tokenizer, num_samples=8)
"""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.utils.data


# ============================================================================
# Constants
# ============================================================================

# NGC checkpoint tags for each model
MODEL_TAGS = {
    "esm2": "esm2/8m:2.0",
    "geneformer": "geneformer/10M_241113:2.0",
    "evo2": "evo2/7b-8k:1.0",
}


# ============================================================================
# Distributed environment initialization (required by Megatron-Core)
# ============================================================================

_DIST_INITIALIZED = False


def init_distributed():
    """Initialize single-GPU distributed environment for Megatron-Core.

    Sets up torch.distributed (NCCL backend) and Megatron's parallel state
    with tensor/pipeline parallelism = 1. BioNeMo models require this even
    for single-GPU inference because they use Megatron-Core internals
    (ColumnParallelLinear, etc.).

    Safe to call multiple times — only the first call has effect.
    """
    global _DIST_INITIALIZED
    if _DIST_INITIALIZED:
        return

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")

    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    from megatron.core import parallel_state
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(1, 1)

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(42)

    _DIST_INITIALIZED = True


# ============================================================================
# Abstract Model Adapter
# ============================================================================

class ModelAdapter(ABC):
    """Abstract base class for BioNeMo model adapters.

    Each concrete adapter encapsulates all model-specific logic:
    loading, tokenization, data preparation, and forward pass handling.

    Attributes:
        name:         Short identifier (e.g., "esm2")
        input_type:   Data domain ("protein", "dna", "gene_ids")
        architecture: Model architecture type ("transformer", "hyena_ssm")
        description:  Human-readable model description
    """

    name: str = "base"
    input_type: str = "unknown"
    architecture: str = "unknown"
    description: str = ""

    # ------------------------------------------------------------------
    # Required methods (subclasses must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def download_checkpoint(self) -> Optional[str]:
        """Download pretrained checkpoint from NGC.

        Returns:
            Local filesystem path to the checkpoint, or None if N/A.
        """

    @abstractmethod
    def load_model(self, ckpt_path: Optional[str] = None) -> Tuple[torch.nn.Module, Any, str]:
        """Load the model onto CUDA in eval mode.

        Args:
            ckpt_path: Path to the pretrained checkpoint.

        Returns:
            (model, tokenizer, input_type):
              - model: nn.Module, bf16, CUDA, eval mode
              - tokenizer: tokenizer instance (or None)
              - input_type: "protein", "dna", or "gene_ids"
        """

    @abstractmethod
    def generate_test_data(
        self, tokenizer: Any, batch_size: int = 4, seq_length: int = 128,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic test data for this model.

        Args:
            tokenizer: Tokenizer instance from load_model().
            batch_size: Number of sequences to generate.
            seq_length: Maximum sequence length.

        Returns:
            (input_ids, attention_mask): Both on CUDA, shape [B, L].
        """

    @abstractmethod
    def run_forward(
        self, model: torch.nn.Module, input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run model forward pass and extract logits.

        Args:
            model: The BioNeMo model.
            input_ids: Token IDs, shape [B, L].
            attention_mask: Attention mask, shape [B, L].

        Returns:
            Logits tensor, typically shape [B, L, V].
        """

    @abstractmethod
    def build_calibration_forward_args(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build keyword arguments for model.forward() during calibration.

        Args:
            input_ids: Token IDs tensor.
            attention_mask: Attention mask tensor.

        Returns:
            Dict of kwargs to pass to model(**kwargs).
        """

    # ------------------------------------------------------------------
    # Shared helper: parse model output into logits
    # ------------------------------------------------------------------

    @staticmethod
    def extract_logits(output) -> torch.Tensor:
        """Extract logits from various model output formats.

        BioNeMo models return different output types:
         - Raw Tensor
         - Dict with 'token_logits' or 'logits'
         - Named object with .token_logits or .logits attributes
        """
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            return output.get("token_logits", output.get("logits", None))
        for attr in ["token_logits", "logits"]:
            if hasattr(output, attr):
                return getattr(output, attr)
        return output


# ============================================================================
# ESM-2 Adapter (Protein Transformer)
# ============================================================================

class ESM2Adapter(ModelAdapter):
    """Adapter for ESM-2 protein language model.

    ESM-2 is a masked language model for protein sequences based on the
    Transformer encoder architecture. It uses the standard ESM tokenizer
    with a 20-amino-acid vocabulary plus special tokens.

    Architecture: Transformer encoder (6 layers, 320 hidden, 20 heads for 8M)
    Input: Amino acid sequences (e.g., "MKTLLILAVVAA...")
    Checkpoint: NGC esm2/8m:2.0
    """

    name = "esm2"
    input_type = "protein"
    architecture = "transformer"
    description = "ESM-2 8M protein language model"

    def download_checkpoint(self) -> str:
        from bionemo.core.data.load import load
        return str(load(MODEL_TAGS["esm2"], source="ngc"))

    def load_model(self, ckpt_path: str) -> Tuple[torch.nn.Module, Any, str]:
        init_distributed()
        import nemo.lightning as nl
        from bionemo.esm2.data.tokenizer import BioNeMoESMTokenizer

        ctx = nl.io.load_context(ckpt_path)
        config = ctx.model.config
        config.initial_ckpt_path = ckpt_path

        tokenizer = BioNeMoESMTokenizer()
        model = config.configure_model(tokenizer)
        model = model.bfloat16().cuda().eval()

        return model, tokenizer, self.input_type

    def generate_test_data(self, tokenizer, batch_size=4, seq_length=128):
        rng = random.Random(42)
        aa = "ACDEFGHIKLMNPQRSTVWY"
        seqs = ["".join(rng.choices(aa, k=seq_length - 2)) for _ in range(batch_size)]
        enc = tokenizer(
            seqs, padding="max_length", max_length=seq_length,
            truncation=True, return_tensors="pt",
        )
        return enc["input_ids"].cuda(), enc["attention_mask"].cuda()

    def run_forward(self, model, input_ids, attention_mask):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        return self.extract_logits(output)

    def build_calibration_forward_args(self, input_ids, attention_mask):
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ============================================================================
# Geneformer Adapter (Gene Expression Transformer)
# ============================================================================

class GeneformerAdapter(ModelAdapter):
    """Adapter for Geneformer single-cell gene expression model.

    Geneformer is a Transformer encoder pretrained on single-cell
    transcriptomic data. It takes gene expression rank tokens as input
    and predicts masked gene tokens.

    Special handling:
      - Requires patching `load_settings_from_checkpoint` to load
        outside of NeMo's full training pipeline.
      - Uses a mock tokenizer with vocab_size=25429 for configure_model().
      - Test data is random integer gene IDs (not text sequences).

    Architecture: Transformer encoder (6 layers, 256 hidden, 4 heads for 10M)
    Input: Gene expression rank tokens (integers 0-25428)
    Checkpoint: NGC geneformer/10M_241113:2.0
    """

    name = "geneformer"
    input_type = "gene_ids"
    architecture = "transformer"
    description = "Geneformer 10M single-cell gene expression model"

    def download_checkpoint(self) -> str:
        from bionemo.core.data.load import load
        return str(load(MODEL_TAGS["geneformer"], source="ngc"))

    def load_model(self, ckpt_path: str) -> Tuple[torch.nn.Module, Any, str]:
        init_distributed()
        from bionemo.geneformer.api import GeneformerConfig
        import bionemo.llm.model.config as bionemo_config

        # Patch: skip checkpoint settings loading (only needed for training)
        original = (
            bionemo_config.MegatronBioNeMoTrainableModelConfig
            .load_settings_from_checkpoint
        )
        bionemo_config.MegatronBioNeMoTrainableModelConfig.load_settings_from_checkpoint = (
            lambda self, path: None
        )

        try:
            class _GeneformerTokenizer:
                """Minimal tokenizer stub for Geneformer's configure_model()."""
                vocab_size = 25429
                def __len__(self):
                    return self.vocab_size

            config = GeneformerConfig(
                num_layers=6, hidden_size=256, num_attention_heads=4,
                ffn_hidden_size=512, initial_ckpt_path=ckpt_path,
            )
            tokenizer = _GeneformerTokenizer()
            model = config.configure_model(tokenizer)
            model = model.bfloat16().cuda().eval()
        finally:
            bionemo_config.MegatronBioNeMoTrainableModelConfig.load_settings_from_checkpoint = original

        return model, None, self.input_type

    def generate_test_data(self, tokenizer, batch_size=4, seq_length=128):
        # Geneformer uses integer gene IDs, not text
        input_ids = torch.randint(0, 25426, (batch_size, seq_length), dtype=torch.long).cuda()
        return input_ids, torch.ones_like(input_ids).cuda()

    def run_forward(self, model, input_ids, attention_mask):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        return self.extract_logits(output)

    def build_calibration_forward_args(self, input_ids, attention_mask):
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# ============================================================================
# Evo2 Adapter (Hyena/Mamba SSM for DNA)
# ============================================================================

class Evo2Adapter(ModelAdapter):
    """Adapter for Evo2 DNA foundation model.

    Evo2 uses a Hyena/Mamba state-space model (SSM) architecture — NOT a
    standard Transformer. This has several important implications:

    Key differences from Transformer models:
      1. Requires position_ids in forward() call
      2. Uses Evo2Tokenizer with byte-level DNA vocabulary (A/C/G/T)
      3. Has NON-DETERMINISTIC initialization: the `configure_model()`
         function initializes some weights randomly, and subsequent
         `load_state_dict` doesn't fully override them due to CUDA RNG
         state differences. This means loading the same checkpoint twice
         in the same process can produce different models.
      4. For quantization testing, we use SUBPROCESS ISOLATION: each
         quant method runs in its own process to ensure a clean load.

    Architecture: Hyena/Mamba SSM (32 layers, mix of Hyena + Attention)
    Input: DNA nucleotide sequences (A/C/G/T)
    Checkpoint: NGC evo2/7b-8k:1.0

    Quantization notes:
      - Default ModelOpt configs only quantize TE Linear layers in the
        5 attention layers (out of 32 total layers).
      - Removing 'default: enable=False' quantizes all 32 MLP layers
        (QuantMLP), including the 27 HyenaLayer MLPs.
      - HyenaFilter's internal nn.Linear layers (inside nn.Sequential)
        are NOT wrapped by ModelOpt because it doesn't traverse Sequential.
      - ImplicitModalFilter (gamma, R, p) and ExplicitSingleDecayFilter (h)
        are nn.Parameter, not Linear layers, and cannot be quantized.
    """

    name = "evo2"
    input_type = "dna"
    architecture = "hyena_ssm"
    description = "Evo2 7B DNA foundation model (Hyena/Mamba SSM)"

    def download_checkpoint(self) -> str:
        from bionemo.core.data.load import load
        return str(load(MODEL_TAGS["evo2"], source="ngc"))

    def load_model(self, ckpt_path: str) -> Tuple[torch.nn.Module, Any, str]:
        init_distributed()
        import nemo.lightning as nl
        from bionemo.evo2.data.tokenizer import Evo2Tokenizer

        ctx = nl.io.load_context(ckpt_path)
        config = ctx.model.config
        config.initial_ckpt_path = ckpt_path

        tokenizer = Evo2Tokenizer()
        # Evo2 tokenizer requires explicit vocab_size setting
        tokenizer.vocab_size = tokenizer.tokenizer.vocab_size

        model = config.configure_model(tokenizer)
        model = model.bfloat16().cuda().eval()

        return model, tokenizer, self.input_type

    def generate_test_data(self, tokenizer, batch_size=4, seq_length=256):
        rng = random.Random(42)
        seqs = ["".join(rng.choices("ACGT", k=seq_length - 2)) for _ in range(batch_size)]
        all_ids = []
        for s in seqs:
            ids = tokenizer.tokenize(s)
            # Evo2 tokenizer returns list-of-lists
            if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                ids = ids[0]
            ids = ids[:seq_length]
            ids = ids + [0] * (seq_length - len(ids))
            all_ids.append(ids)
        input_ids = torch.tensor(all_ids, dtype=torch.long).cuda()
        return input_ids, (input_ids != 0).long().cuda()

    def run_forward(self, model, input_ids, attention_mask):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .unsqueeze(0).expand_as(input_ids)
            )
            output = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )
        return self.extract_logits(output)

    def build_calibration_forward_args(self, input_ids, attention_mask):
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0).expand_as(input_ids)
        )
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
        }


# ============================================================================
# Adapter Registry
# ============================================================================

ADAPTER_REGISTRY: Dict[str, type] = {
    "esm2": ESM2Adapter,
    "geneformer": GeneformerAdapter,
    "evo2": Evo2Adapter,
}


def get_adapter(model_name: str) -> ModelAdapter:
    """Get the adapter instance for a given model name.

    Args:
        model_name: One of "esm2", "geneformer", "evo2".

    Returns:
        Instantiated ModelAdapter subclass.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            f"Available: {list(ADAPTER_REGISTRY.keys())}"
        )
    return ADAPTER_REGISTRY[model_name]()


def list_models() -> List[str]:
    """Return list of supported model names."""
    return list(ADAPTER_REGISTRY.keys())
